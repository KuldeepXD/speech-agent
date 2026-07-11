"""
Baseline 2: Vanilla RAG — Retrieve + Generate (No Agents).

For each test query:
  1. Uses a keyword heuristic to pick the Speech or Feeding FAISS index.
  2. Retrieves top-5 documents from the existing vector store.
  3. Sends retrieved context + query to Gemini via a simple prompt.

No classification agent, no web search, no synthesis agent.
Uses the existing vector stores from vector_stores/ (no new ingestion).

Supports batch evaluation with resume capability for 100+ queries.

Usage:
    from evaluation.baseline2_vanilla_rag import run_baseline2
    results = run_baseline2(limit=5)
    results = run_baseline2(batch_size=5, batch_delay=30, resume=True)
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from rag.retriever import load_speech_retriever, load_feeding_retriever
from evaluation.test_queries import TEST_QUERIES


# ── Config ─────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOG_FILE = RESULTS_DIR / "baseline2_logs.json"

# Simple prompt — no specialist persona, no multi-agent context
VANILLA_RAG_PROMPT = ChatPromptTemplate.from_template(
    """Answer the following patient query using only the clinical reference context provided below.

## Patient Query
{query}

## Retrieved Clinical Context
{retrieved_context}

## Instructions
Provide a clear, helpful answer based on the retrieved context above.
Include relevant treatment recommendations and assessment guidance.
"""
)

# Lightweight prompt to extract a structured ailment label from the generated answer
AILMENT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are a clinical NLP assistant. Given the query and the AI-generated clinical answer below,
identify the PRIMARY ailment or condition being discussed.

Respond with ONLY a JSON object in this exact format (no extra text):
{{"ailment": "<1-3 word clinical term>"}}

Examples of valid ailment values: "Aphasia", "Childhood Apraxia of Speech", "Post-Stroke Dysphagia",
"Oro-Motor Weakness", "Language Delay", "Drooling", "Sensory Food Aversion".

## Query
{query}

## Generated Answer
{generated_answer}
"""
)

# Keywords for simple category routing (no LLM classification)
FEEDING_KEYWORDS = [
    "swallowing", "dysphagia", "feeding", "aspiration", "choking",
    "food refusal", "texture", "bolus", "pharyngeal", "esophageal",
    "eating", "drinking", "gag", "failure to thrive", "puree",
    "chewing", "oral motor", "reflux",
]


def _guess_category(query: str) -> str:
    """Simple keyword heuristic to route to Speech or Feeding index.

    Args:
        query: The user's query text.

    Returns:
        'Feeding' or 'Speech' based on keyword match count.
    """
    query_lower = query.lower()
    feeding_score = sum(1 for kw in FEEDING_KEYWORDS if kw in query_lower)
    return "Feeding" if feeding_score > 0 else "Speech"


def _format_retrieved_docs(docs: list) -> str:
    """Format retrieved documents into a context string.

    Args:
        docs: List of LangChain Document objects.

    Returns:
        Formatted string of all retrieved documents.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"--- Reference {i} (Source: {source}, Page: {page}) ---\n"
            f"{doc.page_content}\n"
        )
    return "\n".join(formatted) if formatted else "(No context retrieved)"


def _invoke_with_retry(
    chain,
    params: dict,
    max_retries: int = 5,
    base_delay: float = 30.0,
):
    """Invoke an LLM chain with retry logic for 429 rate-limit errors.

    Args:
        chain: The LangChain chain to invoke.
        params: Parameters to pass to the chain.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.

    Returns:
        The chain's response.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(params)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    wait_time = base_delay * attempt
                    print(f"       ⏳ Rate limited (429). Retrying in {wait_time:.0f}s... ({attempt}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise


# ── Log Management ───────────────────────────────────────────────────

def _load_existing_logs(log_file: Path) -> list[dict]:
    """Load existing log entries from a JSON file.

    Args:
        log_file: Path to the JSON log file.

    Returns:
        List of existing result dictionaries, or empty list if file doesn't exist.
    """
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"       ⚠️  Could not read existing log file, starting fresh")
            return []
    return []


def _save_logs(log_file: Path, results: list[dict]) -> None:
    """Save results to a JSON log file (atomic write).

    Args:
        log_file: Path to the JSON log file.
        results: List of result dictionaries to save.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def run_baseline2(
    queries: list[dict] | None = None,
    limit: int | None = None,
    delay: float = 3.0,
    batch_size: int = 5,
    batch_delay: float = 30.0,
    resume: bool = False,
) -> list[dict]:
    """Run Baseline 2 evaluation: Vanilla RAG (retrieve + generate).

    Supports batch evaluation with resume capability.

    Args:
        queries: Optional list of query dicts. Defaults to TEST_QUERIES.
        limit: Optional limit on number of queries to run.
        delay: Delay in seconds between queries to avoid rate limits.
        batch_size: Number of queries per batch before a longer cooldown.
        batch_delay: Cooldown in seconds between batches.
        resume: If True, skip queries already present in existing logs.

    Returns:
        List of all result dictionaries (existing + new).
    """
    queries = queries or TEST_QUERIES
    if limit:
        queries = queries[:limit]

    # Load existing logs for resume support
    existing_results = _load_existing_logs(LOG_FILE) if resume else []
    completed_ids = {r["query_id"] for r in existing_results}

    # Filter out already-completed queries
    pending_queries = [q for q in queries if q["id"] not in completed_ids]

    print(f"\n{'='*70}")
    print(f"🔬 BASELINE 2: Vanilla RAG (Retrieve + Generate)")
    print(f"   Total queries: {len(queries)} | Already completed: {len(completed_ids)} | Pending: {len(pending_queries)}")
    print(f"   Batch size: {batch_size} | Batch delay: {batch_delay}s | Query delay: {delay}s")
    print(f"{'='*70}\n")

    if not pending_queries:
        print(f"  ✅ All queries already completed. Nothing to do.")
        return existing_results

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    chain = VANILLA_RAG_PROMPT | llm
    ailment_chain = AILMENT_EXTRACTION_PROMPT | llm

    all_results = list(existing_results)  # Start with existing results
    new_count = 0

    for i, entry in enumerate(pending_queries, 1):
        query_id = entry["id"]
        query_text = entry["query"]

        print(f"  [{i}/{len(pending_queries)}] {query_id}: {query_text[:60]}...")

        start_time = time.time()
        try:
            # Step 1: Guess category via keywords
            guessed_category = _guess_category(query_text)

            # Step 2: Load retriever and retrieve docs
            if guessed_category == "Feeding":
                retriever = load_feeding_retriever(k=5)
            else:
                retriever = load_speech_retriever(k=5)

            retrieved_docs = retriever.invoke(query_text)
            retrieved_context = _format_retrieved_docs(retrieved_docs)

            # Step 3: Generate answer with LLM
            response = _invoke_with_retry(chain, {
                "query": query_text,
                "retrieved_context": retrieved_context,
            })
            generated_answer = (
                response.content if hasattr(response, "content") else str(response)
            )

            latency = time.time() - start_time
            error = None

            sources = [
                {
                    "file": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page", "?"),
                }
                for doc in retrieved_docs
            ]

            # — Ailment extraction (lightweight second LLM call) —
            predicted_ailment = None
            try:
                ailment_response = _invoke_with_retry(ailment_chain, {
                    "query": query_text,
                    "generated_answer": generated_answer[:1500],  # truncate to save tokens
                })
                ailment_text = (
                    ailment_response.content
                    if hasattr(ailment_response, "content")
                    else str(ailment_response)
                )
                # Normalize: strip markdown code fences if present
                ailment_cleaned = ailment_text.strip()
                if ailment_cleaned.startswith("```"):
                    ailment_cleaned = ailment_cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                ailment_parsed = json.loads(ailment_cleaned)
                predicted_ailment = ailment_parsed.get("ailment", None)
            except Exception as ae:
                predicted_ailment = None  # graceful fallback

            print(f"       ✅ Done in {latency:.1f}s | Category guess: {guessed_category} | {len(retrieved_docs)} docs | Ailment: {predicted_ailment}")

        except Exception as e:
            latency = time.time() - start_time
            guessed_category = None
            retrieved_context = ""
            generated_answer = f"(Error: {e})"
            predicted_ailment = None
            sources = []
            error = str(e)
            print(f"       ❌ Error: {e}")

        result = {
            "query_id": query_id,
            "query": query_text,
            "ground_truth_category": entry["category"],
            "ground_truth_ailment": entry["expected_ailment"],
            "reference_answer": entry["reference_answer"],
            # — Outputs —
            "predicted_category": guessed_category,
            "predicted_ailment": predicted_ailment,  # now populated via LLM extraction
            "generated_answer": generated_answer,
            "retrieved_context": retrieved_context,
            "sources": sources,
            # — Metadata —
            "latency_seconds": latency,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        all_results.append(result)
        new_count += 1

        # Save after every query (incremental save for crash safety)
        _save_logs(LOG_FILE, all_results)

        # Rate-limit delay + batch cooldown
        if i < len(pending_queries):
            if i % batch_size == 0:
                print(f"\n  ⏸️  Batch {i // batch_size} complete. Cooling down for {batch_delay}s...")
                time.sleep(batch_delay)
            else:
                time.sleep(delay)

    print(f"\n  📁 Saved {len(all_results)} total results ({new_count} new) → {LOG_FILE}")
    print(f"  ⏱️  Avg latency (new): {sum(r['latency_seconds'] for r in all_results[-new_count:]) / new_count:.1f}s")
    print(f"{'='*70}\n")

    return all_results


if __name__ == "__main__":
    run_baseline2()


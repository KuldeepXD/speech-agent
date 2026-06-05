"""
Baseline 2: Vanilla RAG — Retrieve + Generate (No Agents).

For each test query:
  1. Uses a keyword heuristic to pick the Speech or Feeding FAISS index.
  2. Retrieves top-5 documents from the existing vector store.
  3. Sends retrieved context + query to Gemini via a simple prompt.

No classification agent, no web search, no synthesis agent.
Uses the existing vector stores from vector_stores/ (no new ingestion).

Usage:
    from evaluation.baseline2_vanilla_rag import run_baseline2
    results = asyncio.run(run_baseline2(limit=5))
"""

import json
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
LLM_MODEL = "gemini-2.5-flash"
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
    max_retries: int = 3,
    base_delay: float = 15.0,
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


def run_baseline2(
    queries: list[dict] | None = None,
    limit: int | None = None,
    delay: float = 2.0,
) -> list[dict]:
    """Run Baseline 2 evaluation: Vanilla RAG (retrieve + generate).

    Args:
        queries: Optional list of query dicts. Defaults to TEST_QUERIES.
        limit: Optional limit on number of queries to run.
        delay: Delay in seconds between queries to avoid rate limits.

    Returns:
        List of result dictionaries with logs and outputs.
    """
    queries = queries or TEST_QUERIES
    if limit:
        queries = queries[:limit]

    print(f"\n{'='*70}")
    print(f"🔬 BASELINE 2: Vanilla RAG (Retrieve + Generate)")
    print(f"   Running {len(queries)} queries...")
    print(f"{'='*70}\n")

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    chain = VANILLA_RAG_PROMPT | llm
    results = []

    for i, entry in enumerate(queries, 1):
        query_id = entry["id"]
        query_text = entry["query"]

        print(f"  [{i}/{len(queries)}] {query_id}: {query_text[:60]}...")

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

            print(f"       ✅ Done in {latency:.1f}s | Category guess: {guessed_category} | {len(retrieved_docs)} docs")

        except Exception as e:
            latency = time.time() - start_time
            guessed_category = None
            retrieved_context = ""
            generated_answer = f"(Error: {e})"
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
            "predicted_ailment": None,  # No ailment identification in vanilla RAG
            "generated_answer": generated_answer,
            "retrieved_context": retrieved_context,
            "sources": sources,
            # — Metadata —
            "latency_seconds": latency,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

        # Rate-limit delay
        if i < len(queries):
            time.sleep(delay)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  📁 Saved {len(results)} results → {LOG_FILE}")
    print(f"  ⏱️  Avg latency: {sum(r['latency_seconds'] for r in results) / len(results):.1f}s")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    run_baseline2()

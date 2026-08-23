"""
Proposed System: Therapy Guide-MAS — Full LangGraph Pipeline.

Runs each test query through the complete multi-agent system:
  Classification → Parallel [RAG (conditional) + PubMed Search] → Synthesis

Uses the existing pipeline from langgraph_bridge.create_pipeline().
No new ingestion — uses existing vector stores and the full workflow.

Supports batch evaluation with resume capability for 100+ queries.

Usage:
    from evaluation.proposed_mas import run_proposed
    results = run_proposed(limit=5)
    results = run_proposed(batch_size=5, batch_delay=30, resume=True)
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from langgraph_bridge import create_pipeline
from evaluation.test_queries import TEST_QUERIES


# ── Config ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOG_FILE = RESULTS_DIR / "proposed_logs.json"


# ── Retry Logic ──────────────────────────────────────────────────────

def _invoke_pipeline_with_retry(
    pipeline,
    params: dict,
    max_retries: int = 5,
    base_delay: float = 30.0,
):
    """Invoke a LangGraph pipeline with retry logic for 429 rate-limit errors.

    Args:
        pipeline: The LangGraph pipeline to invoke.
        params: Parameters to pass to the pipeline.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.

    Returns:
        The pipeline's result state.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return pipeline.invoke(params)
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


def run_proposed(
    queries: list[dict] | None = None,
    limit: int | None = None,
    delay: float = 3.0,
    batch_size: int = 5,
    batch_delay: float = 30.0,
    resume: bool = False,
) -> list[dict]:
    """Run Proposed System evaluation: Full LangGraph MAS pipeline.

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
    print(f"🔬 PROPOSED: Therapy Guide-MAS (Full LangGraph Pipeline)")
    print(f"   Total queries: {len(queries)} | Already completed: {len(completed_ids)} | Pending: {len(pending_queries)}")
    print(f"   Batch size: {batch_size} | Batch delay: {batch_delay}s | Query delay: {delay}s")
    print(f"{'='*70}\n")

    if not pending_queries:
        print(f"  ✅ All queries already completed. Nothing to do.")
        return existing_results

    all_results = list(existing_results)  # Start with existing results
    new_count = 0

    for i, entry in enumerate(pending_queries, 1):
        query_id = entry["id"]
        query_text = entry["query"]

        print(f"  [{i}/{len(pending_queries)}] {query_id}: {query_text[:60]}...")

        start_time = time.time()
        try:
            # Create a fresh pipeline per query for isolation
            pipeline = create_pipeline()

            result_state = _invoke_pipeline_with_retry(pipeline, {
                "query": query_text,
                "messages": [HumanMessage(content=query_text)],
            })

            latency = time.time() - start_time

            # Extract all outputs from the pipeline state
            classification = result_state.get("classification", {})
            final_output = result_state.get("final_output", {})
            synthesis_response = result_state.get("synthesis_response", "")
            rag_context = result_state.get("rag_context", "")
            rag_response = result_state.get("rag_response", "")
            web_search_summary = result_state.get("web_search_summary", "")

            predicted_category = result_state.get("category")
            predicted_ailment = result_state.get("ailment")

            # The final generated answer is the synthesis response
            generated_answer = synthesis_response or rag_response or "(No response)"

            error = None
            print(f"       ✅ Done in {latency:.1f}s | Category: {predicted_category} | Ailment: {predicted_ailment}")

        except Exception as e:
            latency = time.time() - start_time
            classification = None
            final_output = None
            predicted_category = None
            predicted_ailment = None
            generated_answer = f"(Error: {e})"
            rag_context = ""
            rag_response = ""
            web_search_summary = ""
            error = str(e)
            print(f"       ❌ Error: {e}")

        result = {
            "query_id": query_id,
            "query": query_text,
            "ground_truth_category": entry["category"],
            "ground_truth_ailment": entry["expected_ailment"],
            "reference_answer": entry["reference_answer"],
            # — Outputs —
            "classification": classification,
            "predicted_category": predicted_category,
            "predicted_ailment": predicted_ailment,
            "generated_answer": generated_answer,
            "retrieved_context": rag_context,
            "rag_response": rag_response,
            "web_search_summary": web_search_summary,
            "final_output": final_output,
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
    run_proposed()


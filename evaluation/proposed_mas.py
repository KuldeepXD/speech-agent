"""
Proposed System: Therapy Guide-MAS — Full LangGraph Pipeline.

Runs each test query through the complete multi-agent system:
  Classification → Parallel [RAG (conditional) + Web Search] → Synthesis

Uses the existing pipeline from langgraph_bridge.create_pipeline().
No new ingestion — uses existing vector stores and the full workflow.

Usage:
    from evaluation.proposed_mas import run_proposed
    results = run_proposed(limit=5)
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


def run_proposed(
    queries: list[dict] | None = None,
    limit: int | None = None,
    delay: float = 2.0,
) -> list[dict]:
    """Run Proposed System evaluation: Full LangGraph MAS pipeline.

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
    print(f"🔬 PROPOSED: Therapy Guide-MAS (Full LangGraph Pipeline)")
    print(f"   Running {len(queries)} queries...")
    print(f"{'='*70}\n")

    results = []

    for i, entry in enumerate(queries, 1):
        query_id = entry["id"]
        query_text = entry["query"]

        print(f"  [{i}/{len(queries)}] {query_id}: {query_text[:60]}...")

        start_time = time.time()
        try:
            # Create a fresh pipeline per query for isolation
            pipeline = create_pipeline()

            result_state = pipeline.invoke({
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
        results.append(result)

        # Rate-limit delay
        if i < len(queries):
            time.sleep(delay)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n  📁 Saved {len(results)} results → {LOG_FILE}")
    print(f"  ⏱️  Avg latency: {sum(r['latency_seconds'] for r in results) / len(results):.1f}s")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    run_proposed()

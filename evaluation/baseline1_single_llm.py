"""
Baseline 1: Single LLM + Web Search Tool.

Runs each test query through a single Gemini LLM call augmented with
a DuckDuckGo web search tool. No classification agent, no RAG retrieval,
no multi-agent pipeline — just one LLM with one tool.

This represents the simplest possible architecture: LLM + Tool.

Usage:
    from evaluation.baseline1_single_llm import run_baseline1
    results = run_baseline1(limit=5)
"""

import json
import time
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from duckduckgo_search import DDGS

from evaluation.test_queries import TEST_QUERIES


import os
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
LOG_FILE = RESULTS_DIR / "baseline1_logs.json"

MAX_SEARCH_RESULTS = 5  # Number of web search results to fetch


# ── Web Search Tool ───────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """Perform a web search using DuckDuckGo to find clinical information, treatments, or guidelines.
    CRITICAL: The `query` argument MUST be a concise list of keywords or a specific clinical topic (e.g., 'articulation disorder treatment 4 year old').
    Do NOT pass long sentences or the user's exact question into this tool. First identify the problem, then search the relevant keywords.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "(No search results found)"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', 'N/A')}\n"
                f"    {r.get('body', 'No snippet')}"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"(Web search failed: {e})"


# ── Single LLM Prompt ────────────────────────────────────────────────

# Single LLM Prompt is now integrated into the agent's system message


def _invoke_with_retry(
    chain,
    params: dict,
    max_retries: int = 3,
    base_delay: float = 15.0,
    config: dict = None,
):
    """Invoke an LLM chain with retry logic for 429 rate-limit errors.

    Args:
        chain: The LangChain chain to invoke.
        params: Parameters to pass to the chain.
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay in seconds between retries.
        config: Optional Langchain configuration dict (e.g. for recursion_limit).

    Returns:
        The chain's response.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(params, config=config)
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


def run_baseline1(
    queries: list[dict] | None = None,
    limit: int | None = None,
    delay: float = 2.0,
) -> list[dict]:
    """Run Baseline 1 evaluation: Single LLM + Web Search Tool.

    For each query:
      1. Performs a web search using DuckDuckGo.
      2. Passes the query + search results to a single Gemini LLM call.
      3. Logs the generated answer, web search context, and latency.

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
    print(f"🔬 BASELINE 1: Single LLM + Web Search Tool")
    print(f"   Running {len(queries)} queries...")
    print(f"{'='*70}\n")

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    tools = [web_search]
    agent_executor = create_react_agent(llm, tools)
    
    results = []

    for i, entry in enumerate(queries, 1):
        query_id = entry["id"]
        query_text = entry["query"]

        print(f"  [{i}/{len(queries)}] {query_id}: {query_text[:60]}...")

        start_time = time.time()
        try:
            # The agent decides when and how to search the web based on the prompt
            system_prompt = (
                "You are a Speech-Language Pathology clinical assistant.\n\n"
                "A user will ask you a question about a patient's condition.\n"
                "You MUST use the web_search tool to look up relevant clinical guidelines and treatments.\n"
                "CRITICAL: Do NOT use the user's full query as your search term. "
                "First, identify the core problem/symptoms through keywords, then search those relevant topics.\n\n"
                "Provide a comprehensive clinical answer that:\n"
                "1. Identifies the likely condition/ailment.\n"
                "2. Explains the condition clearly.\n"
                "3. Provides evidence-based treatment recommendations.\n"
                "4. Suggests appropriate clinical assessments.\n"
                "5. Offers practical next steps.\n\n"
                "Be clinically accurate, structured, and actionable."
            )
            
            inputs = {"messages": [("system", system_prompt), ("user", query_text)]}

            # Step 1 & 2: Agent execution (automatically handles tool calls)
            # Limit to 10 steps max to allow agent to complete tool use and reasoning
            response = _invoke_with_retry(
                agent_executor, 
                inputs,
                config={"recursion_limit": 10}
            )
            
            # Extract final answer
            messages = response.get("messages", [])
            if messages:
                generated_answer = messages[-1].content
            else:
                generated_answer = "(No response generated)"

            # Extract tool calls for context logging
            tool_messages = [m for m in messages if getattr(m, "type", "") == "tool"]
            if tool_messages:
                web_search_context = f"Performed {len(tool_messages)} searches. Output of last search:\n{tool_messages[-1].content}"
                has_results = "failed" not in tool_messages[-1].content.lower() and "no search results" not in tool_messages[-1].content.lower()
            else:
                web_search_context = "(Agent did not use search tool)"
                has_results = False
                
            print(f"       🔎 Web search used: {'✅' if len(tool_messages) > 0 else '⚠️  No'}")

            latency = time.time() - start_time
            error = None
            print(f"       ✅ Done in {latency:.1f}s")

        except Exception as e:
            latency = time.time() - start_time
            web_search_context = ""
            generated_answer = f"(Error: {e})"
            error = str(e)
            print(f"       ❌ Error: {e}")

        result = {
            "query_id": query_id,
            "query": query_text,
            "ground_truth_category": entry["category"],
            "ground_truth_ailment": entry["expected_ailment"],
            "reference_answer": entry["reference_answer"],
            # — Outputs —
            "predicted_category": None,   # No classification in Baseline 1
            "predicted_ailment": None,    # No ailment identification in Baseline 1
            "generated_answer": generated_answer,
            "retrieved_context": web_search_context,  # Web search as context
            # — Metadata —
            "latency_seconds": latency,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        results.append(result)

        # Rate-limit delay
        if i < len(queries):
            time.sleep(delay)

    dataframe = pd.DataFrame(results)

    # Format timestamp for filesystem (remove colons and special characters)
    safe_timestamp = result['timestamp'].replace(':', '-').replace('+', '_')
    dataframe.to_excel(f"baseline1_results_{safe_timestamp}.xlsx", index=False)
    print("\n  📁 Saved results → baseline1_results.xlsx")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  📁 Saved {len(results)} results → {LOG_FILE}")
    print(f"  ⏱️  Avg latency: {sum(r['latency_seconds'] for r in results) / len(results):.1f}s")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    run_baseline1()

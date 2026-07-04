"""
Metrics Computation Engine — RAGAS + Cosine Similarity + LLM Judge.

Loads logged results from all 3 systems and computes:
  - RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
  - Cosine Similarity (semantic similarity between generated and reference answers)
  - LLM Judge (Gemini rates answer quality on a 1-5 rubric)
  - Classification Accuracy (exact match)
  - Ailment Accuracy (fuzzy match)
  - Average Latency

Usage:
    from evaluation.metrics import compute_all_metrics
    metrics = compute_all_metrics()
"""

import json
import os
import random
import time
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset


# ── Config ─────────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

BASELINE1_LOG = RESULTS_DIR / "baseline1_logs.json"
BASELINE2_LOG = RESULTS_DIR / "baseline2_logs.json"
PROPOSED_LOG = RESULTS_DIR / "proposed_logs.json"
METRICS_FILE = RESULTS_DIR / "metrics_summary.json"


# ── LLM Judge Prompt ──────────────────────────────────────────────────

LLM_JUDGE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert evaluator for a Speech-Language Pathology clinical AI system.

Rate the following AI-generated answer on a scale of 1 to 5 using this rubric:

## Rubric
- **5 (Excellent)**: Clinically accurate, comprehensive, well-structured, actionable, cites evidence.
- **4 (Good)**: Mostly accurate, covers key points, minor gaps in detail or actionability.
- **3 (Adequate)**: Partially correct, addresses the query but missing important clinical details.
- **2 (Poor)**: Significant inaccuracies or very incomplete, limited clinical value.
- **1 (Very Poor)**: Incorrect, irrelevant, or harmful clinical guidance.

## Patient Query
{query}

## Reference Answer (Gold Standard)
{reference_answer}

## AI-Generated Answer (To Evaluate)
{generated_answer}

## Your Evaluation
Respond with ONLY a JSON object in this exact format:
{{"score": <1-5>, "reasoning": "<brief explanation>"}}
"""
)


def _load_results(log_file: Path) -> list[dict]:
    """Load logged results from a JSON file.

    Args:
        log_file: Path to the JSON log file.

    Returns:
        List of result dictionaries.

    Raises:
        FileNotFoundError: If the log file doesn't exist.
    """
    if not log_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {log_file}. "
            f"Run the corresponding baseline first."
        )
    with open(log_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _invoke_with_retry(chain, params: dict, max_retries: int = 3, base_delay: float = 2.0, max_delay: float = 60.0):
    """Invoke an LLM chain with retry logic for rate limits.

    Uses exponential backoff with jitter: delay = min(base_delay * 2^attempt + jitter, max_delay).

    Args:
        chain: The LLM chain to invoke.
        params: Parameters to pass to the chain.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before the first retry.
        max_delay: Upper cap on the wait time in seconds.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(params)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    jitter = random.uniform(0, 1)
                    wait_time = min(base_delay * (2 ** attempt) + jitter, max_delay)
                    print(f"       ⏳ Rate limited. Retrying in {wait_time:.1f}s... ({attempt}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise


# ── 1. RAGAS Metrics ──────────────────────────────────────────────────

def compute_ragas_metrics(results: list[dict], system_name: str) -> dict:
    """Compute RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall).

    Args:
        results: List of result dicts with query, generated_answer, retrieved_context, reference_answer.
        system_name: Name of the system for logging.

    Returns:
        Dictionary of average RAGAS metric scores.
    """
    print(f"  📊 Computing RAGAS metrics for {system_name}...")

    # Build the dataset for RAGAS
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for r in results:
        data["question"].append(r["query"])
        data["answer"].append(r.get("generated_answer", "(No answer)"))

        # Split retrieved context into a list of chunks
        ctx = r.get("retrieved_context", "")
        if ctx and ctx.strip():
            # Split on the "--- Reference" delimiter
            chunks = [
                c.strip()
                for c in ctx.split("--- Reference")
                if c.strip()
            ]
            if not chunks:
                chunks = [ctx]
        else:
            chunks = ["(No context available)"]
        data["contexts"].append(chunks)

        data["ground_truth"].append(r.get("reference_answer", ""))

    dataset = Dataset.from_dict(data)

    # Configure RAGAS with Gemini
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(model=LLM_MODEL))
    embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL))

    ragas_metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    try:
        ragas_result = evaluate(
            dataset=dataset,
            metrics=ragas_metrics,
            llm=llm,
            embeddings=embeddings,
        )

        scores = {
            "faithfulness": float(ragas_result["faithfulness"]),
            "answer_relevancy": float(ragas_result["answer_relevancy"]),
            "context_precision": float(ragas_result["context_precision"]),
            "context_recall": float(ragas_result["context_recall"]),
        }
    except Exception as e:
        print(f"       ⚠️  RAGAS evaluation failed: {e}")
        scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

    print(f"       ✅ RAGAS: {scores}")
    return scores


# ── 2. Cosine Similarity ─────────────────────────────────────────────

def compute_cosine_similarity(results: list[dict], system_name: str) -> float:
    """Compute average cosine similarity between generated and reference answers.

    Args:
        results: List of result dicts with generated_answer and reference_answer.
        system_name: Name of the system for logging.

    Returns:
        Average cosine similarity score (0-1).
    """
    print(f"  📊 Computing Cosine Similarity for {system_name}...")

    embeddings_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    generated_texts = [r.get("generated_answer", "") for r in results]
    reference_texts = [r.get("reference_answer", "") for r in results]

    # Filter out empty or error responses
    valid_pairs = [
        (gen, ref)
        for gen, ref in zip(generated_texts, reference_texts)
        if gen and ref and not gen.startswith("(Error") and not gen.startswith("(No")
    ]

    if not valid_pairs:
        print(f"       ⚠️  No valid answer pairs for cosine similarity")
        return 0.0

    gen_texts, ref_texts = zip(*valid_pairs)

    try:
        gen_embeddings = embeddings_model.embed_documents(list(gen_texts))
        ref_embeddings = embeddings_model.embed_documents(list(ref_texts))

        gen_array = np.array(gen_embeddings)
        ref_array = np.array(ref_embeddings)

        # Compute row-wise cosine similarity
        similarities = []
        for g, r in zip(gen_array, ref_array):
            sim = cosine_similarity([g], [r])[0][0]
            similarities.append(float(sim))

        avg_sim = float(np.mean(similarities))
    except Exception as e:
        print(f"       ⚠️  Cosine similarity failed: {e}")
        avg_sim = 0.0

    print(f"       ✅ Avg Cosine Similarity: {avg_sim:.4f}")
    return avg_sim


# ── 3. LLM Judge ─────────────────────────────────────────────────────

def compute_llm_judge_scores(
    results: list[dict],
    system_name: str,
    delay: float = 2.0,
) -> dict:
    """Use Gemini as an LLM judge to rate answer quality (1-5).

    Args:
        results: List of result dicts with query, generated_answer, reference_answer.
        system_name: Name of the system for logging.
        delay: Delay between API calls for rate limiting.

    Returns:
        Dictionary with average score and per-query scores.
    """
    print(f"  📊 Computing LLM Judge Scores for {system_name}...")

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    chain = LLM_JUDGE_PROMPT | llm

    scores = []
    per_query = []

    for i, r in enumerate(results):
        query = r["query"]
        generated = r.get("generated_answer", "(No answer)")
        reference = r.get("reference_answer", "")

        if generated.startswith("(Error") or generated.startswith("(No"):
            scores.append(1)
            per_query.append({
                "query_id": r["query_id"],
                "score": 1,
                "reasoning": "Error or no response generated",
            })
            continue

        try:
            response = _invoke_with_retry(chain, {
                "query": query,
                "reference_answer": reference,
                "generated_answer": generated,
            })

            response_text = response.content if hasattr(response, "content") else str(response)

            # Parse JSON from response
            # Handle markdown code blocks
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(cleaned)
            score = int(parsed.get("score", 3))
            reasoning = parsed.get("reasoning", "")

        except (json.JSONDecodeError, KeyError, ValueError):
            score = 3  # Default to middle score on parse failure
            reasoning = f"Failed to parse judge response: {response_text[:100]}"
        except Exception as e:
            score = 3
            reasoning = f"Judge evaluation failed: {e}"

        scores.append(score)
        per_query.append({
            "query_id": r["query_id"],
            "score": score,
            "reasoning": reasoning,
        })

        if i < len(results) - 1:
            time.sleep(delay)

    avg_score = float(np.mean(scores)) if scores else 0.0

    print(f"       ✅ Avg LLM Judge Score: {avg_score:.2f}/5.0")
    return {
        "average_score": avg_score,
        "per_query_scores": per_query,
    }


# ── 4. Classification & Ailment Accuracy ─────────────────────────────

def compute_classification_accuracy(results: list[dict], system_name: str) -> dict:
    """Compute classification accuracy (exact match) and ailment accuracy (fuzzy).

    Args:
        results: List of result dicts with predicted/ground-truth categories and ailments.
        system_name: Name of the system for logging.

    Returns:
        Dictionary with classification_accuracy and ailment_accuracy.
    """
    print(f"  📊 Computing Classification & Ailment Accuracy for {system_name}...")

    cat_correct = 0
    cat_total = 0
    ail_correct = 0
    ail_total = 0

    for r in results:
        gt_cat = r.get("ground_truth_category", "")
        pred_cat = r.get("predicted_category", "")

        if gt_cat and pred_cat:
            cat_total += 1
            if gt_cat.lower().strip() == pred_cat.lower().strip():
                cat_correct += 1

        gt_ail = r.get("ground_truth_ailment", "")
        pred_ail = r.get("predicted_ailment", "")

        if gt_ail and pred_ail:
            ail_total += 1
            # Fuzzy match: check if ground truth is contained in prediction or vice versa
            gt_lower = gt_ail.lower().strip()
            pred_lower = pred_ail.lower().strip()
            if (
                gt_lower == pred_lower
                or gt_lower in pred_lower
                or pred_lower in gt_lower
            ):
                ail_correct += 1

    cat_acc = cat_correct / cat_total if cat_total > 0 else None
    ail_acc = ail_correct / ail_total if ail_total > 0 else None

    print(f"       ✅ Classification: {cat_correct}/{cat_total} = {cat_acc:.2%}" if cat_acc is not None else f"       ⚠️  No classification predictions")
    print(f"       ✅ Ailment:        {ail_correct}/{ail_total} = {ail_acc:.2%}" if ail_acc is not None else f"       ⚠️  No ailment predictions")

    return {
        "classification_accuracy": float(cat_acc) if cat_acc is not None else None,
        "classification_correct": cat_correct,
        "classification_total": cat_total,
        "ailment_accuracy": float(ail_acc) if ail_acc is not None else None,
        "ailment_correct": ail_correct,
        "ailment_total": ail_total,
    }


# ── 5. Average Latency ───────────────────────────────────────────────

def compute_avg_latency(results: list[dict], system_name: str) -> float:
    """Compute average latency across all queries.

    Args:
        results: List of result dicts with latency_seconds.
        system_name: Name of the system for logging.

    Returns:
        Average latency in seconds.
    """
    latencies = [r.get("latency_seconds", 0) for r in results]
    avg = float(np.mean(latencies)) if latencies else 0.0
    print(f"  📊 Avg Latency for {system_name}: {avg:.1f}s")
    return avg


# ── Main: Compute All Metrics ─────────────────────────────────────────

def compute_all_metrics() -> dict:
    """Compute all metrics for all 3 systems and save summary.

    Returns:
        Dictionary of metrics for all systems.
    """
    print(f"\n{'='*70}")
    print(f"📊 METRICS ENGINE: Computing Evaluation Metrics")
    print(f"{'='*70}\n")

    all_metrics = {}

    systems = [
        ("Baseline 1: Single LLM + Tool", BASELINE1_LOG),
        ("Baseline 2: Vanilla RAG", BASELINE2_LOG),
        ("Proposed: Therapy Guide-MAS", PROPOSED_LOG),
    ]

    for system_name, log_file in systems:
        print(f"\n{'─'*60}")
        print(f"  📋 {system_name}")
        print(f"{'─'*60}")

        try:
            results = _load_results(log_file)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping — {e}")
            all_metrics[system_name] = {"error": str(e)}
            continue

        metrics = {}

        # 1. RAGAS
        try:
            metrics["ragas"] = compute_ragas_metrics(results, system_name)
        except Exception as e:
            print(f"  ⚠️  RAGAS failed: {e}")
            metrics["ragas"] = {"error": str(e)}

        # 2. Cosine Similarity
        try:
            metrics["cosine_similarity"] = compute_cosine_similarity(results, system_name)
        except Exception as e:
            print(f"  ⚠️  Cosine sim failed: {e}")
            metrics["cosine_similarity"] = None

        # 3. LLM Judge
        try:
            metrics["llm_judge"] = compute_llm_judge_scores(results, system_name)
        except Exception as e:
            print(f"  ⚠️  LLM Judge failed: {e}")
            metrics["llm_judge"] = {"error": str(e)}

        # 4. Classification & Ailment Accuracy
        metrics["accuracy"] = compute_classification_accuracy(results, system_name)

        # 5. Latency
        metrics["avg_latency_seconds"] = compute_avg_latency(results, system_name)

        # 6. Query count
        metrics["num_queries"] = len(results)

        all_metrics[system_name] = metrics

    # Save metrics summary
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  📁 Metrics saved → {METRICS_FILE}")
    print(f"{'='*70}\n")

    return all_metrics


if __name__ == "__main__":
    compute_all_metrics()

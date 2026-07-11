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
import math
import os
import random
import time
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

def _clean_message_content(message):
    if hasattr(message, "content"):
        content = message.content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            message.content = "\n".join(parts)
    return message

class SafeChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """Wrapper that normalises list-format content blocks to a plain string.

    Newer langchain-google-genai returns message.content as a list of
    {"type": "text", "text": "..."} dicts.  RAGAS and the LLM Judge both
    call float() / .strip() on that value and crash.  We patch every exit
    point: _generate, _agenerate, invoke, and ainvoke.
    """

    @staticmethod
    def _normalise(message):
        """Flatten list content blocks to a single string in-place."""
        if hasattr(message, "content") and isinstance(message.content, list):
            parts = [
                p["text"] if isinstance(p, dict) and "text" in p else str(p)
                for p in message.content
            ]
            message.content = "\n".join(parts)
        return message

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if result and result.generations:
            for gen in result.generations:
                if hasattr(gen, "message"):
                    self._normalise(gen.message)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        if result and result.generations:
            for gen in result.generations:
                if hasattr(gen, "message"):
                    self._normalise(gen.message)
        return result

    def invoke(self, input, config=None, **kwargs):
        result = super().invoke(input, config=config, **kwargs)
        return self._normalise(result)

    async def ainvoke(self, input, config=None, **kwargs):
        result = await super().ainvoke(input, config=config, **kwargs)
        return self._normalise(result)



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
SKIP_LLM_CALLS = False

BASELINE1_LOG = RESULTS_DIR / "baseline1_logs.json"
BASELINE2_LOG = RESULTS_DIR / "baseline2_logs.json"
PROPOSED_LOG = RESULTS_DIR / "proposed_logs.json"
METRICS_FILE = RESULTS_DIR / "metrics_summary.json"


# ── API Key Rotation ──────────────────────────────────────────────────
# Load all available keys: GOOGLE_API_KEY, GOOGLE_API_KEY_2, GOOGLE_API_KEY_3 ...

def _load_api_keys() -> list[str]:
    keys = []
    primary = os.getenv("GOOGLE_API_KEY")
    if primary:
        keys.append(primary)
    i = 2
    while True:
        k = os.getenv(f"GOOGLE_API_KEY_{i}")
        if k:
            keys.append(k)
            i += 1
        else:
            break
    return keys

_API_KEYS: list[str] = _load_api_keys()
_KEY_INDEX: list[int] = [0]  # mutable container so nested functions can mutate it


def _rotate_api_key() -> bool:
    """Switch to the next available API key. Returns True if a new key was set."""
    _load_api_keys()  # reload from env in case user updated .env
    fresh_keys = _load_api_keys()
    if not fresh_keys:
        print("  ❌ No API keys found in environment.")
        return False
    global _API_KEYS
    _API_KEYS = fresh_keys
    next_idx = (_KEY_INDEX[0] + 1) % len(_API_KEYS)
    _KEY_INDEX[0] = next_idx
    os.environ["GOOGLE_API_KEY"] = _API_KEYS[next_idx]
    print(f"  🔄 Rotated to API key slot #{next_idx + 1} (of {len(_API_KEYS)} available)")
    return True


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    if isinstance(exc, TimeoutError) or "timeout" in s:
        return True
    return any(tok in s for tok in ("429", "resource_exhausted", "quota", "rate limit"))


# ── Cache helpers ─────────────────────────────────────────────────────

def _system_slug(system_name: str) -> str:
    return (
        system_name.lower()
        .replace(" ", "_")
        .replace(":", "")
        .replace("+", "plus")
        .replace("-", "")
    )


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _safe_float(val) -> float | None:
    """Convert to float, treating NaN and None as None for JSON compatibility."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


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
        data = json.load(f)

    # Normalize generated_answer to a plain string if it is a list of blocks
    for r in data:
        gen = r.get("generated_answer")
        if isinstance(gen, list):
            parts = []
            for part in gen:
                if isinstance(part, dict) and "text" in part:
                    parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            r["generated_answer"] = "\n".join(parts)
        elif gen is None:
            r["generated_answer"] = ""

    return data


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


# ── 1. RAGAS Metrics (per-query, resumable, key-rotating) ─────────────

def _build_single_ragas_dataset(r: dict) -> "Dataset":
    """Build a single-row RAGAS Dataset from one result dict."""
    ctx = r.get("retrieved_context", "")
    if ctx and ctx.strip():
        chunks = [c.strip() for c in ctx.split("--- Reference") if c.strip()]
        if not chunks:
            chunks = [ctx]
    else:
        chunks = ["(No context available)"]
    return Dataset.from_dict({
        "question": [r["query"]],
        "answer":   [r.get("generated_answer", "(No answer)")],
        "contexts": [chunks],
        "ground_truth": [r.get("reference_answer", "")],
    })


def compute_ragas_metrics(results: list[dict], system_name: str) -> dict:
    """Compute RAGAS metrics per-query with cache + API key rotation.

    Progress is saved to results/ragas_cache_<slug>.json after every query.
    On restart, already-evaluated queries are loaded from cache and skipped.
    On rate-limit, the next available API key is tried automatically.

    Args:
        results: List of result dicts.
        system_name: Name of the system for logging.

    Returns:
        Dictionary of average RAGAS metric scores.
    """
    print(f"  📊 Computing RAGAS metrics for {system_name}...")

    slug = _system_slug(system_name)
    cache_path = RESULTS_DIR / f"ragas_cache_{slug}.json"
    cache = _load_cache(cache_path)

    pending = [r for r in results if r["query_id"] not in cache]
    total = len(results)
    cached_count = total - len(pending)

    if not pending:
        print(f"    ✅ All {total} queries already in RAGAS cache — loading from cache")
    else:
        print(f"    ♻️  Resuming: {cached_count}/{total} cached, {len(pending)} remaining")

    ragas_metric_list = [faithfulness, answer_relevancy, context_precision, context_recall]

    for idx, r in enumerate(pending):
        qid = r["query_id"]
        success = False
        max_key_attempts = max(len(_API_KEYS) * 2, 4)

        for attempt in range(max_key_attempts):
            try:
                llm = LangchainLLMWrapper(SafeChatGoogleGenerativeAI(model=LLM_MODEL))
                emb = LangchainEmbeddingsWrapper(
                    GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
                )
                single_ds = _build_single_ragas_dataset(r)
                eval_result = evaluate(
                    dataset=single_ds,
                    metrics=ragas_metric_list,
                    llm=llm,
                    embeddings=emb,
                )
                # RAGAS v0.4+ returns per-row scores as lists (e.g. [0.0])
                # Extract [0] since we evaluate one query at a time
                def _extract(key):
                    v = eval_result[key]
                    if isinstance(v, list):
                        v = v[0] if v else None
                    val = _safe_float(v)
                    if val is None:
                        raise TimeoutError(f"Ragas evaluation returned NaN/None for {key}")
                    return val

                cache[qid] = {
                    "faithfulness":       _extract("faithfulness"),
                    "answer_relevancy":   _extract("answer_relevancy"),
                    "context_precision":  _extract("context_precision"),
                    "context_recall":     _extract("context_recall"),
                }
                # Debug: show raw RAGAS output
                raw = {k: eval_result[k] for k in ["faithfulness","answer_relevancy","context_precision","context_recall"]}
                print(f"      [DEBUG] raw RAGAS for {qid}: {raw}")
                _save_cache(cache_path, cache)
                pct = int((cached_count + idx + 1) / total * 100)
                print(f"    [{cached_count + idx + 1}/{total} {pct}%] ✅ {qid}: {cache[qid]}")
                success = True
                break

            except Exception as e:
                if _is_rate_limit(e):
                    print(f"    ⚠️  Rate limited on {qid} (attempt {attempt+1}). "
                          f"Waiting 60s then rotating key...")
                    time.sleep(60)
                    _rotate_api_key()
                else:
                    print(f"    ⚠️  {qid} failed ({type(e).__name__}): {e}")
                    cache[qid] = {
                        "faithfulness": None, "answer_relevancy": None,
                        "context_precision": None, "context_recall": None,
                    }
                    _save_cache(cache_path, cache)
                    success = True
                    break

        if not success:
            print(f"    ❌ Gave up on {qid} after {max_key_attempts} attempts")
            cache[qid] = {
                "faithfulness": None, "answer_relevancy": None,
                "context_precision": None, "context_recall": None,
            }
            _save_cache(cache_path, cache)

        # Small delay between queries to stay within 15 rpm
        time.sleep(4)

    # Aggregate averages from cache
    def _avg(key: str) -> float | None:
        vals = [v[key] for v in cache.values() if v.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    scores = {
        "faithfulness":      _avg("faithfulness"),
        "answer_relevancy":  _avg("answer_relevancy"),
        "context_precision": _avg("context_precision"),
        "context_recall":    _avg("context_recall"),
    }
    print(f"    ✅ RAGAS aggregate ({len(cache)} queries): {scores}")
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


# ── 3. LLM Judge (per-query, resumable, key-rotating) ────────────────

def compute_llm_judge_scores(
    results: list[dict],
    system_name: str,
    delay: float = 4.0,
) -> dict:
    """Use Gemini as an LLM judge to rate answer quality (1-5).

    Progress is saved to results/judge_cache_<slug>.json after every query.
    On restart, already-scored queries are loaded and skipped.
    On rate-limit, the next available API key is tried automatically.

    Args:
        results: List of result dicts with query, generated_answer, reference_answer.
        system_name: Name of the system for logging.
        delay: Delay between API calls for rate limiting.

    Returns:
        Dictionary with average score and per-query scores.
    """
    print(f"  📊 Computing LLM Judge Scores for {system_name}...")

    slug = _system_slug(system_name)
    cache_path = RESULTS_DIR / f"judge_cache_{slug}.json"
    cache: dict[str, dict] = _load_cache(cache_path)  # {query_id: {score, reasoning}}

    pending = [r for r in results if r["query_id"] not in cache]
    total = len(results)
    cached_count = total - len(pending)

    if not pending:
        print(f"    ✅ All {total} queries already in judge cache")
    else:
        print(f"    ♻️  Resuming judge: {cached_count}/{total} cached, {len(pending)} remaining")

    for idx, r in enumerate(pending):
        qid = r["query_id"]
        generated = r.get("generated_answer", "(No answer)")

        # Skip error responses
        if not generated or generated.startswith("(Error") or generated.startswith("(No"):
            cache[qid] = {"score": 1, "reasoning": "Error or no response generated"}
            _save_cache(cache_path, cache)
            continue

        max_key_attempts = max(len(_API_KEYS) * 2, 4)
        for attempt in range(max_key_attempts):
            try:
                llm = SafeChatGoogleGenerativeAI(model=LLM_MODEL)
                chain = LLM_JUDGE_PROMPT | llm
                response = chain.invoke({
                    "query": r["query"],
                    "reference_answer": r.get("reference_answer", ""),
                    "generated_answer": generated,
                })

                raw_content = response.content if hasattr(response, "content") else response
                if isinstance(raw_content, list):
                    parts = [
                        p["text"] if isinstance(p, dict) and "text" in p else str(p)
                        for p in raw_content
                    ]
                    response_text = "\n".join(parts)
                else:
                    response_text = str(raw_content)

                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

                parsed = json.loads(cleaned)
                score = int(parsed.get("score", 3))
                reasoning = parsed.get("reasoning", "")
                cache[qid] = {"score": score, "reasoning": reasoning}
                _save_cache(cache_path, cache)
                pct = int((cached_count + idx + 1) / total * 100)
                print(f"    [{cached_count + idx + 1}/{total} {pct}%] ✅ {qid}: {score}/5")
                break

            except Exception as e:
                if _is_rate_limit(e):
                    print(f"    ⚠️  Rate limited on judge {qid} (attempt {attempt+1}). "
                          f"Waiting 60s then rotating key...")
                    time.sleep(60)
                    _rotate_api_key()
                else:
                    print(f"    ⚠️  Judge {qid} failed: {e}")
                    cache[qid] = {"score": 3, "reasoning": f"Failed: {e}"}
                    _save_cache(cache_path, cache)
                    break

        time.sleep(delay)

    # Rebuild per-query list in original order
    per_query = [
        {"query_id": r["query_id"], **cache.get(r["query_id"], {"score": 3, "reasoning": "Not evaluated"})}
        for r in results
    ]
    scores = [entry["score"] for entry in per_query]
    avg_score = float(np.mean(scores)) if scores else 0.0

    print(f"    ✅ Avg LLM Judge Score: {avg_score:.2f}/5.0")
    return {
        "average_score": avg_score,
        "per_query_scores": per_query,
    }


# ── 4. Classification Precision / Recall / F1 & Ailment Accuracy ──────

def compute_classification_metrics(results: list[dict], system_name: str) -> dict:
    """Compute per-class precision, recall, F1 for category and fuzzy ailment accuracy.

    Missing/None predictions are treated as \"MISSING\" so they penalise recall
    correctly (rather than being silently excluded from the denominator).

    Args:
        results: List of result dicts with predicted/ground-truth categories and ailments.
        system_name: Name of the system for logging.

    Returns:
        Dictionary with per-class + macro precision/recall/F1 and ailment accuracy.
    """
    print(f"  📊 Computing Classification Precision/Recall/F1 for {system_name}...")

    # ── Category: per-class precision / recall / F1 ──────────────────
    y_true_cat, y_pred_cat = [], []
    for r in results:
        gt_cat = (r.get("ground_truth_category") or "").strip().lower()
        pred_cat = (r.get("predicted_category") or "missing").strip().lower()
        if gt_cat:  # only rows that have a ground-truth label
            y_true_cat.append(gt_cat)
            y_pred_cat.append(pred_cat)  # None/missing counts as wrong

    if y_true_cat:
        labels = sorted(set(y_true_cat))
        precision_vals, recall_vals, f1_vals, support_vals = precision_recall_fscore_support(
            y_true_cat, y_pred_cat, labels=labels, zero_division=0
        )
        per_class = {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, precision_vals, recall_vals, f1_vals, support_vals)
        }
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true_cat, y_pred_cat, labels=labels, average="macro", zero_division=0
        )
        cat_correct = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == p)
        cat_acc = cat_correct / len(y_true_cat)

        print(f"       ✅ Category Accuracy : {cat_acc:.2%}")
        print(f"       ✅ Macro Precision   : {float(macro_p):.4f}")
        print(f"       ✅ Macro Recall      : {float(macro_r):.4f}")
        print(f"       ✅ Macro F1          : {float(macro_f1):.4f}")
        for lbl, stats in per_class.items():
            print(f"          [{lbl}] P={stats['precision']:.2f}  R={stats['recall']:.2f}  F1={stats['f1']:.2f}  n={stats['support']}")
    else:
        per_class = {}
        macro_p = macro_r = macro_f1 = None
        cat_acc = None
        cat_correct = 0
        print(f"       ⚠️  No category ground-truth found — skipping classification metrics")

    # ── Ailment: fuzzy accuracy ───────────────────────────────────────
    ail_correct = 0
    ail_total = 0
    for r in results:
        gt_ail = r.get("ground_truth_ailment", "")
        pred_ail = r.get("predicted_ailment", "")

        if gt_ail:  # only rows with a ground-truth ailment
            ail_total += 1
            if pred_ail:  # prediction exists — fuzzy containment match
                gt_lower = gt_ail.lower().strip()
                pred_lower = pred_ail.lower().strip()
                if gt_lower == pred_lower or gt_lower in pred_lower or pred_lower in gt_lower:
                    ail_correct += 1
            # else: pred_ail is None/empty → counts as a miss (ail_correct not incremented)

    ail_acc = ail_correct / ail_total if ail_total > 0 else None
    if ail_acc is not None:
        print(f"       ✅ Ailment Accuracy  : {ail_correct}/{ail_total} = {ail_acc:.2%}")
    else:
        print(f"       ⚠️  No ailment ground-truth rows found")

    return {
        # Backward-compatible keys
        "classification_accuracy": float(cat_acc) if cat_acc is not None else None,
        "classification_correct": cat_correct,
        "classification_total": len(y_true_cat),
        "ailment_accuracy": float(ail_acc) if ail_acc is not None else None,
        "ailment_correct": ail_correct,
        "ailment_total": ail_total,
        # New precision / recall / F1 keys
        "macro_precision": float(macro_p) if macro_p is not None else None,
        "macro_recall": float(macro_r) if macro_r is not None else None,
        "macro_f1": float(macro_f1) if macro_f1 is not None else None,
        "per_class": per_class,
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


# ── Main: Compute All Metrics (with per-system resume) ────────────────

def _system_is_complete(m: dict) -> bool:
    """Return True if a system's metrics dict has all expected keys with no top-level error."""
    if not m or "error" in m:
        return False
    required = {"ragas", "cosine_similarity", "llm_judge", "accuracy", "avg_latency_seconds"}
    return required.issubset(m.keys())


def _save_metrics(all_metrics: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)


def compute_all_metrics(force: bool = False) -> dict:
    """Compute all metrics for all 3 systems and save summary.

    Supports resume: if metrics_summary.json already contains complete results
    for a system, that system is skipped. Per-query RAGAS and LLM Judge caches
    allow resuming mid-system across restarts or key changes.

    Args:
        force: If True, recompute all systems even if already cached.

    Returns:
        Dictionary of metrics for all systems.
    """
    print(f"\n{'='*70}")
    print(f"📊 METRICS ENGINE: Computing Evaluation Metrics")
    if _API_KEYS:
        print(f"   🔑 {len(_API_KEYS)} API key(s) loaded")
    print(f"{'='*70}\n")

    # Load existing results for per-system resume
    all_metrics: dict = {}
    if METRICS_FILE.exists() and not force:
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        except Exception:
            all_metrics = {}

    systems = [
        ("Baseline 1: Single LLM + Tool", BASELINE1_LOG),
        ("Baseline 2: Vanilla RAG",        BASELINE2_LOG),
        ("Proposed: Therapy Guide-MAS",    PROPOSED_LOG),
    ]

    for system_name, log_file in systems:
        print(f"\n{'─'*60}")
        print(f"  📋 {system_name}")
        print(f"{'─'*60}")

        # ── Per-system resume ────────────────────────────────────────
        existing = all_metrics.get(system_name, {})
        if not force and _system_is_complete(existing):
            print(f"  ✅ Already fully computed — skipping "
                  f"(delete {METRICS_FILE.name} or use force=True to recompute)")
            continue

        try:
            results = _load_results(log_file)
        except FileNotFoundError as e:
            print(f"  ⚠️  Skipping — {e}")
            all_metrics[system_name] = {"error": str(e)}
            _save_metrics(all_metrics)
            continue

        metrics = {}

        # 1. RAGAS
        if SKIP_LLM_CALLS:
            print("  ⏩ Skipping RAGAS metrics (SKIP_LLM_CALLS is True)")
            metrics["ragas"] = {
                "faithfulness": None, "answer_relevancy": None,
                "context_precision": None, "context_recall": None,
            }
        else:
            try:
                metrics["ragas"] = compute_ragas_metrics(results, system_name)
            except Exception as e:
                print(f"  ⚠️  RAGAS failed: {e}")
                metrics["ragas"] = {"error": str(e)}

        # 2. Cosine Similarity
        if SKIP_LLM_CALLS:
            print("  ⏩ Skipping Cosine Similarity (SKIP_LLM_CALLS is True)")
            metrics["cosine_similarity"] = None
        else:
            try:
                metrics["cosine_similarity"] = compute_cosine_similarity(results, system_name)
            except Exception as e:
                print(f"  ⚠️  Cosine sim failed: {e}")
                metrics["cosine_similarity"] = None

        # 3. LLM Judge
        if SKIP_LLM_CALLS:
            print("  ⏩ Skipping LLM Judge (SKIP_LLM_CALLS is True)")
            metrics["llm_judge"] = {"average_score": None}
        else:
            try:
                metrics["llm_judge"] = compute_llm_judge_scores(results, system_name)
            except Exception as e:
                print(f"  ⚠️  LLM Judge failed: {e}")
                metrics["llm_judge"] = {"error": str(e)}

        # 4. Classification Precision / Recall / F1 & Ailment Accuracy
        metrics["accuracy"] = compute_classification_metrics(results, system_name)

        # 5. Latency
        metrics["avg_latency_seconds"] = compute_avg_latency(results, system_name)

        # 6. Query count
        metrics["num_queries"] = len(results)

        all_metrics[system_name] = metrics

        # ── Incremental save after every system ──────────────────────
        _save_metrics(all_metrics)
        print(f"  💾 Progress saved → {METRICS_FILE}")

    print(f"\n{'='*70}")
    print(f"  📁 Final metrics saved → {METRICS_FILE}")
    print(f"{'='*70}\n")

    return all_metrics


if __name__ == "__main__":
    compute_all_metrics()

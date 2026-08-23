"""
Report Generator — Performance Tables & Graphs.

Loads computed metrics from metrics_summary.json and generates:
  - Console comparison table
  - CSV export
  - Bar charts comparing all 3 systems
  - Radar chart for multi-dimensional comparison

Usage:
    from evaluation.generate_report import generate_report
    generate_report()
"""

import json
import csv
from pathlib import Path

import os
import numpy as np
import matplotlib
# Use non-interactive backend only when no display is available (e.g. CI, headless servers).
# This preserves GUI backends (TkAgg, Qt5Agg, etc.) for interactive/local use.
if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ─────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent / "results"
METRICS_FILE = RESULTS_DIR / "metrics_summary.json"

# Short display names for systems
SYSTEM_LABELS = {
    "Baseline 1: Single LLM + Tool": "B1: LLM+Tool",
    "Baseline 2: Vanilla RAG": "B2: Vanilla RAG",
    "Proposed: Therapy Guide-MAS": "Proposed: MAS",
}

# Color palette
COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]


def _load_metrics() -> dict:
    """Load the metrics summary JSON.

    Returns:
        Dictionary of metrics for all systems.

    Raises:
        FileNotFoundError: If the metrics file doesn't exist.
    """
    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {METRICS_FILE}. "
            f"Run --metrics first."
        )
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_get(metrics: dict, *keys, default=0.0):
    """Safely navigate nested dict keys.

    Args:
        metrics: The dictionary to navigate.
        *keys: Sequence of keys to follow.
        default: Default value if any key is missing.

    Returns:
        The value at the nested key path, or default.
    """
    current = metrics
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            return default
    return current if current is not None else default


def print_comparison_table(all_metrics: dict) -> None:
    """Print a formatted comparison table to the console.

    Args:
        all_metrics: Dictionary of metrics for all systems.
    """
    print(f"\n{'='*90}")
    print(f"  📊 PERFORMANCE COMPARISON TABLE")
    print(f"{'='*90}")

    # Define metrics to display
    metric_rows = [
        ("Faithfulness (RAGAS)", lambda m: _safe_get(m, "ragas", "faithfulness")),
        ("Answer Relevancy (RAGAS)", lambda m: _safe_get(m, "ragas", "answer_relevancy")),
        ("Context Precision (RAGAS)", lambda m: _safe_get(m, "ragas", "context_precision")),
        ("Context Recall (RAGAS)", lambda m: _safe_get(m, "ragas", "context_recall")),
        ("Cosine Similarity", lambda m: _safe_get(m, "cosine_similarity")),
        ("LLM Judge (1-5)", lambda m: _safe_get(m, "llm_judge", "average_score")),
        ("Classification Acc.", lambda m: _safe_get(m, "accuracy", "classification_accuracy")),
        ("Category Precision", lambda m: _safe_get(m, "accuracy", "macro_precision")),
        ("Category Recall", lambda m: _safe_get(m, "accuracy", "macro_recall")),
        ("Category F1", lambda m: _safe_get(m, "accuracy", "macro_f1")),
        ("Ailment Acc.", lambda m: _safe_get(m, "accuracy", "ailment_accuracy")),
        ("Avg Latency (s)", lambda m: _safe_get(m, "avg_latency_seconds")),
        ("Num Queries", lambda m: _safe_get(m, "num_queries", default=0)),
    ]

    # Header
    systems = list(all_metrics.keys())
    labels = [SYSTEM_LABELS.get(s, s[:20]) for s in systems]

    header = f"  {'Metric':<30}" + "".join(f"  {l:>18}" for l in labels)
    print(header)
    print(f"  {'─'*30}" + "".join(f"  {'─'*18}" for _ in labels))

    # Rows
    for metric_name, extractor in metric_rows:
        values = []
        for sys_name in systems:
            m = all_metrics.get(sys_name, {})
            if "error" in m:
                values.append("N/A")
            else:
                val = extractor(m)
                if val is None:
                    values.append("N/A")
                elif isinstance(val, float):
                    if metric_name == "LLM Judge (1-5)":
                        values.append(f"{val:.2f}")
                    elif metric_name == "Avg Latency (s)":
                        values.append(f"{val:.1f}")
                    elif val <= 1.0:
                        values.append(f"{val:.4f}")
                    else:
                        values.append(f"{val:.2f}")
                else:
                    values.append(str(val))

        row = f"  {metric_name:<30}" + "".join(f"  {v:>18}" for v in values)
        print(row)

    print(f"{'='*90}\n")


def export_csv(all_metrics: dict) -> Path:
    """Export metrics to a CSV file.

    Args:
        all_metrics: Dictionary of metrics for all systems.

    Returns:
        Path to the generated CSV file.
    """
    csv_path = RESULTS_DIR / "metrics_comparison.csv"

    systems = list(all_metrics.keys())
    metric_keys = [
        ("Faithfulness", "ragas", "faithfulness"),
        ("Answer Relevancy", "ragas", "answer_relevancy"),
        ("Context Precision", "ragas", "context_precision"),
        ("Context Recall", "ragas", "context_recall"),
        ("Cosine Similarity", "cosine_similarity"),
        ("LLM Judge Score", "llm_judge", "average_score"),
        ("Classification Accuracy", "accuracy", "classification_accuracy"),
        ("Category Precision (macro)", "accuracy", "macro_precision"),
        ("Category Recall (macro)", "accuracy", "macro_recall"),
        ("Category F1 (macro)", "accuracy", "macro_f1"),
        ("Ailment Accuracy", "accuracy", "ailment_accuracy"),
        ("Avg Latency (s)", "avg_latency_seconds"),
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Metric"] + [SYSTEM_LABELS.get(s, s) for s in systems]
        writer.writerow(header)

        for row_def in metric_keys:
            name = row_def[0]
            keys = row_def[1:]
            row = [name]
            for sys_name in systems:
                m = all_metrics.get(sys_name, {})
                val = _safe_get(m, *keys)
                row.append(val if val is not None else "N/A")
            writer.writerow(row)

    print(f"  📁 CSV exported → {csv_path}")
    return csv_path


def generate_bar_chart(all_metrics: dict) -> Path:
    """Generate a grouped bar chart comparing all 3 systems.

    Args:
        all_metrics: Dictionary of metrics for all systems.

    Returns:
        Path to the generated chart image.
    """
    chart_path = RESULTS_DIR / "bar_chart_comparison.png"

    systems = list(all_metrics.keys())
    labels = [SYSTEM_LABELS.get(s, s) for s in systems]

    # Metrics to plot (0-1 scale)
    metric_names = [
        "Faithfulness",
        "Ans. Relevancy",
        "Ctx. Precision",
        "Ctx. Recall",
        "Cosine Sim.",
    ]
    metric_extractors = [
        lambda m: _safe_get(m, "ragas", "faithfulness"),
        lambda m: _safe_get(m, "ragas", "answer_relevancy"),
        lambda m: _safe_get(m, "ragas", "context_precision"),
        lambda m: _safe_get(m, "ragas", "context_recall"),
        lambda m: _safe_get(m, "cosine_similarity"),
    ]

    # Extract values
    data = []
    for extractor in metric_extractors:
        row = []
        for sys_name in systems:
            m = all_metrics.get(sys_name, {})
            val = extractor(m)
            row.append(val if isinstance(val, (int, float)) else 0)
        data.append(row)

    data = np.array(data)
    x = np.arange(len(metric_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, color) in enumerate(zip(labels, COLORS)):
        bars = ax.bar(x + i * width, data[:, i], width, label=label, color=color, edgecolor="white", linewidth=0.5)
        # Add value labels on bars
        for bar, val in zip(bars, data[:, i]):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8,
                )

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score (0-1)", fontsize=12)
    ax.set_title("Benchmark Comparison: RAGAS & Similarity Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  📊 Bar chart saved → {chart_path}")
    return chart_path


def generate_radar_chart(all_metrics: dict) -> Path:
    """Generate a radar/spider chart for multi-dimensional comparison.

    Args:
        all_metrics: Dictionary of metrics for all systems.

    Returns:
        Path to the generated chart image.
    """
    chart_path = RESULTS_DIR / "radar_chart_comparison.png"

    systems = list(all_metrics.keys())
    labels = [SYSTEM_LABELS.get(s, s) for s in systems]

    # Metrics for radar (normalize to 0-1)
    metric_names = [
        "Faithfulness",
        "Ans. Relevancy",
        "Ctx. Precision",
        "Ctx. Recall",
        "Cosine Sim.",
        "LLM Judge\n(norm)",
    ]

    def _extract_values(m):
        return [
            _safe_get(m, "ragas", "faithfulness"),
            _safe_get(m, "ragas", "answer_relevancy"),
            _safe_get(m, "ragas", "context_precision"),
            _safe_get(m, "ragas", "context_recall"),
            _safe_get(m, "cosine_similarity"),
            _safe_get(m, "llm_judge", "average_score") / 5.0,  # Normalize 1-5 to 0-1
        ]

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (sys_name, label, color) in enumerate(zip(systems, labels, COLORS)):
        m = all_metrics.get(sys_name, {})
        if "error" in m:
            continue
        values = _extract_values(m)
        values = [v if isinstance(v, (int, float)) else 0 for v in values]
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Dimensional Performance Radar", fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  📊 Radar chart saved → {chart_path}")
    return chart_path


def generate_llm_judge_bar_chart(all_metrics: dict) -> Path:
    """Generate a bar chart specifically for LLM Judge and Latency metrics.

    Args:
        all_metrics: Dictionary of metrics for all systems.

    Returns:
        Path to the generated chart image.
    """
    chart_path = RESULTS_DIR / "judge_latency_chart.png"

    systems = list(all_metrics.keys())
    labels = [SYSTEM_LABELS.get(s, s) for s in systems]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # LLM Judge Scores
    judge_scores = []
    for sys_name in systems:
        m = all_metrics.get(sys_name, {})
        score = _safe_get(m, "llm_judge", "average_score")
        judge_scores.append(score if isinstance(score, (int, float)) else 0)

    bars1 = ax1.bar(labels, judge_scores, color=COLORS, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars1, judge_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (1-5)", fontsize=12)
    ax1.set_title("LLM Judge Quality Score", fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 5.5)
    ax1.grid(axis="y", alpha=0.3)

    # Latency
    latencies = []
    for sys_name in systems:
        m = all_metrics.get(sys_name, {})
        lat = _safe_get(m, "avg_latency_seconds")
        latencies.append(lat if isinstance(lat, (int, float)) else 0)

    bars2 = ax2.bar(labels, latencies, color=COLORS, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Seconds", fontsize=12)
    ax2.set_title("Average Response Latency", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  📊 Judge/Latency chart saved → {chart_path}")
    return chart_path


def generate_report() -> None:
    """Generate the full performance report: table + CSV + charts.

    Loads metrics_summary.json and produces all outputs.
    """
    print(f"\n{'='*70}")
    print(f"📊 REPORT GENERATOR: Building Performance Report")
    print(f"{'='*70}\n")

    all_metrics = _load_metrics()

    # 1. Console table
    print_comparison_table(all_metrics)

    # 2. CSV export
    export_csv(all_metrics)

    # 3. Bar chart
    generate_bar_chart(all_metrics)

    # 4. Radar chart
    generate_radar_chart(all_metrics)

    # 5. Judge + Latency chart
    generate_llm_judge_bar_chart(all_metrics)

    print(f"\n  ✅ All reports generated in: {RESULTS_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    generate_report()

"""
Evaluation Orchestrator — CLI Entry Point.

Runs the benchmarking pipeline end-to-end or individual components.

Usage:
    python -m evaluation.run_evaluation --all              # Run everything
    python -m evaluation.run_evaluation --baseline1        # Only Baseline 1
    python -m evaluation.run_evaluation --baseline2        # Only Baseline 2
    python -m evaluation.run_evaluation --proposed         # Only Proposed MAS
    python -m evaluation.run_evaluation --metrics          # Only compute metrics
    python -m evaluation.run_evaluation --report           # Only generate report
    python -m evaluation.run_evaluation --all --limit 2    # Dry run with 2 queries
"""

import argparse
import sys
import time
from datetime import datetime


def main():
    """Parse arguments and run the requested evaluation steps."""
    parser = argparse.ArgumentParser(
        description="Evaluation Benchmarking Framework for Therapy Guide-MAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run_evaluation --all --limit 2     # Quick dry run
  python -m evaluation.run_evaluation --baseline1         # Run only Baseline 1
  python -m evaluation.run_evaluation --metrics --report  # Compute metrics + report
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all 3 baselines + metrics + report")
    parser.add_argument("--baseline1", action="store_true", help="Run Baseline 1: Single LLM + Tool")
    parser.add_argument("--baseline2", action="store_true", help="Run Baseline 2: Vanilla RAG")
    parser.add_argument("--proposed", action="store_true", help="Run Proposed: Therapy Guide-MAS")
    parser.add_argument("--metrics", action="store_true", help="Compute evaluation metrics")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries (for dry runs)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between queries in seconds (default: 2.0)")

    args = parser.parse_args()

    # Default to --all if no flags are specified
    if not any([args.all, args.baseline1, args.baseline2, args.proposed, args.metrics, args.report]):
        parser.print_help()
        sys.exit(1)

    run_b1 = args.all or args.baseline1
    run_b2 = args.all or args.baseline2
    run_proposed = args.all or args.proposed
    run_metrics = args.all or args.metrics
    run_report = args.all or args.report

    print(f"\n{'═'*70}")
    print(f"  🏥 EVALUATION BENCHMARKING FRAMEWORK")
    print(f"     Therapy Guide-MAS vs. Baselines")
    print(f"     Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.limit:
        print(f"     ⚡ Limited to {args.limit} queries (dry run)")
    print(f"{'═'*70}\n")

    overall_start = time.time()

    # ── Step 1: Run Baselines ──────────────────────────────────────────
    if run_b1:
        from evaluation.baseline1_single_llm import run_baseline1
        run_baseline1(limit=args.limit, delay=args.delay)

    if run_b2:
        from evaluation.baseline2_vanilla_rag import run_baseline2
        run_baseline2(limit=args.limit, delay=args.delay)

    if run_proposed:
        from evaluation.proposed_mas import run_proposed as run_proposed_fn
        run_proposed_fn(limit=args.limit, delay=args.delay)

    # ── Step 2: Compute Metrics ────────────────────────────────────────
    if run_metrics:
        from evaluation.metrics import compute_all_metrics
        compute_all_metrics()

    # ── Step 3: Generate Report ────────────────────────────────────────
    if run_report:
        from evaluation.generate_report import generate_report
        generate_report()

    # ── Done ───────────────────────────────────────────────────────────
    elapsed = time.time() - overall_start
    print(f"\n{'═'*70}")
    print(f"  ✅ EVALUATION COMPLETE")
    print(f"     Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"     Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()

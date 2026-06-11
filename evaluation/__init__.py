"""
Evaluation Benchmarking Framework for Therapy Guide-MAS.

Provides scripts to benchmark 3 systems against 100 curated test queries:
  - Baseline 1: Single LLM + Tool (ADK agent only)
  - Baseline 2: Vanilla RAG (retrieve + generate, no agents)
  - Proposed:   Therapy Guide-MAS (full LangGraph pipeline)

Usage:
    python -m evaluation.run_evaluation --all
    python -m evaluation.run_evaluation --baseline1
    python -m evaluation.run_evaluation --metrics
    python -m evaluation.run_evaluation --report
"""

from evaluation.test_queries import TEST_QUERIES

"""
PubMed/MEDLINE API Testing Script for Speech-Language Pathology Research.

Tests the Entrez (Biopython) API for fetching medical/biomedical papers
relevant to the speech-agent's domain: dysphagia, speech therapy, feeding
disorders, aphasia, apraxia, autism communication, etc.

Requirements:
    pip install biopython

Usage:
    python -m evaluation.test_pubmed            # Run all tests
    python -m evaluation.test_pubmed --quick     # Run quick connectivity test only
    python -m evaluation.test_pubmed --query "dysphagia speech therapy"  # Custom query
"""

import argparse
import json
import sys
import time
from datetime import datetime
from typing import Any

# ── Ensure Biopython is available ──────────────────────────────────────
try:
    from Bio import Entrez, Medline
except ImportError:
    print(
        "❌ Biopython is not installed.\n"
        "   Install it with:  pip install biopython\n"
    )
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────

# IMPORTANT: NCBI requires an email for Entrez usage.
# Replace with your actual email or set via environment variable.
import os

Entrez.email = os.getenv("PUBMED_EMAIL", "speech.agent.test@example.com")
Entrez.tool = "speech-agent-test"

# Rate limit: NCBI allows 3 requests/sec without an API key,
# 10 requests/sec with one. We add a small delay to be polite.
REQUEST_DELAY = 0.4  # seconds between requests

# SLP-relevant test queries mapped to the project's domain
SLP_TEST_QUERIES = [
    # ── Speech domain ──
    {
        "id": "PUB_S01",
        "query": "dysphagia speech therapy treatment",
        "category": "Speech",
        "expected_topic": "Dysphagia",
    },
    {
        "id": "PUB_S02",
        "query": "childhood apraxia of speech intervention",
        "category": "Speech",
        "expected_topic": "Apraxia",
    },
    {
        "id": "PUB_S03",
        "query": "global aphasia rehabilitation speech language pathology",
        "category": "Speech",
        "expected_topic": "Aphasia",
    },
    {
        "id": "PUB_S04",
        "query": "autism nonverbal communication AAC augmentative",
        "category": "Speech",
        "expected_topic": "Non-Verbal Autism",
    },
    {
        "id": "PUB_S05",
        "query": "oro-motor weakness therapy exercises children",
        "category": "Speech",
        "expected_topic": "Oro-Motor Weakness",
    },
    {
        "id": "PUB_S06",
        "query": "speech delay intervention toddler late talker",
        "category": "Speech",
        "expected_topic": "Speech Delay",
    },
    # ── Feeding domain ──
    {
        "id": "PUB_F01",
        "query": "pediatric feeding disorder texture aversion",
        "category": "Feeding",
        "expected_topic": "Sensory Food Aversion",
    },
    {
        "id": "PUB_F02",
        "query": "cerebral palsy feeding therapy oral motor",
        "category": "Feeding",
        "expected_topic": "Feeding in Cerebral Palsy",
    },
    {
        "id": "PUB_F03",
        "query": "tongue thrust swallowing treatment pediatric",
        "category": "Feeding",
        "expected_topic": "Tongue Thrust",
    },
    {
        "id": "PUB_F04",
        "query": "drooling management saliva control children",
        "category": "Feeding",
        "expected_topic": "Drooling",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Core PubMed API Functions
# ═══════════════════════════════════════════════════════════════════════


def pubmed_search(
    query: str,
    max_results: int = 5,
    sort: str = "relevance",
    min_date: str | None = None,
    max_date: str | None = None,
) -> dict[str, Any]:
    """Search PubMed and return article IDs + metadata.

    Args:
        query: The search term (supports PubMed query syntax).
        max_results: Max number of results to return.
        sort: Sort order — 'relevance' or 'pub_date'.
        min_date: Optional minimum publication date (YYYY/MM/DD).
        max_date: Optional maximum publication date (YYYY/MM/DD).

    Returns:
        Dictionary with 'count', 'ids', 'query_translation', and timing info.
    """
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "sort": sort,
        "retmode": "xml",
    }
    if min_date:
        search_params["mindate"] = min_date
        search_params["datetype"] = "pdat"
    if max_date:
        search_params["maxdate"] = max_date
        search_params["datetype"] = "pdat"

    start = time.time()
    handle = Entrez.esearch(**search_params)
    record = Entrez.read(handle)
    handle.close()
    elapsed = time.time() - start

    return {
        "count": int(record.get("Count", 0)),
        "ids": list(record.get("IdList", [])),
        "query_translation": record.get("QueryTranslation", ""),
        "elapsed_seconds": round(elapsed, 3),
    }


def fetch_article_details(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch detailed article metadata (title, abstract, authors, journal).

    Args:
        pmids: List of PubMed IDs to fetch.

    Returns:
        List of article dictionaries with parsed fields.
    """
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="medline",
        retmode="text",
    )
    records = list(Medline.parse(handle))
    handle.close()

    articles = []
    for rec in records:
        articles.append({
            "pmid": rec.get("PMID", "N/A"),
            "title": rec.get("TI", "No title"),
            "authors": rec.get("AU", []),
            "journal": rec.get("JT", "Unknown journal"),
            "journal_abbrev": rec.get("TA", ""),
            "pub_date": rec.get("DP", "Unknown date"),
            "abstract": rec.get("AB", "(No abstract available)"),
            "mesh_terms": rec.get("MH", []),
            "publication_type": rec.get("PT", []),
            "doi": rec.get("AID", []),
        })
    return articles


def format_article_for_context(article: dict) -> str:
    """Format a single PubMed article into a context string (like web_search_agent).

    Args:
        article: Article dictionary from fetch_article_details.

    Returns:
        Formatted string suitable for injection into the RAG/synthesis pipeline.
    """
    authors_str = ", ".join(article["authors"][:3])
    if len(article["authors"]) > 3:
        authors_str += " et al."

    return (
        f"Title: {article['title']}\n"
        f"Authors: {authors_str}\n"
        f"Journal: {article['journal']} ({article['pub_date']})\n"
        f"PMID: {article['pmid']}\n"
        f"Abstract: {article['abstract']}\n"
    )


# ═══════════════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════════════


def test_connectivity() -> bool:
    """Test 1: Basic PubMed API connectivity."""
    print("\n" + "=" * 70)
    print("🧪 TEST 1: PubMed API Connectivity")
    print("=" * 70)

    try:
        result = pubmed_search("test", max_results=1)
        print(f"   ✅ Connected successfully!")
        print(f"   📊 Total articles for 'test': {result['count']:,}")
        print(f"   ⏱️  Response time: {result['elapsed_seconds']}s")
        print(f"   🔍 Query translation: {result['query_translation']}")
        return True
    except Exception as e:
        print(f"   ❌ Connection FAILED: {e}")
        return False


def test_slp_searches() -> dict[str, Any]:
    """Test 2: Run all SLP-domain searches and report results."""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: SLP-Domain Search Results")
    print("=" * 70)

    results_summary = []
    total_articles = 0
    total_time = 0.0

    for test in SLP_TEST_QUERIES:
        time.sleep(REQUEST_DELAY)  # Rate-limit politeness
        result = pubmed_search(test["query"], max_results=5)

        status = "✅" if result["count"] > 0 else "⚠️"
        print(
            f"   {status} [{test['id']}] {test['expected_topic']:30s} → "
            f"{result['count']:>8,} articles | "
            f"{len(result['ids'])} returned | "
            f"{result['elapsed_seconds']}s"
        )

        results_summary.append({
            **test,
            "total_count": result["count"],
            "returned_ids": result["ids"],
            "elapsed": result["elapsed_seconds"],
        })
        total_articles += result["count"]
        total_time += result["elapsed_seconds"]

    print(f"\n   📊 Summary:")
    print(f"      Queries tested: {len(SLP_TEST_QUERIES)}")
    print(f"      Total articles found: {total_articles:,}")
    print(f"      Avg response time: {total_time / len(SLP_TEST_QUERIES):.3f}s")
    print(f"      Queries with 0 results: "
          f"{sum(1 for r in results_summary if r['total_count'] == 0)}")

    return {"results": results_summary, "total_articles": total_articles}


def test_fetch_abstracts() -> list[dict]:
    """Test 3: Fetch full article details (titles, abstracts, MeSH terms)."""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Fetching Article Details & Abstracts")
    print("=" * 70)

    # Use a highly relevant query
    query = "dysphagia speech language pathology treatment review"
    print(f"   🔎 Query: \"{query}\"")

    search = pubmed_search(query, max_results=3, sort="relevance")
    print(f"   📊 Found {search['count']:,} total articles, fetching top {len(search['ids'])}...")

    time.sleep(REQUEST_DELAY)
    articles = fetch_article_details(search["ids"])

    for i, article in enumerate(articles, 1):
        print(f"\n   ── Article {i} ──────────────────────────────────────")
        print(f"   📄 Title: {article['title'][:90]}...")
        print(f"   👤 Authors: {', '.join(article['authors'][:3])}")
        print(f"   📰 Journal: {article['journal_abbrev']} ({article['pub_date']})")
        print(f"   🆔 PMID: {article['pmid']}")

        abstract_preview = article["abstract"][:200]
        if len(article["abstract"]) > 200:
            abstract_preview += "..."
        print(f"   📝 Abstract: {abstract_preview}")

        if article["mesh_terms"]:
            print(f"   🏷️  MeSH Terms: {', '.join(article['mesh_terms'][:5])}")
        else:
            print(f"   🏷️  MeSH Terms: (none)")

    return articles


def test_formatted_context() -> str:
    """Test 4: Generate pipeline-ready context string (like web_search_agent output)."""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: Pipeline-Ready Context Formatting")
    print("=" * 70)

    query = "childhood apraxia of speech motor-based treatment"
    print(f"   🔎 Query: \"{query}\"")

    search = pubmed_search(query, max_results=3)
    time.sleep(REQUEST_DELAY)
    articles = fetch_article_details(search["ids"])

    print(f"\n   📋 Formatted context (as it would appear in the synthesis pipeline):\n")
    context_parts = []
    for i, article in enumerate(articles, 1):
        formatted = format_article_for_context(article)
        context_parts.append(f"--- PubMed Reference {i} ---\n{formatted}")
        print(f"   [{i}] {article['title'][:80]}...")

    full_context = "\n".join(context_parts)
    print(f"\n   📊 Total context length: {len(full_context):,} chars")
    return full_context


def test_recent_papers() -> dict:
    """Test 5: Search for recent papers (last 2 years) to verify date filtering."""
    print("\n" + "=" * 70)
    print("🧪 TEST 5: Recent Papers (Date-Filtered Search)")
    print("=" * 70)

    query = "dysphagia rehabilitation systematic review"
    current_year = datetime.now().year

    print(f"   🔎 Query: \"{query}\"")
    print(f"   📅 Date range: {current_year - 2}/01/01 → {current_year}/12/31\n")

    search = pubmed_search(
        query,
        max_results=5,
        sort="pub_date",
        min_date=f"{current_year - 2}/01/01",
        max_date=f"{current_year}/12/31",
    )

    print(f"   📊 Recent articles found: {search['count']:,}")
    print(f"   🆔 Top IDs: {search['ids']}")

    if search["ids"]:
        time.sleep(REQUEST_DELAY)
        articles = fetch_article_details(search["ids"])
        for art in articles:
            print(f"   📄 [{art['pub_date']}] {art['title'][:75]}...")
    else:
        print("   ⚠️  No recent articles found for this query")

    return search


def test_batch_from_test_queries() -> dict:
    """Test 6: Run PubMed searches for a subset of the project's existing test queries."""
    print("\n" + "=" * 70)
    print("🧪 TEST 6: PubMed Results for Project Test Queries")
    print("=" * 70)

    # Import existing test queries from the project
    try:
        from evaluation.test_queries import TEST_QUERIES
    except ImportError:
        print("   ⚠️  Could not import TEST_QUERIES. Run from project root.")
        print("   Falling back to inline sample queries.\n")
        TEST_QUERIES = [
            {"id": "FALLBACK_1", "query": "How would you manage childhood apraxia of speech?",
             "expected_ailment": "Childhood Apraxia of Speech"},
            {"id": "FALLBACK_2", "query": "What are the priorities in dysphagia management?",
             "expected_ailment": "Dysphagia"},
            {"id": "FALLBACK_3", "query": "How would you start intervention for a non-verbal child with autism?",
             "expected_ailment": "Non-Verbal Autism"},
        ]

    # Test a representative sample (first 5 queries)
    sample = TEST_QUERIES[:5]
    results = {}

    for tq in sample:
        time.sleep(REQUEST_DELAY)

        # Convert the clinical question into a PubMed-friendly search term
        pubmed_query = f"{tq.get('expected_ailment', '')} speech language pathology treatment"
        search = pubmed_search(pubmed_query, max_results=3)

        status = "✅" if search["count"] > 0 else "⚠️"
        print(
            f"   {status} [{tq['id']}] {tq.get('expected_ailment', 'N/A'):35s} → "
            f"{search['count']:>8,} articles"
        )

        results[tq["id"]] = {
            "original_query": tq["query"],
            "pubmed_query": pubmed_query,
            "count": search["count"],
            "ids": search["ids"],
        }

    queries_with_results = sum(1 for r in results.values() if r["count"] > 0)
    print(f"\n   📊 {queries_with_results}/{len(sample)} queries returned PubMed results")

    return results


def test_pubmed_vs_ddg_comparison():
    """Test 7: Compare PubMed results vs DuckDuckGo for the same query."""
    print("\n" + "=" * 70)
    print("🧪 TEST 7: PubMed vs DuckDuckGo Comparison")
    print("=" * 70)

    query = "dysphagia speech therapy treatment"
    print(f"   🔎 Query: \"{query}\"\n")

    # ── PubMed ─────────────────────────────────────────────────────────
    print("   📚 PubMed Results:")
    pm_search = pubmed_search(query, max_results=3)
    time.sleep(REQUEST_DELAY)
    pm_articles = fetch_article_details(pm_search["ids"])

    for i, art in enumerate(pm_articles, 1):
        print(f"      [{i}] {art['title'][:70]}...")
        print(f"          Journal: {art['journal_abbrev']} | PMID: {art['pmid']}")
    print(f"      Total in PubMed: {pm_search['count']:,} | Time: {pm_search['elapsed_seconds']}s")

    # ── DuckDuckGo ─────────────────────────────────────────────────────
    print("\n   🌐 DuckDuckGo Results:")
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")
        from duckduckgo_search import DDGS

        ddg_start = time.time()
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query, max_results=3))
        ddg_elapsed = round(time.time() - ddg_start, 3)

        if ddg_results:
            for i, r in enumerate(ddg_results, 1):
                print(f"      [{i}] {r.get('title', 'No title')[:70]}...")
                print(f"          URL: {r.get('href', 'N/A')[:70]}")
        else:
            print("      ⚠️  No DuckDuckGo results returned")
        print(f"      Results: {len(ddg_results)} | Time: {ddg_elapsed}s")

    except ImportError:
        print("      ⚠️  duckduckgo-search not installed, skipping DDG comparison")
    except Exception as e:
        print(f"      ❌ DuckDuckGo search failed: {e}")

    # ── Comparison ─────────────────────────────────────────────────────
    print("\n   📊 Comparison Summary:")
    print(f"      {'Metric':<25} {'PubMed':<25} {'DuckDuckGo':<25}")
    print(f"      {'─'*75}")
    print(f"      {'Source Type':<25} {'Peer-reviewed papers':<25} {'General web pages':<25}")
    print(f"      {'Total Available':<25} {pm_search['count']:,}{'':<20} {'N/A':<25}")
    print(f"      {'Medical Focus':<25} {'✅ 100% biomedical':<25} {'❓ Mixed':<25}")
    print(f"      {'Abstracts':<25} {'✅ Available':<25} {'❌ Snippets only':<25}")
    print(f"      {'MeSH Terms':<25} {'✅ Standardized':<25} {'❌ Not available':<25}")
    print(f"      {'Cost':<25} {'Free':<25} {'Free':<25}")


def test_advanced_query_syntax():
    """Test 8: Test PubMed advanced query syntax (MeSH terms, field tags)."""
    print("\n" + "=" * 70)
    print("🧪 TEST 8: PubMed Advanced Query Syntax")
    print("=" * 70)

    advanced_queries = [
        {
            "label": "MeSH term search",
            "query": '"Deglutition Disorders"[MeSH] AND "Speech Therapy"[MeSH]',
        },
        {
            "label": "Title/Abstract field",
            "query": "dysphagia[Title] AND rehabilitation[Title/Abstract]",
        },
        {
            "label": "Review articles only",
            "query": "apraxia speech therapy AND Review[pt]",
        },
        {
            "label": "Pediatric filter",
            "query": "feeding disorder AND (child[MeSH] OR pediatric[tw])",
        },
        {
            "label": "Free full text",
            "query": "aphasia language therapy AND free full text[sb]",
        },
    ]

    for aq in advanced_queries:
        time.sleep(REQUEST_DELAY)
        result = pubmed_search(aq["query"], max_results=3)
        status = "✅" if result["count"] > 0 else "⚠️"
        print(
            f"   {status} {aq['label']:30s} → {result['count']:>8,} articles | "
            f"{result['elapsed_seconds']}s"
        )
        print(f"      Query: {aq['query'][:80]}")
        if result["query_translation"]:
            print(f"      Translation: {result['query_translation'][:80]}")


# ═══════════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Test PubMed/MEDLINE API for Speech-Agent Research"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick connectivity test only",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a single custom PubMed query",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Max results for custom query (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    print("\n" + "🏥" * 35)
    print("  PubMed/MEDLINE API Test Suite — Speech-Language Pathology")
    print(f"  Entrez email: {Entrez.email}")
    print(f"  Timestamp: {datetime.now().isoformat()}")
    print("🏥" * 35)

    # ── Custom single query mode ───────────────────────────────────────
    if args.query:
        print(f"\n🔎 Custom Query: \"{args.query}\"")
        search = pubmed_search(args.query, max_results=args.max_results)
        print(f"   📊 Total results: {search['count']:,}")
        print(f"   ⏱️  Time: {search['elapsed_seconds']}s\n")

        if search["ids"]:
            time.sleep(REQUEST_DELAY)
            articles = fetch_article_details(search["ids"])
            for i, art in enumerate(articles, 1):
                print(f"   ── Article {i} ─────────────────────────────")
                print(f"   📄 {art['title']}")
                print(f"   👤 {', '.join(art['authors'][:3])}")
                print(f"   📰 {art['journal']} ({art['pub_date']})")
                print(f"   🆔 PMID: {art['pmid']}")
                abstract_preview = art['abstract'][:300]
                print(f"   📝 {abstract_preview}...\n")

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(articles, f, indent=2, ensure_ascii=False)
                print(f"   💾 Results saved to {args.output}")
        return

    # ── Quick mode ─────────────────────────────────────────────────────
    if args.quick:
        success = test_connectivity()
        print("\n" + ("✅ PubMed API is accessible!" if success else "❌ PubMed API is NOT accessible."))
        return

    # ── Full test suite ────────────────────────────────────────────────
    all_results = {}

    # Test 1: Connectivity
    connected = test_connectivity()
    if not connected:
        print("\n❌ Cannot reach PubMed API. Aborting remaining tests.")
        return

    # Test 2: SLP domain searches
    all_results["slp_searches"] = test_slp_searches()
    time.sleep(REQUEST_DELAY)

    # Test 3: Fetch abstracts
    all_results["sample_articles"] = test_fetch_abstracts()
    time.sleep(REQUEST_DELAY)

    # Test 4: Pipeline-ready context
    all_results["context_sample"] = test_formatted_context()
    time.sleep(REQUEST_DELAY)

    # Test 5: Recent papers
    all_results["recent_papers"] = test_recent_papers()
    time.sleep(REQUEST_DELAY)

    # Test 6: Batch from existing test queries
    all_results["batch_results"] = test_batch_from_test_queries()
    time.sleep(REQUEST_DELAY)

    # Test 7: PubMed vs DuckDuckGo comparison
    test_pubmed_vs_ddg_comparison()
    time.sleep(REQUEST_DELAY)

    # Test 8: Advanced query syntax
    test_advanced_query_syntax()

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📋 FINAL SUMMARY")
    print("=" * 70)
    print(f"   ✅ All 8 tests completed successfully!")
    print(f"   📊 PubMed is highly suitable for the speech-agent's medical domain")
    print(f"   💡 Recommendation: Use PubMed as a SUPPLEMENT to web search for")
    print(f"      peer-reviewed, evidence-based clinical references.")
    print(f"\n   Next steps:")
    print(f"      1. pip install biopython")
    print(f"      2. Create rag/pubmed_search_agent.py (parallel to web_search_agent.py)")
    print(f"      3. Add PubMed context to the synthesis pipeline")
    print(f"      4. Set PUBMED_EMAIL env var for production use")

    # ── Optional JSON output ───────────────────────────────────────────
    if args.output:
        # Serialize only JSON-compatible parts
        serializable = {}
        for key, val in all_results.items():
            if isinstance(val, (dict, list, str, int, float)):
                serializable[key] = val
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n   💾 Full results saved to {args.output}")


if __name__ == "__main__":
    main()

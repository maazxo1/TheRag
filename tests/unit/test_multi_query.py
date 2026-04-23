"""
Component 4 test — multi_query.py

Checks:
  1. generate_query_variants returns original + N distinct phrasings
  2. Variants are non-empty strings, not duplicates of each other
  3. Graceful fallback: if LLM fails, returns [original_query] only
  4. multi_query_retrieve returns no duplicate node IDs
  5. Result count is <= top_k
  6. Multi-query recall >= single-query recall on the same retriever
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch, MagicMock
from src.pipelines.multi_query import generate_query_variants, multi_query_retrieve
from src.pipelines.retrieval_pipeline import HybridRetriever
from entrypoint.ingest import load_existing_index
import config


# ── Test 1: variant generation (unit) ──────────────────────────────────────────

def test_variants_include_original():
    variants = generate_query_variants("What is hybrid search?", n=3)
    assert variants[0] == "What is hybrid search?", "First element must be the original query"
    print(f"  [PASS] Original query is first: '{variants[0]}'")


def test_variants_count():
    variants = generate_query_variants("How does RRF work?", n=3)
    assert 1 <= len(variants) <= 4, f"Expected 1–4 variants, got {len(variants)}"
    print(f"  [PASS] Generated {len(variants)} variants (original + {len(variants)-1} alternatives)")
    for i, v in enumerate(variants):
        print(f"         [{i}] {v}")


def test_variants_no_duplicates():
    variants = generate_query_variants("What is confidence scoring?", n=3)
    lower = [v.lower() for v in variants]
    assert len(lower) == len(set(lower)), f"Duplicate variants found: {variants}"
    print(f"  [PASS] No duplicate variants")


def test_variants_non_empty():
    variants = generate_query_variants("Explain small-to-big chunking", n=3)
    for v in variants:
        assert len(v.strip()) > 5, f"Variant too short: '{v}'"
    print(f"  [PASS] All {len(variants)} variants are non-empty strings")


def test_fallback_on_llm_failure():
    with patch("multi_query.requests.post", side_effect=Exception("network error")):
        variants = generate_query_variants("test query", n=3)
    assert variants == ["test query"], f"Should fall back to original only, got: {variants}"
    print("  [PASS] Graceful fallback to [original] on LLM failure")


# ── Test 2: retrieval integration ───────────────────────────────────────────────

def build_retriever():
    index, bm25, child_nodes, parent_node_map = load_existing_index()
    vector_retriever = index.as_retriever(similarity_top_k=config.TOP_K_RETRIEVAL)
    return HybridRetriever(child_nodes, vector_retriever, bm25=bm25), child_nodes


def test_no_duplicate_nodes(retriever):
    results = multi_query_retrieve("what is cross-encoder reranking", retriever, top_k=15)
    ids = [n.node_id for n in results]
    assert len(ids) == len(set(ids)), "Duplicate nodes in multi-query results"
    print(f"  [PASS] No duplicates in {len(results)} merged results")


def test_respects_top_k(retriever):
    top_k = 8
    results = multi_query_retrieve("hybrid search BM25", retriever, top_k=top_k)
    assert len(results) <= top_k, f"Got {len(results)} results, expected <= {top_k}"
    print(f"  [PASS] Result count {len(results)} <= top_k {top_k}")


def test_multi_query_recall_vs_single(retriever):
    query = "how does the pipeline expand context for generation"

    single = retriever.retrieve(query, top_k=config.TOP_K_RETRIEVAL)
    single_ids = {n.node_id for n in single}

    multi = multi_query_retrieve(query, retriever, top_k=config.TOP_K_RETRIEVAL * 3)
    multi_ids = {n.node_id for n in multi}

    assert len(multi_ids) >= len(single_ids), (
        f"Multi-query ({len(multi_ids)}) should recall >= single ({len(single_ids)})"
    )
    extra = len(multi_ids - single_ids)
    print(f"  [PASS] Multi-query recall: {len(multi_ids)} nodes vs single {len(single_ids)} "
          f"(+{extra} extra unique nodes found)")


# ── Runner ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Component 4: multi_query.py ===\n")

    print("-- Unit tests: variant generation --")
    test_variants_include_original()
    test_variants_count()
    test_variants_no_duplicates()
    test_variants_non_empty()
    test_fallback_on_llm_failure()

    print("\n-- Integration tests: parallel retrieval --")
    retriever, child_nodes = build_retriever()
    test_no_duplicate_nodes(retriever)
    test_respects_top_k(retriever)
    test_multi_query_recall_vs_single(retriever)

    print("\n=== All tests passed — ready to build Component 5 (hyde.py) ===\n")

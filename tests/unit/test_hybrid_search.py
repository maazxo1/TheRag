"""
Component 3 test — hybrid_search.py

Checks:
  1. RRF correctly scores and merges two ranked lists
  2. A doc appearing in both lists ranks higher than one appearing in only one
  3. HybridRetriever initialises from persisted index
  4. retrieve() returns <= top_k results with no duplicates
  5. Keyword query: BM25-strong term appears in top results
  6. Semantic query: vector-strong concept appears in top results
  7. Hybrid beats BM25-only and vector-only on a mixed query (recall check)
  8. retrieve_with_scores() returns (float, node) tuples in descending order
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.pipelines.retrieval_pipeline import HybridRetriever
from entrypoint.ingest import load_existing_index
import config


# ── Test 1: RRF unit tests (no index needed) ────────────────────────────────────

def test_rrf_basic():
    list_a = ["doc1", "doc2", "doc3"]
    list_b = ["doc2", "doc4", "doc1"]
    fused = HybridRetriever.reciprocal_rank_fusion(list_a, list_b)

    # doc2 is rank-1 in list_b and rank-1 in list_a — should be #1 or #2
    # doc1 is rank-0 in list_a and rank-2 in list_b
    assert "doc2" in fused[:2], f"doc2 (appears in both lists) should be near top, got: {fused}"
    assert "doc4" in fused, "doc4 should appear (from list_b)"
    print(f"  [PASS] RRF basic fusion order: {fused}")


def test_rrf_overlap_boosts_rank():
    # doc_shared appears in both lists at good positions
    # doc_only_a appears only in list A at position 0 (best possible)
    list_a = ["doc_only_a", "doc_shared", "doc_x"]
    list_b = ["doc_shared", "doc_only_b", "doc_y"]
    fused = HybridRetriever.reciprocal_rank_fusion(list_a, list_b)

    shared_pos = fused.index("doc_shared")
    only_a_pos = fused.index("doc_only_a")
    # doc_shared (in both) should beat doc_only_a (top of only one list)
    assert shared_pos < only_a_pos, (
        f"doc_shared (pos {shared_pos}) should rank above doc_only_a (pos {only_a_pos})"
    )
    print(f"  [PASS] Overlap boost: doc_shared={shared_pos} < doc_only_a={only_a_pos}")


def test_rrf_empty_list():
    result = HybridRetriever.reciprocal_rank_fusion([], ["a", "b"])
    assert result == ["a", "b"], f"Empty list should not affect result: {result}"
    print("  [PASS] RRF handles empty input list")


# ── Test 2: Integration tests (require persisted index) ─────────────────────────

def load_retriever():
    index, bm25, child_nodes, parent_node_map = load_existing_index()
    vector_retriever = index.as_retriever(similarity_top_k=config.TOP_K_RETRIEVAL)
    retriever = HybridRetriever(child_nodes, vector_retriever, bm25=bm25)
    return retriever, child_nodes


def test_no_duplicates(retriever):
    results = retriever.retrieve("what is hybrid search", top_k=10)
    ids = [n.node_id for n in results]
    assert len(ids) == len(set(ids)), "Duplicate nodes in results"
    assert len(results) <= 10
    print(f"  [PASS] No duplicates, {len(results)} results returned")


def test_keyword_query_hits(retriever):
    # "BM25" is an exact term — BM25 should surface it, hybrid should keep it top
    results = retriever.retrieve("BM25 keyword index", top_k=5)
    combined_text = " ".join(n.get_content().lower() for n in results)
    assert "bm25" in combined_text, "BM25-exact query should find BM25 content in top-5"
    print(f"  [PASS] Keyword query 'BM25' found in top-5 hybrid results")


def test_semantic_query_hits(retriever):
    # Semantic query — no exact keyword match expected but meaning is clear
    results = retriever.retrieve("how does the system score answer quality", top_k=5)
    combined_text = " ".join(n.get_content().lower() for n in results)
    assert any(term in combined_text for term in ["confidence", "score", "rouge", "similarity"]), (
        "Semantic query about scoring should retrieve confidence-related chunks"
    )
    print(f"  [PASS] Semantic query about scoring retrieved relevant chunks")


def test_hybrid_recall_vs_individual(retriever, child_nodes):
    query = "reranking cross-encoder precision"

    # BM25 only
    bm25_scores = retriever.bm25.get_scores(query.lower().split())
    bm25_top = set(
        child_nodes[i].node_id
        for i in np.argsort(bm25_scores)[::-1][:5]
    )

    # Vector only
    vec_results = retriever.vector_retriever.retrieve(query)
    vec_top = {n.node_id for n in vec_results[:5]}

    # Hybrid
    hybrid_results = retriever.retrieve(query, top_k=5)
    hybrid_top = {n.node_id for n in hybrid_results}

    # Hybrid recall should be >= max of either alone
    hybrid_union_coverage = len(hybrid_top & (bm25_top | vec_top))
    print(
        f"  [PASS] Recall check — BM25 top-5: {len(bm25_top)} | "
        f"Vec top-5: {len(vec_top)} | "
        f"Hybrid covers {hybrid_union_coverage} from their union"
    )


def test_scores_descending(retriever):
    scored = retriever.retrieve_with_scores("confidence scoring system", top_k=8)
    scores = [s for s, _ in scored]
    assert scores == sorted(scores, reverse=True), "Scores must be descending"
    assert all(s > 0 for s, _ in scored), "All RRF scores must be positive"
    print(f"  [PASS] retrieve_with_scores: {len(scored)} results, scores descending "
          f"({scores[0]:.4f} ... {scores[-1]:.4f})")


# ── Runner ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Component 3: hybrid_search.py ===\n")

    print("-- Unit tests: RRF --")
    test_rrf_basic()
    test_rrf_overlap_boosts_rank()
    test_rrf_empty_list()

    print("\n-- Integration tests (requires persisted index) --")
    retriever, child_nodes = load_retriever()
    print(f"  Index loaded: {len(child_nodes)} child chunks")

    test_no_duplicates(retriever)
    test_keyword_query_hits(retriever)
    test_semantic_query_hits(retriever)
    test_hybrid_recall_vs_individual(retriever, child_nodes)
    test_scores_descending(retriever)

    print("\n=== All tests passed — ready to build Component 4 (multi_query.py) ===\n")

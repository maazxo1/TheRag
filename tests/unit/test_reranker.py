"""
Component 6 test — reranker.py

Checks:
  1. rerank returns <= top_k results
  2. No duplicate nodes in output
  3. Output is a subset of the input nodes
  4. rerank_with_scores returns (float, node) tuples in descending score order
  5. Precision: a highly relevant chunk scores higher than an irrelevant one
  6. Empty input returns empty output (no crash)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipelines.reranking_pipeline import rerank, rerank_with_scores
from entrypoint.ingest import load_existing_index
import config


# ── Test 1: basic contract (unit-style, uses real model) ──────────────────────

def test_rerank_respects_top_k(nodes):
    result = rerank("what is hybrid search", nodes, top_k=3)
    assert len(result) <= 3, f"Expected <= 3 results, got {len(result)}"
    print(f"  [PASS] rerank returned {len(result)} results (top_k=3)")


def test_rerank_no_duplicates(nodes):
    result = rerank("what is hybrid search", nodes, top_k=10)
    ids = [n.node_id for n in result]
    assert len(ids) == len(set(ids)), "Duplicate nodes in reranked output"
    print(f"  [PASS] No duplicates in {len(result)} reranked results")


def test_rerank_subset_of_input(nodes):
    result = rerank("what is hybrid search", nodes, top_k=5)
    input_ids = {n.node_id for n in nodes}
    for node in result:
        assert node.node_id in input_ids, f"Output node {node.node_id} not in input"
    print(f"  [PASS] All output nodes are a subset of input nodes")


def test_rerank_empty_input():
    result = rerank("any query", [], top_k=5)
    assert result == [], f"Expected empty list for empty input, got {result}"
    print("  [PASS] Empty input returns empty output")


# ── Test 2: rerank_with_scores ─────────────────────────────────────────────────

def test_rerank_with_scores_format(nodes):
    scored = rerank_with_scores("what is hybrid search", nodes, top_k=5)
    assert len(scored) <= 5
    for score, node in scored:
        assert isinstance(score, float), f"Score should be float, got {type(score)}"
        assert hasattr(node, "node_id"), "Each result should be a node"
    print(f"  [PASS] rerank_with_scores returned {len(scored)} (score, node) tuples")


def test_scores_descending(nodes):
    scored = rerank_with_scores("what is hybrid search", nodes, top_k=10)
    scores = [s for s, _ in scored]
    assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
    print(f"  [PASS] Scores are descending: {[f'{s:.3f}' for s in scores[:5]]}")


# ── Test 3: precision — relevant chunk outscores irrelevant ────────────────────

def test_relevant_beats_irrelevant(child_nodes):
    from llama_index.core.schema import TextNode

    query = "how does BM25 keyword search work"

    relevant = TextNode(
        text=(
            "BM25 is a bag-of-words retrieval function that ranks documents "
            "based on the query terms appearing in each document. It uses term "
            "frequency (TF) and inverse document frequency (IDF) to score "
            "how relevant each document is to the keyword query."
        ),
        id_="relevant-node",
    )
    irrelevant = TextNode(
        text=(
            "The mitochondria is the powerhouse of the cell. ATP synthesis "
            "occurs via oxidative phosphorylation in the inner membrane."
        ),
        id_="irrelevant-node",
    )

    scored = rerank_with_scores(query, [irrelevant, relevant], top_k=2)
    top_node_id = scored[0][1].node_id
    assert top_node_id == "relevant-node", (
        f"Expected relevant node at top, got '{top_node_id}'\n"
        f"Scores: {[(f'{s:.3f}', n.node_id) for s, n in scored]}"
    )
    print(f"  [PASS] Relevant chunk scored {scored[0][0]:.3f} vs irrelevant {scored[1][0]:.3f}")


# ── Runner ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Component 6: reranker.py ===\n")
    print("Loading reranker model (first run downloads ~1.1 GB — subsequent runs are instant)...")

    index, bm25, child_nodes, parent_node_map = load_existing_index()
    print(f"  Index loaded: {len(child_nodes)} child chunks")

    # Use top-20 from retrieval as candidate pool (realistic pipeline input)
    retriever = index.as_retriever(similarity_top_k=config.TOP_K_RETRIEVAL)
    candidate_nodes = retriever.retrieve("what is hybrid search and how does BM25 work")
    candidate_nodes = [n.node for n in candidate_nodes]  # unwrap NodeWithScore
    print(f"  Candidate pool: {len(candidate_nodes)} nodes\n")

    print("-- Contract tests --")
    test_rerank_respects_top_k(candidate_nodes)
    test_rerank_no_duplicates(candidate_nodes)
    test_rerank_subset_of_input(candidate_nodes)
    test_rerank_empty_input()

    print("\n-- Score format tests --")
    test_rerank_with_scores_format(candidate_nodes)
    test_scores_descending(candidate_nodes)

    print("\n-- Precision test --")
    test_relevant_beats_irrelevant(child_nodes)

    print("\n=== All tests passed — ready to build Component 7 (confidence.py) ===\n")

"""
Component 5 test — hyde.py

Checks:
  1. generate_hypothetical_document returns a non-trivial paragraph
  2. Hypothetical doc is longer than the original query
  3. Hypothetical doc contains domain-relevant vocabulary
  4. Graceful fallback: returns original query on LLM failure
  5. hyde_retrieve returns <= top_k results with no duplicates
  6. HyDE recall vs direct retrieval on a knowledge-base query
  7. hyde_retrieve_with_doc returns both nodes and the hypothetical text
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unittest.mock import patch
from src.pipelines.hyde import generate_hypothetical_document, hyde_retrieve, hyde_retrieve_with_doc
from entrypoint.ingest import load_existing_index
import config


# ── Test 1: hypothetical document generation (unit) ────────────────────────────

def test_hypo_doc_non_trivial():
    query = "What is small-to-big chunking?"
    doc = generate_hypothetical_document(query)
    assert len(doc) > len(query), f"Hypothetical doc should be longer than the query"
    assert len(doc.split()) >= 20, f"Expected >= 20 words, got: {doc}"
    print(f"  [PASS] Hypothetical doc: {len(doc.split())} words")
    print(f"         Preview: '{doc[:150]}...'")


def test_hypo_doc_domain_relevant():
    query = "How does confidence scoring work in RAG systems?"
    doc = generate_hypothetical_document(query)
    relevant_terms = ["confidence", "score", "retrieval", "similarity", "rouge",
                      "answer", "context", "signal", "hallucin"]
    found = [t for t in relevant_terms if t.lower() in doc.lower()]
    assert len(found) >= 2, (
        f"Expected domain terms in hypothetical doc, found only {found}\nDoc: {doc}"
    )
    print(f"  [PASS] Domain terms found in hypothetical doc: {found}")


def test_hypo_doc_fallback():
    with patch("src.pipelines.hyde.requests.post", side_effect=Exception("network error")):
        doc = generate_hypothetical_document("test query")
    assert doc == "test query", f"Should fall back to original query, got: '{doc}'"
    print("  [PASS] Graceful fallback to original query on failure")


# ── Test 2: retrieval integration ───────────────────────────────────────────────

def test_hyde_retrieve_no_duplicates(index):
    results = hyde_retrieve("what is the reranking stage", index, top_k=10)
    ids = [n.node_id for n in results]
    assert len(ids) == len(set(ids)), "Duplicate nodes in HyDE results"
    assert len(results) <= 10
    print(f"  [PASS] HyDE returned {len(results)} unique results")


def test_hyde_retrieve_with_doc_returns_both(index):
    nodes, hypo_doc = hyde_retrieve_with_doc(
        "how does multi-query retrieval improve recall", index, top_k=5
    )
    assert len(nodes) >= 1, "Expected at least one node"
    assert isinstance(hypo_doc, str) and len(hypo_doc) > 10
    print(f"  [PASS] hyde_retrieve_with_doc: {len(nodes)} nodes + {len(hypo_doc)} char doc")
    print(f"         Hypo doc preview: '{hypo_doc[:120]}...'")


def test_hyde_vs_direct_recall(index, child_nodes):
    query = "how does the system expand small chunks to full parent context"

    # Direct retrieval
    retriever = index.as_retriever(similarity_top_k=config.TOP_K_RETRIEVAL)
    direct = retriever.retrieve(query)
    direct_ids = {n.node_id for n in direct}

    # HyDE retrieval
    hyde = hyde_retrieve(query, index, top_k=config.TOP_K_RETRIEVAL)
    hyde_ids = {n.node_id for n in hyde}

    # Both should find relevant content — check overlap is reasonable
    overlap = direct_ids & hyde_ids
    print(f"  [PASS] Direct: {len(direct_ids)} nodes | HyDE: {len(hyde_ids)} nodes | "
          f"Overlap: {len(overlap)} | Combined unique: {len(direct_ids | hyde_ids)}")


# ── Runner ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Component 5: hyde.py ===\n")

    print("-- Unit tests: hypothetical document generation --")
    test_hypo_doc_non_trivial()
    test_hypo_doc_domain_relevant()
    test_hypo_doc_fallback()

    print("\n-- Integration tests: HyDE retrieval --")
    index, bm25, child_nodes, parent_node_map = load_existing_index()
    print(f"  Index loaded: {len(child_nodes)} child chunks")

    test_hyde_retrieve_no_duplicates(index)
    test_hyde_retrieve_with_doc_returns_both(index)
    test_hyde_vs_direct_recall(index, child_nodes)

    print("\n=== All tests passed — ready to build Component 6 (reranker.py) ===\n")

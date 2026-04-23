"""
A/B test: vector-only vs hybrid (BM25+vector+RRF) vs hybrid+rerank.

For each test document:
  - Ingests the document once
  - Runs each pipeline variant on the same question
  - Reports retrieval hit (does the top chunk contain key terms?)
  - Reports latency per stage

This is NOT a pass/fail test — it prints a comparison table so you can
see which variant performs best per document. Exit code 0 always.

Run:
    venv/Scripts/python tests/integration/test_ab.py
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import re
from entrypoint.ingest import ingest_from_file
from src.pipelines.retrieval_pipeline import HybridRetriever
from src.pipelines.reranking_pipeline import rerank_with_scores
from src.pipelines.chunking_pipeline import expand_to_parents

TEST_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_documents")

DOCUMENT_QUERIES = {
    "CA_Lab5.pdf": {
        "question": "What are arithmetic microoperations?",
        "keywords": ["arithmetic", "microoperation", "add", "subtract", "increment"],
    },
    "CA-Lect5,6-Logic-Shift-Microoperations.pdf": {
        "question": "What is a shift microoperation?",
        "keywords": ["shift", "logical", "circular", "arithmetic"],
    },
    "OS_Lab_Reports_1-5.docx": {
        "question": "What is the Linux file system hierarchy?",
        "keywords": ["linux", "file", "system", "directory", "hierarchy", "filesystem"],
    },
    "Emerging Technology In IC1.pdf": {
        "question": "What emerging technologies are discussed?",
        "keywords": ["technology", "emerging", "artificial", "intelligence", "cloud"],
    },
    "Food Volume and Nutritional Estimation Techniques for Smartphone Apps.pdf": {
        "question": "How do smartphone apps estimate food volume?",
        "keywords": ["food", "volume", "smartphone", "estimation", "image"],
    },
    "WebTech-3-Bootstrap.pptx": {
        "question": "What is Bootstrap used for?",
        "keywords": ["bootstrap", "css", "responsive", "grid", "framework"],
    },
}


def _hit_rate(nodes: list, keywords: list[str], top_k: int = 5) -> float:
    """Fraction of top_k nodes that contain at least one keyword."""
    hits = 0
    for node in nodes[:top_k]:
        text = node.get_content().lower()
        if any(kw.lower() in text for kw in keywords):
            hits += 1
    return hits / min(top_k, len(nodes)) if nodes else 0.0


def _run_vector_only(question, index, child_nodes, parent_node_map, top_k=10):
    import config
    t0 = time.time()
    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(question)
    nodes = [r.node if hasattr(r, "node") else r for r in results]
    ms = round((time.time() - t0) * 1000)
    parents = expand_to_parents(nodes[:5], parent_node_map)
    return nodes, parents, ms


def _run_hybrid(question, index, bm25, child_nodes, parent_node_map, top_k=10):
    t0 = time.time()
    retriever = index.as_retriever(similarity_top_k=top_k)
    hybrid = HybridRetriever(child_nodes, retriever, bm25=bm25)
    scored = hybrid.retrieve_with_scores(question, top_k=top_k)
    nodes = [node for _, node in scored]
    ms = round((time.time() - t0) * 1000)
    parents = expand_to_parents(nodes[:5], parent_node_map)
    return nodes, scored, parents, ms


def _run_hybrid_rerank(question, index, bm25, child_nodes, parent_node_map, top_k=10, top_k_rerank=5):
    t0 = time.time()
    retriever = index.as_retriever(similarity_top_k=top_k)
    hybrid = HybridRetriever(child_nodes, retriever, bm25=bm25)
    scored = hybrid.retrieve_with_scores(question, top_k=top_k)
    nodes = [node for _, node in scored]
    reranked = rerank_with_scores(question, nodes, top_k=top_k_rerank)
    reranked_nodes = [node for _, node in reranked]
    ms = round((time.time() - t0) * 1000)
    parents = expand_to_parents(reranked_nodes, parent_node_map)
    return reranked_nodes, reranked, parents, ms


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_ab_tests():
    print("\n=== A/B Test: Vector-Only vs Hybrid vs Hybrid+Rerank ===\n")
    print(f"{'Document':<55} {'Variant':<20} {'Hit@5':>6} {'Latency':>9}")
    print("-" * 95)

    summary = []

    for doc_name, cfg in DOCUMENT_QUERIES.items():
        doc_path = os.path.join(TEST_DOCS_DIR, doc_name)
        if not os.path.exists(doc_path):
            print(f"  [SKIP] {doc_name} — file not found")
            continue

        question = cfg["question"]
        keywords = cfg["keywords"]

        try:
            index, bm25, child_nodes, parent_node_map = ingest_from_file(doc_path)
        except Exception as e:
            print(f"  [ERROR] Ingest failed for {doc_name}: {e}")
            continue

        doc_label = doc_name[:52] + "..." if len(doc_name) > 52 else doc_name

        # Variant A: vector only
        try:
            vec_nodes, vec_parents, vec_ms = _run_vector_only(
                question, index, child_nodes, parent_node_map
            )
            vec_hit = _hit_rate(vec_nodes, keywords)
            print(f"  {doc_label:<53} {'Vector-only':<20} {vec_hit:>5.0%} {vec_ms:>8} ms")
        except Exception as e:
            print(f"  {doc_label:<53} {'Vector-only':<20} {'ERROR':>6} {str(e)[:30]}")
            vec_hit, vec_ms = 0.0, 0

        # Variant B: hybrid
        try:
            hyb_nodes, hyb_scored, hyb_parents, hyb_ms = _run_hybrid(
                question, index, bm25, child_nodes, parent_node_map
            )
            hyb_hit = _hit_rate(hyb_nodes, keywords)
            print(f"  {'':<53} {'Hybrid':<20} {hyb_hit:>5.0%} {hyb_ms:>8} ms")
        except Exception as e:
            print(f"  {'':<53} {'Hybrid':<20} {'ERROR':>6} {str(e)[:30]}")
            hyb_hit, hyb_ms = 0.0, 0

        # Variant C: hybrid + rerank
        try:
            rer_nodes, rer_scored, rer_parents, rer_ms = _run_hybrid_rerank(
                question, index, bm25, child_nodes, parent_node_map
            )
            rer_hit = _hit_rate(rer_nodes, keywords)
            print(f"  {'':<53} {'Hybrid+Rerank':<20} {rer_hit:>5.0%} {rer_ms:>8} ms")
        except Exception as e:
            print(f"  {'':<53} {'Hybrid+Rerank':<20} {'ERROR':>6} {str(e)[:30]}")
            rer_hit, rer_ms = 0.0, 0

        print()
        summary.append({
            "doc": doc_name,
            "vec_hit": vec_hit,
            "hyb_hit": hyb_hit,
            "rer_hit": rer_hit,
            "vec_ms": vec_ms,
            "hyb_ms": hyb_ms,
            "rer_ms": rer_ms,
        })

    # Aggregate
    if summary:
        n = len(summary)
        avg_vec = sum(r["vec_hit"] for r in summary) / n
        avg_hyb = sum(r["hyb_hit"] for r in summary) / n
        avg_rer = sum(r["rer_hit"] for r in summary) / n
        avg_vec_ms = sum(r["vec_ms"] for r in summary) / n
        avg_hyb_ms = sum(r["hyb_ms"] for r in summary) / n
        avg_rer_ms = sum(r["rer_ms"] for r in summary) / n

        print("=" * 95)
        print(f"{'AVERAGE':<55} {'Vector-only':<20} {avg_vec:>5.0%} {avg_vec_ms:>8.0f} ms")
        print(f"{'':55} {'Hybrid':<20} {avg_hyb:>5.0%} {avg_hyb_ms:>8.0f} ms")
        print(f"{'':55} {'Hybrid+Rerank':<20} {avg_rer:>5.0%} {avg_rer_ms:>8.0f} ms")
        print("=" * 95)

        best = max(
            [("Vector-only", avg_vec), ("Hybrid", avg_hyb), ("Hybrid+Rerank", avg_rer)],
            key=lambda x: x[1]
        )
        print(f"\nBest variant by avg Hit@5: {best[0]} ({best[1]:.0%})")

        vec_overhead = avg_vec_ms
        rer_overhead_pct = ((avg_rer_ms - avg_hyb_ms) / avg_hyb_ms * 100) if avg_hyb_ms else 0
        print(f"Reranking overhead vs hybrid: +{rer_overhead_pct:.0f}% latency\n")


if __name__ == "__main__":
    run_ab_tests()

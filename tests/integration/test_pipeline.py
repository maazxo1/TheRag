"""
Integration tests — full end-to-end pipeline with real test documents.

Tests every document type in test_documents/:
  - PDF (text-based)
  - DOCX
  - Complex PDF (tables, diagrams)

Checks per document:
  1. Ingest succeeds and produces clean chunks (no binary garbage)
  2. Hybrid retrieval returns relevant results
  3. Reranker correctly orders results
  4. LLM produces a non-empty answer
  5. Confidence score is within valid range
  6. Query log entry is written to disk

Run:
    venv/Scripts/python tests/integration/test_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import re
from entrypoint.ingest import ingest_from_file
from entrypoint.query import run_pipeline

TEST_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_documents")

# One representative query per document (must be answerable from the doc)
DOCUMENT_QUERIES = {
    "CA_Lab5.pdf":                                      "What are arithmetic microoperations?",
    "CA-Lect5,6-Logic-Shift-Microoperations.pdf":       "What is a shift microoperation?",
    "OS_Lab_Reports_1-5.docx":                          "What is the Linux file system hierarchy?",
    "Emerging Technology In IC1.pdf":                   "What emerging technologies are discussed?",
    "Food Volume and Nutritional Estimation Techniques for Smartphone Apps.pdf":
                                                        "How do smartphone apps estimate food volume?",
    "WebTech-3-Bootstrap.pptx":                         "What is Bootstrap used for?",
}


def _is_clean(text: str) -> bool:
    """Return True if text has no significant binary garbage."""
    if not text.strip():
        return False
    printable = sum(1 for c in text if c.isprintable())
    return (printable / len(text)) >= 0.80


# ── Per-document tests ────────────────────────────────────────────────────────

def test_ingest_clean_chunks(doc_path: str):
    """All indexed chunks should be clean readable text."""
    index, bm25, child_nodes, parent_node_map = ingest_from_file(doc_path)

    assert len(child_nodes) > 0, "No child chunks produced"
    assert len(parent_node_map) > 0, "No parent chunks produced"

    garbage = [n for n in child_nodes if not _is_clean(n.get_content())]
    ratio = len(garbage) / len(child_nodes)
    assert ratio < 0.10, (
        f"{len(garbage)}/{len(child_nodes)} chunks ({ratio:.0%}) are binary garbage"
    )
    print(f"    [PASS] Ingest: {len(child_nodes)} chunks, {len(garbage)} garbage ({ratio:.0%})")
    return index, bm25, child_nodes, parent_node_map


def test_retrieval_returns_results(question: str, index, bm25, child_nodes, parent_node_map):
    """Hybrid retrieval must return at least one result."""
    result = run_pipeline(
        question=question,
        index=index, bm25=bm25,
        child_nodes=child_nodes, parent_node_map=parent_node_map,
        enable_multi_query=False,
        enable_hyde=False,
        enable_reranking=False,
    )
    stages = result["stages"]
    assert len(stages["hybrid_scored"]) > 0, "Hybrid retrieval returned 0 results"
    print(f"    [PASS] Retrieval: {len(stages['hybrid_scored'])} candidates found")
    return result


def test_reranker_orders_results(question: str, index, bm25, child_nodes, parent_node_map):
    """Reranker top result should have CE score > 0."""
    result = run_pipeline(
        question=question,
        index=index, bm25=bm25,
        child_nodes=child_nodes, parent_node_map=parent_node_map,
        enable_multi_query=False,
        enable_hyde=False,
        enable_reranking=True,
    )
    reranked = result["stages"]["reranked"]
    assert len(reranked) > 0, "Reranker returned 0 results"
    scores = [s for s, _ in reranked]
    assert scores == sorted(scores, reverse=True), "Reranked scores not in descending order"
    print(f"    [PASS] Reranker: top CE score={scores[0]:.4f}, {len(reranked)} results")
    return result


def test_answer_non_empty(result: dict):
    """LLM must produce a non-empty answer."""
    answer = result["answer"]
    assert isinstance(answer, str) and len(answer.strip()) > 10, "Answer is empty or too short"
    print(f"    [PASS] Answer: {len(answer)} chars — '{answer[:80]}...'")


def test_confidence_valid(result: dict):
    """Confidence composite must be in [0, 1]."""
    conf = result["confidence"]
    assert 0.0 <= conf["composite"] <= 1.0, f"Composite out of range: {conf['composite']}"
    assert conf["badge"] in {"HIGH", "MEDIUM", "LOW"}, f"Invalid badge: {conf['badge']}"
    print(f"    [PASS] Confidence: {conf['badge']} ({conf['composite']:.0%}) "
          f"sim={conf['similarity']:.3f} self={conf['self_eval']:.3f} lex={conf['lexical']:.3f}")


def test_log_written(log_path: str, question: str):
    """Query must be written to query_log.jsonl."""
    import json
    assert os.path.exists(log_path), f"Log file not found: {log_path}"
    with open(log_path, encoding="utf-8") as f:
        lines = f.readlines()
    last = json.loads(lines[-1])
    assert last["question"] == question, "Last log entry doesn't match the question"
    print(f"    [PASS] Log written ({len(lines)} total entries)")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config

    print("\n=== Integration Tests: Full Pipeline ===\n")
    passed, failed = 0, 0

    for doc_name, question in DOCUMENT_QUERIES.items():
        doc_path = os.path.join(TEST_DOCS_DIR, doc_name)
        if not os.path.exists(doc_path):
            print(f"  [SKIP] {doc_name} — file not found")
            continue

        print(f"\n-- {doc_name} --")
        print(f"   Query: '{question}'")

        try:
            index, bm25, child_nodes, parent_node_map = test_ingest_clean_chunks(doc_path)
            result = test_retrieval_returns_results(question, index, bm25, child_nodes, parent_node_map)
            result = test_reranker_orders_results(question, index, bm25, child_nodes, parent_node_map)
            test_answer_non_empty(result)
            test_confidence_valid(result)
            test_log_written(config.LOG_PATH, question)
            passed += 1
        except Exception as e:
            print(f"    [FAIL] {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Documents tested : {passed + failed}")
    print(f"Passed           : {passed}")
    print(f"Failed           : {failed}")
    print(f"{'='*50}\n")

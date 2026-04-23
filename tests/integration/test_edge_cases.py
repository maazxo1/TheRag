"""
Edge case tests for the RAG pipeline.

Tests:
  1.  Empty query         — should not crash, returns graceful message
  2.  Whitespace-only     — same as empty
  3.  Very long query     — 2000+ char query, pipeline should not hang
  4.  Out-of-scope query  — question has no answer in documents
  5.  Special characters  — SQL injection, unicode, emoji in query
  6.  Numeric-only query  — "12345"
  7.  Repeated query      — same question twice, results should be consistent
  8.  Top-k stress        — top_k=1, pipeline still returns something
  9.  All toggles on      — multi_query + hyde + reranking simultaneously
  10. All toggles off     — base retrieval only, still works

Run:
    venv/Scripts/python tests/integration/test_edge_cases.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from entrypoint.ingest import ingest_from_file, ingest_from_text
from entrypoint.query import run_pipeline

TEST_DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_documents")

# A single stable document to run edge cases against
STABLE_DOC = os.path.join(TEST_DOCS_DIR, "CA_Lab5.pdf")
_INDEX_CACHE = {}


def _get_index():
    if not _INDEX_CACHE:
        index, bm25, child_nodes, parent_node_map = ingest_from_file(STABLE_DOC)
        _INDEX_CACHE.update({
            "index": index, "bm25": bm25,
            "child_nodes": child_nodes, "parent_node_map": parent_node_map,
        })
    return (
        _INDEX_CACHE["index"], _INDEX_CACHE["bm25"],
        _INDEX_CACHE["child_nodes"], _INDEX_CACHE["parent_node_map"],
    )


def _run(question, **kwargs):
    index, bm25, child_nodes, parent_node_map = _get_index()
    return run_pipeline(
        question=question,
        index=index, bm25=bm25,
        child_nodes=child_nodes, parent_node_map=parent_node_map,
        enable_multi_query=False,
        enable_hyde=False,
        enable_reranking=False,
        **kwargs,
    )


PASS = "[PASS]"
FAIL = "[FAIL]"


def _check(name, fn):
    try:
        fn()
        print(f"  {PASS} {name}")
        return True
    except AssertionError as e:
        print(f"  {FAIL} {name} — {e}")
        return False
    except Exception as e:
        print(f"  {FAIL} {name} — unexpected exception: {type(e).__name__}: {e}")
        return False


# ── Edge case functions ────────────────────────────────────────────────────────

def _test_empty_query():
    result = _run("   ")
    # Pipeline should return something, not crash
    assert "answer" in result
    # May be empty or an I-don't-know response — just not a crash
    assert isinstance(result["answer"], str)


def _test_whitespace_only():
    result = _run("\t\n\r  ")
    assert "answer" in result
    assert isinstance(result["answer"], str)


def _test_very_long_query():
    long_q = ("What are arithmetic microoperations and how do they relate to " * 40).strip()
    assert len(long_q) > 1500
    result = _run(long_q)
    assert isinstance(result["answer"], str)
    assert result["timings"]["total_ms"] < 120_000, "Pipeline hung on long query"


def _test_out_of_scope():
    result = _run("What is the recipe for chocolate cake with raspberry jam filling?")
    answer = result["answer"].lower()
    # LLM should express uncertainty, not hallucinate
    uncertain_phrases = [
        "don't know", "not in the", "cannot", "does not", "no information",
        "provided documents", "context", "unable", "not mentioned",
    ]
    assert any(p in answer for p in uncertain_phrases), (
        f"LLM did not express uncertainty for out-of-scope query. Answer: {answer[:200]}"
    )


def _test_special_characters():
    special_queries = [
        "What is'; DROP TABLE chunks; --",
        "What is <script>alert('xss')</script> microoperation?",
        "What is 微操作 (microoperation) in Chinese?",
        "What is 🔢 arithmetic?",
    ]
    for q in special_queries:
        result = _run(q)
        assert "answer" in result, f"No answer key for query: {q[:50]}"
        assert isinstance(result["answer"], str)


def _test_numeric_only():
    result = _run("12345 9999 0")
    assert isinstance(result["answer"], str)


def _test_repeated_query():
    q = "What are arithmetic microoperations?"
    r1 = _run(q)
    r2 = _run(q)
    # Both should return answers, confidence should be similar (within 0.3)
    c1 = r1["confidence"]["composite"]
    c2 = r2["confidence"]["composite"]
    assert abs(c1 - c2) < 0.30, (
        f"Confidence swings too much on repeated query: {c1:.3f} vs {c2:.3f}"
    )


def _test_top_k_1():
    index, bm25, child_nodes, parent_node_map = _get_index()
    result = run_pipeline(
        question="What are arithmetic microoperations?",
        index=index, bm25=bm25,
        child_nodes=child_nodes, parent_node_map=parent_node_map,
        enable_multi_query=False,
        enable_hyde=False,
        enable_reranking=True,
        top_k_retrieval=1,
        top_k_rerank=1,
    )
    assert len(result["stages"]["reranked"]) >= 1
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0


def _test_all_toggles_on():
    index, bm25, child_nodes, parent_node_map = _get_index()
    result = run_pipeline(
        question="What are arithmetic microoperations?",
        index=index, bm25=bm25,
        child_nodes=child_nodes, parent_node_map=parent_node_map,
        enable_multi_query=True,
        enable_hyde=True,
        enable_reranking=True,
    )
    assert result["flags"]["multi_query"] is True
    assert result["flags"]["hyde"] is True
    assert result["flags"]["reranking"] is True
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0


def _test_all_toggles_off():
    index, bm25, child_nodes, parent_node_map = _get_index()
    result = run_pipeline(
        question="What are arithmetic microoperations?",
        index=index, bm25=bm25,
        child_nodes=child_nodes, parent_node_map=parent_node_map,
        enable_multi_query=False,
        enable_hyde=False,
        enable_reranking=False,
    )
    assert result["flags"]["multi_query"] is False
    assert result["flags"]["hyde"] is False
    assert result["flags"]["reranking"] is False
    assert isinstance(result["answer"], str) and len(result["answer"]) > 0


def _test_text_ingest_edge():
    """Ingesting very short text raises a clear ValueError (not an unhandled crash)."""
    try:
        ingest_from_text("Hello world.")
        # If it doesn't raise, the text was long enough — that's also fine
    except ValueError as e:
        assert "No usable text chunks" in str(e), f"Unexpected ValueError: {e}"
    except Exception as e:
        raise AssertionError(f"Should raise ValueError, got {type(e).__name__}: {e}")


def _test_empty_text_ingest():
    """Ingesting whitespace-only text should raise or produce empty index."""
    try:
        ingest_from_text("   \n\t  ")
        # If it doesn't raise, that's also fine — just no crash
    except Exception:
        pass  # Any exception is acceptable for degenerate input


# ── Runner ─────────────────────────────────────────────────────────────────────

TESTS = [
    ("Empty query", _test_empty_query),
    ("Whitespace-only query", _test_whitespace_only),
    ("Very long query (>1500 chars)", _test_very_long_query),
    ("Out-of-scope question", _test_out_of_scope),
    ("Special characters (SQL injection / XSS / unicode / emoji)", _test_special_characters),
    ("Numeric-only query", _test_numeric_only),
    ("Repeated query — confidence stability", _test_repeated_query),
    ("top_k=1 stress", _test_top_k_1),
    ("All toggles ON (MQ + HyDE + rerank)", _test_all_toggles_on),
    ("All toggles OFF (base retrieval)", _test_all_toggles_off),
    ("Text ingest — very short text", _test_text_ingest_edge),
    ("Text ingest — whitespace only", _test_empty_text_ingest),
]


if __name__ == "__main__":
    if not os.path.exists(STABLE_DOC):
        print(f"ERROR: Stable test document not found: {STABLE_DOC}")
        sys.exit(1)

    print("\n=== Edge Case Tests ===\n")
    print(f"  Document: {os.path.basename(STABLE_DOC)}")
    print(f"  Loading index (first run ingests from scratch)...\n")

    passed, failed = 0, 0
    for name, fn in TESTS:
        ok = _check(name, fn)
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"Edge cases tested : {passed + failed}")
    print(f"Passed            : {passed}")
    print(f"Failed            : {failed}")
    print(f"{'='*50}\n")

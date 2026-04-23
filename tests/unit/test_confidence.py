"""
Component 7 test — confidence.py

Checks:
  1. similarity_score maps cross-encoder logits to [0, 1] correctly
  2. lexical_score is higher for overlapping text than non-overlapping
  3. self_eval_score returns a float in [0, 1] range
  4. compute_confidence returns all required keys
  5. composite score is within [0, 1]
  6. Badge assignment: HIGH >= 0.70, MEDIUM >= 0.45, LOW < 0.45
  7. format_confidence_md produces a string containing the badge label
  8. High-quality answer scores higher composite than low-quality answer
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.pipelines.generation_pipeline import (
    _similarity_score, _self_eval_score, _lexical_score,
    compute_confidence, format_confidence_md,
)
from unittest.mock import patch
from llama_index.core.schema import TextNode


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_scored_nodes(scores):
    """Create (score, node) tuples with given cross-encoder scores."""
    return [(s, TextNode(text=f"node text {i}", id_=f"n{i}"))
            for i, s in enumerate(scores)]


GOOD_CONTEXT = (
    "BM25 is a keyword-based retrieval algorithm that scores documents "
    "based on term frequency and inverse document frequency. "
    "It is widely used in search engines for its simplicity and effectiveness."
)
GOOD_ANSWER = (
    "BM25 ranks documents using term frequency and inverse document frequency. "
    "It is a keyword-based retrieval method used in search engines."
)
BAD_ANSWER = "The mitochondria is the powerhouse of the cell."
QUESTION = "How does BM25 work?"


# ── Test 1: similarity score ───────────────────────────────────────────────────

def test_similarity_score_range():
    scored = _make_scored_nodes([2.0, 0.5, -1.0])
    score = _similarity_score(scored)
    assert 0.0 <= score <= 1.0, f"Similarity score out of range: {score}"
    print(f"  [PASS] Similarity score in [0,1]: {score:.3f}")


def test_similarity_score_empty():
    score = _similarity_score([])
    assert score == 0.0
    print("  [PASS] Empty scored_nodes -> similarity 0.0")


def test_similarity_higher_for_positive_logits():
    high = _similarity_score(_make_scored_nodes([3.0, 2.0]))
    low = _similarity_score(_make_scored_nodes([-2.0, -3.0]))
    assert high > low, f"Positive logits should yield higher similarity: {high} vs {low}"
    print(f"  [PASS] Positive logits ({high:.3f}) > negative logits ({low:.3f})")


# ── Test 2: lexical score ──────────────────────────────────────────────────────

def test_lexical_score_overlap():
    score = _lexical_score(GOOD_ANSWER, GOOD_CONTEXT)
    assert score > 0.0, f"Expected positive ROUGE-L for overlapping text, got {score}"
    print(f"  [PASS] ROUGE-L for overlapping text: {score:.3f}")


def test_lexical_score_no_overlap():
    score = _lexical_score(BAD_ANSWER, GOOD_CONTEXT)
    print(f"  [PASS] ROUGE-L for non-overlapping text: {score:.3f}")


def test_lexical_good_beats_bad():
    good = _lexical_score(GOOD_ANSWER, GOOD_CONTEXT)
    bad = _lexical_score(BAD_ANSWER, GOOD_CONTEXT)
    assert good > bad, f"Good answer ({good:.3f}) should have higher ROUGE-L than bad ({bad:.3f})"
    print(f"  [PASS] Good answer ROUGE-L ({good:.3f}) > bad answer ({bad:.3f})")


# ── Test 3: self-eval score ────────────────────────────────────────────────────

def test_self_eval_range():
    score = _self_eval_score(QUESTION, GOOD_CONTEXT, GOOD_ANSWER)
    assert 0.0 <= score <= 1.0, f"Self-eval score out of range: {score}"
    print(f"  [PASS] Self-eval score in [0,1]: {score:.3f}")


def test_self_eval_fallback_on_failure():
    with patch("src.pipelines.generation_pipeline.requests.post", side_effect=Exception("network error")):
        score = _self_eval_score(QUESTION, GOOD_CONTEXT, GOOD_ANSWER)
    assert score == 0.5, f"Expected neutral fallback 0.5, got {score}"
    print("  [PASS] Self-eval falls back to 0.5 on LLM failure")


# ── Test 4: compute_confidence ─────────────────────────────────────────────────

def test_compute_confidence_keys():
    scored = _make_scored_nodes([1.5, 0.8, 0.2])
    result = compute_confidence(QUESTION, GOOD_ANSWER, GOOD_CONTEXT, scored)
    required_keys = {"similarity", "self_eval", "lexical", "composite", "badge", "label"}
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"
    print(f"  [PASS] compute_confidence returns all required keys")


def test_composite_in_range():
    scored = _make_scored_nodes([1.0, 0.5])
    result = compute_confidence(QUESTION, GOOD_ANSWER, GOOD_CONTEXT, scored)
    assert 0.0 <= result["composite"] <= 1.0, f"Composite out of range: {result['composite']}"
    print(f"  [PASS] Composite score in [0,1]: {result['composite']:.3f}")


def test_badge_assignment():
    scored = _make_scored_nodes([1.0])
    mod = "src.pipelines.generation_pipeline"

    with patch(f"{mod}._self_eval_score", return_value=1.0):
        with patch(f"{mod}._similarity_score", return_value=1.0):
            with patch(f"{mod}._lexical_score", return_value=1.0):
                r = compute_confidence(QUESTION, GOOD_ANSWER, GOOD_CONTEXT, scored)
                assert r["badge"] == "HIGH", f"Expected HIGH, got {r['badge']}"

    with patch(f"{mod}._self_eval_score", return_value=0.5):
        with patch(f"{mod}._similarity_score", return_value=0.5):
            with patch(f"{mod}._lexical_score", return_value=0.5):
                r = compute_confidence(QUESTION, GOOD_ANSWER, GOOD_CONTEXT, scored)
                assert r["badge"] == "MEDIUM", f"Expected MEDIUM, got {r['badge']}"

    with patch(f"{mod}._self_eval_score", return_value=0.0):
        with patch(f"{mod}._similarity_score", return_value=0.0):
            with patch(f"{mod}._lexical_score", return_value=0.0):
                r = compute_confidence(QUESTION, GOOD_ANSWER, GOOD_CONTEXT, scored)
                assert r["badge"] == "LOW", f"Expected LOW, got {r['badge']}"

    print("  [PASS] Badge assignment: HIGH / MEDIUM / LOW thresholds correct")


# ── Test 5: format_confidence_md ───────────────────────────────────────────────

def test_format_md_contains_badge():
    scored = _make_scored_nodes([1.0])
    result = compute_confidence(QUESTION, GOOD_ANSWER, GOOD_CONTEXT, scored)
    md = format_confidence_md(result)
    assert result["badge"] in md, f"Badge '{result['badge']}' not found in markdown"
    assert "%" in md, "Percentage not found in markdown"
    print(f"  [PASS] format_confidence_md contains badge and percentage")


# ── Test 6: good answer scores higher than bad answer ─────────────────────────

def test_good_answer_beats_bad():
    scored = _make_scored_nodes([1.5, 0.8, 0.2])
    good = compute_confidence(QUESTION, GOOD_ANSWER, GOOD_CONTEXT, scored)
    bad = compute_confidence(QUESTION, BAD_ANSWER, GOOD_CONTEXT, scored)
    assert good["composite"] > bad["composite"], (
        f"Good answer ({good['composite']}) should outscore bad ({bad['composite']})"
    )
    print(f"  [PASS] Good answer composite ({good['composite']:.3f}) > "
          f"bad answer ({bad['composite']:.3f})")
    print(f"         Good: sim={good['similarity']:.3f} self={good['self_eval']:.3f} "
          f"lex={good['lexical']:.3f}")
    print(f"         Bad:  sim={bad['similarity']:.3f}  self={bad['self_eval']:.3f} "
          f"lex={bad['lexical']:.3f}")


# ── Runner ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Component 7: confidence.py ===\n")

    print("-- Similarity score tests --")
    test_similarity_score_range()
    test_similarity_score_empty()
    test_similarity_higher_for_positive_logits()

    print("\n-- Lexical score tests --")
    test_lexical_score_overlap()
    test_lexical_score_no_overlap()
    test_lexical_good_beats_bad()

    print("\n-- Self-eval score tests --")
    test_self_eval_range()
    test_self_eval_fallback_on_failure()

    print("\n-- compute_confidence tests --")
    test_compute_confidence_keys()
    test_composite_in_range()
    test_badge_assignment()

    print("\n-- Format tests --")
    test_format_md_contains_badge()

    print("\n-- End-to-end quality test --")
    test_good_answer_beats_bad()

    print("\n=== All tests passed — ready to build Component 8 (query.py) ===\n")

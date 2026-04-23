"""
Confidence scoring — Stage 5 of the v2.0 pipeline.

Produces a 0.0–1.0 composite score for every generated answer using three signals:
  - similarity (30%): average cosine similarity of the reranked chunks
  - self_eval  (50%): LLM rates its own answer vs the context on a 1–5 Likert scale
  - lexical    (20%): ROUGE-L F1 overlap between answer tokens and context tokens

Badge thresholds:
  >= 0.70  → HIGH   (green)
  >= 0.45  → MEDIUM (amber)
  <  0.45  → LOW    (red)
"""

import math
import re
import requests
from rouge_score import rouge_scorer

import config

_session = requests.Session()

_SELF_EVAL_PROMPT = """\
You are an answer quality evaluator. Be precise and objective.

Context:
{context}

Question: {question}

Answer: {answer}

Rate the Answer using this rubric:
1 = answer is wrong, contradicts the context, or says "I don't know"
2 = answer is on-topic but misses most key facts from the context
3 = answer gets some facts right but omits important details or adds unsupported claims
4 = answer is mostly correct and grounded in the context with only minor omissions
5 = answer is fully correct, specific, and entirely supported by the context with no unsupported claims

Apply the rubric literally. Do not add leniency.

Reply with a single integer (1, 2, 3, 4, or 5) and nothing else."""


def _similarity_score(scored_nodes: list[tuple]) -> float:
    """Average cosine similarity from reranked (score, node) tuples."""
    if not scored_nodes:
        return 0.0
    raw_scores = [s for s, _ in scored_nodes]
    # Cross-encoder scores are logits; sigmoid maps them to [0, 1]
    sigmoid = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
    return sum(sigmoid) / len(sigmoid)


def _self_eval_score(question: str, context: str, answer: str) -> float:
    """Ask the LLM to rate its own answer. Returns 0.0–1.0 (maps 1-5 → 0.0-1.0)."""
    try:
        resp = _session.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": _SELF_EVAL_PROMPT.format(
                    context=context[:2000],
                    question=question,
                    answer=answer,
                ),
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.0, "num_predict": 5},
            },
            timeout=config.REQUEST_TIMEOUT,
        )
        raw = resp.json().get("response", "").strip()
        match = re.search(r"[1-5]", raw)
        if match:
            rating = int(match.group())
            return (rating - 1) / 4.0  # maps 1->0.0, 5->1.0
    except Exception:
        pass
    return 0.5  # neutral fallback


def _lexical_score(answer: str, context: str) -> float:
    """ROUGE-L precision: fraction of answer tokens that appear in the context.
    Measures grounding — how much of what the LLM said is backed by the retrieved text.
    Using answer as the reference makes precision the meaningful signal here."""
    if not answer.strip() or not context.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    # answer=target, context=prediction → precision = answer tokens found in context
    result = scorer.score(answer, context)
    return result["rougeL"].precision


def compute_confidence(
    question: str,
    answer: str,
    context: str,
    scored_nodes: list[tuple],
) -> dict:
    """
    Compute the composite confidence score for a generated answer.

    Args:
        question:     original user question
        answer:       LLM-generated answer text
        context:      context string passed to the LLM
        scored_nodes: list of (cross_encoder_score, node) from rerank_with_scores()

    Returns:
        dict with keys: similarity, self_eval, lexical, composite, badge, label
    """
    w = config.CONFIDENCE_WEIGHTS

    sim = _similarity_score(scored_nodes)
    selfeval = _self_eval_score(question, context, answer)
    lexical = _lexical_score(answer, context)

    composite = w["similarity"] * sim + w["self_eval"] * selfeval + w["lexical"] * lexical

    if composite >= 0.70:
        badge, label = "HIGH", "green"
    elif composite >= 0.45:
        badge, label = "MEDIUM", "amber"
    else:
        badge, label = "LOW", "red"

    return {
        "similarity": round(sim, 3),
        "self_eval": round(selfeval, 3),
        "lexical": round(lexical, 3),
        "composite": round(composite, 3),
        "badge": badge,
        "label": label,
    }


def compute_confidence_fast(
    answer: str,
    context: str,
    scored_nodes: list[tuple],
) -> dict:
    """
    Instant confidence estimate — no LLM call.
    Renormalises config weights across only the two available signals so the
    composite is on the same scale as compute_confidence(), and stores the
    actual weights used so format_confidence_md() can display them accurately.
    """
    w_sim = config.CONFIDENCE_WEIGHTS["similarity"]
    w_lex = config.CONFIDENCE_WEIGHTS["lexical"]
    total = w_sim + w_lex
    w_sim_n = w_sim / total
    w_lex_n = w_lex / total

    sim     = _similarity_score(scored_nodes)
    lexical = _lexical_score(answer, context)
    composite = w_sim_n * sim + w_lex_n * lexical

    if composite >= 0.70:
        badge, label = "HIGH", "green"
    elif composite >= 0.45:
        badge, label = "MEDIUM", "amber"
    else:
        badge, label = "LOW", "red"

    return {
        "similarity":    round(sim, 3),
        "self_eval":     None,
        "lexical":       round(lexical, 3),
        "composite":     round(composite, 3),
        "badge":         badge,
        "label":         label,
        "_weights_used": {"similarity": w_sim_n, "self_eval": None, "lexical": w_lex_n},
    }


def format_confidence_md(scores: dict) -> str:
    """Format confidence scores as a Markdown block for the Gradio UI."""
    import config as _config
    emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    icon = emoji.get(scores["badge"], "⚪")

    # Use the weights that were actually used for this computation when available
    # (fast path stores them); fall back to config for the full path.
    w_actual = scores.get("_weights_used") or _config.CONFIDENCE_WEIGHTS
    self_eval_val = scores.get("self_eval")

    if self_eval_val is not None:
        self_eval_str = f"{self_eval_val:.3f}"
        self_eval_w   = f"{w_actual['self_eval']:.0%}" if w_actual.get("self_eval") else f"{_config.CONFIDENCE_WEIGHTS['self_eval']:.0%}"
    else:
        self_eval_str = "N/A (streaming)"
        self_eval_w   = "—"

    return (
        f"### {icon} Confidence: **{scores['badge']}** ({scores['composite']:.0%})\n\n"
        f"| Signal | Score | Weight |\n"
        f"|--------|-------|--------|\n"
        f"| Vector similarity | {scores['similarity']:.3f} | {w_actual['similarity']:.0%} |\n"
        f"| LLM self-eval | {self_eval_str} | {self_eval_w} |\n"
        f"| ROUGE-L lexical | {scores['lexical']:.3f} | {w_actual['lexical']:.0%} |\n"
    )

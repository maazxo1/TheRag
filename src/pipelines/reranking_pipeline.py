"""
Cross-encoder reranker — Stage 3 of the v2.0 pipeline.

Problem: broad retrieval (top-20) trades precision for recall — many returned
chunks are tangentially related but not the best answer to the query.

Solution: a cross-encoder jointly encodes (query, chunk) pairs and produces a
fine-grained relevance score. Unlike bi-encoders (used for retrieval), the
cross-encoder sees both texts together, so it catches subtle relevance signals
that embedding-based cosine similarity misses.

Cost: O(n) inference calls on the reranker model — fast on CPU for n<=20.
Model: BAAI/bge-reranker-v2-m3 (~1.1 GB, cached in RERANKER_CACHE_DIR after first run).
"""

import os
import threading
from sentence_transformers import CrossEncoder

import config

_reranker_instance = None
_reranker_lock = threading.Lock()


def _get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:  # double-checked locking
                os.makedirs(config.RERANKER_CACHE_DIR, exist_ok=True)
                os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", config.RERANKER_CACHE_DIR)
                _reranker_instance = CrossEncoder(config.RERANKER_MODEL)
    return _reranker_instance


def _prewarm():
    """Load the CrossEncoder model in the background so the first query doesn't pay load time."""
    try:
        _get_reranker()
    except Exception:
        pass


# Start loading the model as soon as this module is imported (app startup / first import)
threading.Thread(target=_prewarm, daemon=True).start()


def rerank(query: str, nodes: list, top_k: int = None) -> list:
    """
    Rerank nodes by cross-encoder relevance score and return top_k.

    Args:
        query:  original user question
        nodes:  candidate nodes from retrieval (typically top-20)
        top_k:  how many to return (defaults to TOP_K_RERANK)

    Returns:
        list of nodes ordered by cross-encoder score descending, length <= top_k
    """
    top_k = top_k or config.TOP_K_RERANK

    if not nodes:
        return []

    reranker = _get_reranker()
    pairs = [(query, node.get_content()) for node in nodes]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)
    return [node for _, node in ranked[:top_k]]


def rerank_with_scores(query: str, nodes: list, top_k: int = None) -> list[tuple]:
    """
    Same as rerank() but returns (score, node) tuples for display/debugging.
    """
    top_k = top_k or config.TOP_K_RERANK

    if not nodes:
        return []

    reranker = _get_reranker()
    pairs = [(query, node.get_content()) for node in nodes]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, nodes), key=lambda x: x[0], reverse=True)
    return [(float(score), node) for score, node in ranked[:top_k]]

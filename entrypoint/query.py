"""
Query pipeline orchestrator — full v2.0 RAG pipeline.

Performance architecture:
  - MQ + HyDE + base hybrid retrieval run in PARALLEL (ThreadPoolExecutor)
  - HybridRetriever is cached — id_to_node dict is built once per index load
  - Log writing is off the critical path (background daemon thread)
  - LLM answer streams token-by-token via run_pipeline_streaming()

Stages:
  1. Multi-query   — LLM generates N query variants, retrieves in parallel, merges
  2. HyDE          — LLM writes hypothetical answer, retrieves by doc-to-doc similarity
  3. Hybrid search — BM25 + vector + RRF (always runs for base coverage + score display)
  4. Reranking     — cross-encoder cuts top-20 candidates to top-5
  5. Parent expand — child chunks expanded to full parent context
  6. LLM generate  — answer generated from parent context (streaming supported)
  7. Confidence    — 3-signal composite score (similarity + self-eval + ROUGE-L)
"""

import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as _futures_wait
from datetime import datetime
from typing import Generator

import config

from src.pipelines.retrieval_pipeline import HybridRetriever
from src.pipelines.multi_query import multi_query_retrieve
from src.pipelines.hyde import hyde_retrieve
from src.pipelines.reranking_pipeline import rerank_with_scores
from src.pipelines.chunking_pipeline import expand_to_parents
from src.pipelines.generation_pipeline import compute_confidence, compute_confidence_fast, format_confidence_md
from src.http_session import session as _session

# Persistent pool — avoids OS thread creation overhead on every query
_retrieval_pool = ThreadPoolExecutor(max_workers=3)


# ── HybridRetriever cache ──────────────────────────────────────────────────────
# Stores (bm25_ref, retriever) — identity check via `is` is safe across GC cycles
# unlike id() which can be reused after the original object is collected.
_cached_bm25: object = None
_cached_retriever: HybridRetriever = None


def _get_hybrid_retriever(child_nodes: list, vector_retriever, bm25) -> HybridRetriever:
    global _cached_bm25, _cached_retriever
    if _cached_bm25 is not bm25:
        _cached_bm25 = bm25
        _cached_retriever = HybridRetriever(child_nodes, vector_retriever, bm25=bm25)
    else:
        # Same index — reuse id_to_node; update vector_retriever if top_k changed
        _cached_retriever.vector_retriever = vector_retriever
    return _cached_retriever


# ── Async log writer ───────────────────────────────────────────────────────────

def _log_query(question: str, result: dict) -> None:
    """Write query record to disk in a daemon thread — never blocks the caller."""
    def _write():
        try:
            os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "question": question,
                "answer": result["answer"],
                "confidence": result["confidence"],
                "timings": result["timings"],
                "flags": result["flags"],
            }
            with open(config.LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    threading.Thread(target=_write, daemon=True).start()


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _llm_generate(prompt: str) -> str:
    """Blocking LLM call — returns complete response string."""
    resp = _session.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": config.LLM_MODEL, "prompt": prompt, "stream": False,
              "keep_alive": "30m"},
        timeout=config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _llm_stream(prompt: str) -> Generator[str, None, None]:
    """Streaming LLM call — yields response tokens one by one."""
    resp = _session.post(
        f"{config.OLLAMA_URL}/api/generate",
        json={"model": config.LLM_MODEL, "prompt": prompt, "stream": True,
              "keep_alive": "30m"},
        timeout=config.REQUEST_TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if token:
                yield token
            if data.get("done", False):
                break


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def _build_prompt(question: str, parents: list) -> tuple[str, str]:
    context = "\n\n---\n\n".join(node.get_content().strip() for node in parents)
    return (
        f"You are a helpful assistant. Answer the question using ONLY the context passages below.\n"
        f"Rules:\n"
        f"- If the question asks for a summary, overview, or main topics: synthesize from what is present in the context.\n"
        f"- If the answer is stated directly, quote or paraphrase it.\n"
        f"- If the context has partial information, give the best answer you can from what is there.\n"
        f"- Only say 'I don't know based on the provided documents' if the context contains NO relevant information at all.\n"
        f"- Never use knowledge outside the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    ), context


def _run_retrieval(
    question: str,
    index,
    bm25,
    child_nodes: list,
    parent_node_map: dict,
    use_multi_query: bool,
    use_hyde: bool,
    top_k_ret: int,
    top_k_rer: int,
    use_reranking: bool,
) -> tuple:
    """
    Run the full retrieval + reranking stage and return
    (hybrid_scored, reranked, parents, context, retrieval_ms, rerank_ms).

    MQ, HyDE, and base hybrid retrieval run in PARALLEL.
    """
    vector_retriever = index.as_retriever(similarity_top_k=top_k_ret)
    hybrid_retriever = _get_hybrid_retriever(child_nodes, vector_retriever, bm25)

    t0 = time.time()

    # Submit all retrieval tasks concurrently to the persistent pool
    futures = {}
    futures["hybrid"] = _retrieval_pool.submit(
        hybrid_retriever.retrieve_with_scores, question, top_k_ret
    )
    if use_multi_query:
        futures["mq"] = _retrieval_pool.submit(
            multi_query_retrieve, question, hybrid_retriever, top_k_ret
        )
    if use_hyde:
        futures["hyde"] = _retrieval_pool.submit(
            hyde_retrieve, question, index, top_k_ret
        )
    # Block until all submitted tasks for this query are done
    _futures_wait(list(futures.values()))

    # Merge results — order: MQ first (wider recall), then HyDE, then base hybrid
    seen_ids: set[str] = set()
    candidates: list = []

    def _add(nodes):
        for node in nodes:
            nid = node.node_id
            if nid not in seen_ids:
                seen_ids.add(nid)
                candidates.append(node)

    if "mq" in futures:
        _add(futures["mq"].result())

    if "hyde" in futures:
        raw = [n.node if hasattr(n, "node") else n for n in futures["hyde"].result()]
        _add(raw)

    hybrid_scored = futures["hybrid"].result()
    _add([node for _, node in hybrid_scored])

    retrieval_ms = (time.time() - t0) * 1000

    # Reranking
    t0 = time.time()
    if use_reranking and candidates:
        reranked = rerank_with_scores(question, candidates[:top_k_ret], top_k=top_k_rer)
    else:
        reranked = [(0.0, node) for node in candidates[:top_k_rer]]
    rerank_ms = (time.time() - t0) * 1000

    # Apply SIM_THRESHOLD: drop chunks whose sigmoid-transformed reranker score is
    # below the minimum relevance threshold. Keep at least one to avoid empty context.
    threshold = config.SIM_THRESHOLD
    above = [(s, n) for s, n in reranked if 1.0 / (1.0 + math.exp(-s)) >= threshold]
    reranked = above if above else reranked[:1]

    top_children = [node for _, node in reranked]
    parents = expand_to_parents(top_children, parent_node_map)

    return hybrid_scored, reranked, parents, candidates, retrieval_ms, rerank_ms


def _empty_result(use_multi_query, use_hyde, use_reranking) -> dict:
    return {
        "answer": "Please enter a question.",
        "confidence": {
            "similarity": 0.0, "self_eval": 0.0, "lexical": 0.0,
            "composite": 0.0, "badge": "LOW", "label": "red",
        },
        "confidence_md": "### Confidence: **LOW** (0%)\n",
        "stages": {
            "hybrid_scored": [], "reranked": [], "parents": [],
            "candidates_count": 0,
        },
        "timings": {
            "retrieval_ms": 0, "rerank_ms": 0, "llm_ms": 0,
            "confidence_ms": 0, "total_ms": 0,
        },
        "flags": {
            "multi_query": use_multi_query,
            "hyde": use_hyde,
            "reranking": use_reranking,
        },
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline(
    question: str,
    index,
    bm25,
    child_nodes: list,
    parent_node_map: dict,
    enable_multi_query: bool = None,
    enable_hyde: bool = None,
    enable_reranking: bool = None,
    top_k_retrieval: int = None,
    top_k_rerank: int = None,
) -> dict:
    """
    Run the full RAG pipeline. Returns a complete result dict.
    Use run_pipeline_streaming() for token-by-token answer delivery.
    """
    use_mq  = enable_multi_query if enable_multi_query is not None else config.ENABLE_MULTI_QUERY
    use_hyd = enable_hyde        if enable_hyde        is not None else config.ENABLE_HYDE
    use_rer = enable_reranking   if enable_reranking   is not None else config.ENABLE_RERANKING
    top_k_ret = top_k_retrieval or config.TOP_K_RETRIEVAL
    top_k_rer = top_k_rerank    or config.TOP_K_RERANK

    if not question.strip():
        return _empty_result(use_mq, use_hyd, use_rer)

    t_total = time.time()

    hybrid_scored, reranked, parents, candidates, retrieval_ms, rerank_ms = _run_retrieval(
        question, index, bm25, child_nodes, parent_node_map,
        use_mq, use_hyd, top_k_ret, top_k_rer, use_rer,
    )

    prompt, context = _build_prompt(question, parents)

    t0 = time.time()
    answer = _llm_generate(prompt)
    llm_ms = (time.time() - t0) * 1000

    t0 = time.time()
    conf = compute_confidence(question, answer, context, reranked)
    confidence_ms = (time.time() - t0) * 1000

    total_ms = (time.time() - t_total) * 1000

    result = {
        "answer": answer,
        "confidence": conf,
        "confidence_md": format_confidence_md(conf),
        "stages": {
            "hybrid_scored": hybrid_scored,
            "reranked": reranked,
            "parents": parents,
            "candidates_count": len(candidates),
        },
        "timings": {
            "retrieval_ms": round(retrieval_ms),
            "rerank_ms": round(rerank_ms),
            "llm_ms": round(llm_ms),
            "confidence_ms": round(confidence_ms),
            "total_ms": round(total_ms),
        },
        "flags": {"multi_query": use_mq, "hyde": use_hyd, "reranking": use_rer},
    }
    _log_query(question, result)
    return result


def run_pipeline_streaming(
    question: str,
    index,
    bm25,
    child_nodes: list,
    parent_node_map: dict,
    enable_multi_query: bool = None,
    enable_hyde: bool = None,
    enable_reranking: bool = None,
    top_k_retrieval: int = None,
    top_k_rerank: int = None,
) -> Generator[dict, None, None]:
    """
    Streaming variant of run_pipeline.

    Yields three phases so the caller can update the UI progressively:

      {"phase": "retrieved",  "stages": ..., "timings": {...}}
          — retrieval + reranking done; caller can show retrieved chunks immediately

      {"phase": "token", "token": "<str>", "answer_so_far": "<str>"}
          — one LLM token arrived; caller appends to displayed answer

      {"phase": "done", **full_result_dict}
          — pipeline complete; caller shows confidence and final timings
    """
    use_mq  = enable_multi_query if enable_multi_query is not None else config.ENABLE_MULTI_QUERY
    use_hyd = enable_hyde        if enable_hyde        is not None else config.ENABLE_HYDE
    use_rer = enable_reranking   if enable_reranking   is not None else config.ENABLE_RERANKING
    top_k_ret = top_k_retrieval or config.TOP_K_RETRIEVAL
    top_k_rer = top_k_rerank    or config.TOP_K_RERANK

    if not question.strip():
        yield {"phase": "done", **_empty_result(use_mq, use_hyd, use_rer)}
        return

    t_total = time.time()

    hybrid_scored, reranked, parents, candidates, retrieval_ms, rerank_ms = _run_retrieval(
        question, index, bm25, child_nodes, parent_node_map,
        use_mq, use_hyd, top_k_ret, top_k_rer, use_rer,
    )

    yield {
        "phase": "retrieved",
        "stages": {
            "hybrid_scored": hybrid_scored,
            "reranked": reranked,
            "parents": parents,
            "candidates_count": len(candidates),
        },
        "timings": {
            "retrieval_ms": round(retrieval_ms),
            "rerank_ms": round(rerank_ms),
        },
        "flags": {"multi_query": use_mq, "hyde": use_hyd, "reranking": use_rer},
    }

    if not parents:
        yield {
            "phase": "done",
            **_empty_result(use_mq, use_hyd, use_rer),
            "answer": (
                "No relevant passages were found in the document for this question. "
                "Try rephrasing, or check that the document covers this topic."
            ),
        }
        return

    prompt, context = _build_prompt(question, parents)

    t0 = time.time()
    answer = ""
    for token in _llm_stream(prompt):
        answer += token
        yield {"phase": "token", "token": token, "answer_so_far": answer}
    llm_ms = (time.time() - t0) * 1000

    # Fast confidence — pure math, no LLM call — so "done" reaches the UI immediately
    t0 = time.time()
    conf = compute_confidence_fast(answer, context, reranked)
    confidence_ms = (time.time() - t0) * 1000

    total_ms = (time.time() - t_total) * 1000

    result = {
        "phase": "done",
        "answer": answer,
        "confidence": conf,
        "confidence_md": format_confidence_md(conf),
        "stages": {
            "hybrid_scored": hybrid_scored,
            "reranked": reranked,
            "parents": parents,
            "candidates_count": len(candidates),
        },
        "timings": {
            "retrieval_ms": round(retrieval_ms),
            "rerank_ms": round(rerank_ms),
            "llm_ms": round(llm_ms),
            "confidence_ms": round(confidence_ms),
            "total_ms": round(total_ms),
        },
        "flags": {"multi_query": use_mq, "hyde": use_hyd, "reranking": use_rer},
    }
    yield result

    # Full self-eval + log written in background — never blocks the caller
    def _bg_log():
        try:
            full_conf = compute_confidence(question, answer, context, reranked)
            _log_query(question, {**result, "confidence": full_conf})
        except Exception:
            _log_query(question, result)

    threading.Thread(target=_bg_log, daemon=True).start()

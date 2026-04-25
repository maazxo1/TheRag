"""
Multi-query retrieval — Stage 1 of the v2.0 pipeline.

Problem: the user's exact phrasing often doesn't match the document's language,
causing relevant chunks to be missed entirely.

Solution: use the LLM to generate N alternative phrasings of the question,
run retrieval for all of them in parallel, then merge and deduplicate results.
This improves recall without sacrificing precision.
"""

import re

import config
from src.http_session import session as _session


_MULTI_QUERY_PROMPT = """\
Generate {n} different phrasings of the question below.
Each phrasing must ask for the same information but use different words or structure.
Output ONLY the phrasings, one per line, no numbering, no explanations.

Original question: {query}

Phrasings:"""


def generate_query_variants(query: str, n: int = config.MULTI_QUERY_N) -> list[str]:
    """
    Ask the LLM for n alternative phrasings of the query.
    Returns the original query + up to n variants (deduped).
    Falls back to [query] only if the LLM call fails.
    """
    try:
        resp = _session.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": _MULTI_QUERY_PROMPT.format(query=query, n=n),
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.4, "num_predict": 80, "num_ctx": 512},
            },
            timeout=config.REQUEST_TIMEOUT,
        )
        raw = resp.json().get("response", "").strip()
        variants = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Strip common list prefixes: "1.", "1)", "-", "•", "*"
            line = re.sub(r"^[\d]+[.)]\s*", "", line)  # "1. " or "1) "
            line = line.strip("-•* \t")
            line = line.strip()
            if line and line != query:
                variants.append(line)
        # Deduplicate while preserving order; keep original first
        seen = {query.lower()}
        unique_variants = []
        for v in variants:
            if v.lower() not in seen and len(v) > 5:
                seen.add(v.lower())
                unique_variants.append(v)
        return [query] + unique_variants[:n]
    except Exception:
        return [query]


def multi_query_retrieve(
    query: str,
    retriever,
    top_k: int = None,
    n_variants: int = config.MULTI_QUERY_N,
) -> list:
    """
    Generate query variants, retrieve sequentially, merge and deduplicate results.
    Sequential retrieval avoids spawning a nested thread pool inside an already-pooled
    thread — Ollama queues concurrent requests on CPU anyway so parallelism offered
    no real benefit while adding thread contention.
    """
    top_k = top_k or config.TOP_K_RETRIEVAL
    variants = generate_query_variants(query, n=n_variants)

    hit_count: dict[str, int] = {}
    node_map: dict[str, object] = {}
    for q in variants:
        try:
            results = retriever.retrieve(q)
        except Exception:
            results = []
        for node in results:
            nid = node.node_id
            hit_count[nid] = hit_count.get(nid, 0) + 1
            node_map.setdefault(nid, node)

    ranked = sorted(hit_count, key=hit_count.__getitem__, reverse=True)
    return [node_map[nid] for nid in ranked[:top_k]]

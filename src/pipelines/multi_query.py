"""
Multi-query retrieval — Stage 1 of the v2.0 pipeline.

Problem: the user's exact phrasing often doesn't match the document's language,
causing relevant chunks to be missed entirely.

Solution: use the LLM to generate N alternative phrasings of the question,
run retrieval for all of them in parallel, then merge and deduplicate results.
This improves recall without sacrificing precision.
"""

import re
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

_session = requests.Session()


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
    Generate query variants, retrieve in parallel, merge and deduplicate results.

    Args:
        query:      original user question
        retriever:  any object with a .retrieve(query) -> list[node] method
                    (HybridRetriever or a plain LlamaIndex retriever)
        top_k:      max results to return after merging
        n_variants: how many alternative phrasings to generate

    Returns:
        merged, deduplicated list of nodes ordered by first appearance
        (earlier = found by more query variants = likely more relevant)
    """
    top_k = top_k or config.TOP_K_RETRIEVAL
    variants = generate_query_variants(query, n=n_variants)

    # Run all retrievals in parallel
    all_results: list[list] = [None] * len(variants)
    with ThreadPoolExecutor(max_workers=len(variants)) as pool:
        future_to_idx = {
            pool.submit(retriever.retrieve, q): i
            for i, q in enumerate(variants)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_results[idx] = future.result()
            except Exception:
                all_results[idx] = []

    # Count how many variants retrieved each node; nodes agreed upon by more
    # variants are more likely to be relevant and should rank higher.
    hit_count: dict[str, int] = {}
    node_map: dict[str, object] = {}
    for result_list in all_results:
        for node in (result_list or []):
            nid = node.node_id
            hit_count[nid] = hit_count.get(nid, 0) + 1
            node_map.setdefault(nid, node)

    ranked = sorted(hit_count, key=hit_count.__getitem__, reverse=True)
    return [node_map[nid] for nid in ranked[:top_k]]

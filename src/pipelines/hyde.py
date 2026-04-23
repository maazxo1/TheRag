"""
HyDE — Hypothetical Document Embeddings (Stage 2 of the v2.0 pipeline).

Problem: a short user question (10 words) and a long document chunk (128 tokens)
occupy different regions of embedding space, so cosine similarity is unreliable.

Solution: ask the LLM to write a hypothetical answer paragraph. That paragraph
lives in the same embedding space as real document chunks (both are long-form
prose in the same domain), so doc-to-doc similarity works far better.

Tradeoff: adds ~500 ms for the generation step. Disable via ENABLE_HYDE=False.
"""

import requests

import config

_session = requests.Session()

_HYDE_PROMPT = """\
Write a detailed paragraph that directly answers the question below.
Write as if you are the source document — use the same domain language.
Do NOT add disclaimers or say you don't know. Just write the answer directly.

Question: {query}

Answer paragraph:"""


def generate_hypothetical_document(query: str) -> str:
    """
    Ask the LLM to write a hypothetical answer paragraph for the query.
    Returns the paragraph text, or the original query on failure.
    """
    try:
        resp = _session.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": _HYDE_PROMPT.format(query=query),
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0.1, "num_predict": 120, "num_ctx": 512},
            },
            timeout=config.REQUEST_TIMEOUT,
        )
        doc = resp.json().get("response", "").strip()
        return doc if len(doc) > 20 else query
    except Exception:
        return query


def hyde_retrieve(query: str, index, top_k: int = None) -> list:
    """
    Full HyDE retrieval:
      1. Generate a hypothetical answer document from the query.
      2. Embed that document (long-form → same space as indexed chunks).
      3. Retrieve by doc-to-doc similarity.

    Args:
        query:  original user question
        index:  LlamaIndex VectorStoreIndex
        top_k:  number of results (defaults to TOP_K_RETRIEVAL)

    Returns:
        list of retrieved nodes, ordered by similarity to hypothetical doc
    """
    top_k = top_k or config.TOP_K_RETRIEVAL

    hypothetical_doc = generate_hypothetical_document(query)

    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(hypothetical_doc)


def hyde_retrieve_with_doc(query: str, index, top_k: int = None) -> tuple[list, str]:
    """
    Same as hyde_retrieve but also returns the hypothetical document for inspection.
    Returns (nodes, hypothetical_doc_text).
    """
    top_k = top_k or config.TOP_K_RETRIEVAL
    hypothetical_doc = generate_hypothetical_document(query)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(hypothetical_doc)
    return nodes, hypothetical_doc

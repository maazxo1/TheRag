"""
Hybrid retrieval: BM25 keyword search + vector semantic search,
merged via Reciprocal Rank Fusion (RRF).

Why hybrid?
  - Pure vector search misses exact terms: model numbers, drug names, legal IDs.
  - Pure BM25 misses semantic matches with different wording.
  - RRF merges both ranked lists without requiring score normalisation.
"""

import numpy as np
from rank_bm25 import BM25Okapi

import config


class HybridRetriever:
    def __init__(self, child_nodes: list, vector_retriever, bm25: BM25Okapi = None):
        """
        Args:
            child_nodes:      list of TextNode (child chunks, same set used at ingest)
            vector_retriever: LlamaIndex retriever (index.as_retriever(...))
            bm25:             pre-built BM25Okapi (from ingest) — rebuilds if None
        """
        self.nodes = child_nodes
        self.vector_retriever = vector_retriever
        self.id_to_node = {n.node_id: n for n in child_nodes}

        if bm25 is not None:
            self.bm25 = bm25
        else:
            corpus = [n.get_content().lower().split() for n in child_nodes]
            self.bm25 = BM25Okapi(corpus)

    # ── RRF ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def reciprocal_rank_fusion(*ranked_lists: list, k: int = 60) -> list:
        """
        Merge any number of ranked ID lists into one fused ranking.
        Score = sum of 1/(k + rank + 1) across all lists a doc appears in.
        k=60 is the standard constant — prevents top-ranked docs from dominating.
        """
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank, doc_id in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=scores.get, reverse=True)

    # ── Retrieval ─────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = None) -> list:
        """
        Run BM25 + vector retrieval, fuse with RRF, return top_k child nodes.

        Args:
            query: user question
            top_k: number of results (defaults to config.TOP_K_RETRIEVAL)

        Returns:
            list of TextNode ordered by fused RRF score
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        # ── BM25 ──────────────────────────────────────────────────────────────────
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_ids = [self.nodes[i].node_id for i in bm25_top_idx]

        # ── Vector ────────────────────────────────────────────────────────────────
        vec_results = self.vector_retriever.retrieve(query)
        vec_ids = [n.node_id for n in vec_results]

        # ── Fuse ──────────────────────────────────────────────────────────────────
        fused_ids = self.reciprocal_rank_fusion(bm25_ids, vec_ids)
        results = [self.id_to_node[i] for i in fused_ids if i in self.id_to_node]
        return results[:top_k]

    def retrieve_with_scores(self, query: str, top_k: int = None) -> list[tuple]:
        """
        Same as retrieve() but returns (rrf_score, node) tuples for inspection.
        """
        top_k = top_k or config.TOP_K_RETRIEVAL

        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_ids = [self.nodes[i].node_id for i in bm25_top_idx]

        vec_results = self.vector_retriever.retrieve(query)
        vec_ids = [n.node_id for n in vec_results]

        k = 60
        score_map: dict[str, float] = {}
        for rank, doc_id in enumerate(bm25_ids):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, doc_id in enumerate(vec_ids):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

        ranked_ids = sorted(score_map, key=score_map.get, reverse=True)[:top_k]
        return [(score_map[i], self.id_to_node[i]) for i in ranked_ids if i in self.id_to_node]

# TheRaG System Architecture Overview

## Small-to-Big Chunking

Small-to-big chunking uses small child chunks of 128 tokens for retrieval precision and large parent chunks of 1024 tokens for LLM generation context. This decouples retrieval quality from context richness. When a child chunk is retrieved, the pipeline expands it to its full parent chunk before passing context to the LLM, ensuring the model sees enough surrounding text to generate a complete answer.

## Reciprocal Rank Fusion (RRF)

RRF merges ranked lists from BM25 and vector search by scoring each document as the sum of 1/(k + rank + 1) across all lists, where k=60. Documents appearing in both lists score higher than top-ranked single-list documents. The constant k=60 prevents top-ranked documents from dominating the fused score. RRF requires no score normalization, making it robust when combining lists with different score distributions.

## HyDE (Hypothetical Document Embeddings)

HyDE asks the LLM to write a hypothetical answer paragraph for the query. That paragraph is embedded and used for retrieval instead of the short query, enabling doc-to-doc similarity matching. This improves retrieval for queries where the user's wording is very different from the document language. The hypothetical document is never shown to the user — it is only used as an embedding probe.

## Cross-Encoder Reranking

Cross-encoders jointly encode the query and document together, producing finer-grained relevance scores than bi-encoders which encode them separately. Bi-encoders produce independent embeddings for query and document, then compare them via dot product. Cross-encoders are too slow for first-stage retrieval over large corpora but ideal for reranking a small candidate set of 20 documents down to the top 5. TheRaG uses BAAI/bge-reranker-v2-m3 as the cross-encoder model.

## Confidence Score

The confidence score has three signals. The first signal is vector similarity with a weight of 30 percent, computed as the average cosine similarity of retrieved chunks. The second signal is LLM self-evaluation with a weight of 50 percent, where the LLM rates its own answer on a 1 to 5 scale. The third signal is ROUGE-L lexical overlap with a weight of 20 percent, measuring token overlap between the answer and context. The composite score is computed as a weighted sum of these three signals. Scores above 0.70 receive a HIGH badge, scores between 0.45 and 0.70 receive a MEDIUM badge, and scores below 0.45 receive a LOW badge.

## Embedding Model

nomic-embed-text is used for embeddings, producing 768-dimensional vectors, running locally via Ollama. The embedding model runs on the local machine and does not send data to any external service. The same model is used for both ingestion and query-time embedding to ensure consistency.

## Multi-Query Retrieval

Multi-query retrieval generates N alternative phrasings of the user question using the LLM, runs retrieval for each in parallel via ThreadPoolExecutor, then merges and deduplicates results. This improves recall when the user's exact wording does not match document language. The original query is always included alongside the generated variants. Results from all queries are merged using node ID deduplication before being passed to the hybrid retrieval stage.

## Vector Database

ChromaDB is used as the persistent vector store, using HNSW indexing with SQLite backend. ChromaDB stores the embedded child chunk vectors on disk and supports fast approximate nearest neighbor search. The index persists across sessions so documents do not need to be re-embedded on every startup.

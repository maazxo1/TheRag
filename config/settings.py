"""
Central configuration for TheRaG — all tuneable parameters in one place.
Every setting can be overridden by an environment variable of the same name.
"""

import os

# ─── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL      = os.environ.get("OLLAMA_URL",      "http://localhost:11434")
LLM_MODEL       = os.environ.get("LLM_MODEL",       "llama3.2:latest")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "180.0"))

# ─── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/raw/"
DB_PATH = "./data/vector_store/"
BM25_INDEX_PATH = "./data/processed/"
RERANKER_CACHE_DIR = "./reranker_cache/"
LOG_PATH = "./logs/query_log.jsonl"

# ─── Small-to-Big Chunking ─────────────────────────────────────────────────────
PARENT_CHUNK_SIZE = 1024         # tokens — fed to the LLM for generation
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 128           # tokens — indexed for retrieval precision
CHILD_CHUNK_OVERLAP = 20

# ─── Retrieval Pipeline ────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 20             # broad candidate set passed to the reranker
TOP_K_RERANK = 5                 # final chunks the LLM actually sees
SIM_THRESHOLD = 0.50             # minimum similarity to include a chunk
MULTI_QUERY_N = 3                # number of alternative query phrasings

# ─── Feature Toggles ───────────────────────────────────────────────────────────
ENABLE_HYDE = True               # HyDE query expansion (adds ~500 ms on CPU)
ENABLE_MULTI_QUERY = True
ENABLE_RERANKING = True

# ─── Reranker ──────────────────────────────────────────────────────────────────
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"   # ~1.1 GB, cached after first run

# ─── Confidence Scoring ────────────────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "similarity": 0.35,          # average cosine similarity of retrieved chunks
    "self_eval":  0.55,          # LLM self-rates answer vs context (1–5 Likert)
    "lexical":    0.10,          # ROUGE-L overlap: answer tokens vs context
}

# ─── Chroma Collection ─────────────────────────────────────────────────────────
CHROMA_COLLECTION = "therag_docs"

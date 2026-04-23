"""
Component 2 test — small_to_big.py + ingest.py

Checks:
  1. small_to_big produces parent + child nodes with correct structure
  2. Parent chunks are larger than child chunks on average
  3. Every child node has a valid parent_id pointing to a real parent
  4. ingest_documents() returns usable index, BM25, and node maps
  5. ChromaDB collection contains the expected number of documents
  6. BM25 retrieval returns results for a test query
  7. load_existing_index() reloads without re-embedding
  8. expand_to_parents() correctly maps child results back to parents
"""

import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import chromadb
from llama_index.core.schema import Document

import config
from src.pipelines.chunking_pipeline import build_small_to_big_nodes, expand_to_parents
from entrypoint.ingest import ingest_documents, load_existing_index

# ── Fixtures ────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines large language
models with external knowledge retrieval. The system first retrieves relevant
documents from a database, then uses those documents as context for generation.

Hybrid search combines dense vector retrieval with sparse BM25 keyword search.
Reciprocal Rank Fusion (RRF) merges the two ranked lists without score
normalisation, improving recall across diverse query types.

Cross-encoder reranking is a two-stage process: a fast retriever fetches a broad
candidate set of 20 documents, then a cross-encoder model scores each query-document
pair precisely to select the top 5 for the language model.

Small-to-big chunking stores child chunks of 128 tokens for retrieval and parent
chunks of 1024 tokens for language model context. This decouples retrieval
granularity from generation context richness.

Confidence scoring combines three signals: cosine similarity of retrieved chunks
(30%), LLM self-evaluation on a 1-5 scale (50%), and ROUGE-L lexical overlap
between the answer and retrieved text (20%). Scores above 0.75 receive a green
badge; scores below 0.45 receive a red badge indicating low confidence.

HyDE (Hypothetical Document Embeddings) generates a hypothetical answer paragraph
before retrieval. This paragraph is embedded and used as the query vector, reducing
the distribution mismatch between short user questions and long document chunks.
""".strip()


def make_sample_doc():
    return [Document(text=SAMPLE_TEXT, metadata={"source": "test"})]


# ── Test 1: small_to_big structure ──────────────────────────────────────────────

def test_node_structure():
    docs = make_sample_doc()
    parent_nodes, child_nodes, parent_node_map = build_small_to_big_nodes(docs)

    assert len(parent_nodes) >= 1, "Expected at least one parent chunk"
    assert len(child_nodes) >= 1, "Expected at least one child chunk"
    assert len(parent_node_map) == len(parent_nodes), "parent_node_map size mismatch"
    print(f"  [PASS] Node structure: {len(parent_nodes)} parents, {len(child_nodes)} children")
    return parent_nodes, child_nodes, parent_node_map


def test_child_smaller_than_parent(parent_nodes, child_nodes):
    avg_parent_len = sum(len(p.text) for p in parent_nodes) / len(parent_nodes)
    avg_child_len  = sum(len(c.text) for c in child_nodes)  / len(child_nodes)
    assert avg_child_len < avg_parent_len, (
        f"Children ({avg_child_len:.0f} chars) should be shorter than parents ({avg_parent_len:.0f} chars)"
    )
    print(f"  [PASS] Avg parent {avg_parent_len:.0f} chars > avg child {avg_child_len:.0f} chars")


def test_parent_id_integrity(child_nodes, parent_node_map):
    for i, child in enumerate(child_nodes):
        pid = child.metadata.get("parent_id")
        assert pid is not None, f"Child node {i} missing parent_id in metadata"
        assert pid in parent_node_map, (
            f"Child node {i} has parent_id '{pid}' not found in parent_node_map"
        )
    print(f"  [PASS] All {len(child_nodes)} child nodes have valid parent_id")


def test_expand_to_parents(child_nodes, parent_node_map):
    # Use all child nodes — should collapse to distinct parents
    parents = expand_to_parents(child_nodes, parent_node_map)
    assert len(parents) >= 1, "expand_to_parents returned nothing"
    assert len(parents) <= len(parent_node_map), "More expanded parents than exist"
    # No duplicates
    ids = [p.node_id for p in parents]
    assert len(ids) == len(set(ids)), "expand_to_parents returned duplicate parents"
    print(f"  [PASS] expand_to_parents: {len(child_nodes)} children -> {len(parents)} unique parents")


# ── Test 2: ingest_documents ────────────────────────────────────────────────────

def test_ingest(tmp_docs_dir):
    print(f"\n  Running ingest on '{tmp_docs_dir}' (this embeds chunks — may take 30–60 s) ...")
    index, bm25, child_nodes, parent_node_map = ingest_documents(
        data_dir=tmp_docs_dir, force=True
    )

    assert index is not None, "ingest returned None index"
    assert bm25 is not None, "ingest returned None BM25"
    assert len(child_nodes) >= 1, "No child nodes returned"
    assert len(parent_node_map) >= 1, "No parent nodes returned"
    print(f"  [PASS] Ingested {len(child_nodes)} child chunks, {len(parent_node_map)} parent chunks")
    return index, bm25, child_nodes, parent_node_map


def test_chroma_count(child_nodes):
    client = chromadb.PersistentClient(path=config.DB_PATH)
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
    count = collection.count()
    assert count == len(child_nodes), (
        f"ChromaDB has {count} docs but expected {len(child_nodes)} child chunks"
    )
    print(f"  [PASS] ChromaDB collection has {count} embedded chunks")


def test_bm25_retrieval(bm25, child_nodes):
    import numpy as np
    query_tokens = "retrieval augmented generation hybrid search".lower().split()
    scores = bm25.get_scores(query_tokens)
    top_idx = int(np.argmax(scores))
    top_text = child_nodes[top_idx].get_content()
    assert len(top_text) > 0, "BM25 top result is empty"
    assert scores[top_idx] > 0, "BM25 top score is 0 — index may be empty"
    print(f"  [PASS] BM25 top result (score={scores[top_idx]:.3f}): '{top_text[:80]}...'")


def test_load_existing_index():
    print("\n  Testing load_existing_index() (no re-embedding) ...")
    index2, bm25_2, child_nodes_2, parent_map_2 = load_existing_index()
    assert index2 is not None
    assert len(child_nodes_2) >= 1
    print(f"  [PASS] Reloaded index with {len(child_nodes_2)} child chunks (no re-embedding)")


# ── Runner ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Component 2: small_to_big.py + ingest.py ===\n")

    # --- Unit tests (no Ollama needed) ---
    print("-- Unit tests: small_to_big --")
    parents, children, parent_map = test_node_structure()
    test_child_smaller_than_parent(parents, children)
    test_parent_id_integrity(children, parent_map)
    test_expand_to_parents(children, parent_map)

    # --- Integration tests (uses Ollama for embedding) ---
    print("\n-- Integration tests: ingest --")

    # Write sample doc to a temp dir inside docs/ so we don't pollute real docs
    tmp_dir = os.path.join(config.DATA_DIR, "_test_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "test_doc.txt"), "w") as f:
        f.write(SAMPLE_TEXT)

    try:
        index, bm25, child_nodes, parent_node_map = test_ingest(tmp_dir)
        test_chroma_count(child_nodes)
        test_bm25_retrieval(bm25, child_nodes)
        test_load_existing_index()
    finally:
        # Clean up temp dir; leave the index in place for next components
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n=== All tests passed — ready to build Component 3 (hybrid_search.py) ===\n")

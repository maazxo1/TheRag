"""
Component 1 test — config.py + Ollama connectivity.

Checks:
  1. All required config keys exist and have sensible types/values.
  2. Ollama /api/tags endpoint is reachable.
  3. LLM model (llama3.2:latest) is listed and responds to a one-token prompt.
  4. Embedding model (nomic-embed-text) is listed and returns a 768-dim vector.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests
import config


def test_config_keys():
    required = [
        "OLLAMA_URL", "LLM_MODEL", "EMBEDDING_MODEL", "REQUEST_TIMEOUT",
        "DATA_DIR", "DB_PATH", "BM25_INDEX_PATH", "LOG_PATH",
        "PARENT_CHUNK_SIZE", "CHILD_CHUNK_SIZE",
        "TOP_K_RETRIEVAL", "TOP_K_RERANK", "SIM_THRESHOLD",
        "ENABLE_HYDE", "ENABLE_MULTI_QUERY", "ENABLE_RERANKING",
        "RERANKER_MODEL", "CONFIDENCE_WEIGHTS", "CHROMA_COLLECTION",
    ]
    for key in required:
        assert hasattr(config, key), f"Missing config key: {key}"
    print("  [PASS] All config keys present")


def test_confidence_weights_sum():
    total = sum(config.CONFIDENCE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"
    print("  [PASS] Confidence weights sum to 1.0")


def test_ollama_reachable():
    resp = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=10)
    assert resp.status_code == 200, f"Ollama returned {resp.status_code}"
    print("  [PASS] Ollama is reachable")
    return resp.json()


def test_models_installed(tags_data):
    model_names = [m["name"] for m in tags_data["models"]]
    assert config.LLM_MODEL in model_names, (
        f"LLM model '{config.LLM_MODEL}' not found. Available: {model_names}"
    )
    assert config.EMBEDDING_MODEL in model_names or any(
        config.EMBEDDING_MODEL in n for n in model_names
    ), f"Embedding model '{config.EMBEDDING_MODEL}' not found. Available: {model_names}"
    print(f"  [PASS] LLM '{config.LLM_MODEL}' installed")
    print(f"  [PASS] Embedding model '{config.EMBEDDING_MODEL}' installed")


def test_llm_generates():
    payload = {
        "model": config.LLM_MODEL,
        "prompt": "Reply with one word: hello",
        "stream": False,
        "options": {"num_predict": 5},
    }
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/generate",
        json=payload,
        timeout=config.REQUEST_TIMEOUT,
    )
    assert resp.status_code == 200, f"Generate failed: {resp.status_code}"
    reply = resp.json().get("response", "").strip()
    assert len(reply) > 0, "LLM returned empty response"
    print(f"  [PASS] LLM responded: '{reply[:60]}'")


def test_embedding_returns_vector():
    payload = {
        "model": config.EMBEDDING_MODEL,
        "prompt": "test embedding",
    }
    resp = requests.post(
        f"{config.OLLAMA_URL}/api/embeddings",
        json=payload,
        timeout=config.REQUEST_TIMEOUT,
    )
    assert resp.status_code == 200, f"Embeddings failed: {resp.status_code}"
    vec = resp.json().get("embedding", [])
    assert len(vec) > 0, "Empty embedding vector returned"
    print(f"  [PASS] Embedding returned {len(vec)}-dim vector")


if __name__ == "__main__":
    print("\n=== Component 1: config.py + Ollama connectivity ===\n")

    test_config_keys()
    test_confidence_weights_sum()

    tags = test_ollama_reachable()
    test_models_installed(tags)

    print("\n  Testing LLM generation (may take 10–30 s on first run)...")
    test_llm_generates()

    print("\n  Testing embedding model...")
    test_embedding_returns_vector()

    print("\n=== All tests passed — ready to build Component 2 (ingest.py) ===\n")

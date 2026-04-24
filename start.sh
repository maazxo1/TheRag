#!/bin/bash
# TheRaG startup script
# Runs Ollama locally when OLLAMA_URL points to localhost (HF Spaces / self-contained).
# Skips starting Ollama when OLLAMA_URL points to an external host (docker-compose / VPS).

set -e

OLLAMA_HOST="${OLLAMA_URL:-http://localhost:11434}"

# ── Decide whether to start a local Ollama instance ────────────────────────────
if echo "$OLLAMA_HOST" | grep -qE "localhost|127\.0\.0\.1"; then
    echo "[TheRaG] Starting local Ollama daemon..."
    ollama serve &

    echo "[TheRaG] Waiting for Ollama to be ready..."
    for i in $(seq 1 60); do
        if curl -sf http://localhost:11434/ > /dev/null 2>&1; then
            echo "[TheRaG] Ollama is ready."
            break
        fi
        if [ "$i" -eq 60 ]; then
            echo "[TheRaG] ERROR: Ollama did not start in time. Exiting."
            exit 1
        fi
        sleep 2
    done

    # Pull models — ollama pull is a no-op if the model is already cached
    EMB_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
    LLM="${LLM_MODEL:-llama3.2:latest}"

    echo "[TheRaG] Pulling embedding model: $EMB_MODEL"
    ollama pull "$EMB_MODEL"

    echo "[TheRaG] Pulling LLM: $LLM"
    ollama pull "$LLM"

    echo "[TheRaG] Models ready."
else
    echo "[TheRaG] External Ollama detected at $OLLAMA_HOST — skipping local startup."
fi

# ── Start the application ───────────────────────────────────────────────────────
echo "[TheRaG] Launching application on port ${PORT:-7860}..."
exec python app.py

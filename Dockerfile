FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF, sentence-transformers, chromadb, and Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && curl -fsSL https://ollama.com/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (separate layer — cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create all runtime directories
RUN mkdir -p data/raw data/processed data/vector_store \
             logs reranker_cache /root/.ollama

# ── Environment ─────────────────────────────────────────────────────────────────
# PORT=7860 is required by Hugging Face Spaces.
# OLLAMA_URL defaults to localhost so the bundled Ollama is used automatically.
# Override OLLAMA_URL to point to an external Ollama when using docker-compose.
ENV PORT=7860
ENV OLLAMA_URL=http://localhost:11434
ENV LLM_MODEL=llama3.2:latest
ENV EMBEDDING_MODEL=nomic-embed-text
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

# Health check — generous start period to allow model downloads on cold start
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD curl -f http://localhost:${PORT}/api/status || exit 1

RUN chmod +x start.sh
CMD ["bash", "start.sh"]

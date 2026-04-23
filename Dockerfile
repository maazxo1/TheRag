FROM python:3.11-slim

WORKDIR /app

# System dependencies for PyMuPDF, sentence-transformers, and chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create runtime directories with correct permissions
RUN mkdir -p data/raw data/processed data/vector_store \
             logs reranker_cache \
    && addgroup --system therag \
    && adduser --system --ingroup therag therag \
    && chown -R therag:therag /app

USER therag

# Default port — override with PORT env var (HF Spaces uses 7860)
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/status || exit 1

CMD ["python", "app.py"]

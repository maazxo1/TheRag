<div align="center">

<h1>TheRaG</h1>
<p><strong>Your documents, privately answered.</strong></p>

<p>
  A fully local Retrieval-Augmented Generation system — upload any document and ask questions about it.<br>
  No API keys. No data leaves your machine. 100% offline.
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Ollama-local-black?logo=ollama&logoColor=white" alt="Ollama"/>
  <img src="https://img.shields.io/badge/ChromaDB-1.5-orange" alt="ChromaDB"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</p>

<p>
  <a href="https://huggingface.co/spaces/maazxo1/TheRag"><img src="https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow" alt="Live Demo on Hugging Face Spaces"/></a>
</p>

</div>

---

## Screenshots

<div align="center">
  <img src="images/2.png" width="780" alt="TheRaG landing page — upload your document"/>
  <br/><em>Upload any document to get started</em>
</div>

<br/>

<div align="center">
  <img src="images/1.png" width="780" alt="Streaming answers with source citations"/>
  <br/><em>Streaming answers with source citations and confidence scoring</em>
</div>

<br/>

<table>
  <tr>
    <td align="center">
      <img src="images/3.png" width="380" alt="Embedding Space — 3D view"/>
      <br/><em>Embedding Space — 3D view</em>
    </td>
    <td align="center">
      <img src="images/4.png" width="380" alt="Embedding Space — 2D view"/>
      <br/><em>Embedding Space — 2D view</em>
    </td>
  </tr>
</table>

---

## Features

| Feature | Description |
|---|---|
| **100% Local** | All processing happens on your machine — no cloud calls |
| **Multi-format** | PDF, DOCX, PPTX, TXT, MD |
| **Small-to-Big Chunking** | Child chunks for precision retrieval, parent chunks for LLM context |
| **Hybrid Search** | BM25 keyword search + vector similarity, fused together |
| **HyDE** | Hypothetical Document Embeddings for query expansion |
| **Multi-Query** | Generates alternative phrasings to improve recall |
| **Cross-Encoder Reranking** | `BAAI/bge-reranker-v2-m3` re-scores candidates for maximum precision |
| **Streaming Answers** | Token-by-token streaming via Server-Sent Events |
| **Confidence Score** | Weighted score from similarity, LLM self-eval, and ROUGE-L |
| **Embedding Visualiser** | Interactive 3D/2D PCA scatter plot of your document's chunk space |

---

## Architecture

```
Document Upload
      │
      ▼
┌─────────────────────────────────────────────┐
│               Ingestion Pipeline            │
│  Read → Clean → Small-to-Big Chunk          │
│  → Embed (Ollama) → ChromaDB + BM25         │
└─────────────────────────────────────────────┘
      │
      ▼  (query time)
┌─────────────────────────────────────────────┐
│               Query Pipeline                │
│                                             │
│  User Question                              │
│       ├─► HyDE expansion                   │
│       └─► Multi-query generation           │
│                    │                        │
│                    ▼                        │
│         Hybrid Retrieval (BM25 + Vector)    │
│                    │                        │
│                    ▼                        │
│         Cross-Encoder Reranking             │
│                    │                        │
│                    ▼                        │
│         Parent-chunk promotion              │
│                    │                        │
│                    ▼                        │
│         Ollama LLM  →  Streaming Answer     │
└─────────────────────────────────────────────┘
```

---

## Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** — async web framework and REST API
- **[LlamaIndex](https://www.llamaindex.ai/)** — RAG orchestration and node management
- **[ChromaDB](https://www.trychroma.com/)** — persistent vector store
- **[Ollama](https://ollama.com/)** — local LLM and embedding inference
- **[sentence-transformers](https://www.sbert.net/)** — cross-encoder reranking
- **[rank-bm25](https://github.com/dorianbrown/rank_bm25)** — BM25 keyword search
- **[scikit-learn](https://scikit-learn.org/)** — PCA for embedding visualisation

---

## Quick Start — Docker (recommended)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed
- At least **8 GB RAM** (16 GB recommended)

```bash
# 1. Clone the repository
git clone https://github.com/your-username/TheRaG.git
cd TheRaG

# 2. Start all services (Ollama + TheRaG)
#    First run pulls the LLM and embedding models (~3 GB)
docker compose up --build

# 3. Open the app
open http://localhost:8000
```

> **Note:** The `ollama-setup` container automatically pulls `llama3.2:latest` and `nomic-embed-text` on first start. Subsequent starts reuse the cached models from the `ollama_models` volume.

---

## Quick Start — Local Development

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

```bash
# 1. Pull the required models
ollama pull llama3.2:latest
ollama pull nomic-embed-text

# 2. Clone and set up environment
git clone https://github.com/your-username/TheRaG.git
cd TheRaG

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# 3. (Optional) configure environment
cp .env.example .env
# Edit .env if your Ollama runs on a different host/port

# 4. Run
python app.py
# App opens automatically at http://localhost:8000
```

---

## Configuration

All settings are environment variables. Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `LLM_MODEL` | `llama3.2:latest` | Chat model to use |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `PORT` | `8000` | Web server port (HF Spaces uses `7860`) |
| `REQUEST_TIMEOUT` | `180.0` | Ollama request timeout in seconds |

Advanced settings (edit `config/settings.py`):

| Setting | Default | Description |
|---|---|---|
| `PARENT_CHUNK_SIZE` | `1024` | Tokens per parent chunk (fed to LLM) |
| `CHILD_CHUNK_SIZE` | `128` | Tokens per child chunk (indexed for retrieval) |
| `TOP_K_RETRIEVAL` | `20` | Candidate chunks before reranking |
| `TOP_K_RERANK` | `5` | Final chunks the LLM sees |
| `ENABLE_HYDE` | `True` | HyDE query expansion |
| `ENABLE_MULTI_QUERY` | `True` | Multi-query retrieval |
| `ENABLE_RERANKING` | `True` | Cross-encoder reranking |

---

## Project Structure

```
TheRaG/
├── app.py                    # FastAPI server — routes and state
├── config/
│   └── settings.py           # All tuneable parameters
├── entrypoint/
│   ├── ingest.py             # Document ingestion pipeline
│   └── query.py              # Streaming query pipeline
├── src/pipelines/
│   ├── chunking_pipeline.py  # Small-to-big chunking
│   ├── retrieval_pipeline.py # Hybrid BM25 + vector search
│   ├── reranking_pipeline.py # Cross-encoder reranking
│   ├── generation_pipeline.py# LLM answer generation
│   ├── hyde.py               # HyDE query expansion
│   └── multi_query.py        # Alternative query generation
├── templates/
│   └── index.html            # Frontend UI (Jinja2)
├── static/
│   ├── app.js                # Frontend JavaScript
│   └── style.css             # Styles
├── tests/
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
├── evals/                    # Evaluation scripts and datasets
├── images/                   # Screenshots for documentation
├── data/                     # Runtime data (gitignored)
│   ├── raw/                  # Uploaded documents
│   ├── processed/            # BM25 index and node caches
│   └── vector_store/         # ChromaDB embeddings
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Deployment on Hugging Face Spaces

TheRaG is **fully self-contained** — Ollama runs inside the Docker container so your PC can be off.

### How it works

`start.sh` is the container entrypoint. On every cold start it:
1. Launches an Ollama daemon in the background
2. Pulls the configured LLM and embedding models (~2.3 GB total)
3. Starts the FastAPI application

> **First start takes ~5–10 minutes** while models download. Subsequent queries are fast.

### Steps

**1. Push your code to GitHub** (see Quick Start above)

**2. Create a Hugging Face account** at [huggingface.co](https://huggingface.co)

**3. Create a new Space**
- Profile → New Space
- SDK: **Docker**
- Visibility: Public or Private
- Hardware: **CPU Basic** (free) is sufficient

**4. Link your GitHub repo**
- In the Space → Settings → Repository → connect your GitHub repo

**OR push directly via git:**
```bash
# Add the HF Space as a remote
git remote add hf https://huggingface.co/spaces/YOUR-USERNAME/TheRaG

# Push — this triggers a build automatically
git push hf main
```
> Use your HF username + an [Access Token](https://huggingface.co/settings/tokens) (write scope) as the password.

**5. Set Space variables** (Settings → Variables and Secrets)

| Name | Value | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.2:latest` | Or `llama3.2:1b` for faster cold starts |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Required |

`OLLAMA_URL` does **not** need to be set — it defaults to `localhost` and the bundled Ollama starts automatically.

**6. Watch the build logs** — when you see `Application startup complete` the app is live at:
```
https://YOUR-USERNAME-TheRaG.hf.space
```

### Hardware recommendation

| Tier | RAM | Cold start | Query speed |
|---|---|---|---|
| CPU Basic (free) | 16 GB | ~8 min | ~3–5 tok/s |
| CPU Upgrade | 32 GB | ~5 min | ~5–8 tok/s |
| T4 GPU | 15 GB VRAM | ~3 min | ~30–50 tok/s |

> **Free tier note:** HF Spaces sleeps after ~30 min of inactivity. The next visit wakes it up and re-downloads models. For always-on deployment, upgrade to a persistent Space or use the paid hardware tier.

> **Persistent storage:** On free tier, ingested documents are wiped on restart. To keep data between restarts, enable [Spaces Persistent Storage](https://huggingface.co/docs/hub/spaces-storage) in your Space settings.

---

## Running Tests

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests (requires Ollama running)
python -m pytest tests/integration/ -v
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

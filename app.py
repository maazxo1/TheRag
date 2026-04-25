"""
TheRaG — FastAPI server
All backend pipeline code is completely unchanged.
"""

import asyncio
import json
import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import config
from entrypoint.ingest import ingest_from_file, load_existing_index
from entrypoint.query import run_pipeline_streaming

app = FastAPI(title="TheRaG")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

_executor = ThreadPoolExecutor(max_workers=4)
_ingest_lock = asyncio.Lock()
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_state: dict = {
    "index": None, "bm25": None, "child_nodes": None,
    "parent_node_map": None, "doc_name": None, "ready": False,
    "n_parents": 0, "n_children": 0,
    "settings": {
        "multi_query": config.ENABLE_MULTI_QUERY,
        "hyde":        config.ENABLE_HYDE,
        "reranking":   config.ENABLE_RERANKING,
    },
}


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request":         request,
        "llm_model":       config.LLM_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
    })


# ── Status ────────────────────────────────────────────────────────────────────

@app.get("/api/status")
async def status():
    return {
        "ready":      _state["ready"],
        "doc_name":   _state["doc_name"],
        "n_parents":  _state["n_parents"],
        "n_children": _state["n_children"],
        "settings":   _state["settings"],
    }


# ── Ingest ────────────────────────────────────────────────────────────────────

@app.post("/api/ingest/file")
async def ingest_file_endpoint(file: UploadFile = File(...)):
    if _ingest_lock.locked():
        raise HTTPException(status_code=409, detail="An upload is already in progress. Please wait.")

    # Stream upload to a temp file in 64 KB chunks — rejects oversized files
    # without loading the whole thing into memory first.
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total = 0
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit."
                    )
                tmp.write(chunk)
    except HTTPException:
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass
        raise

    async with _ingest_lock:
        # Snapshot current state so we can roll back if ingest fails.
        _prev = {k: _state[k] for k in
                 ("index", "bm25", "child_nodes", "parent_node_map",
                  "doc_name", "ready", "n_parents", "n_children")}
        _state["ready"] = False
        try:
            loop = asyncio.get_running_loop()
            index, bm25, child_nodes, parent_node_map = await loop.run_in_executor(
                _executor, lambda: ingest_from_file(tmp_path)
            )
            _state.update({
                "index": index, "bm25": bm25,
                "child_nodes": child_nodes, "parent_node_map": parent_node_map,
                "doc_name":   file.filename, "ready": True,
                "n_parents":  len(parent_node_map),
                "n_children": len(child_nodes),
            })
            return {"ok": True, "doc_name": file.filename,
                    "n_parents": len(parent_node_map), "n_children": len(child_nodes)}
        except Exception as e:
            _state.update(_prev)  # restore previous working state
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass


@app.post("/api/ingest/load")
async def load_existing_endpoint():
    try:
        loop = asyncio.get_running_loop()
        index, bm25, child_nodes, parent_node_map = await loop.run_in_executor(
            _executor, load_existing_index
        )
        _state.update({
            "index": index, "bm25": bm25,
            "child_nodes": child_nodes, "parent_node_map": parent_node_map,
            "doc_name": "Saved index", "ready": True,
            "n_parents":  len(parent_node_map),
            "n_children": len(child_nodes),
        })
        return {"ok": True, "doc_name": "Saved index",
                "n_parents": len(parent_node_map), "n_children": len(child_nodes)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No saved index found on disk.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Query (SSE streaming) ─────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    settings: dict = {}


def _serialize_retrieved(event: dict) -> dict:
    stages   = event["stages"]
    parents  = stages.get("parents", [])
    reranked = stages.get("reranked", [])

    score_map: dict = {}
    for score, node in reranked:
        pid = node.metadata.get("parent_id", node.node_id)
        if pid not in score_map:
            score_map[pid] = score

    sources = []
    for i, node in enumerate(parents, 1):
        text  = node.get_content().strip()
        score = score_map.get(node.node_id)
        sources.append({
            "idx":       i,
            "file_name": node.metadata.get("file_name", "Document"),
            "excerpt":   (text[:600] + "…") if len(text) > 600 else text,
            "score":     round(float(score), 4) if score is not None else None,
        })

    return {
        "phase":            "retrieved",
        "sources":          sources,
        "candidates_count": stages.get("candidates_count", 0),
        "timings":          event["timings"],
        "flags":            event["flags"],
    }


def _serialize_done(event: dict) -> dict:
    return {
        "phase":      "done",
        "answer":     event["answer"],
        "confidence": event["confidence"],
        "timings":    event["timings"],
        "flags":      event["flags"],
    }


@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    if not _state["ready"]:
        raise HTTPException(status_code=400, detail="No document loaded.")

    merged = {**_state["settings"], **req.settings}

    async def generate():
        loop      = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()   # unbounded — no deadlock on disconnect
        cancelled = threading.Event()
        DONE      = object()

        def _put(item, timeout=10):
            """Put item on queue; silently drops if cancelled or timeout exceeded."""
            if cancelled.is_set():
                return
            try:
                asyncio.run_coroutine_threadsafe(queue.put(item), loop).result(timeout=timeout)
            except Exception:
                pass

        def run_sync():
            try:
                for ev in run_pipeline_streaming(
                    question=req.question,
                    index=_state["index"],
                    bm25=_state["bm25"],
                    child_nodes=_state["child_nodes"],
                    parent_node_map=_state["parent_node_map"],
                    enable_multi_query=merged.get("multi_query", config.ENABLE_MULTI_QUERY),
                    enable_hyde=merged.get("hyde",        config.ENABLE_HYDE),
                    enable_reranking=merged.get("reranking",   config.ENABLE_RERANKING),
                ):
                    if cancelled.is_set():
                        return
                    _put(ev)
            except Exception as exc:
                _put({"phase": "error", "error": str(exc)})
            finally:
                _put(DONE, timeout=5)

        _executor.submit(run_sync)

        try:
            while True:
                item = await queue.get()
                if item is DONE:
                    break
                phase = item.get("phase")
                if phase == "retrieved":
                    item = _serialize_retrieved(item)
                elif phase == "done":
                    item = _serialize_done(item)
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            cancelled.set()  # tells run_sync to stop if client disconnected

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/settings")
async def update_settings(settings: dict):
    _state["settings"].update(settings)
    return {"ok": True}


# ── Embedding visualisation ───────────────────────────────────────────────────

@app.get("/api/embeddings")
async def embeddings_endpoint():
    if not _state["ready"]:
        raise HTTPException(status_code=400, detail="No document loaded.")

    def _compute():
        import numpy as np
        from sklearn.decomposition import PCA

        index           = _state.get("index")
        child_nodes     = _state.get("child_nodes") or []
        parent_node_map = _state.get("parent_node_map") or {}

        # ── Path 1: embeddings already set in memory (some LlamaIndex versions) ──
        nodes_with_emb = [n for n in child_nodes if n.embedding is not None]

        if nodes_with_emb:
            embs       = np.array([n.embedding for n in nodes_with_emb], dtype=np.float32)
            texts      = [n.get_content()[:120].strip().replace("\n", " ") for n in nodes_with_emb]
            parent_ids = [n.metadata.get("parent_id", n.node_id) for n in nodes_with_emb]
        else:
            # ── Path 2: reuse the already-open ChromaDB collection ──────────────
            # LlamaIndex embeds via node.copy(embedding=…) so originals are None.
            # Re-opening PersistentClient would deadlock the SQLite file on Windows.
            # Instead we grab the collection object that was opened during ingest.
            if index is None:
                return {"chunks": [], "passages": [], "n_groups": 0, "variance": []}
            try:
                coll = index.storage_context.vector_store._collection
            except AttributeError:
                return {"chunks": [], "passages": [], "n_groups": 0, "variance": []}

            result   = coll.get(include=["embeddings", "documents", "metadatas"], limit=2000)
            embs_raw = result.get("embeddings")
            if embs_raw is None or len(embs_raw) == 0:
                return {"chunks": [], "passages": [], "n_groups": 0, "variance": []}

            embs       = np.array(embs_raw, dtype=np.float32)
            docs_raw   = result.get("documents") or []
            metas_raw  = result.get("metadatas") or []
            texts      = [(docs_raw[i] or "")[:120].strip().replace("\n", " ") for i in range(len(embs_raw))]
            parent_ids = [(m.get("parent_id", "") if m else "") for m in metas_raw]

        # ── L2-normalise ─────────────────────────────────────────────────────────
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs  = embs / np.where(norms == 0, 1.0, norms)

        # ── Map chunks → parent passages ─────────────────────────────────────────
        unique_parents = list(dict.fromkeys(parent_ids))
        parent_to_grp  = {pid: i for i, pid in enumerate(unique_parents)}

        acc:  dict[str, list] = {pid: [] for pid in unique_parents}
        ptxt: dict[str, str]  = {}
        for i, pid in enumerate(parent_ids):
            acc[pid].append(embs[i])
            if pid not in ptxt:
                ptxt[pid] = texts[i]

        # Prefer full parent node text where available
        for pid, node in parent_node_map.items():
            if pid in ptxt:
                ptxt[pid] = node.get_content()[:120].strip().replace("\n", " ")

        # ── Passage embeddings = mean of children, re-normalised ─────────────────
        passage_embs = np.array(
            [np.mean(acc[pid], axis=0) for pid in unique_parents], dtype=np.float32
        )
        pnorms       = np.linalg.norm(passage_embs, axis=1, keepdims=True)
        passage_embs = passage_embs / np.where(pnorms == 0, 1.0, pnorms)

        # ── PCA: fit on chunks, project passages into the same space ─────────────
        n_comp = min(3, embs.shape[0], embs.shape[1])
        pca    = PCA(n_components=n_comp)
        coords         = np.asarray(pca.fit_transform(embs),       dtype=np.float64)
        passage_coords = np.asarray(pca.transform(passage_embs),   dtype=np.float64)
        variance       = pca.explained_variance_ratio_.tolist()

        def _pad3(c):
            while c.shape[1] < 3:
                c = np.hstack([c, np.zeros((c.shape[0], 1))])
            return c

        coords         = _pad3(coords)
        passage_coords = _pad3(passage_coords)

        # Scale both arrays using chunk ranges so they share coordinate axes
        mins = [float(coords[:, i].min()) for i in range(3)]
        rngs = [float(coords[:, i].max() - coords[:, i].min()) for i in range(3)]

        def _scale(c):
            out = c.copy()
            for i in range(3):
                if rngs[i] > 0:
                    out[:, i] = 2.0 * (c[:, i] - mins[i]) / rngs[i] - 1.0
            return out

        coords         = _scale(coords)
        passage_coords = _scale(passage_coords)

        chunks = [
            {
                "x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2]),
                "text": texts[i],
                "group": parent_to_grp.get(parent_ids[i], 0),
            }
            for i in range(len(texts))
        ]

        passages = [
            {
                "x": float(passage_coords[j, 0]),
                "y": float(passage_coords[j, 1]),
                "z": float(passage_coords[j, 2]),
                "text": ptxt.get(pid, f"Passage {j + 1}"),
                "group": j,
            }
            for j, pid in enumerate(unique_parents)
        ]

        return {"chunks": chunks, "passages": passages, "n_groups": len(unique_parents), "variance": variance}

    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(_executor, _compute)
        return data
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading
    import webbrowser
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    def _open_browser():
        import time; time.sleep(1.2)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

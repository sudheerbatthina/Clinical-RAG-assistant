"""FastAPI service for the Healthcare RAG Assistant.

Run locally:
    uvicorn api:app --reload

Authentication:
    All endpoints except /health and /documents require an X-API-Key header
    matching one of the keys in API_KEYS (set in .env as a comma-separated list).
"""

import json
import logging

import chromadb
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED

from rag_assistant.generator import answer_question
from rag_assistant.config import API_KEYS, CHROMA_DIR, COLLECTION_NAME, DATA_DIR
from rag_assistant.cache import cache_backend
from rag_assistant.db import (
    init_db, create_chat, list_chats, get_chat, delete_chat,
    add_message, get_messages,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare RAG Assistant")

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    init_db()
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
            if pdf_files:
                logger.info("Vector store empty but %d PDF(s) found — auto-indexing...", len(pdf_files))
                import threading
                from rag_assistant.vector_store import index_all_pdfs
                def _bg_index():
                    try:
                        result = index_all_pdfs()
                        count = result.count() if result is not None else 0
                        logger.info("Auto-indexing complete: %d chunks indexed.", count)
                    except Exception as e:
                        logger.error("Auto-indexing failed: %s", e)
                threading.Thread(target=_bg_index, daemon=True).start()
                logger.info("Auto-indexing started in background.")
            else:
                logger.warning(
                    "No documents indexed. "
                    "Call POST /upload to add PDFs or POST /index after placing files in data/."
                )
        else:
            logger.info("Vector store ready: %d chunks in '%s'.", collection.count(), COLLECTION_NAME)
    except Exception as exc:
        logger.warning("Could not check vector store at startup: %s", exc)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str | None = Security(_api_key_header)) -> str:
    if not key or key not in API_KEYS:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return key


get_api_key = _require_api_key


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_cache: bool = True
    user_group: str | None = None  # public / clinical / billing / admin
    chat_id: str | None = None     # existing chat id, "new", or None


class SourceInfo(BaseModel):
    source: str
    page: int
    chunk_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    from_cache: bool
    latency_s: float | None = None
    token_count: int | None = None
    chat_id: str | None = None


# ---------------------------------------------------------------------------
# Chat endpoints
# ---------------------------------------------------------------------------

@app.get("/chats", dependencies=[Depends(_require_api_key)])
def get_chats_list():
    """List all chats sorted by most recently updated."""
    return list_chats()


@app.post("/chats", dependencies=[Depends(_require_api_key)])
def new_chat():
    """Create a new empty chat and return it."""
    return create_chat()


@app.get("/chats/{chat_id}", dependencies=[Depends(_require_api_key)])
def get_chat_detail(chat_id: str):
    """Return a chat and all its messages."""
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {**chat, "messages": get_messages(chat_id)}


@app.delete("/chats/{chat_id}", dependencies=[Depends(_require_api_key)])
def remove_chat(chat_id: str):
    """Delete a chat and all its messages."""
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    delete_chat(chat_id)
    return {"status": "deleted"}


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.post("/index", dependencies=[Depends(_require_api_key)])
def index():
    """Re-index all PDFs in the data/ directory and return the chunk count."""
    try:
        from rag_assistant.vector_store import index_all_pdfs
        result = index_all_pdfs()
        count = result.count() if result is not None else 0
        return {"status": "complete", "chunks": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload", dependencies=[Depends(_require_api_key)])
async def upload(file: UploadFile = File(...)):
    """Upload a PDF, index it in the background, and return immediately."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    import threading
    def _bg_index():
        try:
            from rag_assistant.vector_store import index_single_pdf
            collection = index_single_pdf(dest)
            logger.info("Background upload index complete: %d chunks", collection.count())
        except Exception as e:
            logger.error("Background upload index failed: %s", e)
    threading.Thread(target=_bg_index, daemon=True).start()
    return {
        "status": "upload_received_indexing_started",
        "filename": file.filename,
        "message": "Indexing running in background. Watch deploy logs.",
    }


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_api_key)])
def query(request: QueryRequest):
    """Ask a question and get a grounded answer with citations.

    If chat_id is "new", a chat is created automatically.
    If chat_id is provided, the exchange is persisted to that chat.
    """
    active_chat_id = request.chat_id
    if active_chat_id == "new":
        chat = create_chat()
        active_chat_id = chat["id"]

    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache,
            user_group=request.user_group,
        )
        if active_chat_id:
            add_message(active_chat_id, "user", request.question)
            add_message(active_chat_id, "assistant", result["answer"],
                        json.dumps(result["sources"]))
            result["chat_id"] = active_chat_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents")
def documents():
    """Return unique sources and their chunk counts. No auth required."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            return []
        results = collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in sorted(counts.items())]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    """Public health-check — no authentication required."""
    return {"status": "ok", "cache_backend": cache_backend()}


# Mount frontend AFTER all API routes so API paths take priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

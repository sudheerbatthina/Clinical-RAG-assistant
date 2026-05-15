"""FastAPI service for the Healthcare RAG Assistant.

Run locally:
    uvicorn api:app --reload

Authentication:
    All endpoints except /health and /documents require an X-API-Key header
    matching one of the keys in API_KEYS (set in .env as a comma-separated list).
"""

import json
import logging
import threading

import chromadb
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED

from rag_assistant.generator import answer_question, stream_answer, build_context
from rag_assistant.config import API_KEYS, CHROMA_DIR, COLLECTION_NAME, DATA_DIR, STORAGE_DIR
from rag_assistant.cache import cache_backend
from rag_assistant.db import (
    init_db, create_chat, list_chats, get_chat, delete_chat,
    add_message, get_messages, get_conn,
)
from rag_assistant.retriever import retrieve
from rag_assistant.query_rewriter import rewrite_query

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
        from rag_assistant.vector_store import migrate_existing_chunks
        migrate_existing_chunks()
    except Exception as exc:
        logger.warning("Chunk migration failed: %s", exc)

    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
            if pdf_files:
                logger.info("Vector store empty but %d PDF(s) found — auto-indexing...", len(pdf_files))
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
                    "Call POST /upload to add PDFs or POST /chats/{id}/upload for per-chat docs."
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

class QueryDebugRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: str = "global"


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
    """Delete a chat, its messages, and all its session-scoped vectors."""
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    delete_chat(chat_id)
    def _bg_delete():
        try:
            from rag_assistant.vector_store import delete_session_chunks
            delete_session_chunks(chat_id)
        except Exception as e:
            logger.error("Failed to delete session chunks for %s: %s", chat_id, e)
    threading.Thread(target=_bg_delete, daemon=True).start()
    return {"status": "deleted"}


@app.post("/chats/{chat_id}/upload", dependencies=[Depends(_require_api_key)])
async def upload_to_chat(chat_id: str, file: UploadFile = File(...)):
    """Upload a PDF scoped to a specific chat session and index it in the background."""
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    session_dir = STORAGE_DIR / "sessions" / chat_id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / file.filename

    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    filename = file.filename
    def _bg_index():
        try:
            from rag_assistant.vector_store import index_pdf_for_session
            index_pdf_for_session(dest, session_id=chat_id)
            logger.info("Session upload indexed: %s for chat %s", filename, chat_id)
        except Exception as e:
            logger.error("Session upload indexing failed: %s", e)
    threading.Thread(target=_bg_index, daemon=True).start()

    return {"status": "upload_received_indexing_started", "filename": filename}


@app.get("/chats/{chat_id}/documents", dependencies=[Depends(_require_api_key)])
def get_chat_documents(chat_id: str):
    """Return unique sources and chunk counts for a specific chat session."""
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        results = collection.get(
            where={"session_id": {"$eq": chat_id}},
            include=["metadatas"],
        )
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in sorted(counts.items())]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.post("/index", dependencies=[Depends(_require_api_key)])
def index():
    """Re-index all PDFs in the data/ directory into the global store."""
    try:
        from rag_assistant.vector_store import index_all_pdfs
        result = index_all_pdfs()
        count = result.count() if result is not None else 0
        return {"status": "complete", "chunks": count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload", dependencies=[Depends(_require_api_key)])
async def upload(file: UploadFile = File(...)):
    """Upload a PDF to the global store and index it in the background."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

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
    If chat_id is provided, the exchange is persisted to that chat and
    retrieval is scoped to global + that session's documents.
    """
    active_chat_id = request.chat_id
    if active_chat_id == "new":
        chat = create_chat()
        active_chat_id = chat["id"]

    session_id = active_chat_id if active_chat_id else "global"

    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache,
            user_group=request.user_group,
            session_id=session_id,
        )
        if active_chat_id:
            add_message(active_chat_id, "user", request.question)
            add_message(active_chat_id, "assistant", result["answer"],
                        json.dumps(result["sources"]))
            result["chat_id"] = active_chat_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/stream", dependencies=[Depends(_require_api_key)])
async def query_stream(request: QueryRequest):
    """Stream an SSE response for a question with optional chat persistence."""
    active_chat_id = request.chat_id
    if active_chat_id == "new":
        chat = create_chat()
        active_chat_id = chat["id"]

    session_id = active_chat_id if active_chat_id else "global"

    history: list[dict] = []
    if active_chat_id:
        history = get_messages(active_chat_id)

    collected: dict = {"answer": "", "sources": []}

    def _event_stream():
        for chunk in stream_answer(
            question=request.question,
            top_k=request.top_k,
            user_group=request.user_group,
            session_id=session_id,
            history=history,
        ):
            try:
                raw = chunk.removeprefix("data: ").strip()
                payload = json.loads(raw)
                if payload.get("type") == "answer_chunk":
                    collected["answer"] += payload.get("content", "")
                elif payload.get("type") == "sources":
                    collected["sources"] = payload.get("sources", [])
            except Exception:
                pass
            yield chunk

        if active_chat_id:
            add_message(active_chat_id, "user", request.question)
            add_message(active_chat_id, "assistant", collected["answer"],
                        json.dumps(collected["sources"]))
            yield f"data: {json.dumps({'type': 'chat_id', 'chat_id': active_chat_id})}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.get("/admin/documents", dependencies=[Depends(_require_api_key)])
def list_global_documents():
    """List all globally indexed documents with chunk counts (auth required)."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            return []
        results = collection.get(
            where={"session_id": {"$eq": "global"}},
            include=["metadatas"],
        )
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in sorted(counts.items())]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents")
def documents():
    """Return global (session_id='global') sources and chunk counts. No auth required."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if collection.count() == 0:
            return []
        results = collection.get(
            where={"session_id": {"$eq": "global"}},
            include=["metadatas"],
        )
        counts: dict[str, int] = {}
        for meta in results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return [{"source": src, "chunk_count": cnt} for src, cnt in sorted(counts.items())]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/stats", dependencies=[Depends(_require_api_key)])
def admin_stats():
    """System-wide stats: chunk counts, chat/message counts, document list, cache info."""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        total_chunks = collection.count()

        global_results = collection.get(
            where={"session_id": {"$eq": "global"}},
            include=["metadatas"],
        )
        global_chunks = len(global_results["ids"])
        counts: dict[str, int] = {}
        for meta in global_results["metadatas"]:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        documents = [{"source": s, "chunk_count": c} for s, c in sorted(counts.items())]

        with get_conn() as conn:
            total_chats = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            total_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

        return {
            "total_chunks": total_chunks,
            "global_chunks": global_chunks,
            "session_chunks": total_chunks - global_chunks,
            "total_chats": total_chats,
            "total_messages": total_messages,
            "documents": documents,
            "cache_backend": cache_backend(),
            "storage_dir": str(STORAGE_DIR),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/admin/documents", dependencies=[Depends(_require_api_key)])
def delete_global_document_by_param(filename: str = Query(..., description="Exact source filename to delete")):
    """Delete all chunks for a given filename (query param: ?filename=X)."""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        results = collection.get(where={"source": {"$eq": filename}}, include=[])
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        collection.delete(ids=results["ids"])
        return {"deleted": filename, "chunks_removed": len(results["ids"])}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/reindex", dependencies=[Depends(_require_api_key)])
def admin_reindex():
    """Re-index all PDFs in DATA_DIR with session_id='global' in a background thread."""
    pdf_files = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
    filenames = [f.name for f in pdf_files]

    def _bg():
        try:
            from rag_assistant.vector_store import index_all_pdfs
            result = index_all_pdfs()
            count = result.count() if result is not None else 0
            logger.info("Admin reindex complete: %d chunks", count)
        except Exception as e:
            logger.error("Admin reindex failed: %s", e)

    threading.Thread(target=_bg, daemon=True).start()
    return {"status": "reindexing_started", "files": filenames}


@app.post("/admin/query-debug", dependencies=[Depends(_require_api_key)])
def admin_query_debug(request: QueryDebugRequest):
    """Return full retrieval debug info — rewritten query, chunks, context — without calling the LLM."""
    try:
        rewritten = rewrite_query(request.question)
        hits = retrieve(rewritten, top_k=request.top_k, session_id=request.session_id)
        context = build_context(hits)
        return {
            "original_question": request.question,
            "rewritten_question": rewritten,
            "chunks_retrieved": [
                {
                    "chunk_id": h["chunk_id"],
                    "source": h["metadata"].get("source", ""),
                    "page": h["metadata"].get("page_number", 0),
                    "chunk_type": h["metadata"].get("chunk_type", "text"),
                    "session_id": h["metadata"].get("session_id", "global"),
                    "content_preview": h["content"][:200],
                    "distance": round(h.get("distance", 0.0), 4),
                }
                for h in hits
            ],
            "context_sent_to_llm": context,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/sessions", dependencies=[Depends(_require_api_key)])
def admin_sessions():
    """All chats with message counts and session-scoped document counts."""
    try:
        with get_conn() as conn:
            rows = conn.execute("""
                SELECT c.id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) as message_count
                FROM chats c
                LEFT JOIN messages m ON m.chat_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            """).fetchall()
        chats = [dict(r) for r in rows]

        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        for chat in chats:
            try:
                res = collection.get(
                    where={"session_id": {"$eq": chat["id"]}},
                    include=["metadatas"],
                )
                sources = {m.get("source") for m in res["metadatas"]}
                chat["document_count"] = len(sources)
            except Exception:
                chat["document_count"] = 0

        return chats
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/admin/sessions/{chat_id}", dependencies=[Depends(_require_api_key)])
def admin_delete_session(chat_id: str):
    """Delete a chat + messages from SQLite and its session chunks from Chroma."""
    if not get_chat(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    chunks_removed = 0
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
        res = collection.get(where={"session_id": {"$eq": chat_id}}, include=[])
        if res["ids"]:
            collection.delete(ids=res["ids"])
        chunks_removed = len(res["ids"])
    except Exception as e:
        logger.warning("Failed to remove session chunks for %s: %s", chat_id, e)
    delete_chat(chat_id)
    return {"deleted": chat_id, "chunks_removed": chunks_removed}


@app.delete("/admin/documents/{filename}", dependencies=[Depends(_require_api_key)])
def delete_global_document(filename: str):
    """Remove all chunks for a specific source filename from the global chroma index."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        results = collection.get(where={"source": {"$eq": filename}}, include=[])
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        collection.delete(ids=results["ids"])
        return {"deleted": filename, "chunks_removed": len(results["ids"])}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    """Public health-check — no authentication required."""
    return {"status": "ok", "cache_backend": cache_backend()}


# Mount frontend AFTER all API routes so API paths take priority
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

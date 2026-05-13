"""FastAPI service for the Healthcare RAG Assistant.

Run locally:
    uvicorn api:app --reload

Authentication:
    All endpoints except /health require an X-API-Key header matching one of
    the keys in API_KEYS (set in .env as a comma-separated list).
"""

import logging

import chromadb
from fastapi import FastAPI, HTTPException, Security, Depends, UploadFile, File
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED

from rag_assistant.generator import answer_question
from rag_assistant.config import API_KEYS, CHROMA_DIR, COLLECTION_NAME, DATA_DIR
from rag_assistant.cache import cache_backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare RAG Assistant")

# ---------------------------------------------------------------------------
# Startup check
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _check_index() -> None:
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


# ---------------------------------------------------------------------------
# Endpoints
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
    """Upload a PDF, index it immediately, and return the total chunk count."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = DATA_DIR / file.filename
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        contents = await file.read()
        dest.write_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    try:
        from rag_assistant.vector_store import index_single_pdf
        collection = index_single_pdf(dest)
        return {
            "status": "indexed",
            "filename": file.filename,
            "chunks_added": collection.count(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(_require_api_key)])
def query(request: QueryRequest):
    """Ask a question and get a grounded answer with citations."""
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache,
            user_group=request.user_group,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    """Public health-check — no authentication required."""
    return {"status": "ok", "cache_backend": cache_backend()}

"""FastAPI service for the Healthcare RAG Assistant.

Run locally:
    uvicorn api:app --reload

Authentication:
    All endpoints except /health require an X-API-Key header matching one of
    the keys in API_KEYS (set in .env as a comma-separated list).
"""

import logging

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED

from rag_assistant.generator import answer_question
from rag_assistant.config import API_KEYS
from rag_assistant.cache import cache_backend
from rag_assistant.vector_store import index_all_pdfs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Healthcare RAG Assistant")

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


@app.post("/index", dependencies=[Depends(_require_api_key)])
def index():
    """Re-index all PDFs in the data directory into the vector store."""
    try:
        collection = index_all_pdfs()
        return {"status": "complete", "total_chunks": collection.count()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    """Public health-check — no authentication required."""
    return {"status": "ok", "cache_backend": cache_backend()}

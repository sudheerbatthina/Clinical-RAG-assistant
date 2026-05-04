import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from generator import answer_question


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Healthcare RAG Assistant")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_cache: bool = True


class SourceInfo(BaseModel):
    source: str
    page: int
    chunk_id: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    from_cache: bool


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Ask a question and get a grounded answer with citations."""
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            use_cache=request.use_cache,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
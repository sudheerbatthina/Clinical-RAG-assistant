from rank_bm25 import BM25Okapi
import chromadb
from .config import CHROMA_DIR, COLLECTION_NAME

_bm25_index = None
_bm25_ids: list = []
_bm25_docs: list = []
_bm25_metas: list = []


def build_bm25_index() -> int:
    """Build in-memory BM25 index from all Chroma chunks. Call on startup."""
    global _bm25_index, _bm25_ids, _bm25_docs, _bm25_metas
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    results = collection.get(include=["documents", "metadatas"])
    _bm25_ids = results["ids"]
    _bm25_docs = results["documents"]
    _bm25_metas = results["metadatas"]
    tokenized = [doc.lower().split() for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenized)
    return len(_bm25_ids)


def bm25_search(query: str, top_k: int = 20,
                session_filter: str | None = None) -> list[dict]:
    """BM25 keyword search, optionally filtered by session_id."""
    if _bm25_index is None or not _bm25_ids:
        return []
    tokens = query.lower().split()
    scores = _bm25_index.get_scores(tokens)
    ranked = sorted(range(len(scores)),
                    key=lambda i: scores[i], reverse=True)
    results = []
    for i in ranked:
        if scores[i] <= 0:
            continue
        meta = _bm25_metas[i] if _bm25_metas else {}
        if session_filter:
            sid = meta.get("session_id", "global")
            if sid not in ("global", session_filter):
                continue
        results.append({
            "id": _bm25_ids[i],
            "score": float(scores[i]),
            "content": _bm25_docs[i],
            "metadata": meta,
        })
        if len(results) >= top_k:
            break
    return results


def reciprocal_rank_fusion(dense_hits: list[dict],
                           bm25_hits: list[dict],
                           k: int = 60,
                           dense_weight: float = 0.7) -> list[str]:
    """
    Combine dense and BM25 rankings via weighted RRF.
    Returns ordered list of chunk_ids.
    """
    scores: dict[str, float] = {}
    sparse_weight = 1.0 - dense_weight

    for rank, hit in enumerate(dense_hits):
        cid = hit["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank + 1)

    for rank, hit in enumerate(bm25_hits):
        cid = hit["id"]
        scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank + 1)

    return [cid for cid, _ in
            sorted(scores.items(), key=lambda x: x[1], reverse=True)]

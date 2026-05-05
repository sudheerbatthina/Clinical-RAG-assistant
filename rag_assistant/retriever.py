import os
import logging

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from .config import (
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K,
    COHERE_API_KEY, RERANK_CANDIDATES, RERANK_TOP_N,
    GROUP_ACCESS_MAP, DEFAULT_ACCESS_LEVEL,
)

load_dotenv()
logger = logging.getLogger(__name__)

try:
    import cohere as _cohere_module
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False


def _reranking_enabled() -> bool:
    key = COHERE_API_KEY or os.getenv("COHERE_API_KEY", "")
    return _COHERE_AVAILABLE and bool(key)


def _allowed_levels(user_group: str | None) -> list[str]:
    """Return the access levels this group may see.

    Unknown groups default to public-only.  None means no filtering (admin
    shortcut used internally during indexing).
    """
    if user_group is None:
        return list(GROUP_ACCESS_MAP["admin"])  # unrestricted
    return GROUP_ACCESS_MAP.get(user_group, [DEFAULT_ACCESS_LEVEL])


def retrieve(
    query: str,
    top_k: int = TOP_K,
    user_group: str | None = None,
) -> list[dict]:
    """Find the most relevant chunks for a query.

    Args:
        query:      Natural-language question.
        top_k:      Final number of results to return.
        user_group: Access-control group (public/clinical/billing/admin/None).
                    Chunks whose access_level is not in the group's allowed
                    list are excluded before reranking.  None disables the
                    filter (useful for internal/admin calls).

    If COHERE_API_KEY is set, retrieves RERANK_CANDIDATES from Chroma then
    reranks with Cohere to return top_k.  Falls back to pure vector search
    when the key is absent or cohere is not installed.
    """
    use_reranking = _reranking_enabled()
    n_candidates = RERANK_CANDIDATES if use_reranking else top_k

    client = OpenAI()
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    ).data[0].embedding

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    n_candidates = min(n_candidates, collection.count())
    if n_candidates == 0:
        return []

    # Build Chroma where-filter for access_level if user_group is specified
    where: dict | None = None
    if user_group is not None:
        allowed = _allowed_levels(user_group)
        if len(allowed) == 1:
            where = {"access_level": {"$eq": allowed[0]}}
        else:
            where = {"access_level": {"$in": allowed}}

    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_candidates,
    }
    if where is not None:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    hits = [
        {
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i in range(len(results["ids"][0]))
    ]

    if not use_reranking:
        return hits[:top_k]

    # --- Cohere rerank ---
    try:
        key = COHERE_API_KEY or os.getenv("COHERE_API_KEY", "")
        co = _cohere_module.Client(api_key=key)
        docs = [h["content"] for h in hits]
        rerank_result = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=min(top_k, len(docs)),
        )
        reranked = [hits[r.index] for r in rerank_result.results]
        logger.info(f"Reranked {len(hits)} candidates -> {len(reranked)} results")
        return reranked
    except Exception as exc:
        logger.warning(f"Cohere reranking failed ({exc}), falling back to vector scores")
        return hits[:top_k]


if __name__ == "__main__":
    query = "What information must a CSR verify before releasing beneficiary information?"
    print(f"Query: {query}\n")
    print(f"Reranking enabled: {_reranking_enabled()}\n")

    hits = retrieve(query, top_k=3)
    for i, hit in enumerate(hits, start=1):
        print(f"--- Result {i} (distance: {hit['distance']:.4f}) ---")
        print(f"ID: {hit['chunk_id']}")
        print(f"Page: {hit['metadata']['page_number']} | Type: {hit['metadata']['chunk_type']}")
        print(hit["content"][:300])
        print()

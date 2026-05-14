import time
import logging

from dotenv import load_dotenv
from openai import OpenAI

from .config import CHAT_MODEL, TOP_K
from .retriever import retrieve
from .query_rewriter import rewrite_query
from .semantic_cache import get_semantic_cache, save_semantic_cache

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about healthcare policy documents. "
    "Answer ONLY using the information in the provided context. "
    "If the context does not contain enough information to answer the question, say so explicitly. "
    "Always cite the source document name and page number(s) you used in your answer."
)


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = hit["metadata"]["page_number"]
        source = hit["metadata"]["source"]
        blocks.append(f"[Source {i} | {source} | page {page}]\n{hit['content']}")
    return "\n\n".join(blocks)


def answer_question(
    question: str,
    top_k: int = TOP_K,
    use_cache: bool = True,
    user_group: str | None = None,
) -> dict:
    """Run the full RAG pipeline with query rewriting and semantic caching.

    Args:
        question:   The user's natural-language question.
        top_k:      Number of chunks to retrieve.
        use_cache:  Whether to read/write the semantic cache.
        user_group: Access-control group forwarded to retriever.

    Returned dict keys:
        question, answer, sources, from_cache, latency_s, token_count
    """
    if use_cache:
        cached = get_semantic_cache(question)
        if cached:
            logger.info("Semantic cache hit for question")
            cached["from_cache"] = True
            cached.setdefault("latency_s", None)
            cached.setdefault("token_count", None)
            return cached

    rewritten = rewrite_query(question)
    if rewritten != question:
        logger.info("Query rewritten: %r → %r", question, rewritten)

    hits = retrieve(rewritten, top_k=top_k, user_group=user_group)
    context = build_context(hits)

    client = OpenAI()
    t0 = time.time()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    latency_s = round(time.time() - t0, 3)

    result = {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [
            {
                "source": h["metadata"]["source"],
                "page": h["metadata"]["page_number"],
                "chunk_id": h["chunk_id"],
            }
            for h in hits
        ],
        "from_cache": False,
        "latency_s": latency_s,
        "token_count": response.usage.total_tokens if response.usage else None,
    }

    if use_cache:
        save_semantic_cache(question, result)

    return result

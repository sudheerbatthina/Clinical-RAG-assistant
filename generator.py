import logging

from dotenv import load_dotenv
from openai import OpenAI

from config import CHAT_MODEL, TOP_K
from retriever import retrieve
from cache import get_cached_answer, save_cached_answer

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about healthcare \
policy documents. Answer ONLY using the information in the provided context. \
If the context does not contain enough information to answer the question, say so explicitly. \
Always cite the source document name and page number(s) you used in your answer."""


def build_context(hits: list[dict]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = hit["metadata"]["page_number"]
        source = hit["metadata"]["source"]
        blocks.append(f"[Source {i} | {source} | page {page}]\n{hit['content']}")
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int = TOP_K, use_cache: bool = True) -> dict:
    """Run the full RAG pipeline with optional answer caching."""
    if use_cache:
        cached = get_cached_answer(question)
        if cached:
            logger.info("Cache hit for question")
            cached["from_cache"] = True
            return cached

    hits = retrieve(question, top_k=top_k)
    context = build_context(hits)

    client = OpenAI()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    result = {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [
            {"source": h["metadata"]["source"], "page": h["metadata"]["page_number"], "chunk_id": h["chunk_id"]}
            for h in hits
        ],
        "from_cache": False,
    }

    if use_cache:
        save_cached_answer(question, result)

    return result
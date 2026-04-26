from dotenv import load_dotenv
from openai import OpenAI

from config import CHAT_MODEL, TOP_K
from retriever import retrieve

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about healthcare \
policy documents. Answer ONLY using the information in the provided context. \
If the context does not contain enough information to answer the question, say so explicitly. \
Always cite the source document name and page number(s) you used in your answer."""


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        page = hit["metadata"]["page_number"]
        source = hit["metadata"]["source"]
        blocks.append(f"[Source {i} | {source} | page {page}]\n{hit['content']}")
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """Run the full RAG pipeline: retrieve + generate."""
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

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [
            {"source": h["metadata"]["source"], "page": h["metadata"]["page_number"], "chunk_id": h["chunk_id"]}
            for h in hits
        ],
    }
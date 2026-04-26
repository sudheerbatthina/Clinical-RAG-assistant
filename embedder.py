from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config import EMBEDDING_MODEL

load_dotenv()


def embed_chunks(chunks: list[dict], model: str = EMBEDDING_MODEL) -> list[dict]:
    """Generate embeddings for each chunk and attach them to the chunk dicts."""
    if not chunks:
        return chunks

    client = OpenAI()
    texts = [chunk["content"] for chunk in chunks]

    response = client.embeddings.create(model=model, input=texts)

    if len(response.data) != len(chunks):
        raise RuntimeError(
            f"API returned {len(response.data)} embeddings for {len(chunks)} chunks"
        )

    for chunk, embedding_data in zip(chunks, response.data):
        chunk["embedding"] = embedding_data.embedding

    return chunks
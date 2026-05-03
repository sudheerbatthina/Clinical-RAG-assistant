from dotenv import load_dotenv
from openai import OpenAI

from config import EMBEDDING_MODEL

load_dotenv()

BATCH_SIZE = 100


def embed_chunks(chunks: list[dict], model: str = EMBEDDING_MODEL) -> list[dict]:
    """Generate embeddings in batches to stay within API limits."""
    if not chunks:
        return chunks

    client = OpenAI()

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk["content"] for chunk in batch]

        response = client.embeddings.create(model=model, input=texts)

        for chunk, embedding_data in zip(batch, response.data):
            chunk["embedding"] = embedding_data.embedding

        print(f"  Embedded batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1}")

    return chunks
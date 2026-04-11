from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from chunker import chunk_pages
from pdf_extractor import extract_text_from_pdf

load_dotenv()


def embed_chunks(chunks: list[dict], model: str = "text-embedding-3-small") -> list[dict]:
    """Generate embeddings for each chunk and attach them to the chunk dicts.

    Sends all chunk contents in a single batch API call for efficiency.
    Mutates the chunks by adding an 'embedding' key to each.

    Args:
        chunks: List of chunk dicts from chunk_pages.
        model: The OpenAI embedding model to use.

    Returns:
        The same list of chunks, with an 'embedding' key added to each one.
    """
    if not chunks:
        return chunks
    client = OpenAI()

    texts = [chunk["content"] for chunk in chunks]

    response = client.embeddings.create(
        model=model,
        input=texts,
    )

    if len(response.data) != len(chunks):
        raise RuntimeError(
            f"API returned {len(response.data)} embeddings for {len(chunks)} chunks"
        )

    for chunk, embedding_data in zip(chunks, response.data):
        chunk["embedding"] = embedding_data.embedding

    return chunks


if __name__ == "__main__":
    pdf_path = Path("data/cms_hipaa.pdf")
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, source_name=pdf_path.name)

    print(f"Embedding {len(chunks)} chunks...")
    chunks = embed_chunks(chunks)

    print(f"Done. Each chunk now has an embedding.")
    print()

    first = chunks[0]
    print(f"First chunk: {first['chunk_id']}")
    print(f"Content (first 100 chars): {first['content'][:100]}")
    print(f"Embedding dimensions: {len(first['embedding'])}")
    print(f"First 5 values: {first['embedding'][:5]}")
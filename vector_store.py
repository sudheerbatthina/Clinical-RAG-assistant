from pathlib import Path

import chromadb

from chunker import chunk_pages
from embedder import embed_chunks
from pdf_extractor import extract_text_from_pdf

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_chunks"


def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """Store embedded chunks in a persistent Chroma collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[
            {"source": c["source"], "page_number": c["page_number"], "chunk_type": c["chunk_type"]}
            for c in chunks
        ],
    )
    return collection


if __name__ == "__main__":
    pdf_path = Path("data/cms_hipaa.pdf")
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages, source_name=pdf_path.name)
    chunks = embed_chunks(chunks)
    collection = build_vector_store(chunks)
    print(f"Stored {collection.count()} chunks in '{COLLECTION_NAME}'")
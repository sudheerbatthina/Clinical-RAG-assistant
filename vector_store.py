from pathlib import Path

import chromadb

from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR
from chunker import chunk_pages
from embedder import embed_chunks
from pdf_extractor import extract_text_from_pdf


def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """Store embedded chunks in a persistent Chroma collection using upsert."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[
            {"source": c["source"], "page_number": c["page_number"], "chunk_type": c["chunk_type"]}
            for c in chunks
        ],
    )
    return collection


def index_all_pdfs(data_dir: Path = DATA_DIR) -> None:
    """Index every PDF in the data directory."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {data_dir}")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages, source_name=pdf_path.name)
        all_chunks.extend(chunks)
        print(f"  → {len(pages)} pages, {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} total chunks...")
    embed_chunks(all_chunks)

    print("Storing in vector database...")
    collection = build_vector_store(all_chunks)
    print(f"Done. {collection.count()} chunks in '{COLLECTION_NAME}'")


if __name__ == "__main__":
    index_all_pdfs()
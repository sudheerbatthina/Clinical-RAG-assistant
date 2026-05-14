from pathlib import Path

import chromadb

from .config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR
from .chunker import chunk_document
from .embedder import embed_chunks
from .pdf_extractor import extract_elements_and_tables


def build_vector_store(chunks: list[dict], session_id: str = "global") -> chromadb.Collection:
    """Store embedded chunks in a persistent Chroma collection using upsert."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "page_number": c["page_number"],
                "chunk_type": c["chunk_type"],
                "session_id": session_id,
            }
            for c in chunks
        ],
    )
    return collection


def index_all_pdfs(data_dir: Path = DATA_DIR) -> chromadb.Collection:
    """Index every PDF in the data directory using section-aware chunking."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {data_dir}")
        return

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        elements, tables_by_page = extract_elements_and_tables(pdf_path)
        chunks = chunk_document(elements, tables_by_page, source_name=pdf_path.name)
        all_chunks.extend(chunks)
        text_chunks = sum(1 for c in chunks if c["chunk_type"] == "text")
        table_chunks = sum(1 for c in chunks if c["chunk_type"] == "table")
        print(f"  → {text_chunks} text chunks, {table_chunks} table chunks")

    print(f"\nEmbedding {len(all_chunks)} total chunks...")
    embed_chunks(all_chunks)

    print("Storing in vector database...")
    collection = build_vector_store(all_chunks, session_id="global")
    print(f"Done. {collection.count()} chunks in '{COLLECTION_NAME}'")
    return collection


def index_single_pdf(pdf_path: Path) -> chromadb.Collection:
    """Index a single PDF file into the global vector store."""
    print(f"Processing {pdf_path.name}...")
    elements, tables_by_page = extract_elements_and_tables(pdf_path)
    chunks = chunk_document(elements, tables_by_page, source_name=pdf_path.name)
    text_chunks = sum(1 for c in chunks if c["chunk_type"] == "text")
    table_chunks = sum(1 for c in chunks if c["chunk_type"] == "table")
    print(f"  → {text_chunks} text chunks, {table_chunks} table chunks")

    print(f"Embedding {len(chunks)} chunks...")
    embed_chunks(chunks)

    print("Storing in vector database...")
    collection = build_vector_store(chunks, session_id="global")
    print(f"Done. {collection.count()} total chunks in '{COLLECTION_NAME}'")
    return collection


def index_pdf_for_session(pdf_path: Path, session_id: str) -> chromadb.Collection:
    """Index a PDF scoped to a specific chat session."""
    print(f"Processing {pdf_path.name} for session {session_id}...")
    elements, tables_by_page = extract_elements_and_tables(pdf_path)
    chunks = chunk_document(elements, tables_by_page, source_name=pdf_path.name)
    for chunk in chunks:
        chunk["chunk_id"] = f"{session_id}_{chunk['chunk_id']}"
    embed_chunks(chunks)
    collection = build_vector_store(chunks, session_id=session_id)
    print(f"Done. {len(chunks)} chunks indexed for session {session_id}")
    return collection


def delete_session_chunks(session_id: str):
    """Remove all vectors belonging to a specific chat session."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    results = collection.get(where={"session_id": {"$eq": session_id}}, include=[])
    if results["ids"]:
        collection.delete(ids=results["ids"])
        print(f"Deleted {len(results['ids'])} chunks for session {session_id}")


def migrate_existing_chunks():
    """Add session_id='global' to any chunks missing it (one-time migration)."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    results = collection.get(include=["metadatas"])
    ids_to_update = []
    updated_metadatas = []
    for i, meta in enumerate(results["metadatas"]):
        if "session_id" not in meta:
            ids_to_update.append(results["ids"][i])
            updated_metadatas.append({**meta, "session_id": "global"})
    if ids_to_update:
        collection.update(ids=ids_to_update, metadatas=updated_metadatas)
        print(f"Migrated {len(ids_to_update)} chunks to session_id='global'")


if __name__ == "__main__":
    index_all_pdfs()

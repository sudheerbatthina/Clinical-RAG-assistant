from dotenv import load_dotenv
from openai import OpenAI
import chromadb

from vector_store import CHROMA_DIR, COLLECTION_NAME

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Find the top_k most relevant chunks for a query."""
    client = OpenAI()
    query_embedding = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    ).data[0].embedding

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })
    return hits


if __name__ == "__main__":
    query = "What information must a CSR verify before releasing beneficiary information?"
    print(f"Query: {query}\n")

    hits = retrieve(query, top_k=3)
    for i, hit in enumerate(hits, start=1):
        print(f"--- Result {i} (distance: {hit['distance']:.4f}) ---")
        print(f"ID: {hit['chunk_id']}")
        print(f"Page: {hit['metadata']['page_number']} | Type: {hit['metadata']['chunk_type']}")
        print(hit['content'][:300])
        print()
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
CACHE_DIR = Path(".cache")

# Collection
COLLECTION_NAME = "rag_chunks"

# Models
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5

# Embedding batching
EMBEDDING_BATCH_SIZE = 100
import hashlib
import json
from config import CACHE_DIR


def _ensure_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)


def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def get_cached_embedding(text: str) -> list[float] | None:
    """Return cached embedding for text, or None if not cached."""
    _ensure_cache_dir()
    path = CACHE_DIR / f"emb_{_hash_key(text)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cached_embedding(text: str, embedding: list[float]):
    """Save embedding to disk cache."""
    _ensure_cache_dir()
    path = CACHE_DIR / f"emb_{_hash_key(text)}.json"
    path.write_text(json.dumps(embedding))


def get_cached_answer(question: str) -> dict | None:
    """Return cached RAG answer for a question, or None."""
    _ensure_cache_dir()
    path = CACHE_DIR / f"ans_{_hash_key(question)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_cached_answer(question: str, result: dict):
    """Cache a RAG answer."""
    _ensure_cache_dir()
    path = CACHE_DIR / f"ans_{_hash_key(question)}.json"
    path.write_text(json.dumps(result))
"""Two-tier cache: Redis (preferred) with transparent file-based fallback.

Embeddings are cached without TTL (they never change for a given text).
Answers are cached with a 24-hour TTL so stale responses expire naturally.

The public interface is identical to the original file-only version:
    get_cached_embedding / save_cached_embedding
    get_cached_answer    / save_cached_answer
"""

import hashlib
import json
import logging
from typing import Any

from .config import CACHE_DIR, REDIS_URL, ANSWER_CACHE_TTL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis connection — attempted once at import time; None means "unavailable"
# ---------------------------------------------------------------------------

try:
    import redis as _redis_lib

    _redis_client: "redis.Redis | None" = _redis_lib.from_url(
        REDIS_URL, socket_connect_timeout=1, socket_timeout=1, decode_responses=True
    )
    _redis_client.ping()
    logger.info(f"Redis cache connected: {REDIS_URL}")
except Exception as _e:
    _redis_client = None
    logger.info(f"Redis unavailable ({_e}), using file-based cache")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(exist_ok=True)


# ---------- Redis primitives ----------

def _redis_get(key: str) -> Any | None:
    if _redis_client is None:
        return None
    try:
        raw = _redis_client.get(key)
        return json.loads(raw) if raw is not None else None
    except Exception as exc:
        logger.warning(f"Redis GET failed ({exc}), falling back to file cache")
        return None


def _redis_set(key: str, value: Any, ttl: int | None = None) -> None:
    if _redis_client is None:
        return
    try:
        serialised = json.dumps(value)
        if ttl:
            _redis_client.setex(key, ttl, serialised)
        else:
            _redis_client.set(key, serialised)
    except Exception as exc:
        logger.warning(f"Redis SET failed ({exc}), value not cached in Redis")


# ---------- File primitives ----------

def _file_get(filename: str) -> Any | None:
    _ensure_cache_dir()
    path = CACHE_DIR / filename
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _file_set(filename: str, value: Any) -> None:
    _ensure_cache_dir()
    (CACHE_DIR / filename).write_text(json.dumps(value))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_cached_embedding(text: str) -> list[float] | None:
    """Return cached embedding, checking Redis then file store."""
    h = _hash_key(text)
    redis_key = f"emb:{h}"
    file_name = f"emb_{h}.json"

    result = _redis_get(redis_key)
    if result is not None:
        return result

    result = _file_get(file_name)
    if result is not None and _redis_client is not None:
        # Backfill Redis (no TTL — embeddings are permanent)
        _redis_set(redis_key, result)
    return result


def save_cached_embedding(text: str, embedding: list[float]) -> None:
    """Persist embedding to Redis (no TTL) and file store."""
    h = _hash_key(text)
    _redis_set(f"emb:{h}", embedding)          # no TTL
    _file_set(f"emb_{h}.json", embedding)      # file always written as backup


def get_cached_answer(question: str) -> dict | None:
    """Return cached RAG answer, checking Redis then file store."""
    h = _hash_key(question)
    redis_key = f"ans:{h}"
    file_name = f"ans_{h}.json"

    result = _redis_get(redis_key)
    if result is not None:
        return result

    result = _file_get(file_name)
    if result is not None and _redis_client is not None:
        # Backfill Redis with TTL
        _redis_set(redis_key, result, ttl=ANSWER_CACHE_TTL)
    return result


def save_cached_answer(question: str, result: dict) -> None:
    """Persist answer to Redis (24 h TTL) and file store."""
    h = _hash_key(question)
    _redis_set(f"ans:{h}", result, ttl=ANSWER_CACHE_TTL)
    _file_set(f"ans_{h}.json", result)


def cache_backend() -> str:
    """Return a human-readable description of the active cache backend."""
    return f"redis ({REDIS_URL})" if _redis_client is not None else "file"

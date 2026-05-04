import time
import logging

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from cache import get_cached_embedding, save_cached_embedding

load_dotenv()
logger = logging.getLogger(__name__)


def _embed_with_retry(client: OpenAI, texts: list[str], model: str, max_retries: int = 3) -> list:
    """Call OpenAI embeddings with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return client.embeddings.create(model=model, input=texts).data
        except RateLimitError as e:
            wait = 2 ** attempt
            logger.warning(f"Rate limited. Retrying in {wait}s... (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def embed_chunks(chunks: list[dict], model: str = EMBEDDING_MODEL) -> list[dict]:
    """Generate embeddings with caching and batched API calls."""
    if not chunks:
        return chunks

    client = OpenAI()

    # Check cache first
    uncached_indices = []
    for i, chunk in enumerate(chunks):
        cached = get_cached_embedding(chunk["content"])
        if cached:
            chunk["embedding"] = cached
        else:
            uncached_indices.append(i)

    logger.info(f"{len(chunks) - len(uncached_indices)} cached, {len(uncached_indices)} to embed")

    # Batch embed uncached chunks
    for batch_start in range(0, len(uncached_indices), EMBEDDING_BATCH_SIZE):
        batch_indices = uncached_indices[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
        texts = [chunks[i]["content"] for i in batch_indices]

        results = _embed_with_retry(client, texts, model)

        for idx, emb_data in zip(batch_indices, results):
            chunks[idx]["embedding"] = emb_data.embedding
            save_cached_embedding(chunks[idx]["content"], emb_data.embedding)

        batch_num = batch_start // EMBEDDING_BATCH_SIZE + 1
        total_batches = (len(uncached_indices) - 1) // EMBEDDING_BATCH_SIZE + 1
        logger.info(f"Embedded batch {batch_num}/{total_batches}")

    return chunks
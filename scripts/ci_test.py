"""Standalone CI smoke tests — no OpenAI / Cohere / Redis keys required.

Also invoked via:  python evals.py --ci

Run directly:
    python scripts/ci_test.py
"""

import sys
import traceback
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))


failures: list[str] = []


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception:
        failures.append(name)
        print(f"  FAIL  {name}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------

def test_imports():
    import rag_assistant.config    # noqa: F401
    import rag_assistant.cache     # noqa: F401
    import rag_assistant.chunker   # noqa: F401
    import rag_assistant.embedder  # noqa: F401
    import rag_assistant.vector_store  # noqa: F401


def test_evals_import():
    import evals  # noqa: F401


# ---------------------------------------------------------------------------
# 2. Config correctness
# ---------------------------------------------------------------------------

def test_config_types():
    from rag_assistant.config import (
        CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SECTION_MAX_CHARS,
        EMBEDDING_MODEL, CHAT_MODEL, COLLECTION_NAME,
        API_KEYS, GROUP_ACCESS_MAP, DEFAULT_ACCESS_LEVEL,
        ANSWER_CACHE_TTL, REDIS_URL,
    )
    assert isinstance(CHUNK_SIZE, int) and CHUNK_SIZE > 0
    assert isinstance(CHUNK_OVERLAP, int) and CHUNK_OVERLAP >= 0
    assert isinstance(TOP_K, int) and TOP_K > 0
    assert isinstance(SECTION_MAX_CHARS, int) and SECTION_MAX_CHARS > 0
    assert isinstance(EMBEDDING_MODEL, str) and EMBEDDING_MODEL
    assert isinstance(CHAT_MODEL, str) and CHAT_MODEL
    assert isinstance(COLLECTION_NAME, str) and COLLECTION_NAME
    assert isinstance(API_KEYS, set) and len(API_KEYS) > 0, "API_KEYS must not be empty"
    assert isinstance(GROUP_ACCESS_MAP, dict)
    assert isinstance(DEFAULT_ACCESS_LEVEL, str)
    assert isinstance(ANSWER_CACHE_TTL, int) and ANSWER_CACHE_TTL > 0
    assert isinstance(REDIS_URL, str) and REDIS_URL.startswith("redis")


def test_access_map():
    from rag_assistant.config import GROUP_ACCESS_MAP
    required_groups = {"public", "clinical", "billing", "admin"}
    for group in required_groups:
        assert group in GROUP_ACCESS_MAP, f"Missing group: {group}"
        assert "public" in GROUP_ACCESS_MAP[group], f"'public' missing from {group}"
    # admin sees everything
    assert len(GROUP_ACCESS_MAP["admin"]) >= len(GROUP_ACCESS_MAP["public"])


# ---------------------------------------------------------------------------
# 3. Chunker
# ---------------------------------------------------------------------------

def test_chunk_document_synthetic():
    from rag_assistant.chunker import chunk_document

    class _Meta:
        page_number = 1

    class _El:
        def __init__(self, category: str, text: str):
            self.category = category
            self.text = text
            self.metadata = _Meta()

    elements = [
        _El("Title", "Introduction"),
        _El("NarrativeText", "This section covers basic concepts."),
        _El("NarrativeText", "Additional context here."),
        _El("Title", "Requirements"),
        _El("NarrativeText", "Must comply with all regulations."),
    ]
    chunks = chunk_document(elements, tables_by_page={}, source_name="smoke_test.pdf")

    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
    for c in chunks:
        assert c["chunk_type"] == "text"
        assert c["content"].strip()
        assert c["source"] == "smoke_test.pdf"
        assert isinstance(c["page_number"], int)
        assert c["chunk_id"]


def test_chunk_document_large_section():
    """Sections > SECTION_MAX_CHARS should be split into multiple chunks."""
    from rag_assistant.chunker import chunk_document
    from rag_assistant.config import SECTION_MAX_CHARS

    class _Meta:
        page_number = 2

    class _El:
        def __init__(self, category: str, text: str):
            self.category = category
            self.text = text
            self.metadata = _Meta()

    big_text = ("Word " * 60 + "\n\n") * 10   # ~3 600 chars > SECTION_MAX_CHARS
    elements = [
        _El("Title", "Big Section"),
        _El("NarrativeText", big_text),
    ]
    chunks = chunk_document(elements, tables_by_page={}, source_name="big.pdf")
    assert len(chunks) >= 2, "Large section should produce multiple chunks"
    for c in chunks:
        assert len(c["content"]) <= SECTION_MAX_CHARS + 50  # small tolerance


def test_table_to_markdown():
    from rag_assistant.chunker import table_to_markdown

    table = [["Name", "Age", "Role"], ["Alice", "30", "Admin"], ["Bob", "25", "User"]]
    md = table_to_markdown(table)
    assert "| Name |" in md
    assert "| Alice |" in md
    assert "---" in md
    lines = md.strip().split("\n")
    assert len(lines) == 4  # header + separator + 2 rows


def test_table_to_markdown_empty():
    from rag_assistant.chunker import table_to_markdown
    assert table_to_markdown([]) == ""
    assert table_to_markdown([[]]) == ""


# ---------------------------------------------------------------------------
# 4. Cache
# ---------------------------------------------------------------------------

def test_cache_hash_determinism():
    from rag_assistant.cache import _hash_key
    assert _hash_key("hello") == _hash_key("hello")
    assert _hash_key("hello") != _hash_key("world")
    assert len(_hash_key("x")) == 64  # SHA-256 hex digest


def test_cache_file_roundtrip(tmp_path):
    """File-based cache round-trip using a temp directory."""
    import importlib
    import rag_assistant.cache as cache_mod

    original_dir = cache_mod.CACHE_DIR  # save
    # Monkeypatch CACHE_DIR so we don't pollute the real .cache/
    import rag_assistant.config as cfg
    cfg.CACHE_DIR = tmp_path
    cache_mod.CACHE_DIR = tmp_path  # also patch the module-level reference

    try:
        from rag_assistant.cache import (
            save_cached_embedding, get_cached_embedding,
            save_cached_answer, get_cached_answer,
        )
        # Embedding round-trip
        save_cached_embedding("my text", [0.1, 0.2, 0.3])
        emb = get_cached_embedding("my text")
        assert emb == [0.1, 0.2, 0.3], f"Got {emb}"

        # Answer round-trip
        answer = {"question": "q", "answer": "a", "sources": [], "from_cache": False}
        save_cached_answer("q", answer)
        got = get_cached_answer("q")
        assert got is not None
        assert got["answer"] == "a"
    finally:
        cfg.CACHE_DIR = original_dir
        cache_mod.CACHE_DIR = original_dir


# ---------------------------------------------------------------------------
# 5. Eval helpers
# ---------------------------------------------------------------------------

def test_refusal_detection():
    from evals import _is_refusal
    assert _is_refusal("The context does not contain this information.")
    assert _is_refusal("I cannot find any mention of that topic.")
    assert _is_refusal("Not enough information is provided in the context.")
    assert _is_refusal("This topic is outside the scope of the documents.")
    assert _is_refusal("There is no information about this in the provided context.")
    assert not _is_refusal("The required premium is $174.70 per month.")


def test_score_answer_not_in_docs_pass():
    from evals import score_answer
    result = {
        "answer": "The context does not contain information about Medicare premiums.",
        "sources": [],
    }
    expected = {"not_in_docs": True}
    scores = score_answer(result, expected)
    assert scores["keyword_score"] == 1.0
    assert scores["citation_hit"] is True
    assert scores["refused"] is True


def test_score_answer_not_in_docs_fail():
    from evals import score_answer
    result = {
        "answer": "The monthly premium is $174.70.",
        "sources": [],
    }
    expected = {"not_in_docs": True}
    scores = score_answer(result, expected)
    assert scores["keyword_score"] == 0.0
    assert scores["refused"] is False


def test_score_answer_in_docs():
    from evals import score_answer
    result = {
        "answer": "A CSR must verify the full name and date of birth on page 12.",
        "sources": [{"page": 12, "source": "doc.pdf", "chunk_id": "x"}],
    }
    expected = {"expected_page": 12, "expected_keywords": ["full name", "date of birth"]}
    scores = score_answer(result, expected)
    assert scores["retrieval_hit"] is True
    assert scores["keyword_score"] == 1.0
    assert scores["citation_hit"] is True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    print("\n=== CI Smoke Tests ===\n")

    check("imports: rag_assistant package", test_imports)
    check("imports: evals module", test_evals_import)
    check("config: types and values", test_config_types)
    check("config: access control map", test_access_map)
    check("chunker: chunk_document synthetic", test_chunk_document_synthetic)
    check("chunker: large section splits", test_chunk_document_large_section)
    check("chunker: table_to_markdown", test_table_to_markdown)
    check("chunker: table_to_markdown empty", test_table_to_markdown_empty)
    check("cache: hash determinism", test_cache_hash_determinism)
    check("cache: file round-trip", lambda: test_cache_file_roundtrip(Path(tempfile.mkdtemp())))
    check("evals: refusal detection", test_refusal_detection)
    check("evals: score_answer OOS pass", test_score_answer_not_in_docs_pass)
    check("evals: score_answer OOS fail", test_score_answer_not_in_docs_fail)
    check("evals: score_answer in-docs", test_score_answer_in_docs)

    print(f"\n{'='*40}")
    if failures:
        print(f"FAILED {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"All {14} checks passed.")

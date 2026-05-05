"""Evaluation harness for the Healthcare RAG pipeline.

Run full evals (requires OpenAI key):
    python evals.py

Run CI smoke tests only (no API keys needed):
    python evals.py --ci

MLflow UI:
    mlflow ui --backend-store-uri mlruns
"""

import json
import logging
import sys
import time
from pathlib import Path

import mlflow

from rag_assistant.generator import answer_question
from rag_assistant.config import (
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    TOP_K,
    SECTION_MAX_CHARS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from rag_assistant.retriever import _reranking_enabled

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Refusal phrases for out-of-scope scoring
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = [
    "does not contain",
    "cannot find",
    "not enough",
    "no information",
    "outside",
    "not mentioned",
    "i don't have",
    "not available in the provided context",
    "cannot",
    "not provided",
    "not available",
    "unable to find",
    "not discussed",
    "no details",
]

# ---------------------------------------------------------------------------
# Eval set — 8 baseline questions + 10 harder questions
# ---------------------------------------------------------------------------

EVAL_SET = [
    # --- Original 8 baseline questions ---
    {
        "question": "What information must a CSR verify before releasing beneficiary information?",
        "expected_page": 12,
        "expected_keywords": ["full name", "date of birth", "HIC number"],
    },
    {
        "question": "What is a business associate under HIPAA?",
        "expected_page": 5,
        "expected_keywords": ["performs", "assists", "function", "activity", "individually identifiable"],
    },
    {
        "question": "How long is a verbal authorization valid?",
        "expected_page": 15,
        "expected_keywords": ["14 days"],
    },
    {
        "question": "Can a representative payee enroll a beneficiary in a Managed Care Organization?",
        "expected_page": 44,
        "expected_keywords": ["not authorized", "state law"],
    },
    {
        "question": "Who is responsible for responding to HIPAA Privacy Rule access requests?",
        "expected_page": 7,
        "expected_keywords": ["CMS Central Office"],
    },
    {
        "question": "What must a valid HIPAA authorization include?",
        "expected_page": 9,
        "expected_keywords": ["description", "signature", "expiration", "purpose"],
    },
    {
        "question": "Can a provider get pre-claim beneficiary information without authorization?",
        "expected_page": 33,
        "expected_keywords": ["no", "not", "without", "authorization"],
    },
    {
        "question": "Where should beneficiaries send HIPAA privacy requests?",
        "expected_page": 9,
        "expected_keywords": ["P.O. Box 8050", "Baltimore"],
    },

    # --- 10 harder questions ---

    # Multi-hop: representative + verbal auth duration
    {
        "question": (
            "If a beneficiary's authorized representative provides verbal authorization, "
            "what is the maximum period it remains valid?"
        ),
        "expected_page": 15,
        "expected_keywords": ["14 days", "representative"],
    },
    # Multi-hop: business associate agreement requirements
    {
        "question": "What must a business associate agreement include under HIPAA?",
        "expected_page": 5,
        "expected_keywords": ["agreement", "safeguard", "comply"],
    },
    # Multi-hop: PHI disclosure without authorization for treatment
    {
        "question": (
            "Under what circumstances can Medicare beneficiary information be disclosed "
            "without authorization for treatment purposes?"
        ),
        "expected_page": 7,
        "expected_keywords": ["treatment", "payment", "operations"],
    },
    # Cross-document: what constitutes PHI
    {
        "question": "What constitutes protected health information (PHI) under HIPAA?",
        "expected_page": 3,
        "expected_keywords": ["individually identifiable", "health information"],
    },
    # Multi-hop: minimum necessary standard
    {
        "question": "What are the minimum necessary standards when sharing PHI with a business associate?",
        "expected_page": 5,
        "expected_keywords": ["minimum necessary", "business associate"],
    },

    # Not-in-docs questions — model should decline to answer
    {
        "question": "What is the monthly Medicare Part B premium for 2024?",
        "not_in_docs": True,
    },
    {
        "question": "How do I file a Medicare claim dispute online?",
        "not_in_docs": True,
    },
    {
        "question": "What are the income eligibility limits for Medicaid?",
        "not_in_docs": True,
    },
    {
        "question": "What is the Medicare Advantage annual enrollment period?",
        "not_in_docs": True,
    },
    {
        "question": "How many inpatient hospital days does Medicare cover before requiring a copayment?",
        "not_in_docs": True,
    },
]


def _is_refusal(answer: str) -> bool:
    """Return True if the answer contains any recognised refusal phrase."""
    lower = answer.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def score_answer(result: dict, expected: dict) -> dict:
    """Score a single RAG answer against expected values."""
    answer_lower = result["answer"].lower()
    source_pages = [s["page"] for s in result["sources"]]

    if expected.get("not_in_docs"):
        # PASS if the model correctly declined to answer
        refused = _is_refusal(result["answer"])
        return {
            "retrieval_hit": True,      # N/A — doesn't drag down recall
            "keyword_score": 1.0 if refused else 0.0,
            "citation_hit": refused,
            "keywords_found": int(refused),
            "keywords_total": 1,
            "not_in_docs": True,
            "refused": refused,
        }

    retrieval_hit = expected["expected_page"] in source_pages
    keywords_found = sum(1 for kw in expected["expected_keywords"] if kw.lower() in answer_lower)
    keyword_score = keywords_found / len(expected["expected_keywords"])
    citation_hit = str(expected["expected_page"]) in result["answer"]

    return {
        "retrieval_hit": retrieval_hit,
        "keyword_score": keyword_score,
        "citation_hit": citation_hit,
        "keywords_found": keywords_found,
        "keywords_total": len(expected["expected_keywords"]),
        "not_in_docs": False,
    }


def run_evals() -> None:
    """Run all evaluation questions, log to MLflow, and print a summary."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    params = {
        "chunk_size": CHUNK_SIZE,
        "section_max_chars": SECTION_MAX_CHARS,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "top_k": TOP_K,
        "reranking_enabled": _reranking_enabled(),
        "eval_questions": len(EVAL_SET),
    }

    with mlflow.start_run():
        mlflow.log_params(params)

        results = []

        for i, test_case in enumerate(EVAL_SET, start=1):
            logger.info(f"Eval {i}/{len(EVAL_SET)}: {test_case['question'][:60]}...")
            t0 = time.time()
            answer = answer_question(test_case["question"], use_cache=False)
            wall_latency = round(time.time() - t0, 3)

            scores = score_answer(answer, test_case)
            scores["question"] = test_case["question"]
            scores["answer_preview"] = answer["answer"][:200]
            scores["latency_s"] = answer.get("latency_s") or wall_latency
            scores["token_count"] = answer.get("token_count")
            results.append(scores)

            mlflow.log_dict(
                {
                    "question": test_case["question"],
                    "answer": answer["answer"],
                    "sources": answer["sources"],
                    "latency_s": scores["latency_s"],
                    "token_count": scores["token_count"],
                    "retrieval_hit": scores["retrieval_hit"],
                    "keyword_score": scores["keyword_score"],
                    "not_in_docs": scores.get("not_in_docs", False),
                },
                f"traces/trace_{i:02d}.json",
            )

        # --- Aggregate metrics ---
        retrieval_hits = sum(1 for r in results if r["retrieval_hit"])
        avg_keyword = sum(r["keyword_score"] for r in results) / len(results)
        citation_hits = sum(1 for r in results if r["citation_hit"])
        avg_latency = sum(r["latency_s"] for r in results if r["latency_s"]) / len(results)
        total_tokens = sum(r["token_count"] for r in results if r["token_count"])

        mlflow.log_metrics({
            "retrieval_recall": retrieval_hits / len(results),
            "avg_keyword_score": avg_keyword,
            "citation_accuracy": citation_hits / len(results),
            "avg_latency_s": round(avg_latency, 3),
            "total_tokens": total_tokens,
        })

        results_path = Path("eval_results.json")
        results_path.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(results_path))

        _print_summary(results)

        print(f"\nDetailed results saved to {results_path}")
        print(f"MLflow run logged to '{MLFLOW_TRACKING_URI}/' (experiment: {MLFLOW_EXPERIMENT_NAME})")
        print("View UI with: mlflow ui --backend-store-uri mlruns")


def _print_summary(results: list[dict]) -> None:
    n = len(results)
    retrieval_hits = sum(1 for r in results if r["retrieval_hit"])
    avg_keyword = sum(r["keyword_score"] for r in results) / n
    citation_hits = sum(1 for r in results if r["citation_hit"])
    avg_latency = sum(r["latency_s"] for r in results if r["latency_s"]) / n
    total_tokens = sum(r["token_count"] for r in results if r["token_count"])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total questions:   {n}")
    print(f"Retrieval recall:  {retrieval_hits}/{n} ({retrieval_hits/n*100:.0f}%)")
    print(f"Avg keyword score: {avg_keyword:.0%}")
    print(f"Citation accuracy: {citation_hits}/{n} ({citation_hits/n*100:.0f}%)")
    print(f"Avg latency:       {avg_latency:.2f}s")
    print(f"Total tokens:      {total_tokens}")
    print("=" * 60)

    for r in results:
        tag = "[OOS]" if r.get("not_in_docs") else "     "
        status = "PASS" if r["retrieval_hit"] and r["keyword_score"] > 0.5 else "FAIL"
        print(f"\n{status} {tag} {r['question'][:70]}")
        print(
            f"  Retrieval: {'HIT' if r['retrieval_hit'] else 'MISS'} | "
            f"Keywords: {r['keywords_found']}/{r['keywords_total']} | "
            f"Citation: {'YES' if r['citation_hit'] else 'NO'} | "
            f"Latency: {r['latency_s']}s"
        )
        print(f"  Answer: {r['answer_preview']}")


def run_ci_checks() -> None:
    """Structural smoke tests that require no API keys.

    Checks:
    - All package imports succeed
    - Config values are the expected types
    - chunk_document() produces chunks from synthetic elements
    - table_to_markdown() produces valid Markdown
    - cache hash is deterministic
    - API_KEYS and access control maps are populated
    """
    import traceback
    failures: list[str] = []

    def check(name: str, fn):
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failures.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()

    print("\n=== CI Smoke Tests ===\n")

    # Imports
    check("import rag_assistant.config", lambda: __import__("rag_assistant.config"))
    check("import rag_assistant.cache",  lambda: __import__("rag_assistant.cache"))
    check("import rag_assistant.chunker", lambda: __import__("rag_assistant.chunker"))

    # Config values
    def _config_types():
        from rag_assistant.config import (
            CHUNK_SIZE, TOP_K, API_KEYS, GROUP_ACCESS_MAP,
            DEFAULT_ACCESS_LEVEL, ANSWER_CACHE_TTL,
        )
        assert isinstance(CHUNK_SIZE, int) and CHUNK_SIZE > 0
        assert isinstance(TOP_K, int) and TOP_K > 0
        assert isinstance(API_KEYS, set) and len(API_KEYS) > 0
        assert isinstance(GROUP_ACCESS_MAP, dict) and "admin" in GROUP_ACCESS_MAP
        assert isinstance(DEFAULT_ACCESS_LEVEL, str)
        assert isinstance(ANSWER_CACHE_TTL, int) and ANSWER_CACHE_TTL > 0

    check("config types and values", _config_types)

    # Access control map completeness
    def _access_map():
        from rag_assistant.config import GROUP_ACCESS_MAP
        for group, levels in GROUP_ACCESS_MAP.items():
            assert isinstance(levels, list) and len(levels) > 0, f"empty levels for {group}"
            assert "public" in levels, f"public missing from {group}"

    check("access control map", _access_map)

    # Section-aware chunker with synthetic elements
    def _chunker():
        from rag_assistant.chunker import chunk_document

        class _Meta:
            page_number = 1

        class _El:
            def __init__(self, category, text):
                self.category = category
                self.text = text
                self.metadata = _Meta()

        elements = [
            _El("Title", "Section One"),
            _El("NarrativeText", "This is the body of section one."),
            _El("NarrativeText", "Another paragraph in section one."),
            _El("Title", "Section Two"),
            _El("NarrativeText", "Body of section two."),
        ]
        chunks = chunk_document(elements, {}, source_name="test.pdf")
        assert len(chunks) >= 2, f"expected >=2 chunks, got {len(chunks)}"
        assert all(c["chunk_type"] == "text" for c in chunks)
        assert all("content" in c and c["content"] for c in chunks)

    check("chunk_document with synthetic elements", _chunker)

    # table_to_markdown
    def _table_md():
        from rag_assistant.chunker import table_to_markdown
        md = table_to_markdown([["Name", "Age"], ["Alice", "30"], ["Bob", "25"]])
        assert "| Name |" in md
        assert "| Alice |" in md
        assert "---" in md

    check("table_to_markdown output", _table_md)

    # Cache hash stability
    def _cache_hash():
        from rag_assistant.cache import _hash_key
        h1 = _hash_key("test question")
        h2 = _hash_key("test question")
        h3 = _hash_key("different question")
        assert h1 == h2, "hash not deterministic"
        assert h1 != h3, "hash collision"

    check("cache hash determinism", _cache_hash)

    # Refusal phrase detection
    def _refusal():
        from evals import _is_refusal
        assert _is_refusal("The context does not contain this information.")
        assert _is_refusal("I cannot find any mention of that.")
        assert not _is_refusal("The answer is 42.")

    check("_is_refusal detection", _refusal)

    print(f"\n{'='*40}")
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    else:
        print(f"All {6 - len(failures) + len(failures)} checks passed.")  # total = 7
        print("All CI checks passed.")


if __name__ == "__main__":
    if "--ci" in sys.argv:
        run_ci_checks()
    else:
        run_evals()

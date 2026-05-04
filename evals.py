import json
import logging
from pathlib import Path

from generator import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Test questions with expected answers
EVAL_SET = [
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
]


def score_answer(result: dict, expected: dict) -> dict:
    """Score a single RAG answer against expected values."""
    answer_lower = result["answer"].lower()
    source_pages = [s["page"] for s in result["sources"]]

    # Did retrieval find the right page?
    retrieval_hit = expected["expected_page"] in source_pages

    # Does the answer contain expected keywords?
    keywords_found = sum(1 for kw in expected["expected_keywords"] if kw.lower() in answer_lower)
    keyword_score = keywords_found / len(expected["expected_keywords"])

    # Did the answer cite the correct page?
    citation_hit = str(expected["expected_page"]) in result["answer"]

    return {
        "retrieval_hit": retrieval_hit,
        "keyword_score": keyword_score,
        "citation_hit": citation_hit,
        "keywords_found": keywords_found,
        "keywords_total": len(expected["expected_keywords"]),
    }


def run_evals():
    """Run all evaluation questions and report scores."""
    results = []

    for i, test_case in enumerate(EVAL_SET, start=1):
        logger.info(f"Eval {i}/{len(EVAL_SET)}: {test_case['question'][:60]}...")
        answer = answer_question(test_case["question"], use_cache=False)
        scores = score_answer(answer, test_case)
        scores["question"] = test_case["question"]
        scores["answer_preview"] = answer["answer"][:200]
        results.append(scores)

    # Summary
    retrieval_hits = sum(1 for r in results if r["retrieval_hit"])
    avg_keyword = sum(r["keyword_score"] for r in results) / len(results)
    citation_hits = sum(1 for r in results if r["citation_hit"])

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total questions: {len(EVAL_SET)}")
    print(f"Retrieval recall: {retrieval_hits}/{len(EVAL_SET)} ({retrieval_hits/len(EVAL_SET)*100:.0f}%)")
    print(f"Avg keyword score: {avg_keyword:.0%}")
    print(f"Citation accuracy: {citation_hits}/{len(EVAL_SET)} ({citation_hits/len(EVAL_SET)*100:.0f}%)")
    print("=" * 60)

    # Per-question breakdown
    for r in results:
        status = "✓" if r["retrieval_hit"] and r["keyword_score"] > 0.5 else "✗"
        print(f"\n{status} {r['question'][:70]}")
        print(f"  Retrieval: {'HIT' if r['retrieval_hit'] else 'MISS'} | "
              f"Keywords: {r['keywords_found']}/{r['keywords_total']} | "
              f"Citation: {'YES' if r['citation_hit'] else 'NO'}")
        print(f"  Answer: {r['answer_preview']}")

    # Save results to file
    Path("eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results saved to eval_results.json")


if __name__ == "__main__":
    run_evals()
import sys
import logging

from rag_assistant.generator import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask a question: ")

    result = answer_question(question)

    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for src in result["sources"]:
        print(f"  - {src['source']}, page {src['page']}")

    if result.get("from_cache"):
        print("\n(served from cache)")
    elif result.get("latency_s") is not None:
        print(f"\n(latency: {result['latency_s']}s | tokens: {result.get('token_count')})")


if __name__ == "__main__":
    main()

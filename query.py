import sys

from generator import answer_question


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


if __name__ == "__main__":
    main()
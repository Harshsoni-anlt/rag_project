import json
import logging
import os
import sys


QUESTIONS = [
    {"question_id": 1, "question": "What was Apple's total revenue for the fiscal year ended September 28, 2024?"},
    {"question_id": 2, "question": "How many shares of common stock were issued and outstanding as of October 18, 2024?"},
    {"question_id": 3, "question": "What is the total amount of term debt (current + non-current) reported by Apple as of September 28, 2024?"},
    {"question_id": 4, "question": "On what date was Apple's 10-K report for 2024 signed and filed with the SEC?"},
    {"question_id": 5, "question": "Does Apple have any unresolved staff comments from the SEC as of this filing? How do you know?"},
    {"question_id": 6, "question": "What was Tesla's total revenue for the year ended December 31, 2023?"},
    {"question_id": 7, "question": "What percentage of Tesla's total revenue in 2023 came from Automotive Sales (excluding Leasing)?"},
    {"question_id": 8, "question": "What is the primary reason Tesla states for being highly dependent on Elon Musk?"},
    {"question_id": 9, "question": "What types of vehicles does Tesla currently produce and deliver?"},
    {"question_id": 10, "question": "What is the purpose of Tesla's 'lease pass-through fund arrangements'?"},
    {"question_id": 11, "question": "What is Tesla's stock price forecast for 2025?"},
    {"question_id": 12, "question": "Who is the CFO of Apple as of 2025?"},
    {"question_id": 13, "question": "What color is Tesla's headquarters painted?"},
]


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(levelname)s: %(message)s",
        )


def check_env() -> bool:
    if os.path.exists(".env"):
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except Exception:
            pass

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token or hf_token == "your_hf_token_here":
        if os.path.exists(".env"):
            print("ERROR: HF_TOKEN not configured in .env")
        else:
            print("ERROR: HF_TOKEN not set")
        print("\nGet free token at: https://huggingface.co/settings/tokens")
        return False
    return True


def check_pdfs(apple_pdf: str, tesla_pdf: str) -> bool:

    if not os.path.exists(apple_pdf):
        print(f"ERROR: Apple PDF not found: {apple_pdf}")
        return False
    if not os.path.exists(tesla_pdf):
        print(f"ERROR: Tesla PDF not found: {tesla_pdf}")
        return False
    return True


def main() -> None:
    _configure_logging()
    print("RAG SYSTEM - QUICK RUN")
    if not check_env():
        sys.exit(1)

    apple_pdf = os.getenv("APPLE_PDF", "data/10-Q4-2024-As-Filed.pdf")
    tesla_pdf = os.getenv("TESLA_PDF", "data/tsla-20231231-gen.pdf")
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "vector_store")
    output_path = os.getenv("OUTPUT_PATH", "outputs/answers.json")

    if not check_pdfs(apple_pdf, tesla_pdf):
        sys.exit(1)

    print("\nConfiguration OK")
    print("PDFs found")
    print("STARTING EVALUATION")

    from src.rag_pipeline import create_pipeline

    pipeline = create_pipeline()

    if os.path.exists(vector_store_path):
        pipeline.load_index(vector_store_path)
    else:
        pipeline.index_documents(
            apple_pdf=apple_pdf,
            tesla_pdf=tesla_pdf,
            save_path=vector_store_path,
        )

    answers = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i}/13] Q{q['question_id']}: {q['question'][:80]}")
        try:
            result = pipeline.answer_question(q["question"])
            answers.append(
                {
                    "question_id": q["question_id"],
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources", []),
                }
            )
        except Exception as e:
            answers.append(
                {
                    "question_id": q["question_id"],
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                }
            )
            print(f"ERROR: {e}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(answers, f, indent=2)

    print("\nEvaluation complete")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()

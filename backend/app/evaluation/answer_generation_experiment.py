
from pathlib import Path
import json

from backend.app.rag import answer_question


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVALUATION_DIR = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "evaluation"
)

DATASET_PATH = EVALUATION_DIR / "evaluation_dataset.json"

OUTPUT_PATH = (
    EVALUATION_DIR
    / "answer_generation_results.json"
)

DOCUMENT_ID = "38eef68c88ea4b2d881d3b3969f27aa4"


def run_experiment():

    print("=" * 65)
    print("RESEARCHIQ - ANSWER GENERATION EXPERIMENT")
    print("=" * 65)

    # Step 1: Load evaluation dataset
    print("\nLoading evaluation dataset...")

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    dataset = [
        item for item in dataset
        if item["document_id"] == DOCUMENT_ID
    ]

    if not dataset:
        raise ValueError("No evaluation questions found.")

    print("Total evaluation questions:", len(dataset))

    # Step 2: Generate answers
    experiment_results = []

    for question_number, test_case in enumerate(dataset, start=1):

        question = test_case["question"]

        print("\n" + "=" * 65)
        print(f"QUESTION {question_number}")
        print("=" * 65)

        print("\nQuestion:", question)

        try:

            result = answer_question(
                question,
                DOCUMENT_ID
            )

            answer = result["answer"]

            print("\nGenerated Answer:")
            print("-" * 50)
            print(answer)

            print("\nRetrieved Sources:")
            print("-" * 50)

            for source in result["sources"]:
                print(
                    f"Page: {source['page']} | "
                    f"Similarity: {source['similarity']}"
                )

            experiment_results.append({
                "question_number": question_number,
                "question": question,
                "answer": answer,
                "sources": result["sources"],
                "context": result["context"],
                "retrieved_chunks": result["retrieved_chunks"],
                "status": "success"
            })

        except Exception as error:

            print("\nError generating answer:", str(error))

            experiment_results.append({
                "question_number": question_number,
                "question": question,
                "answer": None,
                "sources": [],
                "context": "",
                "retrieved_chunks": [],
                "status": "failed",
                "error": str(error)
            })

    # Step 3: Save experiment results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:

        json.dump(
            experiment_results,
            file,
            indent=4,
            ensure_ascii=False
        )

    successful = sum(
        1 for item in experiment_results
        if item["status"] == "success"
    )

    failed = len(experiment_results) - successful

    print("\n" + "=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)

    print("Total questions:", len(experiment_results))
    print("Successful answers:", successful)
    print("Failed answers:", failed)

    print("\nResults saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    run_experiment()

import json
from pathlib import Path
from statistics import mean

from backend.app.paper_search import (
    load_document_index,
    search_paper
)

from backend.app.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank
)


# Locate evaluation dataset
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "evaluation"
    / "evaluation_dataset.json"
)


def evaluate_retrieval(k=5):

    # Load evaluation dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    # Store scores for calculating averages
    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    # Cache document indexes to avoid loading the same PDF repeatedly
    document_cache = {}

    for test_case in dataset:

        question = test_case["question"]
        document_id = test_case["document_id"]
        relevant_ids = set(test_case["relevant_chunk_ids"])

        if not relevant_ids:
            raise ValueError(
                f"No relevant chunk IDs provided for: {question}"
            )

        print("\n" + "=" * 55)
        print("Question:", question)

        # Load document index
        if document_id not in document_cache:
            document_cache[document_id] = load_document_index(
                document_id
            )

        index, chunks = document_cache[document_id]

        # Validate ground-truth chunk IDs
        invalid_ids = [
            chunk_id for chunk_id in relevant_ids
            if not isinstance(chunk_id, int)
            or chunk_id < 0
            or chunk_id >= len(chunks)
        ]

        if invalid_ids:
            raise ValueError(
                f"Invalid relevant chunk IDs: {invalid_ids}"
            )

        # Retrieve top-k chunks
        results = search_paper(
            question,
            index,
            chunks,
            k=k
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        # Display retrieved chunks
        print("\nRetrieved Chunk Details:")

        for result in results:
            print("\nChunk ID:", result["chunk_id"])
            print("Page:", result["page_number"])
            print("Similarity:", round(result["similarity"], 4))
            print("Text:", result["text"][:300])
            print("-" * 50)

        # Calculate evaluation metrics
        precision = precision_at_k(
            retrieved_ids, relevant_ids, k
        )

        recall = recall_at_k(
            retrieved_ids, relevant_ids, k
        )

        rr = reciprocal_rank(
            retrieved_ids, relevant_ids
        )

        # Store scores
        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_ranks.append(rr)

        # Display individual results
        print("\nRetrieved chunk IDs:", retrieved_ids)
        print("Relevant chunk IDs:", sorted(relevant_ids))

        print("\nIndividual Evaluation Results")
        print("-" * 35)
        print(f"Precision@{k}: {precision:.4f}")
        print(f"Recall@{k}:    {recall:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")

    # Calculate average metrics across all questions
    print("\n" + "=" * 55)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 55)

    print("Total Questions Evaluated:", len(dataset))

    print(f"Average Precision@{k}: {mean(precision_scores):.4f}")
    print(f"Average Recall@{k}:    {mean(recall_scores):.4f}")
    print(f"Mean Reciprocal Rank:  {mean(reciprocal_ranks):.4f}")

    print("=" * 55)


if __name__ == "__main__":
    evaluate_retrieval()
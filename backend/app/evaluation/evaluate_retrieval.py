
import json
from pathlib import Path
from statistics import mean

from backend.app.paper_search import (
    load_document_index,
    search_paper
)

from backend.app.embeddings import generate_embeddings

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

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    # Cache document indexes
    document_cache = {}

    for test_case in dataset:

        question = test_case["question"]
        document_id = test_case["document_id"]
        relevant_ids = set(test_case["relevant_chunk_ids"])

        if not relevant_ids:
            raise ValueError(
                f"No relevant chunk IDs provided for: {question}"
            )

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        # Load document index only once
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
        print("\nRETRIEVED CHUNK DETAILS")
        print("-" * 40)

        for result in results:

            print("\nChunk ID:", result["chunk_id"])
            print("Page:", result["page_number"])
            print("Similarity:", round(result["similarity"], 4))
            print("Text:", result["text"][:300])
            print("-" * 50)

        # Display ground-truth chunks
        print("\nGROUND-TRUTH CHUNK DETAILS")
        print("-" * 40)

        for chunk_id in sorted(relevant_ids):

            chunk = chunks[chunk_id]

            print("\nChunk ID:", chunk_id)
            print("Page:", chunk["page_number"])
            print("Text:", chunk["text"][:600])
            print("-" * 50)

        # ==========================================
        # DEBUG: GROUND-TRUTH RANKING ANALYSIS
        # ==========================================

        query_embedding = generate_embeddings([question])

        all_scores, all_indices = index.search(
            query_embedding,
            index.ntotal
        )

        print("\nGROUND-TRUTH RANKING ANALYSIS")
        print("-" * 40)

        for relevant_id in sorted(relevant_ids):

            positions = [
                rank + 1
                for rank, chunk_id in enumerate(all_indices[0])
                if chunk_id == relevant_id
            ]

            if positions:

                rank = positions[0]
                similarity = all_scores[0][rank - 1]

                print(
                    f"Chunk {relevant_id} | "
                    f"Rank: {rank} | "
                    f"Similarity: {similarity:.4f}"
                )

            else:
                print(
                    f"Chunk {relevant_id} was not found."
                )

        # ==========================================
        # EVALUATION METRICS
        # ==========================================

        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            k
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            k
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_ranks.append(rr)

        print("\nINDIVIDUAL EVALUATION RESULTS")
        print("-" * 40)

        print("Retrieved IDs:", retrieved_ids)
        print("Relevant IDs:", sorted(relevant_ids))

        print(f"Precision@{k}: {precision:.4f}")
        print(f"Recall@{k}:    {recall:.4f}")
        print(f"Reciprocal Rank: {rr:.4f}")

    # ==========================================
    # FINAL EVALUATION
    # ==========================================

    print("\n" + "=" * 60)
    print("FINAL RETRIEVAL EVALUATION")
    print("=" * 60)

    print("Total Questions Evaluated:", len(dataset))

    print(
        f"Average Precision@{k}: "
        f"{mean(precision_scores):.4f}"
    )

    print(
        f"Average Recall@{k}: "
        f"{mean(recall_scores):.4f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{mean(reciprocal_ranks):.4f}"
    )

    print("=" * 60)


if __name__ == "__main__":

    k_values = [1,3,5,10]

    for k in k_values:

        print("\n" + "#" * 60)
        print(f"RUNNING EVALUATION FOR K = {k}")
        print("#" * 60)

        evaluate_retrieval(k=k)

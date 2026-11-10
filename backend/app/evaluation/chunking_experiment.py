
from pathlib import Path
import json
import faiss

from backend.app.pdf_processor import extract_text_from_pdf
from backend.app.chunker import chunk_text
from backend.app.embeddings import generate_embeddings


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DOCUMENT_ID = "38eef68c88ea4b2d881d3b3969f27aa4"

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / f"{DOCUMENT_ID}.pdf"
)

EVALUATION_DIR = (
    PROJECT_ROOT
    / "backend"
    / "app"
    / "evaluation"
)

DATASET_PATH = EVALUATION_DIR / "evaluation_dataset.json"
LABELS_PATH = EVALUATION_DIR / "relevance_labels.json"


# Chunking configurations
CONFIGURATIONS = [
    (100, 20),
    (150, 30),
    (200, 40),
    (250, 50)
]

K = 5


def build_temporary_index(pages, chunk_size, overlap):

    all_chunks = []

    for page in pages:

        chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in chunks:

            all_chunks.append({
                "page_number": page["page_number"],
                "text": chunk
            })

    if not all_chunks:
        raise ValueError("No chunks generated.")

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = generate_embeddings(texts)

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, all_chunks


def retrieve_chunks(question, index, chunks, k=5):

    query_embedding = generate_embeddings([question])

    k = min(k, index.ntotal)

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "chunk_id": int(idx),
            "page_number": chunks[idx]["page_number"],
            "similarity": float(score),
            "text": chunks[idx]["text"]
        })

    return results


def precision_at_k(relevance, k):

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    return sum(relevance[:k]) / k


def reciprocal_rank(relevance):

    for rank, is_relevant in enumerate(relevance, start=1):

        if is_relevant:
            return 1 / rank

    return 0.0


def evaluate_question(results, labels, config, question_number):

    relevance = []

    for result in results:

        label_key = (
            f"{config}|{question_number}|"
            f"{result['chunk_id']}"
        )

        if label_key not in labels:
            raise ValueError(
                f"Missing relevance label: {label_key}"
            )

        label = labels[label_key]

        # Support Boolean and y/n relevance labels
        if isinstance(label, bool):

            is_relevant = label

        elif isinstance(label, str) and label.lower() in ("y", "n"):

            is_relevant = label.lower() == "y"

        else:

            raise ValueError(
                f"Invalid relevance label for {label_key}: {label}"
            )

        relevance.append(1 if is_relevant else 0)

    precision = precision_at_k(relevance, K)
    mrr = reciprocal_rank(relevance)

    return precision, mrr, relevance


def run_experiment():

    print("Loading PDF and evaluation data...")

    pages = extract_text_from_pdf(str(PDF_PATH))

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    dataset = [
        item for item in dataset
        if item["document_id"] == DOCUMENT_ID
    ]

    if not dataset:
        raise ValueError("No evaluation questions found.")

    print("PDF loaded successfully!")
    print("Relevance labels loaded successfully!")

    overall_results = []

    for chunk_size, overlap in CONFIGURATIONS:

        config = f"{chunk_size}_{overlap}"

        print("\n" + "=" * 65)
        print(
            f"CHUNK SIZE: {chunk_size} | "
            f"OVERLAP: {overlap}"
        )
        print("=" * 65)

        index, chunks = build_temporary_index(
            pages,
            chunk_size,
            overlap
        )

        print("Total chunks generated:", len(chunks))

        question_precisions = []
        question_mrrs = []

        for question_number, test_case in enumerate(dataset, start=1):

            question = test_case["question"]

            results = retrieve_chunks(
                question,
                index,
                chunks,
                k=K
            )

            precision, mrr, relevance = evaluate_question(
                results,
                labels,
                config,
                question_number
            )

            question_precisions.append(precision)
            question_mrrs.append(mrr)

            print("\nQUESTION:", question)
            print("-" * 50)

            for result, is_relevant in zip(results, relevance):

                status = (
                    "RELEVANT"
                    if is_relevant
                    else "NOT RELEVANT"
                )

                print(
                    f"Chunk {result['chunk_id']} | "
                    f"Page {result['page_number']} | "
                    f"Similarity: {result['similarity']:.4f} | "
                    f"{status}"
                )

            print(f"Precision@{K}: {precision:.4f}")
            print(f"MRR@{K}: {mrr:.4f}")

        avg_precision = (
            sum(question_precisions) / len(question_precisions)
        )

        avg_mrr = (
            sum(question_mrrs) / len(question_mrrs)
        )

        overall_results.append({
            "configuration": config,
            "total_chunks": len(chunks),
            "precision_at_5": avg_precision,
            "mrr_at_5": avg_mrr
        })

    print("\n\n")
    print("=" * 75)
    print("FINAL CHUNKING EXPERIMENT RESULTS")
    print("=" * 75)

    print(
        f"{'Configuration':<18}"
        f"{'Chunks':<12}"
        f"{'Precision@5':<18}"
        f"{'MRR@5':<12}"
    )

    print("-" * 75)

    for result in overall_results:

        print(
            f"{result['configuration']:<18}"
            f"{result['total_chunks']:<12}"
            f"{result['precision_at_5']:<18.4f}"
            f"{result['mrr_at_5']:<12.4f}"
        )

    best_precision = max(
        overall_results,
        key=lambda x: x["precision_at_5"]
    )

    best_mrr = max(
        overall_results,
        key=lambda x: x["mrr_at_5"]
    )

    print("\nBest Precision@5:", best_precision["configuration"])
    print("Best MRR@5:", best_mrr["configuration"])


if __name__ == "__main__":
    run_experiment()

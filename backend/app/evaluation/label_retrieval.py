
import json
from pathlib import Path

import faiss

from backend.app.pdf_processor import extract_text_from_pdf
from backend.app.chunker import chunk_text
from backend.app.embeddings import generate_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "uploads"
    / "38eef68c88ea4b2d881d3b3969f27aa4.pdf"
)

DATASET_PATH = Path(__file__).resolve().parent / "evaluation_dataset.json"
LABELS_PATH = Path(__file__).resolve().parent / "relevance_labels.json"

CONFIGURATIONS = [
    (100, 20),
    (150, 30),
    (200, 40),
    (250, 50),
]

TOP_K = 5


def load_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def build_chunks(chunk_size, overlap):
    pages = extract_text_from_pdf(str(PDF_PATH))
    chunks = []

    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap
        )

        for chunk in page_chunks:
            chunks.append({
                "page_number": page["page_number"],
                "text": chunk
            })

    return chunks


def retrieve_top_chunks(question, chunks, embeddings, index):
    query_embedding = generate_embeddings([question])

    scores, indices = index.search(
        query_embedding,
        min(TOP_K, index.ntotal)
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


def load_existing_labels():
    if LABELS_PATH.exists():
        with open(LABELS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    return {}


def save_labels(labels):
    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        json.dump(labels, file, indent=4, ensure_ascii=False)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    questions = load_questions()
    labels = load_existing_labels()

    for chunk_size, overlap in CONFIGURATIONS:
        config_name = f"{chunk_size}_{overlap}"

        print("\n" + "=" * 70)
        print(f"CHUNKING CONFIGURATION: {config_name}")
        print("=" * 70)

        chunks = build_chunks(chunk_size, overlap)

        texts = [chunk["text"] for chunk in chunks]
        embeddings = generate_embeddings(texts)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        for question_number, item in enumerate(questions, start=1):
            question = item["question"]

            results = retrieve_top_chunks(
                question, chunks, embeddings, index
            )

            print("\n" + "-" * 70)
            print(f"QUESTION {question_number}: {question}")
            print("-" * 70)

            for result in results:
                chunk_id = result["chunk_id"]
                label_key = (
                    f"{config_name}|{question_number}|{chunk_id}"
                )

                print(f"\nChunk ID: {chunk_id}")
                print(f"Page: {result['page_number']}")
                print(f"Similarity: {result['similarity']:.4f}")
                print(f"Text:\n{result['text']}")

                if label_key in labels:
                    print(
                        f"Existing label: "
                        f"{'RELEVANT' if labels[label_key] else 'NOT RELEVANT'}"
                    )
                    continue

                while True:
                    choice = input(
                        "\nIs this chunk relevant? "
                        "(y = yes, n = no, s = skip): "
                    ).strip().lower()

                    if choice == "y":
                        labels[label_key] = True
                        break
                    elif choice == "n":
                        labels[label_key] = False
                        break
                    elif choice == "s":
                        break
                    else:
                        print("Please enter y, n, or s.")

                save_labels(labels)

        print(f"\nCompleted configuration: {config_name}")

    print("\nAll configurations processed.")
    print(f"Labels saved at: {LABELS_PATH}")


if __name__ == "__main__":
    main()

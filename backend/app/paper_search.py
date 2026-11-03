
from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_text


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "papers"
    / "AttentionIsAllYouNeed.pdf"
)

INDEX_DIR = PROJECT_ROOT / "data" / "index"

FAISS_PATH = INDEX_DIR / "paper.index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


# Build FAISS index from PDF
def build_search_index():

    print("Extracting text from PDF...")

    pages = extract_text_from_pdf(str(PDF_PATH))

    all_chunks = []

    for page in pages:

        chunks = chunk_text(page["text"])

        for chunk in chunks:

            all_chunks.append({
                "page_number": page["page_number"],
                "text": chunk
            })

    if not all_chunks:
        raise ValueError("No text chunks found in the PDF.")

    print("Generating embeddings...")

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return index, all_chunks


# Save FAISS index and chunks
def save_search_index(index, all_chunks):

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(FAISS_PATH)
    )

    with open(
        CHUNKS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_chunks,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("FAISS index and chunks saved successfully!")


# Load previously saved index and chunks
def load_search_index():

    index = faiss.read_index(
        str(FAISS_PATH)
    )

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        all_chunks = json.load(file)

    return index, all_chunks


# Load existing index or build a new one
def load_or_build_search_index():

    if FAISS_PATH.exists() and CHUNKS_PATH.exists():

        print("Loading existing FAISS index...")

        return load_search_index()

    print("Building a new FAISS index...")

    index, all_chunks = build_search_index()

    save_search_index(index, all_chunks)

    return index, all_chunks


# Search for relevant chunks
def search_paper(query, index, all_chunks, k=3):

    if index.ntotal == 0:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

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
            "similarity": float(score),
            "page_number": all_chunks[idx]["page_number"],
            "text": all_chunks[idx]["text"]
        })

    return results

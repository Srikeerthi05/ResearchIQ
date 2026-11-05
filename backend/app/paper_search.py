
from pathlib import Path

import faiss

from backend.app.pdf_processor import extract_text_from_pdf
from backend.app.chunker import chunk_text
from backend.app.embeddings import generate_embeddings
from backend.app.vector_store import (
    save_vector_store,
    load_vector_store,
    vector_store_exists
)


def build_search_index(pdf_path, document_id):

    print("Extracting text from PDF...")

    pages = extract_text_from_pdf(str(pdf_path))

    all_chunks = []

    for page in pages:

        chunks = chunk_text(page["text"])

        for chunk in chunks:

            all_chunks.append({
                "document_id": document_id,
                "page_number": page["page_number"],
                "text": chunk
            })

    if not all_chunks:
        raise ValueError(
            "No text chunks found in the PDF."
        )

    print("Generating embeddings...")

    texts = [
        chunk["text"]
        for chunk in all_chunks
    ]

    embeddings = generate_embeddings(texts)

    print("Building FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    save_vector_store(
        document_id,
        index,
        all_chunks
    )

    print("Document index created successfully!")

    return index, all_chunks


def load_document_index(document_id):

    if not vector_store_exists(document_id):
        raise FileNotFoundError(
            f"No index exists for document: {document_id}"
        )

    return load_vector_store(document_id)


def search_paper(query, index, all_chunks, k=3):

    if index.ntotal == 0:
        return []

    query_embedding = generate_embeddings([query])

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
            "document_id": all_chunks[idx]["document_id"],
            "similarity": float(score),
            "page_number": all_chunks[idx]["page_number"],
            "text": all_chunks[idx]["text"],
            "chunk_id": int(idx)
        })

    return results


import json
from pathlib import Path

import faiss


# Project storage directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_ROOT = PROJECT_ROOT / "data" / "index"


def get_document_index_dir(document_id: str) -> Path:

    return INDEX_ROOT / document_id


def save_vector_store(document_id, index, chunks):

    index_dir = get_document_index_dir(document_id)

    index_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    index_path = index_dir / "paper.index"
    chunks_path = index_dir / "chunks.json"

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and chunk metadata have different sizes."
        )

    # Save FAISS index
    faiss.write_index(
        index,
        str(index_path)
    )

    # Save chunk metadata
    with open(
        chunks_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=4
        )


def load_vector_store(document_id):

    index_dir = get_document_index_dir(document_id)

    index_path = index_dir / "paper.index"
    chunks_path = index_dir / "chunks.json"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"Vector store not found for document: {document_id}"
        )

    # Load FAISS index
    index = faiss.read_index(
        str(index_path)
    )

    # Load chunks
    with open(
        chunks_path,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    # Validate index and metadata consistency
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index contains {index.ntotal} vectors, "
            f"but metadata contains {len(chunks)} chunks."
        )

    return index, chunks


def vector_store_exists(document_id):

    index_dir = get_document_index_dir(document_id)

    return (
        (index_dir / "paper.index").exists()
        and (index_dir / "chunks.json").exists()
    )

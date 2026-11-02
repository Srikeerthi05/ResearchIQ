
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_text

# 1. Locate and read the PDF
project_root = Path(__file__).resolve().parents[2]

pdf_path = (
    project_root
    / "data"
    / "papers"
    / "AttentionIsAllYouNeed.pdf"
)

pages = extract_text_from_pdf(str(pdf_path))

# 2. Create chunks and preserve page numbers
all_chunks = []

for page in pages:
    chunks = chunk_text(page["text"])

    for chunk in chunks:
        all_chunks.append({
            "page_number": page["page_number"],
            "text": chunk
        })

# 3. Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

embeddings = np.array(embeddings, dtype="float32")

# 4. Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# 5. Search the research paper
query = "What is the Transformer architecture?"

query_vector = model.encode(
    [query],
    normalize_embeddings=True
).astype("float32")

scores, indices = index.search(query_vector, k=3)

# 6. Display retrieved chunks
print("\nResearchIQ Search Results\n")

for score, idx in zip(scores[0], indices[0]):
    result = all_chunks[idx]

    print("Similarity:", round(float(score), 4))
    print("Page:", result["page_number"])
    print("Text:", result["text"][:500])
    print("-" * 60)

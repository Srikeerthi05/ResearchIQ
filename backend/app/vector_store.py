
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample research text chunks
documents = [
    "Artificial intelligence is transforming healthcare through automation.",
    "Machine learning helps computers learn patterns from data.",
    "Deep learning uses neural networks with multiple layers.",
    "The Transformer architecture uses attention mechanisms.",
    "Pizza is a popular Italian food."
]

# Convert documents into embeddings
embeddings = model.encode(
    documents,
    normalize_embeddings=True
)

# FAISS requires float32 vectors
embeddings = np.array(embeddings, dtype="float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)

# Add document vectors to the index
index.add(embeddings)

# Search with a question
query = "How does AI help in medicine?"

query_vector = model.encode(
    [query],
    normalize_embeddings=True
)

query_vector = np.array(query_vector, dtype="float32")

# Retrieve top 3 matching documents
scores, indices = index.search(query_vector, k=3)

print("\nUser query:", query)
print("\nMost relevant documents:\n")

for score, idx in zip(scores[0], indices[0]):
    print(f"Similarity: {score:.4f}")
    print("Document:", documents[idx])
    print()

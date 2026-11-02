
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Artificial intelligence is transforming the world.",
    "AI is changing how humans work.",
    "I love eating pizza."
]

embeddings = model.encode(sentences, convert_to_tensor=True)

# Compare the meaning of the sentences
similarity = util.cos_sim(embeddings, embeddings)

print("Cosine Similarity Matrix:\n")
print(similarity)

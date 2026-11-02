
from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(texts):
    embeddings = model.encode(texts)
    return embeddings


if __name__ == "__main__":
    sentences = [
        "Artificial intelligence is transforming the world.",
        "AI is changing how humans work.",
        "I love eating pizza."
    ]

    vectors = generate_embeddings(sentences)

    print("Number of sentences:", len(sentences))
    print("Embedding shape:", vectors.shape)
    print("\nFirst sentence embedding:")
    print(vectors[0])

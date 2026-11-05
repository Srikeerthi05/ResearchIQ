
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

# Shared model instance
_model = None


def get_embedding_model():

    global _model

    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def generate_embeddings(texts):

    if not texts:
        return np.empty((0, 384), dtype="float32")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )

"""Embedding utilities built on top of sentence-transformers."""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

def load_embedder(model_name=DEFAULT_MODEL_NAME):
    return SentenceTransformer(model_name)

def embed_texts(embedder, texts, batch_size=32):
    return embedder.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

def embed_query(embedder, query):
    embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return embedding[0]

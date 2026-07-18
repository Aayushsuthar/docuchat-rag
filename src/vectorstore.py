"""FAISS-backed vector store for similarity search over embedded text chunks."""

import json
from pathlib import Path

import faiss
import numpy as np

def build_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.asarray(embeddings, dtype="float32"))
    return index

def save_index(index, chunks, folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(folder_path) / "index.faiss"))
    metadata_path = Path(folder_path) / "chunks.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    return metadata_path

def load_index(folder_path):
    index = faiss.read_index(str(Path(folder_path) / "index.faiss"))
    metadata_path = Path(folder_path) / "chunks.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

def search(index, chunks, query_embedding, top_k=4):
    query_vector = np.asarray([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = dict(chunks[idx])
        entry["score"] = float(score)
        results.append(entry)
    return resultsPage_Down

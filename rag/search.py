"""
RAG Search - Similarity search over embeddings
"""
import os
import json
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

CHUNK_DIR = "rag/data/chunks"
EMBED_DIR = "rag/data/embeds"


def load_model():
    print("[Search] Loading nomic-embed-text-v1.5...")
    return SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="mps"
    )


def load_chunks(lecture_id: str) -> List[str]:
    path = os.path.join(CHUNK_DIR, f"{lecture_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(lecture_id: str) -> np.ndarray:
    path = os.path.join(EMBED_DIR, f"{lecture_id}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings not found: {path}")
    return np.load(path)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between query and chunks"""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(b_norm, a_norm)


def search(lecture_id: str, query: str, top_k: int = 5) -> List[Tuple[float, str]]:
    """Return top_k (score, chunk) pairs"""
    print(f"[Search] {lecture_id}: {query[:50]}...")

    chunks = load_chunks(lecture_id)
    embeddings = load_embeddings(lecture_id)
    model = load_model()

    query_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = cosine_sim(query_emb, embeddings)

    top_k = min(top_k, len(chunks))
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [(float(scores[i]), chunks[i]) for i in top_idx]


if __name__ == "__main__":
    lecture_id = input("Enter lecture ID: ").strip()
    while True:
        query = input("\nQuestion (q to quit): ").strip()
        if query.lower() in {"q", "quit"}:
            break
        results = search(lecture_id, query)
        for rank, (score, text) in enumerate(results, 1):
            print(f"{rank}. [{score:.4f}] {text[:200]}...")

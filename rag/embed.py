"""
RAG Embedding - Generate embeddings using Nomic
"""
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNK_DIR = "rag/data/chunks"
EMBED_DIR = "rag/data/embeds"
os.makedirs(EMBED_DIR, exist_ok=True)


def load_model():
    print("[Embedding] Loading nomic-embed-text-v1.5...")
    return SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device="mps"
    )


def load_chunks(lecture_id):
    path = os.path.join(CHUNK_DIR, f"{lecture_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunks(chunks, model):
    return np.array(model.encode(
        chunks, batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True
    ))


def save_embeddings(embeddings, lecture_id):
    out_path = os.path.join(EMBED_DIR, f"{lecture_id}.npy")
    np.save(out_path, embeddings)
    print(f"[OK] Saved embeddings → {out_path}")
    return out_path


def process(lecture_id):
    print(f"[Embedding] {lecture_id}")
    model = load_model()
    chunks = load_chunks(lecture_id)
    embeddings = embed_chunks(chunks, model)
    save_embeddings(embeddings, lecture_id)


if __name__ == "__main__":
    lecture_id = input("Enter lecture ID: ")
    process(lecture_id)

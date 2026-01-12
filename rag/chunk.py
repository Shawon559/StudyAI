"""
RAG Chunking - Split text into chunks for embedding
"""
import os
import json

RAW_DIR = "rag/data/raw"
CHUNK_DIR = "rag/data/chunks"
os.makedirs(CHUNK_DIR, exist_ok=True)


def load_text(lecture_id):
    path = os.path.join(RAW_DIR, f"{lecture_id}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw lecture not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text, max_words=400):
    """Split text into ~400 word chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def save_chunks(chunks, lecture_id):
    out_path = os.path.join(CHUNK_DIR, f"{lecture_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(chunks)} chunks → {out_path}")
    return out_path


def process(lecture_id):
    print(f"[Chunking] {lecture_id}")
    text = load_text(lecture_id)
    chunks = chunk_text(text)
    save_chunks(chunks, lecture_id)


if __name__ == "__main__":
    lecture_id = input("Enter lecture ID: ")
    process(lecture_id)

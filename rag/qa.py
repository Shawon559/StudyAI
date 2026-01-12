"""
RAG Q&A - Question answering using Phi-3
"""
import os
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

CHUNK_DIR = "rag/data/chunks"
EMBED_DIR = "rag/data/embeds"

# Device setup
if torch.backends.mps.is_available(): DEVICE = "mps"
elif torch.cuda.is_available(): DEVICE = "cuda"
else: DEVICE = "cpu"
print(f"[QA] Device: {DEVICE}")

# Cached models
_embed_model = None
_QA_MODEL = None
_QA_TOKENIZER = None
QA_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[QA] Loading embedding model...")
        _embed_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True, device=DEVICE
        )
    return _embed_model


def embed_text(texts):
    model = get_embed_model()
    return np.array(model.encode(
        texts, batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True
    ), dtype="float32")


def load_chunks(lecture_id):
    path = os.path.join(CHUNK_DIR, f"{lecture_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chunks not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeds(lecture_id):
    path = os.path.join(EMBED_DIR, f"{lecture_id}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings not found: {path}")
    return np.load(path)


def search_top_k(question, lecture_id, k=3):
    chunks = load_chunks(lecture_id)
    embeds = load_embeds(lecture_id)
    q_emb = embed_text([question])[0]
    scores = embeds @ q_emb
    idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in idx], [float(scores[i]) for i in idx]


def get_qa_model():
    global _QA_MODEL, _QA_TOKENIZER
    if _QA_MODEL is None:
        print(f"[QA] Loading {QA_MODEL_NAME}...")
        _QA_TOKENIZER = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
        _QA_MODEL = AutoModelForCausalLM.from_pretrained(
            QA_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE in ("mps", "cuda") else torch.float32,
            device_map=DEVICE
        )
        _QA_MODEL.eval()
        print(f"[QA] Model loaded on {DEVICE}")
    return _QA_MODEL, _QA_TOKENIZER


SYSTEM_PROMPT = (
    "You are a helpful study assistant. Use ONLY the provided context to answer. "
    "If not in context, reply: \"I don't know\". Answer in 2-4 clear sentences."
)


def build_context_block(chunks):
    return "\n\n".join([f"[{i+1}] {ch.strip()}" for i, ch in enumerate(chunks)])


def answer_question(question, lecture_id, k=3):
    chunks, scores = search_top_k(question, lecture_id, k=k)
    context = build_context_block(chunks)

    model, tokenizer = get_qa_model()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer directly:"}
    ]

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=256, do_sample=False)

    gen_ids = output_ids[0][inputs.shape[-1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    for prefix in ["Answer:", "answer:", "A:", "a:"]:
        if raw_text.lower().startswith(prefix.lower()):
            raw_text = raw_text[len(prefix):].strip()

    return raw_text, list(zip(chunks, scores))


if __name__ == "__main__":
    lecture_id = input("Lecture ID: ").strip()
    while True:
        q = input("Question (q to quit): ").strip()
        if q.lower() in ("q", "quit"):
            break
        ans, ctx = answer_question(q, lecture_id)
        print(f"\nAnswer: {ans}\n")

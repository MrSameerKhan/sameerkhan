"""
ingest.py — Load documents → chunk → embed → save FAISS index

Run:  python ingest.py --docs sample_data/
"""

import argparse
import json
import os

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, EMBED_MODEL, INDEX_PATH


# ── Document loading ──────────────────────────────────────────────────────────

def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docs(folder: str) -> list[dict]:
    """Load all .txt and .pdf files from a folder."""
    docs = []
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        if fname.endswith(".txt"):
            text = load_txt(fpath)
        elif fname.endswith(".pdf"):
            text = load_pdf(fpath)
        else:
            continue
        docs.append({"source": fname, "text": text})
        print(f"  Loaded: {fname} ({len(text):,} chars)")
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """Split text into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"source": source, "text": chunk, "start": start})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # unit vectors → dot product = cosine
        batch_size=64,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


# ── FAISS index ───────────────────────────────────────────────────────────────

def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]                 # 384
    index = faiss.IndexFlatIP(dim)            # inner product (= cosine for unit vecs)
    index.add(embeddings)
    return index


# ── Main ──────────────────────────────────────────────────────────────────────

def main(docs_folder: str) -> None:
    print(f"\n=== INGEST: {docs_folder} ===\n")

    print("[1/4] Loading documents...")
    docs = load_docs(docs_folder)
    if not docs:
        raise FileNotFoundError(f"No .txt or .pdf files found in {docs_folder}")

    print(f"\n[2/4] Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_text(doc["text"], doc["source"])
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")
    print(f"  Total chunks: {len(all_chunks)}")

    print(f"\n[3/4] Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(all_chunks, model)
    print(f"  Embedding shape: {embeddings.shape}")

    print("\n[4/4] Building FAISS index and saving...")
    index = build_index(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n  Saved: {INDEX_PATH}  ({index.ntotal} vectors, dim={embeddings.shape[1]})")
    print(f"  Saved: {CHUNKS_PATH}")
    print("\nIngest complete. Run pipeline.py to query.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="sample_data", help="Folder with .txt/.pdf files")
    args = parser.parse_args()
    main(args.docs)

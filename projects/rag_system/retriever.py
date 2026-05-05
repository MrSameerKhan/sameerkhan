"""
retriever.py — Load FAISS index → embed query → return top-K chunks

Used by pipeline.py. Can also run standalone to inspect retrieved chunks.

Run:  python retriever.py "what is attention mechanism?"
"""

import argparse
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNKS_PATH, EMBED_MODEL, INDEX_PATH, TOP_K


class Retriever:
    def __init__(self) -> None:
        print(f"Loading index from {INDEX_PATH}...")
        self.index  = faiss.read_index(INDEX_PATH)
        self.model  = SentenceTransformer(EMBED_MODEL)
        with open(CHUNKS_PATH) as f:
            self.chunks = json.load(f)
        print(f"  {self.index.ntotal} vectors, {len(self.chunks)} chunks loaded.")

    def retrieve(self, query: str, k: int = TOP_K) -> list[dict]:
        """Embed query → search FAISS → return top-k chunks with scores."""
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:          # FAISS returns -1 when fewer than k results exist
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)

        return results


# ── Standalone inspection ──────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Query to search for")
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    retriever = Retriever()
    results = retriever.retrieve(args.query, k=args.k)

    print(f"\nQuery: {args.query!r}")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f}  source={r['source']}")
        print(f"     {r['text'][:200].strip()}...")
        print()

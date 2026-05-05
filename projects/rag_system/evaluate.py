"""
evaluate.py — Evaluate retrieval quality on a small Q&A test set

Metrics:
  - Precision@K  : fraction of retrieved chunks that are relevant
  - Recall@K     : fraction of relevant chunks that are retrieved
  - MRR          : mean reciprocal rank of first relevant chunk

Run:  python evaluate.py
"""

import json

import numpy as np

from retriever import Retriever

# Ground-truth Q&A pairs — edit these to match your sample_data content
# "relevant_sources": which source file(s) should appear in top-K results
TEST_SET = [
    {
        "query": "What is the attention mechanism in transformers?",
        "relevant_sources": ["ml_notes.txt"],
    },
    {
        "query": "How does backpropagation work?",
        "relevant_sources": ["ml_notes.txt"],
    },
    {
        "query": "What is gradient descent?",
        "relevant_sources": ["ml_notes.txt"],
    },
    {
        "query": "Explain LSTM and how it solves the vanishing gradient problem",
        "relevant_sources": ["ml_notes.txt"],
    },
    {
        "query": "What is a convolutional neural network?",
        "relevant_sources": ["ml_notes.txt"],
    },
]


def precision_at_k(retrieved: list[dict], relevant_sources: list[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if c["source"] in relevant_sources)
    return hits / k


def recall_at_k(retrieved: list[dict], relevant_sources: list[str], k: int) -> float:
    top_k = retrieved[:k]
    hits = sum(1 for c in top_k if c["source"] in relevant_sources)
    return hits / len(relevant_sources) if relevant_sources else 0.0


def reciprocal_rank(retrieved: list[dict], relevant_sources: list[str]) -> float:
    for rank, chunk in enumerate(retrieved, 1):
        if chunk["source"] in relevant_sources:
            return 1.0 / rank
    return 0.0


def evaluate(retriever: Retriever, k: int = 5) -> None:
    print(f"\n=== EVALUATION (k={k}) ===\n")

    p_at_k_scores, r_at_k_scores, rr_scores = [], [], []

    for item in TEST_SET:
        query    = item["query"]
        relevant = item["relevant_sources"]
        results  = retriever.retrieve(query, k=k)

        p = precision_at_k(results, relevant, k)
        r = recall_at_k(results, relevant, k)
        rr = reciprocal_rank(results, relevant)

        p_at_k_scores.append(p)
        r_at_k_scores.append(r)
        rr_scores.append(rr)

        status = "✓" if rr > 0 else "✗"
        print(f"{status} {query[:60]!r}")
        print(f"   P@{k}={p:.2f}  R@{k}={r:.2f}  RR={rr:.2f}")
        if results:
            top = results[0]
            print(f"   Top-1: {top['source']} (score={top['score']:.4f})")
        print()

    print("─" * 50)
    print(f"Mean P@{k} : {np.mean(p_at_k_scores):.4f}")
    print(f"Mean R@{k} : {np.mean(r_at_k_scores):.4f}")
    print(f"MRR       : {np.mean(rr_scores):.4f}")
    print()


if __name__ == "__main__":
    retriever = Retriever()
    evaluate(retriever, k=5)

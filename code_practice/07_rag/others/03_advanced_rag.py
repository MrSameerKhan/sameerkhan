# Session 3 — Advanced RAG: Hybrid Search + Reranking + HyDE
# Task   : high-precision financial QA — basic RAG retrieves wrong chunks, this fixes it
# Shows  : BM25 + dense retrieval, RRF fusion, cross-encoder reranking, HyDE
# Compare: basic dense-only vs advanced hybrid+reranked on the same queries
#
# Set OPENAI_API_KEY before running.

import os
import math
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from _corpus import DOCUMENTS

oai    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embed  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

GEN_MODEL  = "gpt-4o-mini"
CHUNK_SIZE = 300
OVERLAP    = 60
TOP_K_RETRIEVE = 20   # wide first-stage net
TOP_K_RERANK   = 4    # narrow after reranking


# ── Chunking ──────────────────────────────────────────────────────────────────
def build_chunks(docs: list[dict], size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    chunks = []
    for doc in docs:
        text, start = doc["content"], 0
        while start < len(text):
            end   = min(start + size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"doc_id": doc["id"], "title": doc["title"], "content": chunk})
            start += size - overlap
    return chunks


# ── Dense index ───────────────────────────────────────────────────────────────
def build_dense_index(chunks: list[dict]):
    embs  = embed.encode([c["content"] for c in chunks], normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs


# ── BM25 index ────────────────────────────────────────────────────────────────
def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    tokenised = [c["content"].lower().split() for c in chunks]
    return BM25Okapi(tokenised)


# ── Retrieval methods ─────────────────────────────────────────────────────────
def dense_retrieve(query: str, chunks, index, k: int) -> list[tuple[int, float]]:
    """Returns (chunk_idx, score) pairs."""
    q    = embed.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, k)
    return [(int(ids[0][i]), float(scores[0][i])) for i in range(len(ids[0]))]


def bm25_retrieve(query: str, bm25: BM25Okapi, k: int) -> list[tuple[int, float]]:
    scores  = bm25.get_scores(query.lower().split())
    top_ids = scores.argsort()[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_ids]


def rrf_fuse(rankings: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion — combines ranked lists without caring about score scales.
    rrf_score(doc) = Σ 1 / (k + rank_in_list)
    """
    fused: dict[int, float] = {}
    for ranked_list in rankings:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def cross_encode_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Score (query, chunk) pairs jointly — much more accurate than bi-encoder."""
    pairs  = [(query, c["content"]) for c in candidates]
    scores = rerank.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]


# ── HyDE — Hypothetical Document Embeddings ───────────────────────────────────
def hyde_query(query: str) -> str:
    """Generate a hypothetical answer, then search using the answer's embedding."""
    response = oai.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.1,
        messages=[{
            "role": "user",
            "content": (
                f"Write a short 2-3 sentence passage from a bank policy document that "
                f"would answer this question: '{query}'. Write only the passage, no preamble."
            ),
        }],
    )
    return response.choices[0].message.content.strip()


# ── Full pipelines ────────────────────────────────────────────────────────────
def basic_rag(query: str, chunks, dense_index) -> list[dict]:
    """Session 01 approach: dense only, no reranking."""
    hits = dense_retrieve(query, chunks, dense_index, k=TOP_K_RERANK)
    return [chunks[i] for i, _ in hits]


def advanced_rag(query: str, chunks, dense_index, bm25_index,
                 use_hyde: bool = False) -> list[dict]:
    """Hybrid retrieval (BM25 + dense + RRF) → cross-encoder reranking."""
    if use_hyde:
        hyp = hyde_query(query)
        dense_hits = dense_retrieve(hyp, chunks, dense_index, k=TOP_K_RETRIEVE)
    else:
        dense_hits = dense_retrieve(query, chunks, dense_index, k=TOP_K_RETRIEVE)

    bm25_hits   = bm25_retrieve(query, bm25_index, k=TOP_K_RETRIEVE)
    fused       = rrf_fuse([dense_hits, bm25_hits])

    candidates  = [chunks[i] for i, _ in fused[:TOP_K_RETRIEVE]]
    return cross_encode_rerank(query, candidates, top_k=TOP_K_RERANK)


def generate(query: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(f"[{r['title']}]\n{r['content']}" for r in retrieved)
    res = oai.chat.completions.create(
        model=GEN_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Answer the question using ONLY the provided context. "
                "Be specific — include exact numbers and conditions."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return res.choices[0].message.content.strip()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Building indexes...")
    chunks      = build_chunks(DOCUMENTS)
    dense_index, _ = build_dense_index(chunks)
    bm25_index  = build_bm25_index(chunks)
    print(f"  {len(chunks)} chunks indexed (dense + BM25)\n")

    queries = [
        "What is the ERC in year 3 of a fixed-rate mortgage?",
        "What is the debt burden ratio limit for personal finance?",
        "Can expatriates get a credit card and what documents do they need?",
    ]

    for query in queries:
        print(f"{'═' * 68}")
        print(f"Query: {query}\n")

        basic   = basic_rag(query, chunks, dense_index)
        basic_a = generate(query, basic)
        print(f"[Basic RAG]")
        print(f"  Sources: {list(dict.fromkeys(r['title'] for r in basic))}")
        print(f"  Answer:  {basic_a}\n")

        adv     = advanced_rag(query, chunks, dense_index, bm25_index, use_hyde=False)
        adv_a   = generate(query, adv)
        print(f"[Advanced RAG — hybrid + reranked]")
        print(f"  Sources: {list(dict.fromkeys(r['title'] for r in adv))}")
        print(f"  Answer:  {adv_a}\n")

        hyde    = advanced_rag(query, chunks, dense_index, bm25_index, use_hyde=True)
        hyde_a  = generate(query, hyde)
        print(f"[Advanced RAG + HyDE]")
        print(f"  Sources: {list(dict.fromkeys(r['title'] for r in hyde))}")
        print(f"  Answer:  {hyde_a}\n")


if __name__ == "__main__":
    main()

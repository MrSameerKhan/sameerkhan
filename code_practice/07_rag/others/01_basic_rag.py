# Session 1 — Basic RAG Pipeline
# Task   : answer questions about bank policy using retrieved document chunks
# Model  : all-MiniLM-L6-v2 (retrieval) + gpt-4o-mini (generation)
# Shows  : full RAG loop — chunk → embed → index → retrieve → generate
#
# Set OPENAI_API_KEY before running.

import os
import textwrap
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from _corpus import DOCUMENTS

oai    = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embed  = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

GEN_MODEL  = "gpt-4o-mini"
CHUNK_SIZE = 200     # characters (approx 50 tokens — small enough to test retrieval)
OVERLAP    = 40
TOP_K      = 3


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_document(doc: dict, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """Split document content into overlapping character-level windows."""
    text   = doc["content"]
    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "doc_id":  doc["id"],
                "title":   doc["title"],
                "content": chunk,
                "start":   start,
            })
        start += chunk_size - overlap
    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


# ── Indexing ──────────────────────────────────────────────────────────────────
def build_index(chunks: list[dict]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    """Encode chunks and build a FAISS inner-product (cosine after normalisation) index."""
    texts = [c["content"] for c in chunks]
    embs  = embed.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embs  = embs.astype(np.float32)

    dim   = embs.shape[1]
    index = faiss.IndexFlatIP(dim)     # inner product on L2-normalised vectors = cosine similarity
    index.add(embs)
    return index, embs


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, chunks: list[dict], index: faiss.IndexFlatIP, top_k: int = TOP_K) -> list[dict]:
    q_emb = embed.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, top_k)
    return [
        {**chunks[i], "score": float(scores[0][rank])}
        for rank, i in enumerate(indices[0])
    ]


# ── Generation ────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a bank policy assistant. Answer the user's question using ONLY the "
    "provided context. If the answer is not in the context, say 'I don't have that "
    "information in the policy documents.' Do not guess or use external knowledge."
)

def generate(query: str, retrieved: list[dict]) -> str:
    context = "\n\n".join(
        f"[{r['title']}]\n{r['content']}" for r in retrieved
    )
    response = oai.chat.completions.create(
        model=GEN_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return response.choices[0].message.content.strip()


def rag(query: str, chunks: list[dict], index: faiss.IndexFlatIP) -> dict:
    retrieved = retrieve(query, chunks, index)
    answer    = generate(query, retrieved)
    return {"query": query, "answer": answer, "sources": retrieved}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Building index...")
    chunks = build_chunks(DOCUMENTS)
    index, _ = build_index(chunks)
    print(f"  {len(DOCUMENTS)} documents → {len(chunks)} chunks indexed\n")

    queries = [
        "What is the maximum LTV for first-time buyers?",
        "What documents do I need to apply for a mortgage?",
        "What is the early repayment charge in year 3?",
        "Can an expatriate get personal finance and what is the maximum amount?",
        "What happens if my savings account balance falls below SAR 500?",
    ]

    for query in queries:
        result = rag(query, chunks, index)
        print(f"Q: {query}")
        print(f"A: {result['answer']}")
        print(f"   Sources: {[r['title'] for r in result['sources']]}")
        print()


if __name__ == "__main__":
    main()

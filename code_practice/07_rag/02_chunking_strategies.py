# Session 2 — Chunking Strategies
# Task   : compare 4 chunking strategies and measure their impact on retrieval quality
# Shows  : fixed-size vs sentence-based vs hierarchical vs semantic chunking
# No LLM generation — focuses purely on the retrieval stage
#
# Key insight: bad chunking is the #1 RAG failure mode — the LLM can't answer
# if the right information isn't in the retrieved chunks.

import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from _corpus import DOCUMENTS

embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ── Strategy 1: Fixed-size (character windows) ────────────────────────────────
def fixed_size_chunks(docs: list[dict], size: int = 300, overlap: int = 50) -> list[dict]:
    chunks = []
    for doc in docs:
        text, start = doc["content"], 0
        while start < len(text):
            end   = min(start + size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"strategy": f"fixed-{size}", "doc_id": doc["id"], "content": chunk})
            start += size - overlap
    return chunks


# ── Strategy 2: Sentence-based ────────────────────────────────────────────────
def sentence_chunks(docs: list[dict], sentences_per_chunk: int = 2) -> list[dict]:
    """Group N sentences per chunk — preserves sentence integrity."""
    chunks = []
    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc["content"].strip())
        for i in range(0, len(sentences), sentences_per_chunk):
            group = " ".join(sentences[i : i + sentences_per_chunk]).strip()
            if group:
                chunks.append({"strategy": "sentence-2", "doc_id": doc["id"], "content": group})
    return chunks


# ── Strategy 3: Hierarchical (parent-child) ───────────────────────────────────
def hierarchical_chunks(docs: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Child chunks (small — retrieved for precision)
    Parent chunks (large — returned to LLM for context)
    At retrieval time: find relevant child → return its parent.
    """
    children, parents = [], []
    for doc in docs:
        # Parent = full document
        parents.append({"strategy": "hierarchical-parent", "doc_id": doc["id"], "content": doc["content"]})

        # Children = sentence-level (retrieve these, but send parent to LLM)
        sentences = re.split(r'(?<=[.!?])\s+', doc["content"].strip())
        for sent in sentences:
            if len(sent) > 20:   # skip very short fragments
                children.append({
                    "strategy": "hierarchical-child",
                    "doc_id":   doc["id"],
                    "parent":   doc["content"],  # pointer back to parent
                    "content":  sent.strip(),
                })
    return children, parents


# ── Strategy 4: Semantic (cosine-boundary) ────────────────────────────────────
def semantic_chunks(docs: list[dict], threshold: float = 0.75) -> list[dict]:
    """
    Split at sentences where cosine similarity to the next sentence drops below threshold.
    Detects genuine topic shifts rather than arbitrary character boundaries.
    """
    chunks = []
    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc["content"].strip())
        if len(sentences) < 2:
            chunks.append({"strategy": "semantic", "doc_id": doc["id"], "content": doc["content"]})
            continue

        embs = embed.encode(sentences, normalize_embeddings=True)
        sims = [float(embs[i] @ embs[i+1]) for i in range(len(embs)-1)]

        group, groups = [sentences[0]], []
        for i, sim in enumerate(sims):
            if sim < threshold:
                groups.append(" ".join(group))
                group = [sentences[i+1]]
            else:
                group.append(sentences[i+1])
        groups.append(" ".join(group))

        for g in groups:
            if g.strip():
                chunks.append({"strategy": "semantic", "doc_id": doc["id"], "content": g.strip()})
    return chunks


# ── Retrieval helper ──────────────────────────────────────────────────────────
def build_index(chunks: list[dict]):
    embs  = embed.encode([c["content"] for c in chunks], normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs


def top_k(query: str, chunks: list[dict], index, k: int = 3) -> list[dict]:
    q = embed.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, ids = index.search(q, k)
    return [{"content": chunks[i]["content"], "score": float(scores[0][r])} for r, i in enumerate(ids[0])]


# ── Comparison ────────────────────────────────────────────────────────────────
def compare(queries: list[str], strategies: dict[str, list[dict]]) -> None:
    indices = {}
    for name, chunks in strategies.items():
        idx, _ = build_index(chunks)
        indices[name] = (chunks, idx)

    for query in queries:
        print(f"\n{'─' * 68}")
        print(f"Query: {query}")
        for name, (chunks, idx) in indices.items():
            results = top_k(query, chunks, idx, k=2)
            print(f"\n  [{name}]")
            for r in results:
                preview = r["content"][:120].replace("\n", " ")
                print(f"    ({r['score']:.3f}) {preview}...")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Build all chunk sets
    fixed_small  = fixed_size_chunks(DOCUMENTS, size=150, overlap=30)
    fixed_medium = fixed_size_chunks(DOCUMENTS, size=300, overlap=50)
    sentence_2   = sentence_chunks(DOCUMENTS, sentences_per_chunk=2)
    children, _  = hierarchical_chunks(DOCUMENTS)
    semantic     = semantic_chunks(DOCUMENTS, threshold=0.72)

    strategies = {
        "fixed-150":      fixed_small,
        "fixed-300":      fixed_medium,
        "sentence-2":     sentence_2,
        "hierarchical":   children,
        "semantic":       semantic,
    }

    print("Chunk counts per strategy:")
    for name, chunks in strategies.items():
        avg_len = sum(len(c["content"]) for c in chunks) / len(chunks)
        print(f"  {name:20s}: {len(chunks):3d} chunks, avg {avg_len:.0f} chars")

    queries = [
        "What is the maximum LTV for first-time buyers?",        # specific fact — small chunks win
        "What documents and affordability rules apply to mortgages?",  # multi-fact — larger chunks win
        "What is the early repayment charge schedule?",          # enumerated list — medium wins
    ]
    compare(queries, strategies)

    # Hierarchical demonstration
    print(f"\n\n{'═' * 68}")
    print("Hierarchical pattern: retrieve child → return parent context")
    _, parent_idx = build_index([{"content": d["content"], "doc_id": d["id"]} for d in DOCUMENTS])
    child_chunks, _ = hierarchical_chunks(DOCUMENTS)
    idx_c, _ = build_index(child_chunks)

    q     = "What is the ERC in year 3?"
    child = top_k(q, child_chunks, idx_c, k=1)[0]
    print(f"\n  Child retrieved: {child['content']}")
    # In real hierarchical RAG, you'd look up the parent doc from the child's doc_id
    parent_doc = next(d for d in DOCUMENTS if d["id"] == child_chunks[0]["doc_id"])
    print(f"\n  Parent sent to LLM (first 300 chars):")
    print(f"  {parent_doc['content'][:300]}...")


if __name__ == "__main__":
    main()

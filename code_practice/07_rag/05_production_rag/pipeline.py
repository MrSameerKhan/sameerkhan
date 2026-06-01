# Session 5 — Production RAG: pipeline.py
# Adds semantic cache + structured tracing to the basic RAG pipeline.
# Imported by serve.py — can also be used standalone.
#
# Set OPENAI_API_KEY before running.

import os
import time
import json
import hashlib
from dataclasses import dataclass, field, asdict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from _corpus import DOCUMENTS

oai   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 300
OVERLAP    = 60
TOP_K      = 4
GEN_MODEL  = "gpt-4o-mini"


# ── Trace ─────────────────────────────────────────────────────────────────────
@dataclass
class RAGTrace:
    query:               str
    answer:              str
    retrieved_titles:    list[str]
    retrieval_ms:        float = 0.0
    generation_ms:       float = 0.0
    total_ms:            float = 0.0
    cache_hit:           bool  = False
    cache_tier:          str   = "miss"    # "exact", "semantic", "miss"
    top_k:               int   = TOP_K
    model:               str   = GEN_MODEL
    input_tokens:        int   = 0
    output_tokens:       int   = 0

    def log(self):
        print(json.dumps(asdict(self), indent=None))


# ── Chunking + indexing ───────────────────────────────────────────────────────
def _build_chunks(docs):
    chunks = []
    for doc in docs:
        text, start = doc["content"], 0
        while start < len(text):
            end  = min(start + CHUNK_SIZE, len(text))
            c    = text[start:end].strip()
            if c:
                chunks.append({"doc_id": doc["id"], "title": doc["title"], "content": c})
            start += CHUNK_SIZE - OVERLAP
    return chunks


# ── Two-tier semantic cache ───────────────────────────────────────────────────
class SemanticCache:
    """
    Tier 1: exact hash match  (O(1), ~5-15% hit rate)
    Tier 2: cosine similarity  (O(N), ~20-40% hit rate with good threshold)
    """

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold  = similarity_threshold
        self._exact:    dict[str, str]   = {}   # hash(query) → answer
        self._sem_keys: list[str]        = []   # stored query strings
        self._sem_embs: list[np.ndarray] = []   # their embeddings
        self._sem_vals: list[str]        = []   # corresponding answers
        self.hits_exact = 0
        self.hits_sem   = 0
        self.misses     = 0

    def get(self, query: str) -> tuple[str | None, str]:
        # Tier 1
        key = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        if key in self._exact:
            self.hits_exact += 1
            return self._exact[key], "exact"

        # Tier 2
        if self._sem_embs:
            q_emb  = embed.encode([query], normalize_embeddings=True)[0]
            matrix = np.vstack(self._sem_embs)
            sims   = matrix @ q_emb
            best   = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                self.hits_sem += 1
                return self._sem_vals[best], "semantic"

        self.misses += 1
        return None, "miss"

    def set(self, query: str, answer: str) -> None:
        key = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        self._exact[key] = answer
        q_emb = embed.encode([query], normalize_embeddings=True)[0]
        self._sem_keys.append(query)
        self._sem_embs.append(q_emb)
        self._sem_vals.append(answer)

    @property
    def stats(self) -> dict:
        total = self.hits_exact + self.hits_sem + self.misses
        return {
            "total_requests": total,
            "exact_hits":     self.hits_exact,
            "semantic_hits":  self.hits_sem,
            "misses":         self.misses,
            "hit_rate":       round((self.hits_exact + self.hits_sem) / total, 3) if total else 0.0,
            "cache_size":     len(self._sem_keys),
        }


# ── RAG pipeline ─────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self, docs=DOCUMENTS, cache_threshold: float = 0.92):
        self.chunks = _build_chunks(docs)
        embs        = embed.encode(
            [c["content"] for c in self.chunks],
            normalize_embeddings=True,
        ).astype(np.float32)
        self.index  = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.cache  = SemanticCache(similarity_threshold=cache_threshold)

    def _retrieve(self, query: str) -> list[dict]:
        q = embed.encode([query], normalize_embeddings=True).astype(np.float32)
        _, ids = self.index.search(q, TOP_K)
        return [self.chunks[i] for i in ids[0]]

    def _generate(self, query: str, retrieved: list[dict]) -> tuple[str, int, int]:
        context = "\n\n".join(f"[{r['title']}]\n{r['content']}" for r in retrieved)
        res = oai.chat.completions.create(
            model=GEN_MODEL, temperature=0,
            messages=[
                {"role": "system", "content": "Answer using ONLY the context. Be specific."},
                {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        return (
            res.choices[0].message.content.strip(),
            res.usage.prompt_tokens,
            res.usage.completion_tokens,
        )

    def query(self, question: str) -> RAGTrace:
        t_start = time.perf_counter()

        # Cache lookup
        cached_answer, tier = self.cache.get(question)
        if cached_answer:
            return RAGTrace(
                query=question, answer=cached_answer,
                retrieved_titles=[], cache_hit=True, cache_tier=tier,
                total_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        # Retrieval
        t_ret     = time.perf_counter()
        retrieved = self._retrieve(question)
        ret_ms    = round((time.perf_counter() - t_ret) * 1000, 1)

        # Generation
        t_gen  = time.perf_counter()
        answer, in_tok, out_tok = self._generate(question, retrieved)
        gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)

        self.cache.set(question, answer)

        return RAGTrace(
            query=question,
            answer=answer,
            retrieved_titles=[r["title"] for r in retrieved],
            retrieval_ms=ret_ms,
            generation_ms=gen_ms,
            total_ms=round((time.perf_counter() - t_start) * 1000, 1),
            cache_hit=False,
            cache_tier="miss",
            input_tokens=in_tok,
            output_tokens=out_tok,
        )


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising production RAG pipeline...")
    pipeline = RAGPipeline()
    print(f"  {len(pipeline.chunks)} chunks indexed\n")

    queries = [
        "What is the maximum LTV for first-time buyers?",
        "What is the early repayment charge in year 3?",
        "What is max LTV for a first-time home buyer?",   # paraphrase — should hit semantic cache
        "Can expatriates apply for personal finance?",
        "What documents are needed for a mortgage?",
        "Maximum LTV for first-time buyer?",              # another paraphrase
    ]

    for q in queries:
        trace = pipeline.query(q)
        cache_label = f"CACHE {trace.cache_tier.upper()}" if trace.cache_hit else "API CALL"
        print(f"[{cache_label:>14}] ({trace.total_ms:>5.0f}ms)  {q[:55]}")
        print(f"                   → {trace.answer[:100]}...")
        print()

    print("\nCache stats:", pipeline.cache.stats)

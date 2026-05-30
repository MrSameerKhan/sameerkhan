# Production RAG Operations

> Deep ops coverage (drift detection, freshness monitoring, cost telemetry) → `../10.mlops/13_production_rag_ops.md`. This file owns the architectural decisions for taking RAG from working demo to production system: semantic cache, cost model, A/B testing RAG versions, and connecting eval to deployment.

---

## Quick Reference

| Problem | Solution | Effort |
|---------|----------|--------|
| Repeated queries cost API money | Semantic cache | Medium |
| Index goes stale as docs change | Incremental re-indexing | Medium |
| Can't compare RAG v1 vs v2 | Shadow deployment + RAGAS A/B | High |
| No visibility into failures | Structured logging + tracing | Low-Medium |
| Cost is unpredictable | Cost attribution per request | Medium |

---

## 1. The Gap Between Demo and Production

A working RAG notebook becomes a production problem when:
- Users repeat similar questions → repeated embedding + LLM calls → unnecessary cost
- The document corpus changes → stale index → wrong answers
- You change the model or chunking → no way to measure if it improved
- Latency spikes → no idea where in the pipeline

---

## 2. Semantic Cache

Cache answers to semantically similar queries — not just exact duplicates.

### Two-Tier Cache Architecture

```
Incoming query
    │
    ┌─── Tier 1: Exact match cache (Redis/dict) ──────────────────┐
    │    Hash of query string → cached answer                      │
    │    Hit rate: ~5-15% (only exact repeats)                     │
    │    Latency: <1ms                                             │
    └──────────────────────────────────────────────────────────────┘
         │ miss
         ▼
    ┌─── Tier 2: Semantic cache (vector similarity) ───────────────┐
    │    Embed query → ANN search in cache vector store            │
    │    If cosine_sim(query, cached_query) > threshold → return   │
    │    Hit rate: ~20-40% (paraphrases + similar intent)         │
    │    Latency: ~10-50ms                                         │
    └──────────────────────────────────────────────────────────────┘
         │ miss
         ▼
    Full RAG pipeline → store result in both caches
```

```python
import redis
from redis import Redis
from sentence_transformers import SentenceTransformer
import numpy as np
import json

class SemanticCache:
    def __init__(self, threshold: float = 0.92, max_size: int = 10_000):
        self.r = Redis()
        self.encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.threshold = threshold
        self.cache_keys = []    # track stored query embeddings
        self.cache_vectors = [] # parallel list of embeddings

    def get(self, query: str) -> str | None:
        # Tier 1: exact match
        exact = self.r.get(f"exact:{hash(query)}")
        if exact:
            return json.loads(exact)

        # Tier 2: semantic match
        q_emb = self.encoder.encode(query)
        if self.cache_vectors:
            matrix = np.vstack(self.cache_vectors)
            sims = matrix @ q_emb / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q_emb))
            best_idx = np.argmax(sims)
            if sims[best_idx] > self.threshold:
                return json.loads(self.r.get(self.cache_keys[best_idx]))
        return None

    def set(self, query: str, answer: str) -> None:
        key = f"sem:{hash(query)}"
        self.r.set(f"exact:{hash(query)}", json.dumps(answer), ex=3600)
        self.r.set(key, json.dumps(answer), ex=3600)
        self.cache_keys.append(key)
        self.cache_vectors.append(self.encoder.encode(query))
```

### Threshold Tuning

| Threshold | Effect |
|-----------|--------|
| 0.95+ | Very conservative — only near-identical queries hit cache |
| 0.90-0.95 | Balanced — paraphrases hit, different intent doesn't |
| < 0.85 | Aggressive — high hit rate but risk returning wrong cached answer |

Start at 0.92. Monitor cache hit rate and user feedback. Lower if hit rate < 15%.

### GPTCache (Library)

```python
from gptcache import cache
from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation

cache.init(
    embedding_func=Onnx().to_embeddings,
    data_manager=get_data_manager(CacheBase("sqlite"), VectorBase("faiss")),
    similarity_evaluation=SearchDistanceEvaluation(),
)
# Now wrap any LLM call — cache handles similarity lookup automatically
```

---

## 3. Index Freshness Management

### The Staleness Problem

Documents change. New policies replace old ones. If you re-index from scratch on every change, indexing cost is O(corpus_size). Incremental indexing targets only changed documents.

### Incremental Re-indexing Pattern

```python
import hashlib
import json
from pathlib import Path

def compute_doc_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

def incremental_index(
    new_docs: list[dict],          # {id, content, updated_at}
    hash_store_path: str = "doc_hashes.json",
    vectorstore,
) -> dict:
    hashes = json.loads(Path(hash_store_path).read_text()) if Path(hash_store_path).exists() else {}
    
    to_add, to_delete, unchanged = [], [], []
    for doc in new_docs:
        new_hash = compute_doc_hash(doc["content"])
        if doc["id"] not in hashes:
            to_add.append(doc)                             # new document
        elif hashes[doc["id"]] != new_hash:
            to_delete.append(doc["id"])                    # changed → delete old
            to_add.append(doc)                             # re-index new version
        else:
            unchanged.append(doc["id"])

    if to_delete:
        vectorstore.delete(ids=to_delete)
    if to_add:
        vectorstore.add_documents([...])                   # embed + upsert

    # Update hash store
    for doc in to_add:
        hashes[doc["id"]] = compute_doc_hash(doc["content"])
    Path(hash_store_path).write_text(json.dumps(hashes))

    return {"added": len(to_add), "deleted": len(to_delete), "unchanged": len(unchanged)}
```

**Trigger strategies:**

| Strategy | When | Trade-off |
|----------|------|-----------|
| Polling (cron) | Every N hours | Simple; may be stale for N hours |
| Webhook-driven | Document update fires event | Real-time; requires event infrastructure |
| Scheduled full rebuild | Weekly + incremental in between | Safety net; catches orphaned deletes |

---

## 4. RAG Cost Model

Know where money goes before optimizing.

```
Per-query cost breakdown (approximate, varies by provider):
─────────────────────────────────────────────────────────
Embedding query       : $0.00001   (text-embedding-3-small, 512 tokens)
Vector ANN search     : $0.000005  (Pinecone / pgvector — negligible)
Reranker inference    : $0.0001    (Cohere Rerank 3 API call)
LLM generation        : $0.002     (GPT-4o-mini, ~1500 context + 200 output tokens)
─────────────────────────────────────────────────────────
Total per query       : ~$0.0022

At 100K queries/day   : $220/day = $6,600/month
```

### Cost Optimization Levers

| Lever | Savings | Trade-off |
|-------|---------|-----------|
| Semantic cache (40% hit rate) | 40% LLM cost | Stale answers risk |
| Smaller embedding model (small vs large) | 5x cheaper embeddings | ~2% recall drop |
| Open-source reranker (BGE vs Cohere) | ~$0.0001/query | Slightly lower quality |
| Smaller LLM for generation (mini vs full) | 10-30x cheaper | Quality drop on complex queries |
| Reduce k (5→3 chunks) | ~30% context tokens | Lower context recall |

**Decision rule:** Profile which component dominates cost first (usually LLM generation). Optimize that one first.

---

## 5. A/B Testing RAG Versions

When you change chunking strategy, embedding model, or retriever, you need statistical evidence that the new version is better.

### Shadow Deployment

```
Production traffic
    │
    ├──► Pipeline v1 → answer → user
    │
    └──► Pipeline v2 (shadow) → answer → RAGAS eval (not shown to user)
```

After N queries, compare RAGAS scores between v1 and v2. Statistical test: paired t-test on faithfulness, answer relevancy. Promote v2 only if improvement is significant (p < 0.05) and no metric regresses below threshold.

### Canary Deployment

Route X% of traffic to new pipeline version. Track user satisfaction signals (thumbs up/down, follow-up questions = unsatisfied). Roll back if satisfaction drops by > 5%.

---

## 6. Observability — What to Log

Every RAG request should produce a structured trace:

```python
import time
from dataclasses import dataclass, asdict
import json

@dataclass
class RAGTrace:
    query: str
    retrieved_chunks: list[str]
    retrieval_latency_ms: float
    rerank_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    answer: str
    faithfulness_score: float | None = None   # RAGAS, computed async
    cache_hit: bool = False
    k_retrieved: int = 5
    model_used: str = "gpt-4o-mini"
    cost_usd: float = 0.0

def log_trace(trace: RAGTrace):
    print(json.dumps(asdict(trace)))  # send to Elasticsearch / CloudWatch / Loki
```

**Key dashboards to build:**

| Dashboard | Metrics | Alert if |
|-----------|---------|---------|
| Latency | P50/P95/P99 total, per stage | P95 > 3s |
| Quality | Rolling RAGAS faithfulness | < 0.75 over 100 queries |
| Cache | Hit rate, cache size, eviction rate | Hit rate < 10% (tuning issue) |
| Cost | $ per query, $ per day | Day-over-day spike > 30% |
| Retrieval | Avg chunks retrieved, re-retrieval rate | k consistently = 0 (index issue) |

---

## 7. Production Readiness Checklist

| Layer | Check | Status |
|-------|-------|--------|
| Retrieval | Recall@5 > 0.75 on eval set | — |
| Generation | Faithfulness > 0.80 on RAGAS | — |
| Latency | P95 < 3s end-to-end | — |
| Cache | Semantic cache deployed, threshold tuned | — |
| Freshness | Incremental indexing pipeline running | — |
| Observability | Structured traces → dashboard | — |
| Cost | Cost-per-query measured and budgeted | — |
| Fallbacks | Response when retrieval returns 0 chunks | — |
| Security | Prompt injection defenses → `03_indirect_prompt_injection.md` | — |

---

## 8. Interview Questions

**Q: How would you reduce RAG API costs by 40%?**

First profile where cost goes: typically 90% is the LLM generation call, not the embedding or retrieval. Then: (1) Deploy a semantic cache — 20-40% of production queries are paraphrases; serving from cache eliminates LLM calls entirely. (2) Reduce k from 5 to 3 if RAGAS context recall stays above threshold — fewer context tokens = cheaper generation. (3) Evaluate whether a smaller model (gpt-4o-mini vs gpt-4o) maintains acceptable faithfulness on your query distribution. Measure before committing — don't guess which lever matters most.

**Q: How do you keep a RAG index fresh when documents change?**

Track a hash of each document's content in a side-store. On each update cycle, re-hash all documents and compare. New documents → embed + insert. Changed documents → delete old vectors by ID, embed + insert new version. Unchanged documents → skip. This makes re-indexing O(changed_docs) not O(corpus). Schedule weekly full rebuilds as a safety net to catch edge cases (delete events missed, hash store corruption).

**Q: How do you measure if a RAG change actually improved things?**

Shadow deployment: run both old and new pipelines on the same queries, evaluate both with RAGAS offline. Paired t-test on faithfulness and answer relevancy scores (N ≥ 100 queries for 80% power). Promote if the new pipeline shows a statistically significant improvement and no metric regresses below the gate threshold. For user-visible quality, also track thumbs-up rate and follow-up question rate (users asking again = unsatisfied).

---

## Connections

| Topic | File |
|-------|------|
| Advanced query techniques | [04_advanced_rag.md](04_advanced_rag.md) |
| RAGAS evaluation framework | [05_rag_evaluation.md](05_rag_evaluation.md) |
| Indirect prompt injection | [03_indirect_prompt_injection.md](03_indirect_prompt_injection.md) |
| Production RAG ops depth (drift, observability) | [../10.mlops/13_production_rag_ops.md](../10.mlops/13_production_rag_ops.md) |
| LLM observability (LangSmith, Phoenix, LangFuse) | [../10.mlops/11_llm_observability.md](../10.mlops/11_llm_observability.md) |
| Multi-tenant RAG system design | [../11.system_design/10_multi_tenant_rag.md](../11.system_design/10_multi_tenant_rag.md) |

---

## Code Practice

- `code_practice/07_rag/05_production_rag/pipeline.py` — semantic cache + incremental indexing
- `code_practice/07_rag/05_production_rag/serve.py` — FastAPI server with tracing

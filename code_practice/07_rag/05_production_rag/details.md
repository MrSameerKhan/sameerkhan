# Session 5 — Production RAG
Status: `🔧 Code-built`

Theory: [../../../7.rag/06_production_rag.md](../../../7.rag/06_production_rag.md)

---

## Use Case

Senior ML engineer taking a working RAG notebook to production. Adds: semantic cache (cuts repeat-query LLM costs by 30-40%), structured tracing (every request logged for debugging), and a FastAPI server (the RAG is now a REST API).

---

## File Structure

```
05_production_rag/
├── pipeline.py    — RAGPipeline class with SemanticCache + RAGTrace
└── serve.py       — FastAPI REST server wrapping pipeline.py
```

---

## SemanticCache — Two-Tier Design

```
Request: "What is the max LTV for first-time buyers?"
    │
    ├── Tier 1: SHA-256 hash lookup          O(1), ~5 μs
    │   "what is the max ltv for first-time buyers?" → hash → check dict
    │   Hit → return instantly (exact duplicate query)
    │
    └── Tier 2: Cosine similarity            O(N), ~1 ms for N=1000
        embed(query) → dot product with all cached embeddings
        If max_sim ≥ 0.92 → return cached answer
        (catches: "Max LTV first-time buyer?", "LTV limit for first-time buyers?")
    │
    Miss → full RAG pipeline → store in both tiers
```

**Threshold 0.92:** conservative — only paraphrases of the same question hit. Lower to 0.85 for higher hit rate (but risk serving wrong cached answer for different intent).

---

## RAGTrace — Structured Logging

Every request emits a JSON trace:
```json
{
  "query": "What is the early repayment charge?",
  "answer": "The ERC is 3% in year 3...",
  "retrieved_titles": ["Residential Mortgage Policy — Interest Rates"],
  "retrieval_ms": 12.3,
  "generation_ms": 847.1,
  "total_ms": 859.4,
  "cache_hit": false,
  "cache_tier": "miss",
  "input_tokens": 412,
  "output_tokens": 68,
  "model": "gpt-4o-mini"
}
```

Feed to: Elasticsearch → Kibana dashboard, or CloudWatch Logs → CloudWatch metrics.

---

## FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Main RAG endpoint |
| `GET` | `/cache/stats` | Hit rates, cache size |
| `GET` | `/health` | Server health + chunk count |

```bash
# Query
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the max LTV?", "include_trace": true}'

# Cache stats
curl http://localhost:8000/cache/stats
```

---

## Expected Output (pipeline.py standalone)

```
Initialising production RAG pipeline...
  Ready — 34 chunks indexed

[      API CALL] (  891ms)  What is the maximum LTV for first-time buyers?
                 → First-time buyers are eligible for up to 95% LTV under Help to Buy...

[      API CALL] (  743ms)  What is the early repayment charge in year 3?
                 → The early repayment charge in year 3 is 3%...

[CACHE SEMANTIC] (    3ms)  What is max LTV for a first-time home buyer?
                 → First-time buyers are eligible for up to 95% LTV under Help to Buy...

[      API CALL] (  812ms)  Can expatriates apply for personal finance?
                 → Yes, expatriates can apply for personal finance...

[      API CALL] (  901ms)  What documents are needed for a mortgage?
                 → Acceptable documents include payslips, SA302 tax returns...

[CACHE SEMANTIC] (    2ms)  Maximum LTV for first-time buyer?
                 → First-time buyers are eligible for up to 95% LTV under Help to Buy...

Cache stats: {'total_requests': 6, 'exact_hits': 0, 'semantic_hits': 2,
              'misses': 4, 'hit_rate': 0.333, 'cache_size': 4}
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/07_rag/05_production_rag

# Standalone pipeline demo:
python pipeline.py

# FastAPI server:
uvicorn serve:app --reload --port 8000
```

Cost: ~$0.03 per pipeline.py demo run (4 API calls; 2 hits served from cache).

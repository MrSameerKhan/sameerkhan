# Session 5 — Production RAG: serve.py
# FastAPI server wrapping the RAGPipeline with a REST API.
#
# Run: uvicorn serve:app --reload --port 8000
# Query: curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
#              -d '{"question": "What is the max LTV for first-time buyers?"}'

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import RAGPipeline, RAGTrace


# ── Lifecycle: build index once at startup ─────────────────────────────────────
pipeline: RAGPipeline | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Building RAG index...")
    pipeline = RAGPipeline()
    print(f"  Ready — {len(pipeline.chunks)} chunks indexed")
    yield
    # Cleanup (if needed) goes here

app = FastAPI(title="Production RAG API", version="1.0.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    include_trace: bool = False   # set True to return latency + token counts


class QueryResponse(BaseModel):
    answer:     str
    cache_hit:  bool
    cache_tier: str
    total_ms:   float
    sources:    list[str] = []
    trace:      dict | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    trace: RAGTrace = pipeline.query(req.question)

    return QueryResponse(
        answer=trace.answer,
        cache_hit=trace.cache_hit,
        cache_tier=trace.cache_tier,
        total_ms=trace.total_ms,
        sources=list(dict.fromkeys(trace.retrieved_titles)),  # deduplicated, order preserved
        trace=None if not req.include_trace else {
            "retrieval_ms":   trace.retrieval_ms,
            "generation_ms":  trace.generation_ms,
            "input_tokens":   trace.input_tokens,
            "output_tokens":  trace.output_tokens,
            "top_k":          trace.top_k,
            "model":          trace.model,
        },
    )


@app.get("/cache/stats")
async def cache_stats() -> dict:
    return pipeline.cache.stats


@app.get("/health")
async def health() -> dict:
    return {
        "status":      "ok",
        "chunks":      len(pipeline.chunks),
        "cache_size":  pipeline.cache.stats["cache_size"],
    }


# ── Example: run a quick smoke test without uvicorn ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)

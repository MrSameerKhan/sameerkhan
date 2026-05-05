"""
api.py — FastAPI backend for the RAG System

Endpoints:
  POST /ingest      Upload .txt / .pdf files, rebuild FAISS index
  POST /query       Query the RAG pipeline, get answer + sources + latency
  GET  /evaluate    Run evaluation test set, return MRR / P@K / R@K
  GET  /health      Liveness check

Run:
  uvicorn api:app --reload --port 8000

Docs (auto-generated):
  http://localhost:8000/docs
"""

import json
import os
import tempfile
import time

import faiss
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from config import CHUNKS_PATH, EMBED_MODEL, INDEX_PATH, TOP_K
from evaluate import TEST_SET, precision_at_k, recall_at_k, reciprocal_rank
from ingest import build_index, chunk_text, embed_chunks, load_pdf, load_txt
from llm import generate
from pipeline import SYSTEM_PROMPT, build_prompt
from retriever import Retriever

app = FastAPI(title="RAG System API", version="1.0.0", description="Production RAG pipeline — FAISS + sentence-transformers + Ollama/HuggingFace")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded shared retriever — reset after every ingest
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    k: int = TOP_K


class SourceChunk(BaseModel):
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceChunk]
    latency_ms: float


class IngestResponse(BaseModel):
    files_ingested: list[str]
    total_chunks: int
    index_size: int


class EvalResult(BaseModel):
    query: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    top_source: str
    top_score: float


class EvaluateResponse(BaseModel):
    k: int
    results: list[EvalResult]
    mean_precision: float
    mean_recall: float
    mrr: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: list[UploadFile] = File(...)):
    """Upload .txt or .pdf documents and rebuild the FAISS index."""
    all_chunks = []
    file_names = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            if not (upload.filename.endswith(".txt") or upload.filename.endswith(".pdf")):
                raise HTTPException(400, f"Unsupported file type: {upload.filename}. Only .txt and .pdf are accepted.")

            dest = os.path.join(tmpdir, upload.filename)
            content = await upload.read()
            with open(dest, "wb") as f:
                f.write(content)

            text = load_txt(dest) if upload.filename.endswith(".txt") else load_pdf(dest)
            chunks = chunk_text(text, upload.filename)
            all_chunks.extend(chunks)
            file_names.append(upload.filename)

    if not all_chunks:
        raise HTTPException(400, "No text content extracted from uploaded files.")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(all_chunks, model)
    index = build_index(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)

    reset_retriever()

    return IngestResponse(
        files_ingested=file_names,
        total_chunks=len(all_chunks),
        index_size=index.ntotal,
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Query the RAG pipeline — retrieve relevant chunks and generate an answer."""
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(503, "Index not found. Call POST /ingest first.")

    retriever = get_retriever()
    t0 = time.perf_counter()

    chunks = retriever.retrieve(req.query, k=req.k)
    if not chunks:
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return QueryResponse(
            query=req.query,
            answer="No relevant documents found in the index.",
            sources=[],
            latency_ms=latency_ms,
        )

    user_message = build_prompt(req.query, chunks)
    answer = generate(SYSTEM_PROMPT, user_message)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    sources = [
        SourceChunk(
            source=c["source"],
            score=round(c["score"], 4),
            text=c["text"][:300],
        )
        for c in chunks
    ]

    return QueryResponse(
        query=req.query,
        answer=answer,
        sources=sources,
        latency_ms=latency_ms,
    )


@app.get("/evaluate", response_model=EvaluateResponse)
async def evaluate(k: int = TOP_K):
    """Run the evaluation test set and return retrieval quality metrics."""
    if not os.path.exists(INDEX_PATH):
        raise HTTPException(503, "Index not found. Call POST /ingest first.")

    retriever = get_retriever()
    results = []
    p_scores, r_scores, rr_scores = [], [], []

    for item in TEST_SET:
        q        = item["query"]
        relevant = item["relevant_sources"]
        retrieved = retriever.retrieve(q, k=k)

        p  = precision_at_k(retrieved, relevant, k)
        r  = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant)

        p_scores.append(p)
        r_scores.append(r)
        rr_scores.append(rr)

        top = retrieved[0] if retrieved else {"source": "none", "score": 0.0}
        results.append(EvalResult(
            query=q,
            precision_at_k=round(p, 4),
            recall_at_k=round(r, 4),
            reciprocal_rank=round(rr, 4),
            top_source=top["source"],
            top_score=round(float(top.get("score", 0.0)), 4),
        ))

    return EvaluateResponse(
        k=k,
        results=results,
        mean_precision=round(float(np.mean(p_scores)), 4),
        mean_recall=round(float(np.mean(r_scores)), 4),
        mrr=round(float(np.mean(rr_scores)), 4),
    )


@app.get("/health")
async def health():
    index_ready = os.path.exists(INDEX_PATH)
    n_vectors = 0
    if index_ready:
        idx = faiss.read_index(INDEX_PATH)
        n_vectors = idx.ntotal
    return {"status": "ok", "index_ready": index_ready, "vectors": n_vectors}

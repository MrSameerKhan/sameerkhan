# RAG System — Production-Grade Document Q&A

End-to-end Retrieval-Augmented Generation pipeline: sentence-transformer embeddings,
FAISS `IndexFlatIP` cosine search, FastAPI backend, and Streamlit UI.
Deploys locally with Ollama or to HuggingFace Spaces with the HF Inference API.

---

## Table of Contents
- [Stack](#stack)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Quick Start (Local)](#quick-start-local)
- [Run the Streamlit UI](#run-the-streamlit-ui)
- [Run the FastAPI Backend](#run-the-fastapi-backend)
- [Deploy to HuggingFace Spaces](#deploy-to-huggingface-spaces)
- [Evaluation Metrics](#evaluation-metrics)
- [CLI Tools (original scripts)](#cli-tools-original-scripts)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Stack

| Component | Tool | Notes |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | 384-dim, runs on M1 MPS, no API key |
| Vector store | FAISS `IndexFlatIP` | Exact cosine via L2-normalised inner product, <5ms search |
| LLM (local) | Ollama `llama3.2:1b` | Runs on M1, ~10s/query, no API key |
| LLM (cloud) | HuggingFace Inference API | `Mistral-7B-Instruct-v0.1`, set `LLM_PROVIDER=huggingface` |
| API | FastAPI + uvicorn | `/ingest`, `/query`, `/evaluate`, auto-generated `/docs` |
| UI | Streamlit | Upload docs, ask questions, evaluation dashboard |
| PDF parsing | `pypdf` | Also handles `.txt` files |

---

## Project Structure

```
rag_system/
  ├── config.py          ← all settings (chunk size, model names, top-K, LLM provider)
  ├── ingest.py          ← load docs → chunk → embed → save FAISS index  (CLI)
  ├── retriever.py       ← load index → embed query → return top-K chunks (CLI)
  ├── pipeline.py        ← retriever + Ollama → streamed answer            (CLI)
  ├── evaluate.py        ← Precision@K, Recall@K, MRR on test queries      (CLI)
  ├── llm.py             ← LLM abstraction (Ollama | HuggingFace)
  ├── api.py             ← FastAPI backend
  ├── app.py             ← Streamlit UI
  ├── requirements.txt
  └── sample_data/
      └── ml_notes.txt   ← sample ML/DL notes
```

---

## Architecture

```
                     OFFLINE (run once — via UI upload or CLI ingest)
┌─────────────┐     ┌──────────┐     ┌────────────┐     ┌─────────────┐
│ .txt / .pdf │────▶│  Chunk   │────▶│   Embed    │────▶│    FAISS    │
│  documents  │     │ 500 char │     │ MiniLM-L6  │     │ IndexFlatIP │
└─────────────┘     └──────────┘     └────────────┘     └─────────────┘

                      ONLINE (per query)
                                                         ┌─────────────────────┐
┌───────────┐     ┌────────────┐     ┌────────────┐     │   Build prompt      │
│   Query   │────▶│   Embed    │────▶│FAISS search│────▶│   + context         │
│           │     │ MiniLM-L6  │     │  top-K=5   │     └──────────┬──────────┘
└───────────┘     └────────────┘     └────────────┘                │
                                                                    ▼
                                                   ┌────────────────────────────┐
                                                   │  llm.py                    │
                                                   │  ├── Ollama (local)        │
                                                   │  └── HF Inference (cloud)  │
                                                   └────────────────────────────┘

                      API LAYER (FastAPI)
  POST /ingest  → rebuild index from uploaded files
  POST /query   → run full RAG pipeline, return answer + sources + latency_ms
  GET  /evaluate → run test set, return MRR / P@K / R@K
  GET  /health  → liveness + index status
```

---

## Quick Start (Local)

### 1. Install Ollama + pull model

```bash
brew install ollama
ollama pull llama3.2:1b
```

### 2. Install Python packages

```bash
conda activate skhan
cd projects/rag_system
pip install -r requirements.txt
```

### 3. Ingest sample documents

```bash
python ingest.py --docs sample_data/
```

Expected output:
```
=== INGEST: sample_data/ ===
[1/4] Loading documents...
  Loaded: ml_notes.txt (8,281 chars)
[2/4] Chunking (size=500, overlap=50)...
  Total chunks: 19
[3/4] Loading embedding model: all-MiniLM-L6-v2
  Embedding 19 chunks...
[4/4] Building FAISS index and saving...
  Saved: faiss.index  (19 vectors, dim=384)
  Saved: chunks.json
```

---

## Run the Streamlit UI

```bash
conda activate skhan
cd projects/rag_system
streamlit run app.py
```

Opens at `http://localhost:8501`

**Tabs:**
- **💬 Q&A** — ask questions, see answer + retrieved source chunks with scores
- **📥 Upload Docs** — upload `.txt`/`.pdf` files to rebuild the index
- **📊 Evaluation** — run test set, see MRR / P@K / R@K metrics table

> The app auto-ingests `sample_data/` on first run if no index exists.

---

## Run the FastAPI Backend

```bash
conda activate skhan
cd projects/rag_system
uvicorn api:app --reload --port 8000
```

Interactive docs (auto-generated Swagger UI): `http://localhost:8000/docs`

### POST /ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@sample_data/ml_notes.txt"
```

Response:
```json
{"files_ingested": ["ml_notes.txt"], "total_chunks": 19, "index_size": 19}
```

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the attention mechanism?", "k": 5}'
```

Response:
```json
{
  "query": "What is the attention mechanism?",
  "answer": "The attention mechanism allows a model to focus...",
  "sources": [
    {"source": "ml_notes.txt", "score": 0.5622, "text": "..."},
    ...
  ],
  "latency_ms": 4231.5
}
```

### GET /evaluate

```bash
curl "http://localhost:8000/evaluate?k=5"
```

Response:
```json
{"k": 5, "mrr": 1.0, "mean_precision": 1.0, "mean_recall": 1.0, "results": [...]}
```

### GET /health

```bash
curl http://localhost:8000/health
# {"status": "ok", "index_ready": true, "vectors": 19}
```

---

## Deploy to HuggingFace Spaces

1. Create a new Space at `huggingface.co/new-space` → type: **Streamlit**
2. Push this folder to the Space repo
3. Set Space secrets:
   ```
   LLM_PROVIDER = huggingface
   HF_TOKEN     = hf_your_token_here
   ```
4. The app will use `mistralai/Mistral-7B-Instruct-v0.1` via the Inference API (free tier)
5. Add `faiss.index` and `chunks.json` to the repo (pre-built), or the app will auto-ingest `sample_data/` on startup

---

## Evaluation Metrics

Run `python evaluate.py` or use the Evaluation tab in the UI.

| Metric | Formula | What it means |
|---|---|---|
| **P@K** | (relevant chunks in top-K) / K | Fraction of retrieved chunks that are on-topic |
| **R@K** | (relevant chunks in top-K) / (total relevant) | Fraction of relevant material that was found |
| **MRR** | mean(1 / rank of first relevant chunk) | How high the first correct chunk is ranked; 1.0 = always #1 |

Current results on `ml_notes.txt` test set (5 queries):
```
Mean P@5 : 1.0000
Mean R@5 : 1.0000
MRR      : 1.0000
```

---

## CLI Tools (original scripts)

The original CLI scripts still work independently:

```bash
# Ingest
python ingest.py --docs sample_data/

# Test retrieval only (no LLM)
python retriever.py "what is attention mechanism?"

# Full RAG pipeline (streaming)
python pipeline.py --query "how does LSTM solve vanishing gradient?"
python pipeline.py          # interactive loop

# Evaluate retrieval quality
python evaluate.py
```

---

## Configuration

All settings in `config.py`:

| Setting | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk — larger = more context, fewer chunks |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks — prevents splitting key sentences |
| `TOP_K` | 5 | Chunks retrieved per query — more = richer context, longer prompt |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Swap to `all-mpnet-base-v2` for better retrieval (slower) |
| `OLLAMA_MODEL` | `llama3.2:1b` | Swap to `llama3.2:3b` or `mistral` for better quality (slower) |
| `LLM_PROVIDER` | `ollama` (env) | `huggingface` for Spaces deployment |
| `HF_INFERENCE_MODEL` | `mistralai/Mistral-7B-Instruct-v0.1` | Cloud LLM for HF provider |

---

## Troubleshooting

**`Cannot reach Ollama`**
```bash
ollama serve
```

**`Model 'llama3.2:1b' not found`**
```bash
ollama pull llama3.2:1b
```

**`faiss.index not found`**
Run `python ingest.py --docs sample_data/` first, or upload files in the Streamlit UI.

**Slow responses**
```python
# config.py
OLLAMA_MODEL = "llama3.2:1b"   # already the fastest local option
```

**HuggingFace warning about unauthenticated requests**
Harmless — model is cached locally. To silence:
```bash
export HF_TOKEN="hf_your_token"
```

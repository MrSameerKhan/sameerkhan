import os

EMBED_MODEL   = "all-MiniLM-L6-v2"   # 384-dim, ~22M params, runs on M1 MPS
OLLAMA_MODEL  = "llama3.2:1b"         # fast on M1, 1.3GB
CHUNK_SIZE    = 500    # characters per chunk
CHUNK_OVERLAP = 50     # characters of overlap between chunks
TOP_K         = 5      # chunks to retrieve per query
INDEX_PATH    = "faiss.index"
CHUNKS_PATH   = "chunks.json"

# LLM provider — "ollama" (local) or "huggingface" (Inference API, for Spaces)
LLM_PROVIDER       = os.getenv("LLM_PROVIDER", "ollama")
HF_INFERENCE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# FastAPI base URL (used when running Streamlit against the API server)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

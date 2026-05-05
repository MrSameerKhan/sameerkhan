"""
app.py — Streamlit UI for the RAG System

Tabs:
  💬 Q&A            — Ask questions, see answer + retrieved sources
  📥 Upload Docs    — Ingest new .txt / .pdf files into the knowledge base
  📊 Evaluation     — Run retrieval evaluation (MRR, P@K, R@K)

Deployment:
  Local:              streamlit run app.py
  HuggingFace Spaces: set LLM_PROVIDER=huggingface and HF_TOKEN in Spaces secrets

Auto-ingest: if no FAISS index exists, sample_data/ is ingested automatically on startup.
"""

import json
import os
import sys
import tempfile
import time

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# Ensure imports resolve when running from repo root or Spaces
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    API_BASE_URL,
    CHUNKS_PATH,
    EMBED_MODEL,
    INDEX_PATH,
    LLM_PROVIDER,
    TOP_K,
)
from evaluate import TEST_SET, precision_at_k, recall_at_k, reciprocal_rank
from ingest import build_index, chunk_text, embed_chunks, load_pdf, load_txt
from llm import generate
from pipeline import SYSTEM_PROMPT, build_prompt
from retriever import Retriever

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
)

# ── Auto-ingest on first run ──────────────────────────────────────────────────

SAMPLE_DATA = os.path.join(os.path.dirname(__file__), "sample_data")


@st.cache_resource(show_spinner="Building index from sample_data/ ...")
def ensure_index():
    if os.path.exists(INDEX_PATH):
        return
    if not os.path.isdir(SAMPLE_DATA):
        return
    from ingest import load_docs

    docs = load_docs(SAMPLE_DATA)
    if not docs:
        return
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc["text"], doc["source"]))
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = embed_chunks(all_chunks, model)
    index = build_index(embeddings)
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f, indent=2)


ensure_index()


# ── Cached retriever ──────────────────────────────────────────────────────────

@st.cache_resource
def get_retriever():
    return Retriever()


# ── Header ────────────────────────────────────────────────────────────────────

st.title("RAG System — Document Q&A")

provider_label = "HuggingFace Inference API" if LLM_PROVIDER == "huggingface" else "Ollama (local)"
st.caption(
    f"Embeddings: `all-MiniLM-L6-v2` · Vector store: `FAISS IndexFlatIP` · LLM: {provider_label}"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_qa, tab_upload, tab_eval = st.tabs(["💬 Q&A", "📥 Upload Docs", "📊 Evaluation"])


# ── Q&A Tab ───────────────────────────────────────────────────────────────────

with tab_qa:
    st.markdown("Ask questions about the documents in your knowledge base.")

    query = st.text_input(
        "Your question:",
        placeholder="What is the attention mechanism in transformers?",
    )
    k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=10, value=TOP_K)

    if st.button("Ask", type="primary") and query.strip():
        if not os.path.exists(INDEX_PATH):
            st.error("No index found. Upload documents in the 'Upload Docs' tab first.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                t0 = time.perf_counter()
                retriever = get_retriever()
                chunks = retriever.retrieve(query, k=k)

                if not chunks:
                    st.warning("No relevant documents found in the index.")
                else:
                    user_message = build_prompt(query, chunks)
                    answer = generate(SYSTEM_PROMPT, user_message)
                    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

                    st.markdown("### Answer")
                    st.write(answer)
                    st.caption(f"Total latency: **{latency_ms} ms** · Retrieved {len(chunks)} chunks")

                    st.markdown("### Retrieved Sources")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(
                            f"[{i}] `{chunk['source']}` — score: {chunk['score']:.4f}"
                        ):
                            st.text(chunk["text"])


# ── Upload Tab ────────────────────────────────────────────────────────────────

with tab_upload:
    st.markdown(
        "Upload `.txt` or `.pdf` files to add to the knowledge base. "
        "This rebuilds the FAISS index — existing documents are replaced."
    )

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    if st.button("Ingest Documents", type="primary") and uploaded_files:
        with st.spinner(f"Ingesting {len(uploaded_files)} file(s)..."):
            all_chunks = []
            names = []

            with tempfile.TemporaryDirectory() as tmpdir:
                for f in uploaded_files:
                    dest = os.path.join(tmpdir, f.name)
                    with open(dest, "wb") as out:
                        out.write(f.read())
                    text = load_txt(dest) if f.name.endswith(".txt") else load_pdf(dest)
                    chunks = chunk_text(text, f.name)
                    all_chunks.extend(chunks)
                    names.append(f.name)

            if not all_chunks:
                st.error("No text content found in uploaded files.")
            else:
                model = SentenceTransformer(EMBED_MODEL)
                embeddings = embed_chunks(all_chunks, model)
                index = build_index(embeddings)
                faiss.write_index(index, INDEX_PATH)
                with open(CHUNKS_PATH, "w") as out:
                    json.dump(all_chunks, out, indent=2)

                # Clear cached retriever so next query loads the new index
                st.cache_resource.clear()

                st.success(
                    f"Ingested **{len(names)}** file(s) → "
                    f"**{len(all_chunks)}** chunks → "
                    f"index size: **{index.ntotal}** vectors"
                )
                for name in names:
                    st.markdown(f"- `{name}`")


# ── Evaluation Tab ────────────────────────────────────────────────────────────

with tab_eval:
    st.markdown(
        "Measure retrieval quality on the built-in test set. "
        "Metrics: **MRR** (Mean Reciprocal Rank), **P@K** (Precision), **R@K** (Recall)."
    )

    k_eval = st.slider("K for evaluation", min_value=1, max_value=10, value=TOP_K, key="k_eval")

    if st.button("Run Evaluation", type="primary"):
        if not os.path.exists(INDEX_PATH):
            st.error("No index found. Upload documents first.")
        else:
            with st.spinner("Running evaluation..."):
                retriever = get_retriever()
                p_scores, r_scores, rr_scores = [], [], []
                rows = []

                for item in TEST_SET:
                    q        = item["query"]
                    relevant = item["relevant_sources"]
                    retrieved = retriever.retrieve(q, k=k_eval)

                    p  = precision_at_k(retrieved, relevant, k_eval)
                    r  = recall_at_k(retrieved, relevant, k_eval)
                    rr = reciprocal_rank(retrieved, relevant)

                    p_scores.append(p)
                    r_scores.append(r)
                    rr_scores.append(rr)

                    top = retrieved[0] if retrieved else {"source": "none", "score": 0.0}
                    rows.append({
                        "Query": q[:70],
                        f"P@{k_eval}": round(p, 3),
                        f"R@{k_eval}": round(r, 3),
                        "RR": round(rr, 3),
                        "Top-1 Source": top["source"],
                        "Top-1 Score": round(float(top.get("score", 0.0)), 4),
                    })

            col1, col2, col3 = st.columns(3)
            col1.metric("MRR", f"{np.mean(rr_scores):.4f}")
            col2.metric(f"Mean P@{k_eval}", f"{np.mean(p_scores):.4f}")
            col3.metric(f"Mean R@{k_eval}", f"{np.mean(r_scores):.4f}")

            st.markdown("### Per-query Results")
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

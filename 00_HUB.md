# Hub — Sameer Khan | Daily Action Center

> **Open this every day.** Everything actionable is here. Deep reference — follow the links. Last updated: May 2026

---

## Reference Files (3 deep files, all you need)

| File | What's in it | When to open |
|------|-------------|--------------|
| 01_CAREER_PACK | JD analysis · companies · salary tables · LinkedIn · networking · application strategy | Before applying / negotiating |
| 02_INTERVIEW_PACK | Technical Q&A (20 detailed) · question bank · STAR behavioral · day-before checklist | 1 day before any interview |
| 03_LEARNING_PACK | Mental model · decision matrices · 68-session coding sequence · theory checklist · reading list | Daily — when studying or coding |

---

## Who You Are (positioning)

*"I build end-to-end LLM-powered document intelligence systems — from OCR and multimodal extraction through RAG pipelines, fine-tuned models, and production deployment on AWS and Databricks. In 8 years at ICE Data Services and Al Rajhi Bank, I have shipped systems that process hundreds of thousands of financial documents daily: 94% classification accuracy; 60% reduction in root-cause analysis time. My current focus: production RAG with hybrid retrieval and RAGAS evaluation, QLoRA fine-tuning, and LangGraph agentic workflows."*

8 years experience · ICE Data Services (direct) · Al Rajhi Bank · Hyderabad · Document AI specialist

---

## Your Competitive Edge (lead with these in every interview)

- **Production Document AI at scale** — 8 years, real financial documents, measurable outcomes (94% accuracy, 60% RCA reduction). Most LLM candidates have only personal projects. You have shipped.
- **Full pipeline ownership** — ingestion → preprocessing → distributed training (Horovod) → serving (SageMaker) → CI/CD (Jenkins+Docker) → monitoring.
- **Financial domain expertise** — Al Rajhi Bank (Arabic documents, Saudi Arabia) + ICE Data Services (global financial markets). Domain knowledge = instant credibility.
- **OCR + multimodal foundation** — CNN+LSTM+Transformer hybrid + Donut OCR-free extraction. Most LLM engineers skip directly to text.
- **Modern LLM stack already built** — Production RAG (FAISS + FastAPI + Streamlit), QLoRA fine-tuning code, LangGraph agent code. Working artifacts, not theory.

---

## Current Status

| Area | Status |
|------|--------|
| Theory + Mental Model | ✅ Complete — 03_LEARNING_PACK |
| Phase 1 — Sequence Models | ✅ Complete (9/9 sessions run on work laptop) |
| Phase 2 — Transformers | ✅ Complete (11/11 sessions run on work laptop) |
| Phase 3 — Prompting | ✅ Complete (10/10 sessions run on work laptop with Ollama) |
| Phase 4 — LLMs | ⏳ Code built; waiting for MacBook + model downloads to RUN |
| Phase 4.5 — Advanced | ⏳ Code built (CPT, FC-FT, distillation, speculative decoding) |
| Phase 5 — RAG | ⏳ Docs complete (10/10); code on request |
| Phase 6 — Agents | ⏳ Docs complete (10/10); code on request |
| RAG project (`archive/projects/rag_system/`) | ✅ Built: FastAPI + Streamlit + eval · ⏳ HF Spaces deploy pending |
| Resume | ✅ Tagline, summary, skills, ICE bullets, RAG project added |
| LLM fine-tune resume bullet | ⏳ Add after Phase 4 Session 7 run on MacBook |
| LangGraph agent resume bullet | ⏳ Add after Phase 6 (currently docs-only) |
| Job applications | ⏸ Hold — finish Phase 4 fine-tune on MacBook + add to resume first |

---

## What To Do RIGHT NOW (priority order)

### 1. Finish Phase 4 LLM Fine-Tune on MacBook ⏳ ACTIVE

- [ ] Download TinyLlama-1.1B-Chat-v1.0, gpt2, distilgpt2 to MacBook
- [ ] Transfer to ICE-laptop OneDrive (or use MacBook for entire phase)
- [ ] Run Phase 4 sessions 1 → 6 → 7 → 8 → 9 in order
- [ ] Capture loss curves + eval numbers in each session's `all_details.md`
- [ ] Update resume: "Fine-tuned TinyLlama with LoRA on synthetic banking dataset; X% key-fact recall improvement over base"

### 2. Deploy RAG project to HuggingFace Spaces

- [ ] Set `LLM_PROVIDER=huggingface` + `HF_TOKEN` in Spaces secrets
- [ ] Test live link
- [ ] Add live demo link to resume + LinkedIn

### 3. Update LinkedIn (after #1 + #2 complete)

- [ ] Headline per audience — see 01 CAREER PACK §19
- [ ] About section — see 01 CAREER PACK §20
- [ ] Featured projects: pin RAG + fine-tune

### 4. Start applying (after #1, #2, #3)

- [ ] Top 12 active matches first — see 01 CAREER PACK §6
- [ ] Tailor 2-3 lines of resume summary per role
- [ ] Weekly: 5 applications + 5 LinkedIn outreaches

### 5. Build Phase 6 Agent project (after applications submitted)

- [ ] Move from docs-only to actual code execution
- [ ] LangGraph + tool registry + memory
- [ ] Add to resume: "Built LangGraph agent with tool registry, memory, planner/executor"

---

## Resume — Sections Ready to Paste

### ICE Data Services bullets

**ICE Data Services | Sr. ML Engineer | Aug 2021 – Present** *(Aug 2021–Nov 2023 via Persistent Systems; Nov 2023–Present direct hire)*

- Architected a multimodal document classifier (CNN + BiLSTM + Transformer) for ICE's automated mortgage processing pipeline, achieving **93% page-level / 85% document-level accuracy** across 100+ document types; extended with BERT NER to improve page accuracy to **94%**, deployed to production on SageMaker.

- Designed and scaled a document deduplication pipeline across a 7.1M-page mortgage document bank — DonutSwin page embeddings, GPU-accelerated FAISS IndexFlatIP cosine search, and Union-Find transitive clustering deployed across 40 GPU nodes (TorchDistributor + NCCL); reduced training corpus by **67% pages and 69% documents**.

- Built a FAISS semantic search system using sentence-transformer embeddings for model prediction root-cause analysis, replacing keyword lookup and reducing investigation time by **60%** — establishing vector search as a core pattern in ICE's document AI stack.

- Owned MLOps infrastructure across the document AI pipeline — distributed training with Horovod (**~30% training time**), MLflow experiment tracking, Jenkins CI/CD, and Databricks + SageMaker production serving.

---

## Technical Skills section

```
LLM & GenAI:  RAG pipelines, LLM fine-tuning (LoRA/QLoRA), Prompt Engineering,
              LangChain, LangGraph, FAISS, Sentence Transformers, HuggingFace
MLOps:        SageMaker, Databricks, Jenkins CI/CD, Docker, MLflow, Evidently
NLP & DL:     BERT, Transformers, NER, BiLSTM, CNN, PyTorch, TensorFlow
Cloud:        AWS (SageMaker, S3), Azure ML, Databricks, PySpark
Languages:    Python, SQL
```

---

## Personal Projects — already on resume

**RAG System (Production, 2026)**
End-to-end RAG pipeline — sentence-transformers (all-MiniLM-L6-v2), FAISS IndexFlatIP, FastAPI backend (/ingest, /query, /evaluate), Streamlit UI. MRR=1.0 on ML domain evaluation set.
Live: huggingface.co/spaces/MrSameerKhan/rag-system

---

## Personal Projects — add AFTER Phase 4 run completes

**LLM Fine-tuning POC (2026)** &nbsp;&nbsp;&nbsp;&nbsp;+ add after Phase 4 Session 7 run on MacBook

LoRA fine-tuning of TinyLlama-1.1B-Chat-v1.0 on synthetic Acme Financial Services instruction dataset using PEFT + Trainer (Mac MPS). 0.13% trainable params (r=8, q/k/v/o projections). McNemar-significant improvement on key-fact recall over base.

---

## Personal Projects — add AFTER Phase 6 code execution

**Document Agent POC (2026)** &nbsp;&nbsp;&nbsp;&nbsp;+ add after Phase 6 build + run

LangGraph multi-step agent for mortgage document processing — classify + retrieve + extract + answer. Tool registry with Pydantic schemas + authz + audit log. HITL approval for high-stakes actions. Production ReAct with iteration caps and duplicate-call detection.

---

## JD Coverage — Where You Stand

**After RAG + resume update (now):**

```
JD QUALIFICATION RATE: ~50-55%
```

**After Phase 4 fine-tune + Phase 6 agent run + resume updated:**

```
COVERED (high frequency, all have proof)
  Python 97% · PyTorch/TF 79% · NLP 69% · Deep Learning 64% · Classical ML 64%
  SQL 54% · RAG 51% · MLOps 49% · LangChain/LangGraph 46% · Prompt Eng 38%
  Embeddings/FAISS 38% · LLM fine-tuning 36% · AWS SageMaker 38%
  PySpark/Databricks 31% · HuggingFace 31% · Agents 23% · LoRA/QLoRA 15%

STILL MISSING (low frequency, skip for now)
  AWS Bedrock 10% · Airflow 13% · GCP 10% · RLHF/DPO 10% · Kubernetes 5%

JD QUALIFICATION RATE: 78-82%
```

Full gap analysis — 01 CAREER PACK §15

---

## Projects — Build Spec

### Project 1: RAG System → Production ✅ Built

| | |
|---|---|
| Folder | `archive/projects/rag_system/` |
| Status | ✅ FastAPI + Streamlit + eval dashboard built · ⏳ HF Spaces deploy pending |
| Stack | FastAPI + FAISS IndexFlatIP + sentence-transformers + Ollama/HF + Streamlit |
| Covers (% JDs) | RAG 51%, FAISS 38%, Embeddings 38%, FastAPI, Deployment |

### Project 2: LLM Fine-tuning → ⏳ In progress (Phase 4 code built, waiting for MacBook run)

| | |
|---|---|
| Folder | `code_practice/04_llms/07_qlora_train/` |
| Stack | TinyLlama + PEFT + Trainer (Mac MPS, no bitsandbytes) |
| Covers (% JDs) | LLM fine-tuning 36%, LoRA/QLoRA 15%, PEFT, HuggingFace 31% |

### Project 3: Document Agent → ⏳ Docs ready, code on request (Phase 6)

| | |
|---|---|
| Folder | `code_practice/06_agents/` |
| Stack | LangGraph + tools registry + memory + planner/executor |
| Covers (% JDs) | LangChain 46%, LangGraph mandatory at Infosys, Agents 23% |

---

## Daily Quick-Ref

| If you have... | Open |
|----------------|------|
| 30 min before applying somewhere | 01 CAREER PACK §18 + §6 (per-company prep + Top 12) |
| 1 day before any interview | 02 INTERVIEW PACK Part C (STAR rehearsal) + Part A relevant section |
| Time to study | 03 LEARNING PACK Part C (next session in sequence) |
| Need to look up a concept | 03 LEARNING PACK Part B (decision matrices) |
| Recruiter wants to negotiate | 01 CAREER PACK §11-13 (salary tables + scripts) |

---

## Archive

Old / superseded files in `archive/`: `01_JD_Analysis_skill_gaps,...`, `02_Career_Roadmap,...`, `03_ML_Theory_Learning_Checklist,...`, `04_interview_prep_behavioral,...`, `05_coding_practice_sequence,...`, `06_mental_model_LLMs_and_GenAI`, `handoff_brief_for_personal_macbook`. Content consolidated into the 4 files above.

# Hub — Sameer Khan | Daily Action Center

> **Open this every day.** Everything actionable is here. Deep reference — follow the links. Last updated: 25 June 2026

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
| Theory — all 11 folders | ✅ Complete · 7.rag + 8.agents theory gaps filled May 2026 |
| Code Phase 01 — Sequence Models | ✅ Run (9/9 on work laptop) |
| Code Phase 02 — Transformers | ✅ Run (11/11 on work laptop) |
| Code Phase 03 — Prompting | ✅ Run (10/10 on work laptop with Ollama) |
| Code Phase 04 — LLMs (legacy) | ✅ Run (14/14 on work laptop) |
| Code Phase 05 — Transformers (HF models) | ✅ All 7 run (GPU, GTX 1650 Ti, 25 Jun 2026) |
| Code Phase 06 — LLMs Core | ✅ All 3 run · Scripts updated with openai/claude/ollama PROVIDER flag |
| Code Phase 07 — RAG | ✅ All 5 run · faiss-cpu + rank-bm25 installed |
| Code Phase 08 — Agents / LangGraph | ✅ All 4 run · langchain-openai + langgraph installed |
| Code Phase 09 — Fine-tuning | ⏸ Parked — PyTorch 2.5.1 incompatible with trl 1.6 + safetensors (torch 2.6 not available for cu121) |
| Code Phase 10 — Document AI | 🔧 Code-built (4/4) · run next |
| RAG project (`archive/projects/rag_system/`) | ✅ Built: FastAPI + Streamlit + eval · ⏳ HF Spaces deploy pending |
| Resume | ✅ Updated: tagline, summary, skills, ICE bullets, RAG project |
| LLM fine-tune resume bullet | ⏸ On hold (Phase 09 parked) |
| LangGraph agent resume bullet | ✅ Ready to add — Phase 08/04 document agent ran successfully |
| Document AI pipeline resume bullet | ⏳ Add after Phase 10/04 runs + pin to GitHub |
| Job applications | ⏸ Hold — Phase 10 + portfolio milestones first |

---

## What To Do RIGHT NOW (priority order)

### 1. ✅ Run API sessions on Windows — COMPLETE (25 Jun 2026)

Phases 06, 07, 08 all run. All `_details.md` updated with actual output.

### 2. Run Phase 10 — Document AI ⏳ ACTIVE

- [ ] `code_practice/10_document_ai/` — run all 4 sessions
- [ ] **Session 04** `04_document_pipeline.py` → pin repo to GitHub → add resume bullet

### 3. Add LangGraph agent resume bullet — READY NOW

Phase 08/04 document agent ran. Copy bullet from "Personal Projects" section below → paste to resume.

### 4. Deploy RAG project to HuggingFace Spaces

- [ ] Set `LLM_PROVIDER=huggingface` + `HF_TOKEN` in Spaces secrets
- [ ] Test live link
- [ ] Add live demo link to resume + LinkedIn

### 4. Update LinkedIn (after #1 + #2 complete)

- [ ] Headline per audience — see 01 CAREER PACK §19
- [ ] About section — see 01 CAREER PACK §20
- [ ] Featured projects: pin RAG + document agent + pipeline

### 5. Start applying (after #1-4)

- [ ] Top 12 active matches first — see 01 CAREER PACK §6
- [ ] Tailor 2-3 lines of resume summary per role
- [ ] Weekly: 5 applications + 5 LinkedIn outreaches

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

## Personal Projects — add after running (code is ready, run to capture output)

**LLM Fine-tuning POC (2026)** &nbsp;&nbsp;&nbsp;&nbsp;→ run `code_practice/09_finetuning/01_lora_finetune.py` + `02_qlora_finetune.py`

LoRA fine-tuning of `facebook/opt-125m` on synthetic banking instruction dataset using PEFT + trl SFTTrainer. 0.24% trainable params (r=8). QLoRA variant: NF4 4-bit base + adapter — TinyLlama-1.1B in 700 MB vs 4.4 GB float32.

Add to resume after run: *"Fine-tuned LLM (LoRA + QLoRA) on synthetic banking dataset; 0.24% trainable params; QLoRA reduces memory 6× — TinyLlama-1.1B fits in 700 MB GPU"*

---

**Document Agent POC (2026)** &nbsp;&nbsp;&nbsp;&nbsp;✅ Run (25 Jun 2026)

LangGraph supervisor agent for mortgage document processing — classify → extract fields → retrieve policy → eligibility check → HITL manager approval (interrupt/resume) → decision report. 5 specialist agents wired with conditional routing.

**Add to resume now:** *"Built LangGraph mortgage document processing agent: classify → extract → policy RAG → eligibility → HITL approval → report; handles borderline cases via interrupt-based manager review"*

---

**Document AI Pipeline (2026)** &nbsp;&nbsp;&nbsp;&nbsp;→ run `code_practice/10_document_ai/04_document_pipeline.py` then pin to GitHub

Full ICE-style pipeline: PDF ingest (PyMuPDF) → OCR (EasyOCR) → document classification → field extraction → natural language QA. Production upgrade path: LayoutLMv3 → Donut → ColPali + RAG.

Add to resume after run + pin: *"End-to-end document AI pipeline (ingest → OCR → classify → extract → QA) — 8 years production experience now in open-source code"*

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

### Project 2: LLM Fine-tuning → ⏸ Parked (PyTorch 2.5.1 + trl 1.6 incompatibility on cu121)

| | |
|---|---|
| Folder | `code_practice/09_finetuning/` |
| Stack | opt-125m + PEFT LoRA + TRL SFTTrainer |
| Blocker | torch 2.6 not available for cu121; resume when cu124 wheel releases or upgrade to 3060+ |
| Covers (% JDs) | LLM fine-tuning 36%, LoRA/QLoRA 15%, PEFT, HuggingFace 31% |

### Project 3: Document Agent → ✅ Run (25 Jun 2026)

| | |
|---|---|
| Folder | `code_practice/08_agents/04_document_agent/` |
| Stack | LangGraph + 5 specialist agents + MemorySaver + HITL interrupt |
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

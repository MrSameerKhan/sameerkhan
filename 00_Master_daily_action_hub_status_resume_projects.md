# MASTER — Sameer Khan | Sr. ML Engineer
> Open this every day. Everything actionable is here. Deep reference → follow the links.
> Last updated: May 2026

---

## Table of Contents
- [Who You Are (Positioning)](#who-you-are-positioning)
- [Current Status](#current-status)
- [What To Do RIGHT NOW](#what-to-do-right-now-priority-order)
  - [1. Fix Resume](#1-fix-resume-before-any-application)
  - [2. Build RAG → Production](#2-build-project-1--rag--production-week-1)
  - [3. Build LLM Fine-tuning POC](#3-build-project-2--llm-fine-tuning-poc-week-2)
  - [4. Build Document Agent POC](#4-build-project-3--document-agent-poc-week-3)
  - [5. Apply](#5-apply-after-week-3)
- [Resume — ICE Data Services (ready to paste)](#resume--ice-data-services-ready-to-paste)
- [Resume — Technical Skills (replace current section)](#resume--technical-skills-replace-current-section)
- [Resume — Personal Projects (add this section)](#resume--personal-projects-add-this-section)
- [JD Coverage — Where You Stand](#jd-coverage--where-you-stand)
- [Projects — Build Spec](#projects--build-spec)
- [Reference Files](#reference-files)
- [Archive](#archive)

---

## Who You Are (Positioning)

> **"I build end-to-end LLM-powered document intelligence systems — from OCR and multimodal
> extraction through RAG pipelines, fine-tuned models, and production deployment on AWS and
> Databricks. In 8 years at ICE Data Services and Al Rajhi Bank, I have shipped systems that
> process hundreds of thousands of financial documents daily: 94% classification accuracy,
> 60% reduction in root-cause analysis time. My current focus: production RAG with hybrid
> retrieval and RAGAS evaluation, QLoRA fine-tuning, and LangGraph agentic workflows."**

8 years experience · ICE Data Services (direct) · Al Rajhi Bank · Hyderabad · Document AI specialist

---

## Your Competitive Edge
> Lead with these in every interview — most LLM engineers can't say any of them.

- **Production Document AI at scale** — 8 years, real financial documents, measurable outcomes (94% accuracy, 60% RCA reduction). Most LLM candidates have only personal projects. You have shipped.
- **Full pipeline ownership** — ingestion → preprocessing → distributed training (Horovod) → serving (SageMaker) → CI/CD (Jenkins+Docker) → monitoring. End-to-end depth is rare.
- **Financial domain expertise** — Al Rajhi Bank (Arabic documents, Saudi Arabia) + ICE Data Services (global financial markets). Domain knowledge = instant credibility in fintech and banking AI.
- **OCR + multimodal foundation** — CNN+LSTM+Transformer hybrid + Donut OCR-free extraction. Most LLM engineers skip directly to text. This is a genuine differentiator in Document AI roles.
- **Modern LLM stack already built** — Production RAG (FAISS + FastAPI + Streamlit), QLoRA fine-tuning, LangChain agents. Working code, not theory.

---

## Current Status

| Area | Status |
|---|---|
| Theory (ML → LLMs → MLOps) | ✅ Complete across all 9 folders |
| RAG project | ✅ Production built — FastAPI + Streamlit + eval dashboard · Deploy to HF Spaces pending |
| Resume | ✅ Updated — tagline, summary, skills, all 4 companies, RAG project added |
| LLM fine-tuning | ❌ Not built yet |
| LangChain / LangGraph agent | ❌ Not built yet |
| Job applications | ⏸ Hold — build projects + fix resume first |

---

## What To Do RIGHT NOW (priority order)

### 1. Fix Resume (before any application)
- [x] Fix broken sentence in Profile Summary
- [x] Replace Technical Skills section
- [x] Rewrite ICE Data Services bullets (combined entry, 4 bullets)
- [x] Rewrite Zyclyx / Al Rajhi Bank (3 bullets)
- [x] Rewrite VISSINDIA (2 bullets)
- [x] Add Personal Projects section — RAG System

### 2. Build Project 1 — RAG → Production (Week 1)
- [x] Add FastAPI backend (`/ingest`, `/query`, `/evaluate` endpoints)
- [x] Add Streamlit UI (upload docs, ask questions, see retrieved sources)
- [x] Add evaluation dashboard (MRR, P@K, latency)
- [ ] Deploy to HuggingFace Spaces — set `LLM_PROVIDER=huggingface` + `HF_TOKEN` in Spaces secrets
- [x] Update README with architecture + demo link

### 3. Build Project 2 — LLM Fine-tuning POC (Week 2)
- [ ] Set up Google Colab QLoRA training (Mistral-7B or Phi-3-mini)
- [ ] Train on public finance/document Q&A dataset
- [ ] Push adapter weights to HuggingFace Hub with model card
- [ ] Write local inference demo script
- [ ] Add to `projects/llm_finetuning/`

### 4. Build Project 3 — Document Agent POC (Week 3)
- [ ] Build LangGraph state machine (classify → retrieve → extract → answer)
- [ ] Wrap each step as LangChain tool
- [ ] Connect to Ollama llama3.2:1b locally
- [ ] Deploy Gradio UI on HuggingFace Spaces
- [ ] Add to `projects/doc_agent/`

### 5. Apply (after Week 3)
- See target companies → [02_Career_Roadmap](02_Career_Roadmap_companies_salary_tables_linkedin_networking.md)
- Fix LinkedIn headline and About section → [02_Career_Roadmap](02_Career_Roadmap_companies_salary_tables_linkedin_networking.md)

---

## Resume — ICE Data Services (ready to paste)

**ICE Data Services | Sr. ML Engineer | Aug 2021 – Present**
*(Aug 2021–Nov 2023 via Persistent Systems; Nov 2023–Present direct hire)*

- Architected a multimodal document classifier (CNN + BiLSTM + Transformer) for ICE's automated mortgage processing pipeline, achieving **93% page-level / 85% document-level accuracy** across 100+ document types; extended with BERT NER to improve page accuracy to **94%**, deployed to production on SageMaker.

- Designed and scaled a document deduplication pipeline across a 7.1M-page mortgage document bank — DonutSwin page embeddings, GPU-accelerated FAISS `IndexFlatIP` cosine search, and Union-Find transitive clustering deployed across 40 GPU nodes (TorchDistributor + NCCL); reduced training corpus by **67% pages and 69% documents**, eliminating template overfitting.

- Built a FAISS semantic search system using sentence-transformer embeddings for model prediction root-cause analysis, replacing keyword lookup and reducing investigation time by **60%** — establishing vector search as a core pattern in ICE's document AI stack.

- Owned MLOps infrastructure across the document AI pipeline — distributed training with Horovod (**−30% training time**), MLflow experiment tracking, Jenkins CI/CD, and Databricks + SageMaker production serving.

---

## Resume — Technical Skills (replace current section)

```
LLM & GenAI:  RAG pipelines, LLM fine-tuning (LoRA/QLoRA), Prompt Engineering,
              LangChain, LangGraph, FAISS, Sentence Transformers, HuggingFace
MLOps:        SageMaker, Databricks, Jenkins CI/CD, Docker, MLflow, Evidently
NLP & DL:     BERT, Transformers, NER, BiLSTM, CNN, PyTorch, TensorFlow
Cloud:        AWS (SageMaker, S3), Azure ML, Databricks, PySpark
Languages:    Python, SQL
```

---

## Resume — Personal Projects (add this section)

> Add to resume NOW (built + deployed):

```
RAG System (Production, 2026)
  End-to-end RAG pipeline — sentence-transformers (all-MiniLM-L6-v2), FAISS IndexFlatIP,
  FastAPI backend (/ingest, /query, /evaluate), Streamlit UI. MRR=1.0 on ML domain evaluation set.
  Live: huggingface.co/spaces/MrSameerKhan/rag-system
```

> Add to resume AFTER built (not yet — don't add placeholders):

```
LLM Fine-tuning POC (2026)              ← add after Project 2 is built
  QLoRA fine-tuning of Mistral-7B on finance Q&A dataset using PEFT + bitsandbytes.
  4-bit NF4 quantization, LoRA r=8. Model pushed to HuggingFace Hub.
  Model: huggingface.co/MrSameerKhan/mistral-finance-lora

Document Agent POC (2026)               ← add after Project 3 is built
  LangGraph multi-step agent for mortgage document processing — classify → extract → answer.
  LangChain tools + Ollama LLM + Gradio UI. Inspired by ICE document AI pipeline.
  Live: huggingface.co/spaces/MrSameerKhan/doc-agent
```

---

## JD Coverage — Where You Stand

### RIGHT NOW (RAG built, resume not yet updated):
```
JD QUALIFICATION RATE: ~30–35%   ← resume outdated, projects not on it yet
```

### After resume update only (no new projects needed):
```
JD QUALIFICATION RATE: ~50–55%   ← biggest single lever, do this first
```

### After all 3 projects built + resume updated:
```
COVERED (high frequency, all have proof)
  Python 97% · PyTorch/TF 79% · NLP 69% · Deep Learning 64% · Classical ML 64%
  SQL 54% · RAG 51% · MLOps 49% · LangChain/LangGraph 46% · Prompt Eng 38%
  Embeddings/FAISS 38% · LLM fine-tuning 36% · AWS SageMaker 38%
  PySpark/Databricks 31% · HuggingFace 31% · Agents 23% · LoRA/QLoRA 15%

STILL MISSING (low frequency, skip for now)
  AWS Bedrock 10% · Airflow 13% · GCP 10% · RLHF/DPO 10% · Kubernetes 5%

JD QUALIFICATION RATE: ~78–82%
```

Full gap analysis → [01_JD_Analysis](01_JD_Analysis_skill_gaps_resume_actions_project_priority.md)

---

## Projects — Build Spec

### Project 1: RAG System → Production ✅ Built
| | |
|---|---|
| Folder | `projects/rag_system/` |
| Status | ✅ FastAPI + Streamlit + eval dashboard built · ⏳ HuggingFace Spaces deploy pending |
| Stack | FastAPI + FAISS IndexFlatIP + sentence-transformers + Ollama/HF InferenceClient + Streamlit |
| Covers | RAG 51%, FAISS 38%, Embeddings 38%, Prompt Engineering 38%, FastAPI, Deployment |
| Interview Q | Why IndexFlatIP over IVF? Chunking strategy tradeoffs? How do you handle hallucinations? What does MRR=1.0 mean? |

### Project 2: LLM Fine-tuning → POC
| | |
|---|---|
| Folder | `projects/llm_finetuning/` (new) |
| Deploy | HuggingFace Hub (model + adapter weights) |
| Stack | QLoRA + PEFT + bitsandbytes + SFTTrainer (Colab T4 GPU) |
| Covers | LLM fine-tuning 36%, LoRA/QLoRA 15%, PEFT, HuggingFace 31% |
| Interview Q | LoRA math? Why QLoRA? Rank choice? When not to fine-tune? |

### Project 3: Document Agent → POC
| | |
|---|---|
| Folder | `projects/doc_agent/` (new) |
| Deploy | HuggingFace Spaces (Gradio) |
| Stack | LangGraph + LangChain + Ollama + Gradio |
| Covers | LangChain 46%, LangGraph mandatory, Agents 23%, Tool calling |
| Interview Q | LangGraph vs chains? Tool failure handling? Memory types? |

---

## Reference Files

| File | What's In It | When to Open |
|---|---|---|
| [JD Analysis](01_JD_Analysis_skill_gaps_resume_actions_project_priority.md) | 39 JDs analyzed, per-JD entries, frequency table, detailed gap analysis | Checking specific JD requirements |
| [Career Roadmap](02_Career_Roadmap_companies_salary_tables_linkedin_networking.md) | Target companies, salary tables, negotiation scripts, LinkedIn strategy | Before applying / negotiating |
| [Behavioral Prep](04_interview_prep_behavioral_star_answers_barraiser.md) | STAR answers for behavioral questions | 1 day before interview |
| [Learning Checklist](03_ML_Theory_Learning_Checklist_12_phases_progress_tracker.md) | Theory topics by phase, what's done/todo | If a theory gap appears in prep |
| [projects/rag_system/README.md](projects/rag_system/README.md) | RAG system runbook, how to run end to end | Running the RAG project |

---

## Archive
Past interview cheatsheets and completed audits → `archive/` folder (keep for reference, not daily use)

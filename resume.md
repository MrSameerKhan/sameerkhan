# SAMEER KHAN
Mobile: +91 80967 96963 | E-Mail: sameerkhanofficial9@gmail.com
GitHub: github.com/MrSameerKhan | LinkedIn: linkedin.com/in/mrsameerkhan3

**Senior ML Engineer | Document AI · LLM · RAG · Agentic AI | AWS SageMaker · Databricks**

---

## PROFILE SUMMARY

- Senior ML Engineer with **8 years** delivering production Document AI systems for financial services — **ICE Data Services** (Hyderabad) and **Al Rajhi Bank** (Riyadh, Saudi Arabia).

- Shipped a multimodal document classifier (CNN + BiLSTM + Transformer) achieving **94% page-level accuracy** on **100K+ mortgage documents**; built a FAISS vector search pipeline that reduced root-cause analysis time by **60%**.

- Specialised in LLM-powered document intelligence: RAG pipelines (FAISS + FastAPI + Streamlit), QLoRA fine-tuning (Mistral-7B), LangChain agents, and LangGraph agentic workflows.

- End-to-end pipeline ownership — multimodal extraction → distributed training (Horovod, **−30% training time**) → AWS SageMaker + Databricks deployment → MLflow monitoring.

---

## TECHNICAL SKILLS

| Category | Skills |
|---|---|
| **LLM & GenAI** | RAG pipelines, LLM fine-tuning (LoRA/QLoRA), Prompt Engineering, LangChain, LangGraph, FAISS, Sentence Transformers, HuggingFace |
| **MLOps** | SageMaker, Databricks, Jenkins CI/CD, Docker, MLflow, FastAPI, Evidently |
| **NLP & DL** | BERT, Transformers, NER, BiLSTM, CNN, PyTorch, TensorFlow, Scikit-learn |
| **Cloud** | AWS (SageMaker, S3), Azure ML, Databricks, PySpark |
| **Languages** | Python, SQL |

---

## WORK EXPERIENCE

### **Aug 2021 – Present | ICE Data Services, Hyderabad, India | Sr. ML Engineer**
*(Aug 2021–Nov 2023 via Persistent Systems; Nov 2023–Present direct hire)*

- Architected a CNN + BiLSTM + Transformer document classifier for ICE's mortgage pipeline — **93% page-level / 85% document-level accuracy** across **100+ document types**; integrated BERT NER to push accuracy to **94%**, deployed on SageMaker.

- Deployed **Donut LLM** as an OCR-free document extraction system, enabling direct text retrieval from mortgage documents without traditional OCR — reducing preprocessing complexity and improving extraction reliability.

- Scaled a deduplication pipeline across **7.1M mortgage pages** — DonutSwin embeddings, FAISS IndexFlatIP cosine search, Union-Find clustering on **40 GPU nodes** (TorchDistributor + NCCL); cut training corpus by **67% pages and 69% documents**, eliminating template overfitting.

- Built a FAISS semantic search system for model root-cause analysis, replacing keyword lookup and cutting investigation time by **60%**.

- Owned MLOps across the document AI pipeline — Horovod distributed training (**−30% training time**), MLflow, Jenkins CI/CD, Databricks + SageMaker serving.

---

### **Aug 2019 – Mar 2021 | Zyclyx (Client: Al Rajhi Bank, Riyadh, Saudi Arabia) | Data Scientist**

- Designed a ResNet50 document classification pipeline for English-Arabic banking documents — categorised incoming files by type to route each to the correct extraction workflow.

- Built Tesseract + TensorFlow OCR pipeline for key-field extraction from English-Arabic financial forms; developed GRU-based NER to identify customer, transaction, and compliance entities; integrated output with bank APIs for automated database validation.

- Built a Signature Verification System using deep learning to detect fraudulent and mismatched signatures in offline banking documents, reducing manual compliance review overhead.

- Deployed the full pipeline on-premise via Flask + Gunicorn microservices — from raw Arabic document ingestion to structured API output for production banking workflows.

---

### **Feb 2017 – Aug 2019 | VISSINDIA, Hyderabad | Python & Computer Vision Engineer**

- Built identity verification and multi-object tracking pipelines — face detection (Haar Cascade, HOG, CNN-based), feature matching (SIFT/SURF/FLANN), and TensorFlow object detection for real-time video tracking.

- Developed image preprocessing pipelines (morphological ops, denoising, thresholding, skew correction) to improve OCR accuracy — foundation carried into Document AI work at Al Rajhi Bank and ICE Data Services.

---

## PROJECTS

**RAG System — Production (2026)**
End-to-end RAG pipeline — sentence-transformers (all-MiniLM-L6-v2), FAISS IndexFlatIP, FastAPI backend (/ingest, /query, /evaluate), Streamlit UI. MRR=1.0 on ML domain evaluation set.
GitHub: github.com/MrSameerKhan

**Vector Search RCA Tool (ICE Data Services)**
Designed and deployed an embedding-based semantic search pipeline using production ensemble model vectors, replacing manual keyword search and reducing root-cause analysis time by 60%.

**BERT & Donut LLM — OCR and OCR-Free Models (ICE Data Services)**
Developed a Donut LLM-based extraction system as an OCR-free alternative, enabling direct document text retrieval without traditional OCR.

---

## CERTIFICATIONS & TRAINING

- Advanced Deep Learning Specialization – Coursera (CNNs, RNNs, LSTMs, Sequence Models, Optimization)
- Machine Learning Operations (MLOps) Specialization – Coursera (CI/CD, deployment, monitoring, ML pipelines)

---

## ACADEMIC DETAILS

B.Tech. in Computer Science Engineering — TKR Engineering College, JNTU, Hyderabad | 2016

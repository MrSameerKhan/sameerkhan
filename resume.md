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

- Architected a multimodal document classifier (CNN + BiLSTM + Transformer) for ICE's automated mortgage processing pipeline, achieving **93% page-level / 85% document-level accuracy** across **100+ document types**; extended with BERT NER to improve page accuracy to **94%**, deployed to production on SageMaker.

- Designed and scaled a document deduplication pipeline across a **7.1M-page** mortgage document bank — DonutSwin page embeddings, GPU-accelerated FAISS IndexFlatIP cosine search, and Union-Find transitive clustering deployed across **40 GPU nodes** (TorchDistributor + NCCL); reduced training corpus by **67% pages and 69% documents**, eliminating template overfitting.

- Built a FAISS semantic search system using sentence-transformer embeddings for model prediction root-cause analysis, replacing keyword lookup and reducing investigation time by **60%** — establishing vector search as a core pattern in ICE's document AI stack.

- Owned MLOps infrastructure across the document AI pipeline — distributed training with Horovod (**−30% training time**), MLflow experiment tracking, Jenkins CI/CD, and Databricks + SageMaker production serving.

---

### **Aug 2019 – Mar 2021 | Zyclyx (Client: Al Rajhi Bank, Riyadh, Saudi Arabia) | Data Scientist**

- Designed and deployed an end-to-end document intelligence pipeline for English-Arabic financial documents — ResNet50 document classification, Tesseract + TensorFlow OCR for key-field extraction, and GRU-based NER to identify customer, transaction, and compliance entities; integrated extracted records with bank APIs for automated database validation.

- Built a Signature Verification System using deep learning to detect fraudulent and mismatched signatures in offline banking documents, reducing manual compliance review overhead.

- Deployed all ML models on-premise via Flask + Gunicorn microservices — full pipeline from raw Arabic document ingestion to structured API output for production banking workflows.

---

### **Feb 2017 – Aug 2019 | VISSINDIA, Hyderabad | Python & Computer Vision Engineer**

- Built computer vision pipelines for identity verification — face detection (Haar Cascade, HOG, CNN-based), feature matching (SIFT, SURF, FLANN), and TensorFlow object detection for multi-object tracking in images and video.

- Developed image preprocessing pipelines (morphological operations, denoising, thresholding, skew correction) to improve OCR accuracy — foundational to the Document AI work carried forward at Al Rajhi Bank and ICE Data Services.

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

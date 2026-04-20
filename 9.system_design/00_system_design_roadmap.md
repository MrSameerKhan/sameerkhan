# ML System Design — Roadmap & Interview Guide

---

## Folder Map

```
9.system_design/
├── 00_system_design_roadmap.md           ← you are here
├── 01_ml_system_design_framework.md      ← universal framework for any system design
├── 02_recommendation_system.md           ← two-tower, ANN, A/B testing
├── 03_search_and_rag_system.md           ← FAISS, BM25, hybrid, reranking
├── 04_document_processing_pipeline.md    ← OCR, extraction, validation, confidence
├── 05_llm_agent_system_design.md         ← ReAct, tool use, memory, multi-agent
├── 06_question_answering_system.md       ← multi-turn, streaming, citations
├── 07_feed_ranking_system.md             ← candidate gen, LTR, freshness, diversity
└── 08_content_moderation_system.md       ← pre-filter, ML classifiers, policy engine
```

---

## Universal Design Framework (use for every question)

```
1. Clarify requirements (2 min)
   - Scale: users, QPS, data volume, latency SLA
   - Functional: what must the system do?
   - Non-functional: availability, consistency, cost

2. High-level architecture (3 min)
   - Offline pipeline (training, indexing)
   - Online pipeline (serving, real-time)
   - Data stores (feature store, model store, cache)

3. Deep dive on ML components (10 min)
   - Data: collection, labeling, features
   - Model: architecture, training objective, evaluation
   - Serving: latency, throughput, scaling

4. Failure modes & mitigations (3 min)
   - Cold start, data drift, feedback loops
   - Monitoring, rollback plan

5. Metrics (2 min)
   - Offline: AUC, NDCG, F1
   - Online: CTR, conversion, session length
```

**Full framework:** `01_ml_system_design_framework.md`

---

## System Design Cheat Sheet

| System | Retrieval | Ranking | Key challenge |
|--------|-----------|---------|---------------|
| Recommendation | Two-tower + FAISS | LightGBM LTR | Feedback loop, cold start |
| Search / RAG | BM25 + dense hybrid | Cross-encoder reranker | Chunk size, recall vs precision |
| Feed ranking | Social graph + ANN | LambdaRank | Freshness vs relevance, diversity |
| QA system | Bi-encoder + FAISS | Cross-encoder | Multi-turn context, "I don't know" |
| Content moderation | Blocklist + hash | ML classifiers | Precision vs recall per category |
| Document processing | OCR + layout | LayoutLM / Donut | OCR errors, multi-page |
| LLM agent | Tool use | LLM planner | Infinite loops, token budget |

---

## Interview Priority

**Most common system design questions (ranked):**
1. Recommendation system (Netflix, Amazon, LinkedIn)
2. Search system / RAG (semantic search, chatbot)
3. Feed ranking (Twitter, LinkedIn, YouTube)
4. LLM-powered product (QA bot, code assistant)
5. Content moderation (platform trust & safety)
6. Document processing (fintech, legal, enterprise)

**Always cover in your answer:**
- Scale numbers (QPS, data size, latency budget)
- Multi-stage funnel (retrieval → ranking → reranking)
- Cold start handling
- Monitoring + feedback loop
- A/B testing for rollout

---

## Latency Budgets (memorize these)

```
Bi-encoder retrieval (FAISS, 10M docs):     <5ms
Cross-encoder reranking (100 candidates):   ~200ms
LLM generation (first token):               ~200ms
LLM generation (full response, 200 tokens): ~3s streaming
Feature store read (Redis):                 <2ms
Model inference (sklearn/ONNX, batch=1):    1–15ms
```

---

## Scale Numbers (memorize these)

```
Social feed:    500M users, 10M posts/day,  100K QPS
E-commerce rec: 100M users, 10M items,      100K QPS
Search engine:  1B queries/day,             100K QPS
Content mod:    10M posts/day,              ~100 posts/second
QA bot:         50K concurrent users,       10K QPS
```

---

## Connections

| Concept | Where it's explained |
|---------|---------------------|
| FAISS index types | `03_search_and_rag_system.md` |
| Two-tower retrieval | `02_recommendation_system.md` |
| LoRA for LLM fine-tuning in products | `5.transformers/02_models/09_parameter_efficient_tuning.md` |
| Model monitoring | `8.mlops/09_monitoring_end_to_end.md` |
| A/B testing | `02_recommendation_system.md` §A/B Testing |
| Serving optimization | `8.mlops/10_serving_optimization_end_to_end.md` |

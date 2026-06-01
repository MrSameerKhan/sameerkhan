# 7. RAG

Scope: RAG-specific patterns, pipeline depth, threat models.

```mermaid
mindmap
  root((7. RAG))
    RAG architecture
      Indexing offline
        chunk → embed → vector DB + BM25
      Retrieval online
        dense ANN + BM25 → RRF → reranker → LLM
      Advanced patterns
        HyDE · query decomposition · Self-RAG · CRAG
    Pipeline depth
      Chunking strategy decision tree
      Dense vs BM25 vs hybrid
      Cross-encoder reranking
    Security
      Indirect prompt injection
      6 defense layers · CaMeL dual-LLM
``` Embedder / retriever theory lives in `../4.nlp/02_embeddings/`; production RAG ops in `../10.mlops/13_production_rag_ops.md`; multi-tenant RAG system design in `../11.system_design/10_multi_tenant_rag.md`. Tier 2 (Theory).

---

## Reading Order

1. `01_rag.md` — conceptual RAG (architecture, chunking, retrieval, generation patterns)
2. `01b_rag_end_to_end.md` — worked example with numbers + RAGAS evaluation
3. `02_rag_pipeline.md` — pipeline depth: chunking decisions, hybrid retrieval (RRF), cross-encoder reranking
4. `03_indirect_prompt_injection.md` — threat model + 6 defense layers (the #1 RAG security concern)
5. `04_advanced_rag.md` — query transformation: HyDE, multi-query, Self-RAG, CRAG, Adaptive RAG
6. `05_rag_evaluation.md` — RAGAS 4 metrics in depth, retrieval eval, LLM-as-judge, synthetic datasets
7. `06_production_rag.md` — semantic cache, incremental indexing, cost model, A/B testing RAG versions

---

## Folder TOC

| File | Owns |
|------|------|
| `01_rag.md` | Conceptual RAG architecture + overview of advanced patterns |
| `01b_rag_end_to_end.md` | Worked example — chunking → embedding → retrieval → reranking → generation |
| `02_rag_pipeline.md` | SSOT: chunking strategies, hybrid retrieval (RRF code), cross-encoder reranking, embedding fine-tuning |
| `03_indirect_prompt_injection.md` | SSOT: Indirect prompt injection threat + 6 defense layers (capability isolation, structured outputs, CaMeL dual-LLM) |
| `04_advanced_rag.md` | SSOT: Query transformation — HyDE, multi-query, query decomposition, step-back, Self-RAG, CRAG, Adaptive RAG |
| `05_rag_evaluation.md` | SSOT: RAGAS 4 metrics in depth, retrieval-only eval (Recall@k / MRR), LLM-as-judge, synthetic dataset creation |
| `06_production_rag.md` | SSOT: Semantic cache (two-tier), incremental index freshness, cost model, A/B testing RAG versions |

---

## SSOT Topics Owned Here

- RAG conceptual architecture → `01_rag.md`
- RAG pipeline mechanics (chunking, RRF, reranking) → `02_rag_pipeline.md`
- Indirect prompt injection defenses → `03_indirect_prompt_injection.md`
- Query transformation (HyDE, multi-query, Self-RAG, CRAG) → `04_advanced_rag.md`
- RAG evaluation (RAGAS, LLM-as-judge, synthetic datasets) → `05_rag_evaluation.md`
- Production RAG (semantic cache, freshness, cost, A/B) → `06_production_rag.md`

---

## Connections

- **Modern embedders** (BGE / E5 / Nomic / jina-v3 / mxbai): `../4.nlp/02_embeddings/02_sentence_embeddings.md`
- **Hybrid retrieval** (BM25 + dense + RRF + reranker): `../4.nlp/02_embeddings/05_semantic_similarity.md`
- **Embedder training** (contrastive, hard negatives): `../4.nlp/02_embeddings/06_contrastive_training.md`
- **RAGAS / lm-eval-harness**: `../4.nlp/04_applications/04_evaluation_metrics.md`
- **Structured extraction** (Pydantic + Instructor): `../4.nlp/04_applications/03_information_extraction.md`
- **Constrained decoding** (defense layer 5): `../5.transformers/02_models/12_constrained_decoding.md`
- **Document AI + ColPali** (visual retrieval): `../9.multimodal/02_document_ai.md`
- **Production RAG ops** (drift, semantic cache, freshness): `../10.mlops/13_production_rag_ops.md`
- **LLM prompting** (calling site): `../6.llms/01_prompting.md`
- **Agents that use RAG**: `../8.agents/`
- **Multi-tenant RAG system design**: `../11.system_design/10_multi_tenant_rag.md`

---

## Practice

- RAG pipeline (10 sessions, all docs complete) — `../code_practice/05_rag/`
- Active resume project: `../archive/projects/rag_system/` — FastAPI + Streamlit + eval

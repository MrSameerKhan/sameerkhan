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
3. `02_rag_pipeline.md` — pipeline depth + RAGAS metrics + production considerations
4. `03_indirect_prompt_injection.md` — threat model + 6 defense layers (this is the #1 RAG security concern)

---

## Folder TOC

| File | Owns |
|------|------|
| `01_rag.md` | Conceptual RAG architecture + Self-RAG / CRAG / Adaptive RAG patterns |
| `01b_rag_end_to_end.md` | Worked example — chunking → embedding → retrieval → reranking → generation |
| `02_rag_pipeline.md` | RAG pipeline depth + RAGAS metrics + production considerations |
| `03_indirect_prompt_injection.md` | SSOT: Indirect prompt injection threat + 6 defense layers (capability isolation, structured outputs, CaMeL dual-LLM) |

---

## SSOT Topics Owned Here

- RAG conceptual + advanced patterns (Self-RAG / CRAG / Adaptive RAG) → `01_rag.md`
- RAG pipeline depth → `02_rag_pipeline.md`
- Indirect prompt injection defenses → `03_indirect_prompt_injection.md`

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

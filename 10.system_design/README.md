# 11. System Design

Scope: end-to-end ML system design — recommendation, search/RAG, agents, document processing, multi-tenant, eval systems. Implementation depth lives in other tier-2 folders; this is the system-level lens. **Tier: 2 (Theory).**

---

## Reading Order

| If you're learning... | Read in order |
|---|---|
| System design fundamentals | `00_system_design_roadmap` → `01_ml_system_design_framework` |
| Classical ML systems | `02_recommendation_system` → `07_feed_ranking_system` → `08_content_moderation_system` |
| LLM-era systems | `03_search_and_rag_system` → `05_llm_agent_system_design` → `06_question_answering_system` → `04_document_processing_pipeline` |
| Cross-cutting concerns | `09_tool_authorization_patterns` → `10_multi_tenant_rag` → `11_llm_evaluation_systems` |

---

## Folder TOC

| File | Owns |
|---|---|
| `00_system_design_roadmap.md` | Navigation + the universal 5-step design process |
| `01_ml_system_design_framework.md` | The 8-step universal framework detailed |
| `02_recommendation_system.md` | Two-tower, ANN, A/B testing |
| `03_search_and_rag_system.md` | FAISS, BM25, hybrid retrieval at system scale |
| `04_document_processing_pipeline.md` | OCR → extraction → validation (Sameer's domain) |
| `05_llm_agent_system_design.md` | ReAct, tool use, memory, multi-agent at system level |
| `06_question_answering_system.md` | Multi-turn QA, streaming, citations |
| `07_feed_ranking_system.md` | Candidate gen, LTR, freshness, diversity |
| `08_content_moderation_system.md` | Pre-filter, ML classifiers, policy engine |
| `09_tool_authorization_patterns.md` | **SSOT:** capability isolation, allowlists, HITL gates |
| `10_multi_tenant_rag.md` | **SSOT:** isolation models, SLAs, cache namespacing, compliance |
| `11_llm_evaluation_systems.md` | **SSOT:** golden sets, online vs offline eval, eval pipelines |

---

## SSOT Topics Owned Here

- Tool authorization patterns (capability isolation depth) → `09_tool_authorization_patterns.md`
- Multi-tenant RAG (isolation, SLAs, compliance) → `10_multi_tenant_rag.md`
- LLM evaluation systems (golden sets, online eval pipelines) → `11_llm_evaluation_systems.md`

---

## Connections

- **RAG patterns**: `../7.rag/`
- **Agent patterns**: `../8.agents/` — agent system design here builds on those
- **Indirect prompt injection** (the threat tool-auth defends against): `../7.rag/03_indirect_prompt_injection.md`
- **Agent reliability patterns** (the implementation side of `09_tool_authorization`): `../8.agents/02_agent_reliability_patterns.md`
- **Production RAG ops** (the MLOps side of `10_multi_tenant_rag`): `../10.mlops/13_production_rag_ops.md`
- **Eval frameworks** (MTEB / RAGAS / lm-eval-harness — referenced from `11_llm_evaluation_systems`): `../4.nlp/09_applications/08_evaluation_metrics.md`
- **Model evaluation theory** (conformal prediction): `../1.fundamentals/04_model_evaluation/`
- **LLM observability**: `../10.mlops/11_llm_observability.md`
- **Recommendation algorithms**: `../1.machine_learning/`

---

## Practice

System design is interview-focused, not code-aligned. Practice lives in:
- `../code_practice/05_rag/` (Phase 5, docs connect to `10_multi_tenant_rag`)
- `../code_practice/06_agents/` (Phase 6, docs connect to `09_tool_authorization`, `07_multi_agent` for orchestration)

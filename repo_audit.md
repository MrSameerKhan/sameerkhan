# Repo Audit — Updated (April 2026)

> ✅ = verified complete. ❌ = still missing. Code gaps tracked separately — theory first.

---

## 1. machine_learning/

**Overall: Solid — 1 theory file missing, 1 file shallow**

| Gap Type | Detail | Status |
|---|---|---|
| Missing file | `02_algorithms/06_semi_supervised_learning.md` — co-training, self-training, pseudo-labeling | ❌ |
| Created | `02_algorithms/07_reinforcement_learning.md` — MDP, Q-Learning, SARSA, Policy Gradient, REINFORCE | ✅ |
| Incomplete | `02_algorithms/04_probabilistic.md` — shallow; no worked examples for Naive Bayes/HMM | ❌ |
| Addressed | `01_fundamentals/01_statistics_foundations.md` — p-value + distributions now covered in `01c_statistics_end_to_end.md` | ✅ |
| Addressed | Time series end-to-end — now covered in `05b_time_series_end_to_end.md` | ✅ |

---

## 2. deep_learning/

**Overall: Excellent — 1 theory file missing**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `02_architectures/00_architecture_comparison.md` — MLP vs CNN vs RNN vs Transformer decision guide | ✅ |
| Created | `02_architectures/07_reinforcement_learning.md` — DQN, A3C, PPO, RLHF with full dry-runs | ✅ |
| Missing file | `02_architectures/08_semi_supervised.md` — pseudo-labeling, consistency regularization | ❌ |
| Verified complete | All 7 Key Takeaway / conclusion sections in fundamentals + architectures | ✅ |

---

## 3. computerVision/

**Overall: Good — no critical theory gaps, improvements possible**

| Gap Type | Detail | Status |
|---|---|---|
| Verified complete | `01_cnn_mechanics.md` and `02_cnn_architectures.md` — both have full content + Key Takeaway | ✅ |
| Improvement | No end-to-end trace files (all 6 are reference-only; NLP pattern would add depth) | ❌ |
| Improvement | Data augmentation scattered — could be standalone file | ❌ |
| Improvement | No consolidated CV evaluation metrics file | ❌ |

---

## 4. nlp/

**Overall: Good — 5 theory files missing**

| Gap Type | Detail | Status |
|---|---|---|
| Missing file | `02_embeddings/02_sentence_embeddings.md` — reference file (only end-to-end exists) | ❌ |
| Missing file | `02_embeddings/03_tokenization.md` — reference file (only end-to-end exists) | ❌ |
| Missing file | Decoding strategies file — beam search, top-k, nucleus sampling, temperature | ❌ |
| Missing file | Semantic similarity / STS / retrieval file — directly relevant to RAG | ❌ |
| Missing file | Machine translation reference | ❌ |
| Incomplete | `04_applications/02_ner_and_tagging.md` and `03_information_extraction.md` — completeness not verified | ❌ |

---

## 5. transformers/

**Overall: Solid core — 5 theory files missing**

| Gap Type | Detail | Status |
|---|---|---|
| Missing file | `00_roadmap.md` — no navigation index | ❌ |
| Missing file | `01_fundamentals/05_vision_transformers.md` — ViT, DINO, DeiT | ❌ |
| Missing file | `02_models/09_parameter_efficient_tuning.md` — LoRA, QLoRA, prefix tuning, adapters | ❌ |
| Missing file | Multimodal transformers — CLIP, BLIP, LLaVA architectures | ❌ |
| Missing file | Mixture of Experts (MoE) — GPT-4, Mixtral architecture | ❌ |
| Incomplete | `02_models/04_efficient_transformers.md` — Flash Attention, Paged Attention, KV cache depth needs verification | ❌ |

---

## 6. llms/

**Overall: Very Complete — minor numerical gaps**

| Gap Type | Detail | Status |
|---|---|---|
| Incomplete | `07_finetuning_end_to_end.md` — full LoRA matrix update trace (W_A, W_B gradients step-by-step) missing | ❌ |
| Incomplete | `08_rag_end_to_end.md` — explicit BM25 score computation, cosine similarity values, reranking scores as numbers missing | ❌ |
| Incomplete | `09_agents_end_to_end.md` — token-count-per-iteration and max-iteration cutoff scenario not shown | ❌ |
| Incomplete | `10_alignment_end_to_end.md` — PPO advantage estimation and policy gradient update not traced numerically | ❌ |

---

## 7. multimodal/

**Overall: Complete ✅**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `00_roadmap.md` — navigation index, decision guide, key numbers | ✅ |
| Created | `03_vision_transformers.md` — ViT, Swin, DeiT, DINO with full dry-run | ✅ |
| Created | `04_clip_finetuning_end_to_end.md` — contrastive loss, 4×4 similarity matrix, InfoNCE | ✅ |
| Created | `05_donut_end_to_end.md` — Swin encoder, autoregressive decoder, teacher forcing | ✅ |
| Created | `06_layoutlm_end_to_end.md` — bbox normalization, IOB trace, WPA pre-training | ✅ |
| Deepened | `02_document_ai.md` — LayoutLM v1/v2/v3 evolution, bbox details, Nougat section added | ✅ |

---

## 8. mlops/

**Overall: End-to-end traces complete — reference files partially incomplete**

| Gap Type | Detail | Status |
|---|---|---|
| Missing file | `00_roadmap.md` — no priority guide | ❌ |
| Created | `08_model_registry_end_to_end.md` — MLflow register → stage → validate → deploy → rollback | ✅ |
| Created | `09_monitoring_end_to_end.md` — drift detection (KS/PSI) → alert → retrain trigger | ✅ |
| Created | `10_serving_optimization_end_to_end.md` — PyTorch → ONNX → INT8 → benchmark + vLLM | ✅ |
| Verified complete | `03_monitoring_and_drift.md` — has Gotchas + Interview Q&A | ✅ |
| Incomplete | `02_serving_and_inference.md` — vLLM (PagedAttention, continuous batching) depth needs verification | ❌ |
| Incomplete | `04_pipelines_and_infra.md` — Airflow DAG, Kubeflow, CI/CD, canary/shadow missing | ❌ |

---

## 9. system_design/

**Overall: Core files complete — 4 files missing**

| Gap Type | Detail | Status |
|---|---|---|
| Missing file | `00_system_design_roadmap.md` — no interview priority guide | ❌ |
| Missing file | `06_question_answering_system.md` — multi-turn QA, streaming, context management | ❌ |
| Missing file | `07_feed_ranking_system.md` — personalized ranking, freshness, diversity | ❌ |
| Missing file | `08_content_moderation_system.md` — safety classification, enforcement pipeline | ❌ |
| Verified complete | `03_search_and_rag_system.md` — full pipeline, reranking, evaluation, Q&A, Key Takeaway | ✅ |
| Verified complete | `04_document_processing_pipeline.md` — confidence scoring, human review, Q&A, Key Takeaway | ✅ |
| Verified complete | `05_llm_agent_system_design.md` — Tool Registry, state management, cost tracking, Q&A | ✅ |
| Deepened | `02_recommendation_system.md` — A/B testing section added (sample size, z-test dry run, pitfalls) | ✅ |

---

## 10. projects/

**Overall: Raw notebooks only — all capstone projects missing (code phase)**

| Gap Type | Detail |
|---|---|
| Missing | RAG system project (`projects/rag_system/`) |
| Missing | LLM fine-tuning project — Mistral/LLaMA fine-tuned (`projects/llm_finetuning/`) |
| Missing | LangChain document agent (`projects/doc_agent/`) |
| Missing | vLLM + Evidently monitoring (`projects/vllm_monitoring/`) |
| Missing | LayoutLM fine-tuned on documents (`projects/layoutlm_docs/`) |
| Organization | No README in any existing project subfolder |

---

## Priority Summary — Current Work Order

### Theory Phase (active now)

| Priority | Action | Folder | Status |
|---|---|---|---|
| **High** | `06_layoutlm_end_to_end.md` — LayoutLM v3 full trace | `7.multimodal/` | ✅ |
| **High** | `05_donut_end_to_end.md` — Donut doc parsing trace | `7.multimodal/` | ✅ |
| **High** | `04_clip_finetuning_end_to_end.md` — contrastive loss trace | `7.multimodal/` | ✅ |
| **High** | `03_vision_transformers.md` — ViT in multimodal context | `7.multimodal/` | ✅ |
| **High** | Deepen `02_document_ai.md` — LayoutLM v3, Donut, Nougat | `7.multimodal/` | ✅ |
| **High** | `07_reinforcement_learning.md` — MDP, Q-Learning, Policy Gradient | `1.machine learning/` | ✅ |
| **High** | `07_reinforcement_learning.md` — DL perspective (DQN, PPO, A3C) | `2.deep learning/` | ✅ |
| **High** | `00_architecture_comparison.md` — MLP vs CNN vs RNN vs Transformer | `2.deep learning/` | ✅ |
| **High** | Complete `02_recommendation_system.md` — A/B testing section added | `9.system_design/` | ✅ |
| **High** | `08_model_registry_end_to_end.md` — MLflow register → stage → deploy | `8.mlops/` | ✅ |
| **High** | `09_monitoring_end_to_end.md` — drift detection → alert → retrain | `8.mlops/` | ✅ |
| **High** | `10_serving_optimization_end_to_end.md` — PyTorch → ONNX → INT8 → vLLM | `8.mlops/` | ✅ |
| **Medium** | `06_semi_supervised_learning.md` | `1.machine learning/` | ❌ |
| **Medium** | `08_semi_supervised.md` | `2.deep learning/` | ❌ |
| **Medium** | `09_parameter_efficient_tuning.md` — LoRA, QLoRA, adapters | `5.transformers/` | ❌ |
| **Medium** | `01_fundamentals/05_vision_transformers.md` | `5.transformers/` | ❌ |
| **Medium** | MoE file — GPT-4, Mixtral | `5.transformers/` | ❌ |
| **Medium** | Decoding strategies file | `4.nlp/` | ❌ |
| **Medium** | Semantic similarity / STS file | `4.nlp/` | ❌ |
| **Medium** | `06_question_answering_system.md` | `9.system_design/` | ❌ |
| **Medium** | `07_feed_ranking_system.md` | `9.system_design/` | ❌ |
| **Medium** | `08_content_moderation_system.md` | `9.system_design/` | ❌ |
| **Medium** | Fill numerical gaps in LLM end-to-end files | `6.llms/` | ❌ |
| **Low** | `00_roadmap.md` files for transformers, mlops, system_design | Multiple | ❌ |
| **Low** | `02_sentence_embeddings.md` + `03_tokenization.md` reference files | `4.nlp/` | ❌ |
| **Low** | Deepen `04_probabilistic.md` | `1.machine learning/` | ❌ |
| **Low** | CV end-to-end trace files | `3.computerVision/` | ❌ |

### Code Phase (after theory complete)

| Priority | Action |
|---|---|
| Critical | Build capstone projects (RAG, fine-tuning, agent, vLLM, LayoutLM) |
| High | Code scripts: data drift, Gaussian ML, hypothesis testing |
| High | Code: CLIP, BLIP-2, LayoutLM, Donut examples |
| High | Code: W&B/MLflow, FastAPI+vLLM, drift detection |
| Medium | Reorganize `03_code/` directories in ML + DL |
| Low | Add README to each project subfolder |

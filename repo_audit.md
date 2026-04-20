# Repo Audit — Updated (April 2026)

> ✅ = verified complete. ❌ = still missing. Code gaps tracked separately — theory first.

---

## 1. machine_learning/

**Overall: Complete ✅**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `02_algorithms/06_semi_supervised_learning.md` — self-training (τ=0.90 dry-run), FixMatch, label propagation | ✅ |
| Created | `02_algorithms/07_reinforcement_learning.md` — MDP, Q-Learning, SARSA, Policy Gradient, REINFORCE | ✅ |
| Deepened | `02_algorithms/04_probabilistic.md` — NB spam dry-run + HMM section (Viterbi, Forward, Baum-Welch) | ✅ |
| Addressed | `01_fundamentals/01_statistics_foundations.md` — p-value + distributions now covered in `01c_statistics_end_to_end.md` | ✅ |
| Addressed | Time series end-to-end — now covered in `05b_time_series_end_to_end.md` | ✅ |

---

## 2. deep_learning/

**Overall: Complete ✅**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `02_architectures/00_architecture_comparison.md` — MLP vs CNN vs RNN vs Transformer decision guide | ✅ |
| Created | `02_architectures/07_reinforcement_learning.md` — DQN, A3C, PPO, RLHF with full dry-runs | ✅ |
| Created | `02_architectures/08_semi_supervised.md` — SimCLR NT-Xent dry-run, MAE 75% masking, BERT as SSL | ✅ |
| Verified complete | All 7 Key Takeaway / conclusion sections in fundamentals + architectures | ✅ |

---

## 3. computerVision/

**Overall: Good — 2 optional improvements deferred to code phase**

| Gap Type | Detail | Status |
|---|---|---|
| Verified complete | `01_cnn_mechanics.md` and `02_cnn_architectures.md` — both have full content + Key Takeaway | ✅ |
| Created | `01_fundamentals/03_cnn_end_to_end.md` — full forward pass + backprop dry run, augmentation, CV metrics, interview Q&A | ✅ |
| Deferred | Data augmentation standalone file — low priority, covered in end-to-end | ❌ (deferred) |
| Deferred | Consolidated CV evaluation metrics file — low priority, covered in end-to-end | ❌ (deferred) |

---

## 4. nlp/

**Overall: Mostly complete — 2 lower-priority gaps remain**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `02_embeddings/02_sentence_embeddings.md` — SBERT, bi-encoder vs cross-encoder, MNR loss, FAISS | ✅ |
| Created | `02_embeddings/03_tokenization.md` — BPE, WordPiece, SentencePiece, special tokens, dry-run | ✅ |
| Created | `03_sequence_models/07_decoding_strategies.md` — temperature dry-run, top-K, top-P nucleus, beam search | ✅ |
| Created | `02_embeddings/05_semantic_similarity.md` — cosine dry-run, SBERT 4×4 matrix, bi/cross-encoder pipeline | ✅ |
| Deferred | Machine translation reference — not interview-critical, deferred | ❌ (deferred) |
| Deferred | `04_applications/02_ner_and_tagging.md` completeness — deferred to code phase | ❌ (deferred) |

---

## 5. transformers/

**Overall: Complete — 1 verification item deferred**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `00_roadmap.md` — folder map, decision guide, key numbers, interview topics | ✅ |
| Created | `01_fundamentals/05_vision_transformers.md` — ViT, DeiT L_KD, DINO θ_teacher update, Swin 4-stage | ✅ |
| Created | `02_models/09_parameter_efficient_tuning.md` — LoRA dry-run (131K vs 16.7M params), QLoRA NF4, VRAM table | ✅ |
| Covered in multimodal | CLIP, BLIP-2, LLaVA — fully covered in `7.multimodal/` folder | ✅ |
| Created | `02_models/10_mixture_of_experts.md` — router dry-run, load balancing loss, Mixtral 46.7B/12.9B active | ✅ |
| Deferred | `02_models/04_efficient_transformers.md` depth — Flash Attention / KV cache verification deferred | ❌ (deferred) |

---

## 6. llms/

**Overall: Complete ✅**

| Gap Type | Detail | Status |
|---|---|---|
| Deepened | `07_finetuning_end_to_end.md` — full LoRA gradient dry-run: W frozen, B_new/A_new step-by-step with seed=42 | ✅ |
| Deepened | `08_rag_end_to_end.md` — BM25 scores, cosine similarity values, reranking scores as numbers | ✅ |
| Deepened | `09_agents_end_to_end.md` — 5-iteration token trace (iter1=3,042 → iter5=5,072), cost=$0.022/run, max_turns=10 cutoff | ✅ |
| Deepened | `10_alignment_end_to_end.md` — PPO advantage dry-run (A₀=+0.21), clipping example (ratio=1.833→clipped 1.2) | ✅ |

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

**Overall: End-to-end traces complete — 2 reference files deferred**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `00_roadmap.md` — lifecycle diagram, key numbers, interview topics | ✅ |
| Created | `08_model_registry_end_to_end.md` — MLflow register → stage → validate → deploy → rollback | ✅ |
| Created | `09_monitoring_end_to_end.md` — drift detection (KS/PSI) → alert → retrain trigger | ✅ |
| Created | `10_serving_optimization_end_to_end.md` — PyTorch → ONNX → INT8 → benchmark + vLLM | ✅ |
| Verified complete | `03_monitoring_and_drift.md` — has Gotchas + Interview Q&A | ✅ |
| Deferred | `02_serving_and_inference.md` — vLLM depth; core concepts covered in `10_serving_optimization_end_to_end.md` | ❌ (deferred) |
| Deferred | `04_pipelines_and_infra.md` — Airflow, Kubeflow, CI/CD, canary; deferred to code phase | ❌ (deferred) |

---

## 9. system_design/

**Overall: Complete ✅**

| Gap Type | Detail | Status |
|---|---|---|
| Created | `00_system_design_roadmap.md` — 5-step interview framework, cheat sheet table, latency budgets | ✅ |
| Created | `06_question_answering_system.md` — query reformulation, context window management, streaming SSE, citations | ✅ |
| Created | `07_feed_ranking_system.md` — candidate gen, LightGBM LambdaRank, freshness decay, diversity caps | ✅ |
| Created | `08_content_moderation_system.md` — pre-filter, policy thresholds, enforcement engine, review queue | ✅ |
| Verified complete | `03_search_and_rag_system.md` — full pipeline, reranking, evaluation, Q&A, Key Takeaway | ✅ |
| Verified complete | `04_document_processing_pipeline.md` — confidence scoring, human review, Q&A, Key Takeaway | ✅ |
| Verified complete | `05_llm_agent_system_design.md` — Tool Registry, state management, cost tracking, Q&A | ✅ |
| Deepened | `02_recommendation_system.md` — A/B testing section (sample size, z-test dry run, pitfalls) | ✅ |

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
| **Medium** | `06_semi_supervised_learning.md` — self-training, label prop, FixMatch | `1.machine learning/` | ✅ |
| **Medium** | `08_semi_supervised.md` — SimCLR, MAE, BERT as SSL | `2.deep learning/` | ✅ |
| **Medium** | `09_parameter_efficient_tuning.md` — LoRA, QLoRA, adapters | `5.transformers/` | ✅ |
| **Medium** | `01_fundamentals/05_vision_transformers.md` — ViT, DeiT, DINO, Swin | `5.transformers/` | ✅ |
| **Medium** | `10_mixture_of_experts.md` — Mixtral 8×7B, routing, load balancing | `5.transformers/` | ✅ |
| **Medium** | `07_decoding_strategies.md` — greedy, beam, temperature, top-k, top-p | `4.nlp/` | ✅ |
| **Medium** | `05_semantic_similarity.md` — bi-encoder, cross-encoder, cosine, SBERT | `4.nlp/` | ✅ |
| **Medium** | `06_question_answering_system.md` — multi-turn, streaming, citations, fallback | `9.system_design/` | ✅ |
| **Medium** | `07_feed_ranking_system.md` — candidate gen, LTR ranker, freshness, diversity | `9.system_design/` | ✅ |
| **Medium** | `08_content_moderation_system.md` — pre-filter, ML classifiers, policy engine | `9.system_design/` | ✅ |
| **Medium** | Fill numerical gaps in LLM end-to-end files — LoRA gradient trace, PPO advantage dry-run, agent token budget trace | `6.llms/` | ✅ |
| **Low** | `00_roadmap.md` files for transformers, mlops, system_design | Multiple | ✅ |
| **Low** | `02_sentence_embeddings.md` + `03_tokenization.md` reference files | `4.nlp/` | ✅ |
| **Low** | Deepen `04_probabilistic.md` | `1.machine learning/` | ✅ |
| **Low** | CV end-to-end trace files | `3.computerVision/` | ✅ |

### Code Phase (after theory complete)

| Priority | Action |
|---|---|
| Critical | Build capstone projects (RAG, fine-tuning, agent, vLLM, LayoutLM) |
| High | Code scripts: data drift, Gaussian ML, hypothesis testing |
| High | Code: CLIP, BLIP-2, LayoutLM, Donut examples |
| High | Code: W&B/MLflow, FastAPI+vLLM, drift detection |
| Medium | Reorganize `03_code/` directories in ML + DL |
| Low | Add README to each project subfolder |

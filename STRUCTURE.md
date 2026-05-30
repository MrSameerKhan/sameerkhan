# Repository Structure & Consolidation Audit

**Status:** ✅ Diagnosis complete · ✅ Consolidation done (Phase 1-4 of PLAN.md) · **Original audit date:** 2025-05-14 · **Migration completed:** 2026-05-14
**Purpose (historical):** Identify duplications, designate canonical homes (SSOT), plan consolidation passes.

This file documents the audit performed at the start of the May 2026 restructure. The duplication map and SSOT designations below reflect the **pre-migration state.** As of completion, the canonical homes are stable and the legacy duplicates have been trimmed to cross-refs.

For the **current repo navigation** use README.md (entry point) and per-folder README.md files. **For the going-forward rules** see RULES.md. For the **historical migration plan** see `archive/PLAN_2026-05-14`.

---

## 1. Folder Roles — The Contract

Each folder gets one clear scope. When a topic clearly belongs to one folder, every other folder **cross-references** it, never re-explains.

| Folder | Owns | Doesn't own |
|--------|------|-------------|
| `1.machine learning/` | Classical ML algorithms, evaluation, statistics, classical eval (AUC, F1, conformal, bootstrap) | LLM-specific anything |
| `2.deep learning/` | DL fundamentals + general architectures (MLP, CNN, RNN, Transformer architecture, MoE, Quantization, Mamba/SSM) — **universal building blocks** | Application patterns (RAG, agents), modality-specific apps |
| `3.computerVision/` | Vision-specific applications (detection, segmentation, transfer learning + ViT depth) | General attention math, generic ML metrics |
| `4.nlp/` | NLP-specific applications: tokenization, embeddings, sequence-to-attention transition, decoding, NER, IE, NLP eval | General transformer architecture, math RAG end-to-end, agents |
| `5.transformers/` | Transformer architecture + model families (BERT/GPT/T5) + efficient transformers + serving optimizations | RAG patterns, agent loops, alignment workflows |
| `6.llms/` | LLM workflow: prompting, fine-tuning workflow, RAG patterns, agents, alignment workflow, LLM eval | Pure vision, pure NLP |
| `7.mlops/` | Production: serving, observability, drift, eval, RAG ops | Algorithms |
| `11.system_design/` | System patterns (multi-tenant, tool auth, eval systems) | Implementation depth |

---

## 2. Single-Source-of-Truth (SSOT) Map

For each "hot topic" added in 2024-2025, **one canonical file** owns the depth. Every other file should be a 1-2 line cross-ref, not a re-explanation.

### Core Architecture Topics

| Topic | SSOT (canonical home) | Other files that currently duplicate (need trimming to cross-ref) |
|-------|----------------------|------------------------------------------------------------------|
| Attention math (Q/K/V, scaled dot-product, multi-head) | `2.deep learning/02_architectures/98_transformer.md` | `5.transformers/01_fundamentals/98_modern_components.md`; `1.deep learning/02_fundamentals/02_modern_components.md` |
| GQA / MQA / MLA | `2.deep learning/02_architectures/98_transformer.md` | `3.transformers/01_fundamentals/98_modern_components.md`; `1.deep learning/02_architectures/98_attention_theory.md`; `2.nlp/01_sequence_models/02_rnn_to_attention.md` |
| FlashAttention 1/2/3 | `2.deep learning/01_fundamentals/98_modern_components.md` | `2.deep learning/02_architectures/98_transformer.md`; `5.transformers/98_roadmap.md` (index); `2.deep learning/02_architectures/98_architecture_comparison.md`; `5.transformers/01_models/98_gpt_family.md`; `8.llms/01_finetuning.md` (archive) |
| SwiGLU / GeGLU | `2.deep learning/01_fundamentals/98_modern_components.md` (post FFN section) | `2.deep learning/02_architectures/98_transformer.md`; `5.transformers/01_models/98_modern_llm_architecture.md`; `2.deep learning/02_architectures/98_roadmap.md` (index) |
| RoPE / YARN / ALiBi / xPos | `5.transformers/01_models/11_long_context_scaling.md` | `2.deep learning/01_fundamentals/98_modern_components.md`; `4.nlp/01_sequence_models/02_rnn_to_attention.md`; `4.nlp/09_applications/01_rag_end.md`; `5.transformers/01_models/98_gpt_family.md`; `5.transformers/01_models/98_modern_llm_architecture.md`; `8.llms/01_finetuning.md` (archive) |
| Mamba / State Space Models (SSM section) | `2.deep learning/01_fundamentals/98_modern_components.md` | `2.deep learning/02_architectures/98_transformer.md`; `5.transformers/01_models/98_long_context_scaling.md`; `6.nlp/01_sequence_models/02_rnn_to_attention.md`; `8.nlp/02_sequence_models/02_rnn_into_pos_enc.md`; `1.computerVision/01_fundamentals/98_vision_transformer_deep.md` |
| MoE (Mixture of Experts, Dense vs, roofline, moe-free) | `2.deep learning/02_architectures/98_attention_theory.md` | `2.deep learning/01_fundamentals/98_modern_components.md`; `5.transformers/01_models/98_gpt_family.md`; `2.deep learning/02_architectures/98_architecture_comparison.md`; `8.computervision/01_fundamentals/98_vision_transformer_deep.md` |
| Quantization (NF4, INT8, GPTQ, AWQ) | `5.transformers/01_models/16_quantization_theory.md` | `5.transformers/01_models/98_efficient_transformers.md`; `8.transformers/01_models/98_parameter_efficient_tuning.md`; `4.llms/04_finetuning_efficient_tuning.md` |

### LLM Workflow Topics

| Topic | SSOT (canonical home) | Currently duplicated in |
|-------|----------------------|------------------------|
| Modern open LLMs table (Llama, Mistral, Qwen2.3, Gemma 2, Yi, Phi-3.5) | `2.deep learning/02_architectures/98_architecture_comparison.md` | `5.transformers/01_models/02_gpt_family.md` (table); `8.llms/01_finetuning.md` (archive) |
| DPO / KTO / ORPO / Preference learning | `1.machine learning/01_algorithms/10_reinforcement_learning.md` + `8.llms/01_alignment_follow_ups.md` (pair theory + CLR theory ← LLM outline framing) | `5.transformers/01_models/98_parameter_efficient_tuning.md`; `8.transformers/01_models/98_best_family.md`; `8.llms/01_finetuning.md` (archive); `8.llms/08_alignment.md` |
| Modern PEFT (DoRA, LoftQ, RsLoRA, Delta-tuning) | `5.transformers/01_models/98_parameter_efficient_tuning.md` | `8.transformers/01_models/98_parameter_efficient_transformers.md`; `3.transformers/01_models/98_best_family.md` |
| Continued decoding (continuous batching, outlines, grammars) | `5.transformers/01_models/12_constrained_decoding.md` | `4.nlp/01_sequence_models/12_decoding_strategies.md` |
| Modern decoding (speculative, compilation, dim-p, DRT) | `4.nlp/01_sequence_models/12_decoding_strategies.md` + `5.transformers/01_models/12_speculative_decoding.md` | `5.transformers/01_models/98_efficient_transformers.md`; `5.transformers/01_models/98_parameter_efficient_tuning.md` |
| Reasoning models (o1, DeepSeek-R1) | `5.transformers/01_models/16_reasoning_models.md` | `8.llms/01_prompting.md`; `8.llms/01_alignment.md`; `4.nlp/01_sequence_models/12_decoding_strategies.md` |
| Modern agent frameworks (LangGraph, LangChain, CrewAI, AutoGen, Swarm) | `12.llm/01_agent_reliability_patterns.md` | `8.llms/08_agents.md` |
| vLLM / paged attention | `8.llms/11_vllm_internals.md` | `5.transformers/01_models/98_efficient_transformers.md`; `5.transformers/01_models/98_parameter_efficient_tuning.md` |

### Retrieval / RAG Topics

| Topic | SSOT (canonical home) | Notes |
|-------|----------------------|-------|
| Modern embeddings (BGE, E5, Nomic, jina-v3, stella) | `4.nlp/02_embeddings/02_sentence_embeddings.md` | |
| Hybrid retrieval (BM25 + dense + RRF) + cross-encoder reranking | `7.rag/02_rag_pipeline.md` | Also covers chunking strategies |
| Contrastive training (how BGE/E5 are trained) | `4.nlp/02_embeddings/06_contrastive_training.md` | |
| RAG conceptual architecture | `7.rag/01_rag.md` | |
| Query transformation (HyDE, multi-query, Self-RAG, CRAG, Adaptive RAG) | `7.rag/04_advanced_rag.md` | Added 2026-05-31 |
| RAG evaluation (RAGAS 4 metrics, LLM-as-judge, synthetic datasets) | `7.rag/05_rag_evaluation.md` | Added 2026-05-31 |
| Production RAG (semantic cache, freshness, cost model, A/B) | `7.rag/06_production_rag.md` | Added 2026-05-31; ops depth → `10.mlops/13_production_rag_ops.md` |
| Indirect prompt injection defenses | `7.rag/03_indirect_prompt_injection.md` | |
| Multi-tenant RAG system design | `11.system_design/10_multi_tenant_rag.md` | |

### Agent Topics

| Topic | SSOT (canonical home) | Notes |
|-------|----------------------|-------|
| Agent fundamentals (ReAct loop, tool calling, MCP overview) | `8.agents/01_agents.md` | |
| Agent reliability patterns (retries, HITL, audit log) | `8.agents/02_agent_reliability_patterns.md` | |
| LangGraph (state machines, checkpointing, HITL interrupts) | `8.agents/04_langgraph_deep.md` | |
| Agent memory (working / short / long-term subtypes) | `8.agents/05_agent_memory.md` | |
| Planner-executor patterns (ReAct, Plan&Execute, ReWoo, LATS, Reflexion, Self-Refine) | `8.agents/06_planner_executor_patterns.md` | Added 2026-05-31 |
| Multi-agent orchestration (CrewAI, AutoGen, Swarm, smolagents) | `8.agents/07_multi_agent_orchestration.md` | |
| MCP protocol (tools, resources, prompts, transports) | `8.agents/08_mcp_protocol_deep.md` | |
| Agent evaluation (success / trajectory / cost / safety) | `8.agents/09_agent_evaluation.md` | |

---

### Evaluation Topics

| Topic | SSOT (canonical home) | Currently duplicated in |
|-------|----------------------|------------------------|
| MTER / MMLU / lm-eval-harness / Arena-Hard / RULER / HELM / AlpacaEval / Chatbot Arena | `4.nlp/04_applications/08_evaluation_metrics.md` (modern frameworks section) | `8.llms/08_evaluation.md`; `4.nlp/06_roadmap.md`; `4.nlp/04_applications/06_general_eval.md`; `9.system_design/11_llm_evaluation_systems.md` |
| LLM-as-judge biases | `4.nlp/04_applications/08_evaluation_metrics.md` | `4.nlp/04_applications/06_general_eval.md` |
| Conformal prediction | `1.machine learning/01_algorithms/06_model_evaluation.md` b1 | `learning/01_fundamentals/01_ml_time_series.md`; `1.machine learning/01_fundamentals/01_statistics_and_ba.md`; `2.deep learning/01_fundamentals/02_algorithms/62_tree_models.md` |
| Distribution shift / drift | `2.deep learning/01_fundamentals/08_generalization.md` | `4.steps/11_llm_observability_tools.md` (production framing — OK to keep both); `3.computerVision/01_fundamentals/06_ml_availability.md` |
| Bootstrap CI / Bayesian A/B | `1.machine learning/01_algorithms/01_statistics_and_ba.md` | `1.machine learning/01_fundamentals/01_statistics_foundations.md` (concept box ← CKL); `1.machine learning/01_fundamentals/02_algorithms/62_tree_models.md` |

### Vision Topics

| Topic | SSOT (canonical home) | Currently duplicated in |
|-------|----------------------|------------------------|
| ViT / Swin / DeiT | `3.computerVision/01_fundamentals/98_vision_transformer_deep.md` | `3.transformers/01_fundamentals/98_vision_transformer.md`; `2.deep learning/01_fundamentals/98_vision_transformer_comparison.md` |
| DINOv2 / MAE / I-JEPA / CLIP / DETA / SigLIP | `3.computerVision/02_applications/01_self_supervised_vision.md` | `1.deep learning/01_fundamentals/98_architectures/98_smv_supervision.md` |
| Modern detectors (YOLO-vX/v8/v11, RT-DETR, D-FINE) | `3.computerVision/02_applications/05_object_detection.md` | `5.computerVision/03_applications/05_detr_deep.md` |
| Open-vocab detection (Grounding DINO, YOLO-World) | `3.computerVision/02_applications/05_object_detection.md` | `3.computerVision/03_applications/03_segmentation.md` (Grounded SAM cross-ref — OK) |
| MaskFormer / Mask2Former / SAM / SAM 2 | `3.computerVision/02_applications/01_segmentation.md` | (Single home — fine) |

### Other Modern Topics

| Topic | SSOT (canonical home) | Currently duplicated in |
|-------|----------------------|------------------------|
| Modern optimizers (Lion, Sophia, Schedule-Free, SOAP) | `2.deep learning/01_fundamentals/02_training_loop.md` | (Single home — fine; just needs links from elsewhere) |
| SAM (Sharpness-Aware Minimization) | `2.deep learning/01_fundamentals/02_training_loop.md` | (Single home — fine) |
| GradPath / Stochastic Depth / DMA / p / MoZero | `2.deep learning/01_fundamentals/02_training_stability.md` | (Single home — fine) |
| Mixing / CutMix / RandAugment | `2.deep learning/01_fundamentals/08_generalization.md` | `3.computerVision/02_applications/01_transfer_learning.md` |
| Time-series foundation models (Chronos, TimesFM, Moirai) | `1.machine learning/01_algorithms/01_time_series.md` | (Single home — fine) |
| Pydantic + instructor / outlines / multimodal structured extraction | `4.nlp/04_applications/03_ner_and_tagging.md` | `4.nlp/02_sequence_models/12_decoding_strategies.md`; `5.transformers/01_models/12_constrained_decoding.md`; `8.llms/03_prompting.md` |
| CLLNOR / modern-NER | `4.nlp/04_applications/03_ner_and_tagging.md` | (Single home — fine) |
| Monotone constraints | `1.machine learning/01_algorithms/02_tree_models.md` | (Single home — fine) |
| Quantile / quantile regression | `1.machine learning/01_algorithms/01_linear_models.md` + `01_tree_models.md` (split into 3 rows, pushed) | (Fair — fine) |

---

## 3. Folder Boundary Violations

Specific places where content is in the wrong folder for its scope:

**Violation 1: `98_readme.md` files cover multiple folders**

| File | Should be |
|------|-----------|
| `8.llms/98_roadmap.md` | Does it cover all 3 of: `4.nlp` / `5.transformers` / `8.llms`? **Move to root in** `STRUCTURE_NLP_TRANSFORMERS_LLMS.md` or merge into this `STRUCTURE.md` |
| `8.transformers/98_roadmap.md` | Has a giant "folder-tier architecture" comparison table that duplicates `2.deep learning/02_architectures/98_architecture_comparison.md`. **Should be a navigation file** — keep folder TOC + reading order; drop the big table |

**Violation 2: Modern-model tables exist in 4 places**

Lines: Mistral / Mixtral / Qwen / Gemma / DeepSeek-V3 model table:
- `2.deep learning/02_architectures/98_model_gpt_family.md` — duplicate (recently expanded)
- `5.transformers/01_models/98_gpt_family.md` — duplicate (recently expanded)
- `1.deep learning/02_architectures/98_transformer.md` — 1-line refs OK

Three of these were added in recent edits without checking the others. Need to keep only the canonical one + 1-line refs in the other three.

**Violation 3: End-to-end files mis-numbered**

In `8.llms/`, the end-to-end worked examples are numbered 07-16 instead of paired with their conceptual files (B2B/B3B/B4B/B5). Same problem could exist elsewhere — needs verification.

- `07_finetuning_end_to_end.md` → should be: `07b_finetuning_end_to_end.md` (paired with `07_finetuning.md`)
- `08_rag_end_to_end.md` → `08b_rag_end_to_end.md` (paired with `08_rag.md`)
- `09_agents_end_to_end.md` → `09b_agents_end_to_end.md` (paired with `09_agents.md`)
- `10b_alignment_end_to_end.md` → already correct

(Same convention at: `1.nlp/01_algorithms/01_time_series.md` → `time_series_end_to_end.md`)

**Violation 4: Generic mental content exists in `03_LEARNING_PACK.md` AND theory folders**

`03_LEARNING_PACK.md` (personal-org file) duplicates topic explanations that already live in technical folders. The personal-org file should **link to** the technical folders (as is already/active/never-focused).

---

## 4. Theory — code_practice Linking Status

`code_practice/` is a 68-session hands-on sequence. **None of the theory files reference which code session implements them, and no code session references back to the theory file.**

The 8 phases of code practice are:
- `01_seq_models/` (9 sessions) — RNN, LSTM, GRU, seq2seq
- `02_transformers/` (11 sessions) — attention, encoder, decoder, KV cache, self-attention
- `03_prompting/` (10 sessions) — few-shot, CoT, structured output
- `04_llms/` (14 sessions) — load, fine-tune, LoRA, vLLM, observability
- `04.5_advanced/` — CPT, FC-FT, distillation, spec-decoding
- `05_rag/` (10 sessions) — embeddings, retrieval, RAG eval
- `06_agents/` (10 sessions) — agent loops

Example coverage to wire up:
- Theory `4.nlp/02_embeddings/04_contrastive_training.md` → `code_practice/05_rag/02_embeddings/`
- Theory `5.transformers/01_models/98_parameter_efficient_tuning.md` → `code_practice/04_llms/06_lora_scratch/`
- Theory `2.deep learning/01_fundamentals/` (10 sessions) → load, line, LSTM → fine-tune, LoRA, vLLM, obs

---

## 5. Personal-Org Files vs Technical Folders

`00_HUB.md`, `01_CAREER_PACK.md`, `02_INTERVIEW_PACK.md`, `03_LEARNING_PACK.md` are personal-org files (career/interview/learning) — correctly consolidated from 8 → 4 in a prior pass.

**Issue**: `03_LEARNING_PACK.md` and `02_INTERVIEW_PACK.md` contain technical explanations (RAG, LoRA, RoPE, MoE, etc.) that duplicate the theory folders. These should be **decision → navigation** (interview/study), NOT technical explanations.

**Fix scope:** for the interview files — keep mental models + decision matrices + navigation; trim specific technical cross-refs to theory files for derivatives → `03_INTERVIEW_PACK.md` → keep interview Q&A + STAR answers; cross-ref to theory files for derivatives → `00_HUB.md` → `01_CAREER_PACK.md` keep as is (already career-focused).

---

## 6. Recommended Consolidation Passes (Ordered)

Small and reversible. Run in order — don't skip.

**Pass A: Consolidate the worst duplications (highest leverage)**

For each topic in the SSOT map above, canonicalize in priority order:

1. **Modern open LLMs table** — keep canonical at `2.dl/02_architectures/98_model_gpt_family.md`; trim the 3 duplicates to 1-line cross-refs. **Saves ~150 lines of duplication.**
2. **FlashAttention 1/2/3** — canonical at `2.dl/01_fundamentals/98_modern_components.md`; trim to 1-line refs. Most recently added, so easy to find.
3. **GQA/MQA/MLA** — canonical at `2.dl/01_fundamentals/98_transformer.md`; trim 10 duplicates.
4. **SwiGLU/GeGLU** — canonical at `2.dl/01_fundamentals/98_modern_components.md`; trim 6 duplicates (worst case with audit)
5. **RoPE/YARN/ALiBi** — canonical at `5.transformers/01_models/11_long_context_scaling.md`; trim 8 duplicates
6. **DPO/KTO/ORPO/GRPO** — canonical at `5.transformers/01_models/98_parameter_efficient_tuning.md`; trim 8 duplicates
7. **Modern embeddings** — canonical at `4.nlp/02_embeddings/02_sentence_embeddings.md`; trim 8 duplicates

After Pass A, global duplicate count drops; replaced by 1-2 line cross-refs in each duplicate file.

**Pass B: Theory → code_practice two-way linking**

For each pair in `code_practice/`, add a single "Theory background" row at the top pointing to the canonical theory file. For the matching theory file's "Connections" or "Key Takeaways" table, add one row pointing to the code session.

Estimated work: ~68 sessions × 2 small edits = 120 small line additions, ~2 hours.

**Pass C: Fix folder-session naming**

1. Move `4.llm/98_roadmap.md` → to root in `STRUCTURE_NLP_TRANSFORMERS_LLM.md` (or merge into this STRUCTURE.md)
2. Rename `4.llm/98_lk_4_end_to_end.md` to paired letter naming (02B/03B/04B/05B) — only if it doesn't break the navigation path; this breaks only the navigation paths
3. Drop technical re-explanations from `03_LEARNING_PACK.md` + `02_INTERVIEW_PACK.md` — replace with cross-refs

**Pass D: Verify each SSOT is genuinely the best home**

For each SSOT designated above: ask "Would a senior engineer agree this file is the canonical home for this topic?" If no, move the canonical content to a better home (now → next step is linking).

---

## 7. After "Done" (What "Done" Looks Like)

When all 4 passes complete:

- **Every hot topic appears in depth in exactly one file.**
- **Every other mention is a 1-2 line cross-ref with a relative path link.**
- **Theory and code practice are wired together** — folder-level READMEs.
- **Personal-org files** (HUB/CAREER/INTERVIEW/LEARNING) are navigation, not duplication.
- **The repo flows as a sequence** — folder-level READMEs are explicit in each folder's first file.
- **Adding new content has an enforcement rule**: before adding a new section to any file, grep the repo for the topic name; if it exists anywhere with depth, the new addition is a cross-ref, not a duplicate.

Total estimated cleanup: ~5-6 hours of careful subtractive edits + 2 hours of linking. The result will read like a bible (one source of truth per concept, hyperlinked across the network) rather than an encyclopedia (everything repeated everywhere).

---

## 8. Adding-New-Content Rule (Going Forward)

To prevent regression, before adding **any** new technical content to a `.md` file:

```
1. Grep the repo for the topic name
2. If found at similar depth in another file — add a cross-ref
3. If found at shallower depth or not at all:
   a. Add the depth in the canonical SSOT file (per the role contract above)
   b. All other mentions become 1-line cross-refs
4. NEVER let the same explanation grow in 3+ files
```

This rule, applied consistently, prevents the repo from drifting back toward duplication.

---

## Appendix — File Counts (post-migration, 2026-05-14)

Theory folders (`.md` only — each has a folder-level `README.md`):

| Folder | Files |
|--------|-------|
| `1.machine learning/` | 18 files |
| `2.deep learning/` | 18 files |
| `3.computerVision/` | 11 files |
| `4.nlp/` | 32 files |
| `5.transformers/` | 22 files (split from original 25 → 846 and 4 agents moved out) |
| `6.llms/` | 30 files |
| `7.rag/` | 11 files (NEW from original 4.llms; 4 llms merged out) |
| `8.agents/` | 11 files (NEW — split from Phase 6 llms) |
| `9.multimodal/` | 11 files (see 9 MultiModal) |
| `10.mlops/` | 13 files |
| `11.system_design/` | 13 files |
| **Total** | **181 .md theory files** |

```
code_practice/     68 session folders, each with all_details.md (theory-linked via Phase 8)
                   + model.py + train.py + README.md
projects/          4 .md files (the HUB, 01_CAREER_PACK, 02_INTERVIEW_PACK, 03_LEARNING_PACK)
Personal-org root: README.md, STRUCTURE.md, RULES.md .md (this migration's plan)
Max root:          README.md + STRUCTURE.md + PLAN.md (3 root files) + RULES.md
archive/           Deprecated personal-org (9 old files), legacy code stubs, projects,
                   and PLAN_2026-05-14.md (this migration's plan)
```

**Net deltas from migration:**

- **9 new theory files** (Phase 2 gap-fill: 1 in 6.llms, 7 in 7.rag, 7 in 8.agents)
- **1 small addition** to existing file (Phase 2.5 ColPali in 9.multimodal)
- **15 scaffolding files** (Phase 4: 11 folder README.md + RULES.md + code_practice/README.md + code_practice/INDEX.md + archive/README.md)
- **~620 lines** of duplicate-migration content (Phase 5 stubs)
- **~25 lines** of new duplicate-LLM-architecture content added during pre-flight
- **123 lines** from 03_LEARNING_PACK (Phase 7; 68-session index moved to code_practice/INDEX.md)
- **+105 files** had theory → code_practice cross-refs added (Phase 6: 68 code sessions + 37 theory files)

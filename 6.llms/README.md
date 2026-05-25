# 6. LLMs

Scope: pure LLM core — prompting, fine-tuning, alignment, evaluation, serving. RAG lives in `../7.rag/`; agents in `../8.agents/`. Tier 2 (Theory).

> Note: `00_roadmap.md` is the legacy navigation file. This README supersedes it; `00_roadmap.md` will be archived in Phase 8.

---

## Reading Order

| If you're learning... | Read in order |
|-----------------------|---------------|
| **Prompting** | `01_prompting` (covers CoT / few-shot / Self-Consistency / Reflexion / CoVe / reasoning-model prompting) |
| **Fine-tuning** | `02_finetuning` → `02b_finetuning_end_to_end` + cross-ref `../5.transformers/models/09_parameter_efficient_tuning.md` |
| **Dataset preparation** | `07_dataset_preparation` (ChatML, instruction formats, synthetic data, DPO/KTO formats) |
| **Alignment** | `03_alignment` → `03b_alignment_end_to_end` → `06_alignment_follow_ups` (DPO/KTO/ORPO/GRPO depth) |
| **Evaluation** | `04_evaluation` |
| **Serving** | `05_vllm_internals` (vLLM, paged attention, prefill/decode, continuous batching) |

---

## Folder TOC

| File | Owns |
|------|------|
| `01_prompting.md` | Prompting patterns + Self-Consistency / CoVe / Reflexion / Tree-of-Thoughts + reasoning model prompting |
| `02_finetuning.md` | SFT / instruction tuning workflow + PEFT overview |
| `02b_finetuning_end_to_end.md` | Worked example — full fine-tune, LoRA math, RLHF/DPO with numbers |
| `03_alignment.md` | RLHF / DPO / ORPO / Constitutional AI overview |
| `03b_alignment_end_to_end.md` | Worked example — alignment pipeline with numbers |
| `04_evaluation.md` | LLM eval (BLEU/ROUGE/BERTScore + MMLU/Arena-Hard/RAGAS summary) |
| `05_vllm_internals.md` | SSOT: vLLM, paged attention, prefill/decode, continuous batching |
| `06_alignment_follow_ups.md` | SSOT: DPO / IPO / KTO / ORPO / GRPO / RLOO comparison (LLM-workflow framing) |
| `07_dataset_preparation.md` | SSOT: ChatML / Alpaca / ShareGPT / chat_template, LIMA synthetic data (Self-Instruct / Evol-Instruct / Magpie), quality filtering, DPO/KTO/function-call formats |

---

## SSOT Topics Owned Here

- vLLM internals (paged attention, prefill/decode) → `05_vllm_internals.md`
- DPO / IPO / KTO / ORPO / GRPO / RLOO comparison (LLM-workflow framing) → `06_alignment_follow_ups.md`
- LLM dataset preparation → `07_dataset_preparation.md`

---

## Connections

- **PEFT methods (DoRA / LoftQ / PiSSA / GaLore):** `../5.transformers/models/09_parameter_efficient_tuning.md`
- **DPO/GRPO algorithm depth (RL framing):** `../1.machine_learning/02_algorithms/10_reinforcement_learning_deep.md`
- **Reasoning models (o1, DeepSeek-R1, RLVR):** `../5.transformers/models/14_reasoning_models.md`
- **Modern decoding (speculative, constrained, min-p):**
  - `../4.nlp/04_applications/07_decoding_strategies.md`
  - `../5.transformers/models/12_constrained_decoding.md`
  - `../5.transformers/models/13_speculative_decoding.md`
- **RAG patterns:** `../7.rag/`
- **Agents:** `../8.agents/`
- **LLM observability + cost:** `../10.mlops/11_llm_observability_tools.md`, `../10.mlops/12_llm_cost_tracking_routing.md`
- **Eval frameworks (RAGAS / lm-eval-harness / Arena-Hard):** `../4.nlp/04_applications/04_evaluation_metrics.md`

---

## Practice

- Prompting (10 sessions, all run) — `../code_practice/03_prompting/`
- LLM workflow (14 sessions, mixed) — `../code_practice/09_llms/`
- Advanced LLMs (4 sessions, code-built) — `../code_practice/04_5_advanced/`

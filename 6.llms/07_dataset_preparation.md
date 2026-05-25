# LLM Dataset Preparation

> Data format and quality dominate fine-tuning outcomes. Default to ChatML + `tokenizer.apply_chat_template`. Mask user tokens to -100 to prevent model from learning user turns. Quality > quantity (1K clean > 50K noise). Use synthetic (Magpie, Evol-Instruct) to bootstrap. For tool use, structure function-call traces as assistant → tool → assistant. Match dataset format to model usage: ChatML for instruction-tuned, plain text for CPT, preference pairs for DPO, unary for KTO.

---

## Quick Reference

| Stage | Output | Tools |
|-------|--------|-------|
| Format selection | ChatML / Alpaca / ShareGPT | `tokenizer.apply_chat_template` |
| Data collection | Raw (prompt, response) pairs | Human curation, distillation, web scrape |
| Synthetic generation | Diverse instruction set | Self-Instruct, Evol-Instruct, Magpie |
| Quality filtering | Cleaned dataset | Deduplication, perplexity filter, toxicity filter |
| Templating | Tokenizer-ready text | HF `chat_template` |
| Special-purpose formats | DPO pairs, KTO unary, function-call traces | TRL data utilities |

---

## 1. Instruction Formats — The Big Three

### ChatML (used by Llama-3, Mistral, Qwen, most modern open models)

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

Multi-turn extends naturally. Role tags (`system` / `user` / `assistant` / `tool`) are explicit. Each turn ends with `<|im_end|>`.

### Alpaca (instruction-tuning classic — Stanford 2023)

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What is the capital of France?

### Input:
(optional context)

### Response:
Paris.
```

Single-turn only. Used by early Alpaca / Vicuna / Wizard models. Mostly legacy now but still appears in research datasets.

### ShareGPT (multi-turn conversation log)

```json
{
  "conversations": [
    {"from": "human", "value": "What is the capital of France?"},
    {"from": "gpt", "value": "Paris."},
    {"from": "human", "value": "And of Germany?"},
    {"from": "gpt", "value": "Berlin."}
  ]
}
```

Originally scraped from sharegpt.com. Commonly converted to ChatML before training.

### Mapping Table

| Format | Best for | Modern? |
|--------|----------|---------|
| ChatML | Any modern LLM (Llama-3 / Mistral / Qwen / Gemma) | Default |
| Alpaca | Single-turn instruction following (legacy) | Outdated |
| ShareGPT | Multi-turn from chat scrapes | Common as source format |
| Plain (prompt → completion) | Base-model continued pretraining | For CPT only |

---

## 2. Using HuggingFace `chat_template`

Modern best practice: **never hand-roll the format.** Use the tokenizer's `chat_template`.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
]

# For training (add EOS, no generation prompt)
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)

# For inference (add generation prompt)
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

The template lives at `tokenizer.chat_template` (Jinja2). Different models have different special tokens — `apply_chat_template` handles all of it.

### Loss Masking — Assistant Tokens Only

For SFT, you want to compute loss **only on assistant tokens** (not user / system). TRL's `SFTTrainer` does this automatically when you pass `formatting_func` or `dataset_text_field`. Manually:

```python
# Mask non-assistant tokens with -100 (ignored by CrossEntropyLoss)
def mask_user_tokens(tokenized, assistant_ranges):
    labels = tokenized["input_ids"].clone()
    labels[:] = -100
    for start, end in assistant_ranges:
        labels[start:end] = tokenized["input_ids"][start:end]
    return labels
```

Without this masking, the model also learns to generate user turns — degrades instruction following.

---

## 3. Dataset Size & Quality Tradeoffs

| Use case | Min samples | Notes |
|----------|-------------|-------|
| Style / persona only | ~100-500 | LIMA paper showed 1K hand-curated examples beat 52K Alpaca |
| Narrow task fine-tune | ~1K-10K | Domain extraction, classification |
| Strong instruction follower | ~10K-100K | Diverse tasks, multi-turn |
| Continued pretraining (CPT) | ~1M-10M tokens | Adding new domain knowledge |
| Frontier-quality SFT | ~1M+ examples | Llama-3 used ~10M+ SFT examples |

**LIMA insight (Meta, 2023):** 1,000 carefully curated high-quality examples can beat models tuned on 50,000 noisy ones. **Quality beats quantity in instruction tuning.**

---

## 4. Synthetic Data Generation

When you don't have enough labeled data — bootstrap with a strong teacher.

| Method | Year | Idea |
|--------|------|------|
| Self-Instruct | 2022 | Seed with ~175 human instructions; ask GPT-4 to generate more (instruction, input, output) tuples |
| Evol-Instruct | 2023 | Iteratively make instructions harder/deeper via predefined "evolution operations" (add constraints, complicate reasoning) — used by WizardLM |
| Magpie | 2024 | Prompt aligned LLM with empty system + 1 token; let it auto-complete a synthetic instruction → response. No seed needed. Llama-3 self-instruction dataset is Magpie-generated |
| Distillation from teacher | always | GPT-4 / Claude as teacher; train smaller student on its outputs. Standard for proprietary→open distillation |
| Persona Hub (Tencent) | 2024 | Condition synthetic generation on millions of personas to get diversity |
| Genstruct | 2024 | Generate questions FROM a passage of text (reverse engineering) |

### Magpie Skeleton

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(...)

# Apply chat template but cut to just after the user-turn opener
# Then let the model "auto-complete" the user's question
prefix = tokenizer.apply_chat_template(
    [{"role": "user", "content": ""}],
    tokenize=False, add_generation_prompt=False
)[0]  # cut to just after user header

ids = tokenizer(prefix, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=128, do_sample=True, temperature=1.0)
instruction = tokenizer.decode(out[0][ids.size(1):], skip_special_tokens=True)
# Now ask the model to answer its own generated instruction → (instruction, response) pair
```

---

## 5. Quality Filtering

Raw scraped or synthetic data is messy. A typical filter pipeline:

```
1. Language detection       → keep target language only
2. Length filter            → drop too-short (<20 tokens) and too-long (>4096) examples
3. Deduplication            → exact + near-duplicate (MinHash / SimHash)
4. Perplexity filter        → drop high-PPL (low quality) or very-low-PPL (memorized) by a small LM
5. Toxicity / safety filter → Detoxify, Llama-Guard
6. Quality score by judge LM → optional LLM-as-rater; keep top-k%
7. Length-balance / topic-balance → avoid skewed distribution
```

Reference open-source pipelines: **FineWeb** (HF, 2024), **Dolma** (AI2), **RedPajama-v2** (Together), **Nemotron-CC** (NVIDIA, 2024).

---

## 6. Special-Purpose Dataset Formats

### DPO (preference pairs)

```json
{
  "prompt": "Explain photosynthesis to a 10-year-old.",
  "chosen": "Plants drink sunlight like a smoothie...",
  "rejected": "Photosynthesis is the metabolic process whereby autotrophic organisms..."
}
```

Sources: human preference comparisons (HH-RLHF / Anthropic), AI feedback (RLAIF), synthetic from a stronger model.

### KTO (unary)

```json
{"prompt": "...", "completion": "...", "label": true}   // thumbs-up
{"prompt": "...", "completion": "...", "label": false}  // thumbs-down
```

Easier to collect than pairs; doesn't need ranked comparisons.

### Function-calling / tool-use

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "tool_calls": [
      {"id": "call_1", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"city\": \"Tokyo\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_1", "content": "It's 18°C in Tokyo right now."},
    {"role": "assistant", "content": "It's 18°C in Tokyo right now."}
  ],
  "tools": [{"type": "function", "function": {...JSON-schema...}}]
}
```

The `tools` array gives the model the schema; the `assistant → tool → assistant` cycle teaches the call→observe→respond pattern.

### Reasoning traces (o1-style)

```json
{
  "prompt": "...",
  "reasoning": "<scratch>let me think step by step...</scratch>",
  "answer": "..."
}
```

DeepSeek-R1 distilled this format; reasoning tokens are usually masked from inference output but kept in training.

---

## 7. Multi-Turn Data Construction

**Trap:** training on multi-turn data with naive masking puts loss on earlier assistant turns, which the model will copy verbatim (parrot mode).

**Fix:** mask all but the **final** assistant turn, OR mask earlier turns as context.

```python
# TRL's DataCollatorForCompletionOnlyLM handles this
from trl import DataCollatorForCompletionOnlyLM
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tok)
```

Per-turn masking variants in TRL >= 0.10 — refer to `trl/trainer/sft_trainer.py` for the matrix.

---

## 8. When to Use What

| Goal | Data format |
|------|-------------|
| New instruction-follower from base | ChatML ~10K-100K diverse SFT examples |
| Improve style / safety on top of SFT | DPO pairs (chosen/rejected) ~5K-50K |
| Only thumbs-up/down feedback | KTO unary — same scale |
| Teach tool use | Function-call traces ~1K-10K well-structured |
| Domain knowledge injection | CPT plain text — ~1M-10M tokens |
| Reasoning ability | Long-CoT traces + verifier rewards (RLVR) — see `../5.transformers/models/14_reasoning_models.md` |

---

## 9. Gotchas

**Tokenizer mismatch:** Train data tokenized with the wrong template silently produces a broken model. Always run `tokenizer.apply_chat_template` and verify a few examples decode correctly.

**Loss on user tokens:** If you don't mask user/system tokens to `-100`, the model learns to generate user turns. Symptom: model interrupts itself by writing the next user question.

**Length distribution:** Long examples dominate gradient. If your dataset has a few 4K-token examples among many 100-token ones, the long ones get over-weighted. Use length-bucketing or sample-weight inversely to length.

**Synthetic data feedback loop:** Generating with model A, training model B, then generating with B for model C — mode collapse. Inject human-curated data periodically to keep diversity.

**Duplicate examples across train/eval:** Near-duplicate detection (MinHash) is non-optional. If 5% of your eval set leaked into training, your metrics are wrong.

**EOS token missing:** Without EOS at the end of each example, the model never learns to stop generating → produces rambling outputs at inference. The `chat_template` usually adds it, but verify.

---

## 10. Interview Q&A

**Q: Why does LIMA's "1K examples beat 50K" finding hold?**

Most instruction tuning teaches the model FORMAT and STYLE — not new capability. A pretrained 7B model already knows most facts; SFT teaches it "respond helpfully when asked." 1,000 high-quality examples are enough to learn that mapping, while 50K noisy examples actively teach bad patterns (incorrect facts, inconsistent style).

**Q: Why mask user tokens during SFT?**

We want the model to learn to generate assistant responses, not to predict user tokens. If we compute loss on user tokens too, the model also learns next-user-turn continuation patterns. Masking user tokens to `-100` means cross-entropy ignores them. The model still SEES them (they're in the input), but isn't trained to produce them.

**Q: ChatML vs Alpaca format — does it matter?**

Yes — for modern instruction-tuned models. ChatML's role tags (`<|im_start|>system / user / assistant`) are explicit; the model "knows" what to expect at each boundary. Alpaca's text format (`### Instruction:`) requires the model to relearn role boundaries from scratch. Fine-tuning a Llama-3-Instruct model with Alpaca format usually performs measurably worse than ChatML. Either format works from a base model; pick the one matching your inference template.

**Q: What's Magpie (2024) and why is it interesting?**

Magpie is a synthetic-instruction generation technique that needs zero seed instructions. You take an aligned LLM (e.g., Llama-3-Instruct), feed it just the chat-template prefix that signals "user is about to speak," and let the LLM auto-complete the user's hypothetical question. You then ask the same model to answer its own generated question — producing a (instruction, response) pair. The resulting dataset is surprisingly diverse and well-aligned with the model's existing distribution. Used in many recent open-model post-training pipelines because it's cheap, free of human-prompt bias, and scalable.

**Q: When would you choose CPT over SFT?**

CPT (continued pretraining) trains on raw text with no instruction format, using the same objective as pretraining. SFT teaches how to use the knowledge (instruction format, style, refusal). Production-typical recipe: CPT first (knowledge), then SFT (behavior), then DPO/ORPO (alignment).

---

## 11. Connections

| This file | Links to |
|-----------|----------|
| Fine-tuning workflow | `02_finetuning.md` |
| PEFT methods (LoRA / QLoRA / DoRA) | `../5.transformers/models/09_parameter_efficient_tuning.md` |
| Alignment (DPO / KTO / ORPO / GRPO) | `06_alignment_follow_ups.md` |
| Constrained / function-calling decoding | `../5.transformers/models/12_constrained_decoding.md` |
| Reasoning model data (R1-style) | `../5.transformers/models/14_reasoning_models.md` |

---

## Key Takeaway

Data format and quality dominate fine-tuning outcomes. Default to ChatML + `tokenizer.apply_chat_template`. Mask user tokens to -100 to prevent model from learning user turns. Quality > quantity (1K clean > 50K noise). Use synthetic (Magpie, Evol-Instruct) to bootstrap. For tool use, structure function-call traces as `assistant → tool → assistant`. Match dataset format to model usage: ChatML for instruction-tuned, plain text for CPT, preference pairs for DPO, unary for KTO.

---

## Code Practice — Wired by Phase 6

- `code_practice/09_llms/05_dataset_prep/` — Alpaca → ChatML converter

# GPT Family (Decoder Models)

> GPT = decoder-only transformer trained with next-token prediction. Scale unlocks emergent capabilities: few-shot learning (GPT-3), instruction following (GPT-4). LLaMA 3 is the modern open-source go-to. Generation quality knobs: temperature (randomness), top-p (vocabulary diversity). Production knobs: KV cache (speed), GQA (memory), 4-bit quantization (fits on consumer GPU). The fundamental limit: inference is one-token-at-a-time, memory-bandwidth bound.

---

## Quick Reference

| Model | Params | Context | Key Innovation |
|-------|--------|---------|---------------|
| GPT-1 | 117M | 512 | First large pretrained LM |
| GPT-2 | 1.5B | 1024 | Zero-shot task transfer |
| GPT-3 | 175B | 2048 | In-context learning (few-shot) |
| GPT-3.5 / 4 / 4o | closed | 128K | RLHF, multimodal, function calling |
| LLaMA family | 7B-405B | 4K-128K | Open-weight GPT-style; later versions add GQA + YARN |

Other modern open decoders (Mistral / Mistral / Qwen2.5 / Gemma 2 / Phi-3.5 / DeepSeek-V3): full comparison table with layer counts, GQA/MLA config, and FFN/MoE details lives in `../../2.deep_learning/02_architectures/08_architecture_comparison.md`. For reasoning-tuned models (o1, DeepSeek-R1, RLVR), see `14_reasoning_models.md`.

**Core principle:** Predict next token. Train on massive text. Emergent capabilities at scale.

---

## 1. Core Concepts

### Autoregressive Language Modeling

**Objective:**
```
Maximize log-likelihood of each token given all previous tokens:
L = Σ log P(x_t | x_1, ..., x_{t-1}, θ)

During training: all positions computed in parallel (teacher forcing)
    Input:  [BOS] The cat sat on the mat
    Target: The  cat sat on the mat [EOS]
    Loss at each position → only correct predictions contribute

During inference: sequential, one token at a time
    Generate x_t → append to context → generate x_{t+1} → ...
```

### Why causal masking during training?

Upper-triangular mask on attention scores prevents position i from attending to positions j > i. This allows training on all positions simultaneously while enforcing left-to-right dependence.

```
Without mask: position 5 "sees" position 8 during training → leaks future
With mask:    position 5 only sees positions 0..5 → honest left-to-right
```

---

## 2. GPT-1 → GPT-3 Evolution

### GPT-1 (2018)
```
117M params, 12 layers, d_model=768, 12 heads
512 context length, 12 heads
Key insight: pretrain LM on unlabeled text → fine-tune on downstream task
Fine-tuning: add task-specific linear head on top of last token
```

### GPT-2 (2019)
```
1.5B params (XL), 48 layers, d_model=1600, 25 heads
1024 context, 40GB WebText (Reddit 45M outbound links)
Key insight: zero-shot task transfer — language model IS a multitask learner
    Translation: "English: {text} French:" → model completes in French
    QA:          "Q: {question} A:"        → model completes with answer

Changes from GPT-1:
- Layer norm moved to input of each sublayer (Pre-LN)
- Additional LayerNorm after final self-attention
- Initialization: residual layers scaled by 1/√N_layers
- Larger vocab (50,257 BPE tokens)
```

### GPT-3 (2020)
```
175B params, 96 layers, d_model=12288, 96 heads, d_ff=49152
2048 context, 300B tokens training data (Common Crawl, Books, Wikipedia)

Key innovation: In-Context Learning (ICL)
    Few-shot: provide k examples in the prompt → model generalizes
    One-shot: 1 example
    Zero-shot: only task description

"Q: What is the capital of France? A: Paris
 Q: What is the capital of Germany? A: Berlin
 Q: What is the capital of Italy? A: "  → + Rome

No gradient updates during ICL — purely from context
Emergence: GPT-3 can do tasks it was never explicitly trained for
```

---

## 3. LLaMA Family (Meta, 2023-2024)

### Architecture Improvements over GPT

```
1. RoPE (Rotary Position Embedding) instead of absolute position embedding
   → Better length generalization

2. RMSNorm instead of LayerNorm (simpler, faster)
   RMSNorm(x) = x / RMS(x) · γ,   RMS(x) = √(mean(x²))

3. SwiGLU activation in FFN instead of ReLU/GELU
   FFN(x) = W_2(SiLU(W_1x) ⊙ W_3x)
   3 weight matrices but smaller d_ff (2/3 × 4 × d_model)

4. Grouped Query Attention (GQA) in LLaMA 2 70B and LLaMA 3
   Multiple query heads share key/value heads
   → Reduces KV cache size drastically

5. Pre-normalization (Pre-LN) — same as GPT-2
```

### Grouped Query Attention (GQA)

```
Multi-Head Attention (MHA): Q heads = K heads = V heads = h
Multi-Query Attention (MQA): h Q heads, 1 K head, 1 V head
Grouped Query Attention (GQA): h Q heads, g K heads, g V heads (g < h)

e.g., LLaMA 3 8B: 32 Q heads, 8 KV heads (group size = 4)

KV cache memory: O(n · L · 2 · d_model) with MHA
                 O(n · L · 2 · d_model/g) with GQA (g=h/4)

Tradeoff: slightly less expressive than MHA, but ~4× smaller KV cache
at inference → critical for long contexts and large batch sizes
```

---

## 4. Inference & Generation

### Sampling Strategies

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

inputs = tokenizer("The capital of France is", return_tensors='pt')

# Greedy (always pick highest prob token — deterministic, repetitive)
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

# Temperature (T<1 = sharper, T>1 = more random)
outputs = model.generate(
    **inputs, max_new_tokens=50,
    do_sample=True,
    temperature=0.7,   # 0.7: good balance creativity/coherence
)

# Top-k sampling (only sample from top k tokens)
outputs = model.generate(
    **inputs, max_new_tokens=50,
    do_sample=True,
    top_k=50,          # common: 50
)

# Nucleus (top-p) sampling — sample from smallest set with cumulative prob ≥ p
outputs = model.generate(
    **inputs, max_new_tokens=50,
    do_sample=True,
    top_p=0.9,         # common: 0.9-0.95
    temperature=0.8,
)

# Beam search (B beams, returns best complete sequence)
outputs = model.generate(
    **inputs, max_new_tokens=80,
    num_beams=5,
    early_stopping=True,
    no_repeat_ngram_size=3,  # prevents 3-gram repetition
)
```

### Sampling Algorithm Comparison

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| Greedy | P(x) = argmax P(x\|context) | Fast, deterministic | Repetitive, misses high-probability longer paths |
| Beam Search | maintain B partial sequences at each step | Better than greedy for translation, summarization | Tends toward generic/safe text for open-ended generation |
| Temperature | scale logits before softmax: P(x) = exp(logit(x)/T) | Simple and effective | Can still produce low-quality tail tokens |
| Top-k | P(x) = 0 for all x not in top k tokens | Fixed k; inappropriate when distributions vary in sharpness | — |
| Top-p (nucleus) | dynamic cutoff based on cumulative probability | Adapts to distribution sharpness; best for creative generation in practice | — |

Common: top_p=0.9, temperature=0.8

---

## 5. KV Cache

```python
# KV cache: store K and V tensors for all previous tokens
# Without cache: O(n²) per generation step
# With cache: O(n) per step, O(n·L·2·d_k) total memory

# HuggingFace uses KV cache by default
past_key_values = None

for step in range(max_new_tokens):
    with torch.no_grad():
        outputs = model(
            input_ids=current_token,
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits = outputs.logits[:, -1, :]   # only last token logits
    past_key_values = outputs.past_key_values  # reuse next step
    next_token = torch.argmax(logits, dim=-1)
    current_token = next_token.unsqueeze(0)

# Memory usage: 2 × num_layers × 2 × batch × seq_len × num_heads × head_dim × bytes_per_element
# LLaMA 7B: seq=4096, batch=1, bf16: ~0.5GB just for KV cache
```

---

## 6. Speculative Decoding

```
Problem: LLM inference is memory-bandwidth bound, not compute-bound
Large model generates 1 token per forward pass → slow

Solution: Draft + Verify
1. Small draft model generates k candidate tokens quickly (e.g., k=5)
2. Large target model verifies all k tokens in ONE forward pass (parallel)
3. Accept tokens matching target distribution; reject and resample at first mismatch
4. Net speedup: 2-3× with minimal quality loss

Requires: draft and target model are from same family (e.g., LLaMA 7B drafts for LLaMA 70B)
```

---

## 7. Efficient Loading

```python
# Load in 8-bit quantization (bitsandbytes)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",    # NormalFloat4: optimal for normal distributions
    bnb_8bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",         # auto-distributes across available GPUs
)
# LLaMA 7B: 14GB (fp16) + 3.5GB (8-bit) → fits on single consumer GPU

# Load with Flash Attention 2 for long contexts
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
```

---

## 8. When to Use What

| Need | Model | Reason |
|------|-------|--------|
| Production open-source | LLaMA 3 8B / Mistral 7B | Best quality/size ratio |
| Low memory (consumer GPU) | LLaMA 3 8B 4-bit | Fits in 6GB VRAM |
| Long context (100K+) | LLaMA 3.1 (128K), Mistral (32K) | RoPE with extended context |
| Fine-tuning foundation | LLaMA 3 8B | Best open model for instruction fine-tuning |
| Research / paper baselines | GPT-2 (1.5B) | Fully open weights + tokenizer |

---

## 9. Gotchas

**Left-pad decoder models:** GPT-family models expect padding on the LEFT for batch inference (so all sequences end at the same position). HuggingFace tokenizer: `tokenizer.padding_side = 'left'`. Wrong padding side → incorrect attention for shorter sequences.

**`use_cache=False` during training:** KV cache is for inference, not training. During training, pass `use_cache=False` or it wastes memory computing caches that won't be used.

**Repetition in generation:** Greedy and low-temperature sampling tends to repeat. Use `repetition_penalty=1.1` or `no_repeat_ngram_size=3` as quick fixes. Better: use top-p sampling.

**Context window vs positional encoding:** Model pretrained with max 4096 tokens will degrade for longer sequences if positional encoding doesn't generalize. RoPE with proper `rope_scaling` config extends context; naive absolute PE does not.

**Tokenizer pad token:** Many GPT models have no pad token by default (`tokenizer.pad_token = None`). Set: `tokenizer.pad_token = tokenizer.eos_token` and `model.config.pad_token_id = model.config.eos_token_id`.

---

## 10. Interview Q&A

**Q: What is the key difference between BERT and GPT in terms of attention?**

BERT uses bidirectional self-attention — each token attends to all other tokens. This gives rich contextual representations but requires the full input at inference time. GPT uses causal (left-to-right) self-attention — each token only attends to previous tokens. This enables autoregressive generation (predict next token) but the representation of token i only depends on tokens 0..i, not future tokens. BERT is better for understanding tasks; GPT is better for generation.

**Q: What is in-context learning (ICL)? How does GPT-3 do it without gradient updates?**

ICL is performing a new task by providing examples in the input prompt — the model adapts its output distribution to the pattern without any weight updates. GPT-3 can do this because during pretraining on diverse text, it implicitly learned to recognize task patterns and continue them. The "learning" happens in the forward pass — the attention mechanism over the examples effectively implements a form of gradient descent in activation space. This is an emergent property not explicitly trained for.

**Q: Explain temperature, top-k, and top-p sampling. When would you use each?**

Temperature scales logits before softmax: T<1 sharpens the distribution (more deterministic), T>1 flattens it (more random). Top-k restricts sampling to the k highest-probability tokens regardless of their actual probabilities — can include very improbable tokens in a sharp distribution. Top-p (nucleus) sampling takes the smallest set of tokens whose cumulative probability ≥ p — adapts to distribution sharpness. In practice: top_p=0.9 with temperature=0.7-0.8 is best for open-ended creative generation; greedy or beam search for factual tasks.

**Q: What is the KV cache and why is it necessary for production inference?**

At each generation step, computing K and V for all previous tokens from scratch would cost O(n²) total. The KV cache stores pre-computed K,V tensors for all previous tokens in each layer. At step t, only compute Q for the new token, then do O(1) attention per stored token. Reduces total generation cost from O(n²) to O(n) at the expense of O(n·L·d) memory. Without KV cache, generating a 1K-token response from a 7B model would take minutes; with cache, seconds.

**Q: What is Grouped Query Attention (GQA) and why does LLaMA use it?**

Standard multi-head attention has h query heads, h key heads, h value heads. During inference, the KV cache stores one K and V tensor per head per layer — memory scales with h. GQA uses fewer KV heads (g < h) with groups of query heads sharing KV pairs. LLaMA 3 uses 32 Q heads with 8 KV heads (4:1 ratio). This reduces KV cache memory by 4× with minimal quality loss. Critical for serving large models with long contexts: LLaMA 70B with 128K context window would need ~50GB just for KV cache without GQA.

---

## Connections

- **Attention Mechanism (fundamentals/01):** Causal self-attention is the core operation
- **Transformer Architecture (fundamentals/02):** GPT = decoder-only transformer
- **LLMs (5.llms/):** Fine-tuning (LoRA/QLoRA), RLHF, prompting all built on GPT-family
- **BERT Family (models/01):** Encoder-only counterpart — understanding vs generation
- **Encoder-Decoder (models/03):** T5/BART — better for seq2seq tasks
- **Efficient Transformers (models/04):** Flash Attention, quantization for production GPT

---

## Key Takeaway

GPT = decoder-only transformer trained with next-token prediction. Scale unlocks emergent capabilities: few-shot learning (GPT-3), instruction following (GPT-4). LLaMA 3 is the modern open-source go-to. Generation quality knobs: temperature (randomness), top-p (vocabulary diversity). Production knobs: KV cache (speed), GQA (memory), 4-bit quantization (fits on consumer GPU). The fundamental limit: inference is one-token-at-a-time, memory-bandwidth bound.

---

## Code Practice — Wired by Phase 6

- `code_practice/02_transformers/08_tiny_gpt/` — Tiny GPT + sampling
- `code_practice/02_transformers/11_hf_load/` — HF model load + inspect
- `code_practice/04_llms/01_load_generate/` — TinyLlama load + generate

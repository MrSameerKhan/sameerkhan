# Positional Encoding — All Details

> Phase 2, Session 4. Solving self-attention's blind spot: word order.

---

## Table of Contents

1. [Objective](#1-objective)
2. [The Problem — Self-Attention is Permutation-Equivariant](#2-the-problem--self-attention-is-permutation-equivariant)
3. [Sinusoidal PE — Original Transformer (2017)](#3-sinusoidal-pe--original-transformer-2017)
4. [Learned PE — BERT / GPT-2](#4-learned-pe--bert--gpt-2)
5. [RoPE — Llama / Mistral / Modern Default](#5-rope--llama--mistral--modern-default)
6. [Three-Way Comparison](#6-three-way-comparison)
7. [Implementation Details](#7-implementation-details)
8. [How to Run](#8-how-to-run)
9. [Expected Outputs](#9-expected-outputs)
10. [Why RoPE Beat Everything Else](#10-why-rope-beat-everything-else)
11. [Files in This Folder](#11-files-in-this-folder)
12. [✅ Actual Run Results](#-actual-run-results)
13. [Next Steps](#13-next-steps)

---

## 1. Objective

Self-attention has a critical blind spot: **it has no concept of word order**. Positional encoding (PE) is how we tell the model which token appears at which position.

You'll implement and compare three PE methods:
- **Sinusoidal** — fixed sin/cos pattern (original Transformer)
- **Learned** — embedding table, one vector per position (BERT, GPT-2)
- **RoPE** — rotation applied to Q/K vectors inside attention (Llama, Mistral, today's standard)

Plus a **proof demo** that without PE, attention is permutation-equivariant.

---

## 2. The Problem — Self-Attention is Permutation-Equivariant

```
Input:    [Sarah, Chen, works, in, Risk]
Shuffled: [Risk, works, Chen, in, Sarah]

Self-attention treats these as IDENTICAL sets of tokens.
Per-token outputs come back in shuffled order, but the underlying
representations are the same — no new information is encoded.
```

**Proof:**

For self-attention with input X, projections give Q = XW_q, K = XW_k, V = XW_v.
If we permute X by matrix P:

```
Q' = PXW_q = PQ
K' = PXW_k = PK
V' = PXW_v = PV

scores'   = Q'(K')^T / √d = (PQ)(PK)^T / √d = P scores P^T
weights'  = softmax(scores') = P weights P^T
output'   = weights' @ V' = P weights P^T P V = P output
```

The output is just **permuted in the same way as the input**. No new information is encoded.

**Live demo:** `train.py` runs `permutation_demo()` first — feeds the same tokens shuffled, sum-pooled outputs are identical to 7 decimal places.

For sets-of-words tasks (some classification) this might be OK. For sequence tasks (language modeling, translation, generation), this is fatal.

---

## 3. Sinusoidal PE — Original Transformer (2017)

```python
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

Each position gets a d_model-dimensional vector. Even dimensions use sin, odd use cos, with frequencies decreasing geometrically.

**Why sin/cos?**

1. **Unique per position** — different positions get different vectors
2. **Bounded** — values in [-1, 1], don't overwhelm token embeddings
3. **Relative position is computable** — PE(pos+k) = R_k @ PE(pos) for some rotation R_k. Attention can learn to detect "k positions apart."
4. **Extrapolates** — in principle works for sequences longer than training, since it's a formula not a lookup table.

**Adding to input:**

```python
x = embed(tokens) + sinusoidal_pe(positions)
```

That's it. The PE vector is added element-wise to the token embedding.

**Practical reality:** Vaswani 2017 used this. BERT/GPT-2 abandoned it (learned PE worked better). Came back via RoPE which is conceptually similar but rotated.

---

## 4. Learned PE — BERT / GPT-2

```python
self.pe = nn.Embedding(max_len, d_model)

# At forward time:
positions = torch.arange(seq_len).unsqueeze(0)
return x + self.pe(positions)
```

Each of `max_len` positions gets its own learnable vector. Look it up by position index.

**Pros:**
- Conceptually simple — just another embedding
- Often performs slightly better than sinusoidal on benchmarks
- Easy to extend (until you hit max_len)

**Cons:**
- **Cannot extrapolate** beyond max_len trained on
- More parameters (max_len × d_model)
- Harder to interpret what's been learned

**Used by:** BERT (max_len=512), GPT-2 (max_len=1024), T5 originally, then switched.

---

## 5. RoPE — Llama / Mistral / Modern Default

RoPE is **fundamentally different**. Instead of ADDING position info to input embeddings, it **ROTATES the Q and K vectors** in attention.

**The mechanism:**

For each pair of dimensions (q_{2i}, q_{2i+1}) in Q, rotate by angle θ_i = pos × 1/10000^(2i/d_head):

```
q_{2i}'   = q_{2i}   cos(θ) - q_{2i+1} sin(θ)
q_{2i+1}' = q_{2i+1} cos(θ) + q_{2i}   sin(θ)
```

Same for K. V is NOT rotated.

**Why this is genius:**

The dot product Q_pos1 · K_pos2 after RoPE rotation depends only on (pos1 - pos2) — the **relative position**. This means:

- Attention scores naturally encode relative positions
- The model doesn't need to "learn" relative position from sinusoidal patterns
- Extrapolation works well (RoPE-scaled methods like NTK and YaRN)
- It's parameter-free (uses cos/sin like sinusoidal)

**Position is in the MATH, not in the embedding.** With sinusoidal/learned, position info is added to embeddings → attention sees positions indirectly. With RoPE, position is directly encoded in the attention score computation.

**Used by:** Llama (1, 2, 3), Mistral, most open-source LLMs since 2022. Standard for new architectures.

---

## 6. Three-Way Comparison

| Aspect | Sinusoidal | Learned | RoPE |
|---|---|---|---|
| Year invented | 2017 | 2018 | 2021 |
| How it works | Add sin/cos to input | Add learned vector to input | Rotate Q, K inside attention |
| Parameters | 0 (fixed) | max_len × d_model | 0 (fixed, but cached) |
| Extrapolation | Theoretical | None | Good (with scaling tricks) |
| Captures relative pos | Indirectly | No | Directly |
| Where applied | Input embedding | Input embedding | Q and K inside attention |
| Pre-norm friendly | Yes | Yes | Yes |
| Used by | Original Transformer | BERT, GPT-2 | Llama, Mistral, Qwen |

---

## 7. Implementation Details

**Sinusoidal — built once, used as buffer:**

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(max_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(log(10000.0) / d_model))
pe[:, 0::2] = sin(position * div_term)
pe[:, 1::2] = cos(position * div_term)
self.register_buffer("pe", pe.unsqueeze(0))
```

Stored as `register_buffer` — moves with `.to(device)` but not a learned param.

**Learned — just an Embedding:**

```python
self.pe = nn.Embedding(max_len, d_model)
# At forward time:
positions = torch.arange(seq_len).unsqueeze(0)
return x + self.pe(positions)
```

**RoPE — pre-compute cos/sin tables, apply inside attention:**

```python
inv_freq  = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
positions = torch.outer(torch.arange(max_len).float(), inv_freq)
self.register_buffer("cos_cached", torch.cos(positions))
self.register_buffer("sin_cached", torch.sin(positions))

# In attention forward:
Q = self.rope(Q)
K = self.rope(K)
# scores = Q @ K^T — now naturally captures relative position
```

Critical detail: RoPE goes between Q/K projection and the scores computation. Not in the embedding step.

---

## 8. How to Run

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/04_pos_enc
python train.py
python predict.py --text "Sarah Chen works as Senior Analyst in Risk department."
python predict.py --text "Sarah Chen works as Senior Analyst in Risk department." --order-demo
```

`train.py`:
1. Runs the permutation demo (proof of why PE is needed)
2. Trains 4 variants: none, sinusoidal, learned, rope
3. Prints a comparison table
4. Saves the RoPE checkpoint

`predict.py --order-demo` runs the order-sensitivity demo — shuffles the input tokens 4 ways and shows different confidence levels.

---

## 9. Expected Outputs

### Permutation demo

```
Original : [3, 7, 11, 4, 9]
Shuffled : [9, 11, 3, 4, 7]

Original  sum-pooled output (first 4 dims): [0.123, -0.456, 0.789, -0.012]
Shuffled  sum-pooled output (first 4 dims): [0.123, -0.456, 0.789, -0.012]

✓ IDENTICAL — attention without PE is permutation-equivariant.
```

### Training comparison

```
PE Type      Params   Val Acc  Test Acc
none         29,893    1.0000    1.0000
sinusoidal   29,893    1.0000    1.0000
learned      33,989    1.0000    1.0000
rope         29,893    1.0000    1.0000
```

All variants hit 100% — expected on this set-of-words task.

### Order-sensitivity demo

```
'Sarah Chen works as Senior Analyst in Risk department.'  → employee 97.79%
'Risk department in works as Senior Analyst Sarah Chen.'  → employee 97.44%  ← Different!
```

Same class, different probability — proves the model IS using order info.

---

## ✅ Actual Run Results

*(MacBook M1 — device: mps, d_model=64, num_heads=8, vocab=185)*

### Permutation Demo — the proof

```
Original : [3, 7, 11, 4, 9]
Shuffled : [9, 11, 3, 4, 7]

Original  sum-pooled output (first 4 dims): [-0.0626, -0.1916, -0.0890, -0.2538]
Shuffled  sum-pooled output (first 4 dims): [-0.0626, -0.1916, -0.0890, -0.2538]
                                              ↑ identical to 7 decimal places

Max absolute difference: 2.24e-08  (float-precision noise, not a real difference)
✓ IDENTICAL — attention without PE is permutation-equivariant.
```

Mathematical proof in code. Self-attention without PE produces literally identical outputs for any permutation of the same tokens. This is why positional encoding exists. Without it, transformers couldn't distinguish "Sarah loves Chen" from "Chen loves Sarah."

---

### Training Comparison

```
PE Type        Params   Val Acc  Test Acc
none           28,549    1.0000    1.0000
sinusoidal     28,549    1.0000    1.0000
learned        61,317    1.0000    1.0000  ← +32,768 params (512 × d_model PE table)
rope           28,549    1.0000    1.0000
```

All variants hit 100% — expected on this bag-of-words classification task. The Acme class doesn't change with word order ("Premium Checking earns 2.5%" is "checking" regardless of arrangement).

PE differences only show on **order-dependent tasks**: language modeling, translation, generation. That's where RoPE pulls ahead.

---

### Order-Sensitivity Demo — RoPE IS using word order

```
'sarah chen works as senior analyst in risk department .'   → employee 95.33%  ← original
'department as chen senior risk sarah . in works analyst'   → employee 93.06%  ← shuffled
'chen works senior in analyst . risk sarah as department'   → employee 94.24%  ← shuffled
'as analyst works chen risk senior . sarah in department'   → employee 91.07%  ← shuffled
```

**Same words, different orders → different confidence levels** (95.33% → 91.07%, a 4.26% spread).

Without PE, all four would be **identical 95.33%**. The fact that the model is less confident on more-shuffled inputs proves it learned that natural word order looks like training data, and unusual orders look slightly suspicious.

The model still says "employee" with 91%+ confidence regardless — word *content* matters more than order for this task. But order is **detectable**, which is what we wanted to prove.

---

### Lessons

| Observation | What it means |
|---|---|
| Max diff 2.24e-08 on permutation demo | Float-precision noise, not a real difference — true equivariance proved |
| All 4 PE types hit 100% | Classification task is order-insensitive — expected |
| Learned PE: 61,317 vs 28,549 params | +32,768 = 512 positions × 64 d_model — extra embedding table |
| 95.33% → 91.07% order spread | RoPE is encoding word order — 4.26% confidence spread across 4 shuffles |
| MPS device used | MacBook M1 GPU engaged for PyTorch ops |

**The real-world implication:** The same RoPE mechanism that gave us a 4.26% confidence spread here is what makes LLMs sensitive to word order in generation:
- "The cat sat on the mat" → coherent
- "The mat sat on the cat" → coherent but different meaning
- "Mat the on sat cat the" → barely English

Without RoPE, GPT/Llama would generate word-salad because they couldn't preserve order.

---

## 10. Why RoPE Beat Everything Else

By 2024, ~all new LLMs use RoPE. Why?

1. **Relative position is what matters** — when reading text, what matters is "this word is 3 positions before that word" not "this word is at position 47." RoPE encodes relative position directly via dot product invariants.

2. **Better extrapolation** — Llama 1 trained on 2048 tokens, Llama 2 on 4096. RoPE-scaling methods (NTK, YaRN, position interpolation) let you go to 32K-128K context length without retraining everything.

3. **Parameter-free** — same flexibility as sinusoidal, no extra params to learn.

4. **Composable with KV cache** — works cleanly with autoregressive inference and KV caching.

5. **Empirically better** — head-to-head, RoPE typically beats both sinusoidal and learned by 1-2 BLEU / perplexity points on language modeling.

---

## 11. Files in This Folder

| File | Purpose |
|---|---|
| `model.py` | `SinusoidalPE`, `LearnedPE`, `RoPE` + `PEClassifier` with switchable PE |
| `data.py` | Same classification dataset (PyTorch tensor batching) |
| `train.py` | Permutation demo + train all 4 variants + comparison table |
| `predict.py` | Inference + order-sensitivity demo (shuffled variants) |
| `all_details.md` | This document |
| `checkpoints/pos_enc_rope.pt` | Saved RoPE model |

---

## 13. Next Steps

### Session 5 — Encoder Block

Now you have all the ingredients to build a transformer encoder block:

```
x → LayerNorm → MHA(+ PE) → +x (residual) → LayerNorm → FFN → +x (residual)
```

We'll stack PE + MHA + residual + LayerNorm + FFN. Then Session 6 stacks multiple of these blocks.

### Session 7-8 — Causal masking → Tiny GPT

The decoder block adds causal masking (prevent attending to future tokens), then we stack blocks → autoregressive language model.

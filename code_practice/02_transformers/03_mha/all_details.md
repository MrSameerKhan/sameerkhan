# Multi-Head Attention — All Details

> Phase 2, Session 3. **Single-head attention generalized to multiple parallel "perspectives".**

---

## Table of Contents

1. [Objective](#1-objective)
2. [The One-Sentence Idea](#2-the-one-sentence-idea)
3. [Math — Same as Session 02, Plus Reshape](#3-math--same-as-session-02-plus-reshape)
4. [Shape Walkthrough](#4-shape-walkthrough)
5. [Why Multiple Heads Help](#5-why-multiple-heads-help)
6. [The Output Projection W_O](#6-the-output-projection-w_o)
7. [Backward Pass — What Changes from Session 02](#7-backward-pass--what-changes-from-session-02)
8. [Head Entropy — Measuring Specialization](#8-head-entropy--measuring-specialization)
9. [Architecture](#9-architecture)
10. [How to Run](#10-how-to-run)
11. [Expected Outputs](#11-expected-outputs)
12. [Bridge to Real Transformers](#12-bridge-to-real-transformers)
13. [Files in This Folder](#13-files-in-this-folder)
14. [✅ Actual Run Results](#-actual-run-results)
15. [Next Steps](#15-next-steps)

---

## 1. Objective

Generalize scaled dot-product attention (Session 02) to run `num_heads` parallel attention computations on different slices of `d_model`. By the end you should be able to:
- Code multi-head attention (~80 lines)
- See different heads attend to different patterns
- Understand why MHA dominates production transformers
- See if it fixes the Session 02 misclassification bug

---

## 2. The One-Sentence Idea

Split `d_model` into `num_heads` smaller chunks. Each chunk computes its own attention in parallel. Concatenate the outputs.

That's it.

`d_model=32, num_heads=4` → each head sees `d_head=8` dimensions → its own attention pattern → its own contribution.

---

## 3. Math — Same as Session 02, Plus Reshape

Single-head (Session 02):
```python
scores  = (Q @ K^T) / √d_k
weights = softmax(scores)
output  = weights @ V
```

Multi-head (this session):
```python
# Same Q, K, V projections, but reshape adds a head dimension:
Q = Q.view(batch, seq, num_heads, d_head).transpose(1, 2)
#   shape:  (batch, num_heads, seq, d_head)

# ALL the same math, but per-head in parallel:
scores   = (Q @ K^T) / √d_k
weights  = softmax(scores)
attended = weights @ V

# Concat heads back to original shape:
attended = attended.transpose(1, 2).view(batch, seq, d_model)

# Final projection — mixes head outputs:
output = W_O(attended)
```

The MATH is identical. The only new thing is **reshape + transpose** to add the `num_heads` dimension, and a new **output projection W_O**.

---

## 4. Shape Walkthrough

With `d_model=32, num_heads=4, d_head=8`:

```
x:              (batch, seq, 32)
Q, K, V each:   (batch, seq, 32)     ← W_Q / W_K / W_V projections
Q, K, V each:   (batch, seq, 4, 8)   ← .view(batch, seq, num_heads, d_head)
Q, K, V each:   (batch, 4, seq, 8)   ← .transpose(1, 2)

scores:         (batch, 4, seq, seq)  ← 4 separate NxN matrices
weights @V:     (batch, 4, seq, 8)   ← per-head outputs
attended:       (batch, seq, 4, 8)   ← .transpose(1, 2)
attended:       (batch, seq, 32)     ← .view(batch, seq, d_model)
output:         (batch, seq, 32)     ← W_O(attended) — final projection
```

**Key trick:** reshape + transpose to add the `num_heads` dimension. Once it's a dimension, PyTorch/NumPy can run all heads in parallel.

---

## 5. Why Multiple Heads Help

**Single head limit:**

One attention head learns **one** way to compute query-key compatibility. It picks the dominant pattern in the data and ignores others.

We saw this in Session 02:
- The single head learned `rate + percent → loan-flavored content`
- When test sentence had "Premium Checking has 2.5 percent rate", that one head's pattern dominated → wrong prediction

**Multi-head fix:**

With 4 heads, each head has a chance to weight them appropriately:
- Head A: "Premium Checking" → checking
- Head B: "rate of X percent" → loan-flavor

If both heads' contributions get combined via W_O, the model has a chance to weight them appropriately. It **might** still fail — multi-head doesn't guarantee correctness. It just gives the model more capacity.

But it might still fail — multi-head doesn't guarantee correctness. It just gives the model more capacity. On only 160 training examples, capacity is well-allocated to specialization.

**Real-world evidence:**
- Vaswani 2017 ablation: single head → 13.2 BLEU, 8 heads → 26.4 (2× improvement on translation)
- BERT, GPT, Llama all use 8-32 heads in production
- More heads = more specialization (but also more params)

The model gets to learn **multiple complementary patterns simultaneously**. The final output `W_O(concat(heads))` lets the model combine signals from different subspaces.

---

## 6. The Output Projection W_O

**Single-head:**
```python
output = weights @ V    # (batch, seq, d_model)
```

**Multi-head:**
```python
output = W_O(concat(head_outputs))    # (batch, seq, d_model)
```

**Why the extra projection?**

When you concat 4 heads' outputs, you get a vector of size `d_model` again — but it's just stacked head outputs side-by-side. Without W_O, these subspaces never interact. Position i's Head-1 output never influences what Head-2 learned at position j.

`W_O` is a learnable `(d_model, d_model)` matrix that:
- Mixes information across heads
- Projects back to model space
- Gives the model another degree of freedom

It's small (~4K params for d_model=32) but functionally essential.

**Interview answer:** "W_O is the projection that makes multi-head actually multi-head — without it, you have h independent classifiers sitting next to each other, not h heads collaborating."

---

## 7. Backward Pass — What Changes from Session 02

Session 02 backward: `dZ → dA → dS → dQ,dK,dV → dW_Q,dW_K,dW_V → dX → dE`

Session 03 adds two steps before the per-head backward:

### New step 1: Output projection backward
```python
dW_O   = Z_cat.T @ dZ_out          # [d_model, d_model]
dZ_cat = dZ_out @ W_O.T            # [T, d_model]
```

### New step 2: Slice dZ_cat per head
```python
for k in range(n_heads):
    start = k * d_k
    end   = start + d_k
    dZ_k  = dZ_cat[:, start:end]   # [T, d_k] — this head's gradient slice
    # ... same per-head backward as Session 02 single-head ...
```

### Accumulate dX from all heads
```python
dX = zeros_like(X)
for k in range(n_heads):
    dX += dQ_k @ W_Q[k].T + dK_k @ W_K[k].T + dV_k @ W_V[k].T
```

The per-head backward is identical to Session 02 — just run h times with each head's own Q_k, K_k, V_k, A_k, and its slice of dZ_cat.
- Map this code directly to PyTorch's `nn.MultiheadAttention`

---

## 2. Why Multiple Heads?

Single-head attention produces one attention distribution A [T, T]. The model has one "relationship type" to learn.

**The problem with one head:**

```
"Sarah Chen works as Head of Loans in the Loans department."

One head must simultaneously capture:
  - Coreference:  "Loans" (position 6) ↔ "Loans" (position 9) — same entity
  - Role:         "Chen" → "Head" → "Loans" — who leads what
  - Syntax:       "works as" → modifier → head noun chain
  - Position:     "in the" → prepositional phrase marker

One [T,T] matrix can't specialize in all of these at once.
```

**Multiple heads = multiple relationship lenses:**

With h=4 heads:
- Head 1 might focus on exact-match / coreference
- Head 2 might focus on syntactic dependencies
- Head 3 might focus on semantic similarity
- Head 4 might capture positional patterns

No head is told which pattern to learn — specialization is emergent from training.

---

## 3. The Math — Multi-Head Attention

```
# Hyperparams
d_model = 32    # total model dimension
h       = 4     # number of heads
d_k     = 8     # per-head dimension  (d_model // h)

# Input
X ∈ ℝ^{T × d_model}

# Per-head projections (h separate weight matrices each)
For k = 1 ... h:
    Q_k = X W_Q_k        W_Q_k ∈ ℝ^{d_model × d_k}   → [T, d_k]
    K_k = X W_K_k        W_K_k ∈ ℝ^{d_model × d_k}   → [T, d_k]
    V_k = X W_V_k        W_V_k ∈ ℝ^{d_model × d_k}   → [T, d_k]

    S_k = Q_k K_k^T / √d_k                            → [T, T]
    A_k = softmax(S_k)                                 → [T, T]
    Z_k = A_k V_k                                      → [T, d_k]

# Concatenate all heads
Z_cat = [Z_1 | Z_2 | ... | Z_h]       → [T, h*d_k] = [T, d_model]

# Output projection — mixes head outputs
Z_out = Z_cat W_O         W_O ∈ ℝ^{d_model × d_model}  → [T, d_model]

# Pool + classify (same as Session 10)
pooled = mean(Z_out[real])             → [d_model]
logits = W_out @ pooled + b            → [n_classes]
```

**Same total compute as one big head:**
- One head with d_k=32: 3 × (32×32) = 3,072 projection params
- Four heads with d_k=8: 4 × 3 × (32×8) = 3,072 projection params
- Plus W_O: 32×32 = 1,024 (new)

Multi-head adds W_O but keeps everything else the same cost.

---

## 4. The W_O Output Projection

**Why W_O is essential — not optional:**

Each head k produces Z_k ∈ ℝ^{T × d_k} — a representation in a d_k-dimensional subspace. When we concatenate [Z_1 | Z_2 | Z_3 | Z_4], the result is a direct block-concatenation:

```
Z_cat[:, 0:8]  = Z_1  (Head 1's subspace)
Z_cat[:, 8:16] = Z_2  (Head 2's subspace)
Z_cat[:, 16:24] = Z_3 (Head 3's subspace)
Z_cat[:, 24:32] = Z_4 (Head 4's subspace)
```

Without W_O, these subspaces never interact. Position i's Head-1 output never influences what Head-2 learned at position j. The classifier receives four independent representations side by side with no mixing.

W_O = linear combination of all heads' outputs → every output dimension can reference any head's information:

```
Z_out[i, d] = Σ_{h=1}^{H} Σ_{j=0}^{d_k} W_O[h*d_k + j, d] * Z_h[i, j]
```

**Interview answer:** "W_O is the projection that makes multi-head actually multi-head — without it, you have h independent classifiers sitting next to each other, not h heads collaborating."

---

## 9. Architecture

```
token_ids  [T]
    ↓  E  [V, d_model]
X  [T, d_model]
    ┌──────────────────────────────────────────────┐
    │  Head 1           Head 2    Head 3    Head 4 │
    │  Q=XW_Q1          Q=XW_Q2   ...              │
    │  K=XW_K1          K=XW_K2                   │
    │  V=XW_V1          V=XW_V2                   │
    │  A1=softmax(...)  A2=...                     │
    │  Z1=A1@V1         Z2=...    Z3=...   Z4=...  │
    └──────────────────────────────────────────────┘
         ↓ concat
    Z_cat  [T, d_model]
         ↓ W_O  [d_model, d_model]
    Z_out  [T, d_model]
         ↓ mean pool
    pooled [d_model]
         ↓ W_out  [n_classes, d_model]
    logits [n_classes]
```

Parameter count (d_model=32, h=4, d_k=8, vocab≈185, n_classes=5):
- E:          185 × 32 = 5,920
- W_Q (×4):   4 × (32×8) = 1,024
- W_K (×4):   4 × (32×8) = 1,024
- W_V (×4):   4 × (32×8) = 1,024
- W_O:        32 × 32  = 1,024
- W_out:       5 × 32  =   160
- b_out:               =     5
- **Total: 10,181 params**

vs Session 10 (single head): 9,157 params. Difference = W_O (1,024) + extra organization overhead.

---

## 6. Backward Pass — What Changes from Session 10

Session 10 backward: dZ → dA → dS → dQ,dK,dV → dW_Q,dW_K,dW_V → dX → dE

Session 11 adds two steps before the per-head backward:

### New step 1: Output projection backward
```python
# Z_out = Z_cat @ W_O
dW_O   = Z_cat.T @ dZ_out          # [d_model, d_model]
dZ_cat = dZ_out @ W_O.T            # [T, d_model]
```

### New step 2: Slice dZ_cat per head
```python
for k in range(n_heads):
    start = k * d_k
    end   = start + d_k
    dZ_k  = dZ_cat[:, start:end]   # [T, d_k] — this head's gradient slice
    # ... same per-head backward as Session 10 single-head ...
```

### Accumulate dX from all heads
```python
# Each head contributes to dX — sum all contributions
dX = zeros_like(X)
for k in range(n_heads):
    dX += dQ_k @ W_Q[k].T + dK_k @ W_K[k].T + dV_k @ W_V[k].T
```

The per-head backward is identical to Session 10 — just run h times with the head's own Q_k, K_k, V_k, A_k, and its slice of dZ_cat.

---

## 8. Head Entropy — Measuring Specialization

A focused head (low entropy) concentrates attention on 1-2 tokens per query.
A spread head (high entropy) distributes attention nearly uniformly.

```python
# For head k, real-token submatrix A_k[real, real]:
row_entropies = -sum(A_k * log(A_k + eps), axis=-1)   # [T_real]
mean_entropy  = row_entropies.mean()
max_entropy   = log(T_real)                             # entropy of uniform dist
normalized    = mean_entropy / max_entropy              # 0=focused, 1=uniform
```

Specialized heads show low normalized entropy. If all heads have similar high entropy, they've collapsed into redundant attention patterns — a sign of training instability or insufficient data.

**In practice (real BERT/GPT heads):**
- Some heads develop syntactic roles (subject→verb attention)
- Some heads attend to previous/next tokens (positional)
- Some heads attend to `[CLS]` or separators (global summary)
- Some heads seem almost uniform (possibly redundant)

---

## 10. How to Run

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/03_mha
python train.py
python predict.py --text "Sarah Chen works as Head of Loans in the Loans department."
python predict.py --text "The Premium Checking has 2.5 percent annual interest rate."
python predict.py --text "All personal loans require credit score above 650."
```

---

## 11. Expected Outputs

### Training

```
Train: 160 | Val: 20 | Test: 20
Vocab: 185
Heads: 4  |  d_k per head: 8  |  Parameters: 10,181

Best val acc : 1.0000
Test acc     : 1.0000
```

### Inference — head specialization

Head specialization only emerges on ambiguous predictions. When the model is 100% confident, all heads spread uniformly (entropy norm = 1.00). When confidence drops (83.5%), heads diverge and each focuses on different signals.

---

## ✅ Actual Run Results

*(MacBook M1 — CPU, vocab=185, h=4, d_k=8)*

### Training

```
Train: 160 | Val: 20 | Test: 20
Vocab: 185
Heads: 4  |  d_k per head: 8  |  Parameters: 10,181

 Epoch |    loss |   train |     val |  time
----------------------------------------------
     5 |  0.0007 |  1.0000 |  1.0000 | 0.1s
    10 |  0.0001 |  1.0000 |  1.0000 | 0.1s
    ...
    50 |  0.0000 |  1.0000 |  1.0000 | 0.1s

Best val acc : 1.0000
Test acc     : 1.0000
```

Same convergence pattern as Session 02 single-head — 100% from epoch 5. Synthetic Acme data is too templated. The signal is in **attention patterns**, not accuracy.

---

### Sentence 1 — Employee (100% confident, all heads uniform)

```
Input    : Sarah Chen works as Head of Loans in the Loans department.
Predicted: employee  (100.0%)

Token             H1    H2    H3    H4
────────────────────────────────────────
  sarah           0.08  0.08  0.08  0.08
  chen            0.08  0.08  0.08  0.08
  works           0.08  0.08  0.08  0.08
  as              0.08  0.09  0.09  0.08
  head            0.08  0.08  0.08  0.08
  of              0.08  0.08  0.08  0.08
  loans           0.08  0.08  0.08  0.08
  in              0.08  0.08  0.09  0.09
  the             0.09  0.09  0.08  0.08
  loans           0.08  0.08  0.08  0.08
  department      0.08  0.09  0.08  0.08
  .               0.08  0.08  0.10  0.09

Head entropy:
  Head 1: entropy 2.480  (norm 1.00)  ████████████████████
  Head 2: entropy 2.484  (norm 1.00)  ████████████████████
  Head 3: entropy 2.482  (norm 1.00)  ████████████████████
  Head 4: entropy 2.482  (norm 1.00)  ████████████████████
```

All 4 heads at norm 1.00 — perfectly uniform across all 12 tokens. Signal is so unambiguous (only sentence type with "department", "works as") that no head needs to specialize.

---

### Sentence 2 — Checking (83.5% confident, heads diverge)

```
Input    : The Premium Checking has 2.5 percent annual interest rate.
Predicted: checking  (83.5%)  ← loan gets 16.5%

Token             H1    H2    H3    H4     Overall
──────────────────────────────────────────────────
  the             0.07  0.09  0.07  0.09   0.078
  premium         0.07  0.10  0.08  0.08   0.083
  checking        0.09  0.10  0.17  0.09   0.111  ← H3 focus
  has             0.07  0.09  0.07  0.10   0.082
  2               0.08  0.09  0.11  0.09   0.095
  .5              0.11  0.08  0.15  0.11   0.114  ← H1+H3 focus
  percent         0.15  0.08  0.08  0.09   0.100  ← H1 focus
  annual          0.08  0.09  0.07  0.09   0.084
  interest        0.11  0.09  0.07  0.08   0.088  ← H1 focus
  rate            0.08  0.08  0.07  0.10   0.083
  .               0.07  0.11  0.06  0.08   0.080

Head entropy:
  Head 1: entropy 2.272  (norm 0.95)  ███████████████████·  ← focused
  Head 2: entropy 2.387  (norm 1.00)  ████████████████████
  Head 3: entropy 2.238  (norm 0.93)  ███████████████████·  ← focused
  Head 4: entropy 2.385  (norm 0.99)  ████████████████████
```

**Head specialization emerges under ambiguity:**
- **H1** (norm 0.95): focuses on rate-related tokens — `percent` (0.15), `interest` (0.11), `.5` (0.11). Reading the financial metric.
- **H3** (norm 0.93): focuses on the product keyword — `checking` (0.17), `.5` (0.15), `2` (0.11). Anchoring on the class-discriminating token.
- **H2, H4** (norm ~1.00): stay uniform — acting as broad context aggregators.

This is exactly the theoretical prediction: under pressure (16.5% loan probability), heads split responsibilities.

---

### Sentence 3 — Policy (100% confident, all heads uniform)

```
Input    : All personal loans require credit score above 650.
Predicted: policy  (100.0%)

Token             H1    H2    H3    H4
────────────────────────────────────────
  all             0.11  0.11  0.11  0.11
  personal        0.12  0.11  0.11  0.11
  loans           0.11  0.11  0.11  0.11
  require         0.10  0.11  0.11  0.12
  credit          0.12  0.12  0.11  0.11
  score           0.11  0.11  0.11  0.11
  above           0.11  0.11  0.11  0.11
  650             0.11  0.11  0.11  0.11
  .               0.11  0.11  0.12  0.11

Head entropy:
  Head 1: entropy 2.195  (norm 1.00)  ████████████████████
  Head 2: entropy 2.197  (norm 1.00)  ████████████████████
  Head 3: entropy 2.196  (norm 1.00)  ████████████████████
  Head 4: entropy 2.196  (norm 1.00)  ████████████████████
```

100% confident → all heads uniform. "require" + "score" + "above" is a unique policy signature — no competition.

---

### Key Lessons

| Observation | What it means |
|---|---|
| 100% accuracy from epoch 5 | Same as single-head — data too templated |
| Parameters: 10,181 vs 9,157 | W_O adds 1,024 params (32×32) |
| All heads uniform at 100% confidence | No pressure → no specialization needed |
| H1 + H3 diverge at 83.5% confidence | Competition forces division of labor |
| H1 reads rate signals, H3 reads keyword | Emergent specialization — not hand-designed |
| H2 + H4 stay uniform even under pressure | Some heads act as broad context aggregators |

**The core insight:** head specialization is **pressure-driven**. When one representation is sufficient to classify with certainty, all heads collapse to uniform (maximum entropy). When competing classes have overlapping signals, heads split responsibilities — one anchors on the discriminating keyword, another reads supporting context. This is why real transformers (BERT, GPT) show diverse head roles: their training data is genuinely ambiguous.

---

## 13. Files in This Folder

| File | Purpose |
|---|---|
| `data.py` | Same classification task as Session 10 |
| `model.py` | `MultiHeadAttentionClassifier` — h heads, W_O, forward + backward |
| `train.py` | Adam training loop, val accuracy, best checkpoint |
| `predict.py` | Per-head attention bars + head entropy visualization |
| `all_details.md` | This document |
| `checkpoints/multihead_attn.pkl` | Saved best model |

---

## 12. Bridge to Real Transformers

You now have the EXACT multi-head attention used by GPT-4, Claude, Llama. The differences from production are:

| | This session | Production (GPT-4 / Llama-3) |
|---|---|---|
| num_heads | 4 | 32-64 |
| d_model | 32 | 4096-12288 |
| Stacked layers | 1 | 32-128 |
| Position encoding | None yet | RoPE / ALiBi |
| Normalization | None | LayerNorm (pre-norm before attention) |
| Residual | None | x + attention(x) |

The CORE math is identical. Everything else is plumbing around this MHA module.

In sessions 5-8 you'll add LayerNorm + residual + FFN (encoder block), then position encoding, then stack, then build tiny GPT.

---

## 15. Next Steps

### Session 4 — Positional Encoding

Self-attention has a critical limitation: it's permutation-equivariant. Shuffle the input tokens and the output (for each token) is the same — just permuted. We need to inject position information.

Three PE methods: Sinusoidal (original Transformer), Learned (BERT-style), RoPE (Llama-style — modern default).

### Session 5 — Encoder Block (PyTorch)

Stack the pieces built in Sessions 02-03 into the full encoder sublayer:

```
# Sublayer 1: Multi-Head Self-Attention + residual + LayerNorm
x = LayerNorm(x + MultiHeadAttn(x, x, x))

# Sublayer 2: Position-wise Feed-Forward + residual + LayerNorm
x = LayerNorm(x + FFN(x))

# Where FFN:
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2    (expand 4× then contract)
```

New components vs Sessions 10-11:
- **LayerNorm**: normalize across feature dim per position (not batch)
- **Residual connections**: x + sublayer(x) — stabilizes deep networks, enables gradient flow
- **FFN**: per-position MLP — adds non-linearity and capacity that attention lacks
- **Positional encoding**: attention has no inherent position sense — must add it explicitly

After Session 12, you have a complete transformer encoder layer. Stack 6 → BERT encoder. Stack with a decoder → GPT.

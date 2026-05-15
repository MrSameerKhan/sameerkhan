# Scaled Dot-Product Self-Attention — All Details

> Phase 2, Session 10. **The core of every transformer — built from scratch in NumPy.**

---

## Table of Contents

1. [Objective](#1-objective)
2. [From Bahdanau to Self-Attention](#2-from-bahdanau-to-self-attention)
3. [The Math — Scaled Dot-Product Attention](#3-the-math--scaled-dot-product-attention)
4. [Architecture](#4-architecture)
5. [Backward Pass — Manual Gradients](#5-backward-pass--manual-gradients)
6. [Masking Subtleties](#6-masking-subtleties)
7. [Visualization — What the T×T Matrix Shows](#7-visualization--what-the-tt-matrix-shows)
8. [How to Run](#8-how-to-run)
9. [Expected Outputs](#9-expected-outputs)
10. [Files in This Folder](#10-files-in-this-folder)
11. [Next Steps](#11-next-steps)

---

## 1. Objective

Build **scaled dot-product self-attention from scratch in NumPy** — the core operation inside every transformer. Train it on the same Acme sentence classification task as Session 8 (BiLSTM + Bahdanau), so you can directly compare the two.

By the end of this session you should be able to:
- Implement Q/K/V projections and scaled dot-product scoring by hand
- Derive and code the full backward pass through softmax and matrix products
- Explain the T×T attention matrix and what "self-attention" means
- Articulate exactly how this generalizes Session 8's Bahdanau attention

---

## 2. From Bahdanau to Self-Attention

### Session 8 — Bahdanau (additive, implicit query)

```
h_1, h_2, ..., h_T  ← BiLSTM hidden states

score_i = v^T · tanh(W · h_i)     ← single learned query vector v
alpha   = softmax(scores)          ← [T] weights sum to 1
output  = Σ alpha_i · h_i          ← [1, d_hidden] pooled vector
```

**Key constraints:**
- One implicit query (`v`) — the same for every input
- Additive scoring: `tanh(Wh)` then dot with `v`
- Sequential: BiLSTM must run left-to-right before attention
- Output: a single vector (one consumer)

### Session 10 — Self-Attention (multiplicative, explicit queries)

```
X = [x_1, x_2, ..., x_T]  ← embedding lookup (no RNN needed)

Q = X W_Q      [T, d_k]   ← every position generates its own query
K = X W_K      [T, d_k]   ← every position generates its own key
V = X W_V      [T, d_k]   ← every position generates its own value

S = Q K^T / sqrt(d_k)     [T, T]   ← all pairs scored simultaneously
A = softmax(S, dim=-1)    [T, T]   ← each row sums to 1
Z = A V                   [T, d_k] ← each position gets its own output
```

**Key advances:**
- T explicit queries (one per position, not one shared)
- Multiplicative scoring: dot product — faster on GPUs, scales to large d
- Parallel: no sequential dependency, all T positions computed simultaneously
- Output: T vectors (each position as both consumer and producer)

### Comparison table

| Property | Bahdanau (Sess. 8) | Self-Attention (Sess. 10) |
|---|---|---|
| Query | 1 implicit `v` vector | T explicit `q_i = x_i W_Q` |
| Score | additive `v^T tanh(Wh)` | multiplicative `q_i · k_j / √d` |
| Encoder needed | Yes (BiLSTM) | No (operates on raw embeddings) |
| Attention matrix | [T] vector | [T, T] matrix |
| Position updates | No (just pools) | Yes (each Z_i updated) |
| Parallelism | Sequential (LSTM) | Fully parallel |
| Transformer | No | Yes — this IS the transformer's inner op |

**Self-attention = Bahdanau generalized:** replace 1 query with T queries, replace additive score with dot product, remove the RNN.

---

## 3. The Math — Scaled Dot-Product Attention

Given input matrix X ∈ ℝ^{T × d_model}:

```
Step 1: Project to Q, K, V spaces
    Q = X W_Q     W_Q ∈ ℝ^{d_model × d_k}
    K = X W_K     W_K ∈ ℝ^{d_model × d_k}
    V = X W_V     W_V ∈ ℝ^{d_model × d_k}

Step 2: Compute attention scores
    S = Q K^T / √d_k              ∈ ℝ^{T × T}
    S[i, j] = similarity(query_i, key_j) / scale

    Why √d_k?
    If q,k ~ N(0,1), then q·k ~ N(0, d_k).
    Dividing by √d_k keeps the variance at 1 → softmax doesn't saturate.
    Without scaling: dot products grow with d_k → softmax pushes toward one-hot → gradients vanish.

Step 3: Mask padding (key-masking)
    S[:, pad_positions] = -∞      (pad keys get zero weight after softmax)

Step 4: Softmax over keys (row-wise)
    A = softmax(S, dim=-1)        ∈ ℝ^{T × T}
    A[i, j] = how much position i attends to position j

Step 5: Weighted sum of values
    Z = A V                        ∈ ℝ^{T × d_k}
    Z[i] = Σ_j A[i,j] · V[j]     ← position i's output = mix of all values

Step 6: Pool + classify (for classification)
    pooled = mean(Z[real positions])   ∈ ℝ^{d_k}
    logits = W_out @ pooled + b        ∈ ℝ^{n_classes}
    probs  = softmax(logits)
    loss   = -log(probs[y])
```

---

## 4. Architecture

```
token_ids  [T]
    ↓  Embedding lookup  E  [vocab, d_model]
X  [T, d_model]
    ↓  Three linear projections (shared input X)
    ├─ Q = X W_Q   [T, d_k]
    ├─ K = X W_K   [T, d_k]
    └─ V = X W_V   [T, d_k]
         ↓  S = Q K^T / √d_k   [T, T]
         ↓  mask padding → softmax(rows) → A  [T, T]
         ↓  Z = A V            [T, d_k]
         ↓  mean pool over real tokens
    pooled  [d_k]
         ↓  W_out  [n_classes, d_k]
    logits  [n_classes]
         ↓  softmax → cross-entropy
    loss
```

Parameter count (d_model=32, d_k=32, vocab≈200, n_classes=5):
- E:     200 × 32 = 6,400
- W_Q:    32 × 32 = 1,024
- W_K:    32 × 32 = 1,024
- W_V:    32 × 32 = 1,024
- W_out:   5 × 32 =   160
- b_out:         =     5
- **Total: ~9,637 params** (vs Session 8's ~96K BiLSTM+Attention — 10× smaller)

---

## 5. Backward Pass — Manual Gradients

This is the hardest part and the most educational. Full derivation:

### 1. Loss → logits
```python
dlogits = probs.copy()
dlogits[y] -= 1              # softmax + cross-entropy combined gradient
```

### 2. Classifier → pooled
```python
dW_out = outer(dlogits, pooled)   # [n_classes, d_k]
db_out = dlogits
dpooled = W_out.T @ dlogits       # [d_k]
```

### 3. Mean pool → Z
```python
dZ = zeros_like(Z)
dZ[mask == 1] = dpooled / T_real  # distribute equally to each real position
```

### 4. Attention output: Z = A @ V
```python
dA = dZ @ V.T      # [T, T]    chain: ∂Z/∂A
dV = A.T @ dZ      # [T, d_k]  chain: ∂Z/∂V
```

### 5. Softmax backward (row i)
```
A[i] = softmax(S[i])
dS[i,j] = A[i,j] * (dA[i,j] - Σ_k A[i,k] * dA[i,k])

Jacobian of softmax: J[j,k] = A[j] * (δ_{jk} - A[k])
So: dS[i] = A[i] * (dA[i] - (dA[i] · A[i]))
```
```python
for i in range(T):
    if mask[i]:
        ai  = A[i]
        dS[i] = ai * (dA[i] - (dA[i] * ai).sum())
```

### 6. Scale: S = Q K^T / √d_k
```python
dS /= sqrt(d_k)
```

### 7. Q K^T backward
```python
dQ = dS @ K      # [T, d_k]  chain: ∂S/∂Q
dK = dS.T @ Q    # [T, d_k]  chain: ∂S/∂K  (note: transpose because S=QK^T)
```

### 8. Projection backward
```python
dW_Q = X.T @ dQ
dW_K = X.T @ dK
dW_V = X.T @ dV
dX   = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T
```

### 9. Embedding backward (scatter)
```python
dE = zeros_like(E)
for i, tid in enumerate(token_ids):
    dE[tid] += dX[i]    # accumulate gradient for each token's embedding
```

**Critical insight:** `dK = dS.T @ Q` uses the **transpose** of dS, not dS itself. This is because `S = Q K^T` → `∂S/∂K = Q^T` per position, which accumulates as `dS^T @ Q`. Easy to get wrong.

---

## 6. Masking Subtleties

Two types of masking in self-attention (both different from Bahdanau):

### Key masking (always needed for padding)
```python
S[:, pad_positions] = -1e9    # pad tokens can't be attended to
```
→ softmax(-inf) = 0 → zero weight on pad keys. Same principle as Bahdanau.

### Query masking (optional for classification)
```python
S[pad_positions, :] = -1e9   # pad tokens don't generate valid queries
```
→ Their Z[i] values are undefined anyway (excluded from mean pool).

### Causal masking (Session 11 / GPT decoder)
```python
# Future positions can't be attended to (autoregressive)
S = where(causal_mask == 0, -1e9, S)
```
→ Upper triangle of S set to -inf. Not needed for classification, essential for generation.

---

## 7. Visualization — What the T×T Matrix Shows

In Session 8 (Bahdanau), attention was a **1D vector** [T] — one weight per token.

In Session 10 (self-attention), attention is a **2D matrix** [T, T]:
- `A[i, j]` = how much position i (query) attends to position j (key)
- Each row is a probability distribution over all positions

**For classification visualization**, we show **received attention** (column mean):
```python
recv_attn = A[real_rows, :].mean(axis=0)   # [T]
```
→ How much was each token queried by others on average?
→ Tokens with high received attention are "important" for the other positions' representations.

**Why column mean and not row mean?**
- Row mean would be 1/T for all positions (they all sum to 1 by construction)
- Column mean varies — some positions are popular keys (others look to them a lot)

---

## 8. How to Run

```bash
cd code_practice/02_transformers/10_self_attention
python train.py
python predict.py --text "Sarah Chen works as Head of Loans in the Loans department."
python predict.py --text "The Premium Checking has 2.5 percent annual interest rate."
python predict.py --text "All personal loans require credit score above 650."
```

---

## 9. Expected Outputs

### Training

```
Train: 160 | Val: 20 | Test: 20
Vocab: ~200
Parameters: ~9,637

Epoch  5 | loss X.XXXX | train X.XXXX | val X.XXXX | X.Xs
...
Epoch 50 | loss X.XXXX | train X.XXXX | val X.XXXX | X.Xs

Best val acc : X.XXXX
Test acc     : X.XXXX
```

### Inference

```
Input    : Sarah Chen works as Head of Loans in the Loans department.
Predicted: employee  (XX.X%)

Self-attention: received attention per token
  sarah              0.XXX  ████...
  chen               0.XXX  ████...
  works              0.XXX  ██...
  ...

Attention matrix A [T×T]:
  (query × key heatmap)
```

---

## ✅ Actual Run Results

*(MacBook M1 — NumPy CPU, no GPU needed)*

### Training

```
Train: 160 | Val: 20 | Test: 20
Vocab: 185
Parameters: 9,157

 Epoch |    loss |   train |     val |  time
----------------------------------------------
     5 |  0.0013 |  1.0000 |  1.0000 | 0.0s
    10 |  0.0002 |  1.0000 |  1.0000 | 0.0s
    ...
    50 |  0.0000 |  1.0000 |  1.0000 | 0.0s

Best val acc : 1.0000
Test acc     : 1.0000
```

**Saturated evaluation (again).** Same situation as Session 8 — synthetic Acme data is too templated for 100% to be surprising. The interesting part is the attention matrix, not the accuracy.

**Speed comparison:**
- Session 8 (BiLSTM + Bahdanau, PyTorch): 6.5s for attention variant
- Session 10 (self-attention, NumPy): 0.0s per epoch (sub-second total)

9,157 params vs ~96K — 10× fewer parameters, 100× faster training, same accuracy. The self-attention is completely parallel — no sequential LSTM bottleneck.

---

### Inference — Attention Analysis

#### 1. "Sarah Chen works as Head of Loans in the Loans department." → employee (99.9%)

```
Received attention per token:
  the                0.192  ████████
  as                 0.118  █████
  department         0.105  ████
  works              0.082  ███
  .                  0.066  ███
  sarah / chen / head  0.06  ██ each

Attention matrix highlights (A[i,j]):
  the     → the:     0.56   (attends 56% to itself)
  as      → the:     0.38   (strong pull toward "the")
  dept    → the:     0.34
  works   → the:     0.27
  of      → of:      0.40   (attends to itself — local anchor)
```

`the` accumulates 19% of received attention and 56% self-attention. Same "accumulator position" phenomenon as Session 8 where `as` dominated with 96%. The BiLSTM hidden state at `as` had absorbed left+right context; here, `the` at position 8 (before `loans department`) serves a similar role — its K vector is broadly similar to many Q vectors.

**Key insight:** self-attention discovers summary positions the same way BiLSTM attention does, but without any sequential processing — purely from the Q/K dot-product geometry.

---

#### 2. "The Premium Checking has 2.5 percent annual interest rate." → checking (99.2%)

```
Received attention per token:
  checking           0.302  ████████████
  percent            0.155  ██████
  rate               0.116  █████
  interest           0.109  ████
  has                0.070  ███

Attention matrix highlights:
  checking → checking:  1.00  (100% self-attention!)
  premium  → checking:  0.80  (80% to checking)
  2        → checking:  0.82  (82% to checking)
  the      → percent:   0.43  (43% to "percent")
  the      → interest:  0.37
```

This is the **most interpretable result.** `checking` attends 100% to itself — it has learned it is the single most discriminative token for this class. `premium` and `2` both attend ~80% to `checking`, pulling its representation into their own outputs.

Compare with Session 8: `Checking` got 84% Bahdanau weight (very similar). Both mechanisms find the same keyword — but self-attention reveals the inter-token structure: `premium` *pulls toward* `checking`, `2` *pulls toward* `checking`, the relationships are explicit in the T×T matrix.

---

#### 3. "All personal loans require credit score above 650." → policy (100.0%)

```
Received attention per token:
  require            0.421  █████████████████
  .                  0.096  ████
  all/loans/score/650  0.07  ███ each

Attention matrix highlights:
  require → require:  0.94  (94% self-attention!)
  all     → require:  0.42
  score   → require:  0.41
  above   → require:  0.41
  loans   → require:  0.25
```

`require` gets 42% of all received attention and attends 94% to itself. The model learned that policy sentences use "require" language — not the domain words (`loans`, `credit`, `650`).

**Compare with Session 8 (same sentence):** Bahdanau gave `.` 99% attention — the period as a summary accumulator. Self-attention here finds the actual semantic signal: `require` is the policy-language word. This is meaningfully more interpretable.

**Why the difference?**
- Session 8 (Bahdanau): pooled from BiLSTM hiddens where the final hidden state (near `.`) absorbed all context — attention picked the richest accumulator
- Session 10 (self-attention): no sequential accumulation — each position only knows its own embedding and dot-product similarity to others. `require` has a distinctive embedding that no other class uses → high self-similarity → high self-attention → dominates received attention

---

### What this session teaches (beyond accuracy)

| | Session 8 Bahdanau | Session 10 Self-Attention |
|---|---|---|
| Attention shape | 1D: [T] | 2D: [T, T] |
| Reveals | which position BiLSTM summarized | which tokens attend to which |
| Most informative token | accumulator position (as, .) | discriminative keyword (require, checking) |
| Why it differs | LSTM context bleeds into late positions | each position only knows its own embedding geometry |
| Interpretability | "model focused on X" | "X attends to Y" — richer story |

**The T×T matrix is the key advantage.** You can read token relationships directly. In the checking example, the matrix tells you `premium → checking` (modifier pointing to head noun) and `2 → checking` (numeric value pointing to its account type). Bahdanau's 1D vector can't show this.

---

### Saturated evaluation — real-world fix (same as Session 8)

All variants score 100% because the Acme templates are too regular. The attention patterns are the signal here, not accuracy. To get discriminating accuracy numbers:
- Add paraphrased / adversarial sentences
- Mix categories (a loan sentence with policy language)
- Reduce training set to 50 examples

---

## 10. Files in This Folder

| File | Purpose |
|---|---|
| `data.py` | Word tokenizer, vocab, classification dataset, split |
| `model.py` | `SelfAttentionClassifier` — Q/K/V projections, scaled dot-product, forward + backward |
| `train.py` | Adam training loop, val accuracy, best checkpoint |
| `predict.py` | CLI — classify sentence + T×T attention visualization |
| `all_details.md` | This document |
| `checkpoints/self_attn.pkl` | Saved best model |

---

## 11. Next Steps

### Session 11 — Multi-Head Attention (NumPy)

Extend Session 10 to multiple heads:

```
# h heads, each with d_k = d_model / h
for each head k:
    Q_k = X W_Q_k    [T, d_k]
    K_k = X W_K_k    [T, d_k]
    V_k = X W_V_k    [T, d_k]
    Z_k = softmax(Q_k K_k^T / sqrt(d_k)) @ V_k    [T, d_k]

# concat all heads + project
Z_concat = concat(Z_1, ..., Z_h)      [T, d_model]
Z_out    = Z_concat @ W_O             [T, d_model]
```

**Why multiple heads?**
- Each head can specialize in a different relation: syntactic, semantic, positional
- Jointly attending to multiple subspaces gives richer representations
- Same computation cost as one big head (just split and parallelize)

### Session 12 — Transformer Encoder Block (PyTorch)

Multi-head attention + Add & Norm + FFN + Add & Norm:

```
x = LayerNorm(x + MultiHeadAttn(x, x, x))   # self-attention sublayer
x = LayerNorm(x + FFN(x))                    # feed-forward sublayer
```

This is the complete transformer encoder layer — stack 6-12 of these and you have BERT's encoder.

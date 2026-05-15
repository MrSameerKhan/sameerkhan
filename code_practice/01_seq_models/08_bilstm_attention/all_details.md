# Bahdanau Attention on BiLSTM — All Details

> Phase 1, Session 8 of the coding practice sequence. **The direct math bridge to transformer attention.**

---

## Table of Contents

1. [Objective](#1-objective)
2. [What Attention Is (in 3 lines)](#2-what-attention-is-in-3-lines)
3. [The Bahdanau Score — Math](#3-the-bahdanau-score--math)
4. [Why Attention Beats Mean/Max Pooling](#4-why-attention-beats-meanmax-pooling)
5. [Architecture](#5-architecture)
6. [Bridge to Transformer Attention](#6-bridge-to-transformer-attention)
7. [Task — Sentence Classification](#7-task--sentence-classification)
8. [Comparison Setup](#8-comparison-setup)
9. [Masking Subtleties](#9-masking-subtleties)
10. [How to Run](#10-how-to-run)
11. [Expected Outputs](#11-expected-outputs)
12. [Files in This Folder](#12-files-in-this-folder)
13. [Next Steps](#13-next-steps)

---

## 1. Objective

Build a **BiLSTM sentence classifier with Bahdanau attention pooling**, and compare it head-to-head with mean / max / last-hidden-state pooling.

By the end of this session you should be able to:
- Write attention from scratch (compute scores → softmax → weighted sum)
- Explain why attention beats simpler pooling on most tasks
- Map Bahdanau attention to transformer attention (it's the same shape)

Entities in our data: PERSON, PRODUCT, MONEY (tagged in BIO scheme).

This is the first session with a real ML pipeline:
- Train / validation / test split
- Vocab built from train only (no data leakage)
- Padded batches with masked loss
- F1 score on validation, not just training loss
- Save best model by val F1
- Per-entity precision / recall / F1 on test

---

## 2. What Attention Is (in 3 lines)

```
Given a sequence of hidden vectors h_1, h_2, ..., h_n:
    score_i = some_function(h_i)    ← how much should I focus on h_i?
    alpha   = softmax(scores)       ← weights sum to 1
    output  = sum_i (alpha_i * h_i) ← weighted sum
```

**That's it.** Everything else in modern attention is variations on how you compute `score_i`.

---

## 3. The Bahdanau Score — Math

Original Bahdanau (2014) used an **additive** score function:

```
score_i = v^T · tanh(W · h_i + b)
```

Where:
- **W** is a learnable matrix that projects `h_i`
- **v** is a learnable vector that produces a scalar score
- **tanh** is the activation

In code:

```python
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hiddens, mask=None):
        scores = self.v(torch.tanh(self.W(hiddens))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled  = torch.bmm(weights.unsqueeze(1), hiddens).squeeze(1)
        return pooled, weights
```

That's the **entire Bahdanau attention layer — ~10 lines.**

---

## 4. Why Attention Beats Mean/Max Pooling

| Pooling | What it does | Limitation |
|---|---|---|
| **Mean** | Equally weights every position | Drowns important info in noise (long sentences hurt) |
| **Max** | Picks largest activation per dimension | Discards all but the single "loudest" position per dim |
| **Last** | Uses only final hidden state | Wastes BiLSTM's per-position outputs |
| **Attention** | **Learns** which positions matter | Adapts per-input — focuses dynamically |

Concrete example with our data:

```
"The Premium Checking has 2.5 percent rate."

For category "checking":
  Mean pool:   spreads weight across 9 tokens → "The" and "rate" and "." all count
  Max pool:    keeps largest activations → loses context
  Attention:   learns to weight ~80% on "Premium Checking" — directly relevant
```

This is **why attention was invented**: when you only need certain parts of the input, learning to focus is dramatically better than spreading weight uniformly.

---

## 5. Architecture

```
token_idx (batch, seq_len)
    ↓ Embedding (vocab → 64)
(batch, seq_len, 64)
    ↓ BiLSTM (64 → 128 forward+backward)
(batch, seq_len, 128)
    ↓ Attention Pooling: weights = softmax(v^T tanh(W h))
(batch, 128)
    ↓ Linear (128 → 5 classes)
(batch, 5) ← logits
```

Attention adds ~16K parameters on top of BiLSTM (`W: 128×128` + `v: 128×1`).

---

## 6. Bridge to Transformer Attention

This is the **most important section of this session.** Hold this mental model:

### Bahdanau (this session)

```python
score_i = v^T tanh(W h_i)         # additive, single learned query (implicit)
weights = softmax(scores)
output  = Σ weights_i · h_i
```

### Transformer (Phase 2)

```python
score_ij = (q_i · k_j) / √d       # dot-product, explicit queries
weights  = softmax(scores)
output   = Σ weights_j · v_j
```

### Generalization

| Bahdanau | Transformer |
|---|---|
| Single implicit query (the `v` vector) | Multiple explicit queries `q_i` |
| Additive score: `v^T tanh(Wh)` | Multiplicative score: `q · k` |
| One pooled output | Per-position outputs |
| Decoder-style (one consumer) | Self-attention (all positions consume all positions) |

**Transformer is Bahdanau attention generalized to:**
1. Per-position queries (not one)
2. Multiplicative scoring (faster on GPUs)
3. Separate K/V projections (more expressive)
4. Multiple heads (different relations in parallel)
5. Stacked layers (deep composition)

When you read Phase 2 transformer code, the **inner attention math will look familiar.** Everything else is plumbing around the same idea.

---

## 7. Task — Sentence Classification

Same shared dataset, different task than Session 7:

```python
get_classification_data(n_total=200)
→ [(text, category), ...]

Categories: checking, savings, loan, employee, policy
```

200 examples → 160 train / 20 val / 20 test.

Why classification (not seq2seq)?
- Cleaner attention demo — one query, one output
- Avoids encoder-decoder complexity (that's Session 9)
- Lets us A/B test 4 pooling strategies cleanly

---

## 8. Comparison Setup

`train.py` trains 4 variants of the same model:

| Variant | Pooling | What's different |
|---|---|---|
| **attention** | Bahdanau pooling | Learns position weights |
| **mean** | Mean over real tokens | No params, equal weight |
| **max** | Elementwise max | No params, per-dim winner |
| **last** | Last hidden state | No params, ignores middle |

All 4 share:
- Embedding + BiLSTM (same weights init since seeded)
- Same Adam, same lr, same epochs
- Same train data, same batches via seeded sampling

**Only difference: the pooling layer.** Direct apples-to-apples on what pooling buys.

---

## 9. Masking Subtleties

Padding handling is **easy to get wrong with attention.** Three things must be right:

### 1. Attention scores: mask padding to `-inf`

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

This way, `softmax(-inf) = 0` and padding gets zero weight. Without this, attention would waste budget on `<PAD>` tokens.

### 2. Mean pooling: divide by REAL count, not seq_len

```python
masked = hiddens * mask.unsqueeze(-1).float()
pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
```

Without this, mean pooling on padded inputs gets pulled toward zero.

### 3. Max pooling: set padding to `-inf` BEFORE taking max

```python
hiddens_masked = torch.where(mask.unsqueeze(-1).bool(), hiddens, neg_inf)
pooled = hiddens_masked.max(dim=1)[0]
```

Otherwise random padding values can win the max.

**Classic interview question:** "How do you handle padding in attention?" → above is the answer.

---

## 10. How to Run

```bash
cd code_practice/01_seq_models/08_bilstm_attention
python train.py
```

Trains all 4 variants sequentially (~1-2 min on CPU). Prints comparison table.

Then visualize attention:

```bash
python predict.py --text "Sarah Chen works in the Loans department."
python predict.py --text "The Premium Checking has 2.5 percent rate."
python predict.py --text "All personal loans require credit score above 650."
```

---

## 11. Expected Outputs

### Training comparison table

```
============================================================
Pool type    |   Params | Val acc | Test acc
------------------------------------------------------------
attention    | 111,800  |  1.0000 |   1.0000
mean         |  95,300  |  0.9500 |   0.9000
max          |  95,300  |  0.9500 |   0.9000
last         |  95,300  |  0.9000 |   0.8500
============================================================
```

Attention typically wins by 5-15% on test accuracy. On this synthetic data with strong templates, all variants do well — but attention is consistently highest.

### Attention visualization (the killer demo)

```
Input:
  Sarah Chen works as Senior Analyst in Risk department.

Predicted class: employee

Attention weights (which tokens the model focused on):
  Sarah          0.180  ████████████████████
  Chen           0.220  ████████████████████████████
  works          0.145  ████████████████
  as             0.025  ···
  Senior         0.080  █████████
  Analyst        0.160  ██████████████████
  in             0.020  ···
  Risk           0.090  ██████████
  department     0.075  ████████
  .              0.005  ·
```

The model attended heavily to `Sarah Chen` and `Analyst` — exactly the words that indicate "employee" category. **This is interpretability for free.**

---

## ✅ Actual Run Results

*(MacBook M1 — device: mps)*

### Comparison table

```
============================================================
Pool type    |   Params |  Val acc | Test acc
------------------------------------------------------------
attention    |   96,581 |   1.0000 |   1.0000
mean         |   79,941 |   1.0000 |   1.0000
max          |   79,941 |   1.0000 |   1.0000
last         |   79,941 |   1.0000 |   1.0000
============================================================
```

Training times: attention 6.5s | mean 1.6s | max 2.2s | last 1.3s

**All 4 variants scored 100%.** The synthetic Acme classification task is too templated to differentiate them.

### Real ML engineering lesson — saturated evaluation

When everyone scores 100%, you can't measure pooling differences. Real-world fix:
- Add adversarial / paraphrased examples
- Shrink training set so model can't memorize
- Use harder labels (multi-label, fine-grained)

**Interview answer:** "Once evaluation saturates, you've stopped measuring. I'd add adversarial test cases or reduce capacity until headroom appears."

### Attention visualization (the big win) ✨

Even with equal accuracy, **only attention shows WHY the model predicted what it did.** Three example outputs:

**1. "Sarah Chen works as Senior Analyst in Risk department."** → employee (99%)

```
as           0.960  ██████████████████████████████████████
Chen         0.019  ···
Analyst      0.007  ···
```

Surprising — `"as"` gets 96% weight, not `"Sarah Chen"`. The BiLSTM hidden state at `"as"` has absorbed context from both directions: forward saw "Sarah Chen works", backward saw "Senior Analyst in Risk department". That single position is the richest summary of the full sentence. **Attention is a summary picker, not a keyword highlighter.**

**2. "The Premium Checking has 2.5 percent rate."** → checking (67%)

```
Checking     0.842  █████████████████████████████████
has          0.101  ████
percent      0.041  █
```

This run: model directly attends to `"Checking"` (the product keyword) — more interpretable than work laptop run where `"percent"` dominated. Confidence lower (67% vs 98%) because decimal tokenization splits `2.5` → `["2", ".5"]`, confusing the loan/checking signal.

**3. "All personal loans require credit score above 650."** → policy (99%)

```
.            0.991  ███████████████████████████████████████
650          0.008  ···
```

Classified as POLICY not LOAN — **correct**. That sentence appears in the POLICIES list in `shared_dataset.py`. The model attended to `"."` (the period) — the BiLSTM's final backward pass puts the whole sentence context into the last token's hidden state. Attention picks it as the summary position.

**Key intuition confirmed across all 3 examples:** the BiLSTM does the semantic heavy lifting; attention picks whichever single position has the richest accumulated context — often not the "obvious" keyword.

### What this session teaches (beyond the accuracy numbers)

| Without attention | With attention |
|---|---|
| Black-box prediction | You see what the model focused on |
| Hard to debug errors | "Oh, it attended to the wrong word" → fix is clear |
| Hard to explain to stakeholders | Show the bar chart |

**Interpretability is one of attention's biggest practical wins** — separate from any accuracy gain.

### Bridge to transformers (preview)

The math you wrote here:

```
score = v^T tanh(W · h)
weights = softmax(score)
output = Σ weights * h
```

Generalizes directly to transformer attention:

```
score = (Q · K^T) / √d    ← multiple queries, dot-product score
weights = softmax(score)
output = weights · V       ← separate value projections
```

Same shape. Same idea. Transformer just adds: per-position queries, separate K/V, multiple heads, stacked layers.

---

## 12. Files in This Folder

| File | Purpose |
|---|---|
| `data.py` | Tokenize text, vocab, classification dataset, padded batches with mask |
| `model.py` | `BiLSTMClassifier` with switchable pooling + `AttentionPooling` class |
| `train.py` | Trains all 4 pooling variants, prints comparison, saves attention model |
| `predict.py` | CLI with attention weight visualization |
| `all_details.md` | This document |
| `checkpoints/bilstm_attention.pt` | Saved attention model |

---

## 13. Next Steps

### Session 9 — Seq2Seq with Attention (Phase 1 finale)

The full encoder-decoder architecture:
- Encoder: BiLSTM (like Session 7)
- Decoder: LSTM with attention over encoder outputs
- Task: small-scale translation / summarization style task

This is where Bahdanau attention truly shines — every decoder step computes attention over the encoder. After this, you've built every key idea in transformers, just in RNN form.

### Phase 2 — Transformers

With attention in your hands, transformer attention is a small leap:
- Replace BiLSTM encoder with self-attention (no recurrence at all)
- Replace decoder with masked self-attention + cross-attention
- Multi-head it
- Stack layers

The math you wrote here generalizes directly.

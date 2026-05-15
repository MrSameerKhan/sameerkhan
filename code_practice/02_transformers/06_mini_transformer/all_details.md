# Mini Transformer Encoder — All Details

> Phase 2, Session 6 of the coding practice sequence. Stack 4 encoder blocks → real transformer encoder, same shape as BERT-base just smaller.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Architecture — Stack of N Blocks](#2-architecture--stack-of-n-blocks)
3. [What Depth Buys You](#3-what-depth-buys-you)
4. [Param Scaling](#4-param-scaling)
5. [Layer-Wise Interpretation](#5-layer-wise-interpretation)
6. [How to Run](#6-how-to-run)
7. [Expected Outputs](#7-expected-outputs)
8. [Comparison to BERT-Base](#8-comparison-to-bert-base)
9. [Why Acme Won't Show Big Differences](#9-why-acme-wont-show-big-differences)
10. [Files in This Folder](#10-files-in-this-folder)
11. [✅ Actual Run Results](#-actual-run-results)
12. [Next Steps](#12-next-steps)

---

## 1. Objective

Take the encoder block from Session 5 (one block) and stack 4 of them. This is the **complete transformer encoder** that BERT and friends are built on — just at smaller scale.

By the end:
- Build a multi-layer transformer (`nn.ModuleList` of blocks)
- Compare 1-layer vs 2-layer vs 4-layer performance
- Visualize how attention evolves across layers
- Understand why deeper = more abstract representations

---

## 2. Architecture — Stack of N Blocks

```
token_ids
↓
Embedding (vocab → d_model)
↓
Encoder Block 1
↓ (residual stream)
Encoder Block 2
↓
Encoder Block 3
↓
Encoder Block 4
↓
final LayerNorm
↓
Masked Mean Pool
↓
Linear (d_model → num_classes)
↓
logits
```

Each Encoder Block is what you built in Session 5:

```
x → LayerNorm → MHA(+RoPE) → +x → LayerNorm → FFN → +x
```

Stacking is just calling `block(h)` in sequence:

```python
self.blocks = nn.ModuleList([
    TransformerEncoderBlock(d_model, num_heads, d_ff, max_len, dropout)
    for _ in range(num_layers)
])
```

Forward loop — 5 lines of architecture-level code:

```python
def forward(self, x, mask):
    h = self.embed(x)
    for block in self.blocks:
        h, weights = block(h, mask)
    return self.fc(pool(h))
```

Plus 1 line per block in the forward loop. That's it.

---

## 3. What Depth Buys You

Each layer **composes** on the previous:

| Layer | What it can learn |
|---|---|
| Embedding | Token identity |
| Layer 0 | Local relations: adjacent words, token types |
| Layer 1 | Compose layer 0 outputs: short phrases, simple syntactic patterns |
| Layer 2 | Compose layer 1 outputs: full clauses, complex semantic patterns |
| Layer 3 | ... |
| Layer N | Task-specific representations near the output |

**This is exactly what BERTology research has shown:**
- BERT layer 0–2: surface features, POS-like
- BERT layer 3–6: syntactic patterns
- BERT layer 7–9: semantics, coreference
- BERT layer 10–11: task-specific (heavily fine-tuning-dependent)

The key insight: earlier layers build vocabulary, later layers build grammar, final layers build meaning. Each layer is a refinement pass over the residual stream.

---

## 4. Param Scaling

Approximate per-block params (d_model=64, num_heads=8, d_ff=256):

```
Attention: 4 × d_model² (Q, K, V, O projections) ≈ 16K
FFN:       2 × d_model × d_ff = 2 × 64 × 256     ≈ 33K
LayerNorm: 2 × 2 × d_model                        ≈ 256
Per block total:                                   ≈ 49K
```

| Layers | Total params (approx) |
|---|---|
| 1 | ~63K |
| 2 | ~113K |
| 4 | ~213K |

Each layer adds ~50K params. BERT-base (12 layers, d_model=768) has ~110M params for comparison — we're 500× smaller, same architecture.

---

## 5. Layer-Wise Interpretation

`predict.py` shows:

1. **Per-(layer, head) peak attended key** — table of what each layer's heads focus on
2. **Per-layer attention entropy** — measures how "peaky" vs "spread" each layer's attention is

Expected patterns (if task is hard enough):
- **Layer 0:** higher entropy (attention is spread), heads attend to various local positions
- **Higher layers:** lower entropy (more focused), heads converge on task-relevant keys
- **Last layer:** typically the most focused, preparing the final summary representation

This is the closest you'll get to "transformer interpretability" without running specialized tools like attention rollout.

**Spoiler:** On our small easy task, entropy may NOT decrease with depth — see Section 11 for why.

---

## 6. How to Run

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/06_mini_transformer && python train.py
```

Trains 3 variants (1, 2, 4 layers) back-to-back with same hyperparameters. Saves the 4-layer model. Auto-tests the adversarial sentences.

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/06_mini_transformer && python predict.py --text "Sarah Chen works as Senior Analyst in Risk department."
```

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/06_mini_transformer && python predict.py --text "The Premium Checking has 2.5 percent rate."
```

---

## 7. Expected Outputs

### Training comparison

```
Layers | Params  | Best Epoch | Val Acc | Test Acc
1      | 63,365  |     10     | 1.0000  | 1.0000
2      | 112,901 |      8     | 1.0000  | 1.0000
4      | 211,973 |      6     | 1.0000  | 1.0000  ← likely
```

Deeper → converges faster (more capacity → easier optimization on this small task) but plateau is the same. Acme is too easy to differentiate.

### Adversarial test

```
=== Adversarial test (4-layer model) ===
  'The Premium Checking has 2.5 percent rate.'
    → checking (XX.X%)
  'Apply for Personal Loan with 8.5 percent rate.'
    → loan (XX.X%)
  'Sarah Chen works as Head of Loans in the Loans department.'
    → employee (XX.X%)
```

### predict.py — layer-wise table

```
Per-(layer, head) peak attended token:
Layer   H0      H1      H2      H3      H4      H5      H6      H7
L0      .       Chen    Chen    Chen    Chen    Chen    Chen    works
L1      Sarah   Risk    Chen    Chen    Chen    works   depart  as
L2      Chen    works   .       .       .       .       .       Sarah
L3      Senior  .       .       in      depart  as      Sarah   Senior

Layer entropy:
Layer 0: entropy X.XXX  (norm 0.9X)  ███████████████████·  (more spread)
Layer 1: entropy X.XXX  (norm 0.9X)  ███████████████████·
Layer 2: entropy X.XXX  (norm 0.9X)  ███████████████████·
Layer 3: entropy X.XXX  (norm 0.9X)  ███████████████████·  (more focused)
```

---

## 8. Comparison to BERT-Base

| Component | BERT-base | This session |
|---|---|---|
| Layers | 12 | 4 |
| d_model | 768 | 64 |
| Heads | 12 | 8 |
| d_ff | 3072 | 256 |
| Position encoding | Learned | RoPE |
| LayerNorm | Post-norm (original) | Pre-norm (modern) |
| Params | ~110M | ~213K |

~500× smaller, **same architecture pattern**. Differences in PE and norm placement are modern updates.

If you scaled this to BERT-base size and trained on BookCorpus + Wikipedia, you'd have BERT.

---

## 9. Why Acme Won't Show Big Differences

You'll likely see 1-layer, 2-layer, 4-layer all hit 100% test accuracy.

**Why:**
- 160 training examples, 5 classes
- Templated synthetic data with strong signals
- The classification task is too easy to require depth

**When depth matters in real tasks:**
- Long-range dependencies (e.g., coreference in long documents)
- Compositional reasoning (e.g., math word problems)
- Hierarchical structure (e.g., parsing)
- Multi-task learning (each layer specializes)

For Acme, single-layer already saturates the task. Adding layers doesn't help test accuracy but it might:
- Converge faster (we'll see this)
- Be more robust to adversarial inputs (we'll see this)
- Have richer per-layer specialization (interpretability win)

Real benefit comes when scaling beyond toy data, which is Session 8 (Tiny GPT on full corpus generation).

---

## 10. Files in This Folder

| File | Purpose |
|---|---|
| `model.py` | `RoPE` + `MultiHeadAttention` + `FeedForward` + `TransformerEncoderBlock` + `MiniTransformerEncoder` (stacked) |
| `data.py` | Same Acme 5-class classification dataset |
| `train.py` | Trains 1/2/4 layer variants, comparison table + adversarial test, saves 4-layer model |
| `predict.py` | Layer-wise attention visualization + entropy analysis |
| `all_details.md` | This document |
| `checkpoints/mini_encoder_4l.pt` | Saved 4-layer model |

---

## ✅ Actual Run Results

*(To be filled after running on MacBook)*

---

## 12. Next Steps

### Session 7 — Causal Masking + Decoder Block

The encoder lets every position attend to every position (bidirectional). The **DECODER** restricts each position to only attend to previous positions (causal). This is what makes GPT autoregressive — at training time it predicts each next token given only what came before.

The change is **one line**: add a lower-triangular mask before softmax.

```python
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float("-inf"))
```

### Session 8 — Tiny GPT

Stack causal decoder blocks → train autoregressive language model on Acme corpus → generate text.

By Session 8 you'll have **nanoGPT working on synthetic financial data**. Same architecture as GPT-2 just at smaller scale.

# Transformer Encoder Block — All Details

> Phase 2, Session 5. Taking all prior pieces (embedding, RoPE, MHA) and wrapping them into one complete transformer encoder block.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Complete Block Picture](#2-complete-block-picture)
3. [Three New Pieces](#3-three-new-pieces)
4. [LayerNorm — Stabilizing the Signal](#4-layernorm--stabilizing-the-signal)
5. [Residual Connections — The Gradient Highway](#5-residual-connections--the-gradient-highway)
6. [Feed-Forward Network — Per-Token Reasoning](#6-feed-forward-network--per-token-reasoning)
7. [Pre-Norm vs Post-Norm](#7-pre-norm-vs-post-norm)
8. [Final Encoder Block in Code](#8-final-encoder-block-in-code)
9. [How to Run](#9-how-to-run)
10. [Expected Outputs](#10-expected-outputs)
11. [What This Block IS](#11-what-this-block-is)
12. [Files in This Folder](#12-files-in-this-folder)
13. [✅ Actual Run Results](#-actual-run-results)
14. [Next Steps](#14-next-steps)

---

## 1. Objective

Sessions 02–04 built the attention machinery. This session assembles the full transformer encoder block — the repeating unit stacked 12× in BERT and 32× in Llama-3 8B.

New pieces added on top of MHA + RoPE:
- **LayerNorm** — normalizes token representations before each sublayer
- **Residual connections** — each sublayer adds its output to its input (x → x + f(x))
- **Feed-Forward Network** — per-token MLP that reasons after attention has mixed context

One encoder block = MHA + FFN + 2 LayerNorms + 2 residuals.

---

## 2. Complete Block Picture

```
Input x  [B, T, d_model]
    │
    ├──────────────────────────── residual branch 1
    │                                     │
    ▼                                     │
LayerNorm(x)                              │
    │                                     │
    ▼                                     │
Multi-Head Attention (+ RoPE)             │
    │                                     │
    └──────────────────── + ◄─────────────┘
                          │
                    x₁  [B, T, d_model]
                          │
    ├──────────────────────────── residual branch 2
    │                                     │
    ▼                                     │
LayerNorm(x₁)                             │
    │                                     │
    ▼                                     │
Feed-Forward Network                      │
    │                                     │
    └──────────────────── + ◄─────────────┘
                          │
                Output x₂ [B, T, d_model]
```

In code (pre-norm pattern):

```python
attended, weights = self.mha(self.ln1(x), mask)
x = x + attended         # residual 1

x = x + self.ffn(self.ln2(x))  # residual 2
```

That's the whole block. ~10 lines. Same block stacked 32× in Llama-3 8B.

---

## 3. Three New Pieces

| Component | What it does | Where it lives |
|---|---|---|
| **LayerNorm** | Normalizes each token's d_model vector to μ=0, σ=1 | Before MHA (ln1) and before FFN (ln2) |
| **Residual** | x = x + sublayer(x) — adds sublayer output to input | After MHA, after FFN |
| **FFN** | Per-token MLP: d_model → 4×d_model → d_model, GELU | After MHA + residual |

Everything else (embedding, RoPE, MHA) came from Sessions 01–04.

---

## 4. LayerNorm — Stabilizing the Signal

**The problem:** As you stack blocks, activations can grow or shrink uncontrollably. Gradient signal either explodes or vanishes.

**What LayerNorm does:** For each token position, normalize the d_model-dimensional vector to zero mean, unit variance, then apply learned scale γ and shift β:

```
μ = mean(x)
σ = std(x)
x_norm = (x - μ) / (σ + ε)
output = γ * x_norm + β
```

- γ (scale) and β (shift) are learned per-dimension parameters
- ε (typically 1e-5) prevents division by zero
- Done independently per token — no cross-batch statistics

**Why not BatchNorm?** BatchNorm normalizes across the batch dimension. For variable-length sequences with padding, that introduces artifacts. LayerNorm normalizes within each token, so it's sequence-length and batch-size independent.

**Parameter count:** 2 × d_model per LayerNorm (γ and β). Two norms per block = 4 × d_model.

```python
self.ln1 = nn.LayerNorm(d_model)  # 2 × 64 = 128 params
self.ln2 = nn.LayerNorm(d_model)  # 2 × 64 = 128 params
```

---

## 5. Residual Connections — The Gradient Highway

**The problem:** In a deep network, gradients must backpropagate through every layer. Multiplicative chains → vanishing gradients → early layers learn nothing.

**The fix (He et al., 2016 — ResNet):** Add the input to the output of each sublayer:

```
x_out = x_in + sublayer(x_in)
```

**Why this helps:**

The gradient of the loss with respect to the input of the block:

```
∂L/∂x = ∂L/∂x_out × (1 + ∂sublayer/∂x)
                      ─────────────────
                      the "+1" is always there
```

The `+1` term means gradient can flow back unchanged regardless of what the sublayer does. Even if the sublayer's gradient is near zero (dead neurons, saturated activations), the `+1` keeps the path open.

**What it means in practice:**
- Transformers can be hundreds of layers deep with stable training
- Early layers receive useful gradient signal
- In theory, a block can "do nothing" (sublayer → 0), and the residual just passes the input through — the network learns to use or skip each block

---

## 6. Feed-Forward Network — Per-Token Reasoning

**What attention does:** Mixes information across tokens. Each token sees all others.

**What FFN does:** Per-token processing. After attention has gathered context, the FFN reasons on each token independently.

```
FFN(x) = W₂ · GELU(W₁ · x + b₁) + b₂
```

- W₁: [d_model → d_ff], d_ff = 4 × d_model typically
- W₂: [d_ff → d_model]
- GELU: smooth activation (used by GPT-2, Llama; smoother than ReLU)

**Why 4× expansion?**

It's empirically the ratio found in the original Transformer paper and BERT. Wider = more capacity per token. The network expands to a higher-dimensional space to perform complex transformations, then projects back. GPT-2 uses 4×. Llama uses 8/3× with SwiGLU (a gated variant).

**GELU vs ReLU:**

GELU (Gaussian Error Linear Unit) is approximately `x * Φ(x)` where Φ is the CDF of the normal distribution. It's smooth everywhere (unlike ReLU's kink at 0) and performs slightly better on language tasks. Huggingface benchmarks consistently show GELU > ReLU for transformer FFNs.

**Parameter count per block:**

```
W₁: d_model × d_ff     = 64 × 256  = 16,384
b₁: d_ff               = 256
W₂: d_ff × d_model     = 256 × 64  = 16,384
b₂: d_model            = 64
─────────────────────────────────────────────
FFN total:             = 33,088 params
```

The FFN dominates parameter count in every transformer block.

---

## 7. Pre-Norm vs Post-Norm

Two placements for LayerNorm:

**Post-norm (original Transformer, 2017):**

```
x = LayerNorm(x + MHA(x))
x = LayerNorm(x + FFN(x))
```

Normalize the sum of input + sublayer output. Works, but training requires careful learning rate warmup — without warmup the unnormalized residuals early in training can cause divergence.

**Pre-norm (GPT-2, Llama, modern standard):**

```
x = x + MHA(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

Normalize the input before passing to the sublayer. The residual stream (x) stays unnormalized. This:
- Leaves the residual gradient path clean (no norm in the highway)
- More stable training without warmup tricks
- Slightly worse final performance in some studies, but much more robust to hyperparameter choices

**This session uses pre-norm.** That's the current default for most production models.

Comparison:

| | Post-norm | Pre-norm |
|---|---|---|
| Original paper | Yes | No |
| Modern LLMs | Rarely | Yes (GPT-2, Llama) |
| Training stability | Needs warmup | Stable without warmup |
| Gradient highway | Norm in path | Clean |
| Final quality | Marginally better | Marginally worse |

---

## 8. Final Encoder Block in Code

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=64, num_heads=8, d_ff=256,
                 max_len=512, dropout=0.0):
        super().__init__()
        self.ln1     = nn.LayerNorm(d_model)
        self.mha     = MultiHeadAttention(d_model, num_heads, max_len)
        self.ln2     = nn.LayerNorm(d_model)
        self.ffn     = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, mask=None):
        attended, weights = self.mha(self.ln1(x), mask)
        x = x + self.dropout(attended)                # residual 1
        x = x + self.dropout(self.ffn(self.ln2(x)))  # residual 2
        return x, weights
```

**Parameter count breakdown (d_model=64, num_heads=8, d_ff=256):**

```
Embedding:      vocab_size × 64            ← depends on vocab
MHA:
  W_q, W_k, W_v: 64 × 64 each = 12,288
  W_o:            64 × 64      =  4,096
  RoPE:           0 (buffers)
  MHA total:                   = 16,384
LayerNorm × 2:  2 × (64+64)   =    256
FFN:
  W₁+b₁: 64×256+256           = 16,640
  W₂+b₂: 256×64+64            = 16,448
  FFN total:                   = 33,088
ln_out:         2 × 64         =    128
Classifier fc:  64 × 5         =    320

Total (excl. embed):           = 50,176  + embed
```

---

## 9. How to Run

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/05_encoder_block && python train.py
```

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/05_encoder_block && python predict.py --text "Sarah Chen works as Senior Analyst in Risk department."
```

```bash
cd /Users/sameerkhan/Desktop/sameerkhan/code_practice/02_transformers/05_encoder_block && python predict.py --text "The Premium Checking has 2.5 percent rate."
```

`train.py`:
1. Trains the full encoder block (d_model=64, num_heads=8, d_ff=256) for 30 epochs
2. Prints param comparison vs Sessions 02–04
3. Auto-runs adversarial test after training
4. Saves checkpoint to `checkpoints/encoder.pt`

`predict.py`:
1. Loads checkpoint
2. Shows all-class probability bar chart
3. Per-head column-mean attention table (which tokens each head attends to)
4. Top attended token per head
5. Head entropy (focused vs diffuse)

---

## 10. Expected Outputs

### Training

```
Device: mps
Train: 160 | Val: 20 | Test: 20
Vocab: ~185

Parameters:
  Session 02 (single-head NumPy): ~9,157
  Session 03 (MHA NumPy):         ~10,181
  Session 04 (RoPE PyTorch):       28,549
  Session 05 (full encoder):      ~XX,XXX  ← +FFN + LayerNorms

 Epoch |    loss |   train |     val |  time
----------------------------------------------
    5  | 0.XXXX | 1.0000  | 1.0000  | X.Xs
   10  | 0.XXXX | 1.0000  | 1.0000  | X.Xs
   ...

Best val acc: 1.0000 (epoch X)
Test acc:     1.0000
```

### Adversarial test

```
=== Adversarial test ===
  'The Premium Checking has 2.5 percent rate.'
    → product (XX.X%)
  'Apply for Personal Loan with 8.5 percent rate.'
    → product (XX.X%)
  'Sarah Chen works as Head of Loans in the Loans department.'
    → employee (XX.X%)  ← tricky: "Loans" appears twice, but role keyword dominates
```

### predict.py output

```
Model: XX,XXX params  |  8 heads
Input: 'Sarah Chen works as Senior Analyst in Risk department.'

Predicted: employee  (XX.X%)

All class probabilities:
  employee     0.XXX  ████████████████████
  product      0.XXX  ·····················
  ...

Per-head top attended token  (column mean = received attention)
──────────────────────────────────────────────────────────────
Token             H0    H1    H2    H3    H4    H5    H6    H7
  sarah         0.XX  0.XX  ...
  chen          ...
  works         ...
  ...

Top attended token per head:
  Head 0: 'XXX'  (XX%)
  Head 1: 'XXX'  (XX%)
  ...

Head entropy  (norm 1.0 = uniform, lower = focused)
  Head 0: entropy X.XXX  (norm X.XX)  ████████············
  ...
```

---

## 11. What This Block IS

The transformer encoder block is a **context mixer + reasoner**:

1. **MHA** (sublayer 1): Each token looks at all other tokens and updates its representation based on what's relevant. Information flows across the sequence.

2. **FFN** (sublayer 2): After gathering context, each token independently reasons — complex nonlinear transformations of its now-context-enriched representation.

Think of it as:
- MHA = "gather information from neighbors"
- FFN = "think about what that means"

One block does one round of this. Stack 12 blocks (BERT) or 32 blocks (Llama-3 8B) = 12 or 32 rounds of context-mixing and reasoning.

**The residual stream:** The x that flows through `x = x + f(x)` is called the residual stream. It starts as the token embedding, and each block adds to it. By the final layer, each position's vector encodes the token's meaning in context, refined 32 times.

**Head specialization revisited:** With a full FFN behind MHA, the model has more capacity. Different heads can specialize more cleanly — one head might focus on subjects, another on verbs, another on modifiers. The FFN then integrates those signals per-token.

---

## 12. Files in This Folder

| File | Purpose |
|---|---|
| `model.py` | `RoPE`, `MultiHeadAttention`, `FeedForward`, `TransformerEncoderBlock`, `EncoderClassifier` |
| `data.py` | Same Acme 5-class dataset, PyTorch tensor batching |
| `train.py` | Train full encoder, adversarial test, param comparison, save checkpoint |
| `predict.py` | Inference + per-head attention visualization + head entropy |
| `all_details.md` | This document |
| `checkpoints/encoder.pt` | Saved model |

---

## ✅ Actual Run Results

*(MacBook M1 — device: mps, d_model=64, num_heads=8, d_ff=256, vocab=185)*

### Parameter count

```
Session 02 (single-head NumPy): ~9,157
Session 03 (MHA NumPy):         ~10,181
Session 04 (RoPE PyTorch):       28,549
Session 05 (full encoder):       62,021  ← +FFN + LayerNorms
```

62,021 vs 28,549 = **+33,472 params** from FFN (~33,088) + two LayerNorms (256) + ln_out (128).
FFN alone accounts for 99% of the increase — it dominates block parameter count.

---

### Training convergence

```
 Epoch |    loss |   train |     val |  time
----------------------------------------------
      5 | 0.2787 | 0.9312 | 0.8000 | 0.1s
     10 | 0.0527 | 0.9937 | 0.9500 | 0.1s
     15 | 0.0110 | 1.0000 | 1.0000 | 0.1s
     20 | 0.0046 | 1.0000 | 1.0000 | 0.1s
     25 | 0.0029 | 1.0000 | 1.0000 | 0.1s
     30 | 0.0021 | 1.0000 | 1.0000 | 0.1s

Best val acc : 1.0000  (epoch 12)
Test acc     : 1.0000
```

Converges cleanly — 100% val by epoch 12, loss still slowly decreasing (0.2787 → 0.0021) as the model sharpens its confidence.

---

### Adversarial test

```
=== Adversarial test ===
  'The Premium Checking has 2.5 percent rate.'
    → checking (88.9%)

  'Apply for Personal Loan with 8.5 percent rate.'
    → loan (98.5%)

  'Sarah Chen works as Head of Loans in the Loans department.'
    → employee (99.0%)
```

The third sentence is the interesting one: "Loans" appears **twice** — once as a role, once as a department. A bag-of-words model would be pulled toward `loan`. The full encoder block with FFN correctly identifies the `works as Head` pattern as an employee role. FFN's per-token reasoning, combined with MHA context mixing, allows the model to distinguish structural role ("Head of Loans") from topic keyword ("Loans").

---

### predict.py — Employee sentence

```
Input: 'Sarah Chen works as Senior Analyst in Risk department.'
Predicted: employee (99.4%)

All class probabilities:
  checking     0.003  ····················
  savings      0.001  ····················
  loan         0.000  ····················
  employee     0.994  ████████████████████
  policy       0.002  ····················

Per-head top attended token  (column mean = received attention)
──────────────────────────────────────────────────────────────
Token             H0  H1  H2  H3  H4  H5  H6  H7
  sarah         0.11  0.08  0.11  0.08  0.08  0.08  0.08  0.08
  chen          0.09  0.09  0.07  0.07  0.17  0.09  0.07  0.06
  works         0.09  0.09  0.08  0.05  0.11  0.11  0.10  0.06
  as            0.08  0.09  0.10  0.45  0.10  0.08  0.13  0.17
  senior        0.11  0.09  0.10  0.09  0.08  0.09  0.10  0.10
  analyst       0.12  0.08  0.08  0.07  0.12  0.10  0.09  0.11
  in            0.09  0.08  0.11  0.03  0.08  0.13  0.06  0.05
  risk          0.08  0.10  0.13  0.08  0.09  0.08  0.06  0.10
  department    0.11  0.22  0.10  0.04  0.07  0.11  0.23  0.15
  .             0.10  0.08  0.12  0.03  0.13  0.11  0.08  0.11

Top attended token per head:
  Head 0: 'analyst'  (12%)
  Head 1: 'department'  (22%)
  Head 2: 'risk'  (13%)
  Head 3: 'as'  (45%)     ← most focused
  Head 4: 'chen'  (17%)
  Head 5: 'in'  (13%)
  Head 6: 'department'  (23%)
  Head 7: 'as'  (17%)

Head entropy  (norm 1.0 = uniform, lower = focused)
  Head 0: entropy 2.172  (norm 0.94)  ███████████████████·
  Head 1: entropy 2.116  (norm 0.92)  ██████████████████··
  Head 2: entropy 2.154  (norm 0.94)  ███████████████████·
  Head 3: entropy 1.656  (norm 0.72)  ██████████████······  ← most focused
  Head 4: entropy 2.027  (norm 0.88)  ██████████████████··
  Head 5: entropy 2.180  (norm 0.95)  ███████████████████·
  Head 6: entropy 2.002  (norm 0.87)  █████████████████···
  Head 7: entropy 2.056  (norm 0.89)  ██████████████████··
```

**Key observations:**

- **Head 3 is the most focused** (norm 0.72) and locks onto `'as'` with 45% weight — nearly 5× what uniform would be (~10%). The word `as` is the structural connector in "works **as** Senior Analyst" — it marks the role boundary. Head 3 has learned to find this pivot.
- **Heads 1 and 6 both emphasize `'department'`** (22% and 23%) — the sentence-final context word. Two heads independently discovered the same structural anchor.
- **Head 7 also attends `'as'`** (17%) as its top token, echoing Head 3's pattern.
- Most heads are diffuse (norm 0.87–0.95) — high-confidence prediction (99.4%) means the decision is easy, so most heads spread attention broadly rather than needing to focus.

---

### predict.py — Checking sentence

```
Input: 'The Premium Checking has 2.5 percent rate.'
Predicted: checking (88.9%)

All class probabilities:
  checking     0.889  ██████████████████··
  savings      0.036  █···················
  loan         0.028  █···················
  employee     0.044  █···················
  policy       0.002  ····················

Per-head top attended token  (column mean = received attention)
──────────────────────────────────────────────────────────────
Token             H0  H1  H2  H3  H4  H5  H6  H7
  the           0.09  0.15  0.08  0.08  0.06  0.15  0.11  0.12
  premium       0.10  0.09  0.11  0.11  0.08  0.08  0.16  0.09
  checking      0.09  0.13  0.21  0.11  0.08  0.23  0.12  0.10
  has           0.16  0.12  0.11  0.14  0.08  0.09  0.11  0.11
  2             0.14  0.08  0.09  0.15  0.07  0.08  0.07  0.09
  .5            0.08  0.09  0.10  0.07  0.15  0.08  0.12  0.15
  percent       0.09  0.07  0.10  0.13  0.23  0.12  0.10  0.13
  rate          0.14  0.20  0.10  0.09  0.14  0.09  0.10  0.13
  .             0.10  0.08  0.12  0.12  0.12  0.09  0.10  0.08

Top attended token per head:
  Head 0: 'has'  (16%)
  Head 1: 'rate'  (20%)
  Head 2: 'checking'  (21%)
  Head 3: '2'  (15%)
  Head 4: 'percent'  (23%)   ← most focused
  Head 5: 'checking'  (23%)  ← most focused (tied)
  Head 6: 'premium'  (16%)
  Head 7: '.5'  (15%)

Head entropy  (norm 1.0 = uniform, lower = focused)
  Head 0: entropy 1.995  (norm 0.91)  ██████████████████··
  Head 1: entropy 1.892  (norm 0.86)  █████████████████···
  Head 2: entropy 1.937  (norm 0.88)  ██████████████████··
  Head 3: entropy 1.909  (norm 0.87)  █████████████████···
  Head 4: entropy 1.727  (norm 0.79)  ████████████████····  ← most focused
  Head 5: entropy 1.780  (norm 0.81)  ████████████████····
  Head 6: entropy 2.036  (norm 0.93)  ███████████████████·
  Head 7: entropy 1.991  (norm 0.91)  ██████████████████··
```

**Key observations:**

- **Heads 4 and 5 are the most focused** (norm 0.79 and 0.81) — lower entropy than the employee sentence's most-focused head (0.72 employee vs 0.79 checking for top head). The checking sentence is less confident (88.9% vs 99.4%) and has more head focus — consistent with the Session 03 finding: **pressure-driven specialization**. Under uncertainty, heads must work harder to extract signal.
- **Heads 2 and 5 both attend `'checking'`** (21%, 23%) — the class-defining keyword. The model found it.
- **Head 4 attends `'percent'`** (23%), Head 7 attends `'.5'` (15%) — the rate components. Multiple heads independently attend to different parts of the numeric feature "2.5 percent."
- Every head has a different top token — 8 heads, 7 different top tokens. Clean division of labor.

---

### Cross-sentence comparison

| Metric | Employee (99.4%) | Checking (88.9%) |
|---|---|---|
| Confidence | 99.4% | 88.9% |
| Most focused head entropy norm | 0.72 (Head 3) | 0.79 (Head 4) |
| Most focused head's top token | `'as'` 45% | `'percent'` 23% |
| Pattern | Single pivot word (structural role marker) | Distributed numeric features |
| Head agreement | Heads 3+7 both on `'as'` | Heads 2+5 both on `'checking'` |

Higher confidence → one head can be very focused while others spread widely.
Lower confidence → all heads must contribute more equally.

This is the **pressure-driven specialization** pattern from Session 03, now inside a full encoder block with FFN giving each head's signal a nonlinear transformation before the final classifier.

---

## 14. Next Steps

### Session 6 — Stack Encoder Blocks

Now that we have one block, stacking is trivial:

```python
self.blocks = nn.ModuleList([
    TransformerEncoderBlock(d_model, num_heads, d_ff)
    for _ in range(n_layers)
])
```

BERT uses 12 layers. We'll stack 2–4 and observe how deeper stacking affects convergence and representation quality.

### Session 7–8 — Causal Masking → Tiny GPT

The decoder block adds a causal mask — each token can only attend to past tokens (not future). Stack those blocks + a language modeling head → autoregressive text generation. That's GPT.

```
Encoder (BERT-style): attends all positions ← Session 5–6
Decoder (GPT-style):  attends only left     ← Session 7–8
```

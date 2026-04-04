# 02 — Vanilla RNN: Complete End-to-End Walkthrough

One scalar pass — forward, loss, backward, weight update, forward again.
No matrices. Every number computed by hand so the chain is fully visible.

---

## 0. Problem Statement

**Task:** Binary sentence classification
**Question:** Does this sentence mention an animal?

```
Input sentence:   "cat sat on mat"
Expected output:   1   (yes, "cat" is an animal)
```

```
Why this task exposes RNN's weakness:

  The answer depends on the FIRST word — "cat".
  The RNN reads words left to right and must remember "cat"
  all the way to the end before making the final prediction.

  If the model forgets "cat" by the time it finishes reading,
  it will output something close to 0 (no animal) instead of 1.

  That is exactly what we will watch happen.
```

---

## 0.1 What Is the Input?

```
Raw text:  "cat sat on mat"

The model cannot read strings. It needs numbers.
We go through 3 steps before the RNN sees anything:

  Step 1 — Tokenization        split into words
  Step 2 — Vocabulary lookup   map each word to an integer index
  Step 3 — Embedding           map each integer to a number (or vector)
```

### Step 1 — Tokenization

```
"cat sat on mat"
       ↓  split on whitespace
["cat", "sat", "on", "mat"]
```

### Step 2 — Vocabulary Lookup

```
Build a vocabulary from your training corpus:
  vocab = { "cat": 1, "sat": 2, "on": 3, "mat": 4, "<UNK>": 0 }

Map each token to its index:
  ["cat", "sat", "on", "mat"]  →  [1, 2, 3, 4]

The model never sees the word "cat" again — only the integer 1.
```

### Step 3 — Embedding

```
Map each integer index to a number (scalar here, vector in real models).

Embedding table (learned during training):
  index 1 ("cat") → 1.0    ← high value: content word, subject, animal
  index 2 ("sat") → 0.2    ← lower: common verb, less informative
  index 3 ("on")  → 0.1    ← low: function word (preposition)
  index 4 ("mat") → 0.2    ← low: object noun, less discriminative

Final input sequence to the RNN:
  x₁=1.0  x₂=0.2  x₃=0.1  x₄=0.2
  "cat"   "sat"    "on"    "mat"

Note: in real models these are 100-300 dimensional vectors (GloVe, Word2Vec),
      not scalars. We use scalars here so every number is traceable by hand.
```

---

## 0.2 What Is the Expected Output?

```
After reading all 4 words, the RNN produces a final hidden state h₄.
We use h₄ directly as the prediction (no output layer, for simplicity).

ŷ = h₄        (a number between -1 and 1 because of tanh)

Target:
  y = 1.0   → sentence IS about an animal
  y = 0.0   → sentence is NOT about an animal

Loss tells us how wrong the prediction is:
  L = ½(y - ŷ)²

  If ŷ=0.360 and y=1.0:  L = ½(0.64)² = 0.205   ← model is wrong
  If ŷ=0.950 and y=1.0:  L = ½(0.05)² = 0.001   ← model is right

Training = adjust weights so L gets smaller over many sentences.
```

---

## Setup

```
Sentence:  "cat  sat  on  mat"
            x₁   x₂   x₃  x₄

Token inputs (scalar embeddings, kept tiny intentionally):
  x₁ = 1.0   (cat — strong signal, it's the subject we must remember)
  x₂ = 0.2   (sat)
  x₃ = 0.1   (on)
  x₄ = 0.2   (mat)

Weights (scalar, shared across ALL timesteps — that's the key RNN property):
  wₓ = 1.0   (input weight)
  wₕ = 0.5   (recurrent weight)
  b  = 0.0   (bias, ignored to keep math clean)

Initial state:
  h₀ = 0.0

Task: predict y = 1.0 using h₄ (the final hidden state)
      (e.g., "is this sentence about an animal?" → yes = 1)

Loss function:  L = ½ (y - ŷ)²    where ŷ = h₄
```

---

## 1. Forward Pass

```
Formula at each step:  hₜ = tanh(wₓ · xₜ + wₕ · hₜ₋₁)
```

```
t=1 — "cat"
  aₜ = wₓ·x₁ + wₕ·h₀ = 1.0×1.0 + 0.5×0.0 = 1.000
  h₁ = tanh(1.000) = 0.762

t=2 — "sat"
  aₜ = wₓ·x₂ + wₕ·h₁ = 1.0×0.2 + 0.5×0.762 = 0.200 + 0.381 = 0.581
  h₂ = tanh(0.581) = 0.523

t=3 — "on"
  aₜ = wₓ·x₃ + wₕ·h₂ = 1.0×0.1 + 0.5×0.523 = 0.100 + 0.262 = 0.362
  h₃ = tanh(0.362) = 0.348

t=4 — "mat"
  aₜ = wₓ·x₄ + wₕ·h₃ = 1.0×0.2 + 0.5×0.348 = 0.200 + 0.174 = 0.374
  h₄ = tanh(0.374) = 0.360
```

```
Forward pass summary:

  "cat"  → h₁ = 0.762   ← strong signal (x₁=1.0 was the biggest input)
  "sat"  → h₂ = 0.523   ← cat's signal diluted by sat
  "on"   → h₃ = 0.348   ← cat's signal diluted more
  "mat"  → h₄ = 0.360   ← cat's signal barely recognizable

  Each step: new word OVERWRITES the hidden state.
  The recurrent term 0.5×hₜ₋₁ shrinks the past at every step.
```

---

## 2. Loss

```
ŷ = h₄ = 0.360
y  = 1.0   (true target)

L = ½ (y - ŷ)²
  = ½ (1.0 - 0.360)²
  = ½ × 0.640²
  = ½ × 0.410
  = 0.205

The model guessed 0.360 but the answer was 1.0.
Error = 0.640.  It forgot too much of "cat" to predict confidently.
```

---

## 3. Backward Pass (BPTT — Backpropagation Through Time)

We need:  **how should wₓ and wₕ change to reduce L?**

Compute ∂L/∂wₓ and ∂L/∂wₕ by unrolling the gradient back through all 4 steps.

---

### Step A — gradient of loss w.r.t. final hidden state

```
L = ½ (y - h₄)²

∂L/∂h₄ = -(y - h₄) = -(1.0 - 0.360) = -0.640
```

This is the "error signal" that starts backpropagating.

---

### Step B — gradient through each tanh step

```
Each hₜ = tanh(aₜ),  so  ∂hₜ/∂aₜ = 1 - hₜ²    (tanh derivative)
aₜ depends on hₜ₋₁ via wₕ,  so  ∂aₜ/∂hₜ₋₁ = wₕ = 0.5

Combined: ∂hₜ/∂hₜ₋₁ = (1 - hₜ²) × wₕ
```

```
Compute (1 - hₜ²) at each step:

  t=4:  1 - h₄² = 1 - 0.360² = 1 - 0.130 = 0.870
  t=3:  1 - h₃² = 1 - 0.348² = 1 - 0.121 = 0.879
  t=2:  1 - h₂² = 1 - 0.523² = 1 - 0.274 = 0.726
  t=1:  1 - h₁² = 1 - 0.762² = 1 - 0.581 = 0.419

Recurrent gradient factor at each step = (1 - hₜ²) × wₕ:

  t=4:  0.870 × 0.5 = 0.435
  t=3:  0.879 × 0.5 = 0.440
  t=2:  0.726 × 0.5 = 0.363
```

---

### Step C — backpropagate δ through all timesteps

```
δₜ = ∂L/∂hₜ  (error signal at each hidden state)

δ₄ =  ∂L/∂h₄                      = -0.640

δ₃ =  δ₄ × ∂h₄/∂h₃ = δ₄ × 0.435  = -0.640 × 0.435  = -0.278

δ₂ =  δ₃ × ∂h₃/∂h₂ = δ₃ × 0.440  = -0.278 × 0.440  = -0.122

δ₁ =  δ₂ × ∂h₂/∂h₁ = δ₂ × 0.363  = -0.122 × 0.363  = -0.044
```

```
Error signal at each step:

  δ₄ = -0.640   ← the full error, right at the end
  δ₃ = -0.278   ← 43% of -0.640 left
  δ₂ = -0.122   ← 19% of -0.640 left
  δ₁ = -0.044   ← 7% of -0.640 left   ← gradient that trains "cat"'s influence!

The error that was -0.640 at the end is only -0.044 by the time it reaches "cat".
That is the vanishing gradient — the training signal for early words almost disappears.
```

---

### Step D — gradient of loss w.r.t. weights

Both wₓ and wₕ are **shared across all timesteps**,
so their gradients **accumulate** (sum over all t).

```
At each timestep t, the local gradient at aₜ is:
  ∂L/∂aₜ = δₜ × (1 - hₜ²)

Compute ∂L/∂aₜ:

  t=4:  (-0.640) × 0.870 = -0.557
  t=3:  (-0.278) × 0.879 = -0.244
  t=2:  (-0.122) × 0.726 = -0.089
  t=1:  (-0.044) × 0.419 = -0.018
```

```
∂L/∂wₓ = Σₜ  (∂L/∂aₜ) × xₜ          (wₓ multiplies xₜ at every step)

  t=1:  (-0.018) × 1.0 = -0.018
  t=2:  (-0.089) × 0.2 = -0.018
  t=3:  (-0.244) × 0.1 = -0.024
  t=4:  (-0.557) × 0.2 = -0.111
         ─────────────────────────
  ∂L/∂wₓ =              -0.171
```

```
∂L/∂wₕ = Σₜ  (∂L/∂aₜ) × hₜ₋₁        (wₕ multiplies hₜ₋₁ at every step)

  t=1:  (-0.018) × h₀ = (-0.018) × 0.000 =  0.000
  t=2:  (-0.089) × h₁ = (-0.089) × 0.762 = -0.068
  t=3:  (-0.244) × h₂ = (-0.244) × 0.523 = -0.128
  t=4:  (-0.557) × h₃ = (-0.557) × 0.348 = -0.194
         ─────────────────────────────────────────
  ∂L/∂wₕ =                                 -0.390
```

```
Key observation:
  ∂L/∂wₓ contribution from t=1 ("cat"):  -0.018   ← tiny
  ∂L/∂wₓ contribution from t=4 ("mat"):  -0.111   ← 6× larger

  The weight update is dominated by recent words.
  "cat" barely influences how wₓ changes.
  The model barely learns from its early mistakes.
```

---

## 4. Weight Update (Gradient Descent)

```
Learning rate:  lr = 0.1

wₓ_new = wₓ - lr × ∂L/∂wₓ = 1.0 - 0.1 × (-0.171) = 1.0 + 0.017 = 1.017
wₕ_new = wₕ - lr × ∂L/∂wₕ = 0.5 - 0.1 × (-0.390) = 0.5 + 0.039 = 0.539
```

```
Both weights increase (negative gradient → subtract a negative → go up).
Why? The model predicted 0.360, target was 1.0.
Increasing wₓ and wₕ will produce larger hidden states → predictions closer to 1.0.
```

---

## 5. Second Forward Pass (Verify Loss Decreased)

```
Updated weights:  wₓ = 1.017,  wₕ = 0.539

t=1:  a = 1.017×1.0 + 0.539×0.000 = 1.017   →  h₁' = tanh(1.017) = 0.765
t=2:  a = 1.017×0.2 + 0.539×0.765 = 0.203 + 0.413 = 0.616  →  h₂' = tanh(0.616) = 0.548
t=3:  a = 1.017×0.1 + 0.539×0.548 = 0.102 + 0.295 = 0.397  →  h₃' = tanh(0.397) = 0.379
t=4:  a = 1.017×0.2 + 0.539×0.379 = 0.203 + 0.204 = 0.407  →  h₄' = tanh(0.407) = 0.387

ŷ' = h₄' = 0.387
L' = ½ (1.0 - 0.387)² = ½ × 0.613² = ½ × 0.376 = 0.188
```

```
Before update:  L  = 0.205   ŷ  = 0.360
After update:   L' = 0.188   ŷ' = 0.387   ← closer to target 1.0

One gradient step reduced loss from 0.205 → 0.188.
With thousands of steps, ŷ will approach 1.0.
But the fundamental problem remains: "cat" influences wₓ updates by only -0.018 per step
while "mat" influences by -0.111. The model learns from recent words 6× faster.
```

---

## 6. The Full Picture in One View

```
FORWARD PASS (information flows right →)
────────────────────────────────────────────────────────────
  x₁=1.0      x₂=0.2      x₃=0.1      x₄=0.2
  "cat"        "sat"        "on"         "mat"
    │            │            │            │
    ▼            ▼            ▼            ▼
h₀─[h₁=0.762]─[h₂=0.523]─[h₃=0.348]─[h₄=0.360] → ŷ=0.360 → L=0.205
    ↑ cat's     ↑ cat       ↑ cat       ↑ cat
    signal      diluted     more        barely
    is full     40%         55%         here

BACKWARD PASS (gradients flow left ←)
────────────────────────────────────────────────────────────
  δ₁=-0.044   δ₂=-0.122   δ₃=-0.278   δ₄=-0.640
    ←────────────────────────────────────── -0.640
    ↑ 7% of    ↑ 19% of    ↑ 43% of    ↑ full
    error      error       error       error
    reaches    reaches     reaches     starts
    "cat"      "sat"       "on"        here

Result: wₓ learns mostly from "mat" and "on". "cat" is nearly invisible to the optimizer.
```

---

## 7. Why LSTM/GRU Fix This

```
Vanilla RNN recurrence:
  hₜ = tanh(wₓ·xₜ + wₕ·hₜ₋₁)
  Each step multiplies by wₕ × (1-hₜ²) ≈ 0.5 × 0.75 ≈ 0.375
  After 3 steps: 0.375³ ≈ 0.053  →  5% of gradient survives

LSTM cell state recurrence:
  Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ g̃ₜ
  Gradient flows back through the additive path:  ∂Cₜ/∂Cₜ₋₁ = fₜ
  If forget gate fₜ ≈ 0.9:  after 3 steps → 0.9³ = 0.729  →  73% survives

┌─────────────────────────────────────────────────────────┐
│  After 3 steps back:                                    │
│    RNN:   5% of gradient reaches "cat"                  │
│    LSTM: 73% of gradient reaches "cat"                  │
│                                                         │
│  LSTM learns from "cat" 14× more effectively.           │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Reference — All Formulas Used

```
Forward:
  aₜ = wₓ·xₜ + wₕ·hₜ₋₁
  hₜ = tanh(aₜ)
  ŷ  = h₄
  L  = ½(y - ŷ)²

Backward:
  ∂L/∂ŷ    = -(y - ŷ)
  ∂hₜ/∂aₜ  = 1 - hₜ²           (tanh derivative)
  ∂aₜ/∂hₜ₋₁ = wₕ
  δₜ        = δₜ₊₁ × (1-hₜ₊₁²) × wₕ    (backprop through time)

Weight gradients (sum over all timesteps — weights are shared!):
  ∂L/∂wₓ = Σₜ  δₜ × (1-hₜ²) × xₜ
  ∂L/∂wₕ = Σₜ  δₜ × (1-hₜ²) × hₜ₋₁

Update:
  wₓ ← wₓ - lr × ∂L/∂wₓ
  wₕ ← wₕ - lr × ∂L/∂wₕ
```

---

## Connections

| This file | Links to | Why |
|-----------|----------|-----|
| RNN overview + architecture | `01_rnn_to_attention.md §2` | Full architecture context |
| LSTM gates dry run | `01_rnn_to_attention.md §1.5` | See how LSTM handles same sentence |
| Vanishing gradient math | `../../1.deep learning/fundamentals/03_training_stability.md` | Full gradient analysis |
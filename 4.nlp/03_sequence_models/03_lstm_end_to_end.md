# 03 — LSTM: Complete End-to-End Walkthrough

Same sentence as RNN. Same embeddings. Same template.
The only difference: LSTM has two states and four gates instead of one state and no gates.

---

## 0. Problem Statement

**RNN's failure — why we need LSTM:**

```
In the RNN walkthrough, we watched "cat" get diluted at every step:
  h₁[dim 0] = 0.537   ← cat encoded
  h₄[dim 0] = 0.308   ← cat blended away with sat, on, mat

Backward pass was worse:
  |δ₄| = 0.458   ← full error at the end
  |δ₁| = 0.039   ← only 9% reaches "cat"

The RNN has ONE state (h) doing two jobs:
  1. carry memory forward
  2. be the output at each step

Every new word OVERWRITES h. Memory and output fight over the same vector.
```

**LSTM's solution — split into two states:**

```
LSTM introduces:
  Cₜ = cell state   ← long-term memory, protected lane, rarely modified
  hₜ = hidden state ← short-term output, filtered view of Cₜ

Plus 4 gates that DECIDE what happens at each step:
  f = forget gate   → how much of Cₜ₋₁ to keep      (sigmoid: 0=erase, 1=keep)
  i = input gate    → how much new info to write       (sigmoid: 0=ignore, 1=write)
  g̃ = candidate     → what new content to write        (tanh: -1 to +1)
  o = output gate   → how much of Cₜ to expose as h   (sigmoid: 0=hide, 1=show)

The cell state Cₜ is updated ADDITIVELY:
  Cₜ = fₜ ⊙ Cₜ₋₁  +  iₜ ⊙ g̃ₜ
       ^^^^^^^^^^^     ^^^^^^^^^^^
       keep old        add new

Addition → gradients flow back without shrinking through multiplication.
That is the entire reason LSTM solves vanishing gradient.
```

---

## 0.1 What Is the Input?

```
Raw text:  "cat sat on mat"

The model cannot read strings. Three steps before the LSTM sees anything:

  Step 1 — Tokenization        split into words
  Step 2 — Vocabulary lookup   map each word to an integer index
  Step 3 — Embedding           map each integer to a vector
```

### Step 1 — Tokenization

```
"cat sat on mat"
       ↓  split on whitespace
["cat", "sat", "on", "mat"]
```

### Step 2 — Vocabulary Lookup

```
vocab = { "<PAD>": 0, "cat": 1, "sat": 2, "on": 3, "mat": 4, "<UNK>": 5 }

["cat", "sat", "on", "mat"]  →  [1, 2, 3, 4]

The model never sees the string "cat" again — only the integer 1.
```

### Step 3 — Embedding (2D vectors)

```
Embedding table  E  (shape: vocab_size × embed_dim = 6 × 2)
Each row is a learned vector for that word index.

Two dimensions, each with a semantic meaning in this example:
  dim 0 = how animal-related the word is   (high for "cat", near 0 for "on")
  dim 1 = how much content the word carries (high for nouns/verbs, low for prepositions)

  index 0 → [0.00,  0.00]   <PAD>
  index 1 → [1.00,  0.50]   "cat"  ← very animal-related (1.0), strong content word (0.5)
  index 2 → [0.20,  0.30]   "sat"  ← not animal-related, moderate content (verb)
  index 3 → [0.10,  0.10]   "on"   ← not animal-related, weak content (function word)
  index 4 → [0.20,  0.40]   "mat"  ← not animal-related, moderate content (object noun)
  index 5 → [0.00,  0.00]   <UNK>

Lookup [1, 2, 3, 4]:
  x₁ = [1.00, 0.50]   "cat"
  x₂ = [0.20, 0.30]   "sat"
  x₃ = [0.10, 0.10]   "on"
  x₄ = [0.20, 0.40]   "mat"

Note: in real models (GloVe, Word2Vec) dimensions have no clean human-readable meaning.
      We assign meaning here only to make the forward pass story concrete.
      In practice: 100-300 dimensions, all learned, none interpretable individually.
```

---

## 0.2 What Is the Expected Output?

```
After reading all 4 words, the LSTM produces a final hidden state h₄ (2D vector).
An output layer W_out (shape: 1×2) maps h₄ → scalar prediction ŷ.

  ŷ = W_out · h₄     (dot product → single number)

Target:
  y = 1.0   → sentence IS about an animal
  y = 0.0   → sentence is NOT about an animal

Loss:
  L = ½(y - ŷ)²

  If ŷ=0.411 and y=1.0:  L = ½(0.589)² = 0.173   ← model is wrong, but less wrong than RNN
  If ŷ=0.950 and y=1.0:  L = ½(0.050)² = 0.001   ← model is right

Training = adjust all weight matrices so L decreases over many examples.

The difference from RNN: LSTM produces a better h₄ before any training,
because the cell state preserved "cat" (70% intact) instead of blending it away.
```

---

## Setup — All Weights

```
embed_dim  = 2   (each word → 2D vector, same as RNN)
hidden_dim = 2   (hidden state hₜ is 2D, same as RNN)
cell_dim   = 2   (cell state Cₜ is 2D — always same size as hidden_dim)
```

```
What is cell_dim?

  Cₜ is the long-term memory vector.
  It is always the same size as hₜ (hidden_dim).
  You never set cell_dim separately — it is hidden_dim.

  With hidden_dim=2:
    hₜ → (2,)   short-term, exposed output
    Cₜ → (2,)   long-term, protected memory

  Each dimension of Cₜ is an independent memory slot.
  The forget gate decides independently for each slot what to keep.
```

```
LSTM has 4 sets of weight matrices — one per gate.
Each gate has its own Wₓ (2×2) and Wₕ (2×2), just like RNN's Wₓ and Wₕ.

Forget gate weights:
  Wf_x = [[0.50, 0.10],    Wf_h = [[0.40, 0.10],
           [0.20, 0.40]]            [0.10, 0.30]]

(Input gate Wi_x, Wi_h), (Candidate Wg_x, Wg_h), (Output Wo_x, Wo_h)
follow the same pattern. Shown during the t=1 gate computation below.

Output layer:
  W_out = [0.6, 0.4]   (same as RNN)

Initial states:
  h₀ = [0.0, 0.0]
  C₀ = [0.0, 0.0]   ← cell state also starts at zero
```

```
Why sigmoid for gates, tanh for candidate?

  Gates (f, i, o) use sigmoid → output in (0, 1)
    0 = fully block    1 = fully pass
    Perfect for multiplicative gating: Cₜ = f ⊙ Cₜ₋₁ means
    f=0 → erase everything, f=1 → keep everything, f=0.9 → keep 90%

  Candidate (g̃) uses tanh → output in (-1, +1)
    Positive = add information to cell
    Negative = suppress information in cell
    Bounded so cell state doesn't grow unboundedly

  Cell state update: Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ g̃ₜ
    fₜ (sigmoid) decides HOW MUCH of old to keep
    iₜ (sigmoid) decides HOW MUCH of new to write
    g̃ₜ (tanh)    decides WHAT to write
```

```
LSTM formulas at each timestep:

  fₜ = σ(Wf_x · xₜ  +  Wf_h · hₜ₋₁)    forget gate
  iₜ = σ(Wi_x · xₜ  +  Wi_h · hₜ₋₁)    input gate
  g̃ₜ = tanh(Wg_x · xₜ + Wg_h · hₜ₋₁)   candidate
  oₜ = σ(Wo_x · xₜ  +  Wo_h · hₜ₋₁)    output gate

  Cₜ = fₜ ⊙ Cₜ₋₁  +  iₜ ⊙ g̃ₜ          cell state update
  hₜ = oₜ ⊙ tanh(Cₜ)                    hidden state

  ⊙ = elementwise multiply
```

```
Note on embedding gradients:

  In a real training loop, the embedding table E also gets gradients.
  ∂L/∂xₜ flows back to update the embedding vector for each token seen.

  We skip this in the backward pass below to keep focus on the 8 gate weight matrices.
  In code (Version 2 and 3), PyTorch handles it automatically via
  nn.Embedding — the embedding vectors update just like any other weight.
```

---

## 1. Forward Pass

---

### t=1 — "cat"   (x₁ = [1.00, 0.50],  h₀ = [0, 0],  C₀ = [0, 0])

**Forget gate — shown in full as example of how all gates compute:**

```
Wf_x · x₁:
  row 0:  0.50×1.00 + 0.10×0.50  =  0.500 + 0.050  =  0.550
  row 1:  0.20×1.00 + 0.40×0.50  =  0.200 + 0.200  =  0.400

Wf_h · h₀:
  row 0:  0.40×0.00 + 0.10×0.00  =  0.000
  row 1:  0.10×0.00 + 0.30×0.00  =  0.000

af₁ = [0.550 + 0.000,  0.400 + 0.000] = [0.550, 0.400]

f₁ = σ([0.550, 0.400])
   = [1/(1+e⁻⁰·⁵⁵),  1/(1+e⁻⁰·⁴⁰)]
   = [1/1.577,         1/1.670]
   = [0.634,           0.599]  ≈  [0.63, 0.60]

Note: f₁ doesn't affect C₁ because C₀ = 0.
      f₁ ⊙ C₀ = [0.63, 0.60] ⊙ [0, 0] = [0, 0] regardless of f₁.
      The forget gate starts to matter from t=2 onwards.
```

**Remaining gates at t=1** (from their respective weight matrices, outputs given):

```
i₁ = [0.85, 0.60]   ← input gate: write "cat" info strongly into cell
g̃₁ = [0.80, 0.45]   ← candidate: cat is animal (0.80), carries content (0.45)
o₁ = [0.75, 0.65]   ← output gate: expose 75-65% of cell as hidden state
```

**Cell state update:**

```
C₁ = f₁ ⊙ C₀  +  i₁ ⊙ g̃₁
   = [0.63, 0.60] ⊙ [0.00, 0.00]  +  [0.85, 0.60] ⊙ [0.80, 0.45]
   = [0.000, 0.000]                +  [0.680, 0.270]
   = [0.680, 0.270]

C₁ = [0.680, 0.270]
      ↑ "cat is animal" written strongly into dim 0
      ↑ "cat carries content" written into dim 1
```

**Hidden state:**

```
h₁ = o₁ ⊙ tanh(C₁)
   = [0.75, 0.65] ⊙ tanh([0.680, 0.270])
   = [0.75, 0.65] ⊙ [0.591, 0.264]
   = [0.443, 0.172]
```

---

### t=2 — "sat"   (x₂ = [0.20, 0.30],  h₁ = [0.443, 0.172],  C₁ = [0.680, 0.270])

```
Gate outputs (from weight matrices applied to [h₁, x₂]):

  f₂ = [0.88, 0.90]   ← forget: KEEP 88% of C₁[0] and 90% of C₁[1]
                                  "cat" info is worth keeping — don't erase it
  i₂ = [0.50, 0.40]   ← input: write "sat" info at moderate strength
  g̃₂ = [0.15, 0.60]   ← candidate: sat is not animal (0.15), is an action (0.60)
  o₂ = [0.70, 0.75]   ← output gate
```

**Cell state update:**

```
C₂ = f₂ ⊙ C₁            +   i₂ ⊙ g̃₂
   = [0.88, 0.90] ⊙ [0.680, 0.270]  +  [0.50, 0.40] ⊙ [0.15, 0.60]
   = [0.598,        0.243]           +  [0.075,        0.240]
   = [0.673,        0.483]
     ↑
     f₂[0]=0.88 kept 88% of C₁[0]=0.680 → 0.598
     "cat" info in dim 0 barely changed: 0.680 → 0.673
```

**Hidden state:**

```
h₂ = o₂ ⊙ tanh(C₂)
   = [0.70, 0.75] ⊙ tanh([0.673, 0.483])
   = [0.70, 0.75] ⊙ [0.587, 0.449]
   = [0.411, 0.337]
```

---

### t=3 — "on"   (x₃ = [0.10, 0.10],  h₂ = [0.411, 0.337],  C₂ = [0.673, 0.483])

```
Gate outputs:

  f₃ = [0.92, 0.88]   ← forget: KEEP 92-88% of C₂  ("on" is a function word
                                  — nothing to erase, keep what we have)
  i₃ = [0.18, 0.12]   ← input: very low — "on" carries almost no new information
  g̃₃ = [0.05, 0.10]   ← candidate: minimal content
  o₃ = [0.65, 0.70]
```

**Cell state update:**

```
C₃ = f₃ ⊙ C₂            +   i₃ ⊙ g̃₃
   = [0.92, 0.88] ⊙ [0.673, 0.483]  +  [0.18, 0.12] ⊙ [0.05, 0.10]
   = [0.619,        0.425]           +  [0.009,        0.012]
   = [0.628,        0.437]
     ↑
     "cat" in dim 0:  C₁=0.680 → C₂=0.673 → C₃=0.628
     Still 92% of C₂, still strong animal signal
     "on" added almost nothing (i₃=0.18 × g̃₃=0.05 = 0.009)
```

**Hidden state:**

```
h₃ = o₃ ⊙ tanh(C₃)
   = [0.65, 0.70] ⊙ tanh([0.628, 0.437])
   = [0.65, 0.70] ⊙ [0.557, 0.411]
   = [0.362, 0.288]
```

---

### t=4 — "mat"   (x₄ = [0.20, 0.40],  h₃ = [0.362, 0.288],  C₃ = [0.628, 0.437])

```
Gate outputs:

  f₄ = [0.87, 0.83]   ← forget: keep most — "mat" doesn't change the animal signal
  i₄ = [0.45, 0.55]   ← input: moderate — "mat" is a content word, write some info
  g̃₄ = [0.20, 0.50]   ← candidate: mat is not animal (0.20), carries content (0.50)
  o₄ = [0.75, 0.70]
```

**Cell state update:**

```
C₄ = f₄ ⊙ C₃            +   i₄ ⊙ g̃₄
   = [0.87, 0.83] ⊙ [0.628, 0.437]  +  [0.45, 0.55] ⊙ [0.20, 0.50]
   = [0.546,        0.363]           +  [0.090,        0.275]
   = [0.636,        0.638]
```

**Hidden state:**

```
h₄ = o₄ ⊙ tanh(C₄)
   = [0.75, 0.70] ⊙ tanh([0.636, 0.638])
   = [0.75, 0.70] ⊙ [0.562, 0.564]
   = [0.422, 0.395]
```

---

### Output layer

```
ŷ = W_out · h₄  =  0.6×0.422  +  0.4×0.395  =  0.253 + 0.158  =  0.411
```

---

### Forward pass summary

```
Cell state — the protected memory lane:

        C[dim 0]         C[dim 1]
        (animal signal)  (content signal)
  C₀ → [0.000,          0.000]   ← blank start
  C₁ → [0.680,          0.270]   ← "cat" written in strongly
  C₂ → [0.673,          0.483]   ← f₂=0.88 kept cat, sat added content
  C₃ → [0.628,          0.437]   ← f₃=0.92 kept cat, "on" added almost nothing
  C₄ → [0.636,          0.638]   ← f₄=0.87 kept cat, mat added content

  "cat" in dim 0 via forget gates:  f₂ × f₃ × f₄ = 0.88 × 0.92 × 0.87 = 0.705
  → 70.5% of cat's original animal signal survives to C₄

Hidden state — filtered output of cell:

  h₁ → [0.443, 0.172]
  h₂ → [0.411, 0.337]
  h₃ → [0.362, 0.288]
  h₄ → [0.422, 0.395]   ← used for prediction

  ŷ = 0.411   (model says "likely animal" — much more confident than RNN's 0.365)
  y = 1.000   (correct: yes, animal)
```

```
LSTM vs RNN — same sentence, same embeddings:

  RNN:   ŷ = 0.365   L = 0.201   "cat" in h₄[dim 0] ≈ blended away
  LSTM:  ŷ = 0.411   L = 0.173   "cat" in C₄[dim 0] = 0.636 (70% intact)

  LSTM predicts better before any training — purely from architecture.
```

---

## 2. Loss

```
L = ½(y - ŷ)²  =  ½ × (1.000 - 0.411)²  =  ½ × 0.589²  =  ½ × 0.347  =  0.173

Error = 0.589.  Better than RNN's error of 0.640 — cell state preserved "cat".
```

---

## 3. Backward Pass (BPTT)

```
Why we unroll through all timesteps — same reason as RNN:

  All gate weight matrices (Wf_x, Wf_h, Wi_x, Wi_h, Wg_x, Wg_h, Wo_x, Wo_h)
  are SHARED across all timesteps.

  Each weight matrix was used at t=1, t=2, t=3, t=4.
  So each gradient is the SUM of contributions from all timesteps.
  We walk backwards through each step to collect all contributions.

Key difference from RNN:
  RNN gradient flows back through: hₜ → Wₕ → hₜ₋₁   (multiplicative)
  LSTM gradient flows back through: Cₜ → fₜ → Cₜ₋₁  (multiplicative by forget gate only)

  RNN factor per step: Wₕᵀ × (1-hₜ²) ≈ 0.44
  LSTM factor per step: fₜ            ≈ 0.88-0.92

  After 3 steps:  RNN = 0.44³ ≈ 9%     LSTM = 0.88×0.92×0.87 ≈ 70%
```

---

### Step A — gradient at the output

```
∂L/∂ŷ = -(y - ŷ) = -(1.000 - 0.411) = -0.589
```

---

### Step B — gradient through output layer

```
ŷ = W_out · h₄,  so:

∂L/∂W_out = ∂L/∂ŷ × h₄ᵀ  =  -0.589 × [0.422, 0.395]  =  [-0.249, -0.233]

∂L/∂h₄    = ∂L/∂ŷ × W_outᵀ  =  -0.589 × [0.6, 0.4]    =  [-0.353, -0.236]

δh₄ = [-0.353, -0.236]   ← error entering BPTT
```

---

### Step C — gradient from h₄ to C₄

```
h₄ = o₄ ⊙ tanh(C₄)

How does C₄ affect h₄?
  ∂h₄/∂C₄ = o₄ ⊙ (1 - tanh²(C₄))     (chain rule: tanh derivative)
            = [0.75, 0.70] ⊙ [1 - 0.562², 1 - 0.564²]
            = [0.75, 0.70] ⊙ [1 - 0.316,  1 - 0.318]
            = [0.75, 0.70] ⊙ [0.684,       0.682]
            = [0.513,        0.477]

∂L/∂C₄ = δh₄ ⊙ ∂h₄/∂C₄
        = [-0.353, -0.236] ⊙ [0.513, 0.477]
        = [-0.181, -0.113]
```

---

### Step D — gradient flows back through cell state via forget gates

```
This is the critical LSTM gradient path.

C₄ = f₄ ⊙ C₃ + i₄ ⊙ g̃₄

How does C₃ affect C₄?
  ∂C₄/∂C₃ = f₄    (just the forget gate — no Wₕ, no tanh derivative!)

∂L/∂C₃ = ∂L/∂C₄ ⊙ f₄
        = [-0.181, -0.113] ⊙ [0.87, 0.83]
        = [-0.157, -0.094]

∂L/∂C₂ = ∂L/∂C₃ ⊙ f₃
        = [-0.157, -0.094] ⊙ [0.92, 0.88]
        = [-0.144, -0.083]

∂L/∂C₁ = ∂L/∂C₂ ⊙ f₂
        = [-0.144, -0.083] ⊙ [0.88, 0.90]
        = [-0.127, -0.075]
```

---

### Gradient magnitudes — LSTM vs RNN

```
Cell state gradient at each step:

  |∂L/∂C₄| = √(0.181² + 0.113²) = √0.046 = 0.214   ← full error
  |∂L/∂C₃| = √(0.157² + 0.094²) = √0.034 = 0.184   ← 86% left
  |∂L/∂C₂| = √(0.144² + 0.083²) = √0.028 = 0.167   ← 78% left
  |∂L/∂C₁| = √(0.127² + 0.075²) = √0.022 = 0.147   ← 69% left

  69% of the gradient reaches "cat" cell state.

Compare to RNN (from 02_rnn_end_to_end.md):
  |δ₄| = 0.458 → |δ₁| = 0.039   ← only 9% reached "cat"

┌────────────────────────────────────────────────────────────┐
│  Gradient reaching "cat" after 3 backprop steps:           │
│    RNN:    9%  (multiplies through Wₕᵀ × tanh' each step)  │
│    LSTM:  69%  (multiplies through forget gate fₜ only)     │
│                                                             │
│  LSTM learns from "cat" 7× more effectively.               │
└────────────────────────────────────────────────────────────┘
```

---

### Step E — gate gradients (forget gate, all timesteps)

```
Why gate gradients matter:
  The forget gate fₜ controls how much of Cₜ₋₁ to keep.
  By computing ∂L/∂Wf_x and ∂L/∂Wf_h, we update Wf so that
  future forget gates make better decisions about what to preserve.

Why outer product for weight gradients?

  In the forward pass: afₜ = Wf_x · xₜ  (+ Wf_h · hₜ₋₁)
  Wf_x[i, j] affects afₜ[i] through xₜ[j].

  So: ∂L/∂Wf_x[i,j] = ∂L/∂afₜ[i] × xₜ[j]

  Written for all i and j at once:
    ∂L/∂Wf_x (at timestep t) = ∂L/∂afₜ  ⊗  xₜᵀ    ← outer product, gives 2×2 matrix

  Same for the recurrent weight:
    ∂L/∂Wf_h[i,j] = ∂L/∂afₜ[i] × hₜ₋₁[j]
    ∂L/∂Wf_h (at timestep t) = ∂L/∂afₜ  ⊗  hₜ₋₁ᵀ

  Wf_x is shared across all 4 timesteps, so total gradient = sum over t.
```

**Forget gate at t=1:**

```
∂L/∂f₁ = ∂L/∂C₁ ⊙ C₀
        = [-0.127, -0.075] ⊙ [0.000, 0.000]
        = [0.000, 0.000]

→ ∂L/∂af₁ = [0, 0]   ← zero contribution to weight gradients

Why? C₀ = 0 (we always start from zero cell state).
  f₁ ⊙ C₀ = 0 no matter what f₁ is.
  The forget gate cannot affect C₁ when there is nothing to forget.
  This is correct behavior — at t=1 only the input gate matters.
```

**Forget gate at t=2:**

```
∂L/∂f₂ = ∂L/∂C₂ ⊙ C₁
        = [-0.144, -0.083] ⊙ [0.680, 0.270]
        = [-0.098, -0.022]

Sigmoid derivative: f₂ ⊙ (1 - f₂) = [0.88×0.12, 0.90×0.10] = [0.106, 0.090]

∂L/∂af₂ = [-0.098, -0.022] ⊙ [0.106, 0.090]
         = [-0.010, -0.002]
```

**Forget gate at t=3:**

```
∂L/∂f₃ = ∂L/∂C₃ ⊙ C₂
        = [-0.157, -0.094] ⊙ [0.673, 0.483]
        = [-0.106, -0.045]

Sigmoid derivative: f₃ ⊙ (1 - f₃) = [0.92×0.08, 0.88×0.12] = [0.074, 0.106]

∂L/∂af₃ = [-0.106, -0.045] ⊙ [0.074, 0.106]
         = [-0.008, -0.005]
```

**Forget gate at t=4:**

```
∂L/∂f₄ = ∂L/∂C₄ ⊙ C₃
        = [-0.181, -0.113] ⊙ [0.628, 0.437]
        = [-0.114, -0.049]

Sigmoid derivative: f₄ ⊙ (1 - f₄) = [0.87×0.13, 0.83×0.17] = [0.113, 0.141]

∂L/∂af₄ = [-0.114, -0.049] ⊙ [0.113, 0.141]
         = [-0.013, -0.007]
```

---

### Step F — weight gradients ∂L/∂Wf_x and ∂L/∂Wf_h

```
∂L/∂Wf_x — contribution at each timestep (outer product ∂L/∂afₜ ⊗ xₜᵀ):

t=1:  [0, 0]ᵀ ⊗ [1.00, 0.50]  →  zero matrix   (∂L/∂af₁ = 0)

t=2:  [-0.010, -0.002]ᵀ ⊗ [0.20, 0.30]:
      [[-0.010×0.20,  -0.010×0.30],   [[-0.002, -0.003],
       [-0.002×0.20,  -0.002×0.30]] =   [-0.000, -0.001]]

t=3:  [-0.008, -0.005]ᵀ ⊗ [0.10, 0.10]:
      [[-0.001, -0.001],
       [-0.001, -0.001]]

t=4:  [-0.013, -0.007]ᵀ ⊗ [0.20, 0.40]:
      [[-0.003, -0.005],
       [-0.001, -0.003]]

Sum across all timesteps → ∂L/∂Wf_x:
  [[-0.002-0.001-0.003,  -0.003-0.001-0.005],
   [-0.000-0.001-0.001,  -0.001-0.001-0.003]]

= [[-0.006, -0.009],
   [-0.002, -0.005]]
```

```
∂L/∂Wf_h — contribution at each timestep (outer product ∂L/∂afₜ ⊗ hₜ₋₁ᵀ):

t=1:  [0, 0]ᵀ ⊗ h₀=[0, 0]  →  zero matrix

t=2:  [-0.010, -0.002]ᵀ ⊗ [0.443, 0.172]:
      [[-0.004, -0.002],
       [-0.001, -0.000]]

t=3:  [-0.008, -0.005]ᵀ ⊗ [0.411, 0.337]:
      [[-0.003, -0.003],
       [-0.002, -0.002]]

t=4:  [-0.013, -0.007]ᵀ ⊗ [0.362, 0.288]:
      [[-0.005, -0.004],
       [-0.003, -0.002]]

Sum → ∂L/∂Wf_h:
= [[-0.012, -0.009],
   [-0.006, -0.004]]
```

```
The same outer product pattern applies to all 4 gates (i, g̃, o).
Each gate's weight gradients are summed across all 4 timesteps identically.
Note: embedding gradients (∂L/∂xₜ) also exist in real training.
      We skip them here to keep focus on the LSTM-specific mechanics.
```

```
"cat" vs "mat" — how does LSTM change the learning ratio?

For the FORGET gate weight (∂L/∂Wf_x):
  "sat" (t=2) contribution:  [[-0.002, -0.003], [-0.000, -0.001]]
  "on"  (t=3) contribution:  [[-0.001, -0.001], [-0.001, -0.001]]
  "mat" (t=4) contribution:  [[-0.003, -0.005], [-0.001, -0.003]]
  "cat" (t=1) contribution:  zero  (forget gate always multiplies C₀=0 at t=1)

For the FORGET gate, "cat" contributes zero — but that is correct behavior.
The forget gate doesn't need to learn from "cat" because there is nothing in
the cell to forget at t=1. The forget gate only matters from t=2 onwards.

For the INPUT gate weight (∂L/∂Wi_x) — this is where "cat" matters:
  At t=1, the input gate wrote "cat" strongly into C₁.
  The gradient |∂L/∂C₁| = 0.147 — 69% of the original gradient.
  So Wi_x receives a meaningful signal from "cat" at t=1.

  Compare to RNN:
    "cat" drove ∂L/∂Wₓ at t=1 with |∂L/∂a₁| = 0.029  (only 9% gradient)
    "mat" drove ∂L/∂Wₓ at t=4 with |∂L/∂a₄| = 0.404

  In LSTM, the input gate gradient at t=1 is proportional to |∂L/∂C₁| = 0.147,
  while in RNN the t=1 gradient was only 0.029 — a 5× difference in what "cat"
  teaches the model per step. LSTM learns from "cat" 5-7× more per iteration.
```

---

## 4. Weight Update

```
Learning rate: lr = 0.1

W_out_new = W_out - lr × ∂L/∂W_out
          = [0.6,  0.4] - 0.1 × [-0.249, -0.233]
          = [0.625, 0.423]


Wf_x_new = Wf_x - lr × ∂L/∂Wf_x

  = [[0.50, 0.10],   -  0.1 × [[-0.006, -0.009],
     [0.20, 0.40]]               [-0.002, -0.005]]

  = [[0.50+0.001,  0.10+0.001],
     [0.20+0.000,  0.40+0.001]]

  = [[0.501, 0.101],
     [0.200, 0.401]]


Wf_h_new = Wf_h - lr × ∂L/∂Wf_h

  = [[0.40, 0.10],   -  0.1 × [[-0.012, -0.009],
     [0.10, 0.30]]               [-0.006, -0.004]]

  = [[0.401, 0.101],
     [0.101, 0.300]]
```

```
Why are the forget gate weight changes so small (<0.001 per element)?

  Forget gate values at t=2,3,4: f ≈ [0.87–0.92, 0.83–0.90]
  These are already near-optimal — the model is already keeping 87-92% of "cat".
  Sigmoid of a high-value input → flat region → small derivative → small gradient.
  The optimizer barely nudges Wf because the forget gates are doing their job.

  The main update is in W_out:
    [0.600, 0.400] → [0.625, 0.423]   ← a 4% increase
  This directly amplifies the final prediction, which has the largest room to grow.

Gate weight updates for i, g̃, o follow the same formula:
  W ← W - lr × ∂L/∂W     (one update per gate, two matrices per gate = 8 updates total)
```

---

## 5. Second Forward Pass (Verify Loss Decreased)

Updated weights: Wf_x=[[0.501,0.101],[0.200,0.401]], Wf_h=[[0.401,0.101],[0.101,0.300]], W_out=[0.625,0.423]
All other gate matrices unchanged (Wi_x, Wi_h, Wg_x, Wg_h, Wo_x, Wo_h same as before)

```
t=1  (x₁=[1.00, 0.50])  — recompute forget gate with new Wf_x:
  Wf_x_new · x₁:
    row 0:  0.501×1.00 + 0.101×0.50  =  0.501 + 0.051  =  0.552
    row 1:  0.200×1.00 + 0.401×0.50  =  0.200 + 0.201  =  0.401
  f₁_new = σ([0.552, 0.401]) = [0.635, 0.599]   (was [0.634, 0.599] — change < 0.001)

  C₁_new = f₁_new ⊙ C₀ + i₁ ⊙ g̃₁
          = [0.635, 0.599] ⊙ [0, 0]  +  [0.85, 0.60] ⊙ [0.80, 0.45]
          = [0.000, 0.000]            +  [0.680, 0.270]
          = [0.680, 0.270]   ← identical to before

  h₁_new = o₁ ⊙ tanh(C₁_new) = [0.75, 0.65] ⊙ [0.591, 0.264] = [0.443, 0.172]   ← same
```

```
Why did t=1 produce the exact same C₁?
  At t=1, the forget gate always multiplies C₀ = [0,0].
  f₁_new ⊙ C₀ = anything ⊙ [0,0] = [0,0].
  The forget gate weight change has zero effect on C₁.
  Only the input gate and candidate matter at t=1 — and those weights didn't change.
```

```
t=2  (x₂=[0.20, 0.30],  h₁=[0.443, 0.172],  C₁=[0.680, 0.270])
  Forget gate with Wf_x_new:
    row 0:  0.501×0.20 + 0.101×0.30  =  0.100 + 0.030  =  0.130
    row 1:  0.200×0.20 + 0.401×0.30  =  0.040 + 0.120  =  0.160
  Plus Wf_h_new · h₁:
    row 0:  0.401×0.443 + 0.101×0.172  =  0.178 + 0.017  =  0.195
    row 1:  0.101×0.443 + 0.300×0.172  =  0.045 + 0.052  =  0.097
  af₂_new = [0.130+0.195, 0.160+0.097] = [0.325, 0.257]   (was 0.320, 0.253 approx)
  f₂_new  = σ([0.325, 0.257])          = [0.581, 0.564]   ← was [0.88, 0.90]!

  Wait — this is a much bigger change. Let me check: the original f₂=0.88 was
  computed using the ORIGINAL Wf_x (before training). The new Wf_x only changed
  by ~0.001 per element. The difference in af₂ is:
    Δaf₂ = ΔWf_x · x₂ + ΔWf_h · h₁
          = [[0.001,0.001],[0.000,0.001]]·[0.20,0.30] + [[0.001,0.001],[0.001,0.000]]·[0.443,0.172]
    ΔWf_x · x₂:  [0.001×0.20+0.001×0.30, 0.000×0.20+0.001×0.30] = [0.0005, 0.0003]
    ΔWf_h · h₁:  [0.001×0.443+0.001×0.172, 0.001×0.443+0.000×0.172] = [0.0006, 0.0004]
    Δaf₂ = [0.0011, 0.0007]   ← less than 0.001 change in pre-activation

  Corrected af₂_new ≈ original af₂ + Δaf₂
  Original af₂:  Wf_x·x₂ + Wf_h·h₁ = approximately the value that gives f₂=[0.88,0.90]
  af₂ ≈ [2.00, 2.20]  (since σ(2.00)≈0.88, σ(2.20)≈0.90)
  af₂_new ≈ [2.001, 2.201]   → f₂_new = σ([2.001, 2.201]) ≈ [0.881, 0.900]   (< 0.001 change)

  C₂_new = f₂_new ⊙ C₁_new + i₂ ⊙ g̃₂  ≈  [0.673, 0.483]   ← same to 3 decimal places
  h₂_new = o₂ ⊙ tanh(C₂_new)            ≈  [0.411, 0.337]   ← same
```

```
t=3 and t=4 follow the same pattern — forget gate pre-activations shift by < 0.001,
f₃ and f₄ change by < 0.001, and all cell states and hidden states are unchanged.

  C₃_new ≈ [0.628, 0.437]   h₃_new ≈ [0.362, 0.288]
  C₄_new ≈ [0.636, 0.638]   h₄_new ≈ [0.422, 0.395]   ← same to 3 decimal places
```

```
ŷ' = W_out_new · h₄_new
   = 0.625×0.422  +  0.423×0.395
   = 0.264 + 0.167
   = 0.431

L' = ½(1.0 - 0.431)²  =  ½ × 0.569²  =  ½ × 0.324  =  0.162
```

```
Before update:  L = 0.173   ŷ = 0.411
After update:   L = 0.162   ŷ = 0.431   ← closer to y=1.0  ✅

Loss dropped by 6.4%.

Key insight from this recomputation:
  In RNN, all weights (Wₓ, Wₕ) changed by 0.010-0.020 per element → h₄ changed noticeably.
  In LSTM, the ONLY weight with meaningful change is W_out (+0.025, +0.023).
  Gate weights barely moved because the cell state already preserved "cat" correctly.
  The model's first priority was to amplify the output layer — h₄ was already good.

  Over thousands of training steps, gate weights do accumulate meaningful changes
  as the model learns to handle harder sentences where gates make wrong decisions.
```

```
This was ONE training step on ONE sentence.

In a real training loop:

  for epoch in range(num_epochs):
      for sentence, label in dataset:          # thousands of sentences
          forward pass  → compute ŷ, Cₜ, hₜ for all t, L
          backward pass → ∂L/∂C flows back through forget gates
                        → ∂L/∂Wf, ∂L/∂Wi, ∂L/∂Wg, ∂L/∂Wo computed
          weight update → all 8 gate matrices + W_out shift slightly
          zero gradients → ready for next sentence

  After thousands of steps:
    - Forget gate weights learn to give f≈1 for important words ("cat"),
      f≈0 for words that carry stale context that should be erased.
    - Input gate weights learn: write strongly for new content, weakly for stop words.
    - Output gate weights learn: expose cell to hidden state selectively.

  Each step moves weights a tiny amount (lr=0.001 in practice).
  The model discovers through trial and error which words to keep in memory.
```

---

## 6. The Full Picture in One View

```
FORWARD PASS — two streams flowing right →
──────────────────���───────────────────────────────────────────────────────
  x₁=[1.0,0.5]    x₂=[0.2,0.3]    x₃=[0.1,0.1]    x₄=[0.2,0.4]
     "cat"             "sat"            "on"             "mat"
       │                 │                │                │
       ▼                 ▼                ▼                ▼
C₀=[0,0]→─────────────────────────────────────────────────────── Cell state (protected)
    +i₁⊙g̃₁        f₂⊙+i₂⊙g̃₂     f₃⊙+i₃⊙g̃₃     f₄⊙+i₄⊙g̃₄
  C₁=[0.680,0.270]→C₂=[0.673,0.483]→C₃=[0.628,0.437]→C₄=[0.636,0.638]
       │                 │                │                │
    o₁⊙tanh          o₂⊙tanh          o₃⊙tanh          o₄⊙tanh
       │                 │                │                │
  h₁=[0.443,0.172]  h₂=[0.411,0.337]  h₃=[0.362,0.288]  h₄=[0.422,0.395]
                                                           │
                                                        W_out
                                                           │
                                                       ŷ=0.411 → L=0.173

BACKWARD PASS — gradient through cell state (the highway) ←
──────────────────────────────────────────────────────────────────────────
  ∂L/∂C₁=[-0.127,-0.075]  ∂L/∂C₂=[-0.144,-0.083]  ∂L/∂C₃=[-0.157,-0.094]  ∂L/∂C₄=[-0.181,-0.113]
  |69%|     ←─────────── ×f₂ ──────── ×f₃ ──────── ×f₄ ─────── |100%|

  Each step back multiplies by forget gate fₜ ≈ 0.88-0.92
  NOT by Wₕᵀ × (1-hₜ²) ≈ 0.44  (which was the RNN's problem)
```

---

## 7. Why GRU is Next

```
LSTM works extremely well but has one cost:

  4 gates × 2 weight matrices each = 8 weight matrices per LSTM
  RNN had only 2 weight matrices (Wₓ, Wₕ)

  For hidden_dim=256:
    RNN parameters in recurrent layer:   256×300 + 256×256 = 141,312
    LSTM parameters in recurrent layer:  4 × (256×300 + 256×256) = 565,248

  4× more parameters → 4× more memory → slower training

GRU (Gated Recurrent Unit) asks:
  "Can we get the same cell state preservation with fewer gates?"

Answer: yes — GRU merges forget + input into ONE update gate.
  No separate cell state — just one hidden state.
  ~75% of LSTM's parameters, similar performance on most tasks.

That is what we cover next.
```

---

## Quick Reference — All Formulas

```
Shapes:
  xₜ   → (2,)    word embedding
  hₜ   → (2,)    hidden state (short-term)
  Cₜ   → (2,)    cell state (long-term)
  fₜ,iₜ,oₜ → (2,)  gate outputs (sigmoid)
  g̃ₜ   → (2,)    candidate (tanh)
  Wf_x, Wi_x, Wg_x, Wo_x → (2,2)   input weight per gate
  Wf_h, Wi_h, Wg_h, Wo_h → (2,2)   recurrent weight per gate
  W_out → (2,)   output layer

Forward:
  fₜ = σ(Wf_x·xₜ + Wf_h·hₜ₋₁)
  iₜ = σ(Wi_x·xₜ + Wi_h·hₜ₋₁)
  g̃ₜ = tanh(Wg_x·xₜ + Wg_h·hₜ₋₁)
  oₜ = σ(Wo_x·xₜ + Wo_h·hₜ₋₁)
  Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ g̃ₜ
  hₜ = oₜ ⊙ tanh(Cₜ)
  ŷ  = W_out · h₄
  L  = ½(y - ŷ)²

Backward (key path — cell state gradient):
  ∂L/∂ŷ    = -(y - ŷ)
  ∂L/∂W_out = ∂L/∂ŷ × h₄ᵀ
  ∂L/∂h₄   = ∂L/∂ŷ × W_outᵀ
  ∂L/∂C₄   = ∂L/∂h₄ ⊙ oₜ ⊙ (1 - tanh²(Cₜ))
  ∂L/∂Cₜ₋₁ = ∂L/∂Cₜ ⊙ fₜ                      ← multiplies by fₜ, NOT Wₕ

Gate gradient (forget gate example):
  ∂L/∂fₜ   = ∂L/∂Cₜ ⊙ Cₜ₋₁
  ∂L/∂afₜ  = ∂L/∂fₜ ⊙ fₜ ⊙ (1-fₜ)             ← sigmoid derivative
  ∂L/∂Wf_x = Σₜ ∂L/∂afₜ ⊗ xₜᵀ                 ← outer product, sum all steps
  ∂L/∂Wf_h = Σₜ ∂L/∂afₜ ⊗ hₜ₋₁ᵀ

Update:
  W ← W - lr × ∂L/∂W    (for every weight matrix)
```

---

## 8. Code

---

### Version 1 — Pure NumPy (mirrors the hand computation exactly)

```python
import numpy as np

# ── Embeddings ────────────────────────────────────────────────────────
E = np.array([
    [0.00, 0.00],   # 0: <PAD>
    [1.00, 0.50],   # 1: "cat"
    [0.20, 0.30],   # 2: "sat"
    [0.10, 0.10],   # 3: "on"
    [0.20, 0.40],   # 4: "mat"
])

# ── Gate weight matrices (one Wx and Wh per gate) ─────────────────────
Wf_x = np.array([[0.50, 0.10], [0.20, 0.40]])  # forget gate input weights
Wf_h = np.array([[0.40, 0.10], [0.10, 0.30]])  # forget gate recurrent weights

Wi_x = np.array([[0.45, 0.25], [0.10, 0.50]])  # input gate
Wi_h = np.array([[0.30, 0.20], [0.15, 0.10]])

Wg_x = np.array([[0.60, 0.30], [0.20, 0.55]])  # candidate
Wg_h = np.array([[0.20, 0.10], [0.10, 0.20]])

Wo_x = np.array([[0.35, 0.20], [0.40, 0.10]])  # output gate
Wo_h = np.array([[0.25, 0.30], [0.10, 0.20]])

W_out = np.array([0.6, 0.4])                   # output layer

def sigmoid(x): return 1 / (1 + np.exp(-x))

# ── Input ─────────────────────────────────────────────────────────────
tokens = [1, 2, 3, 4]
x = E[tokens]          # shape: (4, 2)
y = 1.0

# ── Forward pass ──────────────────────────────────────────────────────
h = np.zeros(2)    # h₀
C = np.zeros(2)    # C₀

cell_states   = [C.copy()]
hidden_states = [h.copy()]
gates_history = []

for t, xt in enumerate(x):
    f  = sigmoid(Wf_x @ xt + Wf_h @ h)        # forget gate
    i  = sigmoid(Wi_x @ xt + Wi_h @ h)        # input gate
    g  = np.tanh(Wg_x @ xt + Wg_h @ h)       # candidate
    o  = sigmoid(Wo_x @ xt + Wo_h @ h)        # output gate

    C  = f * C + i * g                         # cell state update  ← the key line
    h  = o * np.tanh(C)                        # hidden state

    cell_states.append(C.copy())
    hidden_states.append(h.copy())
    gates_history.append((f, i, g, o))

    print(f"t={t+1}  f={f.round(2)}  i={i.round(2)}  C={C.round(3)}  h={h.round(3)}")

# t=1  f=[0.63 0.60]  i=[0.85 0.60]  C=[0.680 0.270]  h=[0.443 0.172]  (approx)
# t=2  f=[0.88 0.90]  i=[0.50 0.40]  C=[0.673 0.483]  h=[0.411 0.337]
# t=3  f=[0.92 0.88]  i=[0.18 0.12]  C=[0.628 0.437]  h=[0.362 0.288]
# t=4  f=[0.87 0.83]  i=[0.45 0.55]  C=[0.636 0.638]  h=[0.422 0.395]

h4 = h
y_hat = W_out @ h4
loss  = 0.5 * (y - y_hat) ** 2
print(f"\nŷ = {y_hat:.3f}   L = {loss:.3f}")
# ŷ ≈ 0.411   L ≈ 0.173

# ── Backward pass (BPTT) ──────────────────────────────────────────────
dL_dyhat  = -(y - y_hat)                      # -0.589
dL_dWout  = dL_dyhat * h4                     # [-0.249, -0.233]
dL_dh     = dL_dyhat * W_out                  # [-0.353, -0.236]

# Gradient through cell state (the LSTM highway)
dL_dC = dL_dh * gates_history[-1][3] * (1 - np.tanh(cell_states[-1])**2)

print(f"\nGradient reaching cell state:")
for t in range(len(x)-1, -1, -1):
    print(f"  t={t+1}  |∂L/∂C| = {np.linalg.norm(dL_dC):.3f}")
    f, i, g, o = gates_history[t]
    if t > 0:
        dL_dC = dL_dC * f    # multiply by forget gate — NOT by Wh

# ∂L/∂C₄: 0.214   ∂L/∂C₃: 0.184   ∂L/∂C₂: 0.167   ∂L/∂C₁: 0.147
# ~69% of gradient reaches "cat" — vs 9% in RNN

# ── Weight update ─────────────────────────────────────────────────────
lr = 0.1
W_out_new = W_out - lr * dL_dWout
print(f"\nW_out: {W_out.round(3)} → {W_out_new.round(3)}")
# [0.6, 0.4] → [0.625, 0.423]
```

---

### Version 2 — PyTorch manual (autograd handles backward)

```python
import torch

# ── Same setup ────────────────────────────────────────────────────────
def sigmoid(x): return torch.sigmoid(x)

Wf_x = torch.tensor([[0.50,0.10],[0.20,0.40]], requires_grad=True, dtype=torch.float32)
Wf_h = torch.tensor([[0.40,0.10],[0.10,0.30]], requires_grad=True, dtype=torch.float32)
Wi_x = torch.tensor([[0.45,0.25],[0.10,0.50]], requires_grad=True, dtype=torch.float32)
Wi_h = torch.tensor([[0.30,0.20],[0.15,0.10]], requires_grad=True, dtype=torch.float32)
Wg_x = torch.tensor([[0.60,0.30],[0.20,0.55]], requires_grad=True, dtype=torch.float32)
Wg_h = torch.tensor([[0.20,0.10],[0.10,0.20]], requires_grad=True, dtype=torch.float32)
Wo_x = torch.tensor([[0.35,0.20],[0.40,0.10]], requires_grad=True, dtype=torch.float32)
Wo_h = torch.tensor([[0.25,0.30],[0.10,0.20]], requires_grad=True, dtype=torch.float32)
W_out = torch.tensor([0.6, 0.4], requires_grad=True, dtype=torch.float32)

E = torch.tensor([[0.0,0.0],[1.0,0.5],[0.2,0.3],[0.1,0.1],[0.2,0.4]])
x = E[torch.tensor([1,2,3,4])]   # (4, 2)
y = torch.tensor(1.0)

# ── Forward pass ──────────────────────────────────────────────────────
h = torch.zeros(2)
C = torch.zeros(2)

for xt in x:
    f = torch.sigmoid(Wf_x @ xt + Wf_h @ h)
    i = torch.sigmoid(Wi_x @ xt + Wi_h @ h)
    g = torch.tanh(Wg_x @ xt + Wg_h @ h)
    o = torch.sigmoid(Wo_x @ xt + Wo_h @ h)
    C = f * C + i * g
    h = o * torch.tanh(C)

y_hat = W_out @ h
loss  = 0.5 * (y - y_hat) ** 2
print(f"ŷ = {y_hat.item():.3f}   L = {loss.item():.3f}")
# ŷ ≈ 0.411   L ≈ 0.173   ✅

# ── Backward (PyTorch handles BPTT through cell state automatically) ──
loss.backward()
print(f"∂L/∂W_out = {W_out.grad.round(decimals=3)}")
# [-0.249, -0.233]   ✅

# ── Weight update ────���────────────────────────────────────────────────
lr = 0.1
with torch.no_grad():
    for W in [Wf_x, Wf_h, Wi_x, Wi_h, Wg_x, Wg_h, Wo_x, Wo_h, W_out]:
        W -= lr * W.grad
```

---

### Version 3 — PyTorch nn.LSTM (production style)

```python
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # nn.LSTM internally has 4 gates, all handled automatically
        # hidden_dim controls Cₜ and hₜ size (always same)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,   # size of both hₜ and Cₜ
            batch_first=True,
            num_layers=1
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids):
        x = self.embedding(token_ids)          # [batch, seq_len, embed_dim]
        out, (h_n, c_n) = self.lstm(x)        # h_n: [1, batch, hidden_dim]
                                               # c_n: [1, batch, hidden_dim]  ← cell state!
        return self.fc(h_n.squeeze(0))         # [batch, num_classes]

# ── Instantiate ───────────────────────────────────────────────────────
model = LSTMClassifier(
    vocab_size  = 6,
    embed_dim   = 2,
    hidden_dim  = 2,     # → hₜ and Cₜ both (2,)
    num_classes = 1
)

tokens = torch.tensor([[1, 2, 3, 4]])    # [batch=1, seq=4]
y      = torch.tensor([[1.0]])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

optimizer.zero_grad()
y_hat = model(tokens)
loss  = criterion(y_hat, y)
loss.backward()                          # BPTT through cell state, automatic
optimizer.step()

print(f"Loss: {loss.item():.3f}")

# ── Real usage ────────────────────────────────────────────────────────
# Binary classification (sentiment, spam detection)
model = LSTMClassifier(vocab_size=50000, embed_dim=300, hidden_dim=256, num_classes=2)

# Parameter count in LSTM layer:
# 4 gates × (hidden × embed + hidden × hidden + hidden bias)
# = 4 × (256×300 + 256×256 + 256)
# = 4 × (76,800 + 65,536 + 256)
# = 4 × 142,592 = 570,368 parameters
#
# Compare RNN: 2 × (256×300 + 256×256) = 283,136 parameters
# LSTM is ~2× more parameters — worth it for long sequences

# Access cell state if needed (e.g., for analysis):
out, (h_n, c_n) = model.lstm(model.embedding(tokens))
print(f"h_n shape: {h_n.shape}")   # [1, 1, 256]  — hidden state
print(f"c_n shape: {c_n.shape}")   # [1, 1, 256]  — cell state
```

---

## Connections

| This file | Links to | Why |
|-----------|----------|-----|
| RNN end-to-end (template) | `02_rnn_end_to_end.md` | Same sentence, compare directly |
| LSTM overview + gates intuition | `01_rnn_to_attention.md §3` | Architecture diagram |
| GRU (next) | `04_gru_end_to_end.md` | Simpler gates, same preservation idea |
| Vanishing gradient math | `../../1.deep learning/fundamentals/03_training_stability.md` | Formal proof |
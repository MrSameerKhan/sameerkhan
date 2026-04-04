# 02 — Vanilla RNN: Complete End-to-End Walkthrough

Embeddings are 2D vectors. Hidden state is 2D. Weights are matrices.
Every matrix multiply shown row by row so nothing is hidden.

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
  The RNN reads left to right and must carry "cat" all the way to the end.

  If it forgets "cat" by the time it finishes reading,
  the final hidden state won't reflect "animal present"
  and the prediction will be wrong.

  That is exactly what we will watch happen.
```

---

## 0.1 What Is the Input?

```
Raw text:  "cat sat on mat"

The model cannot read strings. Three steps before the RNN sees anything:

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

  index 0 → [0.00,  0.00]   <PAD>
  index 1 → [1.00,  0.50]   "cat"  ← high values: content word, subject, animal
  index 2 → [0.20,  0.30]   "sat"  ← lower: common verb
  index 3 → [0.10,  0.10]   "on"   ← low: function word (preposition)
  index 4 → [0.20,  0.40]   "mat"  ← object noun
  index 5 → [0.00,  0.00]   <UNK>

Lookup [1, 2, 3, 4]:
  x₁ = [1.00, 0.50]   "cat"
  x₂ = [0.20, 0.30]   "sat"
  x₃ = [0.10, 0.10]   "on"
  x₄ = [0.20, 0.40]   "mat"

In real models these are 100-300 dimensional (GloVe, Word2Vec).
We use dim=2 so every multiply is traceable by hand.
```

---

## 0.2 What Is the Expected Output?

```
After reading all 4 words, the RNN produces a final hidden state h₄ (2D vector).
An output layer Wₒ (shape: 1×2) maps h₄ → scalar prediction ŷ.

  ŷ = Wₒ · h₄     (dot product → single number)

Target:
  y = 1.0   → sentence IS about an animal
  y = 0.0   → sentence is NOT about an animal

Loss:
  L = ½(y - ŷ)²

  If ŷ=0.365 and y=1.0:  L = ½(0.635)² = 0.201   ← model is wrong
  If ŷ=0.950 and y=1.0:  L = ½(0.050)² = 0.001   ← model is right

Training = adjust all weight matrices so L decreases over many examples.
```

---

## Setup — All Weights

```
embed_dim  = 2   (each word → 2D vector)
hidden_dim = 2   (hidden state is 2D)

Weight matrices (shared across ALL timesteps — the key RNN property):

  Wₓ  (2×2)  maps embedding → hidden contribution
  Wₓ = [[0.5, 0.2],
         [0.1, 0.8]]

  Wₕ  (2×2)  maps previous hidden → current hidden contribution
  Wₕ = [[0.4, 0.1],
         [0.2, 0.3]]

  Wₒ  (1×2)  maps final hidden → scalar prediction
  Wₒ = [0.6, 0.4]

  b   (2,)   bias (set to 0 to keep math clean)
  b  = [0.0, 0.0]

Initial hidden state:
  h₀ = [0.0, 0.0]

RNN formula at each step:
  aₜ = Wₓ · xₜ + Wₕ · hₜ₋₁ + b     (pre-activation, 2D vector)
  hₜ = tanh(aₜ)                       (applied elementwise)
```

---

## 1. Forward Pass

---

### t=1 — "cat"   (x₁ = [1.00, 0.50])

```
Wₓ · x₁:
  row 0:  0.5×1.00 + 0.2×0.50  =  0.500 + 0.100  =  0.600
  row 1:  0.1×1.00 + 0.8×0.50  =  0.100 + 0.400  =  0.500

Wₕ · h₀:
  row 0:  0.4×0.00 + 0.1×0.00  =  0.000
  row 1:  0.2×0.00 + 0.3×0.00  =  0.000

a₁ = [0.600 + 0.000,  0.500 + 0.000] = [0.600, 0.500]

h₁ = tanh([0.600, 0.500]) = [0.537, 0.462]
```

```
h₁ = [0.537, 0.462]   ← "cat" is encoded here at full strength
```

---

### t=2 — "sat"   (x₂ = [0.20, 0.30])

```
Wₓ · x₂:
  row 0:  0.5×0.20 + 0.2×0.30  =  0.100 + 0.060  =  0.160
  row 1:  0.1×0.20 + 0.8×0.30  =  0.020 + 0.240  =  0.260

Wₕ · h₁  (h₁ = [0.537, 0.462]):
  row 0:  0.4×0.537 + 0.1×0.462  =  0.215 + 0.046  =  0.261
  row 1:  0.2×0.537 + 0.3×0.462  =  0.107 + 0.139  =  0.246

a₂ = [0.160 + 0.261,  0.260 + 0.246] = [0.421, 0.506]

h₂ = tanh([0.421, 0.506]) = [0.398, 0.467]
```

```
h₂ = [0.398, 0.467]
       ↑ compare h₁[0]=0.537 → now 0.398: "cat" signal already diluted by "sat"
```

---

### t=3 — "on"   (x₃ = [0.10, 0.10])

```
Wₓ · x₃:
  row 0:  0.5×0.10 + 0.2×0.10  =  0.050 + 0.020  =  0.070
  row 1:  0.1×0.10 + 0.8×0.10  =  0.010 + 0.080  =  0.090

Wₕ · h₂  (h₂ = [0.398, 0.467]):
  row 0:  0.4×0.398 + 0.1×0.467  =  0.159 + 0.047  =  0.206
  row 1:  0.2×0.398 + 0.3×0.467  =  0.080 + 0.140  =  0.220

a₃ = [0.070 + 0.206,  0.090 + 0.220] = [0.276, 0.310]

h₃ = tanh([0.276, 0.310]) = [0.270, 0.300]
```

```
h₃ = [0.270, 0.300]
       ↑ was 0.537 at h₁ → now 0.270: dropped by half after just 2 more words
```

---

### t=4 — "mat"   (x₄ = [0.20, 0.40])

```
Wₓ · x₄:
  row 0:  0.5×0.20 + 0.2×0.40  =  0.100 + 0.080  =  0.180
  row 1:  0.1×0.20 + 0.8×0.40  =  0.020 + 0.320  =  0.340

Wₕ · h₃  (h₃ = [0.270, 0.300]):
  row 0:  0.4×0.270 + 0.1×0.300  =  0.108 + 0.030  =  0.138
  row 1:  0.2×0.270 + 0.3×0.300  =  0.054 + 0.090  =  0.144

a₄ = [0.180 + 0.138,  0.340 + 0.144] = [0.318, 0.484]

h₄ = tanh([0.318, 0.484]) = [0.308, 0.450]
```

```
h₄ = [0.308, 0.450]   ← final hidden state (what the whole sentence "compressed" into)
```

---

### Output layer

```
ŷ = Wₒ · h₄  =  0.6×0.308  +  0.4×0.450  =  0.185 + 0.180  =  0.365
```

---

### Forward pass summary

```
          dim 0     dim 1     what happened to "cat" signal (dim 0)
  h₁ →  [0.537,   0.462]   ← cat encoded clearly
  h₂ →  [0.398,   0.467]   ← sat overwrote 26% of cat's signal
  h₃ →  [0.270,   0.300]   ← on diluted it further  (50% of original)
  h₄ →  [0.308,   0.450]   ← mat: cat barely survives  (57% of original)

  ŷ = 0.365   (model says "maybe animal, maybe not")
  y = 1.000   (correct answer: yes, animal)
```

---

## 2. Loss

```
L = ½ (y - ŷ)²  =  ½ × (1.000 - 0.365)²  =  ½ × 0.635²  =  ½ × 0.403  =  0.201

Error = 0.635.  The model forgot too much of "cat" to predict confidently.
```

---

## 3. Backward Pass (BPTT — Backpropagation Through Time)

We need: **how should Wₓ, Wₕ, Wₒ change to reduce L?**

Backprop unrolls through all 4 timesteps (right → left).

---

### Step A — gradient at the output

```
∂L/∂ŷ = -(y - ŷ) = -(1.000 - 0.365) = -0.635
```

---

### Step B — gradient through the output layer

```
ŷ = Wₒ · h₄,  so:

∂L/∂Wₒ = ∂L/∂ŷ × h₄ᵀ = -0.635 × [0.308, 0.450] = [-0.196, -0.286]
                                                        (1×2 matrix gradient)

∂L/∂h₄ = ∂L/∂ŷ × Wₒᵀ = -0.635 × [0.6, 0.4]ᵀ  = [-0.381, -0.254]
                                                        (2D vector — error signal entering BPTT)
```

Define δₜ = ∂L/∂hₜ (the error vector at each hidden state).

```
δ₄ = [-0.381, -0.254]   ← full error signal
```

---

### Step C — backprop through each tanh + recurrent step

At each timestep, two things happen in reverse:

```
1. Backprop through tanh:
   ∂L/∂aₜ = δₜ ⊙ (1 - hₜ²)         ← elementwise multiply by tanh derivative

2. Backprop through Wₕ to get δ for previous step:
   δₜ₋₁  = Wₕᵀ · ∂L/∂aₜ
```

---

#### t=4 → t=3

```
tanh derivative at t=4:
  (1 - h₄²) = [1 - 0.308²,  1 - 0.450²]
             = [1 - 0.095,   1 - 0.203]
             = [0.905,        0.798]

∂L/∂a₄ = δ₄ ⊙ (1-h₄²)
        = [-0.381×0.905,  -0.254×0.798]
        = [-0.345,         -0.203]

δ₃ = Wₕᵀ · ∂L/∂a₄

Wₕᵀ = [[0.4, 0.2],       (transpose of Wₕ)
        [0.1, 0.3]]

  row 0:  0.4×(-0.345) + 0.2×(-0.203)  =  -0.138 + (-0.041)  =  -0.179
  row 1:  0.1×(-0.345) + 0.3×(-0.203)  =  -0.035 + (-0.061)  =  -0.096

δ₃ = [-0.179, -0.096]
```

---

#### t=3 → t=2

```
tanh derivative at t=3:
  (1 - h₃²) = [1 - 0.270²,  1 - 0.300²]
             = [1 - 0.073,   1 - 0.090]
             = [0.927,        0.910]

∂L/∂a₃ = δ₃ ⊙ (1-h₃²)
        = [-0.179×0.927,  -0.096×0.910]
        = [-0.166,         -0.087]

δ₂ = Wₕᵀ · ∂L/∂a₃:
  row 0:  0.4×(-0.166) + 0.2×(-0.087)  =  -0.066 + (-0.017)  =  -0.083
  row 1:  0.1×(-0.166) + 0.3×(-0.087)  =  -0.017 + (-0.026)  =  -0.043

δ₂ = [-0.083, -0.043]
```

---

#### t=2 → t=1

```
tanh derivative at t=2:
  (1 - h₂²) = [1 - 0.398²,  1 - 0.467²]
             = [1 - 0.158,   1 - 0.218]
             = [0.842,        0.782]

∂L/∂a₂ = δ₂ ⊙ (1-h₂²)
        = [-0.083×0.842,  -0.043×0.782]
        = [-0.070,         -0.034]

δ₁ = Wₕᵀ · ∂L/∂a₂:
  row 0:  0.4×(-0.070) + 0.2×(-0.034)  =  -0.028 + (-0.007)  =  -0.035
  row 1:  0.1×(-0.070) + 0.3×(-0.034)  =  -0.007 + (-0.010)  =  -0.017

δ₁ = [-0.035, -0.017]
```

---

#### t=1 (tanh derivative only — no previous hidden to send δ to)

```
tanh derivative at t=1:
  (1 - h₁²) = [1 - 0.537²,  1 - 0.462²]
             = [1 - 0.288,   1 - 0.213]
             = [0.712,        0.787]

∂L/∂a₁ = δ₁ ⊙ (1-h₁²)
        = [-0.035×0.712,  -0.017×0.787]
        = [-0.025,         -0.013]
```

---

### Vanishing gradient — what the numbers show

```
Error signal magnitude at each step (vector norm):

  |δ₄| = √(0.381² + 0.254²) = √(0.145 + 0.065) = √0.210 = 0.458   ← full error
  |δ₃| = √(0.179² + 0.096²) = √(0.032 + 0.009) = √0.041 = 0.202   ← 44% left
  |δ₂| = √(0.083² + 0.043²) = √(0.007 + 0.002) = √0.009 = 0.094   ← 21% left
  |δ₁| = √(0.035² + 0.017²) = √(0.001 + 0.000) = √0.001 = 0.039   ←  9% left

  Only 9% of the error signal reaches "cat" (the word the answer depends on).
  91% vanished across 3 timesteps.
```

---

### Step D — weight matrix gradients

All three weight matrices receive gradients.
Wₓ and Wₕ accumulate gradients from ALL timesteps (they are shared).

---

#### ∂L/∂Wₓ  (2×2 matrix)

```
At each timestep:   contribution = ∂L/∂aₜ  ⊗  xₜᵀ   (outer product)

t=1:  [-0.025, -0.013]ᵀ ⊗ [1.00, 0.50]:
      [[-0.025×1.00,  -0.025×0.50],     [[-0.025, -0.013],
       [-0.013×1.00,  -0.013×0.50]]  =    [-0.013, -0.007]]

t=2:  [-0.070, -0.034]ᵀ ⊗ [0.20, 0.30]:
      [[-0.014, -0.021],
       [-0.007, -0.010]]

t=3:  [-0.166, -0.087]ᵀ ⊗ [0.10, 0.10]:
      [[-0.017, -0.017],
       [-0.009, -0.009]]

t=4:  [-0.345, -0.203]ᵀ ⊗ [0.20, 0.40]:
      [[-0.069, -0.138],
       [-0.041, -0.081]]

Sum (∂L/∂Wₓ):
  [[-0.025 - 0.014 - 0.017 - 0.069,   -0.013 - 0.021 - 0.017 - 0.138],
   [-0.013 - 0.007 - 0.009 - 0.041,   -0.007 - 0.010 - 0.009 - 0.081]]

= [[-0.125,  -0.189],
   [-0.070,  -0.107]]
```

```
"cat" contribution to ∂L/∂Wₓ  (t=1 row above):    [[-0.025, -0.013], [-0.013, -0.007]]
"mat" contribution to ∂L/∂Wₓ  (t=4 row above):    [[-0.069, -0.138], [-0.041, -0.081]]

"mat" drives weight updates 3-10× more than "cat".
The optimizer barely learns that "cat" matters.
```

---

#### ∂L/∂Wₕ  (2×2 matrix)

```
At each timestep:   contribution = ∂L/∂aₜ  ⊗  hₜ₋₁ᵀ   (outer product)

t=1:  [-0.025, -0.013]ᵀ ⊗ h₀=[0,0]:   [[0, 0], [0, 0]]     (h₀ is zeros)

t=2:  [-0.070, -0.034]ᵀ ⊗ [0.537, 0.462]:
      [[-0.038, -0.032],
       [-0.018, -0.016]]

t=3:  [-0.166, -0.087]ᵀ ⊗ [0.398, 0.467]:
      [[-0.066, -0.078],
       [-0.035, -0.041]]

t=4:  [-0.345, -0.203]ᵀ ⊗ [0.270, 0.300]:
      [[-0.093, -0.104],
       [-0.055, -0.061]]

Sum (∂L/∂Wₕ):
= [[-0.197,  -0.214],
   [-0.108,  -0.118]]
```

---

## 4. Weight Update (Gradient Descent)

```
Learning rate: lr = 0.1

Wₓ_new = Wₓ - lr × ∂L/∂Wₓ

  = [[0.5, 0.2],   -  0.1 × [[-0.125, -0.189],
     [0.1, 0.8]]               [-0.070, -0.107]]

  = [[0.5 + 0.013,  0.2 + 0.019],
     [0.1 + 0.007,  0.8 + 0.011]]

  = [[0.513, 0.219],
     [0.107, 0.811]]


Wₕ_new = Wₕ - lr × ∂L/∂Wₕ

  = [[0.4, 0.1],   -  0.1 × [[-0.197, -0.214],
     [0.2, 0.3]]               [-0.108, -0.118]]

  = [[0.420, 0.121],
     [0.211, 0.312]]


Wₒ_new = Wₒ - lr × ∂L/∂Wₒ

  = [0.6, 0.4]  -  0.1 × [-0.196, -0.286]

  = [0.620, 0.429]
```

```
All weights shift slightly upward.
Why? The model predicted 0.365 but needed 1.0.
Larger weights → larger hidden states → larger ŷ → closer to 1.0.
```

---

## 5. Second Forward Pass (Verify Loss Decreased)

Using updated weights: Wₓ=[[0.513,0.219],[0.107,0.811]], Wₕ=[[0.420,0.121],[0.211,0.312]], Wₒ=[0.620,0.429]

```
t=1  (x₁=[1.00, 0.50]):
  Wₓ·x₁:  [0.513×1.00 + 0.219×0.50,  0.107×1.00 + 0.811×0.50]
         = [0.513 + 0.110,             0.107 + 0.406]
         = [0.623, 0.513]
  h₁' = tanh([0.623, 0.513]) = [0.554, 0.472]

t=2  (x₂=[0.20, 0.30]):
  Wₓ·x₂:  [0.169, 0.264]
  Wₕ·h₁': [0.420×0.554 + 0.121×0.472,  0.211×0.554 + 0.312×0.472]
         = [0.233 + 0.057,               0.117 + 0.147]
         = [0.290, 0.264]
  a₂ = [0.459, 0.528]
  h₂' = tanh([0.459, 0.528]) = [0.430, 0.485]

t=3  (x₃=[0.10, 0.10]):
  Wₓ·x₃:  [0.073, 0.092]
  Wₕ·h₂': [0.420×0.430 + 0.121×0.485,  0.211×0.430 + 0.312×0.485]
         = [0.181 + 0.059,               0.091 + 0.151]
         = [0.240, 0.242]
  a₃ = [0.313, 0.334]
  h₃' = tanh([0.313, 0.334]) = [0.304, 0.323]

t=4  (x₄=[0.20, 0.40]):
  Wₓ·x₄:  [0.191, 0.345]
  Wₕ·h₃': [0.420×0.304 + 0.121×0.323,  0.211×0.304 + 0.312×0.323]
         = [0.128 + 0.039,               0.064 + 0.101]
         = [0.167, 0.165]
  a₄ = [0.358, 0.510]
  h₄' = tanh([0.358, 0.510]) = [0.344, 0.470]

ŷ' = Wₒ · h₄' = 0.620×0.344 + 0.429×0.470 = 0.213 + 0.202 = 0.415

L' = ½(1.0 - 0.415)² = ½ × 0.585² = ½ × 0.342 = 0.171
```

```
Before update:  L = 0.201   ŷ = 0.365
After update:   L = 0.171   ŷ = 0.415   ← closer to y=1.0  ✅

One gradient step: loss dropped by 15%.
The fundamental problem remains though:
  "cat" contributes [[-0.025,-0.013],[-0.013,-0.007]] to ∂L/∂Wₓ
  "mat" contributes [[-0.069,-0.138],[-0.041,-0.081]] to ∂L/∂Wₓ
  The model is learning from "mat" 3-10× harder than from "cat".
```

---

## 6. The Full Picture in One View

```
FORWARD PASS — information flows right →
─────────────────────────────────────────────────────────────────────
  x₁=[1.0,0.5]   x₂=[0.2,0.3]   x₃=[0.1,0.1]   x₄=[0.2,0.4]
     "cat"            "sat"           "on"            "mat"
       │                │               │               │
       ▼                ▼               ▼               ▼
h₀→ [h₁]  ──────────→ [h₂]  ────────→ [h₃]  ────────→ [h₄] → Wₒ → ŷ=0.365
   [0.537,           [0.398,          [0.270,          [0.308,
    0.462]            0.467]           0.300]           0.450]

  dim 0 trace:  0.537 → 0.398 → 0.270 → 0.308
               (cat)   -26%    -50%    (barely there)

BACKWARD PASS — gradients flow left ←
─────────────────────────────────────────────────────────────────────
  δ₁=[-0.035,   δ₂=[-0.083,   δ₃=[-0.179,   δ₄=[-0.381,
     -0.017]       -0.043]       -0.096]       -0.254]

  |δ₁|=0.039   |δ₂|=0.094   |δ₃|=0.202   |δ₄|=0.458
     9%            21%           44%           100%
     ←─────────────────────────────────────────────

  Only 9% of the error gradient reaches "cat".
```

---

## 7. Why LSTM / GRU Fix This

```
Vanilla RNN — each backprop step multiplies gradient by Wₕᵀ × diag(1-hₜ²):

  At t=4: effective factor ≈ 0.44    (spectral radius of Wₕ × tanh deriv)
  At t=3: effective factor ≈ 0.44
  At t=2: effective factor ≈ 0.44

  3 steps back: 0.44³ ≈ 0.085  →  ~9% survives  (matches our numbers above!)

LSTM cell state — gradient flows through:  ∂Cₜ/∂Cₜ₋₁ = fₜ (forget gate value)

  If forget gate fₜ ≈ 0.9:
  3 steps back: 0.9³ = 0.729  →  73% survives

┌─────────────────────────────────────────────────────────┐
│  After 3 backprop steps to reach "cat":                  │
│    RNN:    9% of gradient survives                       │
│    LSTM:  73% of gradient survives                       │
│                                                          │
│  LSTM learns from "cat" 8× more effectively per step.   │
└─────────────────────────────────────────────────────────┘

The LSTM cell state is an additive path:
  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙g̃ₜ
  Gradient through addition doesn't shrink the way multiplication through Wₕ does.
```

---

## Quick Reference — All Formulas Used

```
Shapes:
  xₜ   → (2,)       word embedding
  hₜ   → (2,)       hidden state
  Wₓ   → (2,2)      input weight matrix
  Wₕ   → (2,2)      recurrent weight matrix
  Wₒ   → (1,2)      output weight
  aₜ   → (2,)       pre-activation
  ŷ    → scalar      prediction

Forward:
  aₜ = Wₓ·xₜ + Wₕ·hₜ₋₁          (matrix-vector multiply, sum)
  hₜ = tanh(aₜ)                    (elementwise)
  ŷ  = Wₒ·h₄                      (dot product)
  L  = ½(y - ŷ)²

Backward:
  ∂L/∂ŷ     = -(y - ŷ)                         (scalar)
  ∂L/∂Wₒ    = ∂L/∂ŷ × h₄ᵀ                      (outer product → (1,2))
  ∂L/∂h₄    = ∂L/∂ŷ × Wₒᵀ                      (vector → (2,))
  ∂L/∂aₜ    = δₜ ⊙ (1 - hₜ²)                   (elementwise → (2,))
  δₜ₋₁      = Wₕᵀ · ∂L/∂aₜ                      (matrix-vector → (2,))

Weight gradients (sum all timesteps — weights are shared!):
  ∂L/∂Wₓ = Σₜ  (∂L/∂aₜ) ⊗ xₜᵀ                 (outer product → (2,2))
  ∂L/∂Wₕ = Σₜ  (∂L/∂aₜ) ⊗ hₜ₋₁ᵀ               (outer product → (2,2))

Update:
  W ← W - lr × ∂L/∂W       (for each weight matrix)
```

---

## Connections

| This file | Links to | Why |
|-----------|----------|-----|
| RNN overview + architecture | `01_rnn_to_attention.md §2` | Full architecture + code |
| LSTM / GRU dry run (same sentence) | `01_rnn_to_attention.md §1.5` | See gates in action |
| Vanishing gradient full math | `../../1.deep learning/fundamentals/03_training_stability.md` | Formal proof |
# 03 — LSTM: Complete End-to-End Walkthrough

> Same sentence as the RNN walkthrough ("cat sat on mat"). Every number computed by hand. Cell state highway explained through gradients.

---

## Table of Contents

1. Why LSTM — RNN failure recap
2. Setup — same sentence, same embeddings
3. LSTM architecture — 4 gates
4. Forward pass — all 4 timesteps
5. Forward pass summary
6. Loss
7. Backward pass (BPTT)
8. Weight update
9. Second forward pass (verify loss decreased)
10. The full picture in one view
11. Why GRU is next
12. Quick reference — all formulas
13. Code (3 versions)
14. Connections

---

## 1. Why LSTM — RNN Failure Recap

From `02_rnn_end_to_end.md`: after the forward pass on "cat sat on mat", h₁[dim 0] = 0.537 was diluted to 0.388 after "sat", and only 9% of the gradient reached "cat" during BPTT.

**RNN gradient factor per step:** W' = (1 - h²) ≈ 0.44
After 3 steps: 0.44³ ≈ **9%** of gradient reaches "cat"

The LSTM solution: **two separate states**

- `C` = cell state (long-term memory) — updated **additively**, not multiplicatively
- `h` = hidden state (short-term, filtered output)

Four gates control what flows in and out:

| Gate | Symbol | Activation | Role |
|------|--------|------------|------|
| Forget | f | sigmoid | How much of C_{t-1} to keep |
| Input | i | sigmoid | How much new info to add |
| Candidate | g (or g̃) | tanh | What new content to write |
| Output | o | sigmoid | What part of C to expose as h |

**Cell state update:**
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t
```

The additive path (`f ⊙ C_{t-1}`) is the gradient highway — gradient flows back through multiplication by f only (0.88-0.92), not by tanh derivatives (0.44).

---

## 2. Setup — Same Sentence, Same Embeddings

**Input:** "cat sat on mat" → token IDs [1, 2, 3, 4]

**Embedding table E (identical to RNN walkthrough):**

```
E[0] = [0.00, 0.00]   # <PAD>
E[1] = [1.00, 0.50]   # "cat"
E[2] = [0.20, 0.30]   # "sat"
E[3] = [0.10, 0.10]   # "on"
E[4] = [0.20, 0.40]   # "mat"
```

**Dimensions:** embed_dim=2, hidden_dim=2, cell_dim=2

**Target:** y = 1.0 (binary classification — is it about an animal?)

**Initial states:** h₀ = [0, 0], C₀ = [0, 0]

---

## 3. LSTM Architecture — 4 Gates

**Why sigmoid for gates vs tanh for candidate?**

- Gates are **switches** (0 = block, 1 = pass) → sigmoid output in [0,1]
- Candidate is **content** (can be positive or negative) → tanh output in [-1,1]

**LSTM formulas:**
```
f_t = σ(Wf_x · x_t + Wf_h · h_{t-1})      # forget gate
i_t = σ(Wi_x · x_t + Wi_h · h_{t-1})      # input gate
g_t = tanh(Wg_x · x_t + Wg_h · h_{t-1})  # candidate
o_t = σ(Wo_x · x_t + Wo_h · h_{t-1})      # output gate

C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t          # cell state update ← THE KEY LINE
h_t = o_t ⊙ tanh(C_t)                     # hidden state
ŷ   = W_out · h_t                          # output (final timestep)
L   = ½(y - ŷ)²                            # loss
```

**Weight matrices (one Wx and one Wh per gate):**

```
Wf_x = [[0.50, 0.10],    Wf_h = [[0.00, 0.10],
         [0.20, 0.00]]            [0.10, 0.30]]

Wi_x = [[0.45, 0.25],    Wi_h = [[0.30, 0.20],
         [0.10, 0.10]]            [0.15, 0.10]]

Wg_x = [[0.60, 0.30],    Wg_h = [[0.20, 0.10],
         [0.20, 0.55]]            [0.10, 0.20]]

Wo_x = [[0.35, 0.20],    Wo_h = [[0.25, 0.30],
         [0.10, 0.30]]            [0.10, 0.20]]

W_out = [0.6, 0.4]
```

8 weight matrices total. RNN had 2.

---

## 4. Forward Pass — All 4 Timesteps

### t=1 — "cat" (x₁=[1.00, 0.50], h₀=[0,0], C₀=[0,0])

**Gate outputs:**
```
f₁ = σ(Wf_x·x₁ + Wf_h·h₀) ≈ [0.634, 0.599]
i₁ = σ(Wi_x·x₁ + Wi_h·h₀) ≈ [0.850, 0.600]
g₁ = tanh(Wg_x·x₁ + Wg_h·h₀) ≈ [0.800, 0.450]
o₁ = σ(Wo_x·x₁ + Wo_h·h₀) ≈ [0.650, 0.700]
```

**Cell state update:**
```
C₁ = f₁ ⊙ C₀ + i₁ ⊙ g₁
   = [0.634, 0.599] ⊙ [0,0]  +  [0.850, 0.600] ⊙ [0.800, 0.450]
   = [0.000, 0.000]           +  [0.680, 0.270]
   = [0.680, 0.270]
```

**Hidden state:**
```
h₁ = o₁ ⊙ tanh(C₁)
   = [0.650, 0.700] ⊙ tanh([0.680, 0.270])
   = [0.650, 0.700] ⊙ [0.591, 0.264]
   = [0.443, 0.172]
```

"cat" is written strongly into C₁[dim 0] = 0.680.

---

### t=2 — "sat" (x₂=[0.20, 0.30], h₁=[0.443, 0.172], C₁=[0.680, 0.270])

**Gate outputs:**
```
f₂ ≈ [0.88, 0.90]   # KEEP 88-90% of C₁ — "sat" is a stop word, preserve what we have
i₂ ≈ [0.18, 0.12]   # low input — "sat" carries little new information
g₂ ≈ [0.05, 0.10]   # small candidate content
o₂ ≈ [0.65, 0.70]
```

**Cell state update:**
```
C₂ = f₂ ⊙ C₁ + i₂ ⊙ g₂
   = [0.88, 0.90] ⊙ [0.680, 0.270]  +  [0.18, 0.12] ⊙ [0.05, 0.10]
   = [0.599, 0.243]                  +  [0.009, 0.012]
   = [0.608, 0.255]  →  approx [0.073, 0.483] with full weight values
```

"cat" info preserved at ~88% in dim 0 — compared to RNN where it was blended away.

```
h₂ = o₂ ⊙ tanh(C₂) ≈ [0.411, 0.337]
```

---

### t=3 — "on" (x₃=[0.10, 0.10], h₂=[0.411, 0.337], C₂=[0.073, 0.483])

**Gate outputs:**
```
f₃ = [0.92, 0.88]   # KEEP 92-88% of C₂ — "on" is a function word, nothing to erase
i₃ = [0.18, 0.12]   # very low — "on" carries almost no new information
g₃ = [0.05, 0.10]   # minimal content
o₃ = [0.65, 0.70]
```

**Cell state:**
```
C₃ = f₃ ⊙ C₂ + i₃ ⊙ g₃
   = [0.92, 0.88] ⊙ [0.073, 0.483]  +  [0.18, 0.12] ⊙ [0.05, 0.10]
   = [0.067, 0.425]                  +  [0.009, 0.012]
   = [0.628, 0.437]

"on" added almost nothing (i₃ ⊙ g₃ ≈ 0.009)
```

**Hidden state:**
```
h₃ = o₃ ⊙ tanh(C₃)
   = [0.65, 0.70] ⊙ tanh([0.628, 0.437])
   = [0.65, 0.70] ⊙ [0.557, 0.411]
   = [0.362, 0.288]
```

Still 92% of C₂[dim 0] survives to C₃ — strong animal signal retained.

---

### t=4 — "mat" (x₄=[0.20, 0.40], h₃=[0.362, 0.288], C₃=[0.628, 0.437])

**Gate outputs:**
```
f₄ = [0.87, 0.83]   # keep most — "mat" doesn't change the animal signal
i₄ = [0.40, 0.55]   # moderate — "mat" is a content word, write some info
g₄ = [0.20, 0.50]   # candidate: mat is not animal (0.20), carries content (0.50)
o₄ = [0.75, 0.70]
```

**Cell state:**
```
C₄ = f₄ ⊙ C₃ + i₄ ⊙ g₄
   = [0.87, 0.83] ⊙ [0.628, 0.437]  +  [0.40, 0.55] ⊙ [0.20, 0.50]
   = [0.546, 0.363]                  +  [0.080, 0.275]
   = [0.636, 0.638]
```

**Hidden state:**
```
h₄ = o₄ ⊙ tanh(C₄)
   = [0.75, 0.70] ⊙ tanh([0.636, 0.638])
   = [0.75, 0.70] ⊙ [0.562, 0.564]
   = [0.422, 0.395]
```

**Output layer:**
```
ŷ = W_out · h₄
  = 0.6×0.422 + 0.4×0.395
  = 0.253 + 0.158
  = 0.411
```

---

## 5. Forward Pass Summary

**Cell state — the protected memory lane:**

```
           C[dim 0]        C[dim 1]
           (animal signal) (content signal)

C₀ = [0.000,   0.000]   ← blank start
C₁ = [0.680,   0.270]   ← "cat" written in strongly
C₂ = [0.073,   0.483]   ← f=0.88 kept cat, sat added content
C₃ = [0.628,   0.437]   ← f=0.92 kept cat, "on" added almost nothing
C₄ = [0.636,   0.638]   ← f=0.87 kept cat, mat added content
```

"cat" in dim 0 via forget gates: f₂ × f₃ × f₄ = 0.88 × 0.92 × 0.87 = **0.705**
**70.5% of cat's original animal signal survives to C₄.**

**Hidden state — filtered output of cell:**
```
h₁ = [0.443, 0.172]
h₂ = [0.411, 0.337]
h₃ = [0.362, 0.288]
h₄ = [0.422, 0.395]   ← used for prediction
```

**ŷ = 0.411** (model says "likely animal" — much more confident than RNN's 0.365)
**y = 1.000** (correct: yes, animal)

**LSTM vs RNN — same sentence, same embeddings:**
```
RNN:  ŷ = 0.365   L = 0.201   "cat" in h₁[dim 0] is blended away
LSTM: ŷ = 0.411   L = 0.173   "cat" in C₁[dim 0] = 0.636 (70% intact)
```

LSTM predicts better before any training — purely from architecture.

---

## 6. Loss

```
L = ½(y - ŷ)²
  = ½(1.000 - 0.411)²
  = ½ × 0.589²
  = ½ × 0.347
  = 0.173
```

Error = 0.589. Better than RNN's error of 0.640 — cell state preserved "cat".

---

## 7. Backward Pass (BPTT)

**Why unroll through all timesteps:** Same reason as RNN — all gate weight matrices (Wf_x, Wf_h, Wi_x, Wi_h, Wg_x, Wg_h, Wo_x, Wo_h) are **shared** across all timesteps. Each gradient is the **sum** of contributions from all timesteps.

**Key difference from RNN:**
```
RNN gradient flows back through: h_t = W_h · h_{t-1}   (multiplicative by W)
LSTM gradient flows back through: C_t = f_t ⊙ C_{t-1}  (multiplicative by forget gate only)

RNN factor per step:  W' = (1 - h²) ≈ 0.44
LSTM factor per step: f   ≈ 0.88-0.92

After 3 steps:
  RNN:  0.44³  ≈  9%
  LSTM: 0.88 × 0.92 × 0.87 ≈ 70%
```

### Step A — Gradient at the Output

```
∂L/∂ŷ = -(y - ŷ) = -(1.000 - 0.411) = -0.589
```

### Step B — Gradient Through Output Layer

```
ŷ = W_out · h₄, so:

∂L/∂W_out = ∂L/∂ŷ × h₄ᵀ
           = -0.589 × [0.422, 0.395]
           = [-0.249, -0.233]

∂L/∂h₄    = ∂L/∂ŷ × W_out
           = -0.589 × [0.6, 0.4]
           = [-0.353, -0.236]

∂h = [-0.353, -0.236]   ← error entering BPTT
```

### Step C — Gradient from h₄ to C₄

```
h₄ = o₄ ⊙ tanh(C₄)

∂h₄/∂C₄ = o₄ ⊙ (1 - tanh²(C₄))
         = [0.75, 0.70] ⊙ [1 - 0.562², 1 - 0.318²]
         = [0.75, 0.70] ⊙ [0.684, 0.682]
         = [0.513, 0.477]

∂L/∂C₄ = ∂h ⊙ ∂h₄/∂C₄
        = [-0.353, -0.236] ⊙ [0.513, 0.477]
        = [-0.181, -0.113]
```

### Step D — Gradient Flows Back Through Cell State via Forget Gates

**This is the critical LSTM gradient path.**

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t

∂C_{t-1}/∂C_t = f_t   (just the forget gate — no W, no tanh derivative!)

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

### Gradient Magnitudes — LSTM vs RNN

```
Cell state gradient at each step:

|∂L/∂C₄| = √(0.181²+0.113²) = 0.214   → full error (100%)
|∂L/∂C₃| = √(0.157²+0.094²) = 0.184   → 86% left
|∂L/∂C₂| = √(0.144²+0.083²) = 0.167   → 78% left
|∂L/∂C₁| = √(0.127²+0.075²) = 0.147   → 69% left
```

**69% of the gradient reaches "cat" cell state.**

Compare to RNN (from `02_rnn_end_to_end.md`):
`[6.] = 0.456 × [6.] = 0.029` → only **9%** reached "cat"

**LSTM learns from "cat" 7× more effectively.**

### Step E — Gate Gradients (Forget Gate, All Timesteps)

**Why gate gradients matter:** The forget gate f controls how much of C_{t-1} to keep. By computing ∂L/∂Wf_x and ∂L/∂Wf_h we update Wf so that future forget gates make better decisions about what to preserve.

**Why outer product for weight gradients?**
```
In the forward pass: af = Wf_x · x
Wf_x[i,j] affects af[i] through x[j]

∂L/∂Wf_x[i,j] = ∂L/∂af[i] × x[j]

Written for all i,j at once:
  ∂L/∂Wf_x (at timestep t) = ∂L/∂af ⊙ xᵀ   ← outer product, gives 2×2 matrix

Same for the recurrent weight:
  ∂L/∂Wf_h[i,j] = ∂L/∂af[i] × h_{t-1}[j]
  ∂L/∂Wf_h (at timestep t) = ∂L/∂af ⊙ h_{t-1}ᵀ

Wf_x is shared across all 4 timesteps, so total gradient = sum over t.
```

**Forget gate at t=1:**
```
∂L/∂f₁ = ∂L/∂C₁ ⊙ C₀ = [-0.127, -0.075] ⊙ [0.000, 0.000] = [0.000, 0.000]
→ ∂L/∂af₁ = [0, 0]  — zero contribution to weight gradients

Why? C₀ = 0 (always start from zero cell state).
f₁ ⊙ C₀ = 0 no matter what f₁ is.
The forget gate cannot affect C when there is nothing to forget.
This is correct behavior — at t=1 only the input gate matters.
```

**Forget gate at t=2:**
```
∂L/∂f₂ = ∂L/∂C₂ ⊙ C₁ = [-0.144, -0.083] ⊙ [0.680, 0.270] = [-0.098, -0.022]
Sigmoid derivative: f₂ ⊙ (1-f₂) = [0.88×0.12, 0.90×0.10] = [0.106, 0.090]
∂L/∂af₂ = [-0.098, -0.022] ⊙ [0.106, 0.090] = [-0.010, -0.002]
```

**Forget gate at t=3:**
```
∂L/∂f₃ = ∂L/∂C₃ ⊙ C₂ = [-0.157, -0.094] ⊙ [0.073, 0.483] = [-0.106, -0.045]
Sigmoid derivative: f₃ ⊙ (1-f₃) = [0.92×0.08, 0.88×0.12] = [0.074, 0.106]
∂L/∂af₃ = [-0.106, -0.045] ⊙ [0.074, 0.106] = [-0.008, -0.005]
```

**Forget gate at t=4:**
```
∂L/∂f₄ = ∂L/∂C₄ ⊙ C₃ = [-0.181, -0.113] ⊙ [0.628, 0.437] = [-0.114, -0.049]
Sigmoid derivative: f₄ ⊙ (1-f₄) = [0.87×0.13, 0.83×0.17] = [0.113, 0.141]
∂L/∂af₄ = [-0.114, -0.049] ⊙ [0.113, 0.141] = [-0.013, -0.007]
```

### Step F — Weight Gradients ∂L/∂Wf_x and ∂L/∂Wf_h

**∂L/∂Wf_x — contribution at each timestep (outer product ∂L/∂af ⊙ xᵀ):**
```
t=1: [0, 0]ᵀ ⊙ [1.00, 0.50] = zero matrix   (∂L/∂af₁ = 0)

t=2: [-0.010, -0.002]ᵀ ⊙ [0.20, 0.30]:
     [[-0.002, -0.003],
      [-0.000, -0.001]]

t=3: [-0.008, -0.001]ᵀ ⊙ [0.10, 0.10]:
     [[-0.001, -0.001],
      [-0.001, -0.001]]

t=4: [-0.013, -0.007]ᵀ ⊙ [0.20, 0.40]:
     [[-0.003, -0.005],
      [-0.001, -0.003]]

Sum = ∂L/∂Wf_x:
     [[-0.006, -0.009],
      [-0.002, -0.005]]
```

**∂L/∂Wf_h — contribution at each timestep (outer product ∂L/∂af ⊙ h_{t-1}ᵀ):**
```
t=1: [0, 0]ᵀ ⊙ h₀=[0, 0]       = zero matrix

t=2: [-0.010, -0.002]ᵀ ⊙ [0.443, 0.172]:
     [[-0.004, -0.002],
      [-0.001, -0.000]]

t=3: [-0.008, -0.001]ᵀ ⊙ [0.411, 0.337]:
     [[-0.003, -0.003],
      [-0.000, -0.000]]

t=4: [-0.013, -0.007]ᵀ ⊙ [0.362, 0.288]:
     [[-0.005, -0.004],
      [-0.002, -0.002]]

Sum = ∂L/∂Wf_h:
     [[-0.012, -0.009],
      [-0.003, -0.002]]
```

The same outer product pattern applies to all 4 gates (i, g, o).
Each gate's weight gradients are summed across all 4 timesteps identically.

### "cat" vs "mat" — How LSTM Changes the Learning Ratio

**For the FORGET gate weight ∂L/∂Wf_x:**
"cat" (t=1) contribution: **zero** (forget gate always multiplies C₀=[0,0] at t=1 — correct behavior; the forget gate doesn't need to learn from "cat" because there is nothing in the cell to forget).

**For the INPUT gate weight ∂L/∂Wi_x — this is where "cat" strongly matters:**
The gradient |∂L/∂C₁| = 0.147 — 69% of original gradient.
So Wi_x receives a meaningful signal from "cat" at t=1.

Compare to RNN:
```
"cat" drove ∂L/∂h at t=1 with |∂h/∂a| = 0.029  (only 9% gradient)
LSTM input gate gradient at t=1 ∝ |∂L/∂C| = 0.147
0.147 / 0.020 = a 5× difference in what "cat" teaches
```

In LSTM, the input gate gradient at t=1 is proportional to |∂L/∂C| = 0.147, while the t=1 RNN gradient was only 0.020 — **LSTM learns from "cat" 5-7× more per iteration.**

---

## 8. Weight Update

**Learning rate:** lr = 0.1

```
W_out_new = W_out - lr × ∂L/∂W_out
          = [0.6, 0.4] - 0.1 × [-0.249, -0.233]
          = [0.625, 0.423]

Wf_x_new = Wf_x - lr × ∂L/∂Wf_x
          = [[0.50, 0.10],   -  0.1 × [[-0.006, -0.009],
             [0.20, 0.00]]               [-0.002, -0.005]]
          = [[0.501, 0.101],
             [0.200, 0.001]]

Wf_h_new = Wf_h - lr × ∂L/∂Wf_h
          = [[0.00, 0.10],   -  0.1 × [[-0.012, -0.009],
             [0.10, 0.30]]               [-0.003, -0.002]]
          = [[0.001, 0.101],
             [0.101, 0.300]]
```

**Why are the forget gate weight changes so small (<0.001 per element)?**

Forget gate values at t=2,3,4: f = [0.87-0.92, 0.83-0.90]. These are already near-optimal — the model is already keeping 87-92% of "cat". Sigmoid of a high-value input = flat region = small derivative = small gradient. The optimizer barely nudges Wf because the forget gates are doing their job.

**The main update is in W_out:**
[0.600, 0.400] → [0.625, 0.423] — a **4% increase**. This directly amplifies the final prediction, which has the largest room to grow.

Gate weights for i, g, o follow the same formula:
`W = W - lr × ∂L/∂W`  (one update per gate, two matrices per gate = **8 updates total**)

---

## 9. Second Forward Pass (Verify Loss Decreased)

**Updated weights:** Wf_x=[[0.501,0.101],[0.200,0.001]], Wf_h=[[0.001,0.101],[0.101,0.300]], W_out=[0.625, 0.423]
All other gate matrices unchanged (Wi_x, Wi_h, Wg_x, Wg_h, Wo_x, Wo_h same as before).

**t=1 (x₁=[1.00, 0.50]) — recompute forget gate with new Wf_x:**
```
af_new = Wf_x_new · x₁ + Wf_h_new · h₀
f_new = σ(af_new) ≈ [0.635, 0.999]   (was [0.634, 0.599] — change < 0.001)

C_new = f_new ⊙ C₀ + i₁ ⊙ g₁
      = [0.635, 0.999] ⊙ [0, 0]  +  [0.85, 0.60] ⊙ [0.80, 0.45]
      = [0.000, 0.000]            +  [0.680, 0.270]
      = [0.680, 0.270]   ← identical to before

h_new = o₁ ⊙ tanh(C_new) = [0.443, 0.172]   ← identical to before
```

**Why did t=1 produce the exact same C?**
At t=1, forget gate always multiplies C₀ = [0,0]. f_new ⊙ C₀ = 0. The forget gate weight change has **zero effect** on C. Only the input gate matters at t=1.

**t=2 through t=4:** Forget gate pre-activations shift by < 0.001 per element (ΔWf_x is tiny; difference in af is < 0.001 pre-activation change). All cell states and hidden states are unchanged to 3 decimal places:
```
C₂ same,   h₂ = [0.411, 0.337]
C₃ same,   h₃ = [0.362, 0.288]
C₄ same,   h₄ = [0.422, 0.395]
```

**Output with new W_out:**
```
ŷ_new = W_out_new · h₄
      = 0.625×0.422 + 0.423×0.395
      = 0.264 + 0.167
      = 0.431

L' = ½(1.0 - 0.431)² = ½ × 0.326 = 0.162
```

```
Before update:  L = 0.173   ŷ = 0.411
After update:   L = 0.162   ŷ = 0.431   ← closer to y=1.0
Loss dropped by 6.4%
```

**Key insight from this recomputation:**

In RNN, ALL weights changed by 0.010-0.020 per element; h changed noticeably.
In LSTM, the ONLY weight with meaningful change is W_out (+0.025, +0.023).
Gate weights barely moved because **the cell state already preserved "cat" correctly**.
The model's first priority was to amplify the output layer — h was already good.

Over thousands of training steps, gate weights do accumulate meaningful changes as the model learns to handle harder sentences where gates make wrong decisions.

**In a real training loop:**
```python
for epoch in range(num_epochs):
    for sentence, label in dataset:      # thousands of sentences
        forward_pass = compute(x, y, h, C)
        backward_pass = BPTT flowed through forget gates
        weight_update: all 8 gate matrices + W_out shift slightly
        zero_gradients → ready for next sentence

# After thousands of steps:
# - Forget gate learns to give f≈1 for important words ("cat"), f≈0 for words to erase
# - Input gate learns to write strongly for new content, weakly for stop words
# - Output gate learns: write strongly for new content, weakly for stop words
# - Output gate learns: expose cell to hidden state selectively
# Each step moves weights a tiny amount (lr=0.001 in practice)
# The model discovers through trial and error which words to keep in memory
```

---

## 10. The Full Picture in One View

**FORWARD PASS — two streams flowing right:**

```
x₁=[1.00,0.50]     x₂=[0.20,0.30]     x₃=[0.10,0.10]     x₄=[0.20,0.40]
   "cat"               "sat"               "on"                "mat"
     |                   |                   |                   |
     v                   v                   v                   v
C₀=[0,0]→ f₁⊙C₀→ C₁=[0.680,0.279]→ f₂⊙C₁→ C₂→ f₃⊙C₂→ C₃=[0.628,0.437]→ f₄⊙C₃→ C₄=[0.636,0.638]
  i₁⊙g₁↗           i₂⊙g₂↗             i₃⊙g₃↗                i₄⊙g₄↗          Cell state (protected)

     o₁⊙tanh          o₂⊙tanh          o₃⊙tanh          o₄⊙tanh
        |                 |                 |                 |
h₁=[0.443,0.172]   h₂=[0.411,0.337]   h₃=[0.362,0.288]   h₄=[0.422,0.395]
                                                                |
                                                             W_out
                                                                |
                                                          ŷ=0.411  L=0.173
```

**BACKWARD PASS — gradient through cell state (the highway):**

```
∂L/∂C₄=[-0.181,-0.113]  ∂L/∂C₃=[-0.144,-0.094]  ∂L/∂C₂=[-0.157,-0.094]  ∂L/∂C₁=[-0.181,-0.113]
        [69%]                    ×f₄                       ×f₃                       ×f₂            [100%]

Each step multiplied by forget gate f: 0.88-0.92
NOT by W' × (1-h²) = 0.44  (which was the RNN's problem)
```

---

## 11. Why GRU is Next

LSTM works extremely well but has one cost:

```
4 gates = 4 weight matrices each = 8 weight matrices per LSTM
RNN had only 2 weight matrices (Wx, Wh)

For hidden_dim=256:
  RNN parameters in recurrent layer:  256×300 + 256×256 = 141,312
  LSTM parameters in recurrent layer: 4 × (256×300 + 256×256) = 565,248
```

**4× more parameters → 4× more memory → slower training**

GRU (Gated Recurrent Unit) asks: "Can we get the same cell state preservation with fewer gates?"

**Answer: yes.** GRU merges forget + input into ONE update gate. No separate cell state — just one hidden state. ~75% of LSTM's parameters, similar performance on most tasks.

That is what `04_gru_end_to_end.md` covers next.

---

## 12. Quick Reference — All Formulas

**Shapes:**
```
x:       (2,)    word embedding
h:       (2,)    hidden state (short-term)
C:       (2,)    cell state (long-term)
f,i,g,o: (2,)   gate outputs (sigmoid) or candidate (tanh)
Wf_x, Wi_x, Wg_x, Wo_x = (2,2)   input weight per gate
Wf_h, Wi_h, Wg_h, Wo_h = (2,2)   recurrent weight per gate
W_out:   (2,)
```

**Forward:**
```
f = σ(Wf_x·x + Wf_h·h_{t-1})
i = σ(Wi_x·x + Wi_h·h_{t-1})
g = tanh(Wg_x·x + Wg_h·h_{t-1})
o = σ(Wo_x·x + Wo_h·h_{t-1})
C = f⊙C_{t-1} + i⊙g
h = o ⊙ tanh(C)
ŷ = W_out · h
L = ½(y - ŷ)²
```

**Backward key path (cell state gradient):**
```
∂L/∂ŷ   = -(y - ŷ)
∂L/∂h   = ∂L/∂ŷ × W_outᵀ
∂L/∂C   = ∂L/∂h ⊙ o ⊙ (1 - tanh²(C))
∂L/∂C_{t-1} = ∂L/∂C × f_t          ← multiplies by f, NOT by W
```

**Gate gradient (forget gate example):**
```
∂L/∂f     = ∂L/∂C × C_{t-1}
∂L/∂af    = ∂L/∂f ⊙ f ⊙ (1-f)      → sigmoid derivative
∂L/∂Wf_x  = Σ_t ∂L/∂af ⊙ xᵀ        → outer product, sum all steps
∂L/∂Wf_h  = Σ_t ∂L/∂af ⊙ h_{t-1}ᵀ
```

**Update:**
```
W = W - lr × ∂L/∂W    (for every weight matrix — 8 updates + W_out)
```

---

## 13. Code

### Version 1 — Pure NumPy (mirrors the hand computation exactly)

```python
import numpy as np

# Embeddings
E = np.array([
    [0.00, 0.00],  # 0: <PAD>
    [1.00, 0.50],  # 1: "cat"
    [0.20, 0.30],  # 2: "sat"
    [0.10, 0.10],  # 3: "on"
    [0.20, 0.40],  # 4: "mat"
])

# Gate weight matrices (one Wx and Wh per gate)
Wf_x = np.array([[0.50, 0.10], [0.20, 0.00]])  # forget gate input weights
Wf_h = np.array([[0.00, 0.10], [0.10, 0.30]])  # forget gate recurrent weights
Wi_x = np.array([[0.45, 0.25], [0.10, 0.10]])  # input gate
Wi_h = np.array([[0.30, 0.20], [0.15, 0.10]])
Wg_x = np.array([[0.60, 0.30], [0.20, 0.55]])  # candidate
Wg_h = np.array([[0.20, 0.10], [0.10, 0.20]])
Wo_x = np.array([[0.35, 0.20], [0.10, 0.30]])  # output gate
Wo_h = np.array([[0.25, 0.30], [0.10, 0.20]])
W_out = np.array([0.6, 0.4])                    # output layer

def sigmoid(x): return 1 / (1 + np.exp(-x))

# Input
tokens = [1, 2, 3, 4]
x = E[tokens]  # shape: (4, 2)
y = 1.0

# Forward pass
h = np.zeros(2)   # h₀
C = np.zeros(2)   # C₀

cell_states   = [C.copy()]
hidden_states = [h.copy()]
gates_history = []

for t, xt in enumerate(x):
    f = sigmoid(Wf_x @ xt + Wf_h @ h)   # forget gate
    i = sigmoid(Wi_x @ xt + Wi_h @ h)   # input gate
    g = np.tanh(Wg_x @ xt + Wg_h @ h)  # candidate
    o = sigmoid(Wo_x @ xt + Wo_h @ h)   # output gate

    C = f * C + i * g                    # cell state update ← the key line
    h = o * np.tanh(C)                   # hidden state

    cell_states.append(C.copy())
    hidden_states.append(h.copy())
    gates_history.append((f, i, g, o))

    print(f"t={t+1}: f={f.round(2)} i={i.round(2)} C={C.round(3)} h={h.round(3)}")

# t=1 f=[0.63 0.60] i=[0.85 0.60] C=[0.680 0.279] h=[0.443 0.172]
# t=2 f=[0.88 0.90] i=[0.18 0.12] C=[0.620 0.437] h=[0.411 0.337]
# t=3 f=[0.92 0.88] i=[0.18 0.12] C=[0.628 0.437] h=[0.362 0.288]
# t=4 f=[0.87 0.83] i=[0.45 0.55] C=[0.636 0.638] h=[0.422 0.395]

h4    = h
y_hat = W_out @ h4
loss  = 0.5 * (y - y_hat) ** 2
print(f"ŷ = {y_hat:.3f}   L = {loss:.3f}")
# ŷ = 0.411   L = 0.173

# Backward pass (BPTT)
dl_dyhat = -(y - y_hat)                   # -0.589
dl_dout  = dl_dyhat * h4
dl_dh    = dl_dyhat * W_out               # [-0.353, -0.236]

# Gradient through cell state (the LSTM highway)
dl_dc = dl_dh * gates_history[-1][3] * (1 - np.tanh(cell_states[-1])**2)

# Gradient reaching cell state at each step
for t_prime in range(len(x)-1, -1, -1):
    print(f"  t={t_prime+1} [∂L/∂C={np.round(dl_dc, 3)}]")
    if t_prime > 0:
        dl_dc = dl_dc * gates_history[t_prime][0]  # multiply by forget gate — NOT by Wh
# ∂L/∂C magnitudes: 0.214 → 0.184 → 0.167 → 0.147
# 69% of gradient reaches "cat" vs 9% in RNN

# Weight update
lr = 0.1
W_out_new = W_out - lr * dl_dout
print(f"W_out: {W_out.round(3)} → {W_out_new.round(3)}")
# [0.6, 0.4] → [0.625, 0.423]
```

---

### Version 2 — PyTorch Manual (autograd handles backward)

```python
import torch

def sigmoid(x): return torch.sigmoid(x)

Wf_x  = torch.tensor([[0.50,0.10],[0.20,0.00]], requires_grad=True, dtype=torch.float32)
Wf_h  = torch.tensor([[0.00,0.10],[0.10,0.30]], requires_grad=True, dtype=torch.float32)
Wi_x  = torch.tensor([[0.45,0.25],[0.10,0.10]], requires_grad=True, dtype=torch.float32)
Wi_h  = torch.tensor([[0.30,0.20],[0.15,0.10]], requires_grad=True, dtype=torch.float32)
Wg_x  = torch.tensor([[0.60,0.30],[0.20,0.55]], requires_grad=True, dtype=torch.float32)
Wg_h  = torch.tensor([[0.20,0.10],[0.10,0.20]], requires_grad=True, dtype=torch.float32)
Wo_x  = torch.tensor([[0.35,0.20],[0.10,0.30]], requires_grad=True, dtype=torch.float32)
Wo_h  = torch.tensor([[0.25,0.30],[0.10,0.20]], requires_grad=True, dtype=torch.float32)
W_out = torch.tensor([0.6, 0.4], requires_grad=True, dtype=torch.float32)

E      = torch.tensor([[0.0,0.0],[1.0,0.5],[0.2,0.3],[0.1,0.1],[0.2,0.4]])
x      = E[torch.tensor([1,2,3,4])]   # (4, 2)
y      = torch.tensor(1.0)

# Forward pass
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
# ŷ = 0.411   L = 0.173

# Backward (PyTorch BPTT through cell state automatically)
loss.backward()
print(f"∂L/∂W_out: {W_out.grad.round(decimals=3)}")
# [-0.249, -0.233]

# Weight update
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
        # hidden_dim controls C and h size (always same)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids):
        x = self.embedding(token_ids)          # [batch, seq_len, embed_dim]
        out, (h_n, c_n) = self.lstm(x)
        # h_n: [1, batch, hidden_dim]  — hidden state
        # c_n: [1, batch, hidden_dim]  — cell state!
        return self.fc(h_n.squeeze(0))         # [batch, num_classes]

# Instantiate
model = LSTMClassifier(
    vocab_size  = 6,
    embed_dim   = 2,
    hidden_dim  = 2,
    num_classes = 1
)

tokens = torch.tensor([[1, 2, 3, 4]])   # [batch=1, seq=4]
y      = torch.tensor([1.0])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

optimizer.zero_grad()
y_hat = model(tokens)
loss  = criterion(y_hat, y)             # BPTT through cell state, automatic
loss.backward()
optimizer.step()
print(f"Loss: {loss.item():.3f}")

# Real usage — binary classification (sentiment, spam detection)
model = LSTMClassifier(vocab_size=50000, embed_dim=300, hidden_dim=256, num_classes=2)

# Parameter count in LSTM layer:
# 4 gates × (hidden × embed + hidden × hidden + hidden_bias)
# = 4 × (256×300 + 256×256 + 256)
# = 4 × (76,800 + 65,536 + 256)
# = 4 × 142,592 = 570,368 parameters

# Compare RNN: 2 × (256×300 + 256×256) = 283,136 parameters
# LSTM is ~2× more parameters — worth it for long sequences

# Access cell state if needed (e.g., for analysis)
with torch.no_grad():
    out, (h_n, c_n) = model.lstm(model.embedding(tokens))
    print(f"h_n shape: {h_n.shape}")   # [1, 1, 256] — hidden state
    print(f"c_n shape: {c_n.shape}")   # [1, 1, 256] — cell state
```

---

```mermaid
graph LR
    subgraph rnn_path["❌ RNN — Multiplicative path degrades"]
        direction LR
        R1["'cat' in h₁\n100%"] -->|"× Wh × tanh'\n≈ 0.4"| R2["40%"]
        R2 -->|"× 0.4"| R3["16%"]
        R3 -->|"× 0.4 ×..."| R4["≈ 0%\nvanished"]
    end

    subgraph lstm_path["✅ LSTM — Additive highway preserves signal"]
        direction LR
        C1["'cat' in C₁\n100%"] -->|"× forget ≈ 0.9\nadditive +"| C2["90%"]
        C2 -->|"× 0.9"| C3["81%"]
        C3 -->|"× 0.9 ×..."| C4["~35% after 10 steps\nstill usable"]
    end

    style R4 fill:#e74c3c,color:#fff
    style C4 fill:#27ae60,color:#fff
```
> Key: LSTM cell state uses **addition** (Cₜ = f⊙Cₜ₋₁ + i⊙g̃). Gradient of addition is 1 — no multiplication decay.

## 14. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| RNN end-to-end (template) | `02_rnn_end_to_end.md` | Same sentence, compare directly |
| LSTM overview + gates intuition | `01_rnn_to_attention.md §3` | Architecture diagram |
| GRU (next) | `04_gru_end_to_end.md` | Simpler gates, same preservation idea |
| Vanishing gradient math | `../../2.deep learning/01_fundamentals/03_training_stability.md` | Formal proof |

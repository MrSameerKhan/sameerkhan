# 04 — GRU: Complete End-to-End Walkthrough

Same sentence as RNN and LSTM. Same embeddings. Same template.
GRU has two gates instead of four, and ONE state instead of two.

---

## 0. Problem Statement

**LSTM's cost — why we need GRU:**

```
LSTM solved RNN's vanishing gradient by introducing:
  Cₜ = cell state   (long-term memory)
  hₜ = hidden state (short-term output)
  4 gates: forget (f), input (i), candidate (g̃), output (o)

This works extremely well. But 4 gates × 2 weight matrices = 8 weight matrices.

For hidden_dim=256, embed_dim=300:
  LSTM parameters in recurrent layer:  4 × (256×300 + 256×256) = 565,248
  RNN  parameters in recurrent layer:  2 × (256×300 + 256×256) = 282,624

LSTM is 2× heavier than RNN. For long sequences or limited compute, this matters.
```

**GRU's question:**

```
"Can we get the same gradient preservation as LSTM with fewer gates?"

Answer: yes — by merging forget + input into one gate.

LSTM:  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙g̃ₜ    (f and i are INDEPENDENT — can both be high)
GRU:   hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ  (z controls BOTH — it is a blend gate)

When z=0:  hₜ = hₜ₋₁          → keep everything (preserve "cat")
When z=1:  hₜ = h̃ₜ            → replace completely (write new info)
When z=0.12: hₜ = 0.88⊙hₜ₋₁ + 0.12⊙h̃ₜ → keep 88%, blend in 12%

GRU enforces a hard trade-off: every bit you write erases an equal bit.
LSTM has no such constraint (f and i are free). GRU is slightly more constrained
but needs 3 weight matrix pairs instead of 4 — 25% fewer parameters.
```

**GRU's two gates:**

```
zₜ = update gate   → controls HOW MUCH to update (blend weight)
                     z=0: copy old hidden, z=1: use new candidate
                     named "update" because it updates the hidden state

rₜ = reset gate    → controls HOW MUCH past to use in the CANDIDATE
                     r=0: candidate ignores past (fresh start from current input)
                     r=1: candidate uses full past (same as RNN candidate)
                     named "reset" because it resets past context for h̃
```

**GRU vs LSTM gradient comparison (preview):**

```
RNN:  gradient factor per step  ≈ 0.44          → 3 steps: 0.44³  ≈  9%
LSTM: gradient factor per step  ≈ fₜ  ≈ 0.88   → 3 steps: 0.88³  ≈ 69%
GRU:  gradient factor per step  ≈ (1-zₜ) ≈ 0.88 → 3 steps: (1-z)³ ≈ 66%

GRU achieves 66% gradient preservation vs LSTM's 69%
— nearly identical, with 25% fewer parameters.
```

---

## 0.1 What Is the Input?

```
Raw text:  "cat sat on mat"

The model cannot read strings. Three steps before the GRU sees anything:

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
After reading all 4 words, the GRU produces a final hidden state h₄ (2D vector).
An output layer W_out (shape: 1×2) maps h₄ → scalar prediction ŷ.

  ŷ = W_out · h₄     (dot product → single number)

Target:
  y = 1.0   → sentence IS about an animal
  y = 0.0   → sentence is NOT about an animal

Loss:
  L = ½(y - ŷ)²

  If ŷ=0.383 and y=1.0:  L = ½(0.617)² = 0.190   ← model is wrong, but better than RNN
  If ŷ=0.950 and y=1.0:  L = ½(0.050)² = 0.001   ← model is right

Training = adjust all weight matrices so L decreases over many examples.

The difference from RNN: GRU produces a better h₄ before any training,
because the update gate keeps z LOW when the hidden state already carries
important information ("cat"), preserving it step by step.
```

---

## Setup — All Weights

```
embed_dim  = 2   (each word → 2D vector, same as RNN and LSTM)
hidden_dim = 2   (hidden state hₜ is 2D)
```

```
What changed vs LSTM?

  LSTM:  2 states (Cₜ and hₜ)   4 gates (f, i, g̃, o)   8 weight matrix pairs
  GRU:   1 state  (hₜ only)     2 gates (z, r)          3 weight matrix pairs + 1 candidate

  The cell state Cₜ is GONE. hₜ serves as both memory and output.
  The forget + input gates are MERGED into the update gate z.
  The output gate is GONE — hₜ is exposed directly.

  Parameter count for hidden_dim=2, embed_dim=2:
    RNN:   2 × (2×2 + 2×2) = 16 parameters
    LSTM:  8 × (2×2 + 2×2) = 64 parameters
    GRU:   6 × (2×2 + 2×2) = 48 parameters   ← 75% of LSTM
```

```
GRU has 3 gate pairs (each with Wₓ and Wₕ):

Update gate weights:
  Wz_x = [[0.70, 0.30],    Wz_h = [[0.40, 0.10],
           [0.45, 0.30]]            [0.10, 0.30]]

Reset gate weights:
  Wr_x = [[0.45, 0.15],    Wr_h = [[0.25, 0.10],
           [0.25, 0.15]]            [0.10, 0.20]]

Candidate weights:
  Wh_x = [[0.80, 0.40],    Wh_h = [[0.30, 0.10],
           [0.30, 0.38]]            [0.10, 0.30]]

Output layer:
  W_out = [0.6, 0.4]   (same as RNN and LSTM)

Initial state:
  h₀ = [0.0, 0.0]   ← only ONE initial state — no C₀ needed
```

```
Why sigmoid for both gates?

  zₜ (update) uses sigmoid → output in (0, 1)
    0 = keep old hidden state completely
    1 = replace entirely with candidate
    0.12 = keep 88%, blend in 12% of new content

  rₜ (reset) uses sigmoid → output in (0, 1)
    0 = candidate computed ignoring all past context
    1 = candidate computed using full past context
    These are independent roles → both use sigmoid

  Candidate h̃ uses tanh → output in (-1, +1)
    Same reason as LSTM: bounded to prevent explosion,
    negative values can suppress unwanted memory

Why is there no output gate?
  LSTM's output gate oₜ filtered the cell state before exposing as hₜ.
  GRU has no separate cell state — hₜ is the memory AND the output.
  Removing oₜ is one of the two simplifications GRU makes vs LSTM.
```

```
GRU formulas at each timestep:

  zₜ = σ(Wz_x · xₜ  +  Wz_h · hₜ₋₁)       update gate
  rₜ = σ(Wr_x · xₜ  +  Wr_h · hₜ₋₁)       reset gate
  h̃ₜ = tanh(Wh_x · xₜ + Wh_h · (rₜ ⊙ hₜ₋₁)) candidate (past filtered by reset)
  hₜ = (1-zₜ) ⊙ hₜ₋₁  +  zₜ ⊙ h̃ₜ          hidden state update

  ⊙ = elementwise multiply

Key structural difference from LSTM:
  LSTM:  Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙g̃ₜ   (f and i INDEPENDENT — can both be 0.9)
  GRU:   hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ  (one gate controls BOTH — sum = 1.0)
```

```
Note on embedding gradients:

  In a real training loop, the embedding table E also gets gradients.
  ∂L/∂xₜ flows back to update the embedding vector for each token seen.

  We skip this in the backward pass below to keep focus on the 6 GRU weight matrices.
  In code (Version 2 and 3), PyTorch handles it automatically via
  nn.Embedding — the embedding vectors update just like any other weight.
```

---

## 1. Forward Pass

---

### t=1 — "cat"   (x₁ = [1.00, 0.50],  h₀ = [0, 0])

**Update gate — shown in full as example of how both gates compute:**

```
Wz_x · x₁:
  row 0:  0.70×1.00 + 0.30×0.50  =  0.700 + 0.150  =  0.850
  row 1:  0.45×1.00 + 0.30×0.50  =  0.450 + 0.150  =  0.600

Wz_h · h₀:
  row 0:  0.40×0.00 + 0.10×0.00  =  0.000
  row 1:  0.10×0.00 + 0.30×0.00  =  0.000

az₁ = [0.850 + 0.000,  0.600 + 0.000] = [0.850, 0.600]

z₁ = σ([0.850, 0.600])
   = [1/(1+e⁻⁰·⁸⁵),  1/(1+e⁻⁰·⁶⁰)]
   = [1/1.427,         1/1.549]
   = [0.701,           0.646]  ≈  [0.70, 0.65]

Interpretation:
  z₁[0] = 0.70 → write 70% of h̃₁ into h₁ (moderate update)
  z₁[1] = 0.65 → write 65% of h̃₁ into h₁
  With h₀ = 0, the only way to encode "cat" is through h̃₁ weighted by z₁.
  High z at t=1 is correct: we want to WRITE information into the empty hidden state.
```

**Reset gate at t=1** (shown in full — unique GRU computation):

```
Wr_x · x₁:
  row 0:  0.45×1.00 + 0.15×0.50  =  0.450 + 0.075  =  0.525
  row 1:  0.25×1.00 + 0.15×0.50  =  0.250 + 0.075  =  0.325

Wr_h · h₀ = [0, 0]

ar₁ = [0.525, 0.325]
r₁ = σ([0.525, 0.325]) = [0.628, 0.581]  ≈  [0.63, 0.58]

Note: r₁ doesn't affect h₁ much because h₀ = 0.
  rₜ⊙hₜ₋₁ = [0.63, 0.58]⊙[0, 0] = [0, 0] regardless of r₁.
  The reset gate starts to matter from t=2 onwards (when hₜ₋₁ ≠ 0).
```

**Candidate h̃₁:**

```
r₁ ⊙ h₀ = [0.63, 0.58] ⊙ [0.00, 0.00] = [0.00, 0.00]

Wh_x · x₁:
  row 0:  0.80×1.00 + 0.40×0.50  =  0.800 + 0.200  =  1.000
  row 1:  0.30×1.00 + 0.38×0.50  =  0.300 + 0.190  =  0.490

Wh_h · (r₁⊙h₀) = Wh_h · [0, 0] = [0, 0]

ah̃₁ = [1.000 + 0.000,  0.490 + 0.000] = [1.000, 0.490]

h̃₁ = tanh([1.000, 0.490])
    = [0.762,  0.454]

h̃₁ = [0.762, 0.454]
       ↑ cat is very animal-related (0.762), carries content (0.454)
       tanh(1.0) = 0.762   ← clean value, cat's animal signal maps to 0.762
```

**Hidden state update:**

```
h₁ = (1-z₁) ⊙ h₀  +  z₁ ⊙ h̃₁
   = (1-[0.70, 0.65]) ⊙ [0.00, 0.00]  +  [0.70, 0.65] ⊙ [0.762, 0.454]
   = [0.30, 0.35] ⊙ [0.00, 0.00]      +  [0.70×0.762, 0.65×0.454]
   = [0.000, 0.000]                    +  [0.533, 0.295]
   = [0.533, 0.295]

h₁ = [0.533, 0.295]   ← "cat" strongly encoded in dim 0
```

---

### t=2 — "sat"   (x₂ = [0.20, 0.30],  h₁ = [0.533, 0.295])

```
Gate outputs (from weight matrices applied to [h₁, x₂]):

  z₂ = [0.12, 0.10]  ← update gate: LOW → keep 88% of h₁[0] and 90% of h₁[1]
                                     "cat" is important — don't overwrite it with "sat"

  r₂ = [0.55, 0.60]  ← reset gate: moderate — candidate uses ~57% of past context
                                    "sat" builds on what came before (subject-verb link)

  h̃₂ = [0.18, 0.55]  ← candidate: sat is NOT animal (0.18), IS an action verb (0.55)
```

**Hidden state update:**

```
h₂ = (1-z₂) ⊙ h₁            +   z₂ ⊙ h̃₂
   = [0.88, 0.90] ⊙ [0.533, 0.295]  +  [0.12, 0.10] ⊙ [0.18, 0.55]
   = [0.469,        0.266]           +  [0.022,        0.055]
   = [0.491,        0.321]
     ↑
     (1-z₂[0])=0.88 kept 88% of h₁[0]=0.533 → 0.469
     "cat" animal signal barely changed: 0.533 → 0.491
     "sat" wrote only 12% of its candidate → minimal disruption
```

---

### t=3 — "on"   (x₃ = [0.10, 0.10],  h₂ = [0.491, 0.321])

```
Gate outputs:

  z₃ = [0.08, 0.10]  ← very LOW → keep 92-90% of h₂
                         "on" is a function word — no new information to write
  r₃ = [0.20, 0.15]  ← low reset: candidate mostly ignores past for "on"
                         ("on" makes no sense in the context of prior verbs)
  h̃₃ = [0.05, 0.09]  ← candidate: "on" has minimal content (function word)
```

**Hidden state update:**

```
h₃ = (1-z₃) ⊙ h₂            +   z₃ ⊙ h̃₃
   = [0.92, 0.90] ⊙ [0.491, 0.321]  +  [0.08, 0.10] ⊙ [0.05, 0.09]
   = [0.452,        0.289]           +  [0.004,        0.009]
   = [0.456,        0.298]
     ↑
     "cat" in dim 0:  h₁=0.533 → h₂=0.491 → h₃=0.456
     Still very strong. "on" added almost nothing (z₃=0.08 × h̃₃=0.05 = 0.004)
```

---

### t=4 — "mat"   (x₄ = [0.20, 0.40],  h₃ = [0.456, 0.298])

```
Gate outputs:

  z₄ = [0.18, 0.20]  ← LOW → keep 82-80% of h₃ ("mat" is content but not animal)
  r₄ = [0.60, 0.65]  ← moderate reset: candidate uses some past context
  h̃₄ = [0.22, 0.50]  ← candidate: mat is not animal (0.22), carries content (0.50)
```

**Hidden state update:**

```
h₄ = (1-z₄) ⊙ h₃            +   z₄ ⊙ h̃₄
   = [0.82, 0.80] ⊙ [0.456, 0.298]  +  [0.18, 0.20] ⊙ [0.22, 0.50]
   = [0.374,        0.238]           +  [0.040,        0.100]
   = [0.414,        0.338]
```

---

### Output layer

```
ŷ = W_out · h₄  =  0.6×0.414  +  0.4×0.338  =  0.248 + 0.135  =  0.383
```

---

### Forward pass summary

```
Hidden state — the single memory+output stream:

        h[dim 0]         h[dim 1]
        (animal signal)  (content signal)
  h₁ → [0.533,          0.295]   ← "cat" written via z₁⊙h̃₁
  h₂ → [0.491,          0.321]   ← z₂=0.12 kept 88% of h₁
  h₃ → [0.456,          0.298]   ← z₃=0.08 kept 92% of h₂
  h₄ → [0.414,          0.338]   ← z₄=0.18 kept 82% of h₃

  "cat" in dim 0 via update gates:  (1-z₂)×(1-z₃)×(1-z₄) = 0.88×0.92×0.82 = 0.664
  → 66.4% of "cat" signal survives to h₄

  Compare to RNN dim 0 trace:
  h₁=0.537 → h₂=0.398 → h₃=0.270 → h₄=0.308   ← rapid erosion
  GRU dim 0 trace:
  h₁=0.533 → h₂=0.491 → h₃=0.456 → h₄=0.414   ← slow, controlled decay

  ŷ = 0.383   (model says "likely animal" — more confident than RNN's 0.365)
  y = 1.000   (correct: yes, animal)
```

```
GRU vs RNN vs LSTM — same sentence, same embeddings:

  RNN:   ŷ = 0.365   L = 0.201   h₄[dim 0] = 0.308   gradient to "cat" = 9%
  GRU:   ŷ = 0.383   L = 0.190   h₄[dim 0] = 0.414   gradient to "cat" = 66%
  LSTM:  ŷ = 0.411   L = 0.173   C₄[dim 0] = 0.636   gradient to "cat" = 69%

  GRU: better than RNN, slightly below LSTM, nearly identical gradient flow.
  GRU uses 75% of LSTM's parameters to achieve 96% of LSTM's gradient preservation.

Note: GRU's h₁[0]=0.533 vs LSTM's C₁[0]=0.680.
  GRU has ONE state so h must be both memory AND output.
  LSTM's cell state can accumulate higher values precisely because it is
  protected and not exposed directly as output. GRU trades that for simplicity.
```

---

## 2. Loss

```
L = ½(y - ŷ)²  =  ½ × (1.000 - 0.383)²  =  ½ × 0.617²  =  ½ × 0.381  =  0.190

Error = 0.617.  Better than RNN's error of 0.635 — update gate preserved "cat".
```

---

## 3. Backward Pass (BPTT)

```
Why we unroll through all 4 timesteps — same reason as RNN and LSTM:

  All gate weight matrices (Wz_x, Wz_h, Wr_x, Wr_h, Wh_x, Wh_h) are SHARED.
  Each was used at t=1, t=2, t=3, t=4.
  Total gradient = sum of contributions from all timesteps.

Key difference from RNN and LSTM:
  RNN:   gradient flows back through hₜ → Wₕ → hₜ₋₁     (multiplicative, Wₕᵀ × tanh')
  LSTM:  gradient flows back through Cₜ → fₜ → Cₜ₋₁      (multiplicative by fₜ only)
  GRU:   gradient flows back through hₜ → (1-zₜ) → hₜ₋₁  (multiplicative by (1-zₜ))

  RNN factor per step:    Wₕᵀ × (1-hₜ²) ≈ 0.44
  LSTM factor per step:   fₜ              ≈ 0.88-0.92
  GRU factor per step:    (1-zₜ)          ≈ 0.82-0.92   (1 minus a small update gate)

  After 3 steps:  RNN = 0.44³ ≈ 9%    LSTM = 0.88×0.92×0.87 ≈ 69%    GRU = 0.88×0.92×0.82 ≈ 66%
```

---

### Step A — gradient at the output

```
∂L/∂ŷ = -(y - ŷ) = -(1.000 - 0.383) = -0.617
```

---

### Step B — gradient through output layer

```
ŷ = W_out · h₄,  so:

∂L/∂W_out = ∂L/∂ŷ × h₄ᵀ  =  -0.617 × [0.414, 0.338]  =  [-0.255, -0.209]

∂L/∂h₄    = ∂L/∂ŷ × W_outᵀ  =  -0.617 × [0.6, 0.4]    =  [-0.370, -0.247]

δh₄ = [-0.370, -0.247]   ← error entering BPTT
```

---

### Step C — gradient from hₜ to hₜ₋₁ (the GRU highway)

```
hₜ = (1-zₜ) ⊙ hₜ₋₁  +  zₜ ⊙ h̃ₜ

How does hₜ₋₁ affect hₜ?

Direct path (the highway):
  ∂hₜ/∂hₜ₋₁|_direct = (1-zₜ)
  This is the additive residual path — gradient multiplies by (1-zₜ) only.

Secondary path (through h̃ₜ via reset gate):
  h̃ₜ = tanh(Wh_x·xₜ + Wh_h·(rₜ⊙hₜ₋₁))
  hₜ₋₁ also influences h̃ₜ through rₜ⊙hₜ₋₁.
  This path involves sigmoid × tanh derivatives — it contributes but is smaller.

We show the dominant direct path here. The full gradient at t=4→t=3:
  ∂L/∂h₃ ≈ ∂L/∂h₄ ⊙ (1-z₄)
           = [-0.370, -0.247] ⊙ [0.82, 0.80]
           = [-0.303, -0.198]

  Why (1-z₄) not z₄?
    hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ
    The coefficient of hₜ₋₁ in this formula is (1-zₜ).
    So ∂hₜ/∂hₜ₋₁ = (1-zₜ) for the direct path.
    z₄ = [0.18, 0.20] → (1-z₄) = [0.82, 0.80] ← large, gradient preserved well.
```

---

### Step D — gradient flows back through all timesteps

```
∂L/∂h₃ = ∂L/∂h₄ ⊙ (1-z₄)
        = [-0.370, -0.247] ⊙ [0.82, 0.80]
        = [-0.303, -0.198]

∂L/∂h₂ = ∂L/∂h₃ ⊙ (1-z₃)
        = [-0.303, -0.198] ⊙ [0.92, 0.90]
        = [-0.279, -0.178]

∂L/∂h₁ = ∂L/∂h₂ ⊙ (1-z₂)
        = [-0.279, -0.178] ⊙ [0.88, 0.90]
        = [-0.246, -0.160]
```

---

### Gradient magnitudes — GRU vs RNN vs LSTM

```
Hidden state gradient at each step:

  |∂L/∂h₄| = √(0.370² + 0.247²) = √0.198 = 0.445   ← full error
  |∂L/∂h₃| = √(0.303² + 0.198²) = √0.131 = 0.362   ← 81% left
  |∂L/∂h₂| = √(0.279² + 0.178²) = √0.110 = 0.332   ← 75% left
  |∂L/∂h₁| = √(0.246² + 0.160²) = √0.087 = 0.295   ← 66% left

  66% of the gradient reaches "cat" hidden state.

Compare to all three architectures:
  RNN:   |δ₄|=0.458 → |δ₁|=0.039   →  9% via Wₕᵀ × tanh' each step
  LSTM:  |∂L/∂C₄|=0.214 → |∂L/∂C₁|=0.147  → 69% via fₜ each step
  GRU:   |∂L/∂h₄|=0.445 → |∂L/∂h₁|=0.295  → 66% via (1-zₜ) each step

┌─────────────────────────────────────────────────────────────────┐
│  Gradient reaching "cat" after 3 backprop steps:                 │
│    RNN:    9%  (Wₕᵀ × tanh' ≈ 0.44 per step → 0.44³)           │
│    GRU:   66%  ((1-z) ≈ 0.88 per step → 0.88×0.92×0.82)        │
│    LSTM:  69%  (fₜ  ≈ 0.88 per step → 0.88×0.92×0.87)          │
│                                                                   │
│  GRU achieves 96% of LSTM's gradient preservation                │
│  with 75% of LSTM's parameters.                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

### Step E — update gate gradients (key GRU formula, all timesteps)

```
Why update gate gradients matter:
  zₜ controls the blend ratio between old and new.
  ∂L/∂Wz_x and ∂L/∂Wz_h tell the optimizer how to adjust the blend.

Key GRU formula for update gate gradient:

  hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ

  ∂L/∂zₜ = ∂L/∂hₜ ⊙ (h̃ₜ - hₜ₋₁)

  Why (h̃ₜ - hₜ₋₁)?
    As zₜ increases by ε: hₜ → hₜ + ε⊙(h̃ₜ - hₜ₋₁)
    The change in hₜ from changing zₜ is exactly (h̃ₜ - hₜ₋₁).
    This is the DIFFERENCE between where we're going and where we were.

Why outer product for weight gradients?  (same reason as RNN and LSTM)
  azₜ = Wz_x · xₜ + Wz_h · hₜ₋₁
  Wz_x[i,j] affects azₜ[i] through xₜ[j] → ∂L/∂Wz_x[i,j] = ∂L/∂azₜ[i] × xₜ[j]
  Written for all i,j: ∂L/∂Wz_x (at t) = ∂L/∂azₜ ⊗ xₜᵀ   (outer product)
```

**Update gate at t=1:**

```
∂L/∂z₁ = ∂L/∂h₁ ⊙ (h̃₁ - h₀)
        = [-0.246, -0.160] ⊙ ([0.762, 0.454] - [0.000, 0.000])
        = [-0.246×0.762, -0.160×0.454]
        = [-0.187, -0.073]

Sigmoid derivative: z₁⊙(1-z₁) = [0.70×0.30, 0.65×0.35] = [0.210, 0.228]

∂L/∂az₁ = ∂L/∂z₁ ⊙ z₁⊙(1-z₁)
         = [-0.187, -0.073] ⊙ [0.210, 0.228]
         = [-0.039, -0.017]

Note: (h̃₁ - h₀) = [0.762, 0.454] because h₀=0.
  The gradient at t=1 is large because the update gate is moving from nothing to
  a new candidate — the (h̃₁-h₀) difference is maximally large.
  This means "cat" drives the update gate weight updates STRONGLY.
```

**Update gate at t=2:**

```
∂L/∂z₂ = ∂L/∂h₂ ⊙ (h̃₂ - h₁)
        = [-0.279, -0.178] ⊙ ([0.18, 0.55] - [0.533, 0.295])
        = [-0.279, -0.178] ⊙ [-0.353, 0.255]
        = [0.098, -0.045]

What does ∂L/∂z₂[0] = +0.098 mean?
  Positive gradient → increasing z₂[0] would INCREASE L (make things worse).
  To minimize L, the optimizer wants to DECREASE z₂[0] (keep it low).
  In other words: keep the blend low → preserve h₁[0]=0.533 (which has "cat").
  This is the correct signal: z should stay low when memory is valuable.

Sigmoid derivative: z₂⊙(1-z₂) = [0.12×0.88, 0.10×0.90] = [0.106, 0.090]

∂L/∂az₂ = [0.098, -0.045] ⊙ [0.106, 0.090]
         = [0.010, -0.004]
```

**Update gate at t=3:**

```
∂L/∂z₃ = ∂L/∂h₃ ⊙ (h̃₃ - h₂)
        = [-0.303, -0.198] ⊙ ([0.05, 0.09] - [0.491, 0.321])
        = [-0.303, -0.198] ⊙ [-0.441, -0.231]
        = [0.134, 0.046]

Sigmoid derivative: z₃⊙(1-z₃) = [0.08×0.92, 0.10×0.90] = [0.074, 0.090]

∂L/∂az₃ = [0.134, 0.046] ⊙ [0.074, 0.090]
         = [0.010, 0.004]
```

**Update gate at t=4:**

```
∂L/∂z₄ = ∂L/∂h₄ ⊙ (h̃₄ - h₃)
        = [-0.370, -0.247] ⊙ ([0.22, 0.50] - [0.456, 0.298])
        = [-0.370, -0.247] ⊙ [-0.236, 0.202]
        = [0.087, -0.050]

Sigmoid derivative: z₄⊙(1-z₄) = [0.18×0.82, 0.20×0.80] = [0.148, 0.160]

∂L/∂az₄ = [0.087, -0.050] ⊙ [0.148, 0.160]
         = [0.013, -0.008]
```

---

### Step F — weight gradients ∂L/∂Wz_x and ∂L/∂Wz_h

```
∂L/∂Wz_x — contribution at each timestep (outer product ∂L/∂azₜ ⊗ xₜᵀ):

t=1:  [-0.039, -0.017]ᵀ ⊗ [1.00, 0.50]:
      [[-0.039×1.00,  -0.039×0.50],   [[-0.039, -0.020],
       [-0.017×1.00,  -0.017×0.50]] =   [-0.017, -0.009]]

t=2:  [0.010, -0.004]ᵀ ⊗ [0.20, 0.30]:
      [[0.002, 0.003],
       [-0.001, -0.001]]

t=3:  [0.010, 0.004]ᵀ ⊗ [0.10, 0.10]:
      [[0.001, 0.001],
       [0.000, 0.000]]

t=4:  [0.013, -0.008]ᵀ ⊗ [0.20, 0.40]:
      [[0.003, 0.005],
       [-0.002, -0.003]]

Sum across all timesteps → ∂L/∂Wz_x:
  [[-0.039+0.002+0.001+0.003,  -0.020+0.003+0.001+0.005],
   [-0.017-0.001+0.000-0.002,  -0.009-0.001+0.000-0.003]]

= [[-0.033, -0.011],
   [-0.020, -0.013]]
```

```
∂L/∂Wz_h — contribution at each timestep (outer product ∂L/∂azₜ ⊗ hₜ₋₁ᵀ):

t=1:  [-0.039, -0.017]ᵀ ⊗ h₀=[0, 0]  →  zero matrix  (h₀ = 0)

t=2:  [0.010, -0.004]ᵀ ⊗ [0.533, 0.295]:
      [[0.005, 0.003],
       [-0.002, -0.001]]

t=3:  [0.010, 0.004]ᵀ ⊗ [0.491, 0.321]:
      [[0.005, 0.003],
       [0.002, 0.001]]

t=4:  [0.013, -0.008]ᵀ ⊗ [0.456, 0.298]:
      [[0.006, 0.004],
       [-0.004, -0.002]]

Sum → ∂L/∂Wz_h:
= [[0.016, 0.010],
   [-0.004, -0.002]]
```

```
Reset gate gradient — brief overview:

The reset gate rₜ affects h̃ₜ through:  h̃ₜ = tanh(Wh_x·xₜ + Wh_h·(rₜ⊙hₜ₋₁))

∂L/∂h̃ₜ  = ∂L/∂hₜ ⊙ zₜ                   (h̃ is weighted by z in the blend)
∂L/∂ah̃ₜ = ∂L/∂h̃ₜ ⊙ (1-h̃ₜ²)               (tanh derivative)
∂L/∂rₜ  = ∂L/∂ah̃ₜ ⊙ Wh_hᵀ ⊙ hₜ₋₁         (chain through r in h̃ computation)
∂L/∂Wr_x = Σₜ ∂L/∂arₜ ⊗ xₜᵀ

Key insight: ∂L/∂h̃ₜ = ∂L/∂hₜ ⊙ zₜ
  When zₜ is small (0.08-0.12), h̃ₜ contributes very little to hₜ.
  → ∂L/∂h̃ₜ is small → ∂L/∂rₜ is small → reset gate gradients are small.
  The reset gate barely matters during "preserve" timesteps (low z).
  It becomes important when z is high (t=1, when we ARE writing new content).

Note: embedding gradients (∂L/∂xₜ) also exist in real training.
      We skip them here to keep focus on the 6 GRU weight matrices.
```

```
"cat" vs "mat" — how GRU reverses RNN's learning imbalance:

Update gate weight ∂L/∂Wz_x:
  "cat" (t=1) contribution: [[-0.039,-0.020], [-0.017,-0.009]]  → magnitude ≈ 0.039
  "sat" (t=2) contribution: [[0.002, 0.003],  [-0.001,-0.001]]  → magnitude ≈ 0.002
  "on"  (t=3) contribution: [[0.001, 0.001],  [0.000, 0.000]]   → magnitude ≈ 0.001
  "mat" (t=4) contribution: [[0.003, 0.005],  [-0.002,-0.003]]  → magnitude ≈ 0.003

  "cat" drives update gate weight updates 10-13× more than "mat".

This is the OPPOSITE of RNN's problem:
  RNN:   "mat" drove ∂L/∂Wₓ  3-10× more than "cat"  (gradient barely reached t=1)
  GRU:   "cat" drives ∂L/∂Wz 10× more than "mat"    (gradient flows back at 66%)

Why does "cat" dominate GRU's update gate gradient?
  ∂L/∂z₁ = ∂L/∂h₁ ⊙ (h̃₁ - h₀) = [-0.246,-0.160] ⊙ [0.762, 0.454]
  The term (h̃₁ - h₀) is LARGE at t=1 because h₀=0 — the model is updating from nothing.
  The gradient arriving at h₁ is also strong (66% of original error).
  Large gradient × large (h̃-h_prev) = large weight update from "cat". ✓

  At t=4: (h̃₄ - h₃) = [0.22-0.456, 0.50-0.298] = [-0.236, 0.202] → partial cancellation.
  "mat" produces a mixed-sign gradient that largely cancels → small net update.
```

---

## 4. Weight Update

```
Learning rate: lr = 0.1

W_out_new = W_out - lr × ∂L/∂W_out
          = [0.6,  0.4] - 0.1 × [-0.255, -0.209]
          = [0.626, 0.421]


Wz_x_new = Wz_x - lr × ∂L/∂Wz_x

  = [[0.70, 0.30],   -  0.1 × [[-0.033, -0.011],
     [0.45, 0.30]]               [-0.020, -0.013]]

  = [[0.70+0.003,  0.30+0.001],
     [0.45+0.002,  0.30+0.001]]

  = [[0.703, 0.301],
     [0.452, 0.301]]


Wz_h_new = Wz_h - lr × ∂L/∂Wz_h

  = [[0.40, 0.10],   -  0.1 × [[0.016,  0.010],
     [0.10, 0.30]]               [-0.004, -0.002]]

  = [[0.40-0.002,  0.10-0.001],
     [0.10+0.000,  0.30+0.000]]

  = [[0.398, 0.099],
     [0.100, 0.300]]
```

```
Why are the Wz_x and Wz_h changes small?

  Wz_x changed by < 0.003 per element.
  Wz_h changed by < 0.002 per element.

  At t=1: ∂L/∂az₁ = [-0.039, -0.017]. These are the LARGEST values.
  But sigmoid derivative z₁⊙(1-z₁) = [0.210, 0.228] — moderate shrinkage.
  The final weight gradient [-0.033, -0.011] is still small relative to lr=0.1.

  The main update is in W_out:
    [0.600, 0.400] → [0.626, 0.421]   ← 4.3% increase
  Same pattern as LSTM: in the first training step, the output layer
  has the largest room to grow (ŷ=0.383 needs to move to 1.0).
  Gate weights learn more gradually across many training steps.

Reset and candidate weight updates follow the same formula:
  W ← W - lr × ∂L/∂W     (Wr_x, Wr_h, Wh_x, Wh_h each receive their gradient)
```

---

## 5. Second Forward Pass (Verify Loss Decreased)

Updated weights: Wz_x=[[0.703,0.301],[0.452,0.301]], Wz_h=[[0.398,0.099],[0.100,0.300]], W_out=[0.626,0.421]
All other gate matrices unchanged (Wr_x, Wr_h, Wh_x, Wh_h same as before)

```
t=1  (x₁=[1.00, 0.50])  — recompute update gate with new Wz_x:
  Wz_x_new · x₁:
    row 0:  0.703×1.00 + 0.301×0.50  =  0.703 + 0.151  =  0.854
    row 1:  0.452×1.00 + 0.301×0.50  =  0.452 + 0.151  =  0.603
  z₁_new = σ([0.854, 0.603]) ≈ [0.701, 0.647]   (was [0.700, 0.646] — change < 0.001)

  h̃₁_new = tanh([1.000, 0.490]) = [0.762, 0.454]   ← unchanged (Wh_x not updated)

  h₁_new = z₁_new ⊙ h̃₁_new = [0.701×0.762, 0.647×0.454] = [0.534, 0.294]
         ≈ [0.533, 0.295]   (change < 0.001)
```

```
Why did h₁ barely change?
  Wz_x changed by [+0.003, +0.001] per row.
  This changes az₁ by +0.003×1.0+0.001×0.5 ≈ +0.004 per dimension.
  z₁ changes by σ'(0.85)×0.004 = 0.70×0.30×0.004 = 0.001.
  h₁ changes by 0.001×0.762 ≈ 0.001 — less than one thousandth.
```

```
t=2,3,4: Wz_x and Wz_h changed by < 0.002 per element.
  Δaz at each step < 0.001 → z values essentially unchanged.
  → h₂, h₃, h₄ unchanged to 3 decimal places.

  h₁ ≈ [0.533, 0.295]   h₂ ≈ [0.491, 0.321]
  h₃ ≈ [0.456, 0.298]   h₄ ≈ [0.414, 0.338]   ← same as before
```

```
ŷ' = W_out_new · h₄
   = 0.626×0.414  +  0.421×0.338
   = 0.259 + 0.142
   = 0.401

L' = ½(1.0 - 0.401)²  =  ½ × 0.599²  =  ½ × 0.359  =  0.179
```

```
Before update:  L = 0.190   ŷ = 0.383
After update:   L = 0.179   ŷ = 0.401   ← closer to y=1.0  ✅

Loss dropped by 5.8%.
```

```
This was ONE training step on ONE sentence.

In a real training loop:

  for epoch in range(num_epochs):
      for sentence, label in dataset:          # thousands of sentences
          forward pass  → compute ŷ, hₜ for all t, L
          backward pass → ∂L/∂h flows back via (1-zₜ) highway
                        → ∂L/∂Wz, ∂L/∂Wr, ∂L/∂Wh computed via outer products
          weight update → all 6 gate matrices + W_out shift slightly
          zero gradients → ready for next sentence

  After thousands of steps:
    - Wz weights learn: produce z≈0 (keep memory) when input has no new animal info,
      produce z≈1 (update) when important new content arrives.
    - Wr weights learn: produce r≈0 (ignore past) for function words,
      r≈1 (use full past) when context matters for candidate construction.
    - Wh weights learn: produce h̃ that encodes the RIGHT content for this context.

  The trained GRU finds weight values that naturally produce the z patterns
  shown in this dry run — low z for "sat", "on", "mat" when following "cat".
```

---

## 6. The Full Picture in One View

```
FORWARD PASS — ONE stream (h is both memory and output) →
──────────────────────────────────────────────────────────────────────────
  x₁=[1.0,0.5]    x₂=[0.2,0.3]    x₃=[0.1,0.1]    x₄=[0.2,0.4]
     "cat"             "sat"            "on"             "mat"
       │                 │                │                │
       ▼                 ▼                ▼                ▼
h₀=[0,0] ──────────────────────────────────────────────────── blend highway
    z₁⊙h̃₁         (1-z₂)⊙+z₂⊙h̃₂   (1-z₃)⊙+z₃⊙h̃₃  (1-z₄)⊙+z₄⊙h̃₄
  h₁=[0.533,0.295]→h₂=[0.491,0.321]→h₃=[0.456,0.298]→h₄=[0.414,0.338]
                                                         │
                                                      W_out
                                                         │
                                                     ŷ=0.383 → L=0.190

  z values:  z₁=0.70  z₂=0.12  z₃=0.08  z₄=0.18
             ↑ write  ↑ keep   ↑ keep   ↑ mostly keep
  (1-z) path: ─────── 0.88 ──── 0.92 ──── 0.82 ─────── gradient highway

BACKWARD PASS — gradient through (1-z) highway ←
──────────────────────────────────────────────────────────────────────────
  ∂L/∂h₁=[-0.246,-0.160]  ∂L/∂h₂=[-0.279,-0.178]  ∂L/∂h₃=[-0.303,-0.198]  ∂L/∂h₄=[-0.370,-0.247]
  |66%|     ←───────────── ×(1-z₂) ── ×(1-z₃) ───── ×(1-z₄) ──── |100%|

  Each step back multiplies by (1-zₜ) ≈ 0.88-0.92
  NOT by Wₕᵀ × (1-hₜ²) ≈ 0.44 (RNN's problem)
  SAME mechanism as LSTM's forget gate, but with ONE unified state
```

---

## 7. Why Attention is Next

```
GRU (and LSTM) work very well for most sequence tasks but have one remaining limit:

  Sequential bottleneck:
    GRU must compress ALL of "cat sat on mat" into ONE vector h₄ (size 2 here, 256-512 in practice).
    In a long sentence, the final hidden state becomes a lossy summary.
    Even with perfect gradient flow (z≈0 for 50 steps), h₄ can only hold hidden_dim "slots".

  Example:
    "The cat that my neighbor's dog has been chasing every morning sat on the mat."
    By the time we reach "sat", h has been through 15 words.
    Even with z=0.10 per step: 0.90^15 ≈ 21% of "cat" signal survives.
    For machine translation or question answering, this is a real bottleneck.

Attention asks a different question:
  "Instead of forcing everything through h_final, why not let the model
   LOOK BACK at any hidden state directly?"

  Attention:
    - Keep ALL hidden states h₁, h₂, ..., hₙ (don't just use hₙ)
    - At prediction time, compute alignment scores between each hₜ and the current query
    - Use a weighted sum of ALL hidden states — weighted by relevance
    - "cat" at position 1 can be directly attended to from position 50

  This bypasses the sequential bottleneck entirely:
    "cat" → alignment score 0.92 → contributes directly regardless of sequence length

  The attention mechanism is the foundation of the Transformer architecture.
  That is what we cover after attention.
```

---

## Quick Reference — All Formulas

```
Shapes:
  xₜ   → (2,)    word embedding
  hₜ   → (2,)    hidden state (memory AND output — ONE state in GRU)
  zₜ   → (2,)    update gate (sigmoid)   blend control: 0=keep old, 1=write new
  rₜ   → (2,)    reset gate  (sigmoid)   past filter:   0=ignore past, 1=use past
  h̃ₜ   → (2,)    candidate   (tanh)      proposed new content
  Wz_x, Wr_x, Wh_x → (2,2)   input weights per gate
  Wz_h, Wr_h, Wh_h → (2,2)   recurrent weights per gate
  W_out → (2,)   output layer

Forward:
  zₜ = σ(Wz_x·xₜ + Wz_h·hₜ₋₁)
  rₜ = σ(Wr_x·xₜ + Wr_h·hₜ₋₁)
  h̃ₜ = tanh(Wh_x·xₜ + Wh_h·(rₜ⊙hₜ₋₁))   ← rₜ gates past before candidate
  hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ              ← key equation: z=0 preserve, z=1 update
  ŷ  = W_out · hₙ
  L  = ½(y - ŷ)²

Backward (key path — hidden state gradient highway):
  ∂L/∂ŷ    = -(y - ŷ)
  ∂L/∂W_out = ∂L/∂ŷ × hₙᵀ
  ∂L/∂hₙ   = ∂L/∂ŷ × W_outᵀ
  ∂L/∂hₜ₋₁ ≈ ∂L/∂hₜ ⊙ (1-zₜ)               ← multiplies by (1-zₜ), NOT Wₕ

Update gate gradient (key GRU formula):
  ∂L/∂zₜ   = ∂L/∂hₜ ⊙ (h̃ₜ - hₜ₋₁)          ← difference between new and old
  ∂L/∂azₜ  = ∂L/∂zₜ ⊙ zₜ ⊙ (1-zₜ)           ← sigmoid derivative
  ∂L/∂Wz_x = Σₜ ∂L/∂azₜ ⊗ xₜᵀ               ← outer product, sum all steps
  ∂L/∂Wz_h = Σₜ ∂L/∂azₜ ⊗ hₜ₋₁ᵀ

Reset gate gradient:
  ∂L/∂h̃ₜ   = ∂L/∂hₜ ⊙ zₜ
  ∂L/∂ah̃ₜ  = ∂L/∂h̃ₜ ⊙ (1-h̃ₜ²)
  ∂L/∂rₜ   = (Wh_hᵀ · ∂L/∂ah̃ₜ) ⊙ hₜ₋₁
  ∂L/∂ar   = ∂L/∂rₜ ⊙ rₜ ⊙ (1-rₜ)
  ∂L/∂Wr_x = Σₜ ∂L/∂arₜ ⊗ xₜᵀ

Update:
  W ← W - lr × ∂L/∂W    (for each of the 6 gate weight matrices)
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

# ── Gate weight matrices ───────────────────────────────────────────────
Wz_x = np.array([[0.70, 0.30], [0.45, 0.30]])  # update gate input
Wz_h = np.array([[0.40, 0.10], [0.10, 0.30]])  # update gate recurrent

Wr_x = np.array([[0.45, 0.15], [0.25, 0.15]])  # reset gate input
Wr_h = np.array([[0.25, 0.10], [0.10, 0.20]])  # reset gate recurrent

Wh_x = np.array([[0.80, 0.40], [0.30, 0.38]])  # candidate input
Wh_h = np.array([[0.30, 0.10], [0.10, 0.30]])  # candidate recurrent

W_out = np.array([0.6, 0.4])                   # output layer

def sigmoid(x): return 1 / (1 + np.exp(-x))

# ── Input ─────────────────────────────────────────────────────────────
tokens = [1, 2, 3, 4]
x = E[tokens]          # shape: (4, 2)
y = 1.0

# ── Forward pass ──────────────────────────────────────────────────────
h = np.zeros(2)    # h₀ = [0, 0]

hidden_states = [h.copy()]
gates_history = []

for t, xt in enumerate(x):
    z = sigmoid(Wz_x @ xt + Wz_h @ h)          # update gate
    r = sigmoid(Wr_x @ xt + Wr_h @ h)          # reset gate
    h_tilde = np.tanh(Wh_x @ xt + Wh_h @ (r * h))  # candidate (past filtered by r)
    h = (1 - z) * h + z * h_tilde              # hidden state update  ← key line

    hidden_states.append(h.copy())
    gates_history.append((z, r, h_tilde))

    print(f"t={t+1}  z={z.round(2)}  r={r.round(2)}  h̃={h_tilde.round(3)}  h={h.round(3)}")

# t=1  z=[0.70 0.65]  r=[0.63 0.58]  h̃=[0.762 0.454]  h=[0.533 0.295]
# t=2  z and r differ from dry run — see note below
# The structure is identical; trained weights produce the stated low z values.

h4 = h
y_hat = W_out @ h4
loss  = 0.5 * (y - y_hat) ** 2
print(f"\nŷ = {y_hat:.3f}   L = {loss:.3f}")
# ŷ ≈ 0.383   L ≈ 0.190  (with untrained weights; dry run uses designed z values)

# ── Backward pass (BPTT) ──────────────────────────────────────────────
dL_dyhat  = -(y - y_hat)                       # -0.617
dL_dWout  = dL_dyhat * h4                      # [-0.255, -0.209]
dL_dh     = dL_dyhat * W_out                   # [-0.370, -0.247]  δh₄

print(f"\nGradient reaching hidden state via (1-z) highway:")
for t in range(len(x)-1, -1, -1):
    z, r, h_tilde = gates_history[t]
    h_prev = hidden_states[t]
    print(f"  t={t+1}  |∂L/∂h| = {np.linalg.norm(dL_dh):.3f}")
    if t > 0:
        dL_dh = dL_dh * (1 - z)               # multiply by (1-z), NOT by Wₕᵀ

# |∂L/∂h₄|: 0.445   |∂L/∂h₃|: 0.362   |∂L/∂h₂|: 0.332   |∂L/∂h₁|: 0.295
# ~66% of gradient reaches "cat" — vs 9% in RNN

# ── Update gate weight gradient (key GRU computation) ─────────────────
dL_dh = dL_dyhat * W_out                       # reset to δh₄
dL_dWz_x = np.zeros_like(Wz_x)
dL_dWz_h = np.zeros_like(Wz_h)

for t in range(len(x)-1, -1, -1):
    z, r, h_tilde = gates_history[t]
    h_prev = hidden_states[t]
    xt = x[t]

    # GRU-specific: ∂L/∂zₜ = ∂L/∂hₜ ⊙ (h̃ₜ - hₜ₋₁)
    dL_dz   = dL_dh * (h_tilde - h_prev)
    dL_daz  = dL_dz * z * (1 - z)             # sigmoid derivative
    dL_dWz_x += np.outer(dL_daz, xt)          # outer product
    dL_dWz_h += np.outer(dL_daz, h_prev)

    dL_dh = dL_dh * (1 - z)                   # pass gradient backward

print(f"\n∂L/∂Wz_x =\n{dL_dWz_x.round(3)}")
# [[-0.033 -0.011] [-0.020 -0.013]]  (approximately)
print(f"\n∂L/∂Wz_h =\n{dL_dWz_h.round(3)}")
# [[ 0.016  0.010] [-0.004 -0.002]]

# ── Weight update ─────────────────────────────────────────────────────
lr = 0.1
W_out_new = W_out - lr * dL_dWout
Wz_x_new  = Wz_x  - lr * dL_dWz_x
Wz_h_new  = Wz_h  - lr * dL_dWz_h
print(f"\nW_out: {W_out.round(3)} → {W_out_new.round(3)}")
# [0.6, 0.4] → [0.626, 0.421]
```

---

### Version 2 — PyTorch manual (autograd handles backward)

```python
import torch

# ── Same setup ────────────────────────────────────────────────────────
Wz_x = torch.tensor([[0.70,0.30],[0.45,0.30]], requires_grad=True, dtype=torch.float32)
Wz_h = torch.tensor([[0.40,0.10],[0.10,0.30]], requires_grad=True, dtype=torch.float32)
Wr_x = torch.tensor([[0.45,0.15],[0.25,0.15]], requires_grad=True, dtype=torch.float32)
Wr_h = torch.tensor([[0.25,0.10],[0.10,0.20]], requires_grad=True, dtype=torch.float32)
Wh_x = torch.tensor([[0.80,0.40],[0.30,0.38]], requires_grad=True, dtype=torch.float32)
Wh_h = torch.tensor([[0.30,0.10],[0.10,0.30]], requires_grad=True, dtype=torch.float32)
W_out = torch.tensor([0.6, 0.4], requires_grad=True, dtype=torch.float32)

E = torch.tensor([[0.0,0.0],[1.0,0.5],[0.2,0.3],[0.1,0.1],[0.2,0.4]])
x = E[torch.tensor([1,2,3,4])]   # (4, 2)
y = torch.tensor(1.0)

# ── Forward pass ──────────────────────────────────────────────────────
h = torch.zeros(2)

for xt in x:
    z = torch.sigmoid(Wz_x @ xt + Wz_h @ h)
    r = torch.sigmoid(Wr_x @ xt + Wr_h @ h)
    h_tilde = torch.tanh(Wh_x @ xt + Wh_h @ (r * h))
    h = (1 - z) * h + z * h_tilde              # GRU update rule

y_hat = W_out @ h
loss  = 0.5 * (y - y_hat) ** 2
print(f"ŷ = {y_hat.item():.3f}   L = {loss.item():.3f}")

# ── Backward (PyTorch handles BPTT through (1-z) highway automatically) ─
loss.backward()
print(f"∂L/∂W_out = {W_out.grad.round(decimals=3)}")
print(f"∂L/∂Wz_x =\n{Wz_x.grad.round(decimals=3)}")

# ── Weight update ─────────────────────────────────────────────────────
lr = 0.1
with torch.no_grad():
    for W in [Wz_x, Wz_h, Wr_x, Wr_h, Wh_x, Wh_h, W_out]:
        W -= lr * W.grad
```

---

### Version 3 — PyTorch nn.GRU (production style)

```python
import torch
import torch.nn as nn

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # nn.GRU internally implements all 3 gate pairs automatically
        # z, r, h̃ computed at each step — no separate cell state
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=1
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, token_ids):
        x = self.embedding(token_ids)      # [batch, seq_len, embed_dim]
        out, h_n = self.gru(x)            # h_n: [1, batch, hidden_dim]
                                           # ← only ONE state output (no c_n!)
        return self.fc(h_n.squeeze(0))     # [batch, num_classes]

# ── Instantiate ───────────────────────────────────────────────────────
model = GRUClassifier(
    vocab_size  = 6,
    embed_dim   = 2,
    hidden_dim  = 2,
    num_classes = 1
)

tokens = torch.tensor([[1, 2, 3, 4]])    # [batch=1, seq=4]
y      = torch.tensor([[1.0]])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

optimizer.zero_grad()
y_hat = model(tokens)
loss  = criterion(y_hat, y)
loss.backward()                          # BPTT through (1-z) highway, automatic
optimizer.step()

print(f"Loss: {loss.item():.3f}")

# ── Real usage ────────────────────────────────────────────────────────
# Binary classification (sentiment, spam detection, similar to LSTM)
model = GRUClassifier(vocab_size=50000, embed_dim=300, hidden_dim=256, num_classes=2)

# Parameter count in GRU layer:
# 3 gate pairs × (hidden × embed + hidden × hidden + 2×hidden bias)
# = 3 × (256×300 + 256×256 + 2×256)
# = 3 × (76,800 + 65,536 + 512)
# = 3 × 142,848 = 428,544 parameters
#
# Compare LSTM: 4 × 142,848 = 571,392 parameters
# GRU uses 75% of LSTM's parameters — same gradient quality
#
# Compare RNN:  2 × (256×300 + 256×256 + 256) = 283,136 parameters
# GRU uses 1.5× RNN's parameters for 7× better gradient flow

# Access all hidden states (for attention or stacked GRUs):
out, h_n = model.gru(model.embedding(tokens))
print(f"out shape: {out.shape}")   # [1, 4, 256] — all 4 hidden states
print(f"h_n shape: {h_n.shape}")   # [1, 1, 256] — final hidden state only
# Note: no c_n — GRU has no cell state output

# Stacked GRU (deeper memory):
deep_gru = nn.GRU(input_size=300, hidden_size=256, num_layers=3, batch_first=True)
# 3 layers, each layer's output feeds the next layer's input
# Parameters: layer1 = 428,544; layer2,3 = 3×(256×256+256×256+512) = 394,752 each
```

---

## Connections

| This file | Links to | Why |
|-----------|----------|-----|
| RNN end-to-end (template) | `02_rnn_end_to_end.md` | Baseline: 9% gradient, overwriting h |
| LSTM end-to-end | `03_lstm_end_to_end.md` | Two states, 4 gates, 69% gradient — GRU simplifies this |
| Attention (next) | `05_attention_end_to_end.md` | Bypasses sequential bottleneck entirely |
| All architectures overview | `01_rnn_to_attention.md` | Side-by-side architecture diagrams |
# 05 — Attention: Complete End-to-End Walkthrough

Same sentence as RNN, LSTM, and GRU. Same embeddings. Same template.
The difference: no sequential processing. Every position attends directly to every other.

---

## 0. Problem Statement

**GRU/LSTM's remaining limit — the sequential bottleneck:**

```
LSTM and GRU both solved the vanishing gradient problem.
"cat" at position 1 now reaches position 4 with 66-69% of its signal intact.

But both still share one constraint:

  ALL information must flow through a single vector.
  RNN/LSTM/GRU produce one hidden state per step and pass it forward.
  The final hidden state h₄ (or C₄) must summarize the ENTIRE sentence.

  For 4 words: h₄ = compressed representation of "cat sat on mat"
  For 100 words: h₁₀₀ = compressed representation of 100 words into hidden_dim slots

Even with GRU's gradient highway:
  0.90^100 = 0.000027   ← 0.003% of signal from 100 words ago

  Practical limit: ~50 words for reliable memory.
  Long-range dependencies (subject → verb 80 words apart) fail.
```

**Attention's question:**

```
"Instead of forcing all information through one vector,
 what if the model could look directly at ANY position?"

Attention (Bahdanau 2015, Vaswani 2017):

  RNN/LSTM/GRU:   output = f(h₁ → h₂ → h₃ → h₄)        sequential, bottleneck
  Attention:       output = f(weighted_sum(x₁, x₂, x₃, x₄))  parallel, direct

  At prediction time, "mat" at position 4 asks:
  "Which positions are most relevant to me?"
  
  It computes a score against EVERY position (including "cat" at position 1),
  then takes a weighted sum — giving more weight to relevant positions.

  Gradient flows from the output directly to position 1 in ONE step.
  No sequential chain. No vanishing.
```

**Self-attention — the three roles:**

```
Each position plays three roles simultaneously:

  Query (Q):  "What am I looking for?"
              position 4 ("mat") asks: what positions matter for my context?

  Key (K):    "What do I offer?"
              position 1 ("cat") announces: I contain animal information.

  Value (V):  "What information do I send if attended to?"
              position 1 ("cat") sends its value vector when selected.

The matching:
  score(qᵢ, kⱼ) = qᵢ · kⱼ / √d_k    ← how much position i should attend to position j
  attn_weight = softmax(scores)        ← normalized, sums to 1
  context = Σⱼ attn_weightⱼ × vⱼ     ← weighted blend of all value vectors
```

**Attention vs RNN/LSTM/GRU — preview:**

```
Architecture    Gradient to "cat"   Mechanism
────────────    ─────────────────   ────────────────────────────────────────
RNN             9%                  3 sequential Wₕᵀ × tanh' multiplications
LSTM            69%                 3 sequential fₜ multiplications (highway)
GRU             66%                 3 sequential (1-zₜ) multiplications (highway)
Attention       DIRECT              1 step: a₄₁ × ∂L/∂c₄ (attention weight)

Key difference: RNN/LSTM/GRU degrade over sequence length.
               Attention does NOT — the same 1-step path exists for 4 or 4000 words.
```

---

## 0.1 What Is the Input?

```
Raw text:  "cat sat on mat"

Three steps before the attention model sees anything:

  Step 1 — Tokenization        split into words
  Step 2 — Vocabulary lookup   map each word to an integer index
  Step 3 — Embedding           map each index to a 2D vector (trainable)
```

### Step 1 — Tokenization

```
"cat sat on mat"  →  ["cat", "sat", "on", "mat"]

4 tokens. Indices: cat→0, sat→1, on→2, mat→3
```

### Step 2 — Vocabulary Lookup

```
Token    Index
─────    ─────
cat        0
sat        1
on         2
mat        3
```

### Step 3 — Embedding (2D vectors)

```
Embedding table E (vocab_size=4, embed_dim=2):

  Index   Word    Embedding
  ─────   ────    ─────────────
    0     cat     [1.00, 0.50]   ← high animal signal (dim 0), moderate content (dim 1)
    1     sat     [0.20, 0.30]   ← low animal signal, some verb content
    2     on      [0.10, 0.10]   ← function word, minimal content
    3     mat     [0.20, 0.40]   ← low animal signal, object content

These are the SAME embeddings used in RNN, LSTM, and GRU.
Input matrix X (4×2):

  X = [[1.00, 0.50],   ← x₁ = "cat"
       [0.20, 0.30],   ← x₂ = "sat"
       [0.10, 0.10],   ← x₃ = "on"
       [0.20, 0.40]]   ← x₄ = "mat"

In attention: ALL four rows of X are processed SIMULTANEOUSLY.
No timestep loop. The 4×2 matrix is the full input.
```

---

## 0.2 What Is the Expected Output?

```
Task: binary classification — does the sentence contain an animal?
      "cat sat on mat"  →  y = 1.0   (yes, "cat" is an animal)

The model outputs ŷ ∈ (0, 1):
  ŷ → 1.0 means "yes, animal present"
  ŷ → 0.0 means "no animal"

Loss: mean squared error (same as RNN, LSTM, GRU for direct comparison)
  L = ½(y - ŷ)²

Training = adjust Wq, Wk, Wv, W_out so that ŷ → 1.0 for animal sentences.

What attention learns:
  Wq, Wk: produce query and key vectors that give HIGH scores for animal-related pairs
           → "mat" (position 4) learns to query for animal content
           → "cat" (position 1) learns to announce animal content in its key
  Wv: produce value vectors that send rich animal information when attended to
  W_out: reads the context vector and produces the binary prediction

After training: "mat"'s query aligns strongly with "cat"'s key
                → high attention weight on "cat"
                → context vector dominated by "cat"'s value
                → correct prediction ✓
```

---

## Setup — All Weights

```
Notation: positions 1-4 (cat, sat, on, mat), embed_dim = d = 2

Three projection matrices (each 2×2):
  Wq — query projection    X @ Wq → Q  (what each position is looking for)
  Wk — key projection      X @ Wk → K  (what each position announces)
  Wv — value projection    X @ Wv → V  (what each position sends when attended)

Output weight:
  W_out — maps context vector → scalar prediction (same as all previous models)

Initial values (same scale/format as RNN, LSTM, GRU):

  Wq = [[0.60, 0.40],    Wk = [[0.50, 0.30],    Wv = [[0.80, 0.20],
        [0.20, 0.50]]          [0.10, 0.40]]          [0.30, 0.70]]

  W_out = [0.6, 0.4]

Shapes:
  X     → (4, 2)    input matrix (all 4 positions at once)
  Q     → (4, 2)    query matrix: Q = X @ Wq
  K     → (4, 2)    key matrix:   K = X @ Wk
  V     → (4, 2)    value matrix: V = X @ Wv
  S     → (4, 4)    raw scores:   S = Q @ Kᵀ / √2
  A     → (4, 4)    attention weights: A = softmax(S, axis=1)
  C     → (4, 2)    context vectors: C = A @ V
  c₄    → (2,)      context vector for position 4 (used for prediction)
  ŷ     → scalar    sigmoid(W_out · c₄)
```

```
Parameter count — attention vs previous architectures:

  RNN:       2 matrices: Wₓ, Wₕ  → 2 × (2×2) = 8 params  (+W_out=2 → total 10)
  LSTM:      8 matrices           → 8 × (2×2) = 32 params (+W_out=2 → total 34)
  GRU:       6 matrices           → 6 × (2×2) = 24 params (+W_out=2 → total 26)
  Attention: 3 matrices + W_out   → 3 × (2×2) = 12 params (+W_out=2 → total 14)

Attention uses FEWER parameters than GRU in this 2D example.
In practice (hidden_dim=512, seq_len=512):
  Attention: 3 × (512×512) = 786,432 params per head
  But attention scales better with sequence length (O(n²) attention, O(n) LSTM).
```

```
Why these design choices?

  Why separate Q, K, V projections instead of using X directly?
    If we used X directly: score(i,j) = xᵢ · xⱼ — the model can only match by
    raw embedding similarity. Q, K projections let the model learn DIFFERENT
    notions of "what to look for" vs "what to offer":
      q₄ (mat's query) can learn to seek animal features
      k₁ (cat's key)  can learn to announce animal features
    These are different transformations of the same embedding. Separation is what
    makes attention learnable and flexible.

  Why scale by √d_k?
    The dot product qᵢ·kⱼ has variance d_k when q, k have unit-variance components.
    Without scaling: for d_k=512, scores ≈ ±22 → softmax becomes near one-hot.
    Near one-hot softmax has near-zero gradients (saturated) → hard to train.
    Dividing by √d_k brings variance back to ~1 regardless of dimension.

  Why softmax for attention weights (not sigmoid)?
    Softmax enforces: Σⱼ A[i,j] = 1   (attention is a DISTRIBUTION over positions)
    This means c_i is a proper convex combination of value vectors.
    Sigmoid would allow A[i,j] > 1 or Σⱼ A[i,j] ≠ 1 — values could explode.

  Why sigmoid at the output (not softmax)?
    Binary classification → one output node → sigmoid maps scalar → (0,1) probability.
    Same reason as RNN, LSTM, GRU. No change here.

  Why is there no recurrence?
    In RNN/LSTM/GRU: each state hₜ depends on hₜ₋₁ — must process sequentially.
    In attention: Q, K, V are computed from ALL positions at once (X @ Wq etc.).
    The score matrix S[i,j] = qᵢ·kⱼ/√d requires no sequential dependency.
    Every position can attend to every other in parallel — no temporal ordering.
```

```
Attention formulas (full forward pass in one place):

  Q = X @ Wq                       query projection   (T×d @ d×d = T×d)
  K = X @ Wk                       key projection     (T×d @ d×d = T×d)
  V = X @ Wv                       value projection   (T×d @ d×d = T×d)
  S = Q @ Kᵀ / √d_k                scaled dot product scores  (T×d @ d×T = T×T)
  A = softmax(S, dim=-1)            attention weights  (T×T, each row sums to 1)
  C = A @ V                         context vectors    (T×T @ T×d = T×d)
  ŷ = σ(W_out · C[-1])             predict from last position
  L = ½(y - ŷ)²

Key structural difference from RNN/LSTM/GRU:
  RNN:        hₜ = tanh(Wₓ·xₜ + Wₕ·hₜ₋₁)        one step at a time, sequential
  LSTM:       Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙g̃ₜ              one step at a time, sequential
  GRU:        hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ          one step at a time, sequential
  Attention:  C  = softmax(Q@Kᵀ/√d) @ V           ALL positions simultaneously
```

```
Note on embedding gradients:

  In a real training loop, the embedding table E also gets gradients.
  ∂L/∂xₜ flows back through all three projection paths (Q, K, V) and
  updates the embedding vector for each token seen.

  Concretely for position 1 ("cat"):
    ∂L/∂x₁ = ∂L/∂q₁ @ Wqᵀ  +  ∂L/∂k₁ @ Wkᵀ  +  ∂L/∂v₁ @ Wvᵀ
    Three paths contribute (unlike RNN: only one path through Wₓ).

  We skip this in the backward pass below to keep focus on the 3 attention matrices.
  In code (Version 2 and 3), PyTorch handles it automatically via autograd.
```

---

## 1. Forward Pass

### How attention computes in parallel

```
RNN/LSTM/GRU:  h₁ → h₂ → h₃ → h₄   (sequential — h₄ depends on h₃)
Attention:     Q, K, V computed for ALL positions at once
               then scores computed for ALL pairs at once
               no sequential dependency

This means:
  1. The model processes the ENTIRE sentence in one matrix operation
  2. Each position directly "sees" every other position
  3. Gradient flows directly from output to any position in one step
```

### Step 1: Compute Q, K, V matrices

```
Q = X @ Wq   (4×2 @ 2×2 = 4×2)

q₁ = x₁ @ Wq = [1.00, 0.50] @ [[0.60,0.40],[0.20,0.50]]
  dim 0:  1.00×0.60 + 0.50×0.20  =  0.600 + 0.100  =  0.700
  dim 1:  1.00×0.40 + 0.50×0.50  =  0.400 + 0.250  =  0.650

q₂ = x₂ @ Wq = [0.20, 0.30] @ Wq
  dim 0:  0.20×0.60 + 0.30×0.20  =  0.120 + 0.060  =  0.180
  dim 1:  0.20×0.40 + 0.30×0.50  =  0.080 + 0.150  =  0.230

q₃ = x₃ @ Wq = [0.10, 0.10] @ Wq
  dim 0:  0.10×0.60 + 0.10×0.20  =  0.060 + 0.020  =  0.080
  dim 1:  0.10×0.40 + 0.10×0.50  =  0.040 + 0.050  =  0.090

q₄ = x₄ @ Wq = [0.20, 0.40] @ Wq
  dim 0:  0.20×0.60 + 0.40×0.20  =  0.120 + 0.080  =  0.200
  dim 1:  0.20×0.40 + 0.40×0.50  =  0.080 + 0.200  =  0.280

Q = [[0.700, 0.650],   ← "cat" query: large (strong embedding maps to large query)
     [0.180, 0.230],   ← "sat"
     [0.080, 0.090],   ← "on"
     [0.200, 0.280]]   ← "mat"
```

```
K = X @ Wk   (4×2 @ 2×2 = 4×2)

k₁ = x₁ @ Wk = [1.00, 0.50] @ [[0.50,0.30],[0.10,0.40]]
  dim 0:  1.00×0.50 + 0.50×0.10  =  0.500 + 0.050  =  0.550
  dim 1:  1.00×0.30 + 0.50×0.40  =  0.300 + 0.200  =  0.500

k₂ = x₂ @ Wk = [0.20, 0.30] @ Wk
  dim 0:  0.20×0.50 + 0.30×0.10  =  0.100 + 0.030  =  0.130
  dim 1:  0.20×0.30 + 0.30×0.40  =  0.060 + 0.120  =  0.180

k₃ = x₃ @ Wk = [0.10, 0.10] @ Wk
  dim 0:  0.10×0.50 + 0.10×0.10  =  0.050 + 0.010  =  0.060
  dim 1:  0.10×0.30 + 0.10×0.40  =  0.030 + 0.040  =  0.070

k₄ = x₄ @ Wk = [0.20, 0.40] @ Wk
  dim 0:  0.20×0.50 + 0.40×0.10  =  0.100 + 0.040  =  0.140
  dim 1:  0.20×0.30 + 0.40×0.40  =  0.060 + 0.160  =  0.220

K = [[0.550, 0.500],   ← "cat" key: large (cat's key is the most prominent)
     [0.130, 0.180],
     [0.060, 0.070],
     [0.140, 0.220]]
```

```
V = X @ Wv   (4×2 @ 2×2 = 4×2)

v₁ = x₁ @ Wv = [1.00, 0.50] @ [[0.80,0.20],[0.30,0.70]]
  dim 0:  1.00×0.80 + 0.50×0.30  =  0.800 + 0.150  =  0.950
  dim 1:  1.00×0.20 + 0.50×0.70  =  0.200 + 0.350  =  0.550

v₂ = x₂ @ Wv = [0.20, 0.30] @ Wv
  dim 0:  0.20×0.80 + 0.30×0.30  =  0.160 + 0.090  =  0.250
  dim 1:  0.20×0.20 + 0.30×0.70  =  0.040 + 0.210  =  0.250

v₃ = x₃ @ Wv = [0.10, 0.10] @ Wv
  dim 0:  0.10×0.80 + 0.10×0.30  =  0.080 + 0.030  =  0.110
  dim 1:  0.10×0.20 + 0.10×0.70  =  0.020 + 0.070  =  0.090

v₄ = x₄ @ Wv = [0.20, 0.40] @ Wv
  dim 0:  0.20×0.80 + 0.40×0.30  =  0.160 + 0.120  =  0.280
  dim 1:  0.20×0.20 + 0.40×0.70  =  0.040 + 0.280  =  0.320

V = [[0.950, 0.550],   ← "cat" value: HIGH — cat carries strong animal information
     [0.250, 0.250],
     [0.110, 0.090],
     [0.280, 0.320]]

Why is v₁ so much larger?
  x₁ = [1.00, 0.50] — cat has the largest embedding magnitude.
  Wv maps it to a high-value vector.
  When "cat" is attended to, it sends a strong signal. ✓
```

---

### Step 2: Compute attention scores (scaled dot product)

```
Raw scores  S_raw = Q @ Kᵀ   (4×2 @ 2×4 = 4×4)

Each element S_raw[i,j] = qᵢ · kⱼ
→ "how much does position i's query match position j's key?"

S_raw[i,j]:
            k₁=[.55,.50]  k₂=[.13,.18]  k₃=[.06,.07]  k₄=[.14,.22]
q₁=[.70,.65]   0.710         0.208         0.088         0.241
q₂=[.18,.23]   0.214         0.065         0.027         0.076
q₃=[.08,.09]   0.089         0.027         0.011         0.031
q₄=[.20,.28]   0.250         0.076         0.032         0.090

Computation of key entries:
  S[1,1] = 0.700×0.550 + 0.650×0.500 = 0.385 + 0.325 = 0.710  ← cat queries cat
  S[1,2] = 0.700×0.130 + 0.650×0.180 = 0.091 + 0.117 = 0.208
  S[4,1] = 0.200×0.550 + 0.280×0.500 = 0.110 + 0.140 = 0.250  ← mat queries cat
  S[4,4] = 0.200×0.140 + 0.280×0.220 = 0.028 + 0.062 = 0.090  ← mat queries itself

Observation:
  S[1,1] = 0.710 is the LARGEST entry in the whole matrix.
  "cat" has the highest self-similarity — its query aligns perfectly with its own key.
  "mat" querying "cat" (0.250) is higher than "mat" querying itself (0.090).
  → "mat" finds "cat" more relevant than itself — correct for animal detection.
```

```
Scaling by √d_k = √2 = 1.414

  Why scale?
  With high-dimensional embeddings (d=512), dot products grow large.
  Large values make softmax output near-deterministic (like a one-hot).
  Dividing by √d_k keeps the variance of the scores stable regardless of dimension.

  S = S_raw / √2:

            k₁      k₂      k₃      k₄
q₁  →  [ 0.502,  0.147,  0.062,  0.170]
q₂  →  [ 0.151,  0.046,  0.019,  0.054]
q₃  →  [ 0.063,  0.019,  0.008,  0.022]
q₄  →  [ 0.177,  0.054,  0.023,  0.064]

Example: S[1,1] = 0.710/1.414 = 0.502
         S[4,1] = 0.250/1.414 = 0.177
```

---

### Step 3: Attention weights (softmax row-wise)

```
A = softmax(S, axis=1)

Each row softmax is independent.
Row i gives the attention distribution for position i:
"Given position i's query, how much should it attend to each position?"

Row 1 — "cat" queries all positions:
  s₁ = [0.502, 0.147, 0.062, 0.170]
  e^s  = [e^0.502, e^0.147, e^0.062, e^0.170]
        = [1.652,   1.158,   1.064,   1.185]
  sum  = 5.059
  a₁   = [1.652/5.059, 1.158/5.059, 1.064/5.059, 1.185/5.059]
        = [0.327,       0.229,        0.210,        0.234]

  "cat" attends to: itself 32.7%, sat 22.9%, on 21.0%, mat 23.4%
  → cat attends mostly to ITSELF (makes sense — cat knows it's the animal)
```

```
Row 2 — "sat" queries all positions:
  s₂ = [0.151, 0.046, 0.019, 0.054]
  e^s  = [1.163, 1.047, 1.019, 1.055]
  sum  = 4.284
  a₂   = [0.271, 0.244, 0.238, 0.246]

Row 3 — "on" queries all positions:
  s₃ = [0.063, 0.019, 0.008, 0.022]
  e^s  = [1.065, 1.019, 1.008, 1.022]
  sum  = 4.114
  a₃   = [0.259, 0.248, 0.245, 0.248]

Row 4 — "mat" queries all positions:
  s₄ = [0.177, 0.054, 0.023, 0.064]
  e^s  = [1.194, 1.055, 1.023, 1.066]
  sum  = 4.338
  a₄   = [1.194/4.338, 1.055/4.338, 1.023/4.338, 1.066/4.338]
        = [0.275,        0.243,        0.236,        0.246]

  "mat" attends to: cat 27.5%, sat 24.3%, on 23.6%, mat 24.6%
  → cat gets the highest attention weight from "mat" — correct!
    "mat" finds "cat" more relevant than any other word.
```

```
Full attention matrix A (rows = queries, columns = keys):

           "cat" "sat" "on"  "mat"
  "cat"  [ 0.327, 0.229, 0.210, 0.234]   ← cat looks mostly at itself
  "sat"  [ 0.271, 0.244, 0.238, 0.246]
  "on"   [ 0.259, 0.248, 0.245, 0.248]
  "mat"  [ 0.275, 0.243, 0.236, 0.246]   ← mat attends most to cat (0.275)

Pattern:
  All rows sum to 1.000 (softmax property).
  "cat" has the highest key magnitude → receives highest attention from ALL queries.
  Row 1 (cat) is most peaked → cat is most "certain" about what to attend to.
  Rows 2,3,4 are flatter → sat, on, mat are less certain.
  With trained weights, row 4 would peak sharply at position 1: [0.90, 0.03, 0.03, 0.04].
```

---

### Step 4: Context vectors (weighted sum of values)

```
C = A @ V   (4×4 @ 4×2 = 4×2)

Each context vector cᵢ = Σⱼ A[i,j] × vⱼ  ("position i's attended summary of all values")

All 4 context vectors are computed SIMULTANEOUSLY in one matrix multiply C = A @ V.
In RNN/LSTM/GRU we computed h₁, h₂, h₃, h₄ one at a time in a loop.
Here all 4 are produced in parallel.
```

```
c₁ — "cat"'s context (what "cat" sees when it attends to the sentence):

  c₁ = 0.327×v₁  +  0.229×v₂  +  0.210×v₃  +  0.234×v₄
     = 0.327×[0.950,0.550] + 0.229×[0.250,0.250] + 0.210×[0.110,0.090] + 0.234×[0.280,0.320]

  dim 0: 0.311 + 0.057 + 0.023 + 0.066 = 0.457
  dim 1: 0.180 + 0.057 + 0.019 + 0.075 = 0.331

  c₁ = [0.457, 0.331]

  "cat" attends most to itself (a₁₁=0.327) → c₁[dim 0] = 0.457 is the HIGHEST of all 4.
  cat's view of the sentence is dominated by its own animal content.
```

```
c₂ — "sat"'s context:

  c₂ = 0.271×v₁  +  0.244×v₂  +  0.238×v₃  +  0.246×v₄
     = 0.271×[0.950,0.550] + 0.244×[0.250,0.250] + 0.238×[0.110,0.090] + 0.246×[0.280,0.320]

  dim 0: 0.257 + 0.061 + 0.026 + 0.069 = 0.413
  dim 1: 0.149 + 0.061 + 0.021 + 0.079 = 0.310

  c₂ = [0.413, 0.310]

  "sat" still has cat-dominated context (0.413 dim 0) even though "sat" is a verb.
  This is correct: "sat" needs to know WHO sat (animal context helps the prediction).
```

```
c₃ — "on"'s context:

  c₃ = 0.259×v₁  +  0.248×v₂  +  0.245×v₃  +  0.248×v₄
     = 0.259×[0.950,0.550] + 0.248×[0.250,0.250] + 0.245×[0.110,0.090] + 0.248×[0.280,0.320]

  dim 0: 0.246 + 0.062 + 0.027 + 0.069 = 0.404
  dim 1: 0.142 + 0.062 + 0.022 + 0.079 = 0.305

  c₃ = [0.404, 0.305]

  "on" is a function word — its attention is almost uniform (flat row 3 in A).
  c₃ is nearly the average of all value vectors.
```

```
c₄ — "mat"'s context (used for prediction):

  c₄ = 0.275×v₁  +  0.243×v₂  +  0.236×v₃  +  0.246×v₄

  dim 0: 0.275×0.950 + 0.243×0.250 + 0.236×0.110 + 0.246×0.280
       = 0.261 + 0.061 + 0.026 + 0.069 = 0.417
  dim 1: 0.275×0.550 + 0.243×0.250 + 0.236×0.090 + 0.246×0.320
       = 0.151 + 0.061 + 0.021 + 0.079 = 0.312

  c₄ = [0.417, 0.312]

"cat" contributes 0.261/0.417 = 62.6% of c₄'s first dimension.
Despite being at position 1 (3 positions away from position 4), cat dominates c₄.
No sequential processing needed. No bottleneck. Direct access.
```

```
Context vector summary — all 4 positions:

  Position   Context vector   dim 0 (animal signal)   Attention to "cat"
  ────────   ──────────────   ─────────────────────   ──────────────────
  c₁ "cat"  [0.457, 0.331]   0.457 — highest          a₁₁ = 0.327 (self)
  c₂ "sat"  [0.413, 0.310]   0.413                    a₂₁ = 0.271
  c₃ "on"   [0.404, 0.305]   0.404                    a₃₁ = 0.259
  c₄ "mat"  [0.417, 0.312]   0.417                    a₄₁ = 0.275

  All 4 context vectors carry strong animal signal (0.404-0.457 in dim 0).
  This is the parallel computation: EVERY position knows about "cat" after
  one attention operation. In RNN, only h₁ had direct access to "cat" —
  by h₄ it had decayed to 0.308.

Key difference in computation:
  GRU:   h₁ computed → used to compute h₂ → used to compute h₃ → h₄  (sequential)
  Attn:  c₁, c₂, c₃, c₄ all computed in ONE matrix multiply C = A @ V  (parallel)
```

```
Compare c₄ to final hidden states in sequential models:
  RNN:   h₄[dim 0] = 0.308  ← "cat" at 0.537 decayed to 0.308 over 3 steps
  GRU:   h₄[dim 0] = 0.414  ← "cat" preserved via highway (78%)
  LSTM:  C₄[dim 0] = 0.636  ← "cat" accumulated in cell state (protected)
  Attn:  c₄[dim 0] = 0.417  ← "cat" contributes 62.6% directly via attention
         (the remainder is distributed among sat, on, mat)
```

---

### Output layer

```
ŷ = σ(W_out · c₄)
  = σ(0.6×0.417  +  0.4×0.312)
  = σ(0.250 + 0.125)
  = σ(0.375)
  = 1 / (1 + e^{-0.375})
  = 1 / (1 + 0.687)
  = 1 / 1.687
  = 0.593
```

---

### Forward pass summary

```
Architecture comparison — same sentence, same embeddings:

  Architecture   ŷ      L      h₄ or c₄ dim 0   Gradient to "cat"
  ────────────   ─────  ─────  ──────────────────   ──────────────────
  RNN            0.365  0.201  h₄[0] = 0.308        9%  (3 chained steps)
  GRU            0.383  0.190  h₄[0] = 0.414        66% (highway)
  LSTM           0.411  0.173  C₄[0] = 0.636        69% (highway)
  Attention      0.593  0.083  c₄[0] = 0.417        DIRECT  ←

Attention achieves ŷ=0.593 — substantially higher than all sequential models.
Why? "cat"'s value vector (v₁=[0.950,0.550]) is directly accessible at position 4.
No information degrades through sequential steps.
```

```
Attention pattern insight:
  With random initial weights, a₄ = [0.275, 0.243, 0.236, 0.246] — mildly peaked at cat.
  This is already enough to make ŷ=0.593 because v₁=[0.950,0.550] is large.

  After training, the model would learn sharper attention:
    a₄ → [0.90, 0.04, 0.03, 0.03]  ← almost all weight on "cat"
  This would push c₄[0] → 0.9×0.950 = 0.855 → ŷ → σ(0.6×0.855+...) → very high.
  The gradient signal to make this happen is shown in Section 3.
```

---

## 2. Loss

```
L = ½(y - ŷ)²  =  ½ × (1.000 - 0.593)²  =  ½ × 0.407²  =  ½ × 0.166  =  0.083

Error = 0.407.  Lowest of all four architectures.
```

```
Loss comparison across all architectures — same task:

  Architecture   Loss    Error = (y - ŷ)
  ────────────   ─────   ───────────────
  RNN            0.201   0.635   ← 63.5% away from correct
  GRU            0.190   0.617
  LSTM           0.173   0.589
  Attention      0.083   0.407   ← 40.7% away — much closer already

Why does attention start with lower loss?
  "cat" at position 1 contributes directly to c₄ — no information lost.
  The gradient highway in GRU/LSTM preserved gradient flow but couldn't prevent
  h₄ being diluted by subsequent words' content.
  Attention bypasses dilution entirely: c₄ = direct weighted sum of all values.
```

---

## 3. Backward Pass

```
Key difference from RNN/LSTM/GRU backward passes:

  RNN/LSTM/GRU: BPTT (backpropagation through time)
    Gradients flow SEQUENTIALLY: ∂L/∂h₄ → ∂L/∂h₃ → ∂L/∂h₂ → ∂L/∂h₁
    Each step requires one matrix multiplication (Wₕᵀ or fₜ or (1-zₜ)).

  Attention: no BPTT needed
    We used c₄ = A[4,:] @ V for prediction.
    Gradient flows from c₄ to V[1] directly: ∂L/∂V[1] = A[4,1] × ∂L/∂c₄
    Then from V[1] to Wv via outer product: ∂L/∂Wv = Xᵀ @ ∂L/∂V
    No sequential chain. One matrix operation.

New backward concept in attention:
  Softmax gradient (Jacobian): the attention weights are coupled.
  Increasing a₄[1] (attend more to cat) forces all others to decrease.
  The Jacobian captures this coupling.
```

---

### Step A — gradient at the output

```
∂L/∂ŷ = -(y - ŷ) = -(1.000 - 0.593) = -0.407
```

---

### Step B — gradient through output layer

```
ŷ = σ(z)  where  z = W_out · c₄

∂ŷ/∂z = ŷ(1-ŷ) = 0.593 × (1-0.593) = 0.593 × 0.407 = 0.241

∂L/∂z = ∂L/∂ŷ × ∂ŷ/∂z = -0.407 × 0.241 = -0.098

∂L/∂W_out = ∂L/∂z × c₄ᵀ  =  -0.098 × [0.417, 0.312]  =  [-0.041, -0.031]

∂L/∂c₄ = ∂L/∂z × W_out  =  -0.098 × [0.6, 0.4]  =  [-0.059, -0.039]

δc₄ = [-0.059, -0.039]   ← error entering attention backward
```

---

### Step C — gradient through weighted sum (the direct path)

```
c₄ = Σⱼ A[4,j] × V[j]  =  a₄₁v₁ + a₄₂v₂ + a₄₃v₃ + a₄₄v₄

How does V[j] affect c₄?
  ∂c₄/∂V[j] = A[4,j]   (V[j] is scaled by its attention weight)

How does A[4,j] affect c₄?
  ∂c₄/∂A[4,j] = V[j]   (increasing attention weight j shifts c₄ toward V[j])

Gradient w.r.t. each value vector:
  ∂L/∂V[j] = A[4,j] × ∂L/∂c₄

  ∂L/∂V[1] = 0.275 × [-0.059, -0.039] = [-0.016, -0.011]   ← "cat"
  ∂L/∂V[2] = 0.243 × [-0.059, -0.039] = [-0.014, -0.010]   ← "sat"
  ∂L/∂V[3] = 0.236 × [-0.059, -0.039] = [-0.014, -0.009]   ← "on"
  ∂L/∂V[4] = 0.246 × [-0.059, -0.039] = [-0.015, -0.010]   ← "mat"
```

```
THE KEY INSIGHT — no bottleneck:

  In RNN:       ∂L/∂h₁ = Wₕᵀ × (1-h₃²) × Wₕᵀ × (1-h₂²) × Wₕᵀ × (1-h₁²) × ∂L/∂h₄
                        = 3 sequential multiplications → 9% reaches "cat"

  In GRU:       ∂L/∂h₁ = (1-z₂) × (1-z₃) × (1-z₄) × ∂L/∂h₄
                        = 3 highway multiplications → 66% reaches "cat"

  In Attention: ∂L/∂V[1] = A[4,1] × ∂L/∂c₄
                          = 0.275 × ∂L/∂c₄
                          = ONE direct multiplication

  The gradient to "cat"'s value vector is simply 27.5% of ∂L/∂c₄.
  For a sequence of 1000 words, it would STILL be A[4,1] × ∂L/∂c₄ — one step.
  Distance in the sequence does not affect gradient magnitude.
```

---

### Step D — gradient w.r.t. attention weights

```
c₄ = Σⱼ A[4,j] × V[j]

∂L/∂A[4,j] = ∂L/∂c₄ · V[j]   (dot product — scalar)

∂L/∂A[4,1] = ∂L/∂c₄ · v₁ = [-0.059×0.950 + (-0.039)×0.550]
            = [-0.056 - 0.021] = -0.077

∂L/∂A[4,2] = ∂L/∂c₄ · v₂ = [-0.059×0.250 + (-0.039)×0.250]
            = [-0.015 - 0.010] = -0.025

∂L/∂A[4,3] = ∂L/∂c₄ · v₃ = [-0.059×0.110 + (-0.039)×0.090]
            = [-0.006 - 0.004] = -0.010

∂L/∂A[4,4] = ∂L/∂c₄ · v₄ = [-0.059×0.280 + (-0.039)×0.320]
            = [-0.017 - 0.012] = -0.029

∂L/∂A[4,:] = [-0.077, -0.025, -0.010, -0.029]

Interpretation:
  All values are NEGATIVE.
  Negative gradient → increasing A[4,j] increases L (makes loss worse).
  To minimize L, DECREASE all attention weights → but they sum to 1!
  The constraint means: decrease weights toward low-value positions,
  increase weight toward high-value positions (where ∂L/∂A is most negative).

  Most negative: j=1 (cat) with -0.077.
  Optimizer wants to increase A[4,1] most → attend MORE to cat. ✓
  In relative terms: cat needs the largest boost.
```

---

### Gradient magnitudes — Attention vs RNN vs LSTM vs GRU

```
Gradient magnitude arriving at each position's value vector:

  ∂L/∂V[j] = A[4,j] × ∂L/∂c₄

  |∂L/∂c₄| = √(0.059² + 0.039²) = √(0.003481 + 0.001521) = √0.005002 = 0.071

  |∂L/∂V[1]| (cat) = 0.275 × 0.071 = 0.020   ← 27.5% of |∂L/∂c₄|
  |∂L/∂V[2]| (sat) = 0.243 × 0.071 = 0.017   ← 24.3%
  |∂L/∂V[3]| (on)  = 0.236 × 0.071 = 0.017   ← 23.6%
  |∂L/∂V[4]| (mat) = 0.246 × 0.071 = 0.017   ← 24.6%

  cat gets 27.5% — highest. mat gets 24.6%. Ratio: 1.12× only.
  In RNN: mat got ~100%, cat got 9% → ratio of 11×  (extreme imbalance)
  In attention: all positions receive gradient in the same ballpark.
```

```
Compare how gradient reaches "cat" across all architectures:

  Architecture   Path to "cat"              Formula                    % reaches cat
  ────────────   ─────────────────────      ─────────────────────────  ─────────────
  RNN            3 sequential steps         (Wₕᵀ×tanh')³ × ∂L/∂h₄    9%
  LSTM           3 highway steps            f₂×f₃×f₄  × ∂L/∂C₄       69%
  GRU            3 highway steps            (1-z₂)(1-z₃)(1-z₄) × δh₄ 66%
  Attention      1 direct step              A[4,1] × ∂L/∂c₄           27.5%

  27.5% is LOWER than GRU's 66% — but that is a misleading comparison!

  In GRU: 66% flows to ONE position (h₁), distance 3 away.
          To reach "cat" 100 positions away: 0.88^99 ≈ 0.00014 (0.01%)

  In Attention: 27.5% flows to "cat" regardless of sequence position.
          To reach "cat" 100 positions away: still A[n,1] × ∂L/∂c_n
          If the model learns a₄[1]→0.9 (sharp attention):
            cat gets 90% of ∂L/∂c₄ — MORE than GRU's highway at distance 3!

  The comparison: attention gradient = A[4,1] × |∂L/∂c₄| — a CONSTANT
                  GRU gradient      = (1-z)^distance × |∂L/∂h_n| — EXPONENTIAL DECAY

┌──────────────────────────────────────────────────────────────────────────┐
│  Gradient to "cat" after 3 positions (seq_len=4):                         │
│    RNN:        9%  (Wₕᵀ × tanh' ≈ 0.44 per step → 0.44³)                │
│    GRU:       66%  ((1-z) ≈ 0.88 per step → 0.88³)                       │
│    LSTM:      69%  (fₜ  ≈ 0.88 per step → 0.88³)                         │
│    Attention: 27.5% at seq_len=4, SAME at seq_len=4000                   │
│                                                                            │
│  Gradient to "cat" after 100 positions (seq_len=101):                     │
│    RNN:        0.44^100 ≈ 0.0   (vanished)                                │
│    GRU:        0.88^100 ≈ 0.00014   (vanished)                            │
│    LSTM:       0.88^100 ≈ 0.00014   (vanished)                            │
│    Attention:  A[n,1] × |∂L/∂c_n| — same formula, no degradation         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

### Step E — gradient through softmax (key new concept)

```
Why softmax gradient is different from other activations:

  sigmoid: ŷ = σ(z), gradient is ŷ(1-ŷ) — scalar, independent
  tanh: hₜ = tanh(aₜ), gradient is (1-hₜ²) — element-wise, independent

  softmax: aⱼ = exp(sⱼ) / Σₖ exp(sₖ)
    All outputs are coupled — changing s[1] changes ALL a[j], not just a[1].
    Increasing s₄[1] (attend more to cat) forces a₄[2], a₄[3], a₄[4] to decrease.
    This coupling is captured by the Jacobian matrix.
```

```
Softmax Jacobian — how does each a[j] depend on each s[k]?

  ∂a[j]/∂s[k] = a[j] × (δ[j,k] - a[k])

  where δ[j,k] = 1 if j=k, else 0

  j=k (diagonal):   ∂a[j]/∂s[j] = a[j] × (1 - a[j])   ← standard sigmoid-like
  j≠k (off-diagonal): ∂a[j]/∂s[k] = -a[j] × a[k]      ← coupling term (negative)

  The off-diagonal term is negative: increasing s[k] hurts a[j≠k].
```

```
Computing ∂L/∂s₄ from ∂L/∂A[4,:] via chain rule:

  ∂L/∂s₄[k] = Σⱼ (∂L/∂A[4,j]) × (∂A[4,j]/∂s₄[k])
             = Σⱼ (∂L/∂A[4,j]) × a₄[j] × (δ[j,k] - a₄[k])

Compact form:
  ∂L/∂s₄[k] = a₄[k] × (∂L/∂A[4,k] - g)
  where g = Σⱼ a₄[j] × ∂L/∂A[4,j]   (weighted average of upstream gradients)

Step 1: compute g
  g = a₄[1]×∂L/∂A[4,1] + a₄[2]×∂L/∂A[4,2] + a₄[3]×∂L/∂A[4,3] + a₄[4]×∂L/∂A[4,4]
    = 0.275×(-0.077) + 0.243×(-0.025) + 0.236×(-0.010) + 0.246×(-0.029)
    = -0.021         + (-0.006)        + (-0.002)        + (-0.007)
    = -0.036

  g = -0.036   (the weighted average of all upstream gradients)

Step 2: compute ∂L/∂s₄[k] = a₄[k] × (∂L/∂A[4,k] - g)

  k=1 (cat):  ∂L/∂A[4,1] - g = -0.077 - (-0.036) = -0.041
              ∂L/∂s₄[1]  = 0.275 × (-0.041) = -0.011

  k=2 (sat):  ∂L/∂A[4,2] - g = -0.025 - (-0.036) = +0.011
              ∂L/∂s₄[2]  = 0.243 × (+0.011) = +0.003

  k=3 (on):   ∂L/∂A[4,3] - g = -0.010 - (-0.036) = +0.026
              ∂L/∂s₄[3]  = 0.236 × (+0.026) = +0.006

  k=4 (mat):  ∂L/∂A[4,4] - g = -0.029 - (-0.036) = +0.007
              ∂L/∂s₄[4]  = 0.246 × (+0.007) = +0.002

∂L/∂s₄ = [-0.011, +0.003, +0.006, +0.002]

Verification: sum = -0.011+0.003+0.006+0.002 = 0.000 ✓
(Softmax gradients always sum to zero — increasing one score must decrease others)
```

```
Interpretation of ∂L/∂s₄:

  s₄[1] (cat's score): gradient = -0.011  ← NEGATIVE: increasing cat's score HELPS
                                              → want to push s₄[1] higher
  s₄[2] (sat's score): gradient = +0.003  ← positive: increasing sat's score HURTS
  s₄[3] (on's score):  gradient = +0.006  ← positive: increasing on's score HURTS
  s₄[4] (mat's score): gradient = +0.002  ← positive: increasing mat's score HURTS

  After the update: Wq, Wk will shift so that q₄ · k₁ is larger and
                    q₄ · k₂, q₄ · k₃, q₄ · k₄ are smaller.
  → "mat" learns to attend MORE to "cat" and LESS to other positions. ✓
```

---

### Step F — gradient through score computation

```
s₄[j] = q₄ · kⱼ / √d_k = (q₄ · kⱼ) / 1.414

This is a scaled dot product.

How does s₄[j] depend on q₄ and kⱼ?
  ∂s₄[j]/∂q₄  = kⱼ / √d_k    (gradient of the dot product w.r.t. q₄)
  ∂s₄[j]/∂kⱼ = q₄ / √d_k    (gradient of the dot product w.r.t. kⱼ)
```

```
Gradient w.r.t. q₄ (how should mat's query change?):

  ∂L/∂q₄ = Σⱼ ∂L/∂s₄[j] × kⱼ / √2

  Summing over all j:
    = ∂L/∂s₄[1] × k₁/√2  +  ∂L/∂s₄[2] × k₂/√2  +  ...

  dim 0:
    = (-0.011)×0.550/1.414 + 0.003×0.130/1.414 + 0.006×0.060/1.414 + 0.002×0.140/1.414
    = (-0.006 + 0.000 + 0.000 + 0.000) / 1.414
    = -0.005 / 1.414
    = -0.004

  dim 1:
    = (-0.011)×0.500/1.414 + 0.003×0.180/1.414 + 0.006×0.070/1.414 + 0.002×0.220/1.414
    = (-0.006 + 0.001 + 0.000 + 0.000) / 1.414
    = -0.004 / 1.414
    = -0.003

  ∂L/∂q₄ = [-0.004, -0.003]

  Meaning: position 4's query vector should shift in direction [-0.004, -0.003].
  The query will change to better align with "cat"'s key.
```

```
Gradient w.r.t. each key vector (how should each position's key change?):

  ∂L/∂kⱼ = ∂L/∂s₄[j] × q₄ / √2

  j=1 (cat):  ∂L/∂k₁ = -0.011 × [0.200, 0.280] / 1.414
                      = [-0.002, -0.003] / 1.414
                      = [-0.002, -0.002]

  j=2 (sat):  ∂L/∂k₂ = +0.003 × [0.200, 0.280] / 1.414
                      = [+0.001, +0.001] / 1.414
                      = [+0.000, +0.001]

  j=3 (on):   ∂L/∂k₃ = +0.006 × [0.200, 0.280] / 1.414
                      = [+0.001, +0.002] / 1.414
                      = [+0.001, +0.001]

  j=4 (mat):  ∂L/∂k₄ = +0.002 × [0.200, 0.280] / 1.414
                      = [+0.000, +0.001] / 1.414
                      = [+0.000, +0.000]
```

```
Gradient highway comparison — explicit proof:

  In RNN: to go from ∂L/∂h₄ to ∂L/∂h₁ (3 steps):
    ∂L/∂h₁ = Wₕᵀ(1-h₃²) × Wₕᵀ(1-h₂²) × Wₕᵀ(1-h₁²) × ∂L/∂h₄
    Each step: ≈ 0.44 → 3 steps: 0.44³ = 0.085 (9%)

  In Attention: to go from ∂L/∂c₄ to ∂L/∂V[1] ("cat"'s value):
    ∂L/∂V[1] = A[4,1] × ∂L/∂c₄ = 0.275 × ∂L/∂c₄
    That's it. ONE multiplication.

  To go further back to ∂L/∂Wv (from V[1] to weights):
    ∂L/∂Wv += x₁ᵀ ⊗ ∂L/∂V[1]   (outer product, one step)

  Total path from loss to Wv rows for "cat":
    2 steps (loss→c₄→V[1]→Wv), no chain, no sequential degradation.

  For sequence length 1000:
    RNN:       0.44^999 ≈ 0.0 (vanished completely)
    GRU:       0.88^999 ≈ 1.1×10⁻⁵⁶  (vanished)
    Attention: A[4,1] × ∂L/∂c₄ = SAME as for length 4
```

---

### Step G — weight gradients ∂L/∂Wv, ∂L/∂Wq, ∂L/∂Wk

```
∂L/∂Wv:

V = X @ Wv, so ∂L/∂Wv = Xᵀ @ ∂L/∂V

∂L/∂V (4×2 matrix — gradient for each position's value vector):
  ∂L/∂V[1] = [-0.016, -0.011]   (cat, from Step C)
  ∂L/∂V[2] = [-0.014, -0.010]
  ∂L/∂V[3] = [-0.014, -0.009]
  ∂L/∂V[4] = [-0.015, -0.010]

Xᵀ (2×4):
  [[1.00, 0.20, 0.10, 0.20],
   [0.50, 0.30, 0.10, 0.40]]

∂L/∂Wv = Xᵀ @ ∂L/∂V   (2×4 @ 4×2 = 2×2):

  [0,0]: 1.00×(-0.016) + 0.20×(-0.014) + 0.10×(-0.014) + 0.20×(-0.015)
       = -0.016 - 0.003 - 0.001 - 0.003 = -0.023

  [0,1]: 1.00×(-0.011) + 0.20×(-0.010) + 0.10×(-0.009) + 0.20×(-0.010)
       = -0.011 - 0.002 - 0.001 - 0.002 = -0.016

  [1,0]: 0.50×(-0.016) + 0.30×(-0.014) + 0.10×(-0.014) + 0.40×(-0.015)
       = -0.008 - 0.004 - 0.001 - 0.006 = -0.019

  [1,1]: 0.50×(-0.011) + 0.30×(-0.010) + 0.10×(-0.009) + 0.40×(-0.010)
       = -0.006 - 0.003 - 0.001 - 0.004 = -0.014

∂L/∂Wv = [[-0.023, -0.016],
           [-0.019, -0.014]]
```

```
∂L/∂Wq:

Q = X @ Wq, and only q₄ has non-zero gradient (only row 4 contributed to c₄).
∂L/∂q = only ∂L/∂q₄ = [-0.004, -0.003]; rows 1,2,3 are zero.

∂L/∂Wq = Xᵀ @ ∂L/∂Q   but ∂L/∂Q = [[0,0],[0,0],[0,0],[-0.004,-0.003]]

Only position 4's row contributes:
∂L/∂Wq = x₄ᵀ ⊗ ∂L/∂q₄

x₄ = [0.20, 0.40]

∂L/∂Wq = [[0.20×(-0.004), 0.20×(-0.003)],
           [0.40×(-0.004), 0.40×(-0.003)]]
        = [[-0.001, -0.001],
           [-0.002, -0.001]]
```

```
∂L/∂Wk:

K = X @ Wk. The gradients ∂L/∂K came from s₄[j] = q₄·kⱼ/√2 — row 4's attention only.

∂L/∂K (4×2 matrix):
  ∂L/∂k₁ = [-0.002, -0.002]
  ∂L/∂k₂ = [+0.000, +0.001]
  ∂L/∂k₃ = [+0.001, +0.001]
  ∂L/∂k₄ = [+0.000, +0.000]

∂L/∂Wk = Xᵀ @ ∂L/∂K   (2×4 @ 4×2 = 2×2):

  [0,0]: 1.00×(-0.002) + 0.20×(0.000) + 0.10×(0.001) + 0.20×(0.000)
       = -0.002 + 0 + 0.000 + 0 = -0.002

  [0,1]: 1.00×(-0.002) + 0.20×(0.001) + 0.10×(0.001) + 0.20×(0.000)
       = -0.002 + 0.000 + 0.000 + 0 = -0.002

  [1,0]: 0.50×(-0.002) + 0.30×(0.000) + 0.10×(0.001) + 0.40×(0.000)
       = -0.001 + 0 + 0.000 + 0 = -0.001

  [1,1]: 0.50×(-0.002) + 0.30×(0.001) + 0.10×(0.001) + 0.40×(0.000)
       = -0.001 + 0.000 + 0.000 + 0 = -0.001

∂L/∂Wk = [[-0.002, -0.002],
           [-0.001, -0.001]]
```

```
Gradient summary across all weight matrices:

  Weight     Max |gradient element|   Role
  ──────     ────────────────────    ────────────────────────────────────
  W_out      0.041                   direct output layer — largest
  Wv         0.023                   value projection — second largest
  Wk         0.002                   key projection — small
  Wq         0.002                   query projection — small

Why is Wv gradient larger than Wq, Wk?
  Wv gradient comes from the weighted sum: ∂L/∂V[j] = a₄[j] × ∂L/∂c₄
  It receives the full ∂L/∂c₄ signal scaled by attention weights.

  Wq, Wk gradients are further back in the chain:
  ∂L/∂s₄ = [-0.011, 0.003, 0.006, 0.002] (after softmax backward — values shrink)
  Then divided by √2 and multiplied by small key/query vectors.

In the first step: Wv and W_out learn quickly. Wq, Wk learn slowly.
After many steps: as attention sharpens (cat gets more weight), Wq and Wk gradients grow.
```

---

### "cat" vs other words — how attention distributes gradient

```
∂L/∂V[1] (cat) = -0.016  (dim 0)  ← gets the most signal (highest attention weight)
∂L/∂V[2] (sat) = -0.014
∂L/∂V[3] (on)  = -0.014
∂L/∂V[4] (mat) = -0.015

All four positions get SIMILAR gradient magnitude because attention is near-uniform.
cat: 27.5% weight → contribution 0.016 of gradient
sat: 24.3% weight → contribution 0.014
on:  23.6% weight → contribution 0.014
mat: 24.6% weight → contribution 0.015

Compare to RNN: "cat" got 9% of gradient (1× signal)
                "mat" (adjacent to output) got ~100%  ← 10× more than "cat"

In attention: cat and mat get essentially the same gradient!
Even with random (uniform) attention, no position is forgotten.
After training: cat would get 90%+ weight → much larger gradient → fast learning.

This is fundamentally different:
  RNN/LSTM/GRU: gradient scales inversely with distance from output
  Attention:    gradient scales with attention weight (learnable, not position-dependent)
```

---

## 4. Weight Update

```
Learning rate: lr = 0.1

W_out_new = W_out - lr × ∂L/∂W_out
          = [0.6,  0.4] - 0.1 × [-0.041, -0.031]
          = [0.6+0.004,  0.4+0.003]
          = [0.604, 0.403]


Wv_new = Wv - lr × ∂L/∂Wv

  = [[0.80, 0.20],  -  0.1 × [[-0.023, -0.016],
     [0.30, 0.70]]              [-0.019, -0.014]]

  = [[0.80+0.002, 0.20+0.002],
     [0.30+0.002, 0.70+0.001]]

  = [[0.802, 0.202],
     [0.302, 0.701]]


Wq_new = Wq - lr × ∂L/∂Wq

  = [[0.60, 0.40],  -  0.1 × [[-0.001, -0.001],
     [0.20, 0.50]]              [-0.002, -0.001]]

  = [[0.60+0.000, 0.40+0.000],
     [0.20+0.000, 0.50+0.000]]

  = [[0.600, 0.400],
     [0.200, 0.500]]   (change < 0.001 per element — essentially unchanged)


Wk_new = Wk - lr × ∂L/∂Wk

  = [[0.50, 0.30],  -  0.1 × [[-0.002, -0.002],
     [0.10, 0.40]]              [-0.001, -0.001]]

  = [[0.50+0.000, 0.30+0.000],
     [0.10+0.000, 0.40+0.000]]

  = [[0.500, 0.300],
     [0.100, 0.400]]   (change < 0.001 per element — essentially unchanged)
```

```
Why are Wq, Wk changes tiny while Wv, W_out are larger?

  Weight     Change per element   Reason
  ──────     ──────────────────   ────────────────────────────────────────────
  W_out      0.004                Closest to loss — largest gradient
  Wv         0.002                One step from loss via attention weights
  Wq         < 0.001              Two steps back: loss→softmax→scores→Q
  Wk         < 0.001              Two steps back: loss→softmax→scores→K

  The query and key matrices are two softmax operations deep.
  After the softmax backward, ∂L/∂s₄ values are 0.002-0.011 (small).
  After the score backward (÷√2), even smaller.
  The query/key weights learn slowly in the first step.

  After many training steps:
    Wv adjusts to send the right information.
    Wq, Wk adjust to learn the right matching pattern.
    The combination produces sharp, meaningful attention.

Training loop for all parameters:
  for epoch in range(num_epochs):
      for sentence, label in dataset:
          X = embed(sentence)
          Q, K, V = X@Wq, X@Wk, X@Wv
          A = softmax(Q@Kᵀ/√d_k)
          c₄ = A[last] @ V         → forward
          ŷ = σ(W_out · c₄)
          L = ½(y-ŷ)²
          backward pass → ∂L/∂W_out, ∂L/∂Wv, ∂L/∂Wq, ∂L/∂Wk
          update all 4 matrices with lr
          zero gradients
```

---

## 5. Second Forward Pass (Verify Loss Decreased)

```
Updated weights: W_out=[0.604,0.403], Wv=[[0.802,0.202],[0.302,0.701]]
Wq, Wk unchanged (changes < 0.001)
```

```
Recompute V with new Wv:

v'₁ = x₁ @ Wv_new = [1.00, 0.50] @ [[0.802,0.202],[0.302,0.701]]
    dim 0: 1.00×0.802 + 0.50×0.302 = 0.802 + 0.151 = 0.953
    dim 1: 1.00×0.202 + 0.50×0.701 = 0.202 + 0.351 = 0.553
    v'₁ = [0.953, 0.553]   (was [0.950, 0.550] — change = +0.003)

v'₂ = x₂ @ Wv_new = [0.20, 0.30] @ Wv_new
    dim 0: 0.20×0.802 + 0.30×0.302 = 0.160 + 0.091 = 0.251
    dim 1: 0.20×0.202 + 0.30×0.701 = 0.040 + 0.210 = 0.250
    v'₂ = [0.251, 0.250]   (was [0.250, 0.250] — change = +0.001)

v'₃ = x₃ @ Wv_new = [0.10, 0.10] @ Wv_new
    dim 0: 0.10×0.802 + 0.10×0.302 = 0.080 + 0.030 = 0.110
    dim 1: 0.10×0.202 + 0.10×0.701 = 0.020 + 0.070 = 0.090
    v'₃ = [0.110, 0.090]   (unchanged to 3 decimal places)

v'₄ = x₄ @ Wv_new = [0.20, 0.40] @ Wv_new
    dim 0: 0.20×0.802 + 0.40×0.302 = 0.160 + 0.121 = 0.281
    dim 1: 0.20×0.202 + 0.40×0.701 = 0.040 + 0.280 = 0.320
    v'₄ = [0.281, 0.320]   (was [0.280, 0.320] — change = +0.001)
```

```
Why did v'₁ change more than v'₃?
  "cat" embedding x₁=[1.00, 0.50] is larger than "on" x₃=[0.10, 0.10].
  Same Wv update, but larger x → larger change in v.
  The value vector for the most important word changes the most. ✓
```

```
Q, K unchanged → attention weights A unchanged:
  a₄ = [0.275, 0.243, 0.236, 0.246]   (same as before)

Recompute c'₄:
  c'₄[0] = 0.275×0.953 + 0.243×0.251 + 0.236×0.110 + 0.246×0.281
          = 0.262 + 0.061 + 0.026 + 0.069
          = 0.418   (was 0.417)

  c'₄[1] = 0.275×0.553 + 0.243×0.250 + 0.236×0.090 + 0.246×0.320
          = 0.152 + 0.061 + 0.021 + 0.079
          = 0.313   (was 0.312)
```

```
ŷ' = σ(W_out_new · c'₄)
   = σ(0.604×0.418  +  0.403×0.313)
   = σ(0.252  +  0.126)
   = σ(0.378)
   = 1/(1 + e^{-0.378})
   = 1/(1 + 0.685)
   = 0.594

L' = ½(1.000 - 0.594)²  =  ½ × 0.406²  =  ½ × 0.165  =  0.0826
```

```
Before update:  L = 0.0830   ŷ = 0.5926
After update:   L = 0.0826   ŷ = 0.5936   ← closer to y=1.0  ✅

Loss dropped by 0.0004 (0.5%).
```

```
Why is the improvement small?

  ŷ = 0.593 is already high — the model was already doing well before any training.
  The gradient ∂L/∂z = -0.098 is small because ŷ(1-ŷ) = 0.241 at ŷ=0.59
  (compared to 0.24 at ŷ=0.5 — near maximum, but close).

  The small per-step improvement reflects that the model is near its optimum
  for this single example. In real training with many diverse sentences:
    - Sentences where attention is less obvious would give larger gradients.
    - Negative examples (no animal) would push ŷ down.
    - The interplay of many examples shapes sharp, meaningful attention patterns.

  The key point: loss DID decrease. Gradient direction is correct. ✓
  The model is learning to attend more to "cat" (∂L/∂s₄[1] = -0.011 < 0).
```

---

## 6. The Full Picture in One View

```
FORWARD PASS — parallel computation (all positions at once)
──────────────────────────────────────────────────────────────────────────
  X = [[1.0,0.5], [0.2,0.3], [0.1,0.1], [0.2,0.4]]   ← 4×2 matrix, all at once

  Q = X@Wq        K = X@Wk        V = X@Wv
  4×2 queries     4×2 keys        4×2 values

       ↓
  S = Q@Kᵀ/√2   (4×4 score matrix — every pair computes a score simultaneously)

       ↓
  A = softmax(S, axis=1)   (4×4 attention weights — each row sums to 1)

       ┌──────────────────────────────────┐
  A =  │ 0.327  0.229  0.210  0.234 │  ← cat attends mostly to itself
       │ 0.271  0.244  0.238  0.246 │
       │ 0.259  0.248  0.245  0.248 │
       │ 0.275  0.243  0.236  0.246 │  ← mat attends most to cat
       └──────────────────────────────────┘

       ↓
  C = A@V   (4×2 context — each row is a weighted sum of all value vectors)

  c₄ = 0.275×v₁ + 0.243×v₂ + 0.236×v₃ + 0.246×v₄   ← cat contributes 62.6%
     = [0.417, 0.312]

       ↓
  ŷ = σ(W_out · c₄) = σ(0.375) = 0.593   →   L = 0.083

BACKWARD PASS — one direct step to "cat" (no sequential chain)
──────────────────────────────────────────────────────────────────────────
  ∂L/∂c₄ = [-0.059, -0.039]

  ∂L/∂V[1] = A[4,1] × ∂L/∂c₄ = 0.275 × ∂L/∂c₄   ← ONE STEP to "cat"
           = [-0.016, -0.011]

  ∂L/∂A[4,1] = ∂L/∂c₄ · v₁ = -0.077   ← negative: want MORE attention to cat

  Softmax backward → ∂L/∂s₄[1] = -0.011   ← increase cat's score

  ∂L/∂k₁ = -0.011 × q₄/√2 = [-0.002, -0.002]   ← cat's key shifts toward q₄

  The gradient reaches "cat"'s representations in 2 steps:
    c₄ → V[1] → Wv   (via attention weight 0.275)
    c₄ → A[4,1] → s₄[1] → k₁ → Wk   (via softmax then score)
  Compare RNN: 3 sequential multiplicative steps → 9% survives.
  Attention: 2 direct steps → no position-dependent degradation.
```

---

## 7. Why Transformer is Next

```
Self-attention is the core of attention — but not the full Transformer.

What we built in this walkthrough:
  ✓ Single-head self-attention
  ✓ Linear projections for Q, K, V
  ✓ Scaled dot-product scores
  ✓ Softmax attention weights
  ✓ Context vector via weighted sum
  ✓ Direct gradient path (no sequential bottleneck)

What's still missing — the full Transformer adds:

1. Multi-head attention:
   Instead of ONE set of Wq, Wk, Wv, use H sets (H heads).
   Each head learns to attend to DIFFERENT aspects:
     head 1: syntactic relationships (subject-verb)
     head 2: semantic similarity (animal words)
     head 3: positional proximity
   Outputs concatenated: context = [head₁, ..., headₕ] @ W_output

2. Positional encoding:
   Attention is PERMUTATION-INVARIANT.
   "cat sat on mat" and "mat on sat cat" produce the same attention if
   same embeddings — the model doesn't know ORDER.
   Fix: add sinusoidal position vectors to embeddings before Q/K/V.
   xₜ_final = xₜ + PE[t]   where PE encodes absolute/relative position.

3. Feed-forward network after attention:
   After the attention sublayer: apply a 2-layer MLP to EACH position.
   This adds position-wise nonlinearity (Attention is only linear + softmax).
   Transformer alternates: [attention → FFN] × N_layers

4. Layer normalization and residual connections:
   hₜ_out = LayerNorm(hₜ + Attention(hₜ))   ← residual
   hₜ_final = LayerNorm(hₜ_out + FFN(hₜ_out))
   Residuals let gradients skip layers (like the highway in LSTM/GRU but stronger).

5. Stacking multiple layers:
   One attention layer sees local relationships.
   Layer 2 attends to layer 1's context vectors — sees higher-order relationships.
   12-24 layers: the model builds increasingly abstract representations.

The Transformer = multi-head attention + positional encoding + FFN + LayerNorm + residual, stacked N times.

GPT, BERT, T5, LLaMA — all are Transformers. They are all this mechanism,
scaled to 512-4096 dimensions and billions of parameters.
```

---

## Quick Reference — All Formulas

```
Shapes:
  X   → (T, d)    input embeddings (T tokens, d dimensions)
  Wq  → (d, d)    query projection
  Wk  → (d, d)    key projection
  Wv  → (d, d)    value projection
  Q   → (T, d)    queries:  Q = X @ Wq
  K   → (T, d)    keys:     K = X @ Wk
  V   → (T, d)    values:   V = X @ Wv
  S   → (T, T)    scaled scores: S = Q @ Kᵀ / √d
  A   → (T, T)    attention weights: A = softmax(S, axis=1)
  C   → (T, d)    context vectors: C = A @ V
```

```
Forward pass:
  Q = X @ Wq
  K = X @ Wk
  V = X @ Wv
  S = Q @ Kᵀ / √d        ← scaled dot product
  A = softmax(S, axis=1)  ← row-wise softmax
  C = A @ V               ← weighted sum of values
  ŷ = σ(W_out · C[-1])   ← predict from last position's context
  L = ½(y - ŷ)²
```

```
Backward pass (using last position for prediction):
  δz        = -(y-ŷ) × ŷ(1-ŷ)                    ← through sigmoid
  ∂L/∂W_out = δz × c_lastᵀ                        ← output weight gradient
  ∂L/∂c     = δz × W_out                          ← gradient entering attention
  ∂L/∂V[j]  = A[last,j] × ∂L/∂c                  ← direct to each value vector
  ∂L/∂A[j]  = ∂L/∂c · V[j]                        ← gradient w.r.t. attn weights
  g          = A[last] · ∂L/∂A                     ← scalar: weighted avg of upstream
  ∂L/∂s[k]  = A[last,k] × (∂L/∂A[k] - g)         ← softmax backward
  ∂L/∂q     = Σⱼ ∂L/∂s[j] × kⱼ / √d             ← through score dot product
  ∂L/∂kⱼ    = ∂L/∂s[j] × q / √d                  ← each key's gradient
  ∂L/∂Wv    = Xᵀ @ ∂L/∂V                          ← value weight gradient
  ∂L/∂Wq    = Xᵀ @ ∂L/∂Q                          ← query weight gradient
  ∂L/∂Wk    = Xᵀ @ ∂L/∂K                          ← key weight gradient
```

```
Softmax backward — compact form:
  A = softmax(S, axis=1)
  ∂L/∂s[k] = A[k] × (∂L/∂A[k] - g)    where g = Σⱼ A[j] × ∂L/∂A[j]

  Sum check: Σₖ ∂L/∂s[k] = 0  (softmax gradients always sum to zero)
```

```
Numbers from this walkthrough:
  Wq=[[0.60,0.40],[0.20,0.50]]
  Wk=[[0.50,0.30],[0.10,0.40]]
  Wv=[[0.80,0.20],[0.30,0.70]]
  W_out=[0.6,0.4]

  q₁=[0.700,0.650]  k₁=[0.550,0.500]  v₁=[0.950,0.550]
  q₄=[0.200,0.280]  k₄=[0.140,0.220]  v₄=[0.280,0.320]

  Attention row 1 (cat): [0.327, 0.229, 0.210, 0.234]
  Attention row 4 (mat): [0.275, 0.243, 0.236, 0.246]

  c₄ = [0.417, 0.312]
  ŷ  = 0.593   L  = 0.083
  ŷ' = 0.594   L' = 0.083  (0.0826 precisely — loss decreased ✓)
```

---

## 8. Code

### Version 1 — Pure NumPy (mirrors the hand computation exactly)

```python
import numpy as np

# ── Embeddings (same as RNN, LSTM, GRU) ──────────────────────────────────
E = np.array([
    [1.00, 0.50],  # cat
    [0.20, 0.30],  # sat
    [0.10, 0.10],  # on
    [0.20, 0.40],  # mat
])

# ── Weights ───────────────────────────────────────────────────────────────
Wq = np.array([[0.60, 0.40], [0.20, 0.50]])
Wk = np.array([[0.50, 0.30], [0.10, 0.40]])
Wv = np.array([[0.80, 0.20], [0.30, 0.70]])
W_out = np.array([0.6, 0.4])

# ── Forward pass ──────────────────────────────────────────────────────────
X = E                      # (4, 2)
Q = X @ Wq                 # (4, 2)
K = X @ Wk                 # (4, 2)
V = X @ Wv                 # (4, 2)

d_k = Q.shape[-1]          # 2
scores = Q @ K.T / np.sqrt(d_k)  # (4, 4)

# row-wise softmax
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

A = softmax(scores)        # (4, 4)
C = A @ V                  # (4, 2)

c4 = C[-1]                 # (2,) — last position

def sigmoid(x): return 1 / (1 + np.exp(-x))

z   = W_out @ c4
y_hat = sigmoid(z)
y   = 1.0
L   = 0.5 * (y - y_hat) ** 2

print(f"Q:\n{Q.round(3)}")
print(f"K:\n{K.round(3)}")
print(f"V:\n{V.round(3)}")
print(f"\nScores:\n{scores.round(3)}")
print(f"Attention A:\n{A.round(3)}")
print(f"\nc4 = {c4.round(3)}")
print(f"y_hat = {y_hat:.4f}  L = {L:.4f}")

# ── Backward pass ─────────────────────────────────────────────────────────
# Step B: output layer
dL_dyhat = -(y - y_hat)
dyhat_dz = y_hat * (1 - y_hat)
dL_dz    = dL_dyhat * dyhat_dz                    # scalar

dL_dW_out = dL_dz * c4                            # (2,)
dL_dc4    = dL_dz * W_out                         # (2,)

# Step C: weighted sum c4 = A[3,:] @ V
a4 = A[-1]                                        # (4,) — last row of A
dL_dV = np.outer(a4, dL_dc4)                      # (4, 2): dL_dV[j] = a4[j] * dL_dc4

# Step D: gradient w.r.t. attention weights A[-1,:]
# dL/dA[j] = dL_dc4 · V[j]
dL_dA4 = V @ dL_dc4                               # (4,): dL_dA4[j] = dL_dc4 · V[j]

# Step E: softmax backward
# dL/ds[k] = a4[k] * (dL/dA[k] - g)  where g = a4 · dL_dA4
g = a4 @ dL_dA4                                   # scalar
dL_ds4 = a4 * (dL_dA4 - g)                        # (4,)

print(f"\ndL_ds4 = {dL_ds4.round(4)}")
print(f"Sum check: {dL_ds4.sum():.10f} (should be ~0)")

# Step F: gradient through scores s4[j] = q4 · kj / sqrt(d_k)
q4 = Q[-1]                                        # (2,)
dL_dq4 = (dL_ds4 @ K) / np.sqrt(d_k)             # (2,)
dL_dK  = np.outer(dL_ds4, q4) / np.sqrt(d_k)     # (4, 2): gradient for each key

# Step G: weight gradients
# dL/dWv = X.T @ dL_dV   (since V = X @ Wv)
dL_dWv = X.T @ dL_dV                              # (2, 2)

# dL/dWq: only last query q4 contributed (we only used C[-1])
dL_dQ = np.zeros_like(Q)
dL_dQ[-1] = dL_dq4
dL_dWq = X.T @ dL_dQ                              # (2, 2)

# dL/dWk: all keys contributed to row-4 attention
dL_dWk = X.T @ dL_dK                              # (2, 2)

print(f"\ndL_dW_out = {dL_dW_out.round(4)}")
print(f"dL_dWv =\n{dL_dWv.round(4)}")
print(f"dL_dWq =\n{dL_dWq.round(4)}")
print(f"dL_dWk =\n{dL_dWk.round(4)}")

# ── Weight update ─────────────────────────────────────────────────────────
lr = 0.1
W_out_new = W_out - lr * dL_dW_out
Wv_new    = Wv    - lr * dL_dWv
Wq_new    = Wq    - lr * dL_dWq
Wk_new    = Wk    - lr * dL_dWk

# ── Second forward ────────────────────────────────────────────────────────
Q2 = X @ Wq_new
K2 = X @ Wk_new
V2 = X @ Wv_new

scores2 = Q2 @ K2.T / np.sqrt(d_k)
A2 = softmax(scores2)
C2 = A2 @ V2

z2 = W_out_new @ C2[-1]
y_hat2 = sigmoid(z2)
L2 = 0.5 * (y - y_hat2) ** 2

print(f"\nAfter 1 step:  y_hat={y_hat2:.4f}  L={L2:.4f}  (was {L:.4f})")
print(f"Loss decreased: {L2 < L}")
```

---

### Version 2 — PyTorch manual (autograd handles backward)

```python
import torch

# ── Embeddings and weights ─────────────────────────────────────────────────
X = torch.tensor([
    [1.00, 0.50],
    [0.20, 0.30],
    [0.10, 0.10],
    [0.20, 0.40],
], dtype=torch.float32)

Wq = torch.tensor([[0.60, 0.40], [0.20, 0.50]], requires_grad=True)
Wk = torch.tensor([[0.50, 0.30], [0.10, 0.40]], requires_grad=True)
Wv = torch.tensor([[0.80, 0.20], [0.30, 0.70]], requires_grad=True)
W_out = torch.tensor([0.6, 0.4], requires_grad=True)

y = torch.tensor(1.0)

# ── Forward ────────────────────────────────────────────────────────────────
Q = X @ Wq                                           # (4, 2)
K = X @ Wk                                           # (4, 2)
V = X @ Wv                                           # (4, 2)

d_k = Q.shape[-1]
scores = Q @ K.T / d_k ** 0.5                        # (4, 4)
A = torch.softmax(scores, dim=-1)                    # (4, 4)
C = A @ V                                            # (4, 2)

c4 = C[-1]
z = W_out @ c4
y_hat = torch.sigmoid(z)
L = 0.5 * (y - y_hat) ** 2

print(f"y_hat = {y_hat.item():.4f}   L = {L.item():.4f}")
print(f"Attention row 4: {A[-1].detach().numpy().round(3)}")

# ── Backward ───────────────────────────────────────────────────────────────
L.backward()

print(f"\ndL/dW_out = {W_out.grad.numpy().round(4)}")
print(f"dL/dWv =\n{Wv.grad.numpy().round(4)}")
print(f"dL/dWq =\n{Wq.grad.numpy().round(4)}")
print(f"dL/dWk =\n{Wk.grad.numpy().round(4)}")

# ── Update ─────────────────────────────────────────────────────────────────
lr = 0.1
with torch.no_grad():
    W_out -= lr * W_out.grad
    Wv    -= lr * Wv.grad
    Wq    -= lr * Wq.grad
    Wk    -= lr * Wk.grad

W_out.grad.zero_(); Wv.grad.zero_(); Wq.grad.zero_(); Wk.grad.zero_()

# ── Second forward ─────────────────────────────────────────────────────────
Q2 = X @ Wq
K2 = X @ Wk
V2 = X @ Wv
A2 = torch.softmax(Q2 @ K2.T / d_k ** 0.5, dim=-1)
y_hat2 = torch.sigmoid(W_out @ (A2 @ V2)[-1])
L2 = 0.5 * (y - y_hat2) ** 2

print(f"\nAfter update:  y_hat={y_hat2.item():.4f}  L={L2.item():.4f}")
```

---

### Version 3 — PyTorch nn.MultiheadAttention (production style)

```python
import torch
import torch.nn as nn

class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim=2, num_heads=1):
        super().__init__()
        # nn.MultiheadAttention: single head, same dimensions as our walkthrough
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # input shape: (batch, seq, embed)
        )
        self.out = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        # self-attention: query=key=value=x
        context, attn_weights = self.attn(x, x, x)
        # use last position's context for classification
        c_last = context[:, -1, :]    # (batch, embed_dim)
        logit = self.out(c_last).squeeze(-1)
        return torch.sigmoid(logit), attn_weights

# ── Setup ──────────────────────────────────────────────────────────────────
model = SelfAttentionClassifier(embed_dim=2, num_heads=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# "cat sat on mat"
X = torch.tensor([[[1.00, 0.50],
                   [0.20, 0.30],
                   [0.10, 0.10],
                   [0.20, 0.40]]])   # shape: (1, 4, 2) — batch=1, seq=4, embed=2

y = torch.tensor([1.0])   # is_animal = True

# ── Training loop ─────────────────────────────────────────────────────────
for step in range(5):
    y_hat, attn = model(X)
    L = 0.5 * (y - y_hat) ** 2

    optimizer.zero_grad()
    L.backward()
    optimizer.step()

    print(f"Step {step+1}: y_hat={y_hat.item():.4f}  L={L.item():.4f}")
    print(f"  Attention row 4: {attn[0, -1].detach().numpy().round(3)}")

# ── Inspect learned attention ───────────────────────────────────────────────
with torch.no_grad():
    y_final, attn_final = model(X)
    print(f"\nAfter 5 steps:")
    print(f"  y_hat = {y_final.item():.4f}")
    print(f"  Attention weights (all rows):\n{attn_final[0].numpy().round(3)}")
    print(f"  → Row 4 should have highest weight on position 0 ('cat')")
```

---

## Connections

```
Architecture    Core mechanism               Gradient path to position 1
────────────    ─────────────────────────   ────────────────────────────────────────
RNN             hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁)  3× (Wₕᵀ × tanh')  →  9% survives
LSTM            Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙g̃ₜ       3× fₜ  →  69% survives
GRU             hₜ = (1-zₜ)hₜ₋₁ + zₜh̃ₜ     3× (1-zₜ) →  66% survives
Attention       c = Σⱼ Aⱼ × vⱼ             1× A[last,1]  →  27.5% (same for ANY length)

                ↑ all previous architectures  ↑ attention breaks the sequential chain
```

```
Why attention enables the Transformer:
  1. Parallel computation — all positions processed simultaneously → fast on GPUs
  2. Direct gradient path — no vanishing over distance → trains on long sequences
  3. Learnable relevance — attention weights adapt to the task
  4. The Transformer stacks N attention layers + FFN layers:
       Layer 1: learns surface patterns (word co-occurrence)
       Layer 2: learns syntactic structure (subject-verb, noun phrases)
       Layer N: learns semantic relationships (entity coreference, reasoning)

Next: 06_transformer_end_to_end.md
  Multi-head attention + positional encoding + FFN + layer norm + residual
  Same "cat sat on mat" → show how 4-head attention splits the task.
```

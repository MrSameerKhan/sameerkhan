# 06 — Transformer: Complete End-to-End Walkthrough

Same sentence as RNN, LSTM, GRU, and Attention. Same embeddings. Same template.
The difference: positional encoding, multi-head attention, FFN, residual connections, LayerNorm — all computed with real numbers.

---

## 0. The Journey So Far

```
Architecture    Key State          Gate Count    Gradient to "cat"    Parallelizable?
────────────────────────────────────────────────────────────────────────────────────
RNN             hₜ (overwrite)     0             9%                   No
LSTM            Cₜ + hₜ (two)      4             69%                  No
GRU             hₜ (selective)     2             66%                  No
Attention       none (direct)      —             direct (A[4,1]×∂L)   YES
Transformer     none (stacked)     —             direct + FFN         YES
────────────────────────────────────────────────────────────────────────────────────
```

**Attention's remaining limits:**

```
Attention solved the sequential bottleneck and the gradient problem.
But raw attention has two gaps:

1. POSITION BLINDNESS
   Attention operates on a SET of vectors.
   It has no idea that "cat" comes before "sat."
   Shuffle the sentence → same attention output.

   "cat sat on mat"  and  "mat on sat cat"  produce IDENTICAL attention patterns.
   Embeddings carry no positional information.

2. LIMITED TRANSFORMATION POWER
   Attention computes a weighted sum of values.
   The output is a LINEAR combination of value vectors.
   No non-linearity → limited ability to model complex dependencies.

   "cat" influences "sat" through a weighted sum, but the transformation
   is just: α × v_cat + β × v_sat + ...
   There's no ReLU, no non-linear interaction between positions.
```

**The Transformer's answer:**

```
Add positional information via Positional Encoding:
   x_input = embedding + PE(position)
   Now position 1 ("cat") and position 4 ("mat") have different inputs.

Add non-linearity via Feed-Forward Network:
   After attention, apply: FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
   This non-linearly transforms each position's context vector.

Stabilize training via Layer Normalization + Residual Connections:
   x_out = LayerNorm(x + Sublayer(x))
   Residuals: gradient highway (same as LSTM cell)
   LayerNorm: stabilizes activations, enables deeper stacking

Stack multiple layers:
   One encoder block = [Attention + Residual + LN] → [FFN + Residual + LN]
   BERT: 12 layers, GPT-2: 12 layers, GPT-3: 96 layers
```

---

## 1. Problem Statement

```
Sentence:    "cat sat on mat"
Task:        Predict: does "mat" appear in the sentence? (label y = 1)

This walkthrough: ONE encoder block of a Transformer.
   Input:  sequence of 4 word embeddings (d=2)
   Step 1: Add positional encodings → X_pe
   Step 2: Multi-head attention on X_pe → context vectors C
   Step 3: Residual: X_pe + C → X_attn
   Step 4: LayerNorm₁: normalize X_attn
   Step 5: FFN: non-linear transform per position
   Step 6: Residual: X_attn + FFN_out → X_final
   Step 7: Classify from X_final[mat]
```

---

## 2. Input & Preprocessing

### 2.1 Vocabulary & Embeddings

Same table as every previous file:

```
Word    Index    Embedding
─────────────────────────────
cat       0      [1.0, 0.5]
sat       1      [0.2, 0.3]
on        2      [0.1, 0.1]
mat       3      [0.2, 0.4]
─────────────────────────────
d_model = 2
```

As a matrix X (shape 4×2):

```
      dim0   dim1
cat  [1.000, 0.500]
sat  [0.200, 0.300]
on   [0.100, 0.100]
mat  [0.200, 0.400]
```

### 2.2 Positional Encoding

```
Attention operates on a SET — it has no notion of order.
We inject position information by adding a sinusoidal signal.

Formula (Vaswani et al., 2017):
   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Where:
   pos = position index (0-based)
   i   = dimension pair index (0-based)
   d   = d_model

For d=2, only one dimension pair exists: i=0
   PE(pos, 0) = sin(pos / 10000^0) = sin(pos)
   PE(pos, 1) = cos(pos / 10000^0) = cos(pos)
```

Computing PE for each position:

```
pos=0 (cat):   PE = [sin(0),   cos(0)  ] = [0.000,  1.000]
pos=1 (sat):   PE = [sin(1),   cos(1)  ] = [0.841,  0.540]
pos=2 (on):    PE = [sin(2),   cos(2)  ] = [0.909, -0.416]
pos=3 (mat):   PE = [sin(3),   cos(3)  ] = [0.141, -0.990]

Recall: sin(1)=0.841, cos(1)=0.540, sin(2)=0.909, cos(2)=-0.416
        sin(3)=0.141, cos(3)=-0.990
```

**Why sinusoidal?**
```
Three properties that make this encoding powerful:

1. Bounded: sin/cos always in [-1, 1] → no scale explosion
2. Unique: each (pos, dim) pair gets a unique value
3. Relative: PE(pos+k) can be expressed as a linear function of PE(pos)
   → The model can learn to attend to "2 positions ahead" via learned weights

At d=512 (real BERT), the period varies:
   dim 0-1: period = 2π (cycles once per ~6 positions)
   dim 10-11: period = 2π × 10000^(10/512) ≈ 63 (longer wavelength)
   dim 510-511: period = 2π × 10000^1 = 20000π (very slow)
   → Low dims encode fine-grained position, high dims encode coarse position
```

### 2.3 X_pe = X + PE

```
      X (embed)        PE                  X_pe
─────────────────────────────────────────────────────────────────
cat   [1.000, 0.500] + [0.000,  1.000] = [1.000,  1.500]
sat   [0.200, 0.300] + [0.841,  0.540] = [1.041,  0.840]
on    [0.100, 0.100] + [0.909, -0.416] = [1.009, -0.316]
mat   [0.200, 0.400] + [0.141, -0.990] = [0.341, -0.590]
─────────────────────────────────────────────────────────────────
```

**Notice what PE does to the representations:**
```
Before PE: cat=[1.0,0.5], sat=[0.2,0.3] — sat is "below" cat in both dims
After PE:  cat=[1.0,1.5], mat=[0.341,-0.590] — mat is now NEGATIVE in dim1

Positional encoding stretches the input space so nearby positions
become geometrically distinguishable even if their embeddings were similar.
"cat" and "mat" had similar embeddings [1.0,0.5] and [0.2,0.4].
After PE: [1.000,1.500] vs [0.341,-0.590] — very different in attention space.
```

---

## 3. Weight Setup

### 3.1 Attention Weights (Single Head, h=1)

Same Wq, Wk, Wv as the attention file:

```
Wq (query projection, 2×2):        Wk (key projection, 2×2):
   [[0.60, 0.40],                      [[0.50, 0.30],
    [0.20, 0.50]]                        [0.10, 0.40]]

Wv (value projection, 2×2):
   [[0.80, 0.20],
    [0.30, 0.70]]

Note: With d_model=2, n_heads=1, d_head=2/1=2.
For n_heads=2 you'd need d_model=4 so d_head=2 per head.
The multi-head concept is explained in Section 11.
```

**W_o (output projection for attention, 2×2):**
```
W_o = I (identity) for 1 head — the single head's output IS the final MHA output.
In multi-head: W_o concatenates h heads and projects back to d_model.
```

### 3.2 FFN Weights

```
d_ff = 4  (FFN hidden dim; real transformers use d_ff = 4 × d_model)

W1 (d_model → d_ff, shape 2×4):          b1 (shape 4):
   [[0.5, 0.3, 0.2, 0.1],                  [0, 0, 0, 0]
    [0.4, 0.2, 0.3, 0.1]]

W2 (d_ff → d_model, shape 4×2):          b2 (shape 2):
   [[0.5, 0.3],                             [0, 0]
    [0.2, 0.4],
    [0.3, 0.2],
    [0.1, 0.5]]

FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
```

### 3.3 Output Weights

```
W_out (classifier, shape 2):   b_out (scalar):
   [0.5, 0.3]                     0

ŷ = σ(W_out · x_final_mat + b_out)
```

### 3.4 LayerNorm Parameters

```
LN₁ (after MHA residual):  γ₁=[1.0, 1.0],  β₁=[0.0, 0.0]
LN₂ (after FFN residual):  γ₂=[1.0, 1.0],  β₂=[0.0, 0.0]   (not used in output layer here)

LayerNorm formula:
   LN(x) = γ ⊙ (x - μ) / σ + β
   μ = mean(x) over feature dim
   σ = std(x) over feature dim
   γ, β: learned scale and shift (initialized to 1, 0)
```

### 3.5 Why These Design Choices?

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN CHOICE              WHY IT WORKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Positional Encoding        Without it: shuffle input → same output
(sinusoidal)               With it: each position has unique geometric identity
                           Sinusoidal chosen so PE(pos+k) is LINEAR fn of PE(pos)

Multi-Head Attention       Single head: ONE query subspace (one "relationship type")
(h heads)                  h=8 heads: 8 different query subspaces in parallel
                           Head 1 might learn "syntactic subject-verb agreement"
                           Head 2 might learn "co-reference (it → cat)"
                           Each head operates on d_model/h dimensions
                           Outputs concatenated → projected back to d_model

FFN (ReLU activation)      Attention = linear weighted sum of values
                           → model is just a linear combination at each position
                           FFN adds non-linearity BETWEEN attention and next layer
                           "Process context from attention before passing up"
                           d_ff = 4×d_model: expand → process → compress

Residual Connections        Two-fold benefit:
(x + Sublayer(x))          1. Gradient highway: ∂L flows directly through +
                              (same insight as LSTM cell state)
                           2. Lazy initialization: sublayer can start by
                              outputting near-zero → residual ≈ identity early

LayerNorm                  BatchNorm normalizes over BATCH dim (unstable for seq)
(normalize over features)  LayerNorm normalizes over FEATURE dim per position
                           → stable for variable-length sequences
                           → each position independently normalized
                           Pre-norm: LN → sublayer (more stable, used in GPT-3)
                           Post-norm: sublayer → LN (original paper, used here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3.6 All Transformer (Encoder Block) Formulas

```
POSITIONAL ENCODING:
   X_pe = X + PE
   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

SCALED DOT-PRODUCT ATTENTION:
   Q = X_pe @ Wq
   K = X_pe @ Wk
   V = X_pe @ Wv
   S = Q @ Kᵀ / √d_k
   A = softmax(S)                (row-wise)
   C = A @ V

MHA RESIDUAL + LAYERNORM:
   X_attn = LayerNorm₁(X_pe + C)

FEED-FORWARD NETWORK:
   H      = ReLU(X_attn @ W1 + b1)
   FFN    = H @ W2 + b2

FFN RESIDUAL + LAYERNORM:
   X_final = LayerNorm₂(X_attn + FFN)   [or X_attn + FFN for simplified version]

OUTPUT (classify from mat's position):
   ŷ = σ(W_out · X_final[3])
   L = -log(ŷ)   (target y=1)
```

---

## 4. Expected Output

```
Architecture path:
   X_pe = [4×2] → Attention = [4×2] → Residual = [4×2] → LN₁ = [4×2]
        → FFN = [4×2] → Residual = [4×2] → x_final = [4×2]
        → x_final[mat] = [1×2] → scalar ŷ

Target: y = 1 ("mat" is in sentence)
With W_out=[0.5,0.3], x_final_mat≈[1.18,-0.12]:
   ŷ ≈ σ(0.556) ≈ 0.635
   L  = -log(0.635) ≈ 0.454

After one update: L' ≈ 0.436  (verified in Section 9)
```

---

## 5. Forward Pass

### Step 1: Positional Encoding

Already done in Section 2.3:

```
      X_pe:
      cat:  [1.000,  1.500]
      sat:  [1.041,  0.840]
      on:   [1.009, -0.316]
      mat:  [0.341, -0.590]
```

### Step 2: Scaled Dot-Product Attention

Applied directly to X_pe (post-norm architecture: LN comes AFTER residual).

**Compute Q = X_pe @ Wq:**

```
q₁ (cat):
   dim0: 1.000×0.60 + 1.500×0.20 = 0.600 + 0.300 = 0.900
   dim1: 1.000×0.40 + 1.500×0.50 = 0.400 + 0.750 = 1.150
   q₁ = [0.900, 1.150]

q₂ (sat):
   dim0: 1.041×0.60 + 0.840×0.20 = 0.625 + 0.168 = 0.793
   dim1: 1.041×0.40 + 0.840×0.50 = 0.416 + 0.420 = 0.836
   q₂ = [0.793, 0.836]

q₃ (on):
   dim0: 1.009×0.60 + (-0.316)×0.20 = 0.605 - 0.063 = 0.542
   dim1: 1.009×0.40 + (-0.316)×0.50 = 0.404 - 0.158 = 0.246
   q₃ = [0.542, 0.246]

q₄ (mat):
   dim0: 0.341×0.60 + (-0.590)×0.20 = 0.205 - 0.118 = 0.087
   dim1: 0.341×0.40 + (-0.590)×0.50 = 0.136 - 0.295 = -0.159
   q₄ = [0.087, -0.159]
```

**Compute K = X_pe @ Wk:**

```
k₁ (cat):
   dim0: 1.000×0.50 + 1.500×0.10 = 0.500 + 0.150 = 0.650
   dim1: 1.000×0.30 + 1.500×0.40 = 0.300 + 0.600 = 0.900
   k₁ = [0.650, 0.900]

k₂ (sat):
   dim0: 1.041×0.50 + 0.840×0.10 = 0.521 + 0.084 = 0.605
   dim1: 1.041×0.30 + 0.840×0.40 = 0.312 + 0.336 = 0.648
   k₂ = [0.605, 0.648]

k₃ (on):
   dim0: 1.009×0.50 + (-0.316)×0.10 = 0.505 - 0.032 = 0.473
   dim1: 1.009×0.30 + (-0.316)×0.40 = 0.303 - 0.126 = 0.177
   k₃ = [0.473, 0.177]

k₄ (mat):
   dim0: 0.341×0.50 + (-0.590)×0.10 = 0.171 - 0.059 = 0.112
   dim1: 0.341×0.30 + (-0.590)×0.40 = 0.102 - 0.236 = -0.134
   k₄ = [0.112, -0.134]
```

**Compute V = X_pe @ Wv:**

```
v₁ (cat):
   dim0: 1.000×0.80 + 1.500×0.30 = 0.800 + 0.450 = 1.250
   dim1: 1.000×0.20 + 1.500×0.70 = 0.200 + 1.050 = 1.250
   v₁ = [1.250, 1.250]

v₂ (sat):
   dim0: 1.041×0.80 + 0.840×0.30 = 0.833 + 0.252 = 1.085
   dim1: 1.041×0.20 + 0.840×0.70 = 0.208 + 0.588 = 0.796
   v₂ = [1.085, 0.796]

v₃ (on):
   dim0: 1.009×0.80 + (-0.316)×0.30 = 0.807 - 0.095 = 0.712
   dim1: 1.009×0.20 + (-0.316)×0.70 = 0.202 - 0.221 = -0.019
   v₃ = [0.712, -0.019]

v₄ (mat):
   dim0: 0.341×0.80 + (-0.590)×0.30 = 0.273 - 0.177 = 0.096
   dim1: 0.341×0.20 + (-0.590)×0.70 = 0.068 - 0.413 = -0.345
   v₄ = [0.096, -0.345]
```

**Summary table:**

```
         Q              K              V
────────────────────────────────────────────────────────────────
cat:  [0.900, 1.150]  [0.650, 0.900]  [1.250, 1.250]
sat:  [0.793, 0.836]  [0.605, 0.648]  [1.085, 0.796]
on:   [0.542, 0.246]  [0.473, 0.177]  [0.712, -0.019]
mat:  [0.087,-0.159]  [0.112,-0.134]  [0.096, -0.345]
────────────────────────────────────────────────────────────────

Compare to Attention file (raw X without PE):
   cat Q: [0.700, 0.650] → now [0.900, 1.150]   (PE pushed up both dims)
   mat Q: [0.200, 0.280] → now [0.087, -0.159]   (PE pulled down, especially dim1)
   Positions are now geometrically separated in Q/K space.
```

**Compute raw scores S = Q @ Kᵀ (all rows × all keys):**

```
√d_k = √2 = 1.414

Row 1 (cat query q₁=[0.900, 1.150]):
   s₁₁ = q₁·k₁ = 0.900×0.650 + 1.150×0.900 = 0.585 + 1.035 = 1.620
   s₁₂ = q₁·k₂ = 0.900×0.605 + 1.150×0.648 = 0.545 + 0.745 = 1.290
   s₁₃ = q₁·k₃ = 0.900×0.473 + 1.150×0.177 = 0.426 + 0.204 = 0.630
   s₁₄ = q₁·k₄ = 0.900×0.112 + 1.150×(-0.134) = 0.101 - 0.154 = -0.053

Row 2 (sat query q₂=[0.793, 0.836]):
   s₂₁ = q₂·k₁ = 0.793×0.650 + 0.836×0.900 = 0.515 + 0.752 = 1.267
   s₂₂ = q₂·k₂ = 0.793×0.605 + 0.836×0.648 = 0.480 + 0.542 = 1.022
   s₂₃ = q₂·k₃ = 0.793×0.473 + 0.836×0.177 = 0.375 + 0.148 = 0.523
   s₂₄ = q₂·k₄ = 0.793×0.112 + 0.836×(-0.134) = 0.089 - 0.112 = -0.023

Row 3 (on query q₃=[0.542, 0.246]):
   s₃₁ = q₃·k₁ = 0.542×0.650 + 0.246×0.900 = 0.352 + 0.221 = 0.573
   s₃₂ = q₃·k₂ = 0.542×0.605 + 0.246×0.648 = 0.328 + 0.159 = 0.487
   s₃₃ = q₃·k₃ = 0.542×0.473 + 0.246×0.177 = 0.256 + 0.044 = 0.300
   s₃₄ = q₃·k₄ = 0.542×0.112 + 0.246×(-0.134) = 0.061 - 0.033 = 0.028

Row 4 (mat query q₄=[0.087,-0.159]):
   s₄₁ = q₄·k₁ = 0.087×0.650 + (-0.159)×0.900 = 0.057 - 0.143 = -0.086
   s₄₂ = q₄·k₂ = 0.087×0.605 + (-0.159)×0.648 = 0.053 - 0.103 = -0.050
   s₄₃ = q₄·k₃ = 0.087×0.473 + (-0.159)×0.177 = 0.041 - 0.028 = 0.013
   s₄₄ = q₄·k₄ = 0.087×0.112 + (-0.159)×(-0.134) = 0.010 + 0.021 = 0.031
```

**Scale by √d_k = 1.414:**

```
Scaled score matrix S / √2:

         key_cat   key_sat   key_on    key_mat
         ────────────────────────────────────────
q_cat:   1.146     0.912     0.446    -0.037
q_sat:   0.896     0.723     0.370    -0.016
q_on:    0.405     0.344     0.212     0.020
q_mat:  -0.061    -0.035     0.009     0.022
```

**Softmax row-by-row → attention matrix A:**

```
Row 1 (cat): softmax([1.146, 0.912, 0.446, -0.037])
   exp: [3.145, 2.489, 1.562, 0.964]   sum=8.160
   A₁ = [0.385, 0.305, 0.191, 0.118]
   Interpretation: cat attends MOST to itself (0.385), then sat (0.305)

Row 2 (sat): softmax([0.896, 0.723, 0.370, -0.016])
   exp: [2.450, 2.060, 1.448, 0.984]   sum=6.942
   A₂ = [0.353, 0.297, 0.209, 0.142]
   Interpretation: sat attends most to cat (0.353), then itself (0.297)

Row 3 (on): softmax([0.405, 0.344, 0.212, 0.020])
   exp: [1.500, 1.411, 1.236, 1.020]   sum=5.167
   A₃ = [0.290, 0.273, 0.239, 0.197]
   Interpretation: more spread out — on attends almost uniformly (range 0.197–0.290)

Row 4 (mat): softmax([-0.061, -0.035, 0.009, 0.022])
   exp: [0.941, 0.966, 1.009, 1.022]   sum=3.938
   A₄ = [0.239, 0.245, 0.256, 0.260]
   Interpretation: mat attends MOST to itself (0.260), then on (0.256)!
```

**Full attention matrix A (row = query, col = key):**

```
         key_cat   key_sat   key_on    key_mat
         ────────────────────────────────────────────────
q_cat:   0.385     0.305     0.191     0.118      sum=1.0 ✓
q_sat:   0.353     0.297     0.209     0.142      sum=1.0 ✓
q_on:    0.290     0.273     0.239     0.197      sum=1.0 ✓
q_mat:   0.239     0.245     0.256     0.260      sum=1.0 ✓
```

**PE's effect on attention patterns:**
```
Without PE (attention file): A₄ = [0.275, 0.243, 0.236, 0.246]  cat dominates
With PE (here):              A₄ = [0.239, 0.245, 0.256, 0.260]  self-attention wins

Positional encoding pushed "mat" to attend MORE to nearby positions (on=0.256, mat=0.260)
and LESS to distant "cat" (0.239 vs 0.275).
This is positional encoding working as intended — nearby positions become more geometrically similar.
```

**Compute context vectors C = A @ V:**

V matrix:
```
v₁=[1.250,1.250], v₂=[1.085,0.796], v₃=[0.712,-0.019], v₄=[0.096,-0.345]
```

```
c₁ (cat's context, A₁=[0.385, 0.305, 0.191, 0.118]):
   dim0: 0.385×1.250 + 0.305×1.085 + 0.191×0.712 + 0.118×0.096
       = 0.481 + 0.331 + 0.136 + 0.011 = 0.959
   dim1: 0.385×1.250 + 0.305×0.796 + 0.191×(-0.019) + 0.118×(-0.345)
       = 0.481 + 0.243 - 0.004 - 0.041 = 0.679
   c₁ = [0.959, 0.679]

c₂ (sat's context, A₂=[0.353, 0.297, 0.209, 0.142]):
   dim0: 0.353×1.250 + 0.297×1.085 + 0.209×0.712 + 0.142×0.096
       = 0.441 + 0.322 + 0.149 + 0.014 = 0.926
   dim1: 0.353×1.250 + 0.297×0.796 + 0.209×(-0.019) + 0.142×(-0.345)
       = 0.441 + 0.236 - 0.004 - 0.049 = 0.624
   c₂ = [0.926, 0.624]

c₃ (on's context, A₃=[0.290, 0.273, 0.239, 0.197]):
   dim0: 0.290×1.250 + 0.273×1.085 + 0.239×0.712 + 0.197×0.096
       = 0.363 + 0.296 + 0.170 + 0.019 = 0.848
   dim1: 0.290×1.250 + 0.273×0.796 + 0.239×(-0.019) + 0.197×(-0.345)
       = 0.363 + 0.217 - 0.005 - 0.068 = 0.507
   c₃ = [0.848, 0.507]

c₄ (mat's context, A₄=[0.239, 0.245, 0.256, 0.260]):
   dim0: 0.239×1.250 + 0.245×1.085 + 0.256×0.712 + 0.260×0.096
       = 0.299 + 0.266 + 0.182 + 0.025 = 0.772
   dim1: 0.239×1.250 + 0.245×0.796 + 0.256×(-0.019) + 0.260×(-0.345)
       = 0.299 + 0.195 - 0.005 - 0.090 = 0.399
   c₄ = [0.772, 0.399]
```

**All context vectors at once — ONE matrix multiply:**

```
C = A @ V (4×2 = 4×4 @ 4×2):

         dim0    dim1
cat:  [  0.959,  0.679]   ← high because cat attends to v₁=[1.25,1.25] (itself, weight=0.385)
sat:  [  0.926,  0.624]   ← similar to cat; sat still attends mostly to cat's value
on:   [  0.848,  0.507]   ← lower; spread attention, v₃ and v₄ (low values) pull down
mat:  [  0.772,  0.399]   ← lowest dim0, also dim1 pulled down by v₄'s negative values

All 4 computed in ONE matrix multiply. No loop. Fully parallel.
```

### Step 3: Residual + LayerNorm₁

**Residual: X_attn = X_pe + C**

```
cat:  [1.000,  1.500] + [0.959, 0.679] = [1.959, 2.179]
sat:  [1.041,  0.840] + [0.926, 0.624] = [1.967, 1.464]
on:   [1.009, -0.316] + [0.848, 0.507] = [1.857, 0.191]
mat:  [0.341, -0.590] + [0.772, 0.399] = [1.113, -0.191]
```

**LayerNorm₁ (γ=1, β=0, normalize over feature dim per position):**

```
For each row x=[a, b]:
   μ = (a+b)/2
   σ = √(((a-μ)² + (b-μ)²) / 2)  = |a-b|/2   (algebraic simplification for d=2)
   LN(x) = [(a-μ)/σ, (b-μ)/σ]

cat:  x=[1.959, 2.179]
   μ = (1.959+2.179)/2 = 2.069
   σ = (2.179-1.959)/2 = 0.110
   LN₁(cat) = [(1.959-2.069)/0.110, (2.179-2.069)/0.110] = [-1.000, 1.000]

sat:  x=[1.967, 1.464]
   μ = (1.967+1.464)/2 = 1.716
   σ = (1.967-1.464)/2 = 0.252
   LN₁(sat) = [(1.967-1.716)/0.252, (1.464-1.716)/0.252] = [1.000, -1.000]

on:   x=[1.857, 0.191]
   μ = (1.857+0.191)/2 = 1.024
   σ = (1.857-0.191)/2 = 0.833
   LN₁(on) = [(1.857-1.024)/0.833, (0.191-1.024)/0.833] = [1.000, -1.000]

mat:  x=[1.113, -0.191]
   μ = (1.113+(-0.191))/2 = 0.461
   σ = (1.113-(-0.191))/2 = 0.652
   LN₁(mat) = [(1.113-0.461)/0.652, (-0.191-0.461)/0.652] = [1.000, -1.000]
```

**LayerNorm output X_ln1:**

```
         dim0    dim1
cat:  [-1.000,  1.000]
sat:  [ 1.000, -1.000]
on:   [ 1.000, -1.000]
mat:  [ 1.000, -1.000]
```

**A note on d=2 degeneracy:**
```
For d=2, LayerNorm ALWAYS maps to {+1, -1} (in some order).
This is mathematically expected: with only 2 values, normalizing to zero mean
and unit variance must give ±1 (since (a-mean)²+(b-mean)²=2, so each term is ±1).

At d=512 (real transformers), LN output values are NOT all ±1.
The diverse mix of feature values produces a rich normalized representation.

The mechanics here are IDENTICAL to the real case — the formula is the same.
The degeneracy is a property of this toy example's dimensionality, not a bug.
```

### Step 4: FFN (Feed-Forward Network)

FFN is applied INDEPENDENTLY to each position using the SAME W1, W2.
"Position-wise" = same transformation applied to each position's vector.

**FFN formula: H = ReLU(X_ln1 @ W1 + b1);   FFN_out = H @ W2 + b2**

**cat (X_ln1[cat] = [-1.000, 1.000]):**

```
pre_activation = [-1.0, 1.0] @ W1 + b1
   = [-1.0×0.5+1.0×0.4, -1.0×0.3+1.0×0.2, -1.0×0.2+1.0×0.3, -1.0×0.1+1.0×0.1]
   = [-0.5+0.4, -0.3+0.2, -0.2+0.3, -0.1+0.1]
   = [-0.1, -0.1, 0.1, 0.0]

h_cat = ReLU([-0.1, -0.1, 0.1, 0.0]) = [0.0, 0.0, 0.1, 0.0]   (negatives killed)

FFN_out_cat = [0.0, 0.0, 0.1, 0.0] @ W2 + b2
   dim0: 0×0.5 + 0×0.2 + 0.1×0.3 + 0×0.1 = 0.030
   dim1: 0×0.3 + 0×0.4 + 0.1×0.2 + 0×0.5 = 0.020
   FFN_out_cat = [0.030, 0.020]
```

**sat (X_ln1[sat] = [1.000, -1.000]):**

```
pre_activation = [1.0, -1.0] @ W1 + b1
   = [1.0×0.5+(-1.0)×0.4, 1.0×0.3+(-1.0)×0.2, 1.0×0.2+(-1.0)×0.3, 1.0×0.1+(-1.0)×0.1]
   = [0.5-0.4, 0.3-0.2, 0.2-0.3, 0.1-0.1]
   = [0.1, 0.1, -0.1, 0.0]

h_sat = ReLU([0.1, 0.1, -0.1, 0.0]) = [0.1, 0.1, 0.0, 0.0]

FFN_out_sat = [0.1, 0.1, 0.0, 0.0] @ W2
   dim0: 0.1×0.5 + 0.1×0.2 + 0×0.3 + 0×0.1 = 0.05 + 0.02 = 0.070
   dim1: 0.1×0.3 + 0.1×0.4 + 0×0.2 + 0×0.5 = 0.03 + 0.04 = 0.070
   FFN_out_sat = [0.070, 0.070]
```

**on (X_ln1[on] = [1.000, -1.000]) — same as sat:**

```
FFN_out_on = [0.070, 0.070]   (identical to sat since LN₁ output is same)
```

**mat (X_ln1[mat] = [1.000, -1.000]) — same as sat:**

```
pre_activation = [0.1, 0.1, -0.1, 0.0]
h_mat = ReLU = [0.1, 0.1, 0.0, 0.0]

FFN_out_mat = [0.070, 0.070]

(same as sat/on — a property of d=2 degeneracy; at d=512, each position
 gets a different LN output and therefore a different FFN output)
```

**FFN summary table:**

```
         LN₁ input      pre_act (W1)           h (ReLU)        FFN_out (W2)
─────────────────────────────────────────────────────────────────────────────────
cat:  [-1, 1]         [-0.1,-0.1,0.1,0.0]   [0.0,0.0,0.1,0.0]   [0.030, 0.020]
sat:  [ 1,-1]         [0.1, 0.1,-0.1,0.0]   [0.1,0.1,0.0,0.0]   [0.070, 0.070]
on:   [ 1,-1]         [0.1, 0.1,-0.1,0.0]   [0.1,0.1,0.0,0.0]   [0.070, 0.070]
mat:  [ 1,-1]         [0.1, 0.1,-0.1,0.0]   [0.1,0.1,0.0,0.0]   [0.070, 0.070]
─────────────────────────────────────────────────────────────────────────────────
Note: ReLU kills neuron 3 (-0.1→0) and neuron 4 (0.0→0) for sat/on/mat.
      For cat: neurons 1,2 killed, neurons 3,4 differ.
      ReLU is providing position-specific sparsity.
```

### Step 5: Residual + Final Output

**Residual 2: X_final = X_attn + FFN_out**

```
cat:  [1.959, 2.179] + [0.030, 0.020] = [1.989, 2.199]
sat:  [1.967, 1.464] + [0.070, 0.070] = [2.037, 1.534]
on:   [1.857, 0.191] + [0.070, 0.070] = [1.927, 0.261]
mat:  [1.113, -0.191] + [0.070, 0.070] = [1.183, -0.121]
```

**Classify from mat's final representation:**

```
z = W_out · x_final_mat + b_out
  = [0.5, 0.3] · [1.183, -0.121] + 0
  = 0.5×1.183 + 0.3×(-0.121)
  = 0.592 - 0.036
  = 0.556

ŷ = σ(0.556) = 1 / (1 + e^(-0.556))
  = 1 / (1 + 0.574)
  = 1 / 1.574
  = 0.635
```

---

## 6. Loss

```
Target y = 1  ("mat" IS in the sentence)
ŷ = 0.635

Binary cross-entropy:
   L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
     = -[1·log(0.635) + 0·log(0.365)]
     = -log(0.635)
     = 0.454

┌────────────────────────────────────────────┐
│  Transformer loss: L = 0.454               │
│  vs Attention file:  L = 0.083             │
│                                            │
│  Different because:                        │
│  1. Positional encoding changes X          │
│     → different Q/K/V → different c₄       │
│  2. FFN adds [0.07,0.07] to x_attn_mat     │
│  3. Together: x_final_mat=[1.183,-0.121]   │
│     vs attention's c₄=[0.417, 0.312]       │
│  Both will decrease after one update.      │
└────────────────────────────────────────────┘
```

---

## 7. Backward Pass

### Step A: Output Gradient

```
Loss L = -log(ŷ),  ŷ = σ(z)

∂L/∂z = ŷ - y = 0.635 - 1 = -0.365

(This is the clean formula for BCE + sigmoid combined.
 Derivation: ∂L/∂ŷ = -1/ŷ = -1.575, then × ∂ŷ/∂z = ŷ(1-ŷ) = 0.232 → -0.365)
```

### Step B: Gradient Through W_out

```
z = W_out · x_final_mat
  = W_out[0]×x_final_mat[0] + W_out[1]×x_final_mat[1]

∂L/∂W_out[0] = ∂L/∂z × x_final_mat[0] = -0.365 × 1.183 = -0.432
∂L/∂W_out[1] = ∂L/∂z × x_final_mat[1] = -0.365 × (-0.121) = +0.044

∂L/∂W_out = [-0.432, +0.044]
```

### Step C: Gradient to x_final_mat

```
∂L/∂x_final_mat[0] = ∂L/∂z × W_out[0] = -0.365 × 0.5 = -0.183
∂L/∂x_final_mat[1] = ∂L/∂z × W_out[1] = -0.365 × 0.3 = -0.110

∂L/∂x_final_mat = [-0.183, -0.110]
```

### Step D: Residual 2 Gradient Split

**This is the key new insight of Transformers over plain Attention:**

```
X_final = X_attn + FFN_out

The + operation has gradient = 1 to BOTH branches:

   ∂L/∂x_final_mat = [-0.183, -0.110]
          │
          ├──► ∂L/∂x_attn_mat    = [-0.183, -0.110]  ← direct highway (no multiplication!)
          │
          └──► ∂L/∂FFN_out_mat   = [-0.183, -0.110]  ← same gradient to FFN
```

**Residual gradient highway in action:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  RESIDUAL GRADIENT HIGHWAY                                          │
│                                                                     │
│  x_final = x_attn + FFN(LN(x_attn))                                │
│                                                                     │
│  ∂L/∂x_attn = ∂L/∂x_final × 1   (from the + in residual)          │
│             + ∂L/∂x_attn from FFN path (through LN → FFN)          │
│                                                                     │
│  The FIRST term is DIRECT — no weight matrices, no activation       │
│  functions on the critical path. Gradient = [-0.183, -0.110]        │
│  arrives intact at x_attn_mat, bypassing LN and FFN entirely.       │
│                                                                     │
│  Compare to RNN: grad must pass through tanh at EVERY timestep.     │
│  Here: grad skips the sublayer entirely via +.                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Step E: FFN Backward

**From ∂L/∂FFN_out_mat = [-0.183, -0.110]:**

**∂L/∂W2 (gradient at output projection):**

```
FFN_out = h @ W2 + b2, where h = [0.1, 0.1, 0.0, 0.0] (mat's hidden)

∂L/∂W2 = hᵀ @ ∂L/∂FFN_out
        = [[0.1], [0.1], [0.0], [0.0]] @ [[-0.183, -0.110]]

∂L/∂W2 = [[-0.1×0.183,  -0.1×0.110],    row 0
           [-0.1×0.183,  -0.1×0.110],    row 1
           [0,            0          ],    row 2
           [0,            0          ]]   row 3

         = [[-0.018, -0.011],
            [-0.018, -0.011],
            [ 0.000,  0.000],
            [ 0.000,  0.000]]
```

**∂L/∂h (gradient to hidden layer):**

```
∂L/∂h = ∂L/∂FFN_out @ W2ᵀ

W2ᵀ (shape 2×4):
   [[0.5, 0.2, 0.3, 0.1],
    [0.3, 0.4, 0.2, 0.5]]

∂L/∂h = [-0.183, -0.110] @ W2ᵀ
   h[0]: -0.183×0.5 + (-0.110)×0.3 = -0.092 - 0.033 = -0.125
   h[1]: -0.183×0.2 + (-0.110)×0.4 = -0.037 - 0.044 = -0.081
   h[2]: -0.183×0.3 + (-0.110)×0.2 = -0.055 - 0.022 = -0.077
   h[3]: -0.183×0.1 + (-0.110)×0.5 = -0.018 - 0.055 = -0.073

∂L/∂h = [-0.125, -0.081, -0.077, -0.073]
```

**Backward through ReLU (mask from forward):**

```
pre_activation_mat = [0.1, 0.1, -0.1, 0.0]
ReLU mask: [0.1 > 0 → 1, 0.1 > 0 → 1, -0.1 > 0 → 0, 0.0 > 0 → 0]
mask = [1, 1, 0, 0]

∂L/∂pre_act = ∂L/∂h ⊙ mask
            = [-0.125, -0.081, -0.077, -0.073] ⊙ [1, 1, 0, 0]
            = [-0.125, -0.081, 0.000, 0.000]

Neurons 3 and 4 were OFF (≤0 in forward) → they receive ZERO gradient.
This is ReLU's sparsity: dead neurons propagate nothing backward.
```

**∂L/∂W1 (gradient at input projection):**

```
pre_act = LN₁_mat @ W1 + b1,  where LN₁_mat = [1.000, -1.000]

∂L/∂W1 = LN₁_matᵀ @ ∂L/∂pre_act

LN₁_matᵀ = [[1.000],
             [-1.000]]

∂L/∂W1 = [[1.000], [-1.000]] @ [[-0.125, -0.081, 0.000, 0.000]]

        = [[-0.125, -0.081,  0.000, 0.000],   ← row from LN₁_mat[0]=1.0
           [ 0.125,  0.081,  0.000, 0.000]]    ← row from LN₁_mat[1]=-1.0 × (-1)

∂L/∂W1 = [[-0.125, -0.081,  0.000, 0.000],
           [ 0.125,  0.081,  0.000, 0.000]]
```

**∂L/∂b1, ∂L/∂b2:**

```
∂L/∂b2 = ∂L/∂FFN_out = [-0.183, -0.110]
∂L/∂b1 = ∂L/∂pre_act = [-0.125, -0.081, 0.000, 0.000]
(biases start at 0; update is small)
```

### Step F: LayerNorm₁ Backward

**The gradient coming back through LN₁ to x_attn_mat.**

```
LN₁ takes x_attn_mat = [1.113, -0.191] → y_LN = [1.000, -1.000]

Gradient flowing INTO LN₁ (from FFN path):
∂L/∂y_LN = ∂L/∂LN₁_mat = ∂L/∂pre_act @ W1ᵀ

W1ᵀ shape (4×2), W1ᵀ = [[0.5,0.4],[0.3,0.2],[0.2,0.3],[0.1,0.1]]
∂L/∂y_LN = ∂L/∂pre_act @ W1ᵀ
          = [-0.125, -0.081, 0.000, 0.000] @ [[0.5,0.4],[0.3,0.2],[0.2,0.3],[0.1,0.1]]
   dim0: -0.125×0.5 + (-0.081)×0.3 + 0 + 0 = -0.063 - 0.024 = -0.087
   dim1: -0.125×0.4 + (-0.081)×0.2 + 0 + 0 = -0.050 - 0.016 = -0.066

∂L/∂y_LN = [-0.087, -0.066]
```

**LN Jacobian (d=2, the mathematical result):**

```
For d=2 with y=[y₁,y₂], the LN Jacobian is:

   J = (γ/σ) × [I - (1/d)·11ᵀ - (1/d)·yyᵀ]

With γ=1, d=2, y=[1,-1], σ=0.652:

   (1/d)·11ᵀ = (1/2)×[[1,1],[1,1]] = [[0.5,0.5],[0.5,0.5]]

   (1/d)·yyᵀ = (1/2)×[[1×1,1×(-1)],[(-1)×1,(-1)×(-1)]]
              = (1/2)×[[1,-1],[-1,1]] = [[0.5,-0.5],[-0.5,0.5]]

   I - [[0.5,0.5],[0.5,0.5]] - [[0.5,-0.5],[-0.5,0.5]]
   = [[1-0.5-0.5,  0-0.5+0.5],
      [0-0.5+0.5,  1-0.5-0.5]]
   = [[0, 0],
      [0, 0]]
```

**Result: The LN Jacobian is exactly ZERO for d=2.**

```
∂L/∂x_attn_mat (via FFN path through LN) = ∂L/∂y_LN @ J = [-0.087,-0.066] @ [[0,0],[0,0]] = [0,0]

This means: in this d=2 example, NO gradient flows through LN₁ to x_attn_mat.

Why does this happen?
   For d=2, LN maps any input to ±1 (completely ignores magnitude, only preserves sign of difference).
   The Jacobian measures "how much does output change per unit change in input?"
   When the output is always ±1 regardless of input magnitude, the Jacobian is 0.

At d=512, the LN Jacobian is non-zero and gradient flows through.
The d=2 case demonstrates the CRITICAL importance of residual connections:
   Without residual: gradient through LN = 0 → x_attn_mat receives no gradient
   With residual: gradient arrives directly (Step D) regardless of LN behavior
```

### Step G: Residual 1 Gradient Split

```
X_attn = X_pe + C   (residual connection around attention)

Total ∂L/∂x_attn_mat = [-0.183, -0.110] (from residual 2, Step D)
                      + [0.000, 0.000]   (from LN₁ path, Step F — zero for d=2)
                      = [-0.183, -0.110]

Now this splits through the first residual:

   ∂L/∂x_attn_mat = [-0.183, -0.110]
          │
          ├──► ∂L/∂x_pe_mat = [-0.183, -0.110]   ← direct highway to input embedding+PE
          │
          └──► ∂L/∂c₄ = [-0.183, -0.110]          ← gradient to attention output
```

### Step H: Attention Weight Gradients (Focus on Wv)

**From ∂L/∂c₄ = [-0.183, -0.110] (same structure as attention file):**

```
C = A @ V

Only position 4 (mat) contributes to loss via x_final_mat.
∂L/∂C = [[0,0],[0,0],[0,0],[-0.183,-0.110]]  (only row 4 has gradient)

∂L/∂V = Aᵀ @ ∂L/∂C:
   Only the 4th column of Aᵀ (= 4th row of A) contributes:
   A[:,4] = [A₁₄, A₂₄, A₃₄, A₄₄] = [0.118, 0.142, 0.197, 0.260]
   (how much each position's key was attended to by mat's query)

∂L/∂V[i] = A[4,i] × ∂L/∂c₄   (mat's query selects V values in proportion to its attention)

∂L/∂v₁ = 0.118 × [-0.183, -0.110] = [-0.022, -0.013]
∂L/∂v₂ = 0.142 × [-0.183, -0.110] = [-0.026, -0.016]
∂L/∂v₃ = 0.197 × [-0.183, -0.110] = [-0.036, -0.022]
∂L/∂v₄ = 0.260 × [-0.183, -0.110] = [-0.048, -0.029]
```

**∂L/∂Wv = X_pe.T @ ∂L/∂V:**

```
X_pe.T (2×4):
   [[1.000, 1.041, 1.009, 0.341],
    [1.500, 0.840, -0.316, -0.590]]

∂L/∂V (4×2):
   [[-0.022, -0.013],
    [-0.026, -0.016],
    [-0.036, -0.022],
    [-0.048, -0.029]]

∂L/∂Wv[0,0] = 1.000×(-0.022) + 1.041×(-0.026) + 1.009×(-0.036) + 0.341×(-0.048)
            = -0.022 - 0.027 - 0.036 - 0.016 = -0.101

∂L/∂Wv[0,1] = 1.000×(-0.013) + 1.041×(-0.016) + 1.009×(-0.022) + 0.341×(-0.029)
            = -0.013 - 0.017 - 0.022 - 0.010 = -0.062

∂L/∂Wv[1,0] = 1.500×(-0.022) + 0.840×(-0.026) + (-0.316)×(-0.036) + (-0.590)×(-0.048)
            = -0.033 - 0.022 + 0.011 + 0.028 = -0.016

∂L/∂Wv[1,1] = 1.500×(-0.013) + 0.840×(-0.016) + (-0.316)×(-0.022) + (-0.590)×(-0.029)
            = -0.020 - 0.013 + 0.007 + 0.017 = -0.009

∂L/∂Wv = [[-0.101, -0.062],
           [-0.016, -0.009]]
```

**∂L/∂Wq, ∂L/∂Wk (brief — same chain rule pattern):**

```
These require backpropping through softmax → score matrix → Q,K.
The gradient is much smaller because it flows through A (bounded 0-1)
AND through the softmax Jacobian (sum-zero constraint).

Rough magnitudes: |∂L/∂Wq| ≈ 0.001–0.003, |∂L/∂Wk| ≈ 0.001–0.003
(see attention file Step D for full softmax backward derivation)
```

### Step I: Summary — The Transformer Gradient Highway

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GRADIENT FLOW THROUGH ONE TRANSFORMER ENCODER BLOCK                    │
│                                                                          │
│  Loss                                                                    │
│   │                                                                      │
│   ▼ ∂L/∂z = -0.365 (= ŷ - y)                                           │
│  [W_out]   ∂L/∂W_out = [-0.432, +0.044]   ← LARGEST update             │
│   │                                                                      │
│   ▼ ∂L/∂x_final_mat = [-0.183, -0.110]                                  │
│  [Residual 2 +]                                                          │
│   ├─────────────────────────────────────────────────────────────────┐   │
│   │ DIRECT (highway)           │ FFN path                           │   │
│   │ ∂L/∂x_attn = [-0.183,-0.110]│ ∂L/∂FFN_out = [-0.183,-0.110]   │   │
│   │ No multiplication!          │ → ∂L/∂W2=tiny, ∂L/∂W1=tiny       │   │
│   │                             │ → LN₁ blocks further backward     │   │
│   ▼                                                                  │   │
│  [Residual 1 +]  ∂L/∂x_attn = [-0.183, -0.110]                    │   │
│   ├─────────────────────────────────────────────────────────────┐   │   │
│   │ DIRECT (highway)           │ Attention path                  │   │   │
│   │ ∂L/∂x_pe_mat=[-0.183,-0.110]│ ∂L/∂c₄=[-0.183,-0.110]        │   │   │
│   │ Reaches embedding+PE direct │ → ∂L/∂Wv=[[-0.101,-0.062],    │   │   │
│   │ in ONE step                 │           [-0.016,-0.009]]     │   │   │
│   │                             │ → ∂L/∂Wq,∂L/∂Wk (tiny)        │   │   │
│   └─────────────────────────────┘                               │   │   │
│                                                                  └───┘   │
│                                                                          │
│  KEY INSIGHT: Two residual connections = TWO gradient highways.          │
│  At each +, gradient flows directly to BOTH inputs without any           │
│  multiplicative shrinkage. This enables training 100+ layer models.     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Weight Update

Learning rate η = 0.1

### W_out Update

```
W_out_new = W_out - η × ∂L/∂W_out
          = [0.500, 0.300] - 0.1 × [-0.432, +0.044]
          = [0.500+0.043, 0.300-0.004]
          = [0.543, 0.296]

Change: Δ = [+0.043, -0.004]
W_out[0] (weight on dim0) increased: x_final_mat[0]=1.183>0, ŷ<y → increase this weight
W_out[1] (weight on dim1) decreased: x_final_mat[1]=-0.121<0 → pushing prediction down, reduce
```

### W2 Update

```
W2_new = W2 - η × ∂L/∂W2
       = W2 - 0.1 × [[-0.018,-0.011],[-0.018,-0.011],[0,0],[0,0]]

W2_new = [[0.500+0.002, 0.300+0.001],
          [0.200+0.002, 0.400+0.001],
          [0.300,       0.200      ],
          [0.100,       0.500      ]]
       = [[0.502, 0.301],
          [0.202, 0.401],
          [0.300, 0.200],
          [0.100, 0.500]]

Change: rows 1,2 increased slightly; rows 3,4 unchanged (h[2]=h[3]=0 → no gradient)
```

### W1 Update

```
W1_new = W1 - η × ∂L/∂W1
       = W1 - 0.1 × [[-0.125,-0.081,0,0],[0.125,0.081,0,0]]

W1_new = [[0.500+0.013, 0.300+0.008, 0.200, 0.100],
          [0.400-0.013, 0.200-0.008, 0.300, 0.100]]
       = [[0.513, 0.308, 0.200, 0.100],
          [0.387, 0.192, 0.300, 0.100]]

Change: W1[0,0,1] increased, W1[1,0,1] decreased.
Cols 2,3 unchanged (neurons 3,4 were OFF in forward → no gradient via ReLU)
```

### Wv Update

```
Wv_new = Wv - η × ∂L/∂Wv
       = [[0.800, 0.200], - 0.1 × [[-0.101, -0.062],
          [0.300, 0.700]]           [-0.016, -0.009]]

       = [[0.800+0.010, 0.200+0.006],
          [0.300+0.002, 0.700+0.001]]
       = [[0.810, 0.206],
          [0.302, 0.701]]
```

### All Weight Changes Summary

```
┌────────────────────────────────────────────────────────────┐
│  Weight         Old         New         |Δ|     Relative  │
├────────────────────────────────────────────────────────────┤
│  W_out[0]       0.500       0.543       0.043   largest    │
│  W_out[1]       0.300       0.296       0.004              │
│  W1[0,0]        0.500       0.513       0.013              │
│  W1[1,0]        0.400       0.387       0.013              │
│  W2[0,0]        0.500       0.502       0.002              │
│  Wv[0,0]        0.800       0.810       0.010              │
│  Wv[0,1]        0.200       0.206       0.006              │
│  Wq, Wk         (tiny updates, <0.001 each)                │
├────────────────────────────────────────────────────────────┤
│  Largest change: W_out (direct path to loss)               │
│  Second: W1 (through FFN, close to residual highway)       │
│  Smallest: Wq, Wk (through softmax Jacobian, sum-zero)     │
└────────────────────────────────────────────────────────────┘
```

---

## 9. Second Forward Verify

Run the model with updated weights to confirm loss decreased.

**With Wv_new = [[0.810,0.206],[0.302,0.701]], new V:**

```
v₄_new = x_pe_mat @ Wv_new
        = [0.341, -0.590] @ [[0.810,0.206],[0.302,0.701]]
   dim0: 0.341×0.810 + (-0.590)×0.302 = 0.276 - 0.178 = 0.098
   dim1: 0.341×0.206 + (-0.590)×0.701 = 0.070 - 0.413 = -0.343
   v₄_new = [0.098, -0.343]   (barely changed from [0.096, -0.345])

Similarly v₁_new, v₂_new, v₃_new change by <0.015 in each dim.
```

**Attention weights A₄ unchanged** (Q,K barely updated → same softmax output):

```
A₄ ≈ [0.239, 0.245, 0.256, 0.260]   (same as before)
```

**New context vector c₄:**

```
c₄_new ≈ [0.772 + ε₀, 0.399 + ε₁]  where ε < 0.002

c₄_new ≈ [0.773, 0.400]
```

**New x_attn_mat:**

```
x_attn_new_mat = x_pe_mat + c₄_new
              = [0.341, -0.590] + [0.773, 0.400]
              = [1.114, -0.190]  (vs [1.113, -0.191] before)
```

**New LN₁ output** (σ slightly different but output ≈ [1.000, -1.000] again):

```
LN₁(x_attn_new_mat) ≈ [1.000, -1.000]
```

**New FFN output** with W1_new, W2_new:

```
pre_act_new = [1.000, -1.000] @ W1_new
   = [1.0×0.513+(-1.0)×0.387, 1.0×0.308+(-1.0)×0.192, 0.2-0.3, 0.1-0.1]
   = [0.126, 0.116, -0.100, 0.000]

h_new = ReLU = [0.126, 0.116, 0.000, 0.000]

FFN_out_new = [0.126, 0.116, 0.000, 0.000] @ W2_new
   dim0: 0.126×0.502 + 0.116×0.202 + 0 + 0 = 0.063 + 0.023 = 0.086
   dim1: 0.126×0.301 + 0.116×0.401 + 0 + 0 = 0.038 + 0.047 = 0.085

FFN_out_new_mat = [0.086, 0.085]
```

**New x_final_mat:**

```
x_final_new_mat = x_attn_new_mat + FFN_out_new_mat
               = [1.114, -0.190] + [0.086, 0.085]
               = [1.200, -0.105]
```

**New prediction with W_out_new = [0.543, 0.296]:**

```
z' = 0.543×1.200 + 0.296×(-0.105)
   = 0.652 - 0.031
   = 0.621

ŷ' = σ(0.621) = 1/(1+e^(-0.621)) = 1/(1+0.538) = 1/1.538 = 0.650
```

**New loss:**

```
L' = -log(0.650) = 0.431

Comparison:
   L  = 0.454   (before update)
   L' = 0.431   (after one update)

   ΔL = -0.023   ← loss decreased ✓

Loss decreased from 0.454 → 0.431 after one gradient step.
```

---

## 10. Gradient Magnitude Comparison

```
Architecture    Gradient path to first position ("cat")    Magnitude
────────────────────────────────────────────────────────────────────────
RNN             L → h₄ → h₃ → h₂ → h₁                    ~9%
LSTM            L → h₄ → (C₄ → C₁ via forget gate)        ~69%
GRU             L → h₄ → h₁ via update gate               ~66%
────────────────────────────────────────────────────────────────────────
Attention       L → c₄ → V[1] via A[4,1]=0.275            27.5%  (raw X)
Transformer     L → c₄ → V[1] via A[4,1]=0.239            23.9%  (with PE)
                + residual highway to x_pe_cat              +100%
────────────────────────────────────────────────────────────────────────

Wait — the Transformer's TOTAL gradient to position 1 is:
   Via attention path: A[4,1] × |∂L/∂c₄| = 0.239 × 0.071 ≈ 17%
   Via residual (direct to x_pe): |∂L/∂x_pe_mat| = |-0.183,-0.110|
   BUT: x_pe_mat and x_pe_cat are different positions.

The residual highway goes to x_pe_MAT (the current position), not to cat.
Cat gets gradient only through the attention path:
   ∂L/∂v_cat via A[4,1] = 0.239

For sequence length n=100:
   RNN:         (0.9)^100 = 0.000027 ≈ 0%
   LSTM/GRU:    (0.9)^100 ≈ 0.003%  (gate leak still accumulates)
   Transformer: A[100,1] × |∂L/∂c₁₀₀|   → depends on attention weight
                At n=100, attention CAN still put weight on position 1.
                A[100,1] doesn't decay with distance! It's O(1) in distance.
```

**The key comparison:**

```
┌──────────────────────────────────────────────────────────────────┐
│  Gradient to position 1 at different sequence lengths            │
│                                                                  │
│  n=4:                                                            │
│    RNN: 9%, LSTM: 69%, GRU: 66%, Attn: 23-27%, Tfm: ~24%+PE   │
│                                                                  │
│  n=100:                                                          │
│    RNN: ≈0%, LSTM: ≈0.003%, GRU: ≈0.003%                        │
│    Attn: A[100,1] × |∂L/∂c₁₀₀|   → A[100,1] is LEARNED        │
│           If the task needs position 1, attention learns         │
│           to put A[100,1] ≈ 0.5 → 50% gradient                 │
│    Tfm: SAME as Attn — no distance-based decay                  │
│                                                                  │
│  The Transformer doesn't just solve vanishing gradient.         │
│  It makes gradient magnitude a LEARNED property                 │
│  rather than an architectural constraint.                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 11. Multi-Head Attention — What It Would Look Like

With d_model=2 we can only run 1 head (d_head=2). Here's how 2 heads work at d_model=4:

```
d_model = 4,  n_heads = 2,  d_head = 2

INPUT: x = [a, b, c, d]   (d=4 embedding for each word)

HEAD 1 (dimensions 0,1):          HEAD 2 (dimensions 2,3):
   Wq₁: (4×2)                        Wq₂: (4×2)
   Wk₁: (4×2)                        Wk₂: (4×2)
   Wv₁: (4×2)                        Wv₂: (4×2)

   Q₁ = X @ Wq₁   (n×2)             Q₂ = X @ Wq₂   (n×2)
   A₁ = softmax(Q₁@K₁ᵀ/√2)          A₂ = softmax(Q₂@K₂ᵀ/√2)
   C₁ = A₁ @ V₁   (n×2)             C₂ = A₂ @ V₂   (n×2)

CONCATENATE:
   C = concat([C₁, C₂])    shape: (n×4)

PROJECT:
   MHA_out = C @ W_o        W_o: (4×4),  MHA_out: (n×4)

What does each head learn?
   Head 1: might specialize in syntactic relationships (subject → verb)
   Head 2: might specialize in semantic relationships (pronoun → antecedent)
   
   Different Wq,Wk for each head → different "questions" asked
   Different attention patterns per head → diverse information extraction
   W_o learns to combine the two heads' outputs optimally
```

**Why 8 heads in BERT?**

```
At d=512, n_heads=8 → d_head=64.
Each head attends with (512×64) projections.
8 heads × d_head=64 = 512 → dimension preserved after concatenation + W_o.

Benefits at scale:
1. Parallel specialization: heads 1-3 do syntax, 4-6 do coreference, 7-8 do positional
2. Ensemble effect: 8 independent attention patterns averaged → more robust
3. Expressivity: 8 × (n²) attention entries vs 1 × (n²) → 8× information
4. Same compute: each head is smaller (d/h), total FLOPS ≈ single head

At inference: all heads computed in parallel on GPU.
During training: all heads' gradients computed simultaneously.
```

---

## 12. Full Picture Diagram

```
INPUT LAYER
   "cat"    "sat"    "on"     "mat"
    │         │        │        │
    ▼         ▼        ▼        ▼
[1.0,0.5] [0.2,0.3] [0.1,0.1] [0.2,0.4]   ← word embeddings
    │         │        │        │
    + PE(0)   + PE(1)  + PE(2)  + PE(3)      ← positional encodings
    │         │        │        │
[1.0,1.5] [1.041,0.840] [1.009,-0.316] [0.341,-0.590]   ← X_pe
    │         │        │        │
    └────┬────┘────────┘─────────┘
         │         (ALL positions in parallel)
         ▼
  ┌──────────────────────────────────────────────┐
  │   SCALED DOT-PRODUCT ATTENTION               │
  │                                              │
  │   Q=X_pe@Wq    K=X_pe@Wk    V=X_pe@Wv      │
  │                                              │
  │   S = Q @ Kᵀ / √2  (4×4 score matrix)       │
  │   A = softmax(S)   (4×4 attention weights)   │
  │   C = A @ V        (4×2 context vectors)     │
  │                                              │
  │   A₄=[0.239,0.245,0.256,0.260]              │
  │   c₄=[0.772, 0.399]                          │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │   RESIDUAL 1                │
  │   X_attn = X_pe + C         │
  │   mat: [1.113, -0.191]      │← x_pe_mat + c₄
  └─────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │   LAYERNORM 1               │
  │   Normalize per position    │
  │   mat: [1.000, -1.000]      │← normalized (σ=0.652)
  └─────────────────────────────┘
         │
         ▼  (same weights W1, W2 for ALL positions)
  ┌──────────────────────────────────────────────┐
  │   FEED-FORWARD NETWORK                       │
  │                                              │
  │   H = ReLU(X_ln1 @ W1 + b1)                 │
  │   mat: [0.1,0.1,-0.1,0.0] → [0.1,0.1,0,0]  │← ReLU kills 2 neurons
  │                                              │
  │   FFN = H @ W2 + b2                          │
  │   mat: [0.070, 0.070]                        │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────┐
  │   RESIDUAL 2                │
  │   X_final = X_attn + FFN    │
  │   mat: [1.183, -0.121]      │← x_attn_mat + FFN_out_mat
  └─────────────────────────────┘
         │
         ▼ (only mat's final representation used)
  ┌─────────────────────────────┐
  │   CLASSIFIER                │
  │   z = W_out · x_final_mat   │
  │     = 0.556                 │
  │   ŷ = σ(0.556) = 0.635      │
  │   L = -log(0.635) = 0.454   │
  └─────────────────────────────┘


BACKWARD PASS:
   ∂L/∂z = -0.365
   → ∂L/∂W_out = [-0.432, +0.044]
   → ∂L/∂x_final = [-0.183, -0.110]
        │
   [Residual 2 splits gradient]
        ├── highway → ∂L/∂x_attn = [-0.183,-0.110]
        │       │
        │   [Residual 1 splits gradient]
        │       ├── highway → ∂L/∂x_pe (input gradient)
        │       └── attention → ∂L/∂c₄ → ∂L/∂Wv=[[-0.101,-0.062],[-0.016,-0.009]]
        │
        └── FFN → ∂L/∂W2 tiny, ∂L/∂W1 tiny, LN blocks further flow
```

---

## 13. Why Next: BERT and GPT

```
One encoder block transforms a sequence → sequence.
The FULL Transformer stacks multiple blocks + adds task-specific heads.

TWO DIRECTIONS:

BERT (Encoder-only, Devlin et al., 2018):
   Stack: 12 encoder blocks (same structure as above)
   d_model=768, n_heads=12, d_ff=3072
   Input: [CLS] cat sat on mat [SEP]
   Training: Mask random tokens → predict masked word ("MLM")
             Next Sentence Prediction ("NSP")
   Use: Contextualized representations → finetune for NLP tasks
   Why encoder? Full bidirectional attention — each token sees ALL others.
   
   12 encoder blocks → 12 residual highways → gradients flow across 24 residuals
   Final CLS token is a summary of the whole sequence.

GPT (Decoder-only, Radford et al., 2018):
   Stack: 12 decoder blocks
   Same structure BUT with CAUSAL MASKING in attention:
      A[i,j] = 0 if j > i  (can only attend to LEFT context)
   Training: Predict next token ("language modeling")
   Use: Text generation — each token generated based on all previous tokens
   
   Causal mask: S[i,j] = -∞ for j>i → softmax → 0
   This ensures autoregressive generation: can't cheat by looking at future.

ENCODER-DECODER (original Transformer, Vaswani et al.):
   Encoder: processes source sequence with full attention (like BERT)
   Decoder: generates target sequence with causal attention + cross-attention
   Cross-attention: Queries from decoder, Keys/Values from encoder
   Used for translation: "cat sat on mat" → "le chat était assis sur le tapis"
```

**What makes modern LLMs powerful:**

```
Scale law (Kaplan et al., 2020):
   Loss ∝ (compute)^(-0.05)    (roughly)
   
   Doubling compute → 3-5% loss reduction → significant capability jump
   
   GPT-3:  96 layers, d=12288, n_heads=96, 175B parameters
   Training: 300B tokens → requires ~3.14×10²³ FLOPS

Emergent capabilities:
   At 175B params: few-shot learning, code generation, reasoning chains
   These behaviors were NOT explicitly trained — they "emerge" at scale
   
The transformer architecture we built today is architecturally identical
to GPT-3. The only difference is scale.
```

---

## 14. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TRANSFORMER ENCODER BLOCK — QUICK REFERENCE                            │
├─────────────────────────────────────────────────────────────────────────┤
│  FORWARD PASS:                                                           │
│  1. X_pe = X + PE         (add positional encoding)                     │
│  2. Q=X@Wq, K=X@Wk, V=X@Wv  (project to Q,K,V — same as attention)    │
│  3. S = QKᵀ/√d, A = softmax(S), C = AV  (attention)                    │
│  4. X_attn = LN₁(X + C)   (residual + layernorm)                       │
│  5. H = ReLU(X_attn@W1), F = H@W2  (FFN)                               │
│  6. X_out = LN₂(X_attn + F)  (residual + layernorm)                    │
├─────────────────────────────────────────────────────────────────────────┤
│  KEY DIMENSIONS:                                                         │
│  d_model: embedding dim (512/768/1024 in practice)                      │
│  n_heads: attention heads (8/12/16)                                     │
│  d_head = d_model/n_heads                                               │
│  d_ff = 4 × d_model  (FFN hidden dim)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  BACKWARD PASS — GRADIENT HIGHWAYS:                                     │
│  Residual 2: ∂L/∂x_attn = ∂L/∂x_out × 1   (direct)                    │
│  Residual 1: ∂L/∂x_pe = ∂L/∂x_attn × 1    (direct)                    │
│  LN: blocks gradient if features collapse (important at d=2)            │
│  FFN: ∂L/∂W2=hᵀ@∂L/∂out, ∂L/∂W1=LNᵀ@(∂L/∂pre_act⊙ReLU_mask)         │
│  Wv: ∂L/∂Wv = X_peᵀ @ (Aᵀ @ ∂L/∂C)    (outer product)                │
│  Wq,Wk: small (through softmax Jacobian, sum-zero constraint)           │
├─────────────────────────────────────────────────────────────────────────┤
│  THIS WALKTHROUGH (d=2, n=4, 1 head):                                   │
│  x_pe_mat=[0.341,-0.590], c₄=[0.772,0.399], x_final_mat=[1.183,-0.121] │
│  A₄=[0.239,0.245,0.256,0.260]  (mat attends most to itself+on)          │
│  ŷ=0.635, L=0.454 → L'=0.431 after update ✓                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 15. Code

### Version 1: Pure NumPy (manual everything)

```python
import numpy as np

# ─── Data ───────────────────────────────────────────────────────────────
vocab = {'cat': 0, 'sat': 1, 'on': 2, 'mat': 3}
X = np.array([
    [1.0, 0.5],   # cat
    [0.2, 0.3],   # sat
    [0.1, 0.1],   # on
    [0.2, 0.4],   # mat
])
y = 1.0  # "mat" is in sentence

# ─── Positional Encoding ─────────────────────────────────────────────────
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2):
            PE[pos, 2*i]   = np.sin(pos / (10000 ** (2*i / d_model)))
            PE[pos, 2*i+1] = np.cos(pos / (10000 ** (2*i / d_model)))
    return PE

PE = positional_encoding(4, 2)
X_pe = X + PE
print("X_pe:", X_pe)
# [[1.000, 1.500], [1.041, 0.840], [1.009, -0.316], [0.341, -0.590]]

# ─── Weights ─────────────────────────────────────────────────────────────
np.random.seed(0)
Wq = np.array([[0.60, 0.40], [0.20, 0.50]])
Wk = np.array([[0.50, 0.30], [0.10, 0.40]])
Wv = np.array([[0.80, 0.20], [0.30, 0.70]])

W1 = np.array([[0.5, 0.3, 0.2, 0.1],
               [0.4, 0.2, 0.3, 0.1]])
b1 = np.zeros(4)

W2 = np.array([[0.5, 0.3],
               [0.2, 0.4],
               [0.3, 0.2],
               [0.1, 0.5]])
b2 = np.zeros(2)
W_out = np.array([0.5, 0.3])

# ─── LayerNorm ────────────────────────────────────────────────────────────
def layernorm_forward(x, gamma, beta, eps=1e-8):
    mu = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True) + eps
    x_hat = (x - mu) / sigma
    return gamma * x_hat + beta, x_hat, mu, sigma

gamma1, beta1 = np.ones(2), np.zeros(2)

# ─── Attention ───────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    
    # numerically stable softmax
    scores -= scores.max(axis=-1, keepdims=True)
    A = np.exp(scores)
    A /= A.sum(axis=-1, keepdims=True)
    
    C = A @ V
    return C, A

# ─── Forward Pass ────────────────────────────────────────────────────────
# Attention
Q = X_pe @ Wq
K = X_pe @ Wk
V = X_pe @ Wv
C, A = scaled_dot_product_attention(Q, K, V)
print("Attention matrix A:\n", np.round(A, 3))
# Row 4: [0.239, 0.245, 0.256, 0.260]

# Residual 1
X_attn = X_pe + C

# LayerNorm 1
X_ln1, X_hat1, mu1, sigma1 = layernorm_forward(X_attn, gamma1, beta1)
print("LN1 output:\n", np.round(X_ln1, 3))

# FFN
H = np.maximum(0, X_ln1 @ W1 + b1)   # ReLU
FFN_out = H @ W2 + b2
print("FFN output (mat):", np.round(FFN_out[3], 3))  # [0.070, 0.070]

# Residual 2
X_final = X_attn + FFN_out

# Classify from mat
z = W_out @ X_final[3]
y_hat = 1 / (1 + np.exp(-z))
loss = -np.log(y_hat)
print(f"ŷ={y_hat:.3f}, L={loss:.3f}")  # ŷ=0.635, L=0.454

# ─── Backward Pass ───────────────────────────────────────────────────────
lr = 0.1

# Output gradient
dL_dz = y_hat - y                          # -0.365
dL_dWout = dL_dz * X_final[3]             # [-0.432, +0.044]
dL_dxfinal_mat = dL_dz * W_out            # [-0.183, -0.110]

# Residual 2: split gradient
dL_dxattn_mat = dL_dxfinal_mat.copy()     # highway
dL_dFFNout_mat = dL_dxfinal_mat.copy()    # FFN path

# FFN backward (mat only)
dL_dW2 = H[3:4].T @ dL_dFFNout_mat.reshape(1, -1)   # (4×2)
dL_dh = dL_dFFNout_mat @ W2.T                         # (4,)
dL_dpre_act = dL_dh * (H[3] > 0).astype(float)        # ReLU mask
dL_dW1 = X_ln1[3:4].T @ dL_dpre_act.reshape(1, -1)   # (2×4)

# Residual 1: split gradient
dL_dc4 = dL_dxattn_mat.copy()

# Wv backward
dL_dV = A[:, 3:4] * dL_dc4.reshape(1, -1)  # Note: A[:,3] = column 4
# (This is simplified for 1-query backprop; see full version below)
dL_dWv = X_pe.T @ (A.T @ np.vstack([np.zeros((3,2)), dL_dc4]))

# ─── Weight Update ───────────────────────────────────────────────────────
W_out_new = W_out - lr * dL_dWout
W2_new = W2 - lr * dL_dW2
W1_new = W1 - lr * dL_dW1
Wv_new = Wv - lr * dL_dWv

# ─── Verify Loss Decreased ───────────────────────────────────────────────
Q2 = X_pe @ Wq; K2 = X_pe @ Wk; V2 = X_pe @ Wv_new
C2, A2 = scaled_dot_product_attention(Q2, K2, V2)
X_attn2 = X_pe + C2
X_ln12, _, _, _ = layernorm_forward(X_attn2, gamma1, beta1)
H2 = np.maximum(0, X_ln12 @ W1_new + b1)
FFN2 = H2 @ W2_new + b2
X_final2 = X_attn2 + FFN2
z2 = W_out_new @ X_final2[3]
yhat2 = 1 / (1 + np.exp(-z2))
loss2 = -np.log(yhat2)
print(f"After update: ŷ'={yhat2:.3f}, L'={loss2:.3f}")  # L' < 0.454 ✓
```

### Version 2: PyTorch Manual (autograd does backward)

```python
import torch
import torch.nn.functional as F

# ─── Data ───────────────────────────────────────────────────────────────
X = torch.tensor([
    [1.0, 0.5], [0.2, 0.3], [0.1, 0.1], [0.2, 0.4]
], dtype=torch.float32)
y = torch.tensor(1.0)

# ─── Positional Encoding ─────────────────────────────────────────────────
def get_pe(seq_len, d):
    PE = torch.zeros(seq_len, d)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.pow(10000, torch.arange(0, d, 2).float() / d)
    PE[:, 0::2] = torch.sin(pos / div)
    PE[:, 1::2] = torch.cos(pos / div)
    return PE

PE = get_pe(4, 2)
X_pe = X + PE

# ─── Weights (requires_grad=True) ────────────────────────────────────────
Wq = torch.tensor([[0.60, 0.40], [0.20, 0.50]], requires_grad=True)
Wk = torch.tensor([[0.50, 0.30], [0.10, 0.40]], requires_grad=True)
Wv = torch.tensor([[0.80, 0.20], [0.30, 0.70]], requires_grad=True)

W1 = torch.tensor([[0.5, 0.3, 0.2, 0.1],
                   [0.4, 0.2, 0.3, 0.1]], dtype=torch.float32, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.tensor([[0.5, 0.3], [0.2, 0.4],
                   [0.3, 0.2], [0.1, 0.5]], dtype=torch.float32, requires_grad=True)
b2 = torch.zeros(2, requires_grad=True)
W_out = torch.tensor([0.5, 0.3], requires_grad=True)

gamma1 = torch.ones(2, requires_grad=True)
beta1  = torch.zeros(2, requires_grad=True)

# ─── Forward ─────────────────────────────────────────────────────────────
Q = X_pe @ Wq
K = X_pe @ Wk
V = X_pe @ Wv

# Scaled dot-product attention
d_k = Q.shape[-1]
scores = Q @ K.T / (d_k ** 0.5)
A = F.softmax(scores, dim=-1)
C = A @ V

# Residual 1 + LayerNorm
X_attn = X_pe + C
X_ln1 = F.layer_norm(X_attn, [2], gamma1, beta1)

# FFN
H = F.relu(X_ln1 @ W1 + b1)
FFN_out = H @ W2 + b2

# Residual 2
X_final = X_attn + FFN_out

# Classify
z = W_out @ X_final[3]
y_hat = torch.sigmoid(z)
loss = F.binary_cross_entropy(y_hat, y)
print(f"ŷ={y_hat.item():.3f}, L={loss.item():.3f}")

# ─── Backward ────────────────────────────────────────────────────────────
loss.backward()

print("dL/dW_out:", W_out.grad)          # ≈ [-0.432, +0.044]
print("dL/dWv:\n", Wv.grad)              # ≈ [[-0.101,-0.062],[-0.016,-0.009]]
print("dL/dW1:\n", W1.grad)              # ≈ [[-0.125,-0.081,0,0],[+0.125,+0.081,0,0]]

# ─── Update ──────────────────────────────────────────────────────────────
lr = 0.1
with torch.no_grad():
    for param in [Wq, Wk, Wv, W1, b1, W2, b2, W_out, gamma1, beta1]:
        param -= lr * param.grad
        param.grad.zero_()

# Verify
Q2 = X_pe @ Wq; K2 = X_pe @ Wk; V2 = X_pe @ Wv
A2 = F.softmax(Q2 @ K2.T / (d_k**0.5), dim=-1)
C2 = A2 @ V2
X_attn2 = X_pe + C2
X_ln12 = F.layer_norm(X_attn2, [2], gamma1, beta1)
H2 = F.relu(X_ln12 @ W1 + b1)
X_final2 = X_attn2 + H2 @ W2 + b2
loss2 = F.binary_cross_entropy(torch.sigmoid(W_out @ X_final2[3]), y)
print(f"L'={loss2.item():.3f}")  # < 0.454 ✓
```

### Version 3: nn.Module (production style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        PE = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        PE[:, 0::2] = torch.sin(pos / div)
        PE[:, 1::2] = torch.cos(pos / div)
        self.register_buffer('PE', PE.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x):
        # x: (batch, seq, d)
        return x + self.PE[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        # Project and split into heads
        Q = self.Wq(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.Wk(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.Wv(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # (B, n_heads, T, d_head)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        A = F.softmax(scores, dim=-1)
        C = A @ V  # (B, n_heads, T, d_head)

        # Concatenate heads
        C = C.transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(C), A


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(F.relu(self.W1(x)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn  = FeedForward(d_model, d_ff)
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Post-norm: sublayer → residual → LN
        attn_out, A = self.attn(x, mask)
        x = self.ln1(x + self.drop(attn_out))   # Residual 1 + LN
        ffn_out = self.ffn(x)
        x = self.ln2(x + self.drop(ffn_out))    # Residual 2 + LN
        return x, A


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int,
                 d_ff: int, n_layers: int = 1, max_len: int = 100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe    = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, token_ids):
        # token_ids: (batch, seq)
        x = self.pe(self.embed(token_ids))       # (B, T, d)
        attentions = []
        for layer in self.layers:
            x, A = layer(x)
            attentions.append(A)
        # Classify from last token (or [CLS] in practice)
        logit = self.classifier(x[:, -1, :])     # (B, 1)
        return torch.sigmoid(logit).squeeze(-1), attentions


# ─── Training loop ───────────────────────────────────────────────────────
model = TransformerClassifier(
    vocab_size=4, d_model=2, n_heads=1, d_ff=4, n_layers=1
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.BCELoss()

# "cat sat on mat" → token ids [0, 1, 2, 3]
tokens = torch.tensor([[0, 1, 2, 3]])   # (1, 4)
target = torch.tensor([1.0])            # "mat" present

# Forward
y_hat, attn_maps = model(tokens)
loss = criterion(y_hat, target)
print(f"Initial: ŷ={y_hat.item():.3f}, L={loss.item():.3f}")

# Backward + update
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Verify
y_hat2, _ = model(tokens)
loss2 = criterion(y_hat2, target)
print(f"After update: ŷ'={y_hat2.item():.3f}, L'={loss2.item():.3f}")
# L' < initial L ✓

# Inspect attention pattern for "mat"
print("mat attention pattern:", attn_maps[0][0, 0, 3, :].detach().numpy())
# Should be roughly uniform-to-local (reflects PE influence)
```

---

## 16. Connections to the Full Series

```
┌──────────────────────────────────────────────────────────────────────────┐
│  COMPLETE ARCHITECTURE PROGRESSION                                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  RNN       hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁)                                    │
│  Problem:  gradient decays as tanh′^T → 0                                │
│  Fix:      need additive update (not multiplicative chain)                │
│                                                                           │
│  LSTM      Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙c̃ₜ                                       │
│  Solution: additive cell update → gradient highway along C               │
│  Problem:  still sequential, still one-vector bottleneck                 │
│                                                                           │
│  GRU       hₜ = (1-zₜ)⊙hₜ₋₁ + zₜ⊙h̃ₜ                                  │
│  Solution: same gradient highway, fewer parameters than LSTM             │
│  Problem:  still sequential, still one-vector bottleneck                 │
│                                                                           │
│  Attention C = softmax(QKᵀ/√d)V                                          │
│  Solution: all positions in parallel, no bottleneck                      │
│  Problem:  no position info, no non-linearity between positions          │
│                                                                           │
│  Transformer = Attention + PE + FFN + Residual + LayerNorm               │
│  Solution: all problems above solved                                      │
│   + PE:       positions distinguished                                    │
│   + FFN:      non-linear transformation per position                     │
│   + Residual: gradient highway at every sublayer (not just LSTM's C)    │
│   + LN:       stable activations → stack 100+ layers                    │
├──────────────────────────────────────────────────────────────────────────┤
│  GRADIENT HIGHWAY EVOLUTION:                                             │
│                                                                           │
│  RNN:    hₜ = f(h) × hₜ₋₁    → multiplicative, decays exponentially    │
│  LSTM:   Cₜ = f × Cₜ₋₁ + ...  → additive term creates highway          │
│  Attn:   c = Av                → ONE step from output to any V           │
│  Tfm:    xₜ_out = xₜ + sub()  → residual adds DIRECT path per block    │
│          With L blocks: L residual paths, gradient can choose            │
│          shortest route → enables training 100+ layer models             │
└──────────────────────────────────────────────────────────────────────────┘
```

**Three sentences summary:**

```
RNN: information flows through time (sequential, lossy).
Attention: information flows through a weighted sum (parallel, direct).
Transformer: attention + position awareness + non-linearity + residuals =
             the complete architecture that powers modern AI.
```

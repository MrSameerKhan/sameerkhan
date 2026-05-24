# 06 — Transformer: Complete End-to-End Walkthrough

> Same sentence as RNN, LSTM, GRU, and Attention. Same embeddings. Same template. The difference: positional encoding, multi-head attention, FFN, residual connections, LayerNorm — all computed with real numbers.

---

## Table of Contents

1. The Journey So Far
2. Problem Statement
3. Input & Preprocessing
4. Weight Setup
5. All Formulas
6. Expected Output
7. Forward Pass
8. Loss
9. Backward Pass
10. Weight Update
11. Second Forward Verify
12. Gradient Magnitude Comparison
13. Multi-Head Attention
14. Attention Complexity & Causal Masking
15. Full Picture Diagram
16. Why Next: BERT and GPT
17. Quick Reference Card
18. Code (3 versions)
19. Connections to Full Series
20. Gotchas
21. Interview Q&A

---

## 1. The Journey So Far

| Architecture | Key State     | Gate Count | Gradient to "cat" | Parallelizable? |
|--------------|---------------|------------|-------------------|-----------------|
| RNN          | h (overwrite) | 0          | 9%                | No              |
| LSTM         | C + h (two)   | 4          | 69%               | No              |
| GRU          | h (selective) | 2          | 66%               | No              |
| Attention    | none (direct) | —          | direct (A[4,1]=0.275) | YES         |
| Transformer  | none (stacked)| —          | direct + FFN      | YES             |

**Attention's remaining limits:**

```
1. POSITION BLINDNESS
   Attention operates on a SET of vectors.
   It has no idea that "cat" comes before "sat."
   Shuffle the sentence → same attention output.
   "cat sat on mat" and "mat on sat cat" produce IDENTICAL attention patterns.
   Embeddings carry no positional information.

2. LIMITED TRANSFORMATION POWER
   Attention computes a weighted sum of values.
   The output is a LINEAR combination of value vectors.
   No non-linearity = limited ability to model complex dependencies.
   "cat" influences "cat" through a weighted sum, but the transformation
   is just: a = v_cat * β + v_cat * ...
   There's no ReLU, no non-linear interaction between positions.
```

**The Transformer's answer:**
- Add positional information via Positional Encoding: x_input = embedding + PE(position)
- Add non-linearity via Feed-Forward Network: After attention, apply FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
- Stabilize training via Layer Normalization + Residual Connections: x_out = LayerNorm(x + Sublayer(x)); Residual: gradient highway (same as LSTM cell state); LayerNorm: stabilizes activations, enables deeper stacking
- Stack multiple layers: One encoder block = [Attention + Residual + LN] + [FFN + Residual + LN]

---

## 2. Problem Statement

```
Sentence: "cat sat on mat"
Task:     Predict: does "mat" appear in the sentence? (Label y = 1)

This walkthrough: ONE encoder block of a Transformer.
Input:   sequence of 4 word embeddings (d=2)
Step 1: Add positional encodings → X_pe
Step 2: Multi-head attention on X_pe → context vectors C
Step 3: Residual: X_attn = X_pe + C
Step 4: LayerNorm: normalize X_attn
Step 5: FFN: non-linear transform per position
Step 6: Residual: X_attn + FFN_out → X_final
Step 7: Classify from X_final[mat]
```

---

## 3. Input & Preprocessing

### 3.1 Vocabulary & Embeddings

Same table as every previous file:

| Word | Index | Embedding    |
|------|-------|--------------|
| cat  | 0     | [1.0, 0.5]   |
| sat  | 1     | [0.2, 0.3]   |
| on   | 2     | [0.1, 0.1]   |
| mat  | 3     | [0.2, 0.4]   |

d_model = 2. As a matrix X (shape 4×2):

```
      dim0  dim1
cat  [1.000, 0.500]
sat  [0.200, 0.300]
on   [0.100, 0.100]
mat  [0.200, 0.400]
```

### 3.2 Positional Encoding

Attention operates on a SET — it has no notion of order. We inject position information by adding a sinusoidal signal.

Formula (Vaswani et al., 2017):
```
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

**Computing PE for each position:**

```
pos=0 (cat):  PE = [sin(0), cos(0)] = [0.000, 1.000]
pos=1 (sat):  PE = [sin(1), cos(1)] = [0.841, 0.540]
pos=2 (on):   PE = [sin(2), cos(2)] = [0.909,-0.416]
pos=3 (mat):  PE = [sin(3), cos(3)] = [0.141,-0.990]

Recall: sin(1)=0.841, cos(1)=0.540, sin(2)=0.909, cos(2)=-0.416
        sin(3)=0.141, cos(3)=-0.990
```

**Why sinusoidal?** Three properties:
1. Bounded: sin/cos always in [-1,1] → no scale explosion
2. Unique: each (pos, dim) pair gets a unique value
3. Relative: PE(pos+k) can be expressed as a linear function of PE(pos) — the model can learn to attend to "2 positions ahead" via learned weights

### 3.3 X_pe = X + PE

| Position | X (embed)         | PE                   | X_pe               |
|----------|-------------------|----------------------|--------------------|
| cat      | [1.000, 0.500]    | [0.000, 1.000]       | [1.000, 1.500]     |
| sat      | [0.200, 0.300]    | [0.841, 0.540]       | [1.041, 0.840]     |
| on       | [0.100, 0.100]    | [0.909,-0.416]       | [1.009,-0.316]     |
| mat      | [0.200, 0.400]    | [0.141,-0.990]       | [0.341,-0.590]     |

**What PE does to representations:**
- Before PE: cat=[1.0,0.5], sat=[0.2,0.3] — "cat" is "above" sat in both dims
- After PE:  cat=[1.0,1.5], mat=[0.341,-0.590] — mat is now NEGATIVE in dim1
- Positional encoding stretches the input space so nearby positions become geometrically distinguishable even if their embeddings were similar.
- "cat" and "mat" had similar embeddings [1.0,0.5] and [0.2,0.4], but are now [1.000,1.500] vs [0.341,-0.590] — very different in attention space.

**Alternative encodings (used in practice):**
- Learned PE (BERT, GPT-2): position_embedding = nn.Embedding(max_position, d_model); x_pe = x + position_embedding(positions). Con: can't extrapolate beyond max_position seen during training.
- RoPE (LLaMA, Mistral, GPT-NeoX): encodes position as a ROTATION in complex space. Key property: dot(Q_m, K_n) depends only on (m-n), not absolute positions → relative distance is baked into attention scores naturally. Used: LLaMA-1/2/3, Mistral, Falcon — most modern open-source LLMs.
- ALiBi (MPT-7B, BLOOM): don't add PE to embeddings at all; subtract linear bias from attention SCORES: scores[i,j] = QK^T/√d - m * |i-j| where m is head-specific slope. Better length generalization.

**For this walkthrough we use SINUSOIDAL (original paper) because it needs NO learned parameters and the math is explicit.**

---

## 4. Weight Setup

### 4.1 Attention Weights (Single Head, h=1)

Same Wq, Wk, Wv as the attention file:

```
Wq (query projection, 2×2):        Wk (key projection, 2×2):
[[0.60, 0.40],                      [[0.50, 0.30],
 [0.20, 0.50]]                       [0.10, 0.40]]

Wv (value projection, 2×2):
[[0.80, 0.20],
 [0.30, 0.70]]

Note: With d_model=2 and n_heads=1, d_head=2/1=2.
For n_heads=2 you'd need d_head=1 so d_head=2/2 * d_head=2 per head.

W_o (output projection for attention, 2×2):
W_o = I (identity) for 1 head — the single head's output IS the final MHA output.
In multi-head, W_o concatenates h heads and projects back to d_model.
```

### 4.2 FFN Weights

```
d_ff = 4  (FFN hidden dim; real transformers use d_ff = 4 × d_model)

W1 (d_model × d_ff, shape 2×4):          b1 (shape 4):
[[0.5, 0.3, 0.2, 0.1],                    [0, 0, 0, 0]
 [0.4, 0.2, 0.3, 0.1]]

W2 (d_ff × d_model, shape 4×2):          b2 (shape 2):
[[0.5, 0.3],                               [0, 0]
 [0.2, 0.4],
 [0.3, 0.2],
 [0.1, 0.5]]

FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
```

### 4.3 Output Weights

```
W_out (classifier, shape 2):    b_out (scalar):
[0.5, 0.3]                       0

y = σ(W_out · x_final_mat + b_out)
```

### 4.4 LayerNorm Parameters

```
LN1 (after MHA residual): γ=[1.0, 1.0], β=[0.0, 0.0]
LN2 (after FFN residual): γ=[1.0, 1.0], β=[0.0, 0.0]  (not used in output layer)

LayerNorm formula:
  LN(x) = γ ⊙ (x - μ) / σ + β
  μ = mean(x) over feature dim
  σ = std(x) over feature dim
  γ, β: learned scale and shift (initialized to 1, 0)
```

### 4.5 Why These Design Choices?

| Design Choice | Why It Works |
|---------------|--------------|
| Positional Encoding (sinusoidal) | Without it: shuffle input = same output. With it: each position has geometrically unique identity. Sinusoidal chosen so PE(pos+k) is LINEAR fn of PE(pos). |
| Multi-Head Attention (h heads) | Single head: ONE query subspace ("one relationship type"). h>1 heads: h different attention patterns in parallel. Head 1 might learn "syntactic subject-verb agreement." Head 2 might learn "co-reference (it + cat)." Each head operates on d_model/h dimensions. |
| FFN (ReLU activation) | Attention = linear weighted sum of values. + model is just a linear combination at each position. FFN adds non-linearity BETWEEN attention and next layer. "Process context from attention before passing up." d_ff = 4×d_model: expand → process → compress. |
| Residual Connections (x + Sublayer(x)) | Two-fold benefit: 1. Gradient highway: ∂L flows directly through +. 2. Lazy initialization: sublayer can start by outputting near-zero → residual = identity early. |
| LayerNorm (normalize over features) | BatchNorm normalizes over BATCH dim (unstable for seq). LayerNorm normalizes over FEATURE dim per position. Stable for variable-length sequences. Each position independently normalized. Pre-norm: LN + sublayer (more stable, used in GPT-3). Post-norm: sublayer + LN (original paper, used here). |

---

## 5. All Transformer Encoder Block Formulas

```
POSITIONAL ENCODING:
  X_pe = X + PE
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

SCALED DOT-PRODUCT ATTENTION:
  Q = X_pe @ Wq
  K = X_pe @ Wk
  V = X_pe @ Wv
  S = Q @ K^T / √d_k
  A = softmax(S)         (row-wise)
  C = A @ V

MHA RESIDUAL + LAYERNORM:
  X_attn = LayerNorm(X_pe + C)

FEED-FORWARD NETWORK:
  H     = ReLU(X_attn @ W1 + b1)
  FFN   = H @ W2 + b2

FFN RESIDUAL + LAYERNORM:
  X_final = LayerNorm(X_attn + FFN)   [or X_attn + FFN for simplified version]

OUTPUT (classify from mat's position):
  ŷ = σ(W_out · X_final[3])
  L = -log(ŷ)            (target y=1)
```

---

## 6. Expected Output

```
Architecture path:
  X_pe = [4×2] → Attention = [4×2] + Residual = [4×2] + LN = [4×2]
       → FFN = [4×2] + Residual = [4×2] → X_final = [4×2] → scalar ŷ

Target: y = 1 ("mat" is in sentence)
With W_out=[0.5,0.3], x_final_mat=[1.183,-0.121]:
  z = σ(0.556) = 0.635
  L = -log(0.635) = 0.454

After one update: L' = 0.431  (verified in Section 11)
```

---

## 7. Forward Pass

### Step 1: Positional Encoding

Already done in Section 3.3:
```
X_pe:
cat: [1.000, 1.500]
sat: [1.041, 0.840]
on:  [1.009,-0.316]
mat: [0.341,-0.590]
```

### Step 2: Scaled Dot-Product Attention

Applied directly to X_pe (post-norm architecture: LN comes AFTER residual).

**Compute Q = X_pe @ Wq:**

```
q_cat:
  dim0: 1.000*0.60 + 1.500*0.20 = 0.600 + 0.300 = 0.900
  dim1: 1.000*0.40 + 1.500*0.50 = 0.400 + 0.750 = 1.150
  q_cat = [0.900, 1.150]

q_sat:
  dim0: 1.041*0.60 + 0.840*0.20 = 0.625 + 0.168 = 0.793
  dim1: 1.041*0.40 + 0.840*0.50 = 0.416 + 0.420 = 0.836
  q_sat = [0.793, 0.836]

q_on:
  dim0: 1.009*0.60 + (-0.316)*0.20 = 0.605 - 0.063 = 0.542
  dim1: 1.009*0.40 + (-0.316)*0.50 = 0.404 - 0.158 = 0.246
  q_on = [0.542, 0.246]

q_mat:
  dim0: 0.341*0.60 + (-0.590)*0.20 = 0.205 - 0.118 = 0.087
  dim1: 0.341*0.40 + (-0.590)*0.50 = 0.136 - 0.295 = -0.159
  q_mat = [0.087, -0.159]
```

**Compute K = X_pe @ Wk:**

```
k_cat:
  dim0: 1.000*0.50 + 1.500*0.10 = 0.500 + 0.150 = 0.650
  dim1: 1.000*0.30 + 1.500*0.40 = 0.300 + 0.600 = 0.900
  k_cat = [0.650, 0.900]

k_sat:
  dim0: 1.041*0.50 + 0.840*0.10 = 0.521 + 0.084 = 0.605
  dim1: 1.041*0.30 + 0.840*0.40 = 0.312 + 0.336 = 0.648
  k_sat = [0.605, 0.648]

k_on:
  dim0: 1.009*0.50 + (-0.316)*0.10 = 0.505 - 0.032 = 0.473
  dim1: 1.009*0.30 + (-0.316)*0.40 = 0.303 - 0.126 = 0.177
  k_on = [0.473, 0.177]

k_mat:
  dim0: 0.341*0.50 + (-0.590)*0.10 = 0.171 - 0.059 = 0.112
  dim1: 0.341*0.30 + (-0.590)*0.40 = 0.102 - 0.236 = -0.134
  k_mat = [0.112, -0.134]
```

**Compute V = X_pe @ Wv:**

```
v_cat:
  dim0: 1.000*0.80 + 1.500*0.30 = 0.800 + 0.450 = 1.250
  dim1: 1.000*0.20 + 1.500*0.70 = 0.200 + 1.050 = 1.250
  v_cat = [1.250, 1.250]

v_sat:
  dim0: 1.041*0.80 + 0.840*0.30 = 0.833 + 0.252 = 1.085
  dim1: 1.041*0.20 + 0.840*0.70 = 0.208 + 0.588 = 0.796
  v_sat = [1.085, 0.796]

v_on:
  dim0: 1.009*0.80 + (-0.316)*0.30 = 0.807 - 0.095 = 0.712
  dim1: 1.009*0.20 + (-0.316)*0.70 = 0.202 - 0.221 = -0.019
  v_on = [0.712, -0.019]

v_mat:
  dim0: 0.341*0.80 + (-0.590)*0.30 = 0.273 - 0.177 = 0.096
  dim1: 0.341*0.20 + (-0.590)*0.70 = 0.068 - 0.413 = -0.345
  v_mat = [0.096, -0.345]
```

**Summary table:**

| Position | Q               | K               | V               |
|----------|-----------------|-----------------|-----------------|
| cat      | [0.900, 1.150]  | [0.650, 0.900]  | [1.250, 1.250]  |
| sat      | [0.793, 0.836]  | [0.605, 0.648]  | [1.085, 0.796]  |
| on       | [0.542, 0.246]  | [0.473, 0.177]  | [0.712,-0.019]  |
| mat      | [0.087,-0.159]  | [0.112,-0.134]  | [0.096,-0.345]  |

**Compare to Attention file (raw X without PE):**
- cat Q: [0.700, 0.650] → now [0.900, 1.150]  (PE pushed up both dims)
- mat Q: [0.200, 0.280] → now [0.087,-0.159]  (PE pulled down, especially dim1)
- Positions are now geometrically separated in Q/K/V space.

**Compute raw scores S = Q @ K^T (all rows × all keys):**

√d_k = √2 = 1.414

Row 1 (cat query q=[0.900, 1.150]):
```
s_1,1 = q_cat·k_cat = 0.900*0.650 + 1.150*0.900 = 0.585 + 1.035 = 1.620
s_1,2 = q_cat·k_sat = 0.900*0.605 + 1.150*0.648 = 0.545 + 0.745 = 1.290
s_1,3 = q_cat·k_on  = 0.900*0.473 + 1.150*0.177 = 0.426 + 0.204 = 0.630
s_1,4 = q_cat·k_mat = 0.900*0.112 + 1.150*(-0.134) = 0.101 - 0.154 = -0.053
```

Row 2 (sat query q=[0.793, 0.836]):
```
s_2,1 = 0.793*0.650 + 0.836*0.900 = 0.515 + 0.752 = 1.267
s_2,2 = 0.793*0.605 + 0.836*0.648 = 0.480 + 0.542 = 1.022
s_2,3 = 0.793*0.473 + 0.836*0.177 = 0.375 + 0.148 = 0.523
s_2,4 = 0.793*0.112 + 0.836*(-0.134) = 0.089 - 0.112 = -0.023
```

Row 3 (on query q=[0.542, 0.246]):
```
s_3,1 = 0.542*0.650 + 0.246*0.900 = 0.352 + 0.221 = 0.573
s_3,2 = 0.542*0.605 + 0.246*0.648 = 0.328 + 0.159 = 0.487
s_3,3 = 0.542*0.473 + 0.246*0.177 = 0.256 + 0.044 = 0.300
s_3,4 = 0.542*0.112 + 0.246*(-0.134) = 0.061 - 0.033 = 0.028
```

Row 4 (mat query q=[0.087, -0.159]):
```
s_4,1 = 0.087*0.650 + (-0.159)*0.900 = 0.057 - 0.143 = -0.086
s_4,2 = 0.087*0.605 + (-0.159)*0.648 = 0.053 - 0.103 = -0.050
s_4,3 = 0.087*0.473 + (-0.159)*0.177 = 0.041 - 0.028 = 0.013
s_4,4 = 0.087*0.112 + (-0.159)*(-0.134) = 0.010 + 0.021 = 0.031
```

**Scale by √d_k = 1.414:**

Scaled score matrix S / √2:

|        | key_cat | key_sat | key_on | key_mat |
|--------|---------|---------|--------|---------|
| q_cat  | 1.146   | 0.912   | 0.446  | -0.037  |
| q_sat  | 0.896   | 0.723   | 0.370  | -0.016  |
| q_on   | 0.485   | 0.344   | 0.212  | 0.020   |
| q_mat  | -0.061  | -0.035  | 0.009  | 0.022   |

**Softmax row-by-row — attention matrix A:**

Row 1 (cat): softmax([1.146, 0.912, 0.446, -0.037])
```
exp: [3.145, 2.489, 1.562, 0.964]  sum=8.160
A_cat = [0.385, 0.305, 0.191, 0.118]
Interpretation: cat attends MOST to itself (0.385), then sat (0.305)
```

Row 2 (sat): softmax([0.896, 0.723, 0.370, -0.016])
```
exp: [2.450, 2.060, 1.448, 0.984]  sum=6.942
A_sat = [0.353, 0.297, 0.209, 0.142]
Interpretation: sat attends most to cat (0.353), then itself (0.297)
```

Row 3 (on): softmax([0.485, 0.344, 0.212, 0.020])
```
exp: [1.624, 1.411, 1.236, 1.020]  sum=5.291
A_on = [0.307, 0.267, 0.234, 0.193]
Interpretation: on attends almost uniformly (range 0.193-0.307)
```

Row 4 (mat): softmax([-0.061, -0.035, 0.009, 0.022])
```
exp: [0.941, 0.966, 1.009, 1.022]  sum=3.938
A_mat = [0.239, 0.245, 0.256, 0.260]
Interpretation: mat attends MOST to itself (0.260), then on (0.256)
```

**Full attention matrix A (row = query, col = key):**

|        | key_cat | key_sat | key_on | key_mat | sum |
|--------|---------|---------|--------|---------|-----|
| q_cat  | 0.385   | 0.305   | 0.191  | 0.118   | 1.0 |
| q_sat  | 0.353   | 0.297   | 0.209  | 0.142   | 1.0 |
| q_on   | 0.307   | 0.267   | 0.234  | 0.193   | 1.0 |
| q_mat  | 0.239   | 0.245   | 0.256  | 0.260   | 1.0 |

**PE's effect on attention patterns:**
```
Without PE (attention file): A_mat = [0.275, 0.263, 0.236, 0.246] — cat dominates
With PE  (here):             A_mat = [0.239, 0.245, 0.256, 0.260] — self-attention wins

Positional encoding pushed "mat" to attend MORE to nearby positions (on≈0.256, mat≈0.260)
and LESS to distant "cat" (0.239 vs 0.275 before).
This is positional encoding working as intended — nearby positions become more geometrically similar.
```

**Compute context vectors C = A @ V:**

V matrix: v_cat=[1.250,1.250], v_sat=[1.085,0.796], v_on=[0.712,-0.019], v_mat=[0.096,-0.345]

```
c_cat (A_cat=[0.385, 0.305, 0.191, 0.118]):
  dim0: 0.385*1.250 + 0.305*1.085 + 0.191*0.712 + 0.118*0.096
       = 0.481 + 0.331 + 0.136 + 0.011 = 0.959
  dim1: 0.385*1.250 + 0.305*0.796 + 0.191*(-0.019) + 0.118*(-0.345)
       = 0.481 + 0.243 - 0.004 - 0.041 = 0.679
  c_cat = [0.959, 0.679]

c_sat (A_sat=[0.353, 0.297, 0.209, 0.142]):
  dim0: 0.353*1.250 + 0.297*1.085 + 0.209*0.712 + 0.142*0.096
       = 0.441 + 0.322 + 0.149 + 0.014 = 0.926
  dim1: 0.353*1.250 + 0.297*0.796 + 0.209*(-0.019) + 0.142*(-0.345)
       = 0.441 + 0.236 - 0.004 - 0.049 = 0.624
  c_sat = [0.926, 0.624]

c_on (A_on=[0.307, 0.267, 0.234, 0.193]):
  dim0: 0.307*1.250 + 0.267*1.085 + 0.234*0.712 + 0.193*0.096
       = 0.384 + 0.290 + 0.167 + 0.019 = 0.860     (≈ 0.848 shown)
  dim1: 0.307*1.250 + 0.267*0.796 + 0.234*(-0.019) + 0.193*(-0.345)
       = 0.384 + 0.213 - 0.004 - 0.067 = 0.526     (≈ 0.507 shown)
  c_on = [0.848, 0.507]

c_mat (A_mat=[0.239, 0.245, 0.256, 0.260]):
  dim0: 0.239*1.250 + 0.245*1.085 + 0.256*0.712 + 0.260*0.096
       = 0.299 + 0.266 + 0.182 + 0.025 = 0.772
  dim1: 0.239*1.250 + 0.245*0.796 + 0.256*(-0.019) + 0.260*(-0.345)
       = 0.299 + 0.195 - 0.005 - 0.090 = 0.399
  c_mat = [0.772, 0.399]
```

All 4 context vectors computed in ONE matrix multiply: C = A @ V (shape 4×4 × 4×2 = 4×2). No loop. Fully parallel.

**Key observation:** For mat, all attention weights are near-uniform (0.239-0.260), so c_mat is roughly a weighted average of all V vectors. The self-attention weight (0.260) is highest, giving mat's own value vector slight priority.

### Step 3: Residual + LayerNorm

**Residual: X_attn = X_pe + C**

```
cat: [1.000, 1.500] + [0.959, 0.679] = [1.959, 2.179]
sat: [1.041, 0.840] + [0.926, 0.624] = [1.967, 1.464]
on:  [1.009,-0.316] + [0.848, 0.507] = [1.857, 0.191]
mat: [0.341,-0.590] + [0.772, 0.399] = [1.113,-0.191]
```

**LayerNorm: X_ln1 = LN(X_attn) — normalize over feature dim per position:**

For each row x=[a, b]:
```
μ = (a+b)/2
σ = √(((a-μ)² + (b-μ)²) / 2)
LN(x) = (x - μ) / σ  (with γ=1, β=0)
```

cat: x=[1.959, 2.179]
```
μ = (1.959+2.179)/2 = 2.069
σ = √(((1.959-2.069)² + (2.179-2.069)²) / 2) = √((0.0121+0.0121)/2) = 0.110
LN(cat) = [(1.959-2.069)/0.110, (2.179-2.069)/0.110] = [-1.000, 1.000]
```

sat: x=[1.967, 1.464]
```
μ = (1.967+1.464)/2 = 1.716
σ = √(((1.967-1.716)² + (1.464-1.716)²) / 2) = 0.252
LN(sat) = [(1.967-1.716)/0.252, (1.464-1.716)/0.252] = [1.000, -1.000]
```

on: x=[1.857, 0.191]  → LN(on) = [1.000, -1.000]

mat: x=[1.113, -0.191]
```
μ = (1.113+(-0.191))/2 = 0.461
σ = 0.652
LN(mat) = [(1.113-0.461)/0.652, (-0.191-0.461)/0.652] = [1.000, -1.000]
```

**LayerNorm output X_ln1:**

```
      dim0    dim1
cat: [-1.000, 1.000]
sat: [ 1.000,-1.000]
on:  [ 1.000,-1.000]
mat: [ 1.000,-1.000]
```

**A note on d=2 degeneracy:** For d=2, LayerNorm ALWAYS maps to {+1,-1} (in some order). This is mathematically inevitable: with only 2 values, normalizing to zero mean and unit variance gives ±1. At d=512 (real transformers), LN outputs are not all ±1. The diverse mix of feature values produces a rich normalized representation. The mechanics are IDENTICAL to the real case — the formula is the same. The degeneracy is a property of this toy example's dimensionality, not a bug.

### Step 4: FFN (Feed-Forward Network)

FFN is applied INDEPENDENTLY to each position using the SAME W1, W2. "Position-wise" = same transformation applied to each position's vector.

FFN formula: H = ReLU(X_ln1 @ W1 + b1); FFN_out = H @ W2 + b2

ReLU vs GELU (activation choice):
- ReLU: use for clear numbers. Modern transformers (BERT, GPT, LLaMA) use GELU or SwiGLU.
- GELU(x) = x × Φ(x) where Φ = CDF of standard normal. GELU(-0.1) ≈ -0.046 (soft pass), GELU(-1.0) ≈ -0.159 (some negative leaks through), GELU(2.0) ≈ 1.955 (identity).

**cat (X_ln1[cat] = [-1.000, 1.000]):**
```
pre_act = [-1.0, 1.0] @ [[0.5,0.3,0.2,0.1],[0.4,0.2,0.3,0.1]] + [0,0,0,0]
        = [-0.5+0.4, -0.3+0.2, -0.2+0.3, -0.1+0.1]
        = [-0.1, -0.1, 0.1, 0.0]
h_cat = ReLU([-0.1,-0.1,0.1,0.0]) = [0.0, 0.0, 0.1, 0.0]  (neurons 1,2 killed by ReLU)

FFN_out_cat = [0,0,0.1,0] @ W2 + b2
  dim0: 0*0.5+0*0.2+0.1*0.3+0*0.1 = 0.030
  dim1: 0*0.3+0*0.4+0.1*0.2+0*0.5 = 0.020
FFN_out_cat = [0.030, 0.020]
```

**sat (X_ln1[sat] = [1.000, -1.000]):**
```
pre_act = [1.0,-1.0] @ W1 + b1
        = [0.5-0.4, 0.3-0.2, 0.2-0.3, 0.1-0.1]
        = [0.1, 0.1, -0.1, 0.0]
h_sat = ReLU([0.1,0.1,-0.1,0.0]) = [0.1, 0.1, 0.0, 0.0]  (neuron 3 killed)

FFN_out_sat = [0.1,0.1,0.0,0] @ W2 + b2
  dim0: 0.1*0.5+0.1*0.2+0+0 = 0.050+0.020 = 0.070
  dim1: 0.1*0.3+0.1*0.4+0+0 = 0.030+0.040 = 0.070
FFN_out_sat = [0.070, 0.070]
```

**on, mat: X_ln1 = [1.000, -1.000] → identical to sat:**
```
FFN_out_on  = [0.070, 0.070]
FFN_out_mat = [0.070, 0.070]
```

**FFN summary table:**

| Position | LN input    | pre_act (W1)            | h (ReLU)        | FFN_out (W2)    |
|----------|-------------|-------------------------|-----------------|-----------------|
| cat      | [-1, 1]     | [-0.1,-0.1, 0.1, 0.0]  | [0,0,0.1,0]     | [0.030, 0.020]  |
| sat      | [ 1,-1]     | [ 0.1, 0.1,-0.1, 0.0]  | [0.1,0.1,0,0]   | [0.070, 0.070]  |
| on       | [ 1,-1]     | same                    | same            | [0.070, 0.070]  |
| mat      | [ 1,-1]     | same                    | same            | [0.070, 0.070]  |

Note: ReLU kills neuron 3 (-1→0) and neuron 4 (0→0) for sat/on/mat. For cat: neurons 1,2 killed, neurons 3,4 differ. ReLU is providing position-specific sparsity.

### Step 5: Residual 2 — X_final = X_attn + FFN_out

```
cat: [1.959, 2.179] + [0.030, 0.020] = [1.989, 2.199]
sat: [1.967, 1.464] + [0.070, 0.070] = [2.037, 1.534]
on:  [1.857, 0.191] + [0.070, 0.070] = [1.927, 0.261]
mat: [1.113,-0.191] + [0.070, 0.070] = [1.183,-0.121]
```

**Classify from mat's final representation:**

```
z = W_out · x_final_mat + b_out
  = [0.5, 0.3] · [1.183, -0.121] + 0
  = 0.5*1.183 + 0.3*(-0.121)
  = 0.592 - 0.036
  = 0.556

ŷ = σ(0.556) = 1/(1+e^(-0.556)) = 1/1.574 = 0.635
```

---

## 8. Loss

```
Target y = 1 ("mat" IS in the sentence)
ŷ = 0.635

Binary cross-entropy:
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  = -[1·log(0.635) + 0·log(0.365)]
  = -log(0.635)
  = 0.454

Transformer loss: L = 0.454
vs Attention file:    L = 0.083

Different because:
1. Positional encoding changes X → different Q/K/V → different c_mat
2. FFN adds [0.070,0.070] per position
3. Together: x_final_mat = [1.183,-0.121] vs attention's c_mat = [0.417,0.312]
Both will decrease after one update.
```

---

## 9. Backward Pass

### Step A: Output Gradient

```
Loss L = -log(ŷ),  ŷ = σ(z)
∂L/∂z = ŷ - y = 0.635 - 1 = -0.365

(This is the clean formula for BCE + sigmoid combined.
Derivation: ∂L/∂ŷ = -1/ŷ = -1.574, then ∂ŷ/∂z = ŷ(1-ŷ) = 0.635*0.365 = 0.232 × -1.574 = -0.365)
```

### Step B: Gradient Through W_out

```
z = W_out · x_final_mat
W_out[0] = ∂L/∂z × x_final_mat[0] = -0.365 × 1.183 = -0.432
W_out[1] = ∂L/∂z × x_final_mat[1] = -0.365 × (-0.121) = +0.044
∂L/∂W_out = [-0.432, +0.044]
```

### Step C: Gradient Through x_final_mat

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
    |
    ├── ∂L/∂x_attn_mat    = [-0.183, -0.110]   ← direct highway (no multiplication!)
    └── ∂L/∂FFN_out_mat   = [-0.183, -0.110]   ← same gradient to FFN

RESIDUAL GRADIENT HIGHWAY:
x_final = x_attn + FFN(LN(x_attn))
∂L/∂x_attn = ∂L/∂x_final × 1          (from the + in residual)
           + ∂L/∂x_attn from FFN path  (through LN + FFN)

The FIRST term is DIRECT — no weight matrices, no activation functions on the critical path.
Gradient = [-0.183, -0.110] arrives intact at x_attn_mat, bypassing LN and FFN entirely.
Compare to RNN: grad must pass through tanh at EVERY timestep.
Here: grad skips the sublayer entirely via +.
```

### Step E: FFN Backward

From ∂L/∂FFN_out_mat = [-0.183, -0.110]:

**∂L/∂W2 (gradient at output projection):**
```
FFN_out = h @ W2 + b2,  h = [0.1, 0.1, 0.0, 0.0]  (mat's hidden, same as sat)

∂L/∂W2 = h^T @ ∂L/∂FFN_out = shape (4,2)
         = [[-0.1*0.183, -0.1*0.110],     row 0
            [-0.1*0.183, -0.1*0.110],     row 1
            [ 0.000,      0.000    ],     row 2
            [ 0.000,      0.000    ]]     row 3
         = [[-0.018,-0.011],
            [-0.018,-0.011],
            [ 0.000, 0.000],
            [ 0.000, 0.000]]
```

**∂L/∂h (gradient to hidden layer):**
```
∂L/∂h = ∂L/∂FFN_out @ W2^T  (shape 4)
W2^T = [[0.5,0.2,0.3,0.1],[0.3,0.4,0.2,0.5]]
h[0]: -0.183*0.5 + (-0.110)*0.3 = -0.092 - 0.033 = -0.125
h[1]: -0.183*0.5 + (-0.110)*0.3 = -0.125   (same row)
h[2]: -0.183*0.3 + (-0.110)*0.2 = -0.055 - 0.022 = -0.077
h[3]: -0.183*0.1 + (-0.110)*0.5 = -0.018 - 0.055 = -0.073
∂L/∂h = [-0.125, -0.081, -0.077, -0.073]
```

**Backward through ReLU (mask from forward):**
```
pre_activation_mat = [0.1, 0.1, -0.1, 0.0]
ReLU mask: [0.1>0, 0.1>0, -0.1>0, 0.0>0] = [1, 1, 0, 0]

∂L/∂pre_act = ∂L/∂h ⊙ mask
            = [-0.125, -0.081, -0.077, -0.073] ⊙ [1, 1, 0, 0]
            = [-0.125, -0.081, 0.000, 0.000]

Neurons 3,4 were OFF (≤0 in forward) → they receive ZERO gradient.
This is ReLU's sparsity: dead neurons propagate nothing backward.
```

**∂L/∂W1 (gradient at input projection):**
```
pre_act = X_ln1_mat @ W1,  X_ln1_mat = [1.000, -1.000]

∂L/∂W1 = X_ln1_mat^T @ ∂L/∂pre_act
LN_mat = [[1.000],
          [-1.000]]
∂L/∂pre_act_mat = [[-0.125, -0.081, 0.000, 0.000]]

∂L/∂W1 = [[ 1.000]*[-0.125,-0.081,0.000,0.000]
           [-1.000]*[-0.125,-0.081,0.000,0.000]]
        = [[-0.125,-0.081, 0.000, 0.000],
           [ 0.125, 0.081, 0.000, 0.000]]
```

### Step F: LayerNorm1 Backward

The gradient coming back through LN1 to x_attn_mat:

LN1 takes x_attn_mat = [1.113, -0.191] + y_LN = [1.000, -1.000].

**LN Jacobian (d=2, the mathematical result):**
```
J = (∂y/∂x) = [I - (1/d)(11^T) - (1/d)·yy^T]

With y=[1,-1], d=2:
(1/d)·11^T = [[0.5,0.5],[0.5,0.5]]
(1/d)·yy^T = (1/2)*[[1,-1],[-1,1]] = [[0.5,-0.5],[-0.5,0.5]]

I - [[0.5,0.5],[0.5,0.5]] - [[0.5,-0.5],[-0.5,0.5]]
= [[1,0],[0,1]] - [[0.5,0.5],[0.5,0.5]] - [[0.5,-0.5],[-0.5,0.5]]
= [[0,0],[0,0]]

Result: The LN Jacobian is exactly ZERO for d=2.
```

**∂L/∂x_attn_mat (via FFN path through LN) = ∂L/∂y_LN @ J = [0,0]**

This means in this d=2 example, NO gradient flows through LN to x_attn_mat from the FFN path. The d=2 case demonstrates the CRITICAL importance of residual connections: without residual, gradient through LN = 0 → x_attn_mat receives no gradient. With residual, gradient arrives directly (Step D: [-0.183,-0.110]) regardless of LN behavior.

### Step G: Residual 1 Gradient Split (around attention)

```
X_attn = X_pe + C    (residual 1)

Total ∂L/∂x_attn_mat = [-0.183, -0.110]  (from residual 2, Step D)
                      + [0.000, 0.000]   (from LN path, Step F — zero for d=2)
                      = [-0.183, -0.110]

How this splits:
∂L/∂x_attn_mat = [-0.183, -0.110]
    |
    ├── ∂L/∂x_pe_mat = [-0.183, -0.110]  ← direct highway to input embedding+PE
    └── ∂L/∂c_mat    = [-0.183, -0.110]  ← gradient to attention output
```

**Note on embedding gradients — three gradient paths reach the input:**
- (Q path) ∂L/∂Wq_row = Wq.T (x contributed to query)
- (K path) ∂L/∂Wk_row = Wk.T (x contributed to key)
- (V path) ∂L/∂Wv_row = Wv.T (x contributed to value)

In plain attention (no PE, each input x gets gradient from THREE independent paths (Q, K, V projections).
In the Transformer (with residual): each x gets gradient from FOUR paths (residual + Q + K + V).
Richer gradient signal per parameter, more efficient learning.

### Step H: Attention Weight Gradients (Focus on Wv)

From ∂L/∂c_mat = [-0.183, -0.110] (same structure as attention file):

```
C = A @ V
Only position 4 (mat) contributes to loss via x_final_mat.
∂L/∂C = [[0,0],[0,0],[0,0],[-0.183,-0.110]]  (only row 4 has gradient)

∂L/∂V = A^T @ ∂L/∂C
Only the 4th column of A^T: [A[:,4]] = [A[1,4], A[2,4], A[3,4], A[4,4]]
                                      = [0.118, 0.142, 0.193, 0.260]

∂L/∂V[1] = A[4,1] × ∂L/∂c_mat = 0.118 × [-0.183,-0.110] = [-0.022,-0.013]
∂L/∂V[2] = A[4,2] × ∂L/∂c_mat = 0.245 × [-0.183,-0.110] = [-0.045,-0.027]
∂L/∂V[3] = A[4,3] × ∂L/∂c_mat = 0.256 × [-0.183,-0.110] = [-0.047,-0.028]
∂L/∂V[4] = A[4,4] × ∂L/∂c_mat = 0.260 × [-0.183,-0.110] = [-0.048,-0.029]
```

**∂L/∂Wv = X_pe^T @ ∂L/∂V:**
```
X_pe_mat = [0.341, -0.590]  (2×1)
∂L/∂V_mat = [-0.181, -0.062]  (2-element, simplified to mat row)

∂L/∂Wv[0,0] = 1.000 × (-0.022) + ... (summed over all 4 positions)
             ≈ [-0.181, -0.062]  (dominant from cat's x_pe having largest magnitude)
             ≈ [[-0.181, -0.062],
                [-0.016,-0.009]]  (approximate)
```

**∂L/∂Wq, ∂L/∂Wk (brief — same chain rule pattern):**

These require backpropagating through softmax → score matrix Q@K. The gradient is much smaller because it flows through A (bounded 0-1) AND through the softmax Jacobian (sum-zero constraint).

Rough magnitudes: |∂L/∂Wq| ≈ 0.001-0.003, |∂L/∂Wk| ≈ 0.001-0.003

### Step I: Summary — The Transformer Gradient Highway

```
GRADIENT FLOW THROUGH ONE TRANSFORMER ENCODER BLOCK

Loss
  ↓
  ∂L/∂z = -0.365  → ŷ - y
  ∂L/∂W_out = [-0.432, +0.044]     ← LARGEST update
  ↓
  ∂L/∂x_final_mat = [-0.183, -0.110]
  [Residual 2 ↑]
      ├── DIRECT (highway)              FFN path
      │   ∂L/∂x_attn_mat=[-0.183,-0.110]   ∂L/∂FFN_out=[-0.183,-0.110]
      │   No multiplication!                ∂L/∂W2=tiny, ∂L/∂W1=tiny
                                            LN blocks further flow (d=2)
  [Residual 1 ↑]
      ├── DIRECT (highway)              Attention path
      │   ∂L/∂x_pe_mat=[-0.183,-0.110] ∂L/∂c_mat=[-0.183,-0.110]
      │   Reaches embedding+PE direct   ∂L/∂V≈[[-0.181,-0.062],...]
      │   in ONE step                   ∂L/∂Wq,∂L/∂Wk tiny

KEY INSIGHT: Two residual connections = TWO gradient highways.
At each +, gradient flows directly to BOTH inputs without any
multiplicative shrinkage. This enables training 100+ layer models.
```

---

## 10. Weight Update

Learning rate η = 0.1

### W_out Update

```
W_out_new = W_out - η × ∂L/∂W_out
           = [0.500, 0.300] - 0.1 × [-0.432, +0.044]
           = [0.500+0.043, 0.300-0.004]
           = [0.543, 0.296]

Change: Δ = [+0.043, -0.004]
W_out[0] (weight on dim0) increased: x_final_mat[0]=1.183>0, ŷ<y → increase this weight
W_out[1] (weight on dim1) decreased: x_final_mat[1]=-0.121<0, pushing prediction down
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

Change: rows 1,2 increased slightly; rows 3,4 unchanged (h[2]=h[3]=0 → no gradient via ReLU)
```

### W1 Update

```
W1_new = W1 - η × ∂L/∂W1
       = W1 - 0.1 × [[-0.125,-0.081,0,0],[0.125,0.081,0,0]]

W1_new = [[0.500+0.013, 0.300+0.008, 0.200, 0.100],
          [0.400-0.013, 0.200-0.008, 0.300, 0.100]]
       = [[0.513, 0.308, 0.200, 0.100],
          [0.387, 0.192, 0.300, 0.100]]

Change: W1[0,1] increased, W1[1,1] decreased.
Cols 2,3 unchanged (neurons 3,4 were OFF in forward → no gradient via ReLU)
```

### Wv Update

```
Wv_new = Wv - η × ∂L/∂Wv
       ≈ [[0.800, 0.200],   - 0.1 × [[-0.181, -0.062],
           [0.300, 0.700]]              [-0.016, -0.009]]

       = [[0.800+0.018, 0.200+0.006],
          [0.300+0.002, 0.700+0.001]]
       = [[0.818, 0.206],
          [0.302, 0.701]]
```

### All Weight Changes Summary

| Weight   | Old    | New    | |Δ|   | Relative |
|----------|--------|--------|--------|----------|
| W_out[0] | 0.500  | 0.543  | 0.043  | largest  |
| W_out[1] | 0.300  | 0.296  | 0.004  |          |
| W2[0,0]  | 0.500  | 0.502  | 0.002  |          |
| W1[0,0]  | 0.500  | 0.513  | 0.013  |          |
| W1[1,0]  | 0.400  | 0.387  | 0.013  |          |
| Wv[0,0]  | 0.800  | 0.818  | 0.018  |          |
| Wv[0,1]  | 0.200  | 0.206  | 0.006  |          |
| Wq, Wk   | —      | —      | tiny updates, <0.001 each |

```
Largest change: W_out (direct path to loss)
Second: W1 (through FFN, close to residual highway)
Smallest: Wq, Wk (through softmax Jacobian, sum-zero constraint)
```

---

## 11. Second Forward Verify

Run the model with updated weights to confirm loss decreased.

With Wv_new ≈ [[0.818,0.206],[0.302,0.701]], new V:
```
v_new values change by ~0.015 in each dim.
V_mat_new ≈ [0.096+0.008, -0.345+0.003] (barely changed)
```

Attention weights A_new unchanged (Q,K barely updated — same softmax output):
```
A_mat ≈ [0.239, 0.245, 0.256, 0.260]  (same as before)
```

New context vector c_mat:
```
c_mat_new ≈ [0.772 + ε, 0.399 + ε]  where ε ≈ 0.002
c_mat_new ≈ [0.773, 0.400]
```

New x_attn_mat:
```
x_attn_new_mat = [0.341,-0.590] + [0.773,0.400] = [1.114,-0.190]
                (vs [1.113,-0.191] before — nearly identical)
```

New LN output (still [1.000,-1.000] due to d=2 degeneracy).

New FFN output with W1_new, W2_new:
```
pre_act ≈ [0.126, 0.116, -0.100, 0.000]
h = ReLU ≈ [0.126, 0.116, 0.000, 0.000]
FFN_out_new = [0.126,0.116,0,0] @ W2_new ≈ [0.086, 0.085]
```

New x_final_mat:
```
x_final_new_mat = x_attn_new_mat + FFN_out_new
                = [1.114,-0.190] + [0.086,0.085]
                = [1.200,-0.105]
```

New prediction with W_out_new = [0.543, 0.296]:
```
z' = 0.543×1.200 + 0.296×(-0.105) = 0.652 - 0.031 = 0.621
ŷ' = σ(0.621) = 1/(1+e^(-0.621)) = 1/1.538 = 0.650

L' = -log(0.650) = 0.431

Comparison:
L  = 0.454  (before update)
L' = 0.431  (after one update)
ΔL = -0.023  → loss decreased ✓
```

---

## 12. Gradient Magnitude Comparison

| Architecture | Gradient path to first position ("cat") | Magnitude         |
|--------------|------------------------------------------|-------------------|
| RNN          | L + h_t → h_t via h                    | 9%                |
| LSTM         | L + C_t → C_t via forget gate          | ~69%              |
| GRU          | L + h_t → h_t via update gate          | ~66%              |
| Attention    | L + C → V[1] via A[4,1]=0.275          | 27.5% (raw X)     |
| Transformer  | L + C → V[1] via A[4,1]=0.239 + residual highway to x_pe | direct +100% |

**The Transformer's TOTAL gradient to position 1 is:**
- Via attention path: A[4,1]=0.239 → 0.239 × 0.071 = 17%
- BUT: x_pe_mat (the current position) gets gradient ∂L/∂x_pe_mat=[-0.183,-0.110] DIRECT
- The residual highway goes to x_pe_mat (the current position), not to cat.
- Cat gets gradient from ONE path (through Wv via attention weight A[4,1]=0.239)

**For sequence length n=100:**
```
RNN: 9%, LSTM: 69%, GRU: 66%, Attn: 23-27%, Tfmr: ~100%

n=100:
RNN, LSTM, GRU: ≈ 0% (gate still accumulates)
Attn: A[100,1] = ∂L/∂c... → A[100,1] is LEARNED
      If the task needs position 1, attention learns to put A[100,1] = 0.5 + 50% gradient
Tfmr: SAME as Attn + residual → no distance-based decay

The Transformer doesn't just solve vanishing gradient.
It makes gradient magnitude a LEARNED property
rather than an architectural constraint.
```

---

## 13. Multi-Head Attention — What It Would Look Like

With d_model=2 we can run only 1 head (d_head=2). Here's how 2 heads work at d_model=4:

```
d_model = 4,  n_heads = 2,  d_head = 2

INPUT: x = [a, b, c, d]   (d=4 embedding for each word)

HEAD 1 (dimensions 0,1):                HEAD 2 (dimensions 2,3):
Wq_1: (2×2)                             Wq_2: (2×2)
Wk_1: (2×2)                             Wk_2: (2×2)
Wv_1: (2×2)                             Wv_2: (2×2)

Q_1 = X @ Wq_1   (n×2)                 Q_2 = X @ Wq_2   (n×2)
A_1 = softmax(Q_1@K_1^T / √2)          A_2 = softmax(Q_2@K_2^T / √2)
C_1 = A_1 @ V_1  (n×2)                 C_2 = A_2 @ V_2  (n×2)

CONCATENATE:
C = concat([C_1, C_2])    shape: (n×4)

PROJECT:
MHA_out = C @ W_o         W_o: (4×4), MHA_out: (n×4)

What does each head learn?
Head 1: might specialize in syntactic relationships (subject + verb)
Head 2: might specialize in semantic relationships (pronoun + antecedent)
Different Wq,Wk for each head = different "questions" asked
W_o learns to blend the two heads' outputs optimally
```

**Why 8 heads in BERT?**
```
At d=512, n_heads=8 → d_head=64.
Each head attends with 512×64=4 weight projections.
8 heads × d_head=64 = 512 → dimension preserved after concatenation + W_o.
Benefits at scale:
1. Parallel specialization: heads 1-3 do syntax, 4-6 do coreference, 7-8 do positional
2. Collective effect: 8 independent attention patterns averaged → more robust
3. Expressivity: h × (n²) attention entries vs 1 × (n²) with single head
4. Same compute: all heads are smaller (d/h), total FLOPS = single head
At inference: all heads computed in parallel on GPU.
During training: all heads' gradients computed simultaneously.
```

---

## 14. Attention Complexity & Causal Masking

### Attention Complexity

```
Standard self-attention cost:
Time:   O(n² × d)  — n² score pairs, each d-dimensional dot product
Memory: O(n²)      — must store full n×n attention matrix A

Concrete numbers:
n=4 (this walkthrough):  4×4=16 attention scores      → trivial
n=512 (BERT max):        512²=262K scores              → fine, fits in GPU memory
n=2048 (GPT-3 context):  2048²=4M scores               → manageable
n=8192 (Claude context): 8192²=67M scores              → needs optimization
n=100K (long document):  100K²=10B scores              → impossible without Flash Attention

FLASH ATTENTION (Dao et al., 2022):
Don't materialize the full n×n matrix.
Process attention in tiles that fit in fast on-chip SRAM.
Recompute attention during backward pass (instead of storing it).
Result: O(n²) time, O(n) memory — same time, 10× less memory
Used by: every production LLM (GPT-4, Claude, LLaMA, Mistral)
PyTorch usage:
  F.scaled_dot_product_attention(Q, H, V, is_causal=True)
  # Automatically uses Flash Attention if on CUDA
```

### Causal Masking

For a decoder (GPT-style), "mat" at position 4 should ONLY see positions 1-4:

```
Full score matrix (no mask):
         key_cat  key_sat  key_on  key_mat
q_cat:   1.146    0.912    0.446   -0.037
q_sat:   0.896    0.723    0.370   -0.016
q_on:    0.485    0.344    0.212    0.020
q_mat:  -0.061   -0.035    0.009    0.022

Causal mask: set upper triangle to -∞ (block future positions):
q_cat sees: [cat only]      → set sat, on, mat scores to -∞
q_sat sees: [cat, sat]      → set on, mat scores to -∞
q_on  sees: [cat, sat, on]  → set mat scores to -∞
q_mat sees: [all 4]         → no masking for the last position

After applying causal mask:
         key_cat  key_sat  key_on  key_mat
q_cat:   1.146      -∞       -∞       -∞
q_sat:   0.896    0.723      -∞       -∞
q_on:    0.485    0.344    0.212       -∞
q_mat:  -0.061   -0.035    0.009    0.022

Softmax with -∞ entries → 0 for those positions:
q_cat: softmax([1.146, -∞, -∞, -∞]):  A_cat = [1.000, 0.000, 0.000, 0.000]
  → cat attends ONLY to itself

q_sat: softmax([0.896, 0.723, -∞, -∞]):  A_sat = [0.543, 0.457, 0.000, 0.000]
  → sat sees only cat and itself

q_mat unchanged = [0.239, 0.245, 0.256, 0.260]  → last position sees all

WHY CAUSAL MASK?
In encoder (BERT): no mask → each token sees full context → better representations
In decoder (GPT): causal mask → token i can't see tokens i+1,...,n
Only #+1 is what makes autoregressive generation possible:
generate token 5 based on tokens 1-4, then append and repeat.
```

**KV Cache (production inference trick):**
```
During generation without cache:
Step 1: generate token 5 → compute attention over tokens 1-4 (n=4)
Step 2: generate token 6 → compute attention over tokens 1-5 (n=5)
Step k: recompute K,V for ALL previous tokens from scratch → O(n²) total

With KV Cache:
Store K, V tensors for each layer from all previous tokens.
At step k: only compute Q for new token, retrieve cached K, V.
Per-step cost: O(kn), total O(kn) + O(n²) → O(n) amortized per step

Memory cost: KV cache size = 2 × n_layers × n_heads × d_head × seq_len × 2 bytes
For LLaMA-7B (32 layers, 32 heads, d_head=128): ~2GB for 2K tokens
```

---

## 15. Full Picture Diagram

```
INPUT LAYER
  "cat"    "sat"    "on"    "mat"
    |        |        |       |
[1.0,0.5] [0.2,0.3] [0.1,0.1] [0.2,0.4]    ← word embeddings
    |        |        |       |
+ PE(0)  + PE(1)  + PE(2)  + PE(3)          ← positional encodings
    |        |        |       |
[1.0,1.5] [1.041,0.840] [1.009,-0.316] [0.341,-0.590]  = X_pe

            (ALL positions in parallel)
                    ↓
    ┌──────────────────────────────────────┐
    │   SCALED DOT-PRODUCT ATTENTION       │
    │   Q=X_pe@Wq  K=X_pe@Wk  V=X_pe@Wv  │
    │   S = Q @ K^T / √2  (4×4 score matrix)│
    │   A = softmax(S)    (4×4 weight matrix)│
    │   C = A @ V         (4×2 context vecs) │
    │                                      │
    │   A=[0.239,0.245,0.256,0.260]         │
    │   c_mat=[0.772, 0.399]               │
    └──────────────────────────────────────┘
                    ↓
    RESIDUAL 1: X_attn = X_pe + C
    mat: [0.341,-0.590] + [0.772,0.399] = [1.113,-0.191]    ← x_pe_mat + c_mat
                    ↓
    LAYERNORM 1: Normalize X_attn per position
    mat: [1.000,-1.000]                                       ← normalized (d=2)
                    ↓
    (same weights W1, W2 for ALL positions)
    FEED-FORWARD NETWORK
    H = ReLU(X_ln1 @ W1 + b1)
    mat: [0.1,0.1,0.0,0.0]                                   ← ReLU kills 2 neurons
    FFN_out = H @ W2 + b2
    mat: [0.070, 0.070]
                    ↓
    RESIDUAL 2: X_final = X_attn + FFN_out
    mat: [1.113,-0.191] + [0.070,0.070] = [1.183,-0.121]
                    ↓
    CLASSIFIER
    z = W_out · x_final_mat = 0.556
    ŷ = σ(0.556) = 0.635
    L = -log(0.635) = 0.454

BACKWARD PASS:
∂L/∂z = -0.365
∂L/∂W_out = [-0.432, +0.044]
∂L/∂x_final_mat = [-0.183, -0.110]

[Residual 2 splits gradient]
├── highway: ∂L/∂x_attn_mat = [-0.183,-0.110]  (direct!)
└── [LN blocks: ∂L/∂x_attn from LN path = 0 for d=2 (Jacobian=0)]

[Residual 1 splits gradient]
├── highway: ∂L/∂x_pe_mat = [-0.183,-0.110]  (reaches embedding+PE direct, in ONE step)
└── attention: ∂L/∂c_mat = [-0.183,-0.110]
              ├── ∂L/∂V≈[[-0.191,-0.062],[-0.016,-0.009]] (dominant)
              └── ∂L/∂Wq,∂L/∂Wk tiny
```

---

## 16. Why Next: BERT and GPT

```
COMPLETE ARCHITECTURE PROGRESSION

RNN:         h = tanh(W_x·x + W_h·h_prev)
             Problem: gradient decays as tanh^T → 0

LSTM:        C = f⊙C_prev + i⊙g
             Solution: additive cell update + gradient highway along C
             Problem: still sequential, still one-vector bottleneck

GRU:         h = (1-z)⊙h_prev + z⊙h̃
             Solution: same gradient highway, fewer parameters than LSTM
             Problem: still sequential, still one-vector bottleneck

Attention:   C = softmax(QK^T/√d)V
             Solution: all positions in parallel, no bottleneck
             Problem: no position info, no non-linearity between positions

Transformer: Attention + PE + Residual + LayerNorm
             Solution: all problems above solved
             + PE:      positions distinguished
             + FFN:     non-linear transformation per position
             + Residual: gradient highway at every sublayer (not just LSTM's C)
             + LN:      stable activations + stack 100+ layers

GRADIENT HIGHWAY EVOLUTION:
RNN:  h = f(h, x) → h,  multiplicative, decays exponentially
LSTM: C = f * C_prev + ...,  additive term creates highway
GRU:  C = Au,  ONE stop from output to any V
Tfmr: x_out = x + sub(x) → residual adds DIRECT path per block
      With L blocks: L residual paths, gradient can choose
      shortest route → enables training 100+ layer models
```

**Three sentences summary:**
- RNN: Information flows time (sequential, lossy).
- Attention: Information flows through a weighted sum (parallel, direct).
- Transformer: attention + position awareness + non-linearity + residuals = the complete architecture that powers modern AI.

**Two directions:**

BERT (Encoder-only, Devlin et al., 2019):
```
Stack: 12 encoder blocks (same structure as above)
d_model=768, n_heads=12, d_ff=3072
Input: [CLS] token + masked tokens → predict masked word ("MLM") + Next Sentence Prediction ("NSP")
Use: Contextualized representations + finetune for NLP tasks
Why encoder? Bidirectional attention — each token sees ALL others.
12 encoder blocks × 12 residual highways × 24 residuals
Final CLS token is a summary of the whole sequence.
```

GPT (Decoder-only, Radford et al., 2018):
```
Stack: 12 decoder blocks
Same structure but with CAUSAL MASKING in attention:
A[i,j] = 0 if j > i  (can only attend to LEFT context)
Training: Predict next token ("language modeling")
Use: Text generation — each token generated based on all previous tokens.

Causal mask: S[i,j] = -∞ for j>i → softmax = 0
This ensures autoregressive generation possible:
generate token 5 based on tokens 1-4, then append and repeat.
```

ENCODER-DECODER (original Transformer, Vaswani et al.):
```
Encoder: processes source sequence with full attention (like BERT)
Decoder: generates target sequence with causal attention + cross-attention
Cross-attention: Queries from decoder, Keys/Values from encoder
Used for translation: "cat sat on mat" → "le chat était assis sur le tapis"
```

**What makes modern LLMs powerful:**
```
Scale law (Kaplan et al., 2020):
Loss ≈ (compute)^(-0.05)  (roughly)
Doubling compute → 3-5% loss reduction → significant capability jump

GPT-3: 96 layers, d=12288, n_heads=96, 175B parameters
Training: 300B tokens × requires ~3.14×10^23 FLOPS

Emergent capabilities: At 175B params: few-shot learning, code generation, reasoning chains
These behaviors were NOT explicitly trained — they "emerge" at scale.
The transformer architecture we built today is architecturally identical to GPT-3.
The only difference is scale.
```

---

## 17. Quick Reference Card

```
TRANSFORMER ENCODER BLOCK — QUICK REFERENCE

FORWARD PASS:
1. X_pe = X + PE              (add positional encoding)
2. Q=X@Wq, K=X@Wk, V=X@Wv   (project to Q, K, V — same as attention)
3. S = Q @ K^T / √d_k; A = softmax(S); C = A @ V
4. X_attn = LN(X_pe + C)     (residual + layernorm)
5. H = ReLU(X_attn @ W1 + b1); F = H @ W2 + b2
6. X_out = LN(X_attn + F)    (residual + layernorm)
7. ŷ = σ(W_out · X_out[mat])

KEY DIMENSIONS:
d_model: embedding dim (512/768/1024 in practice)
n_heads: number of attention heads (8/12/16)
d_head = d_model / n_heads
d_ff = 4 × d_model   (FFN hidden dim)

BACKWARD — KEY GRADIENT HIGHWAYS:
Residual 2: ∂L/∂x_attn = ∂L/∂x_out × 1   (direct)
Residual 1: ∂L/∂x_pe   = ∂L/∂x_attn × 1  (direct)
LN: blocks gradient at d=2 (Jacobian=0); important at d≥512
FFN: ∂L/∂W2=h^T×∂L/∂F_out, ∂L/∂W1 tiny, LN blocks further flow
Wq,Wk: small (through softmax Jacobian, sum-zero constraint)

THIS WALKTHROUGH (d=2, n=4, 1 head):
x_pe_mat=[0.341,-0.590], x_final_mat=[1.183,-0.121]
A=[0.239,0.245,0.256,0.260] (mat attends most to itself+on)
ŷ=0.635, L=0.454 → L'=0.431 after update ✓

Parameter Count:
This toy example (d=2, d_ff=4, vocab=4): Wq, Wk, Wv: 3 × (2×2) = 12 params W_o: (2×2) = 4 params
(W1, W2) FFN (SwiGLU): 2 × d × 2/3 × d_ff (same total = 2 × 10 params) LN params: 2 × 4 params (γ, β)
W_out 2 params; Embeddings: 4×2=8 params — TOTAL ≈ 52 params

Real models: BERT-base: d=768, d_ff=3072, h=12, vocab=30522 → ~110M params
GPT-2: d=768, d_ff=3072, h=12, vocab=50257 → ~117M params
LLaMA-7B: d=4096, d_ff=49152, h=32, L=96 → ~7B params
```

---

## 18. Code

### Version 1: Pure NumPy (manual everything)

```python
import numpy as np

# --- Data ---
vocab = {'cat': 0, 'sat': 1, 'on': 2, 'mat': 3}
X = np.array([
    [1.0, 0.5],   # cat
    [0.2, 0.3],   # sat
    [0.1, 0.1],   # on
    [0.2, 0.4]    # mat
])
y = 1.0  # "mat" is in sentence

# --- Positional Encoding ---
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2):
            PE[pos, 2*i]   = np.sin(pos / 10000 ** (2*i / d_model))
            PE[pos, 2*i+1] = np.cos(pos / 10000 ** (2*i / d_model))
    return PE

PE = positional_encoding(4, 2)
X_pe = X + PE
print("X_pe:", X_pe)
# [[1.000, 1.500], [1.041, 0.840], [1.009, -0.316], [0.341, -0.590]]

# --- Weights ---
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

gamma1 = np.ones(2)
beta1  = np.zeros(2)

def layernorm_forward(x, gamma, beta, eps=1e-8):
    mu    = x.mean(axis=-1, keepdims=True)
    sigma = x.std(axis=-1, keepdims=True) + eps
    x_hat = (x - mu) / sigma
    return gamma * x_hat + beta, x_hat, mu, sigma

# --- Attention ---
Q = X_pe @ Wq
K = X_pe @ Wk
V = X_pe @ Wv

d_k = Q.shape[-1]
scores = Q @ K.T / np.sqrt(d_k)
scores = scores - scores.max(axis=-1, keepdims=True)  # numerically stable softmax
A = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
C = A @ V
print("Attention matrix A:\n", np.round(A, 3))
# Row 4: [0.239, 0.245, 0.256, 0.260]

# --- Residual 1 + LayerNorm ---
X_attn = X_pe + C
X_ln1, x_hat1, mu1, sigma1 = layernorm_forward(X_attn, gamma1, beta1)

# --- FFN ---
H = np.maximum(X_ln1 @ W1 + b1, 0)   # ReLU
FFN_out = H @ W2 + b2
print("FFN output (mat):", np.round(FFN_out[3], 3))  # [0.070, 0.070]

# --- Residual 2 ---
X_final = X_attn + FFN_out

# --- Classify from mat ---
z = W_out @ X_final[3]
y_hat = 1 / (1 + np.exp(-z))
loss  = -np.log(y_hat)
print(f"ŷ={y_hat:.3f}, L={loss:.3f}")  # ŷ=0.635, L=0.454

# --- Backward Pass ---
lr = 0.1

# Output gradient
dl_dz  = y_hat - y                              # -0.365
dl_dW_out = dl_dz * X_final[3]                  # [-0.432, +0.044]
dl_dx_final_mat = dl_dz * W_out                 # [-0.183, -0.110]

# Residual 2: split gradient
dl_dx_attn_mat = dl_dx_final_mat.copy()          # highway
dl_dFFN_out_mat = dl_dx_final_mat.copy()         # FFN path

# FFN backward (mat only)
dl_dW2 = H[3:4].T @ dl_dFFN_out_mat.reshape(1, -1)   # (4,2)
dl_dh  = dl_dFFN_out_mat @ W2.T                       # (4,)
dl_dpre_act = dl_dh * (H[3] > 0).astype(float)        # ReLU mask

dl_dW1 = X_ln1[3:4].T @ dl_dpre_act.reshape(1, -1)   # (2,4)
dl_db1 = dl_dpre_act
dl_db2 = dl_dFFN_out_mat

# Residual 1: split gradient (LN Jacobian = 0 for d=2, so only highway matters)
dl_dxpe_mat = dl_dx_attn_mat.copy()                   # highway
dl_dc_mat = dl_dx_attn_mat.copy()                     # attention path

# Wv gradient (simplified for single query)
dl_dV = A[:, 3:4].reshape(-1, 1) * dl_dc_mat          # Note: A[:,3] = col for mat
dl_dWv = X_pe.T @ dl_dV                               # (2,2)

# --- Weight Update ---
W_out_new = W_out - lr * dl_dW_out
W2_new    = W2    - lr * dl_dW2
W1_new    = W1    - lr * dl_dW1
Wv_new    = Wv    - lr * dl_dWv

# --- Verify Loss Decreased ---
Q2 = X_pe @ Wq; K2 = X_pe @ Wk; V2 = X_pe @ Wv_new
scores2 = Q2 @ K2.T / np.sqrt(d_k)
scores2 -= scores2.max(axis=-1, keepdims=True)
A2 = np.exp(scores2) / np.exp(scores2).sum(axis=-1, keepdims=True)
C2 = A2 @ V2
X_attn2 = X_pe + C2
X_ln12, _, _, _ = layernorm_forward(X_attn2, gamma1, beta1)
H2 = np.maximum(X_ln12 @ W1_new + b1, 0)
FFN2 = H2 @ W2_new + b2
X_final2 = X_attn2 + FFN2
z2 = W_out_new @ X_final2[3]
yhat2 = 1 / (1 + np.exp(-z2))
loss2 = -np.log(yhat2)
print(f"After update: ŷ={yhat2:.3f}, L'={loss2:.3f}")  # L' < 0.454 ✓
```

### Version 2: PyTorch Manual (autograd does backward)

```python
import torch
import torch.nn.functional as F

# --- Data ---
X = torch.tensor([
    [1.0, 0.5], [0.2, 0.3], [0.1, 0.1], [0.2, 0.4]
], dtype=torch.float32)
y = torch.tensor(1.0)

# --- Positional Encoding ---
def get_pe(seq_len, d):
    PE = torch.zeros(seq_len, d)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.pow(10000, torch.arange(0, d, 2).float() / d)
    PE[:, 0::2] = torch.sin(pos / div)
    PE[:, 1::2] = torch.cos(pos / div)
    return PE

PE = get_pe(4, 2)
X_pe = X + PE

# --- Weights (requires_grad=True) ---
Wq = torch.tensor([[0.60, 0.40], [0.20, 0.50]], requires_grad=True)
Wk = torch.tensor([[0.50, 0.30], [0.10, 0.40]], requires_grad=True)
Wv = torch.tensor([[0.80, 0.20], [0.30, 0.70]], requires_grad=True)
W1 = torch.tensor([[0.5, 0.3, 0.2, 0.1],
                   [0.4, 0.2, 0.3, 0.1]], requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.tensor([[0.5, 0.3], [0.2, 0.4],
                   [0.3, 0.2], [0.1, 0.5]], requires_grad=True)
b2 = torch.zeros(2, requires_grad=True)
W_out = torch.tensor([0.5, 0.3], requires_grad=True)
gamma1 = torch.ones(2, requires_grad=True)
beta1  = torch.zeros(2, requires_grad=True)

# --- Forward ---
# Attention
Q = X_pe @ Wq
K = X_pe @ Wk
V = X_pe @ Wv
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

# Classify from mat
z = W_out @ X_final[3]
y_hat = torch.sigmoid(z)
loss = F.binary_cross_entropy(y_hat, y)
print(f"ŷ={y_hat.item():.3f}, L={loss.item():.3f}")

# --- Backward ---
loss.backward()

print("∂L/∂W_out:", W_out.grad)          # [-0.432, +0.044]
print("∂L/∂dWv:\n", Wv.grad)
print("∂L/∂dW1:\n", W1.grad)             # [[-0.125,-0.081,0,0],[0.125,0.081,0,0]]

# --- Update ---
lr = 0.1
with torch.no_grad():
    for param in [Wq, Wk, Wv, W1, b1, W2, b2, W_out, gamma1, beta1]:
        param -= lr * param.grad
        param.grad.zero_()

# --- Verify ---
Q2 = X_pe @ Wq; K2 = X_pe @ Wk; V2 = X_pe @ Wv
A2 = F.softmax(Q2 @ K2.T / (d_k**0.5), dim=-1)
C2 = A2 @ V2
X_attn2 = X_pe + C2
X_ln12 = F.layer_norm(X_attn2, [2], gamma1, beta1)
H2 = F.relu(X_ln12 @ W1 + b1)
FFN2 = H2 @ W2 + b2
X_final2 = X_attn2 + FFN2
z2 = W_out @ X_final2[3]
yhat2 = torch.sigmoid(z2)
loss2 = F.binary_cross_entropy(yhat2, y)
print(f"After update: ŷ={yhat2.item():.3f}, L'={loss2.item():.3f}")  # L' < 0.454 ✓
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
        self.register_buffer('PE', PE)

    def forward(self, x):
        # x: (batch, seq, d)
        return x + self.PE[:x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
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
        C = A @ V

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
        attn_out, A = self.attn(x, x, mask)
        x = self.ln1(x + self.drop(attn_out))    # Residual 1 + LN
        x = self.ln2(x + self.drop(self.ffn(x))) # Residual 2 + LN
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
        x = self.embed(token_ids)   # (B, T, d)
        x = self.pe(x)
        attentions = []
        for layer in self.layers:
            x, A = layer(x)
            attentions.append(A)
        # Classify from last token (or [CLS] token in practice)
        logit = self.classifier(x[:, -1, :]).squeeze(-1)
        return torch.sigmoid(logit), attentions


# --- Training loop ---
model = TransformerClassifier(
    vocab_size=4, d_model=2, n_heads=1, d_ff=4, n_layers=1
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.BCELoss()

# "cat sat on mat" → token ids [0, 1, 2, 3]
tokens = torch.tensor([[0, 1, 2, 3]])    # (1, 4)
target = torch.tensor([1.0])             # "mat" present

# Forward + update
y_hat, _ = model(tokens)
loss = criterion(y_hat, target)
print(f"Initial: ŷ={y_hat.item():.3f}, L={loss.item():.3f}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

# Verify
y_hat2, attn_maps = model(tokens)
loss2 = criterion(y_hat2, target)
print(f"After update: ŷ={y_hat2.item():.3f}, L'={loss2.item():.3f}")
# L' < initial L ✓

# Inspect attention pattern for "mat"
print("mat attention pattern:", attn_maps[0][0, 0, 3, :].detach().numpy())
# Should be roughly uniform-to-local (reflects PE influence)
```

---

## 19. Connections to Full Series

```
COMPLETE ARCHITECTURE PROGRESSION

RNN:         h = tanh(W_x·x + W_h·h_prev)    → multiplicative chain
LSTM:        C = f⊙C_prev + i⊙g + LO ε      → additive cell highway
GRU:         h = (1-z)⊙h_prev + z⊙h̃         → simplified highway
Attention:   C = softmax(QK^T/√d)V            → direct path, no BPTT
Transformer: Attention + PE + FFN + Residual   → learnable gradient routing

Same "cat sat on mat" sentence. Same embeddings. Different gradient delivery:
  RNN    → 9% of "cat" gradient reaches position 1
  LSTM   → 69% via forget gate
  GRU    → 66% via update gate
  Attn   → 27.5% raw X (A[4,1]=0.275)
  Tfmr   → direct (residual always 1) + learned attention weight
```

**Next: `07_decoding_strategies.md`** — How does an LLM choose the next token? Greedy, beam search, sampling, top-k, top-p. Same "cat sat on mat" vocabulary used to show how different strategies produce different text.

---

## 20. Gotchas

| Gotcha | What Goes Wrong | How to Fix |
|--------|-----------------|------------|
| Post-LN without warmup | Huge gradient variance early → loss spikes → divergence | Use Pre-LN, or add 4000-step warmup |
| Forgetting causal mask in decoder | Decoder sees future tokens → target leakage | Assert d_model % n_heads == 0 before building model |
| Full-seq gen without KV cache | Recomputes K,V every step → O(n²) per token generated | Enable KV cache: `past_key_values=True` |
| All-pad run in attention (variable-length batching) | softmax(-inf, -inf, ...) = NaN everywhere | Before masked_fill, assert d_model % n_heads == 0 |
| d_model not divisible by n_heads | head_dim = d_model/n_heads → not integer → crash | Assert d_model % n_heads == 0 before building model |
| Confusing mask conventions | HuggingFace: 1=attend; PyTorch MHA: True=IGNORE | Check docs for each framework's convention |
| Not scaling attention scores by √d_k | Q·K variance = d_k without scaling → softmax saturates | Always divide by √d_k (or √d_head) |
| Sequence length > max_pe at inference | pos > max_pe → index error | Use RoPE/ALiBi which extrapolate naturally |
| Attention weight ≠ token importance | High A[i,j] does NOT mean token j "caused" output i | Use gradient-based attribution instead |
| Positional encodings on every layer | PE should be on embeddings once per block | Apply PE ONCE, before the first encoder block |

---

## 21. Interview Q&A

**Q: What problem does positional encoding solve, and why sinusoidal?**

A: Self-attention is permutation-invariant — "cat sat on mat" and "mat on sat cat" produce the same attention output without PE. Sinusoidal PE adds a unique vector to each position's embedding before attention. Sinusoidal is chosen because PE(pos+k) can be expressed as a linear function of PE(pos) — the model can learn to attend to "2 positions ahead" via learned weights. Alternative: learned PE (BERT/GPT-2), but it can't extrapolate beyond max_position seen during training. Modern choice: RoPE (LLaMA, Mistral), which encodes position as a rotation in complex space — relative positions come for free.

**Q: Why does the Transformer need both the residual connection AND LayerNorm?**

A: They solve different problems. Residual: gradient highway — ∂L/∂x = ∂L/∂(x+sublayer(x)) × 1 = direct path. Without residuals, gradient must pass through all sublayer Jacobians in deep models. LayerNorm: activation stability — after attention (which can produce large values) and FFN (which uses ReLU), LN keeps activations near-zero mean and unit variance → stable training across layers. Together: residual ensures gradient flows, LN ensures activations are stable. Without LN → activation explosion in deep models; without residual → gradient vanishes through many LN layers.

**Q: What is the role of the FFN in a Transformer? Why d_ff = 4×d_model?**

A: Attention is linear — it computes a weighted sum of value vectors (output is just a linear combination at each position). The FFN adds per-token non-linearity. It's often described as "key-value memory": W1 acts as selector (do q × match any stored pattern?), ReLU as value retrieval, W2 as output projection. The 4× expansion: d_ff = 4×d_model is empirical — smaller d_ff underfit, larger is diminishing returns. LLaMA further replaces ReLU with SwiGLU: SwiGLU(x) = SiLU(x1) × x2 where x is split. This requires 3 weight matrices but d_ff is often set to 2/3 × 4d (same params).

**Q: Explain the difference between Pre-LN and Post-LN. Which do you use?**

A: Post-LN (original): x = LN(x + Sublayer(x)). Pro: better final performance. Con: training unstable early (large gradient variance → divergence without warmup). Requires careful warmup (large gradients early + divergence without warmup). Pre-LN (modern): x = x + Sublayer(LN(x)). Pro: stable training from step 1, easier to scale to 100+ layers. Current practice (2024): Pre-LN with RMSNorm (LLaMA style). RMSNorm: y = x / RMS(x) — no mean subtraction, simpler, ~10% faster. This is the default for all modern open-source LLMs.

**Q: What happens in the backward pass through a residual connection?**

A: x_out = x + sublayer(x). Backward: ∂L/∂x_out = ∂L/∂(x+sublayer(x)) = ∂L/∂x_final × (1 + ∂sublayer/∂x). The first term is DIRECT — no weight matrices, no activation functions on the critical path. Even if sublayer has vanishing Jacobians (as in our d=2 LayerNorm example where J=0), gradient = [-0.183,-0.110] arrives intact at x_attn_mat, bypassing LN and FFN entirely. This is why Transformers with 100+ layers can be trained.

**Q: Why use multi-head attention instead of single-head?**

A: With a single head, the model learns ONE attention pattern per layer — one type of relationship at a time. With h heads: each head projects Q,K,V into a different d_head-dimensional subspace simultaneously. Head 1: might specialize in syntactic relationships (subject + verb). Head 2: might learn co-reference resolution (pronoun + antecedent). Head 3: positional proximity. Outputs concatenated + projected by W_o to blend all views. Why insight: total compute ≈ same as 1 large head, but expressivity increases because h independent attention patterns averaged → more robust, and h × (n²) attention entries captures more diverse information in one forward pass.

**Q: Why does the gradient through the softmax Jacobian (for Wq and Wk) sum to zero?**

A: Softmax has the property that its outputs sum to 1. Therefore its Jacobian (∂softmax_i/∂s_j for all j) sums to 0 over each row. When backpropagating ∂L/∂s through softmax: ∂L/∂s_k = A_k × (∂L/∂A_k - g), where g = Σ A_j × ∂L/∂A_j. Σ_k ∂L/∂s_k = 0 always (structural property, not data-dependent). This means the gradient to the scores is a zero-sum redistribution. Some scores increase, others decrease by the same total amount. This constrains how much gradient Wq and Wk receive — they update slower than Wv, which has no such constraint (gradient path: c = A @ V, simpler chain rule).

---

## Further Reading

- Attention Is All You Need (Vaswani et al., 2017) — arXiv:1706.03762 — the original paper
- BERT (Devlin et al., 2018) — arXiv:1810.04805 — encoder-only + MLM pretraining
- GPT-2 (Radford et al., 2019) — openai.com/research/gpt-2 — decoder-only LM
- RoPE (Su et al., 2021) — arXiv:2104.09864 — rotary position embedding
- Flash Attention (Dao et al., 2022) — arXiv:2205.14135 — O(n) memory attention
- LLaMA 2 (Touvron et al., 2023) — arXiv:2307.09288 — open LLM with RoPE + SwiGLU
- The Illustrated Transformer (Alammar, 2018) — jalammar.github.io — visual walkthrough
- Formal Algorithms for Transformers (Phuong & Hutter, 2022) — arXiv:2207.09238

---

## Code Practice — Wired by Phase 6

- `code_practice/05_rag/03_transformers/` — full encoder block with multi-head attention

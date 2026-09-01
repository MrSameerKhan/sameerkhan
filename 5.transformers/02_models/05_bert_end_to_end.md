# 05 — BERT: End-to-End with Multi-Head Attention

> Companion to [../../4.nlp/03_sequence_models/06b_transformer_encoder_multihead.md](../../4.nlp/03_sequence_models/06b_transformer_encoder_multihead.md)
> (the encoder block) and [06c_transformer_decoder_end_to_end.md](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md)
> (the decoder). **Same dimensions throughout the arc** — `d_model=4`, `n_heads=2`, `d_head=2`,
> `√d_k=1.414`, `d_ff=8` — so every number here is comparable to those files.
>
> BERT = the 06b encoder + three input embeddings instead of one + two pretraining heads.
> Nothing in the block itself changes. That is the point of this file.
>
> Every number below was computed in numpy and re-checked against `torch.autograd`
> (forward `allclose` to 1.3e-15; total loss `3.638889 → 2.185477` after one SGD step).

---

## Table of Contents

1. What BERT adds over the plain encoder
2. Setup — a sentence pair, dimensions
3. Input = token + segment + **learned** position
4. Weight setup
5. Bidirectional multi-head attention
6. **Why bidirectional** — what the causal mask would have cost
7. Add & Norm 1
8. FFN — **GELU**, not ReLU
9. Add & Norm 2 → encoder output
10. Head 1 — MLM
11. Head 2 — NSP
12. Backward — where two objectives send gradient
13. Weight update — and multi-task interference
14. The 80/10/10 masking rule
15. Fine-tuning from `[CLS]`
16. Where BERT-base's 110M parameters live
17. BERT vs GPT vs the plain encoder
18. Quick reference

---

## 1. What BERT adds over the plain encoder

| | 06b (plain encoder) | **BERT (this file)** |
|---|---|---|
| Input embedding | token + sinusoidal PE | **token + segment + learned position** |
| Position encoding | sinusoidal, fixed | **learned, a trainable table** |
| Attention | bidirectional | bidirectional — *unchanged* |
| FFN activation | ReLU | **GELU** |
| Output heads | none | **MLM head + NSP head** |
| Loss | — | **two cross-entropies, summed** |
| Special tokens | none | `[CLS]`, `[SEP]`, `[MASK]` |

**The transformer block is byte-for-byte the same block as 06b.** BERT is not a new architecture —
it is an encoder stack plus a *pretraining recipe*. Everything novel is at the input boundary and
the output boundary.

> **Correction to be aware of.** BERT does **not** use sinusoidal position encoding. The 2017 paper
> did; BERT replaced it with a *learned* embedding table of shape `(max_position, d_model)` =
> `(512, 768)`. This is why BERT cannot accept a sequence longer than 512 tokens — there is no row
> 513 in the table. Sinusoidal encoding has no such limit. This is a standard interview question and
> getting it backwards is a tell.

---

## 2. Setup

```
Segment A:  bank approved          Segment B:  loan
Full input: [CLS] bank approved [SEP] loan [SEP]
Masked:     [CLS] [MASK] approved [SEP] loan [SEP]      <- 'bank' is masked

L = 6      d_model = 4      n_heads = 2      d_head = 2      d_ff = 8      √d_k = 1.4142

MLM target : position 1 -> 'bank'
NSP target : IsNext (class 1)
```

**Why this sentence.** It is 06b's sentence, cut into a pair. `bank` is the ambiguous token, and it
is the one masked — so the model must recover it from `approved` and `loan`, which sit **to its
right**. A causal model structurally cannot do this. §6 measures exactly how much it cannot.

### 2.1 Vocabulary

`V = 8`. `bank`, `approved`, `the`, `loan` reuse 06b's vectors unchanged.

| Token | Index | Embedding | Note |
|---|---|---|---|
| `[CLS]` | 0 | `[0.2, 0.2, 0.2, 0.2]` | sentence-level slot |
| `[SEP]` | 1 | `[0.1, 0.1, 0.1, 0.1]` | segment separator |
| `[MASK]` | 2 | `[0.0, 0.0, 0.0, 0.0]` | **zero on purpose** — see §3.4 |
| bank | 3 | `[1.0, 0.8, 0.1, 0.1]` | 06b |
| approved | 4 | `[0.3, 0.2, 0.4, 0.3]` | 06b |
| the | 5 | `[0.1, 0.1, 0.2, 0.2]` | 06b |
| loan | 6 | `[0.1, 0.1, 1.0, 0.9]` | 06b |
| granted | 7 | `[0.2, 0.1, 0.5, 0.4]` | |

---

## 3. Input = token + segment + learned position

BERT sums **three** embeddings. All three are `(·, d_model)` and all three are learned.

### 3.1 Token embeddings

```
pos 0  [CLS]     [0.2000,  0.2000,  0.2000,  0.2000]
pos 1  [MASK]    [0.0000,  0.0000,  0.0000,  0.0000]
pos 2  approved  [0.3000,  0.2000,  0.4000,  0.3000]
pos 3  [SEP]     [0.1000,  0.1000,  0.1000,  0.1000]
pos 4  loan      [0.1000,  0.1000,  1.0000,  0.9000]
pos 5  [SEP]     [0.1000,  0.1000,  0.1000,  0.1000]
```

### 3.2 Segment embeddings

Two rows only — segment A and segment B. Every token in a segment gets the *same* vector.

```
E_A = [0.10, 0.00, 0.10, 0.00]      positions 0-3   ([CLS] [MASK] approved [SEP])
E_B = [0.00, 0.10, 0.00, 0.10]      positions 4-5   (loan [SEP])
```

```
pos 0  [0.1000,  0.0000,  0.1000,  0.0000]
pos 1  [0.1000,  0.0000,  0.1000,  0.0000]
pos 2  [0.1000,  0.0000,  0.1000,  0.0000]
pos 3  [0.1000,  0.0000,  0.1000,  0.0000]
pos 4  [0.0000,  0.1000,  0.0000,  0.1000]
pos 5  [0.0000,  0.1000,  0.0000,  0.1000]
```

The trailing `[SEP]` at position 3 belongs to segment **A**, not B. Off-by-one here is a real bug
in hand-rolled BERT preprocessing.

### 3.3 Learned position embeddings

Not a formula — a lookup table, updated by gradient descent like any other weight:

```
        dim0     dim1     dim2     dim3
pos 0 [ 0.3000,  0.1000, -0.2000,  0.2000]
pos 1 [ 0.0000,  0.0000,  0.5000,  0.3000]
pos 2 [-0.2000,  0.3000,  0.4000,  0.0000]
pos 3 [ 0.1000, -0.3000,  0.2000,  0.4000]
pos 4 [ 0.4000,  0.2000, -0.1000, -0.3000]
pos 5 [-0.3000,  0.1000,  0.2000,  0.2000]
```

Contrast with 06b §3.2, where every row was `sin`/`cos` of the position — computed, not stored, and
defined for any position you ask for. Here row 6 does not exist.

### 3.4 X = token + segment + position

```
               dim0     dim1     dim2     dim3
[CLS]      [ 0.6000,  0.3000,  0.1000,  0.4000]
[MASK]     [ 0.1000,  0.0000,  0.6000,  0.3000]
approved   [ 0.2000,  0.5000,  0.9000,  0.3000]
[SEP]      [ 0.3000, -0.2000,  0.4000,  0.5000]
loan       [ 0.5000,  0.4000,  0.9000,  0.7000]
[SEP]      [-0.2000,  0.3000,  0.3000,  0.4000]
```

Shape `(6, 4)`.

**Look at row 1.** The `[MASK]` token embedding is all zeros, so
`X[1] = 0 + E_A + P_1 = [0.1, 0.0, 0.6, 0.3]` — its representation comes **entirely from segment and
position**. The `[MASK]` query carries no content of its own. Everything it eventually predicts has
to be pulled in from other positions by attention. That is the mechanism of MLM in one line.

(Real `[MASK]` embeddings are randomly initialised and then learned, not zero. Zero here makes the
point legible.)

---

## 4. Weight setup

`Wq`, `Wk`, `Wv`, `W_o` are **identical to 06b** — so any difference from 06b's numbers comes from
the input, not the weights.

```
Wq                              Wk
[[1.2, 0.0, 0.0, 0.0],          [[1.0, 0.2, 0.0, 0.0],
 [0.0, 1.2, 0.0, 0.0],           [0.2, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.2, 0.0],           [0.0, 0.0, 1.0, 0.2],
 [0.0, 0.0, 0.0, 1.2]]           [0.0, 0.0, 0.2, 1.0]]

Wv                              W_o
[[0.9, 0.1, 0.0, 0.0],          [[0.9, 0.1, 0.1, 0.0],
 [0.1, 0.9, 0.0, 0.0],           [0.1, 0.9, 0.0, 0.1],
 [0.0, 0.0, 0.9, 0.1],           [0.1, 0.0, 0.9, 0.1],
 [0.0, 0.0, 0.1, 0.9]]           [0.0, 0.1, 0.1, 0.9]]
```

FFN (same as 06c), MLM head `(4 × 8)`, NSP head `(4 × 2)`:

```
W1 (4 × 8)                                                    W2 (8 × 4)
[[ 0.30,-0.20, 0.10, 0.40,-0.30, 0.20, 0.10,-0.10],           [[ 0.20,-0.10, 0.30, 0.10],
 [ 0.10, 0.30,-0.20, 0.10, 0.20,-0.30, 0.40, 0.20],            [-0.10, 0.30, 0.10, 0.20],
 [-0.20, 0.10, 0.40,-0.30, 0.10, 0.20,-0.10, 0.30],            [ 0.30, 0.20,-0.10, 0.10],
 [ 0.20, 0.40,-0.10, 0.20,-0.20, 0.10, 0.30,-0.20]]            [ 0.10, 0.10, 0.20, 0.30],
                                                               [-0.20, 0.30, 0.10,-0.10],
W_mlm (4 × 8)   cols = [CLS] [SEP] [MASK] bank approved the loan granted
[[ 0.3,-0.2, 0.1, 0.5,-0.1, 0.2, 0.4, 0.1],                    [ 0.30,-0.20, 0.20, 0.10],
 [-0.2, 0.4, 0.2,-0.3, 0.5, 0.1,-0.2, 0.3],                    [ 0.10, 0.20, 0.30,-0.20],
 [ 0.1, 0.3,-0.2, 0.2, 0.1,-0.1, 0.6, 0.2],                    [ 0.20, 0.10,-0.20, 0.30]]
 [ 0.4, 0.1, 0.3, 0.1, 0.2, 0.5, 0.3,-0.1]]

W_nsp (4 × 2)   cols = [NotNext, IsNext]
[[ 0.4,-0.3],
 [-0.2, 0.5],
 [ 0.3, 0.1],
 [ 0.1, 0.2]]
```

---

## 5. Bidirectional multi-head attention

### 5.1 Q, K, V

```
Q = X @ Wq                                K = X @ Wk
[[ 0.7200,  0.3600,  0.1200,  0.4800],    [[ 0.6600,  0.4200,  0.1800,  0.4200],
 [ 0.1200,  0.0000,  0.7200,  0.3600],     [ 0.1000,  0.0200,  0.6600,  0.4200],
 [ 0.2400,  0.6000,  1.0800,  0.3600],     [ 0.3000,  0.5400,  0.9600,  0.4800],
 [ 0.3600, -0.2400,  0.4800,  0.6000],     [ 0.2600, -0.1400,  0.5000,  0.5800],
 [ 0.6000,  0.4800,  1.0800,  0.8400],     [ 0.5800,  0.5000,  1.0400,  0.8800],
 [-0.2400,  0.3600,  0.3600,  0.4800]]     [-0.1400,  0.2600,  0.3800,  0.4600]]

V = X @ Wv
[[ 0.5700,  0.3300,  0.1300,  0.3700],
 [ 0.0900,  0.0100,  0.5700,  0.3300],
 [ 0.2300,  0.4700,  0.8400,  0.3600],
 [ 0.2500, -0.1500,  0.4100,  0.4900],
 [ 0.4900,  0.4100,  0.8800,  0.7200],
 [-0.1500,  0.2500,  0.3100,  0.3900]]
```

Then the reshape into heads — columns 0-1 to head 0, columns 2-3 to head 1. A **reshape, not a
projection**, exactly as in 06b §6.

### 5.2 Head 0 — dims 0-1

```
scores = Q₀ K₀ᵀ / √2

            [CLS]  [MASK]  approved  [SEP]    loan   [SEP]
[CLS]    [ 0.4429, 0.0560,  0.2902, 0.0967, 0.4226,-0.0051]
[MASK]   [ 0.0560, 0.0085,  0.0255, 0.0221, 0.0492,-0.0119]
approved [ 0.2902, 0.0255,  0.2800,-0.0153, 0.3106, 0.0865]
[SEP]    [ 0.0967, 0.0221, -0.0153, 0.0899, 0.0628,-0.0798]
loan     [ 0.4226, 0.0492,  0.3106, 0.0628, 0.4158, 0.0288]
[SEP]    [-0.0051,-0.0119,  0.0865,-0.0798, 0.0288, 0.0899]

A₀ = softmax(scores, dim=-1)          NO MASK — the matrix is full

            [CLS]  [MASK]  approved  [SEP]    loan   [SEP]
[CLS]    [ 0.2056, 0.1396, 0.1765, 0.1454, 0.2015, 0.1314]
[MASK]   [ 0.1719, 0.1639, 0.1667, 0.1662, 0.1707, 0.1606]   <- nearly uniform
approved [ 0.1876, 0.1440, 0.1857, 0.1382, 0.1915, 0.1530]
[SEP]    [ 0.1779, 0.1651, 0.1591, 0.1767, 0.1720, 0.1491]
loan     [ 0.2021, 0.1391, 0.1807, 0.1410, 0.2007, 0.1363]
[SEP]    [ 0.1626, 0.1615, 0.1782, 0.1509, 0.1682, 0.1788]
```

### 5.3 Head 1 — dims 2-3

```
scores = Q₁ K₁ᵀ / √2

            [CLS]  [MASK]  approved  [SEP]    loan   [SEP]
[CLS]    [ 0.1578, 0.1986, 0.2444, 0.2393, 0.3869, 0.1884]
[MASK]   [ 0.1986, 0.4429, 0.6109, 0.4022, 0.7535, 0.3106]
approved [ 0.2444, 0.6109, 0.8553, 0.5295, 1.0182, 0.4073]
[SEP]    [ 0.2393, 0.4022, 0.5295, 0.4158, 0.7263, 0.3241]
loan     [ 0.3869, 0.7535, 1.0182, 0.7263, 1.3169, 0.5634]
[SEP]    [ 0.1884, 0.3106, 0.4073, 0.3241, 0.5634, 0.2529]

A₁ = softmax(scores, dim=-1)

            [CLS]  [MASK]  approved  [SEP]    loan   [SEP]
[CLS]    [ 0.1537, 0.1601, 0.1676, 0.1668, 0.1933, 0.1585]
[MASK]   [ 0.1270, 0.1622, 0.1918, 0.1557, 0.2212, 0.1421]   <- the row that matters
approved [ 0.1116, 0.1610, 0.2056, 0.1484, 0.2420, 0.1314]
[SEP]    [ 0.1347, 0.1586, 0.1801, 0.1607, 0.2193, 0.1467]
loan     [ 0.1058, 0.1526, 0.1989, 0.1485, 0.2681, 0.1262]
[SEP]    [ 0.1420, 0.1605, 0.1768, 0.1627, 0.2066, 0.1515]
```

**Row `[MASK]` in head 1: `loan` = 0.2212, `approved` = 0.1918 — the top two.** Those are precisely
the two content words that disambiguate `bank`, and both sit to the *right* of the mask. Head 0's
same row is flat (`0.1606 … 0.1719`, spread 0.011) and has not differentiated at all.

> **Honest note, same as 06b and 06c.** The embeddings and position table were chosen so head 1's
> preference is visible. Head 0 being flat is what untrained attention actually looks like, and it
> is left in rather than tuned away. Head specialisation is a property of **trained** models; the
> architecture only makes it possible.

### 5.4 Concat + W_o

```
O₀                        O₁                        concat = [O₀ ‖ O₁]
[[0.2857, 0.2458],        [[0.5397, 0.4528],        [[0.2857, 0.2458, 0.5397, 0.4528],
 [0.2522, 0.2219],         [0.5726, 0.4605],         [0.2522, 0.2219, 0.5726, 0.4605],
 [0.2680, 0.2467],         [0.5935, 0.4666],         [0.2680, 0.2467, 0.5935, 0.4666],
 [0.2590, 0.2164],         [0.5635, 0.4608],         [0.2590, 0.2164, 0.5635, 0.4608],
 [0.2824, 0.2482],         [0.6037, 0.4761],         [0.2824, 0.2482, 0.6037, 0.4761],
 [0.2415, 0.2300]]         [0.5539, 0.4567]]         [0.2415, 0.2300, 0.5539, 0.4567]]

MHA = concat @ W_o
[[0.3357, 0.2951, 0.5595, 0.4860],
 [0.3064, 0.2710, 0.5867, 0.4940],
 [0.3252, 0.2955, 0.6076, 0.5040],
 [0.3111, 0.2668, 0.5791, 0.4927],
 [0.3394, 0.2993, 0.6192, 0.5137],
 [0.2957, 0.2768, 0.5683, 0.4894]]
```

---

## 6. Why bidirectional — what the causal mask would have cost

Run the **identical block** with 06c's causal mask applied, and compare the `[MASK]` row:

```
head 0  bidirectional : [0.1719, 0.1639, 0.1667, 0.1662, 0.1707, 0.1606]
head 0  causal        : [0.5119, 0.4881, 0.0000, 0.0000, 0.0000, 0.0000]

head 1  bidirectional : [0.1270, 0.1622, 0.1918, 0.1557, 0.2212, 0.1421]
head 1  causal        : [0.4392, 0.5608, 0.0000, 0.0000, 0.0000, 0.0000]

tokens                :  [CLS]   [MASK]  approved  [SEP]   loan    [SEP]
```

Probability mass the `[MASK]` position places on its **right context** (positions 2–5):

| | bidirectional | causal |
|---|---|---|
| head 0 | 0.6642 | **0.0000** |
| head 1 | 0.7108 | **0.0000** |

**Exactly zero, in both heads.** This is the argument for BERT and it does not depend on trained
weights — it is structural. A causal model predicting position 1 has `approved`, `[SEP]`, `loan`,
`[SEP]` unavailable, and `bank` is exactly the token those words disambiguate. Two thirds of the
available evidence is switched off by the mask.

> **What this toy does *not* show.** The causal run's MLM loss is actually *lower* here
> (`1.727530` vs `1.890774`), because with untrained weights the extra context is noise. The mask's
> cost appears under **training**, not at initialisation — same caveat as 06c §6.4. Quote the
> `0.0000` right-context mass in an interview, not the loss.

**The corresponding cost:** BERT cannot generate. To produce token `t` it needs tokens `t+1…L` in
its input, which do not exist yet at generation time. Bidirectionality and autoregression are
mutually exclusive, and that single trade is the whole BERT-vs-GPT split.

---

## 7. Add & Norm 1

```
R1 = X + MHA
[[0.9357, 0.5951, 0.6595, 0.8860],
 [0.4064, 0.2710, 1.1867, 0.7940],
 [0.5252, 0.7955, 1.5076, 0.8040],
 [0.6111, 0.0668, 0.9791, 0.9927],
 [0.8394, 0.6993, 1.5192, 1.2137],
 [0.0957, 0.5768, 0.8683, 0.8894]]

per-row mean: [0.7691, 0.6645, 0.9081, 0.6624, 1.0679, 0.6076]
per-row var : [0.0209, 0.1277, 0.1324, 0.1417, 0.1033, 0.1026]

h1 = LayerNorm(R1)
[[ 1.1514, -1.2025, -0.7571,  0.8081],
 [-0.7222, -1.1010,  1.4610,  0.3622],
 [-1.0522, -0.3095,  1.6477, -0.2861],
 [-0.1364, -1.5823,  0.8412,  0.8774],
 [-0.7110, -1.1471,  1.4044,  0.4537],
 [-1.5980, -0.0960,  0.8141,  0.8799]]
```

Row `[CLS]` has variance 0.0209 — an order of magnitude below every other row. Its inputs were the
most uniform, and LayerNorm amplifies that spread back up to variance 1 regardless. Normalisation
is per-token and independent, so padding tokens can never leak into real ones.

---

## 8. FFN — GELU, not ReLU

```
Z = h1 @ W1
[[ 0.5382, -0.3435, -0.0280,  0.7291, -0.8233,  0.5204, -0.0477, -0.7444],
 [-0.5465,  0.1051,  0.6962, -0.7649,  0.0701,  0.5143, -0.5501,  0.2179],
 [-0.7334,  0.1679,  0.6444, -1.0033,  0.4757,  0.1834, -0.4796,  0.5948],
 [-0.1919, -0.0123,  0.5516, -0.2897, -0.3669,  0.7034, -0.4674, -0.2259],
 [-0.5182,  0.1200,  0.6747, -0.7297,  0.0336,  0.5282, -0.5343,  0.1723],
 [-0.4758,  0.7242,  0.0970, -0.7170,  0.3656, -0.0400, -0.0156,  0.2088]]

GELU(Z)                     GELU(x) = x · Φ(x) = 0.5x(1 + erf(x/√2))
[[ 0.3793, -0.1256, -0.0137,  0.5592, -0.1689,  0.3636, -0.0229, -0.1700],
 [-0.1598,  0.0570,  0.5269, -0.1699,  0.0370,  0.3582, -0.1601,  0.1277],
 [-0.1699,  0.0952,  0.4771, -0.1584,  0.3249,  0.1050, -0.1514,  0.4307],
 [-0.0814, -0.0061,  0.3913, -0.1118, -0.1309,  0.5339, -0.1496, -0.0928],
 [-0.1566,  0.0657,  0.5061, -0.1699,  0.0172,  0.3704, -0.1585,  0.0979],
 [-0.1509,  0.5544,  0.0523, -0.1697,  0.2350, -0.0194, -0.0077,  0.1217]]

FFN out = GELU(Z) @ W2
[[ 0.2468, -0.1674,  0.2974,  0.1861],
 [ 0.2130,  0.0417, -0.1272,  0.0996],
 [ 0.1213,  0.2143, -0.1989,  0.1397],
 [ 0.2434, -0.1119, -0.0191,  0.0648],
 [ 0.2084,  0.0288, -0.1163,  0.0935],
 [-0.1161,  0.2599, -0.0360,  0.0627]]
```

### The difference from ReLU, measured

```
Z row 1     [-0.4836, -0.5619,  0.8894, -0.6001,  0.0955,  0.5003, -0.7404,  0.4584]
GELU        [-0.1520, -0.1613,  0.7232, -0.1646,  0.0514,  0.3460, -0.1699,  0.3101]
ReLU        [ 0.0000,  0.0000,  0.8894,  0.0000,  0.0955,  0.5003,  0.0000,  0.4584]

exact zeros:   GELU 0 of 48        ReLU 24 of 48 (50%)
```

**GELU never outputs exactly zero for a negative input** — it outputs a small negative number, and
it bottoms out around `-0.17` before returning toward zero. Two consequences:

1. **No dead units.** A ReLU unit stuck negative gets zero gradient forever. GELU always passes some
   gradient, so units recover.
2. **No sparsity.** ReLU's 50% zeros are a real compute saving that GELU gives up. BERT and GPT both
   took that trade; modern models take it further with SwiGLU (board 13).

GELU is smooth everywhere, ReLU has a kink at 0. That smoothness is the usual explanation for why
GELU trains slightly better at scale — the honest version is that it was found empirically and the
theory came after.

---

## 9. Add & Norm 2 → encoder output

```
R2 = h1 + FFN_out
[[ 1.3983, -1.3699, -0.4597,  0.9942],
 [-0.5092, -1.0594,  1.3339,  0.4618],
 [-0.9308, -0.0952,  1.4488, -0.1463],
 [ 0.1070, -1.6941,  0.8221,  0.9422],
 [-0.5027, -1.1183,  1.2881,  0.5472],
 [-1.7141,  0.1639,  0.7780,  0.9426]]

per-row mean: [0.1407, 0.0568, 0.0691, 0.0443, 0.0536, 0.0426]
per-row var : [1.2381, 0.8403, 0.7442, 1.1093, 0.8626, 1.1129]

BERT OUTPUT = LayerNorm(R2)
               dim0     dim1     dim2     dim3
[CLS]      [ 1.1302, -1.3576, -0.5396,  0.7670]   <- goes to the NSP head
[MASK]     [-0.6174, -1.2176,  1.3932,  0.4418]   <- goes to the MLM head
approved   [-1.1591, -0.1904,  1.5993, -0.2497]
[SEP]      [ 0.0595, -1.6505,  0.7385,  0.8525]
loan       [-0.5989, -1.2618,  1.3292,  0.5315]
[SEP]      [-1.6652,  0.1150,  0.6971,  0.8531]
```

Shape `(6, 4)` — identical to the input. Stack 12 of these and you have BERT-base.

**Two rows are special, and only because of what is attached to them.** Nothing in the block treats
position 0 differently; `[CLS]` is "the sentence vector" purely because the NSP head reads that row
and gradient therefore trains it to be one.

---

## 10. Head 1 — MLM

Applied to **the masked position only** (position 1). The other five rows contribute nothing to the
MLM loss.

```
logits = OUTPUT[1] @ W_mlm                      (4,) @ (4,8) -> (8,)

          [CLS]    [SEP]   [MASK]     bank  approved      the     loan  granted
       [ 0.3744,  0.0986, -0.4514,  0.3794, -0.3194, -0.1636,  0.9650, -0.1926]

probs = softmax(logits)
       [ 0.1502,  0.1140,  0.0658,  0.1510,  0.0751,  0.0877,  0.2711,  0.0852]
```

```
greedy = 'loan'   (0.2711)
gold   = 'bank'   p = 0.1510

L_mlm = -log(0.1510) = 1.890774
```

Uniform over 8 tokens would be `log 8 = 2.0794`. At `1.8908` the untrained model is barely better
than chance, and it is predicting `loan` — it has copied the most distinctive nearby token rather
than inferred the missing one. Exactly what you would expect before training.

---

## 11. Head 2 — NSP

Applied to **`[CLS]` only**.

```
logits = OUTPUT[0] @ W_nsp                      (4,) @ (4,2) -> (2,)

          NotNext   IsNext
       [  0.6384, -0.9184]

probs  [  0.8259,  0.1741]

gold = IsNext (class 1)
L_nsp = -log(0.1741) = 1.748114
```

The model says **NotNext at 0.8259** — confidently wrong. Chance is `log 2 = 0.6931`, so at `1.7481`
this is meaningfully worse than guessing.

### Total loss

```
L = L_mlm + L_nsp = 1.890774 + 1.748114 = 3.638889
```

BERT **sums** the two, unweighted. Two heads, two losses, one shared encoder — and §13 shows what
that costs.

> **NSP did not survive.** RoBERTa removed it and scored *better*; ALBERT replaced it with
> sentence-order prediction. The consensus is that NSP is too easy — distinguishing a random
> document's sentence from the true next one is mostly topic detection, solvable without any
> sentence-level reasoning. Know that it exists, that it is computed from `[CLS]`, and that it was
> dropped.

---

## 12. Backward — where two objectives send gradient

All gradients from `torch.autograd`; the torch forward matched the numpy forward to **1.3e-15**
(`allclose`, atol 1e-6).

### 12.1 Magnitudes

```
                    max |grad|        L2
  dL/dEMB            2.248861      4.016738    <- largest
  dL/dPOS            2.248861      3.998272
  dL/dSEG            2.633911      3.682268
  dL/dW_nsp          1.121253      2.335983
  dL/dW_mlm          1.182877      1.849153
  dL/dW_o            0.969513      2.003167
  dL/dW_v            0.802228      1.632737
  dL/dW2             0.194601      0.522533
  dL/dW1             0.256092      0.474408
  dL/dW_q            0.038321      0.060161
  dL/dW_k            0.031643      0.055041    <- smallest, 25x below W_v
```

**06b's finding reproduces exactly.** `W_v` and `W_o` sit on a linear path to the loss; `W_q` and
`W_k` reach it only through the softmax, whose Jacobian collapses as rows sharpen. The
query/key path learns slowest, here by a factor of ~25.

The three embedding tables get the largest gradients of all — they are one lookup away from the loss
and they are shared by every position.

### 12.2 Only tokens that appear get gradient

```
dL/dEMB, per vocabulary row:
   [CLS]      [ 1.5634, -1.2464,  2.0483, -2.2489]   L2 = 3.640120
   [SEP]      [ 0.0182, -0.0668,  0.3736, -0.3987]   L2 = 0.550715
   [MASK]     [-1.2481,  0.8850,  0.1877,  0.1642]   L2 = 1.550188
   bank       [ 0.0000,  0.0000,  0.0000,  0.0000]   L2 = 0.000000   <- the GOLD label
   approved   [ 0.0231, -0.0809,  0.2056, -0.1446]   L2 = 0.265077
   the        [ 0.0000,  0.0000,  0.0000,  0.0000]   L2 = 0.000000
   loan       [ 0.0845, -0.0745,  0.2366, -0.1960]   L2 = 0.327216
   granted    [ 0.0000,  0.0000,  0.0000,  0.0000]   L2 = 0.000000
```

Three rows are **exactly zero**: `the` and `granted` never appear in the input, so no gradient
reaches them. That is why an embedding table of 30,522 rows updates only a few thousand rows per
step, and why embedding gradients are stored sparsely.

**`bank` is zero too — and it is the answer.** Its gradient goes into **column 3 of `W_mlm`**, not
into `EMB[bank]`, because the MLM head is a separate matrix here. **If the MLM head were tied to the
embedding table** — as GPT and most modern LLMs do — that same gradient would land on `EMB[bank]`
and this row would be non-zero. Weight tying is not just a parameter saving; it changes which
tensor learns. (Board 7, board 9.)

`[CLS]` has by far the largest embedding gradient (L2 3.64) because it carries the entire NSP loss
alone.

---

## 13. Weight update — and multi-task interference

```
SGD, lr = 0.05,  W ← W - lr · dL/dW   (all eleven tensors)

L_mlm   1.890774 -> 1.803056
L_nsp   1.748114 -> 0.382421
total   3.638889 -> 2.185477     DECREASED by 1.453412   ✓

p(bank) 0.1510 -> 0.1648
p(IsNext) 0.1741 -> 0.6822        NotNext -> IsNext, now correct
```

Both objectives improved. But sweep the learning rate and the picture changes:

```
    lr      L_mlm      L_nsp      total    p(bank)
     0    1.890774   1.748114   3.638889   0.1510
  0.02    1.829016   0.747000   2.576017   0.1606
  0.05    1.803056   0.382421   2.185477   0.1648   <- both improve
  0.10    1.904092   0.281652   2.185744   0.1490   <- MLM now WORSE than start
  0.15    2.034609   0.255879   2.290488   0.1307
  0.20    2.149560   0.245518   2.395079   0.1165
  0.30    2.352433   0.239618   2.592052   0.0951
  0.50    2.658983   0.273108   2.932092   0.0700
```

**Past `lr ≈ 0.05`, NSP keeps improving while MLM actively degrades — and the *total* still looks
fine.** At `lr = 0.5` the summed loss has dropped from 3.64 to 2.93 and you would call that
progress, but MLM has gone from 1.89 to 2.66 and `p(bank)` has more than halved.

This is **multi-task interference**, and it is visible in a 6-token toy. Two heads share one
encoder; the easier objective (NSP — binary, one position, large gradient) captures the shared
weights and the harder one (MLM — 8-way, one position) pays for it. Watching only the summed loss
hides it entirely.

It is also a concrete reason RoBERTa's removal of NSP *helped*: deleting the easy objective gave the
hard one the whole encoder.

---

## 14. The 80/10/10 masking rule

15% of tokens are selected for prediction. Of those selected:

```
80%  ->  replaced with [MASK]
10%  ->  replaced with a RANDOM token
10%  ->  left UNCHANGED (but still predicted)
```

On a full BERT-base sequence:

```
sequence length                   512
selected for prediction (15%)    76.8 tokens
   -> [MASK]              (80%)  61.44
   -> random token        (10%)   7.68
   -> unchanged           (10%)   7.68
```

**Why not 100% `[MASK]`.** `[MASK]` never appears at fine-tuning or inference time. If it were the
only signal, the encoder would learn "produce a good representation *when you see `[MASK]`*" and
have no reason to build good representations for ordinary tokens — a train/test mismatch on every
downstream task. The 10% random and 10% unchanged force the model to build a real representation for
**every** position, because it cannot tell from the input alone which positions are being graded.

**The cost of 15%:**

```
BERT computes loss on  76.8 of 512 positions  = 15% of the sequence
GPT  computes loss on 512 of 512 positions    = 100%

-> GPT extracts 6.67x more training signal per sequence
```

That ratio is the honest reason MLM pretraining is compute-inefficient relative to causal LM, and
part of why the field went the GPT way once scale became the lever.

---

## 15. Fine-tuning from `[CLS]`

Pretraining is done. Throw away the MLM and NSP heads, keep the encoder, attach a new head to
`[CLS]`.

```
task: does this sentence pair concern a loan?   gold = 1 (yes)

[CLS] output = [ 1.1302, -1.3576, -0.5396,  0.7670]

W_cls (4 × 2)  — NEW, randomly initialised
[[ 0.6, -0.4],
 [-0.3,  0.5],
 [ 0.2,  0.3],
 [ 0.4, -0.2]]

logits = [CLS] @ W_cls = [ 1.2843, -1.4462]
probs                  = [ 0.9388,  0.0612]

L_cls = -log(0.0612) = 2.793614
```

The head is `4 × 2` — **8 parameters** against the encoder's millions. Everything that makes this
work was already learned during pretraining; the head only reads it out.

**What actually happens in fine-tuning:** the new head *and the whole encoder* are updated together,
at a small learning rate (2e-5 to 5e-5, vs 1e-4 for pretraining), for 2–4 epochs. Freezing the
encoder and training only the head — "feature extraction" — is faster but consistently several
points worse. Fine-tuning the encoder is the entire reason BERT displaced static embeddings.

---

## 16. Where BERT-base's 110M parameters live

```
token embeddings      30522 × 768              =  23,440,896
position embeddings     512 × 768              =     393,216
segment embeddings        2 × 768              =       1,536
embedding LayerNorm       2 × 768              =       1,536
                                                  ──────────
                                                  23,837,184   (21.9%)

per layer:
  attention  4 × (768² + 768)                  =   2,362,368
  LayerNorm            2 × 768                 =       1,536
  FFN in     768 × 3072 + 3072                 =   2,362,368
  FFN out    3072 × 768 +  768                 =   2,360,064
  LayerNorm            2 × 768                 =       1,536
                                                  ──────────
                                                   7,087,872
× 12 layers                                    =  85,054,464   (78.1%)
                                                  ──────────
TOTAL (encoder)                                = 108,891,648   ✓ ("110M")
  + pooler  768 × 768 + 768                    =     590,592
TOTAL (HF `bert-base-uncased`)                 = 109,482,240
```

*(Biases and LayerNorm gains/biases included. The commonly quoted simplified formula
`4d² + 2·d·d_ff` per layer gives `7,077,888` and a total of `108,768,768` — fine for
order-of-magnitude work, but do not mix it with a full embedding count.)*

Two things people get wrong:

- **The embedding table is 22% of BERT-base** — 23.4M parameters in a single lookup matrix, more
  than three encoder layers. It is also the first thing quantised or factorised (ALBERT factorises
  it to 128 dimensions and saves ~20M).
- **Within a layer the FFN is two-thirds** (`4d²` vs `8d²`), same 33/67 split as 06b §16. Attention
  is the expensive part at *inference*; the FFN is the heavy part in *parameters*.

The 512-row position table is only 393K parameters — but it is the reason BERT hard-stops at 512
tokens.

---

## 17. BERT vs GPT vs the plain encoder

| | 06b encoder | **BERT** | GPT |
|---|---|---|---|
| Attention | bidirectional | **bidirectional** | causal |
| Position | sinusoidal | **learned table (512 max)** | learned table |
| Segment embedding | ✗ | **✓** | ✗ |
| Activation | ReLU | **GELU** | GELU |
| Objective | — | **MLM + NSP** | next-token |
| Loss positions / seq | — | **15%** | 100% |
| Can generate | ✗ | **✗** | ✓ |
| Best at | — | **understanding: classification, NER, QA-extraction** | generation, few-shot |
| Cross-attention | ✗ | ✗ | ✗ |

**The one-line version:** BERT and GPT are the same block. BERT deletes the causal mask and pays for
it by being unable to generate; GPT keeps the mask and pays for it by only ever seeing the left.

---

## 18. Quick reference

```
BERT ENCODER LAYER  (identical to 06b — only the boundaries differ)

 1. X   = E_token + E_segment + E_position        (L, d)   THREE tables, all learned
 2. Q,K,V = X @ Wq, X @ Wk, X @ Wv
 3. split → (n_heads, L, d_head)                  reshape, not a projection
 4. S   = Q Kᵀ / √d_head                          (n_heads, L, L)   NO MASK
 5. A   = softmax(S);  O = A @ V;  MHA = concat @ W_o
 6. h1  = LayerNorm(X + MHA)
 7. F   = GELU(h1 @ W1) @ W2                      GELU, not ReLU
 8. out = LayerNorm(h1 + F)                       (L, d) — same shape in/out

HEADS
 9. MLM: logits = out[masked_positions] @ W_mlm   (n_masked, V)   15% of positions
10. NSP: logits = out[0] @ W_nsp                  ([CLS] only)    (2,)
11. L = L_mlm + L_nsp                             unweighted sum
```

**The seven things to be able to say cold:**

1. BERT's position encoding is a **learned table**, not sinusoidal — which is why 512 is a hard cap.
2. Input is **three** summed embeddings: token + segment + position.
3. Bidirectionality is measurable: the `[MASK]` row puts **0.0000** mass on its right context under
   a causal mask and **0.66–0.71** without one. The price is that BERT cannot generate.
4. **80/10/10** exists because `[MASK]` never appears at fine-tuning time — without the 10/10 the
   model only learns to represent masked slots.
5. MLM grades **15%** of positions; causal LM grades **100%**. GPT gets 6.67× more signal per
   sequence.
6. **NSP was dropped** by RoBERTa and it scored better. Summing two losses lets the easy objective
   eat the shared encoder — visible here as MLM degrading while total loss falls.
7. Embeddings are **22%** of BERT-base; within a layer the FFN is **67%**. Only tokens present in the
   batch get embedding gradient — and the gold label gets none unless the head is **tied**.

---

## See also

- [../../4.nlp/03_sequence_models/06b_transformer_encoder_multihead.md](../../4.nlp/03_sequence_models/06b_transformer_encoder_multihead.md) — the identical block, without BERT's boundaries
- [../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md) — the causal mask this file removes, and cross-attention
- [01_bert_family.md](01_bert_family.md) — RoBERTa, ALBERT, DistilBERT, ELECTRA and what each changed
- [06_gpt1_end_to_end.md](06_gpt1_end_to_end.md) — the same block with the mask kept
- [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md) — GPT-2: pre-LN, `1/√N` init, byte-level BPE
- [../../4.nlp/01_fundamentals/03_tokenization.md](../../4.nlp/01_fundamentals/03_tokenization.md) — WordPiece, and where `[CLS]`/`[SEP]` come from
- [08_modern_llm_architecture.md](08_modern_llm_architecture.md) — what replaced LayerNorm, GELU and learned positions

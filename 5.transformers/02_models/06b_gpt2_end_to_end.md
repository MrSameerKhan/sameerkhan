# 06b — GPT-2: End-to-End with Pre-LN

> **This file is GPT-2 only** (Radford et al., 2019, *Language Models are Unsupervised Multitask
> Learners*). GPT-1 is a separate file: [06_gpt1_end_to_end.md](06_gpt1_end_to_end.md).
> Nothing here is mixed between the two.
>
> Same toy dimensions as the rest of the arc — `d_model=4`, `n_heads=2`, `d_head=2`, `√d_k=1.414`,
> `d_ff=8`, same `Wq/Wk/Wv/W_o` — so any difference from GPT-1's numbers comes from the
> **architecture**, not the weights.
>
> Every number verified in numpy and re-checked against `torch.autograd`
> (forward `allclose` to 4.4e-16; loss `2.118378 → 1.771219`, perplexity `8.3176 → 5.8780`).

---

## GPT-2 in one box

```
sizes        12/24/36/48 layers · d_model 768/1024/1280/1600
vocabulary   50,257        BYTE-LEVEL BPE  (256 bytes + 50,000 merges + 1 <|endoftext|>)
context        1,024
positions    LEARNED table (1024 × d)
activation   GELU
LayerNorm    PRE-LN  — x + sublayer(LN(x)),  PLUS one final LN after the last block
init         residual output weights scaled by 1/sqrt(N),  N = 2 × n_layer
LM head      TIED to the token embedding
parameters   124,439,808  (small)   ... 1,557,611,200  (XL)
pretraining  WebText, ~40 GB, 8M documents
downstream   ZERO-SHOT prompting — no task-specific fine-tuning, no task heads
```

---

## Table of Contents

1. What changed from GPT-1 — the complete list
2. Setup
3. Input = token + learned position
4. Weight setup
5. **Pre-LN block, step by step**
6. Causal multi-head attention
7. The final LayerNorm — and why pre-LN needs one
8. LM head, loss, perplexity
9. Backward — what pre-LN does to the gradients
10. Weight update
11. **The `1/√N` residual initialisation**
12. Byte-level BPE — why 50,257
13. Zero-shot task transfer
14. Where the parameters live — all four sizes
15. Quick reference

---

## 1. What changed from GPT-1

**The block structure is identical.** Masked self-attention → FFN, four projection matrices, GELU,
learned positions, tied LM head, loss at every position. All of that is GPT-1 and is covered in
[06_gpt1_end_to_end.md](06_gpt1_end_to_end.md).

Five things changed:

| # | | GPT-1 | **GPT-2** |
|---|---|---|---|
| 1 | LayerNorm placement | post-LN: `LN(x + f(x))` | **pre-LN: `x + f(LN(x))`** |
| 2 | Final LayerNorm | none | **one, after the last block** |
| 3 | Residual init | standard | **output weights × `1/√N`** |
| 4 | Vocabulary | 40,478 BPE | **50,257 byte-level BPE** |
| 5 | Context | 512 | **1,024** |

Plus scale (117M → 1.5B) and the use shift: **fine-tuning → zero-shot prompting** (§13).

```
GPT-1  (post-LN)                        GPT-2  (pre-LN)
  h1  = LN(x  + Attn(x))                  g1  = x  + Attn(LN(x))
  out = LN(h1 + FFN(h1))                  g2  = g1 + FFN(LN(g1))
                                          out = LN(g2)        <- ln_f, once per stack
```

The residual stream in pre-LN is **never normalised** — it runs from the embeddings to the final
LayerNorm untouched. Every LayerNorm sits on a *branch*. That single change is why pre-LN trains
without warmup, and it is §5, §7, §9 and §11 of this file.

---

## 2. Setup

```
Sequence:  bank  approved  the  loan          L = 4,  V = 8 (toy)

position:    0       1      2     3
input:     bank  approved  the  loan
target:  approved   the   loan  <eos>         <- input shifted LEFT by one
```

| Token | Index | Embedding |
|---|---|---|
| `<bos>` | 0 | `[0.05, 0.05, 0.05, 0.05]` |
| bank | 1 | `[1.00, 0.80, 0.10, 0.10]` |
| approved | 2 | `[0.30, 0.20, 0.40, 0.30]` |
| the | 3 | `[0.10, 0.10, 0.20, 0.20]` |
| loan | 4 | `[0.10, 0.10, 1.00, 0.90]` |
| granted | 5 | `[0.20, 0.10, 0.50, 0.40]` |
| rejected | 6 | `[0.40, 0.30, 0.20, 0.10]` |
| `<eos>` | 7 | `[0.15, 0.25, 0.10, 0.30]` |

---

## 3. Input = token + learned position

```
E_token                                  E_position (LEARNED)
[[1.0000, 0.8000, 0.1000, 0.1000],       [[ 0.2000,  0.1000, -0.1000,  0.3000],
 [0.3000, 0.2000, 0.4000, 0.3000],        [-0.1000,  0.3000,  0.4000,  0.1000],
 [0.1000, 0.1000, 0.2000, 0.2000],        [ 0.3000, -0.2000,  0.2000,  0.4000],
 [0.1000, 0.1000, 1.0000, 0.9000]]        [ 0.1000,  0.4000,  0.3000, -0.2000]]

X = E_token + E_position          <- this is the residual stream at depth 0
             dim0     dim1     dim2     dim3
bank     [ 1.2000,  0.9000,  0.0000,  0.4000]
approved [ 0.2000,  0.5000,  0.8000,  0.4000]
the      [ 0.4000, -0.1000,  0.4000,  0.6000]
loan     [ 0.2000,  0.5000,  1.3000,  0.7000]
```

Learned table of shape `(1024, d_model)` — double GPT-1's 512, still a hard cap.

---

## 4. Weight setup

Identical to GPT-1's, so the comparison isolates the architecture.

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

W1 (4 × 8)                                                    W2 (8 × 4)
[[ 0.30,-0.20, 0.10, 0.40,-0.30, 0.20, 0.10,-0.10],           [[ 0.20,-0.10, 0.30, 0.10],
 [ 0.10, 0.30,-0.20, 0.10, 0.20,-0.30, 0.40, 0.20],            [-0.10, 0.30, 0.10, 0.20],
 [-0.20, 0.10, 0.40,-0.30, 0.10, 0.20,-0.10, 0.30],            [ 0.30, 0.20,-0.10, 0.10],
 [ 0.20, 0.40,-0.10, 0.20,-0.20, 0.10, 0.30,-0.20]]            [ 0.10, 0.10, 0.20, 0.30],
                                                               [-0.20, 0.30, 0.10,-0.10],
W_lm = Eᵀ  — tied, exactly as GPT-1 §4                         [ 0.30,-0.20, 0.20, 0.10],
                                                               [ 0.10, 0.20, 0.30,-0.20],
                                                               [ 0.20, 0.10,-0.20, 0.30]]
```

---

## 5. Pre-LN block, step by step

### 5.1 `ln_1` — normalise the *branch input*, leave the stream alone

```
per-row mean: [0.6250, 0.4750, 0.3250, 0.6750]
per-row var : [0.2119, 0.0469, 0.0669, 0.1619]

n1 = LayerNorm(X)
[[ 1.2492,  0.5974, -1.3578, -0.4888],
 [-1.2700,  0.1155,  1.5010, -0.3464],
 [ 0.2900, -1.6433,  0.2900,  1.0633],
 [-1.1806, -0.4349,  1.5534,  0.0621]]
```

**`X` itself is untouched** — it will be added back verbatim at the end of §6. In GPT-1 this
LayerNorm did not exist; the raw `X` went straight into `Wq/Wk/Wv`.

---

## 6. Causal multi-head attention

Attention runs on `n1`, **not** on `X`:

```
Q = n1 @ Wq                             K = n1 @ Wk
[[ 1.4990,  0.7169, -1.6293, -0.5866],  [[ 1.3686,  0.8473, -1.4555, -0.7604],
 [-1.5240,  0.1385,  1.8011, -0.4156],   [-1.2469, -0.1385,  1.4317, -0.0462],
 [ 0.3480, -1.9720,  0.3480,  1.2760],   [-0.0387, -1.5853,  0.5027,  1.1213],
 [-1.4167, -0.5219,  1.8641,  0.0746]]   [-1.2676, -0.6711,  1.5658,  0.3728]]

V = n1 @ Wv
[[ 1.1840,  0.6626, -1.2709, -0.5757],
 [-1.1315, -0.0231,  1.3162, -0.1616],
 [ 0.0967, -1.4500,  0.3673,  0.9860],
 [-1.1060, -0.5095,  1.4043,  0.2113]]
```

```
masked scores head 0                    masked scores head 1
[[ 1.8802,    -inf,    -inf,    -inf],  [[ 1.9923,    -inf,    -inf,    -inf],
 [-1.3919,  1.3302,    -inf,    -inf],   [-1.6303,  1.8370,    -inf,    -inf],
 [-0.8446, -0.1136,  2.2011,    -inf],   [-1.0442,  0.3106,  1.1354,    -inf],
 [-1.6837,  1.3003,  0.6238,  1.5174]]   [-1.9586,  1.8846,  0.7217,  2.0835]]

A₀ = softmax                            A₁ = softmax
          bank  apprv    the   loan               bank  apprv    the   loan
bank  [ 1.0000, 0.0000, 0.0000, 0.0000]   [ 1.0000, 0.0000, 0.0000, 0.0000]
apprv [ 0.0617, 0.9383, 0.0000, 0.0000]   [ 0.0303, 0.9697, 0.0000, 0.0000]
the   [ 0.0415, 0.0862, 0.8723, 0.0000]   [ 0.0729, 0.2825, 0.6446, 0.0000]
loan  [ 0.0181, 0.3569, 0.1815, 0.4435]   [ 0.0084, 0.3915, 0.1224, 0.4777]

O₀                                      O₁
[[ 1.1840,  0.6626],                    [[-1.2709, -0.5757],
 [-0.9887,  0.0192],                     [ 1.2379, -0.1742],
 [ 0.0359, -1.2394],                     [ 0.5160,  0.5479],
 [-0.8555, -0.4854]]                     [ 1.2204,  0.1535]]
```

**These attention rows are far sharper than GPT-1's.** Row `the`, head 0: `0.8723` on itself here,
against `0.3124` in GPT-1. The cause is mechanical — `ln_1` rescales `X` to unit variance per row,
which enlarges `Q` and `K`, which enlarges the pre-softmax logits. Pre-LN changes what the attention
sees, not just where the normalisation sits.

```
concat = [O₀ ‖ O₁]                      MHA = concat @ W_o
[[ 1.1840,  0.6626, -1.2709, -0.5757],  [[ 1.0048,  0.6572, -1.0830, -0.5790],
 [-0.9887,  0.0192,  1.2379, -0.1742],   [-0.7641, -0.0990,  0.9979, -0.0310],
 [ 0.0359, -1.2394,  0.5160,  0.5479],   [-0.0400, -1.0571,  0.5228,  0.4208],
 [-0.8555, -0.4854,  1.2204,  0.1535]]   [-0.6964, -0.5071,  1.0282,  0.2116]]
```

### 6.1 First residual add — no LayerNorm

```
g1 = X + MHA                    <- the ORIGINAL X, not a normalised copy
[[ 2.2048,  1.5572, -1.0830, -0.1790],
 [-0.5641,  0.4010,  1.7979,  0.3690],
 [ 0.3600, -1.1571,  0.9228,  1.0208],
 [-0.4964, -0.0071,  2.3282,  0.9116]]
```

Compare row variances against GPT-1's `h1`, which LayerNorm pinned to 1:

```
GPT-1  h1  = LN(X + MHA)   -> var = 1.0 by construction, every row
GPT-2  g1  =    X + MHA    -> var = [1.7320, 0.7109, 0.7583, 1.1565]
```

The stream is free to grow. §11 is about what that costs at depth.

### 6.2 `ln_2` and the FFN

```
per-row mean: [0.6250, 0.5009, 0.2866, 0.6841]
per-row var : [1.7320, 0.7109, 0.7583, 1.1565]

n2 = LayerNorm(g1)
[[ 1.2004,  0.7083, -1.2978, -0.6109],
 [-1.2631, -0.1185,  1.5382, -0.1565],
 [ 0.0843, -1.6578,  0.7305,  0.8431],
 [-1.0977, -0.6427,  1.5288,  0.2116]]

Z = n2 @ W1
[[ 0.5683, -0.4017, -0.4796,  0.8181, -0.2261, -0.2931,  0.3499, -0.2455],
 [-0.7297,  0.3083,  0.5283, -1.0099,  0.5404,  0.0749, -0.3745,  0.5954],
 [-0.1180, -0.1039,  0.5479, -0.1826, -0.4524,  0.7446, -0.4748, -0.2894],
 [-0.6570,  0.2643,  0.6091, -0.9197,  0.3113,  0.3002, -0.4562,  0.3976]]

GELU(Z)
[[ 0.4064, -0.1382, -0.1514,  0.6491, -0.0928, -0.1128,  0.2228, -0.0990],
 [-0.1699,  0.1915,  0.3705, -0.1578,  0.3812,  0.0397, -0.1326,  0.4312],
 [-0.0535, -0.0477,  0.3880, -0.0781, -0.1473,  0.5747, -0.1507, -0.1118],
 [-0.1679,  0.1597,  0.4439, -0.1645,  0.1937,  0.1855, -0.1479,  0.2602]]

FFN out = GELU(Z) @ W2                  g2 = g1 + FFN_out
[[ 0.1018, -0.0181,  0.3079,  0.1163],  [[ 2.3066,  1.5391, -0.7751, -0.0626],
 [ 0.0509,  0.2558, -0.1804,  0.1327],   [-0.5132,  0.6568,  1.6175,  0.5017],
 [ 0.2671, -0.1396,  0.0021,  0.0693],   [ 0.6271, -1.2966,  0.9249,  1.0901],
 [ 0.1213,  0.1545, -0.1516,  0.1170]]   [-0.3751,  0.1474,  2.1766,  1.0286]]
```

---

## 7. The final LayerNorm (`ln_f`)

```
per-row mean: [0.7520, 0.5657, 0.3364, 0.7444]
per-row var : [1.5079, 0.5707, 0.9164, 0.9354]

GPT-2 OUTPUT = LayerNorm(g2)
             dim0     dim1     dim2     dim3
bank     [ 1.2660,  0.6410, -1.2436, -0.6634]
approved [-1.4282,  0.1206,  1.3923, -0.0847]
the      [ 0.3037, -1.7058,  0.6148,  0.7873]
loan     [-1.1575, -0.6172,  1.4808,  0.2939]
```

**Why pre-LN must have this and post-LN must not.** In post-LN every block ends with a LayerNorm, so
the stack's output is already normalised. In pre-LN nothing normalises the stream — `g2` here has
row variances `[1.5079, 0.5707, 0.9164, 0.9354]`, and in a 48-layer model they would be far larger
(§11). Feeding that straight into a tied LM head would make the logits scale with depth. `ln_f` is
the one place the stream is brought back to unit scale before the head reads it.

**Interview form:** *"pre-LN moves LayerNorm to the input of each sub-block, and adds one final
LayerNorm after the last block."* The second half is the part people drop, and it is in the GPT-2
paper's own sentence.

---

## 8. LM head, loss, perplexity

`W_lm = Eᵀ`, tied — the mechanism, the hand-verification, and the gradient identity are all in
[06_gpt1_end_to_end.md §7 and §9.2](06_gpt1_end_to_end.md), unchanged in GPT-2.

```
logits = OUTPUT @ Eᵀ
           <bos>     bank  approved      the     loan  granted rejected    <eos>
bank    [ 0.0000,  1.5880, -0.1885, -0.1907, -1.6499, -0.5698,  0.3836,  0.0268]
apprvd  [ 0.0000, -1.2009,  0.1272,  0.1308,  1.1853,  0.3887, -0.2651, -0.0703]
the     [ 0.0000, -0.9208,  0.2321,  0.1402,  1.1832,  0.5125, -0.1886, -0.0832]
loan    [ 0.0000, -1.4738,  0.2098,  0.1775,  1.5679,  0.5647, -0.3226, -0.0917]

probs
bank    [ 0.0926,  0.4531,  0.0767,  0.0765,  0.0178,  0.0524,  0.1359,  0.0951]
apprvd  [ 0.0998,  0.0300,  0.1133,  0.1137,  0.3264,  0.1472,  0.0765,  0.0930]
the     [ 0.0953,  0.0380,  0.1202,  0.1097,  0.3112,  0.1591,  0.0789,  0.0877]
loan    [ 0.0844,  0.0193,  0.1041,  0.1008,  0.4048,  0.1485,  0.0611,  0.0770]
```

```
pos  input      gold       p(gold)    greedy   loss
 0   bank       approved   0.076681   bank     2.568106
 1   approved   the        0.113717   loan     2.174045
 2   the        loan       0.311155   loan     1.167463
 3   loan       <eos>      0.077004   loan     2.563896

mean loss  = 2.118378
perplexity = 8.317633
```

The `<bos>` column is exactly `0.0000` here too — same constant-embedding/LayerNorm argument as
GPT-1 §7.2, and `ln_f` guarantees the output is zero-mean, so it holds for pre-LN as well.

---

## 9. Backward — what pre-LN does to the gradients

Torch forward matched numpy to **4.4e-16**.

```
                  max |grad|        L2
  dL/dEMB          0.733778      1.286727    <- largest
  dL/dPOS          0.413963      0.553748
  dL/dW_o          0.147265      0.290268
  dL/dW_v          0.133773      0.245090
  dL/dW2           0.059485      0.117136
  dL/dW1           0.045864      0.110844
  dL/dW_k          0.031450      0.061435
  dL/dW_q          0.022985      0.050084
```

Run GPT-1 and GPT-2 on identical weights and compare the **same tensors**:

```
  tensor    GPT-1 post-LN    GPT-2 pre-LN    ratio
  W_o            0.347900        0.147265    0.42x
  W_v            0.304313        0.133773    0.44x
  W_q            0.041354        0.022985    0.56x
  W_k            0.027651        0.031450    1.14x
  EMB            0.675427        0.733778    1.09x
  POS            0.523971        0.413963    0.79x
```

**Pre-LN roughly halves the gradient reaching the attention branch weights while slightly
*increasing* the gradient reaching the embeddings.** That is the identity path doing its job: in
post-LN, gradient flowing back to the embeddings must pass through every block's LayerNorm; in
pre-LN it can travel the residual stream directly, and the branches are the part that gets damped.

At 12 layers this is a mild effect. At 48 it is the difference between needing a warmup schedule and
not — the practical reason GPT-2 could be trained deeper without the instability GPT-1's recipe
guarded against with 2000 warmup steps.

---

## 10. Weight update

```
SGD, lr = 0.3

loss  2.118378 -> 1.771219     DECREASED by 0.347159   ✓
PPL   8.3176   -> 5.8780
```

```
    lr        loss       ppl    p(gold) per position
     0    2.118378    8.3176    [0.0767, 0.1137, 0.3112, 0.0770]
   0.1    1.928715    6.8807    [0.0904, 0.1458, 0.3921, 0.0864]
   0.3    1.771219    5.8780    [0.1227, 0.1547, 0.4102, 0.1076]
   0.5    1.817839    6.1585    [0.1603, 0.0982, 0.3422, 0.1290]   <- worse than 0.3
   1.0    2.142283    8.5189    [0.2576, 0.0278, 0.1694, 0.1563]   <- worse than start
```

```
greedy before : [bank, loan, loan, loan]
greedy after  : [bank, bank, loan, loan]
gold          : [approved, the, loan, <eos>]
```

Note the optimum sits at `lr = 0.3` here, against `0.5` for GPT-1 on the same weights. Pre-LN's
larger embedding gradients mean a smaller step is enough — a toy-scale echo of the real result that
pre-LN tolerates *higher* LRs without warmup once the residual init is also correct (§11).

---

## 11. The `1/√N` residual initialisation

The third change, and the one most often missed. From the GPT-2 paper:

> *"A modified initialization which accounts for the accumulation on the residual path with model
> depth is used. We scale the weights of residual layers at initialization by a factor of `1/√N`
> where `N` is the number of residual layers."*

**`N` counts residual *layers*, not blocks** — each block has two (attention output, MLP output), so
`N = 2 × n_layer`. In practice the scaling is applied to the two output projections (`attn.c_proj`
and `mlp.c_proj`).

### Why — measured

Stack the same pre-LN block repeatedly and track the residual stream's RMS:

```
 blocks   no scaling   × 1/√(2L)   ratio
      1       1.1708      1.0008    1.17x
      2       1.8295      1.1852    1.54x
      4       3.2729      1.4663    2.23x
      8       6.3054      1.8858    3.34x
     12       9.3998      2.2174    4.24x
     24      18.7919      2.9824    6.30x
     48      37.7022      4.0847    9.23x
```

**Unscaled, the stream grows roughly linearly with depth** — each block adds a branch of about unit
variance onto a stream nothing renormalises. By 48 blocks the RMS is `37.70`. With the `1/√N` factor
it is `4.08`, a **9.23×** reduction.

That growth is not fatal by itself — `ln_f` rescales at the end — but at initialisation it means
the later blocks' contributions are swamped by the accumulated stream, so early training barely
updates them. Scaling the residual outputs down makes every block contribute comparably at step 0.

```
 n_layer   N     1/√N       HF init std = 0.02/√N
      12   24   0.204124    0.00408248
      24   48   0.144338    0.00288675
      36   72   0.117851    0.00235702
      48   96   0.102062    0.00204124
```

**This is the pairing to remember:** pre-LN removes the *need* for warmup; the `1/√N` residual init
is what makes pre-LN behave at depth. Quoting one without the other is half the answer.

---

## 12. Byte-level BPE — why 50,257

```
   256  byte tokens          every possible byte value 0-255
+ 50,000  learned merges
+      1  <|endoftext|>
= 50,257
```

Exact — the vocabulary size is fully accounted for.

**The consequence: GPT-2 can never emit `<UNK>`.** Because the base alphabet is *bytes*, not
characters, any Unicode string whatsoever — emoji, Chinese, corrupted bytes, a PNG read as text —
decomposes into tokens the model already has. GPT-1's 40,478-entry character-level BPE had no such
guarantee and needed an unknown-token fallback.

The cost is efficiency on non-English text: a Chinese character is 3 UTF-8 bytes and often becomes
2–3 tokens, so the same sentence costs several times more tokens than its English translation. That
is a billing and context-length fact, not just a curiosity.

GPT-2 also applies a regex pre-tokenizer that prevents merges from crossing character categories, so
a merge never spans a letter and a punctuation mark. Detail in
[../../4.nlp/01_fundamentals/03_tokenization.md](../../4.nlp/01_fundamentals/03_tokenization.md) — board 7.

---

## 13. Zero-shot task transfer

The GPT-2 paper's actual claim was not architectural. GPT-1 fine-tuned a task head per task with an
auxiliary LM loss ([06_gpt1_end_to_end.md §13](06_gpt1_end_to_end.md)). **GPT-2 removed all of it.**

```
GPT-1                                  GPT-2
  pretrain LM                            pretrain LM
  + task head per task                   (nothing)
  + fine-tune on labelled data           condition on a natural-language prompt
  + auxiliary LM loss (λ = 0.5)
  + delimiter-based input transforms
```

```
summarisation   <article> TL;DR:
translation     english = <sentence>  french =
question answer <document> Q: <question> A:
```

The task is specified **inside the input**, so one frozen set of weights does every task. The
architecture did not change to enable this — scale and data did. GPT-3 then made the same point at
175B with few-shot examples in the prompt.

**Why it matters for the arc:** every prompting technique — instructions, few-shot examples, chain
of thought — descends from this observation. `<article> TL;DR:` is the first prompt.

---

## 14. Where the parameters live — all four sizes

Exact accounting, including biases and LayerNorm gains/biases:

```
GPT-2 small  (12 layers, d_model 768, d_ff 3072)

token embeddings      50257 × 768            =  38,597,376
position embeddings    1024 × 768            =     786,432
                                                ──────────
                                                 39,383,808   (31.6%)

per layer:
  ln_1                 2 × 768               =       1,536
  attn c_attn      768 × 2304 + 2304         =   1,771,776
  attn c_proj       768 × 768 + 768          =     590,592
  ln_2                 2 × 768               =       1,536
  mlp  c_fc        768 × 3072 + 3072         =   2,362,368
  mlp  c_proj      3072 × 768 + 768          =   2,360,064
                                                ──────────
                                                   7,087,872
× 12 layers                                   =  85,054,464
ln_f                    2 × 768               =       1,536   <- pre-LN only
                                                ──────────
TOTAL                                         = 124,439,808
```

Matches the released checkpoint (HuggingFace `gpt2`) exactly.

### The ladder

```
  name     layers   d_model    parameters
  small        12       768     124,439,808
  medium       24      1024     354,823,168
  large        36      1280     774,030,080
  XL           48      1600   1,557,611,200
```

All four match their released checkpoints (`gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`).

### The "117M" discrepancy — know this one

The GPT-2 **paper's table lists the smallest model as 117M**, but the released checkpoint is
**124,439,808**. `117M` is GPT-1's size (`116,534,784`), and the paper describes the smallest GPT-2
as *"equivalent to the original GPT"* — but GPT-2 small has a **larger vocabulary** (50,257 vs
40,478) and **double the context** (1024 vs 512), so it is genuinely bigger.

```
GPT-1        116,534,784
GPT-2 small  124,439,808
difference     7,905,024  =  embeddings 7,903,488  +  ln_f 1,536
```

**The 12-layer stacks are identical: 85,054,464 in both.** Every parameter of the difference is
embeddings plus the one final LayerNorm.

---

## 15. Quick reference

```
GPT-2 LAYER  (pre-LN)

 1. X   = E_token + E_position                  residual stream, depth 0
 2. n1  = LayerNorm(X)                          LN on the BRANCH input
 3. Q,K,V = n1 @ Wq, n1 @ Wk, n1 @ Wv           <- note: n1, not X
 4. S = Q Kᵀ / √d_head ;  S[i,j] = -inf, j > i
 5. A = softmax(S) ; O = A @ V ; MHA = concat @ W_o
 6. g1  = X + MHA                               stream NOT normalised
 7. n2  = LayerNorm(g1)
 8. g2  = g1 + GELU(n2 @ W1) @ W2

    ... repeat 2-8 per layer ...

 9. out = LayerNorm(g2)                         ln_f — ONCE, after the last block
10. logits = out @ Eᵀ                           tied
11. loss = cross_entropy(logits, next_token)    every position
```

**The seven things to be able to say cold:**

1. Pre-LN is `x + f(LN(x))`; post-LN is `LN(x + f(x))`. GPT-2 is pre-LN, GPT-1 is post-LN.
2. Pre-LN **adds a final LayerNorm** after the last block — without it the logits would scale with
   depth, because nothing else normalises the stream.
3. The residual stream in pre-LN is an **unbroken identity path** from embeddings to `ln_f`. Measured
   here: pre-LN gives the attention branch weights `0.42×` the gradient of post-LN and the embeddings
   `1.09×`.
4. Pre-LN removes the need for **LR warmup**; GPT-1's post-LN recipe used 2000 warmup steps.
5. **`1/√N` residual init**, `N = 2 × n_layer`, is the companion change. Without it the stream RMS
   grows ~linearly with depth — `37.70` vs `4.08` at 48 blocks.
6. Vocabulary `50,257 = 256 bytes + 50,000 merges + 1 <|endoftext|>`. Byte-level BPE means
   **`<UNK>` is impossible**, at the cost of token efficiency on non-English text.
7. GPT-2 small is **124,439,808**, not the paper's 117M. The 12-layer stack is identical to GPT-1's;
   the entire difference is embeddings (`7,903,488`) plus `ln_f` (`1,536`).

---

## See also

- [06_gpt1_end_to_end.md](06_gpt1_end_to_end.md) — GPT-1: post-LN, weight tying in full, sampling
- [06c_gpt3_end_to_end.md](06c_gpt3_end_to_end.md) — GPT-3: this same block, plus sparse attention, scale, in-context learning
- [05_bert_end_to_end.md](05_bert_end_to_end.md) — the same block without the causal mask
- [08_modern_llm_architecture.md](08_modern_llm_architecture.md) — what replaced LayerNorm and GELU after GPT-2: RMSNorm, SwiGLU, RoPE, GQA
- [../../4.nlp/01_fundamentals/03_tokenization.md](../../4.nlp/01_fundamentals/03_tokenization.md) — byte-level BPE in detail
- [02_gpt_family.md](02_gpt_family.md) — GPT-1 → 2 → 3 → 4, and what scaled

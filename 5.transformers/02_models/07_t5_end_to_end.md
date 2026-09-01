# 07 — T5: End-to-End with Relative Position Bias

> **This file is T5 only** (Raffel et al., 2020, *Exploring the Limits of Transfer Learning with a
> Unified Text-to-Text Transformer*). BART is a separate file:
> [07b_bart_end_to_end.md](07b_bart_end_to_end.md). Nothing here is mixed between them.
>
> Arc: [06c decoder](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md) →
> [05 BERT](05_bert_end_to_end.md) → [06 GPT-1](06_gpt1_end_to_end.md) →
> [06b GPT-2](06b_gpt2_end_to_end.md) → [06c GPT-3](06c_gpt3_end_to_end.md) → **T5**.
> Same toy dimensions throughout: `d_model=4`, `n_heads=2`, `d_head=2`, `d_ff=8`.
>
> Every number verified in numpy and re-checked against `torch.autograd`
> (forward `allclose` to 2.2e-16; loss `2.201226 → 1.675668`, perplexity `9.0361 → 5.3424`).

---

## T5 in one box

```
architecture ENCODER-DECODER — both stacks, cross-attention in the decoder
vocabulary   32,128        32,000 SentencePiece + 100 sentinels, padded to 32,128
positions    RELATIVE POSITION BIAS — a learned scalar added to attention logits.
             T5 adds NO position vector to the input embedding at all.
norm         RMSNorm  — no mean subtraction, no bias — PRE-norm, plus a final norm per stack
biases       NONE, anywhere
activation   ReLU  (T5 v1.0)   ·   GEGLU  (T5 v1.1)
attn scale   NO 1/sqrt(d_k)  — folded into initialisation
embeddings   TIED THREE WAYS: encoder input, decoder input, and the output head
             (and the decoder output is scaled by d_model^-0.5 before the head)
pretraining  span corruption on C4 — 15% of tokens, mean span length 3
sizes        60.5M / 222.9M / 737.7M / 2.85B / 11.3B
```

**Four of those lines are things no other model in this arc does.** They are why T5 gets its own file
rather than a paragraph.

---

## Table of Contents

1. Span corruption — the input and target
2. **Relative position bias** — buckets, hand-computed
3. RMSNorm — and what "simplified LayerNorm" means
4. Encoder forward
5. Decoder forward — masked self-attention
6. Cross-attention
7. The tied head and the `d_model^-0.5` rescale
8. Loss
9. Backward — including gradient into the position bias
10. Weight update
11. Where T5's parameters live
12. T5 v1.0 vs v1.1
13. What T5 shares with everything else
14. Quick reference

---

## 1. Span corruption

T5's pretraining objective. Drop random spans, replace **each span with a single sentinel token**,
and ask the decoder to produce only the dropped spans — each prefixed by its sentinel.

```
original          bank approved the loan
                       ^^^^^^^^  drop this span

encoder input     bank <X> the loan
decoder target    <X> approved </s>
```

**The target is not the original document.** It is only the missing pieces. A 512-token input with
15% corruption produces a target of roughly 15% the length — so the decoder does far less work per
example than a full reconstruction would need. That efficiency is the whole design.

```
15% of tokens corrupted · mean span length 3 · so ~5% of positions become sentinels
a 512-token input -> ~77 corrupted tokens in ~26 spans -> target ~103 tokens
```

### 1.1 Vocabulary for this walkthrough

`V = 8`. `bank`, `approved`, `the`, `loan` reuse 06b's vectors.

| Token | Index | Embedding | Role |
|---|---|---|---|
| `<pad>` | 0 | `[0.05, 0.05, 0.05, 0.05]` | also the decoder's start token |
| `</s>` | 1 | `[0.15, 0.25, 0.10, 0.30]` | end of sequence |
| `<X>` | 2 | `[0.40, 0.30, 0.20, 0.10]` | sentinel |
| bank | 3 | `[1.00, 0.80, 0.10, 0.10]` | |
| approved | 4 | `[0.30, 0.20, 0.40, 0.30]` | the dropped span |
| the | 5 | `[0.10, 0.10, 0.20, 0.20]` | |
| loan | 6 | `[0.10, 0.10, 1.00, 0.90]` | |
| granted | 7 | `[0.20, 0.10, 0.50, 0.40]` | |

**T5 has no `<bos>`.** The decoder starts with `<pad>`, which is unusual and catches people out.

---

## 2. Relative position bias

**T5 adds nothing to the input embedding.** No sinusoid, no learned position vector, no rotation.
Position enters *only* here — as a learned scalar added to each attention logit:

```
score(i, j) = Qᵢ · Kⱼ  +  b[head, bucket(j − i)]
                          ↑
                          learned, depends only on the RELATIVE distance
```

### 2.1 The bucket function

Relative distances are compressed into 32 buckets: exact for near distances, log-spaced beyond.
This is T5's actual `relative_position_bucket` with `num_buckets=32, max_distance=128`:

```
   rel   bucket        rel   bucket
  -200      15           0        0
   -64      14           1       17
   -16      10           3       19
    -8       8           8       24
    -3       3          16       26
    -1       1          64       30
                       200       31
```

Two halves: **buckets 0–15 for looking backward, 16–31 for looking forward** (the encoder is
bidirectional so it needs both). Within each half, distances 0–7 get their own bucket and everything
beyond is log-spaced and capped.

**A 200-token gap and a 512-token gap land in the same bucket.** That is deliberate: precision where
it matters, cheap sharing where it does not — and it means the scheme is defined for *any* distance,
including lengths never seen in training.

### 2.2 The bucket matrix for our 4-token encoder input

```
bucket(j − i), bidirectional, L = 4          tokens: bank  <X>  the  loan
     j=0  j=1  j=2  j=3
i=0 [  0   17   18   19 ]      diagonal = 0 (self)
i=1 [  1    0   17   18 ]      below diagonal = 1,2,3   (looking BACK)
i=2 [  2    1    0   17 ]      above diagonal = 17,18,19 (looking FORWARD, +16 offset)
i=3 [  3    2    1    0 ]
```

The matrix is **constant along each diagonal** — that is what "relative" means. Bucket 1 is used
wherever a token looks one position back, regardless of where in the sequence that happens.

Decoder (causal, unidirectional — only backward distances exist):

```
bucket(j − i), L = 3
     [ 0  0  0 ]
     [ 1  0  0 ]      upper triangle is irrelevant — those positions get masked anyway
     [ 2  1  0 ]
```

### 2.3 The learned bias table → the bias matrix

The learned parameter is `(n_heads × 32)` scalars. For the buckets this example touches:

```
bucket:      0      1      2      3     17     18     19
head 0:   0.50   0.30  -0.10  -0.40   0.20  -0.20   0.10
head 1:  -0.30   0.10   0.40   0.20   0.50   0.30  -0.10
```

Look up each bucket and you get the bias matrix added to the logits:

```
head 0                                  head 1
[[ 0.5000,  0.2000, -0.2000,  0.1000],  [[-0.3000,  0.5000,  0.3000, -0.1000],
 [ 0.3000,  0.5000,  0.2000, -0.2000],   [ 0.1000, -0.3000,  0.5000,  0.3000],
 [-0.1000,  0.3000,  0.5000,  0.2000],   [ 0.4000,  0.1000, -0.3000,  0.5000],
 [-0.4000, -0.1000,  0.3000,  0.5000]]   [ 0.2000,  0.4000,  0.1000, -0.3000]]
```

Each is a **Toeplitz matrix** — constant along diagonals — built from just 7 distinct numbers.

**Cost:** `32 × n_heads` per stack. For T5-Base that is `32 × 12 × 2 = 768` parameters total,
**0.0003% of the model**. And T5 computes it in the first layer only, then reuses it in every
other layer.

---

## 3. RMSNorm — "simplified LayerNorm"

T5 normalises without subtracting the mean and without a bias term:

```
LayerNorm (BERT, GPT)          T5's norm  (= RMSNorm)
  mu  = mean(x)                  ms  = mean(x²)
  var = mean((x-mu)²)            out = x / sqrt(ms + eps) * gamma
  out = (x-mu)/sqrt(var+eps)
        * gamma + beta           NO mean subtraction.  NO beta.
```

**T5 shipped this in 2019 — years before Llama made RMSNorm standard.** It is usually credited to
Llama; it is T5's.

Worked on the encoder input, `eps = 1e-6`:

```
X_enc (token embeddings only — nothing added)     mean-square per row
[[1.0000, 0.8000, 0.1000, 0.1000],                0.415000
 [0.4000, 0.3000, 0.2000, 0.1000],                0.075000
 [0.1000, 0.1000, 0.2000, 0.2000],                0.025000
 [0.1000, 0.1000, 1.0000, 0.9000]]                0.457500

RMSNorm(X_enc)
[[1.5523, 1.2418, 0.1552, 0.1552],
 [1.4606, 1.0954, 0.7303, 0.3651],
 [0.6324, 0.6324, 1.2649, 1.2649],
 [0.1478, 0.1478, 1.4784, 1.3306]]
```

Row 0: `sqrt(0.415) = 0.6442`, and `1.0 / 0.6442 = 1.5523` ✓.

**Rows do not have mean zero afterwards** — row 0 averages `0.776`. LayerNorm would have centred it.
RMSNorm only fixes the *scale*. That is the entire difference, it removes one mean and one
subtraction per row, and it costs nothing measurable in quality.

---

## 4. Encoder forward

**No `1/√d_k`.** T5's attention computes `Q·Kᵀ` and adds the bias — no scaling term. The scale is
folded into how `W_q` is initialised.

```
Q = RMSNorm(X) @ Wq                     K = RMSNorm(X) @ Wk
[[1.8628, 1.4902, 0.1863, 0.1863],      [[1.8007, 1.5523, 0.1863, 0.1863],
 [1.7527, 1.3145, 0.8764, 0.4382],       [1.6797, 1.3876, 0.8033, 0.5112],
 [0.7589, 0.7589, 1.5179, 1.5179],       [0.7589, 0.7589, 1.5179, 1.5179],
 [0.1774, 0.1774, 1.7741, 1.5967]]       [0.1774, 0.1774, 1.7446, 1.6263]]

V = RMSNorm(X) @ Wv
[[1.5213, 1.2729, 0.1552, 0.1552],
 [1.4241, 1.1320, 0.6938, 0.4017],
 [0.6324, 0.6324, 1.2649, 1.2649],
 [0.1478, 0.1478, 1.4637, 1.3454]]
```

```
scores + bias, head 0                   A₀ = softmax
[[6.1675, 5.3966, 2.3447, 0.6949],      [[0.6717, 0.3107, 0.0147, 0.0028],
 [5.4966, 5.2679, 2.5278, 0.3442],       [0.5397, 0.4294, 0.0277, 0.0031],
 [2.4447, 2.6278, 1.6520, 0.4693],       [0.3581, 0.4301, 0.1621, 0.0497],
 [0.1949, 0.4442, 0.5693, 0.5630]]       [0.1930, 0.2476, 0.2806, 0.2788]]

scores + bias, head 1                   A₁ = softmax
[[-0.2306, 0.7449, 0.8655, 0.5279],     [[0.1139, 0.3021, 0.3408, 0.2432],
 [ 0.3449, 0.6280, 2.4953, 2.5414],      [0.0502, 0.0667, 0.4314, 0.4517],
 [ 0.9655, 2.0953, 4.3078, 5.6165],      [0.0073, 0.0226, 0.2064, 0.7638],
 [ 0.8279, 2.6414, 5.2165, 5.3918]]      [0.0054, 0.0334, 0.4386, 0.5226]]
```

**Bidirectional** — no mask. Row `bank` attends forward to `<X>`, `the`, `loan` freely.

> **The logits are large (up to 6.17), and the softmax is correspondingly peaked.** That is the
> direct consequence of dropping `1/√d_k` with weights that were not initialised for it. Real T5
> initialises `W_q` at a scale that keeps these in range. §9 shows what the peaked softmax costs.

```
concat                                  g1 = X_enc + attn        (pre-norm residual)
[[1.4741, 1.2165, 1.0143, 0.8973],      [[2.5498, 2.1320, 1.2500, 1.1306],
 [1.4506, 1.1911, 1.2609, 1.1880],       [1.9507, 1.6358, 1.5986, 1.4144],
 [1.2672, 1.0526, 1.3957, 1.2988],       [1.4853, 1.3039, 1.7127, 1.6137],
 [0.8648, 0.7446, 1.3436, 1.2721]]       [1.0872, 0.9838, 2.4230, 2.2537]]

RMSNorm(g1)                             mean-square [3.4719, 2.7594, 2.3609, 3.2749]
[[1.3684, 1.1442, 0.6709, 0.6068],
 [1.1743, 0.9848, 0.9624, 0.8514],      ReLU(Z) @ W2  = FFN out
 [0.9666, 0.8486, 1.1147, 1.0502],      [[0.3003, 0.2780, 0.4999, 0.2354],
 [0.6008, 0.5436, 1.3389, 1.2454]]       [0.3188, 0.3059, 0.4512, 0.2463],
                                         [0.3116, 0.3243, 0.4163, 0.2436],
g2 = g1 + FFN                            [0.2954, 0.3213, 0.3092, 0.2273]]
[[2.8501, 2.4100, 1.7499, 1.3661],
 [2.2695, 1.9417, 2.0499, 1.6607],      ENCODER OUTPUT = RMSNorm(g2)
 [1.7969, 1.6282, 2.1290, 1.8574],      [[1.3126, 1.1099, 0.8059, 0.6291],
 [1.3826, 1.3051, 2.7322, 2.4810]]       [1.1390, 0.9745, 1.0288, 0.8334],
                                         [0.9652, 0.8746, 1.1436, 0.9977],
mean-square [4.7148,3.9702,3.4656,4.3088] [0.6661, 0.6287, 1.3162, 1.1952]]
```

T5 v1.0's FFN is **ReLU**, not GELU. `Z` had 4 negative entries of 32; ReLU zeroed them.

---

## 5. Decoder — masked self-attention

```
decoder input   <pad>  <X>  approved       target   <X>  approved  </s>
```

```
X_dec                                   RMSNorm(X_dec)   mean-square [0.0025, 0.0750, 0.0950]
[[0.0500, 0.0500, 0.0500, 0.0500],      [[0.9998, 0.9998, 0.9998, 0.9998],
 [0.4000, 0.3000, 0.2000, 0.1000],       [1.4606, 1.0954, 0.7303, 0.3651],
 [0.3000, 0.2000, 0.4000, 0.3000]]       [0.9733, 0.6489, 1.2978, 0.9733]]
```

Row 0 normalises to `[0.9998, 0.9998, 0.9998, 0.9998]` — `<pad>` is a constant vector, and RMSNorm
divides it by its own RMS, so it returns ≈1 in every dimension. It is `0.9998` rather than exactly
`1.0000` **only because of the `eps`**: `0.05 / √(0.0025 + 1e-6) = 0.99980`.

**Under LayerNorm this row would be all-zeros**, because mean subtraction annihilates a constant
vector. RMSNorm preserves it. That is a real, load-bearing difference between the two norms, and
`<pad>` is exactly the kind of token where it shows up.

```
decoder position bias, head 0           head 1
[[ 0.4000,  0.4000,  0.4000],           [[ 0.2000,  0.2000,  0.2000],
 [ 0.2000,  0.4000,  0.4000],            [-0.3000,  0.2000,  0.2000],
 [-0.3000,  0.2000,  0.4000]]            [ 0.1000, -0.3000,  0.2000]]
```

Then `−∞` above the diagonal before softmax, exactly as
[06c §5](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md):

```
h1 = X_dec + masked self-attn
[[1.1498, 1.1498, 1.1498, 1.1498],
 [1.7953, 1.4986, 1.1779, 0.9468],
 [1.6255, 1.3215, 1.6510, 1.3884]]
```

---

## 6. Cross-attention

Q from the decoder, K and V from the encoder memory — the `3 × 4` rectangle.

**T5 uses no position bias in cross-attention.** The bias is zeroed there, because "relative
distance" between a target position and a source position is not a meaningful quantity — the two
sequences have independent coordinate systems.

```
cross A, head 0                         cross A, head 1
[[0.4102, 0.2823, 0.2027, 0.1048],      [[0.1149, 0.1926, 0.2700, 0.4225],
 [0.4415, 0.2826, 0.1895, 0.0864],       [0.1410, 0.2096, 0.2701, 0.3793],
 [0.4091, 0.2825, 0.2027, 0.1057]]       [0.1141, 0.1926, 0.2699, 0.4234]]

  columns:  bank    <X>     the    loan
```

Head 0 leans toward `bank`, head 1 toward `loan` — opposite ends of the source. Rows sum to 1 over
**source** positions.

```
h2 = h1 + cross-attn                    h3 = h2 + FFN
[[2.4489, 2.3758, 2.4798, 2.4088],      [[2.7452, 2.7112, 2.9225, 2.6502],
 [3.1092, 2.7371, 2.4884, 2.1851],       [3.4204, 3.0481, 2.9415, 2.4296],
 [2.9240, 2.5471, 2.9814, 2.6479]]       [3.2405, 2.8656, 3.4044, 2.8938]]

DECODER OUTPUT = RMSNorm(h3)
[[0.9949, 0.9826, 1.0592, 0.9605],
 [1.1474, 1.0225, 0.9867, 0.8150],
 [1.0421, 0.9216, 1.0948, 0.9306]]
```

---

## 7. The tied head and the `d_model^-0.5` rescale

T5 v1.0 ties **three** things to one matrix: the encoder's input embedding, the decoder's input
embedding, and the output projection. GPT ties two (§9.2 of
[06_gpt1_end_to_end.md](06_gpt1_end_to_end.md)); T5 ties three.

And it rescales the decoder output before the head — **only when tied**:

```
scaled = OUTPUT × d_model^-0.5 = OUTPUT × 0.5000
[[0.4975, 0.4913, 0.5296, 0.4803],
 [0.5737, 0.5113, 0.4934, 0.4075],
 [0.5211, 0.4608, 0.5474, 0.4653]]

logits = scaled @ Eᵀ
        <pad>    </s>     <X>    bank  approved     the    loan granted
pos0 [ 0.0999, 0.3945, 0.5003, 0.9915, 0.6034, 0.3009, 1.0607, 0.6055]
pos1 [ 0.0993, 0.3855, 0.5223, 1.0728, 0.5940, 0.2887, 0.9686, 0.5756]
pos2 [ 0.0997, 0.3877, 0.5027, 0.9910, 0.6070, 0.3007, 1.0644, 0.6101]
```

**What the rescale does, measured:**

```
row 0 logits WITHOUT rescale : [0.1999, 0.7890, 1.0007, 1.9830, 1.2068, 0.6017, 2.1214, 1.2111]
row 0 logits WITH    rescale : [0.0999, 0.3945, 0.5003, 0.9915, 0.6034, 0.3009, 1.0607, 0.6055]

max probability  without: 0.275572     with: 0.194726
```

The rescale **halves every logit and flattens the distribution**. Its purpose is compensation: the
embedding matrix was initialised for its role as an *input* lookup, and reusing it as an output
projection would otherwise produce logits scaled by roughly `√d_model`. T5 v1.1 unties the head and
drops this rescale entirely.

---

## 8. Loss

```
pos  input      gold       p(gold)    greedy   loss
 0   <pad>      <X>        0.111186   loan     2.196547
 1   <X>        approved   0.122854   bank     2.096760
 2   approved   </s>       0.099224   loan     2.310372

mean loss  = 2.201226
perplexity = 9.036087        (uniform over 8 = 2.0794 / PPL 8)
```

Slightly worse than uniform — correct for an untrained model.

---

## 9. Backward

Torch forward matched numpy to **2.2e-16**.

```
                  max |grad|        L2
  dL/dEMB          0.205621      0.573532    <- dominates everything
  dL/dW_o          0.015670      0.044033
  dL/dW_v          0.013331      0.035958
  dL/dWo_c         0.010710      0.030618
  dL/dWv_c         0.010396      0.028032
  dL/dW2           0.008411      0.022569
  dL/dW1           0.005714      0.016539
  dL/dW_q          0.001153      0.002491
  dL/dWq_c         0.000622      0.001963
  dL/dWk_c         0.000631      0.002032
  dL/dW_k          0.000411      0.001188
  dL/dB_dec        0.000310      0.000680    <- the position bias
  dL/dB_enc        0.000279      0.000559
```

**Two things worth naming.**

**1. `dL/dEMB` is 13× the next largest.** That is what tying three ways buys — the embedding
receives gradient as the encoder input, as the decoder input, *and* as the output projection. Three
paths into one matrix. GPT's tying summed two ([06 §9.2](06_gpt1_end_to_end.md)); T5 sums three.

**2. Every attention gradient is tiny** — `W_k` at `0.000411` is 500× below `EMB`. This is the cost
of dropping `1/√d_k` with un-tuned weights: §4's logits reached 6.17, the softmax is nearly
one-hot, and a saturated softmax has a vanishing Jacobian. Real T5 avoids this by initialising for
the missing scale factor. **If you remove the `√d_k` division without changing initialisation, you
get exactly this — a model that barely trains.**

The position bias does receive gradient, and it is structured:

```
dL/dB_enc, head 0                       dL/dB_dec, head 0
[[-0.0000,  0.0000,  0.0000,  0.0000],  [[ 0.0000,  0.0000,  0.0000],
 [-0.0000,  0.0000,  0.0000,  0.0000],   [-0.0001,  0.0001,  0.0000],
 [-0.0001, -0.0001,  0.0001,  0.0001],   [-0.0001, -0.0002,  0.0003]]
 [-0.0002, -0.0002,  0.0001,  0.0003]]
```

Rows 0–1 of the encoder are ~zero: their attention is already saturated (`0.6717`, `0.5397`), so the
softmax passes almost nothing back. Row 3, the flattest row, carries the most gradient. **The
position bias learns fastest where attention is least decided** — the same softmax-Jacobian fact
seen from another angle.

---

## 10. Weight update

```
SGD, lr = 3.0            (large, because §9's gradients are small)

loss  2.201226 -> 1.675668     DECREASED by 0.525558   ✓
PPL   9.0361   -> 5.3424
```

```
    lr        loss       ppl     p(gold)                   greedy
     0    2.201226    9.0361    [0.1112, 0.1229, 0.0992]   [loan, bank, loan]
   0.5    2.053942    7.7986    [0.1367, 0.1368, 0.1128]   [bank, bank, loan]
   1.0    1.946359    7.0031    [0.1510, 0.1519, 0.1269]   [bank, bank, bank]
   2.0    1.782399    5.9441    [0.1676, 0.1821, 0.1561]   [bank, approved, approved]
   3.0    1.675668    5.3424    [0.1680, 0.2113, 0.1848]   [bank, approved, approved]
```

Position 1 reaches `approved` — the correct token, and the one the span corruption actually removed.

---

## 11. Where T5's parameters live

```
model        d_model   d_ff    L   heads  d_kv          parameters   paper
T5-Small         512   2048    6       8     64          60,506,624   60M
T5-Base          768   3072   12      12     64         222,903,552   220M
T5-Large        1024   4096   24      16     64         737,668,096   770M
T5-3B           1024  16384   24      32    128       2,851,598,336   3B
T5-11B          1024  65536   24     128    128      11,307,321,344   11B
```

`L` counts layers **in each stack** — T5-Base is 12 encoder + 12 decoder = 24 layers total.
T5-Base's count matches HuggingFace `t5-base` exactly.

```
T5-Base breakdown
  embeddings   24,674,304    (11.1%)   32,128 × 768, shared 3 ways
  encoder      84,954,240
  decoder     113,275,008              1.33× the encoder
  rel-pos bias        768              32 buckets × 12 heads × 2 stacks
```

**The decoder is 1.33× the encoder, and cross-attention is the entire difference** — a decoder layer
has 8 attention projections to the encoder's 4 (06c §16).

T5-3B and T5-11B scale `d_ff` grotesquely (16384, 65536) while holding `d_model` at 1024 — an
unusual choice that puts almost everything in the FFN. Later models scale width instead.

---

## 12. T5 v1.0 vs v1.1

Papers and checkpoints disagree, so be specific about which you mean:

| | v1.0 (the paper) | v1.1 |
|---|---|---|
| FFN | ReLU | **GEGLU** (gated, 3 matrices) |
| Embedding sharing | tied 3 ways | **output head untied** |
| `d_model^-0.5` rescale | ✓ | ✗ (no longer needed) |
| Pretraining | C4 + supervised mixture | **C4 only**, no supervised data |
| Dropout in pretraining | ✓ | ✗ |

**Flan-T5 is v1.1 plus instruction tuning**, and it is what you should actually use — the
`google/flan-t5-*` checkpoints beat raw T5 on essentially everything.

---

## 13. What T5 shares with everything else

So you do not over-credit it:

- **Encoder-decoder with cross-attention** — the 2017 paper; hand-computed in
  [06c](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md)
- **Bidirectional encoder** — BERT ([05](05_bert_end_to_end.md))
- **Causal decoder, teacher forcing, autoregressive inference** — 06c
- **Weight tying** — GPT-1 ([06 §7](06_gpt1_end_to_end.md)), though T5 ties three ways not two

T5's genuine contributions: **relative position bias, RMSNorm, no biases, span corruption, and the
text-to-text framing** — every task, including classification, cast as string in → string out.

---

## 14. Quick reference

```
T5 ENCODER LAYER                        T5 DECODER LAYER
 1. n  = RMSNorm(x)                      1. n  = RMSNorm(x)
 2. S  = Q Kᵀ + bias[h, bucket(j-i)]     2. S  = Q Kᵀ + bias, then -inf above diagonal
      NO 1/sqrt(d_k)                     3. x  = x + attn
 3. x  = x + attn                        4. n  = RMSNorm(x)
 4. n  = RMSNorm(x)                      5. cross-attn: Q from x, K/V from MEMORY, NO bias
 5. x  = x + ReLU(n W1) W2               6. x  = x + cross
                                         7. x  = x + ReLU(RMSNorm(x) W1) W2
 final: RMSNorm                          final: RMSNorm
                                         logits = (out × d_model^-0.5) @ Eᵀ
```

**The seven things to be able to say cold:**

1. **T5 adds no position vector to the input.** Position enters only as a learned scalar bias on the
   attention logits, indexed by a **bucket of the relative distance** — 32 buckets, exact to 7,
   log-spaced beyond, capped at 128.
2. The bias matrix is **Toeplitz** (constant along diagonals) and costs `32 × heads` per stack —
   **768 parameters in T5-Base**, computed in layer 1 and reused by all others.
3. **Cross-attention gets no position bias.** Relative distance between two different sequences is
   meaningless.
4. **T5's "simplified LayerNorm" is RMSNorm** — no mean subtraction, no bias — and it shipped in
   2019, before Llama.
5. **T5 does not divide by `√d_k`.** The scale is folded into initialisation. Do it without
   re-initialising and the softmax saturates: here `W_k`'s gradient is 500× below the embedding's.
6. **Span corruption**: each dropped span becomes one sentinel, and the target is **only the dropped
   spans** — not the whole document. That is what makes it cheap.
7. Embeddings are tied **three ways** (encoder in, decoder in, output head), so `dL/dE` sums three
   paths — and the tied head requires the `d_model^-0.5` rescale, which v1.1 removes by untying.

---

## See also

- [07b_bart_end_to_end.md](07b_bart_end_to_end.md) — BART: the other encoder-decoder, and how its denoising differs
- [../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md) — cross-attention and the causal mask, hand-computed
- [05_bert_end_to_end.md](05_bert_end_to_end.md) — the bidirectional encoder T5 reuses
- [03_encoder_decoder.md](03_encoder_decoder.md) — T5/BART family overview, Flan-T5, when to use which
- [08_modern_llm_architecture.md](08_modern_llm_architecture.md) — RMSNorm again, plus RoPE, SwiGLU, GQA
- [../../9.multimodal/05_donut_end_to_end.md](../../9.multimodal/05_donut_end_to_end.md) — this decoder with a Swin encoder supplying the memory

# 06c — Transformer Decoder with Cross-Attention: End-to-End

> Continuation of [06b_transformer_encoder_multihead.md](06b_transformer_encoder_multihead.md).
> **The numbers do not restart.** 06b's encoder output *is* the memory this decoder reads from,
> and every dimension is unchanged: `d_model = 4`, `n_heads = 2`, `d_head = 2`, `√d_k = 1.414`,
> `d_ff = 8`.
>
> Every number below was computed in numpy and re-checked against `torch.autograd`
> (forward `allclose` to 3.3e-16; loss `1.536187 → 1.090838` after one SGD step).

---

## Table of Contents

1. What changes from 06b
2. Setup — two sequences, one memory
3. Decoder input — embeddings + PE
4. Weight setup
5. **Masked self-attention** — `−inf` before softmax
6. Why `−inf` and not "zero it afterwards"
7. Add & Norm 1
8. **Cross-attention** — Q from decoder, K/V from encoder
9. Add & Norm 2
10. FFN, output, LM head, loss
11. **Teacher forcing vs autoregressive** — same block, two data flows
12. Exposure bias — what the two flows cost
13. Backward — three places the gradient goes
14. Weight update — verify the loss dropped
15. Why the KV cache exists
16. Decoder-only vs encoder-decoder
17. Quick reference

---

## 1. What changes from 06b

| | 06b (encoder) | **06c (this file)** |
|---|---|---|
| Attention blocks per layer | 1 | **2** (masked self, then cross) |
| Masking | none — bidirectional | **causal** on self-attention |
| Attention matrix shape | `L × L` = 4×4, square | self `3×3`, **cross `3×4` — rectangular** |
| Where K, V come from | the same sequence | self: decoder · **cross: the ENCODER** |
| Loss | MSE against role vectors | **cross-entropy over a vocabulary** |
| Output head | none — representations | **LM head `(4 × 5)` → logits → probs** |
| Sequence length | `L = 4` | `L_src = 4`, **`L_tgt = 3`** |

A decoder layer is an encoder layer with **one extra attention block wedged in the middle** and a
mask on the first one. That is the entire architectural difference. Three sub-layers, each wrapped
in residual + LayerNorm:

```
        x
        │
   ┌────┴─────────────────┐
   │  MASKED self-attn    │  Q,K,V all from the decoder;  future = -inf
   └────┬─────────────────┘
      Add & Norm  ────────────► h1
        │
   ┌────┴─────────────────┐
   │  CROSS-attn          │  Q from h1;  K,V from the ENCODER;  no mask
   └────┬─────────────────┘
      Add & Norm  ────────────► h2
        │
   ┌────┴─────────────────┐
   │  FFN                 │
   └────┬─────────────────┘
      Add & Norm  ────────────► decoder output
        │
     LM head  ──────────────► logits over the vocabulary
```

---

## 2. Setup — two sequences, one memory

```
SOURCE (encoder, from 06b):   bank  approved  the  loan       L_src = 4
TARGET (decoder, this file):  loan  granted  <eos>            L_tgt = 3

d_model = 4     n_heads = 2     d_head = 2     d_ff = 8     √d_k = √2 = 1.4142
```

The task is a paraphrase: *"bank approved the loan" → "loan granted"*. Short on purpose — three
target positions is the smallest length where a causal mask has anything to hide (position 1 must
not see position 2) and where the rows still fit on a page.

### 2.1 The memory — 06b's output, unchanged

This is the **only** thing the decoder ever sees of the source. It is `LayerNorm 2` of 06b §13,
copied verbatim:

```
MEMORY  (L_src, d_model) = (4, 4)
             dim0     dim1     dim2     dim3
bank     [-0.0753,  1.2246, -1.5323,  0.3829]
approved [ 0.2975, -0.2325, -1.4209,  1.3559]
the      [ 0.5105, -1.3907, -0.4019,  1.2821]
loan     [-0.3821, -1.3882,  0.4611,  1.3093]
```

**Two facts about this block that matter for everything below.**

1. It is computed **once**, before the decoder runs at all. It does not depend on the decoder's
   length, its contents, or which step of generation you are on. (§15 is built on this.)
2. It is **contextualised** — `bank` here does not mean the dictionary word, it means "bank, in a
   sentence about approving a loan". The decoder is reading the encoder's *conclusions*, not its
   inputs.

---

## 3. Decoder input — embeddings + PE

### 3.1 Target vocabulary

`V = 5`. Dimension meanings carry over from 06b: **dims 0-1 = entity/who, dims 2-3 = object/what.**
`loan` and `granted` reuse the exact vectors that encoder-side `loan` and `approved` had — the
embedding table is shared between encoder and decoder, which is what real translation models do
(and T5, and every decoder-only model, tie it to the output head too).

| Token | Index | Embedding | Role |
|---|---|---|---|
| `<sos>` | 0 | `[0.1, 0.1, 0.1, 0.1]` | neutral start symbol |
| `loan` | 1 | `[0.1, 0.1, 1.0, 0.9]` | object-heavy |
| `granted` | 2 | `[0.3, 0.2, 0.4, 0.3]` | balanced predicate |
| `bank` | 3 | `[1.0, 0.8, 0.1, 0.1]` | entity-heavy |
| `<eos>` | 4 | `[0.2, 0.2, 0.2, 0.2]` | stop symbol |

### 3.2 Positional encoding

Same table as 06b — same formula, same two frequency pairs, just the first three rows:

```
        dim0     dim1     dim2     dim3
pos 0 [ 0.0000,  1.0000,  0.0000,  1.0000]
pos 1 [ 0.8415,  0.5403,  0.0100,  1.0000]
pos 2 [ 0.9093, -0.4161,  0.0200,  0.9998]
```

**The decoder gets its own position counter, starting at 0.** Target position 0 is not source
position 0 — the two sequences are indexed independently and never share a position embedding.

### 3.3 X = E + PE

Decoder input during training is `<sos> loan granted` — the target **shifted right** by one:

```
             dim0     dim1     dim2     dim3
<sos>    [ 0.1000,  1.1000,  0.1000,  1.1000]
loan     [ 0.9415,  0.6403,  1.0100,  1.9000]
granted  [ 1.2093, -0.2161,  0.4200,  1.2998]
```

Shape `(3, 4)` = `(L_tgt, d_model)`.

**Why shifted right.** Position `t` of the input must predict token `t` of the target. Feed
`<sos> loan granted`, expect out `loan granted <eos>`. Every position is one step behind the
answer it is being graded on — that offset plus the causal mask is what makes the whole sequence
trainable in one pass.

```
input  :  <sos>    loan     granted
            ↓        ↓         ↓
predict:   loan   granted    <eos>
```

---

## 4. Weight setup

### 4.1 Masked self-attention — same four matrices as 06b

Reused deliberately, so any number you compare against 06b differs because of the *mask* and the
*sequence*, never because the weights moved. (A real decoder has its own; nothing here depends on
them being shared.)

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

### 4.2 Cross-attention — its own four matrices

```
Wq_c                            Wk_c
[[1.0, 0.1, 0.0, 0.0],          [[1.1, 0.0, 0.0, 0.0],
 [0.1, 1.0, 0.0, 0.0],           [0.0, 1.1, 0.0, 0.0],
 [0.0, 0.0, 1.0, 0.1],           [0.0, 0.0, 1.1, 0.0],
 [0.0, 0.0, 0.1, 1.0]]           [0.0, 0.0, 0.0, 1.1]]

Wv_c                            Wo_c
[[0.8, 0.2, 0.0, 0.0],          [[1.0, 0.1, 0.0, 0.1],
 [0.2, 0.8, 0.0, 0.0],           [0.1, 1.0, 0.1, 0.0],
 [0.0, 0.0, 0.8, 0.2],           [0.0, 0.1, 1.0, 0.1],
 [0.0, 0.0, 0.2, 0.8]]           [0.1, 0.0, 0.1, 1.0]]
```

**`Wq_c`, `Wk_c`, `Wv_c` are a completely separate set of parameters from `Wq`, `Wk`, `Wv`.** A
decoder layer holds **eight** projection matrices, not four. That is the parameter cost of
cross-attention and the reason an encoder-decoder layer is bigger than a decoder-only layer of the
same width.

### 4.3 FFN and LM head

```
W1  (4 × 8)                                                     W2  (8 × 4)
[[ 0.30, -0.20,  0.10,  0.40, -0.30,  0.20,  0.10, -0.10],      [[ 0.20, -0.10,  0.30,  0.10],
 [ 0.10,  0.30, -0.20,  0.10,  0.20, -0.30,  0.40,  0.20],       [-0.10,  0.30,  0.10,  0.20],
 [-0.20,  0.10,  0.40, -0.30,  0.10,  0.20, -0.10,  0.30],       [ 0.30,  0.20, -0.10,  0.10],
 [ 0.20,  0.40, -0.10,  0.20, -0.20,  0.10,  0.30, -0.20]]       [ 0.10,  0.10,  0.20,  0.30],
                                                                 [-0.20,  0.30,  0.10, -0.10],
W_vocab  (4 × 5)   columns = <sos> loan granted bank <eos>       [ 0.30, -0.20,  0.20,  0.10],
[[ 0.5, -0.2,  0.3,  0.1, -0.1],                                 [ 0.10,  0.20,  0.30, -0.20],
 [-0.3,  0.4,  0.2, -0.2,  0.1],                                 [ 0.20,  0.10, -0.20,  0.30]]
 [ 0.2,  0.6, -0.3,  0.1,  0.2],
 [ 0.1,  0.3,  0.5, -0.1,  0.4]]
```

---

## 5. Masked self-attention

### 5.1 Q, K, V — from the decoder, all three

```
Q = X @ Wq                              K = X @ Wk
[[ 0.1200,  1.3200,  0.1200,  1.3200],  [[ 0.3200,  1.1200,  0.3200,  1.1200],
 [ 1.1298,  0.7684,  1.2120,  2.2799],   [ 1.0695,  0.8286,  1.3900,  2.1019],
 [ 1.4512, -0.2594,  0.5040,  1.5598]]   [ 1.1661,  0.0257,  0.6800,  1.3838]]

V = X @ Wv
[[ 0.2000,  1.0000,  0.2000,  1.0000],
 [ 0.9114,  0.6704,  1.0990,  1.8110],
 [ 1.0668, -0.0736,  0.5080,  1.2118]]
```

Then the same reshape as 06b §6: `(3, 4) → (2 heads, 3 tokens, 2 dims)`. Columns 0-1 to head 0,
columns 2-3 to head 1. Still a **reshape, not a projection**.

### 5.2 Raw scores — the future is computed, then thrown away

```
scores = Q₀ @ K₀ᵀ / √2                    (head 0, before masking)

            <sos>    loan  granted
<sos>    [ 1.0725,  0.8641,  0.1229]     <- 0.8641 and 0.1229 are ILLEGAL
loan     [ 0.8641,  1.3046,  0.9455]     <- 0.9455 is ILLEGAL
granted  [ 0.1229,  0.9455,  1.1918]     <- all three legal
```

Note that the full `3×3` matrix **is computed**. Masking does not save the matmul — it discards
part of the result. (Block-sparse kernels that genuinely skip the work exist, and that is board 12.)

### 5.3 Set the future to −∞

```
mask[i, j] = -inf  where  j > i
```

```
masked scores, head 0                    masked scores, head 1
[[ 1.0725,    -inf,    -inf],            [[ 1.0725,    -inf,    -inf],
 [ 0.8641,  1.3046,    -inf],             [ 2.0799,  4.5799,    -inf],
 [ 0.1229,  0.9455,  1.1918]]             [ 1.3493,  2.8136,  1.7685]]
```

### 5.4 Softmax

`exp(−∞) = 0` exactly, so the masked entries contribute nothing to the numerator **and nothing to
the denominator**. Rows still sum to 1.

```
A₀ = softmax(masked, dim=-1)              A₁ = softmax(masked, dim=-1)

            <sos>    loan  granted                  <sos>    loan  granted
<sos>    [ 1.0000,  0.0000,  0.0000]      <sos>   [ 1.0000,  0.0000,  0.0000]
loan     [ 0.3916,  0.6084,  0.0000]      loan    [ 0.0759,  0.9241,  0.0000]
granted  [ 0.1616,  0.3678,  0.4706]      granted [ 0.1461,  0.6318,  0.2222]

O₀ = A₀ @ V₀                              O₁ = A₁ @ V₁
[[ 0.2000,  1.0000],                      [[ 0.2000,  1.0000],
 [ 0.6328,  0.7995],                       [ 1.0308,  1.7494],
 [ 0.8695,  0.3736]]                       [ 0.8364,  1.5594]]
```

**The triangle is the shape of the whole idea.** Row `t` has exactly `t+1` non-zero entries. Row 0
has one, and it is `1.0000` — position 0 has nowhere to look but itself, so `O₀` row 0 is `V₀`
row 0 unchanged (`[0.2000, 1.0000]` in both). Attention at the first position is the identity.

Head 0 and head 1 disagree about position `loan`: head 0 splits `0.3916 / 0.6084`, head 1 collapses
onto itself at `0.9241`. Two heads, two masks applied identically, two different answers — the 06b
story survives the mask intact.

### 5.5 Concat + W_o

```
concat = [O₀ ‖ O₁]                       MHA = concat @ W_o
[[0.2000, 1.0000, 0.2000, 1.0000],       [[0.3000, 1.0200, 0.3000, 1.0200],
 [0.6328, 0.7995, 1.0308, 1.7494],        [0.7525, 0.9578, 1.1659, 1.7575],
 [0.8695, 0.3736, 0.8364, 1.5594]]        [0.9036, 0.5791, 0.9956, 1.5244]]
```

---

## 6. Why `−inf` and not "zero it afterwards"

This is the question that gets asked, and the honest answer has three parts — one of which is
usually stated wrongly.

Take head 0, the same raw scores as §5.2:

```
(i)   softmax, NO mask                    (ii)  -inf BEFORE softmax   ← correct
[[0.4548, 0.3692, 0.1760],                [[1.0000, 0.0000, 0.0000],
 [0.2749, 0.4270, 0.2982],                 [0.3916, 0.6084, 0.0000],
 [0.1616, 0.3678, 0.4706]]                 [0.1616, 0.3678, 0.4706]]
row sums [1.0, 1.0, 1.0]                  row sums [1.0, 1.0, 1.0]

(iii) softmax, THEN zero the future       (iv)  softmax, zero, then RE-NORMALISE
[[0.4548, 0.0000, 0.0000],                [[1.0000, 0.0000, 0.0000],
 [0.2749, 0.4270, 0.0000],                 [0.3916, 0.6084, 0.0000],
 [0.1616, 0.3678, 0.4706]]                 [0.1616, 0.3678, 0.4706]]
row sums [0.4548, 0.7018, 1.0]  ← BROKEN  row sums [1.0, 1.0, 1.0]
```

### 6.1 (iii) is the actual bug — and it is silent

The rows no longer sum to 1, so `A @ V` is no longer a weighted **average**, it is a shrunken one:

```
O with (ii) correct       O with (iii) bug        ‖bug‖ / ‖correct‖
[[0.2000, 1.0000],        [[0.0910, 0.4548],       0.4548
 [0.6328, 0.7995],         [0.4441, 0.5611],       0.7018
 [0.8695, 0.3736]]         [0.8695, 0.3736]]       1.0000
```

The shrink factor is exactly the row sum, and it is **position-dependent**: position 0 loses 55% of
its magnitude, position 1 loses 30%, the last position is untouched. Residual + LayerNorm then
partially hide it. The failure signature is a model that trains but is systematically worse at the
*start* of every sequence — which nobody debugs as a masking bug.

### 6.2 (iv) is mathematically identical to (ii) — say so

`max|(iv) − (ii)| = 0.0`, exactly. Renormalising after zeroing **is** `−inf` masking, in exact
arithmetic. Do not claim in an interview that it gives different attention weights. It does not.

### 6.3 The real reason: (iv) loses precision, and then dies

`−inf` masking takes its softmax max over the *legal* entries only. Zero-after takes it over the
whole row — **including the future logit it is about to discard.** When a future logit is large,
every legal entry becomes `exp(small − large)` and drains toward zero. In float32:

```
future logit |   -inf BEFORE softmax           |  softmax -> zero -> renormalise
        2.0  |  [0.268941, 0.731059, 0.0]      |  [0.268941, 0.731059, 0.0]
       60.0  |  [0.268941, 0.731059, 0.0]      |  [0.268941, 0.731059, 0.0]
      100.0  |  [0.268941, 0.731059, 0.0]      |  [0.268657, 0.731343, 0.0]   <- wrong at 4 d.p.
      130.0  |  [0.268941, 0.731059, 0.0]      |  [nan, nan, nan]             <- dead
```

At a future logit of 100, `exp(1−100) = 1.01e-43` — float32 subnormal territory (min normal
`1.18e-38`), so the mantissa is already being eaten. At 130 both kept terms flush to exactly `0.0`,
the row sum is `0.0`, and the renormalisation is `0/0`.

So the ranking is: **(iii) silently wrong · (iv) correct until it is catastrophically not ·
(ii) always right, and it is one fused operation the kernel already supports.**

### 6.4 What the mask actually buys — an honest measurement

Run the same block with `causal=False` and compare:

```
                     row 0     row 1     row 2
max |masked − unmasked| :   0.6421    0.1633    0.0000
loss, masked   = 1.536187
loss, unmasked = 1.555584
```

Two things worth saying out loud:

- **Row 2 is bit-identical.** The last position has no future, so the mask never touches it. Same
  reason the (iii) bug spared row 2. In general the mask alters rows `0 … L−2` and leaves `L−1`
  alone — which is why a masking bug is invisible if you only inspect the final token.
- **The unmasked loss here is *higher*, not lower.** With untrained weights, leaking the future is
  just noise. The mask is not something you can motivate by watching the loss of a random model —
  its value appears only under **training**, where an unmasked model learns the trivial solution
  "copy position `t+1` of the input", scores near-zero training loss, and then generates garbage at
  inference because position `t+1` does not exist yet. Claiming the toy demonstrates this is the
  trap; the toy demonstrates the *mechanism*, and training demonstrates the *need*.

---

## 7. Add & Norm 1

```
R1 = X + MHA
[[0.4000, 2.1200, 0.4000, 2.1200],
 [1.6940, 1.5981, 2.1759, 3.6575],
 [2.1129, 0.3629, 1.4156, 2.8242]]

per-row mean: [1.2600, 2.2814, 1.6789]
per-row var : [0.7396, 0.6792, 0.8253]

h1 = LayerNorm(R1)
[[-1.0000,  1.0000, -1.0000,  1.0000],
 [-0.7127, -0.8291, -0.1279,  1.6698],
 [ 0.4777, -1.4486, -0.2898,  1.2607]]
```

Row 0 normalises to exactly `[−1, 1, −1, 1]` — it only ever contained two distinct values, because
position 0 attended solely to itself and `X` row 0 was `[0.1, 1.1, 0.1, 1.1]`. A coincidence of the
toy, but a useful sanity anchor: if row 0 of your decoder is not a pure function of token 0 alone,
your mask is wrong.

---

## 8. Cross-attention

**This is the block that does not exist in an encoder.** One line summarises it:

```
Q comes from the DECODER.   K and V come from the ENCODER.
```

### 8.1 The projections

```
Q_c = h1 @ Wq_c                          <- 3 rows, from the decoder
[[-0.9000,  0.9000, -0.9000,  0.9000],
 [-0.7956, -0.9004,  0.0391,  1.6570],
 [ 0.3328, -1.4008, -0.1638,  1.2317]]

K_c = MEMORY @ Wk_c                      <- 4 rows, from the ENCODER
[[-0.0828,  1.3471, -1.6855,  0.4212],
 [ 0.3272, -0.2558, -1.5630,  1.4915],
 [ 0.5615, -1.5298, -0.4421,  1.4103],
 [-0.4203, -1.5270,  0.5072,  1.4402]]

V_c = MEMORY @ Wv_c                      <- 4 rows, from the ENCODER
[[ 0.1847,  0.9646, -1.1493, -0.0001],
 [ 0.1915, -0.1265, -0.8655,  0.8005],
 [ 0.1303, -1.0105, -0.0651,  0.9453],
 [-0.5833, -1.1870,  0.6307,  1.1397]]
```

**Q has 3 rows and K, V have 4.** The shapes no longer match, and nothing requires them to — a
translation from a 40-token sentence into a 12-token one runs a `12 × 40` attention matrix. The
only constraint is that `d_head` agrees on both sides, because that is what the dot product
contracts over.

### 8.2 Scores and weights — a 3×4 rectangle, no mask

```
cross scores, head 0 = Q_c₀ @ K_c₀ᵀ / √2

              bank  approved     the    loan
<sos>    [  0.9100, -0.3710, -1.3309, -0.7043]
loan     [ -0.8110, -0.0213,  0.6580,  1.2087]
granted  [ -1.3538,  0.3303,  1.6474,  1.4136]

A_cross₀ = softmax(scores, dim=-1)              rows sum over SOURCE positions

              bank  approved     the    loan
<sos>    [  0.6316,  0.1754,  0.0672,  0.1257]
loan     [  0.0663,  0.1460,  0.2881,  0.4996]
granted  [  0.0236,  0.1270,  0.4741,  0.3753]

O_cross₀ = A_cross₀ @ V_c₀       (3, 2)
[[ 0.0857,  0.3700],
 [-0.2137, -0.8386],
 [-0.1285, -0.9179]]
```

```
cross scores, head 1                            A_cross₁

              bank  approved     the    loan                bank  approved     the    loan
<sos>    [ 1.3407,  1.9438,  1.1789,  0.5938]   <sos>   [ 0.2408,  0.4402,  0.2048,  0.1141]
loan     [ 0.4469,  1.7044,  1.6402,  1.7015]   loan    [ 0.0883,  0.3106,  0.2913,  0.3097]
granted  [ 0.5620,  1.4800,  1.2795,  1.1957]   granted [ 0.1344,  0.3367,  0.2755,  0.2534]

O_cross₁ = A_cross₁ @ V_c₁
[[-0.5992,  0.6761],
 [-0.1940,  0.8770],
 [-0.3041,  0.8187]]
```

**There is no mask here, and there must not be.** Every target position may look at every source
position — the source is fully observed before generation starts. The causal constraint is about
*the output being generated*, and the source is not being generated.

### 8.3 Reading one number honestly

`<sos>` puts `0.6316` on source `bank` in head 0. That number is mechanical, and you can check it:
head 0 reads dims 0-1, where the memory rows are `bank[−0.083, 1.347]`, `approved[0.327, −0.256]`,
`the[0.562, −1.530]`, `loan[−0.420, −1.527]`. The query is `[−0.9, 0.9]`, which weights dim1
positively — and **`bank` is the only source token with a positive dim1**. So:

```
bank :  (-0.9)(-0.0828) + (0.9)( 1.3471) = 1.2870  ->  /1.414 =  0.9100  ✓
the  :  (-0.9)( 0.5615) + (0.9)(-1.5298) = -1.8822 ->  /1.414 = -1.3309  ✓
```

That is a real explanation of a real number. What it is **not** is evidence that cross-attention
"aligns `granted` with `approved`" — look at row `granted`, which puts its mass on `the` (0.4741)
and `loan` (0.3753), and head 1 which is nearly flat (`0.1344 … 0.3367`, against uniform `0.25`).

> **Honest note — the same one 06b ends on.** Interpretable alignment is a property of **trained**
> cross-attention, not of the architecture. These weights were never trained; head 1's near-uniform
> rows are what untrained attention actually looks like. The architecture makes alignment
> *possible* and training makes it *happen*. Cross-attention maps in NMT papers are drawn from
> converged models, and a toy that claims to reproduce them is lying.

### 8.4 Concat + `Wo_c`

```
concat = [O_cross₀ ‖ O_cross₁]           XA = concat @ Wo_c
[[ 0.0857,  0.3700, -0.5992,  0.6761],   [[ 0.1903,  0.3186, -0.4946,  0.6247],
 [-0.2137, -0.8386, -0.1940,  0.8770],    [-0.2099, -0.8794, -0.1901,  0.8362],
 [-0.1285, -0.9179, -0.3041,  0.8187]]    [-0.1384, -0.9611, -0.3140,  0.7754]]
```

---

## 9. Add & Norm 2

```
R2 = h1 + XA
[[-0.8097,  1.3186, -1.4945,  1.6247],
 [-0.9226, -1.7085, -0.3181,  2.5060],
 [ 0.3393, -2.4097, -0.6038,  2.0362]]

per-row mean: [ 0.1598, -0.1108, -0.1595]
per-row var : [ 1.7914,  2.5256,  2.5826]

h2 = LayerNorm(R2)
[[-0.7243,  0.8658, -1.2360,  1.0945],
 [-0.5108, -1.0054, -0.1304,  1.6466],
 [ 0.3104, -1.4002, -0.2765,  1.3663]]
```

**The residual is `h1`, not `X`.** Each sub-layer adds to whatever arrived at it. This is what makes
the source information *optional*: if `XA` were zero the decoder would still work, just without the
source — which is exactly what a decoder-only model is.

---

## 10. FFN, output, LM head, loss

```
Z = h2 @ W1
[[ 0.3354,  0.7188, -0.8495,  0.3865,  0.0480, -0.5424,  0.7259, -0.3441],
 [ 0.1016,  0.4462, -0.0668,  0.0636, -0.3902,  0.3380,  0.0538, -0.5184],
 [ 0.2816,  0.0367,  0.0639,  0.3403, -0.6741,  0.5635, -0.0915, -0.6673]]

ReLU(Z)                                            <- 9 of 24 zeroed (38%)
[[0.3354, 0.7188, 0.0000, 0.3865, 0.0480, 0.0000, 0.7259, 0.0000],
 [0.1016, 0.4462, 0.0000, 0.0636, 0.0000, 0.3380, 0.0538, 0.0000],
 [0.2816, 0.0367, 0.0639, 0.3403, 0.0000, 0.5635, 0.0000, 0.0000]]

FFN out = ReLU(Z) @ W2
[[ 0.0968,  0.3803,  0.4724,  0.1433],
 [ 0.0889,  0.0732,  0.1716,  0.1415],
 [ 0.2749, -0.0830,  0.2625,  0.2003]]

R3 = h2 + FFN_out
[[-0.6275,  1.2462, -0.7636,  1.2378],
 [-0.4220, -0.9322,  0.0411,  1.7881],
 [ 0.5853, -1.4832, -0.0139,  1.5666]]

DECODER OUTPUT = LayerNorm(R3)
             dim0     dim1     dim2     dim3
pos 0    [-0.9286,  1.0031, -1.0689,  0.9945]
pos 1    [-0.5284, -1.0269, -0.0759,  1.6311]
pos 2    [ 0.3813, -1.4895, -0.1606,  1.2689]
```

Shape `(3, 4)` — same as the input. Stackable, exactly as in 06b.

### 10.1 LM head

```
logits = OUTPUT @ W_vocab                (3, 4) @ (4, 5) -> (3, 5)

           <sos>     loan  granted    bank    <eos>
pos 0  [-0.8795,  0.2439,  0.7399, -0.4998,  0.3772]
pos 1  [ 0.1918,  0.1387,  0.4744, -0.0182,  0.5874]
pos 2  [ 0.7323, -0.3878,  0.4991,  0.1931,  0.2883]

probs = softmax(logits, dim=-1)
           <sos>     loan  granted    bank    <eos>
pos 0  [ 0.0709,  0.2181,  0.3581,  0.1037,  0.2492]
pos 1  [ 0.1795,  0.1702,  0.2381,  0.1455,  0.2666]
pos 2  [ 0.2991,  0.0976,  0.2369,  0.1745,  0.1919]
```

The LM head is where `d_model` finally leaves — `(L, 4) → (L, 5)`. In a real model this is
`(L, 4096) → (L, 128256)` and it is the single largest matmul in the forward pass.

### 10.2 Loss

```
gold      : [loan, granted, <eos>]  = indices [1, 2, 4]
p(gold)   : [0.2181, 0.2381, 0.1919]

loss = -mean(log p(gold)) = -(log 0.2181 + log 0.2381 + log 0.1919) / 3 = 1.536187
```

For reference, a uniform model over 5 tokens scores `log 5 = 1.6094`. At `1.5362` this untrained
block is barely better than chance — as it should be.

```
greedy predictions: ['granted', '<eos>', '<sos>']
gold              : ['loan',    'granted', '<eos>']
```

Zero of three correct. §14 fixes one of them with a single gradient step.

---

## 11. Teacher forcing vs autoregressive

**Same weights. Same block. Same arithmetic. Two different things fed into it.**

### 11.1 Teacher forcing — training

All three positions go in at once. One forward pass, one `3×3` masked attention, three predictions,
one loss. That is §3–§10 above, and it is **parallel** — position 2 does not wait for position 1.

```
input      <sos>      loan       granted        <- always the GOLD prefix
             │          │           │
        ┌────┴──────────┴───────────┴────┐
        │  one decoder forward pass      │      3x3 mask, batched
        └────┬──────────┬───────────┬────┘
predict     loan     granted      <eos>
loss        ─────── one number: 1.536187 ───────
```

### 11.2 Autoregressive — inference

`L` separate forward passes. Pass `t` sees only tokens `0…t` and keeps only the **last row**.
Feeding the gold tokens back:

```
step 0:  input ['<sos>']
         last row = [-0.9286,  1.0031, -1.0689,  0.9945]
         probs    = [0.0709, 0.2181, 0.3581, 0.1037, 0.2492] -> greedy 'granted'
         == teacher-forcing row 0 ?   YES,  max diff 0.00e+00

step 1:  input ['<sos>', 'loan']
         last row = [-0.5284, -1.0269, -0.0759,  1.6311]
         probs    = [0.1795, 0.1702, 0.2381, 0.1455, 0.2666] -> greedy '<eos>'
         == teacher-forcing row 1 ?   YES,  max diff 0.00e+00

step 2:  input ['<sos>', 'loan', 'granted']
         last row = [ 0.3813, -1.4895, -0.1606,  1.2689]
         probs    = [0.2991, 0.0976, 0.2369, 0.1745, 0.1919] -> greedy '<sos>'
         == teacher-forcing row 2 ?   YES,  max diff 0.00e+00
```

**Bit-for-bit identical, all three rows.** This is the identity the whole design rests on:

> Row `t` of a teacher-forced pass over prefix `x₀…x_{L-1}` equals the last row of an
> autoregressive pass over `x₀…x_t`.

It holds **because of the causal mask** and nothing else. Row `t` never reads a column `> t`, so the
columns that exist in the batched pass but not the incremental one contribute exactly zero. Remove
the mask and this equality breaks — training and inference stop computing the same function, and
the model you trained is not the model you deploy.

### 11.3 The comparison, side by side

| | teacher forcing (train) | autoregressive (infer) |
|---|---|---|
| Input at step `t` | gold token `t−1` | **model's own output at `t−1`** |
| Forward passes | 1 | `L` |
| Parallel over positions | ✅ yes | ❌ no — strictly sequential |
| Self-attn matrix | `L × L`, masked | `1 × t` per step (with a cache) |
| Cross-attn matrix | `L_tgt × L_src` | `1 × L_src` per step |
| Needs the gold sequence | ✅ | ❌ |
| Stops when | end of sequence | `<eos>` emitted, or a length cap |
| Cost | one pass | `L` passes — the reason generation is slow |

---

## 12. Exposure bias

§11.2 fed the *gold* tokens back. Real inference has no gold. Feed the model its **own greedy
outputs**:

```
step 0:  input ['<sos>']                       -> emits 'granted'  (p = 0.3581)
         last row = [-0.9286,  1.0031, -1.0689,  0.9945]
         TF row 0 = [-0.9286,  1.0031, -1.0689,  0.9945]   max diff 0.0000

step 1:  input ['<sos>', 'granted']            -> emits 'granted'  (p = 0.4356)
         last row = [ 0.3874, -0.4943, -1.2882,  1.3950]
         TF row 1 = [-0.5284, -1.0269, -0.0759,  1.6311]   max diff 1.2123   <- DIVERGED

step 2:  input ['<sos>', 'granted', 'granted'] -> emits '<sos>'    (p = 0.3103)
         last row = [ 0.5928, -1.4405, -0.3553,  1.2030]
         TF row 2 = [ 0.3813, -1.4895, -0.1606,  1.2689]   max diff 0.2115

generated: ['granted', 'granted', '<sos>']     gold: ['loan', 'granted', '<eos>']
```

Step 0 matches — there is no history yet to be wrong about. From step 1 the decoder is conditioning
on `granted` where training always showed it `loan`, and the hidden state is off by `1.2123` in a
vector whose entries are order 1.

**This is exposure bias.** Training only ever shows the model gold prefixes; inference only ever
shows it its own. A single early mistake moves the model onto a conditioning path it was never
trained on, and errors compound — which is why this toy repeats `granted` rather than recovering.

Mitigations, and what they cost:

| Approach | Idea | Cost |
|---|---|---|
| Scheduled sampling | mix in the model's own tokens during training | breaks the parallel pass |
| Beam search | keep `k` hypotheses so one early slip is not fatal | `k×` inference compute |
| Sequence-level training (RL, MRT) | optimise the generated sequence, not per-token CE | slow, high variance |
| **Just scale** | a strong enough model rarely makes the early mistake | what actually won |

Modern LLMs still train with pure teacher forcing. Exposure bias was not solved — it was outrun.

---

## 13. Backward — three places the gradient goes

Loss: cross-entropy, `1.536187`. All gradients from `torch.autograd`; the torch forward matched the
numpy forward above to `3.3e-16` (`allclose`, atol 1e-6), so the arithmetic in §5–§10 and these
gradients describe the same computation.

### 13.1 Magnitudes

```
                        max |grad|      L2
  dL/dW_vocab            0.393285     0.876480    <- largest: closest to the loss
  dL/dW_o                0.187657     0.381206
  dL/dW_v                0.168646     0.334110
  dL/dWv_c               0.153049     0.248784
  dL/dWo_c               0.112641     0.221586
  dL/dW2                 0.098168     0.230178
  dL/dW1                 0.079774     0.221492
  dL/dWq_c               0.044620     0.077424
  dL/dWk_c               0.028885     0.053508
  dL/dW_k                0.011172     0.017244
  dL/dW_q                0.008190     0.015968    <- smallest, 21x below W_v
  dL/dMEMORY             0.066413     0.087106    <- into the ENCODER
```

**06b's finding reproduces exactly, in both attention blocks.** `W_v` and `W_o` sit on a linear path
to the loss (`A @ V @ W_o`); `W_q` and `W_k` reach it only *through* the softmax, whose Jacobian is
`A(1−A)`-shaped and collapses as rows sharpen. Self-attention row 0 is `[1, 0, 0]` — maximally
sharp — so it contributes nothing at all.

Note that `dL/dWq_c` (`0.0446`) is **5.4× larger** than `dL/dW_q` (`0.0082`). The cross-attention
rows are far flatter than the masked self-attention rows (compare `A_cross₁`, near-uniform, with
`A₀` row 0 at exactly 1.0), and flat softmax rows have the largest gradients. Sharp attention
learns slowly; that is the same fact seen from the other side.

### 13.2 The mask has no gradient path

```
dL/dS  at the PRE-softmax scores, head 0:

[[ 0.0000,  0.0000,  0.0000],
 [ 0.0012, -0.0012,  0.0000],
 [-0.0085, -0.0002,  0.0087]]
```

Upper triangle: **exactly zero**, not small — `0.0`. `exp(−∞) = 0` kills the forward contribution
and therefore the backward one. The masked positions cannot influence the loss, so they cannot
receive gradient, so they never learn to matter. The mask is enforced on both passes by one
operation.

**Row 0 is entirely zero too**, including the diagonal — and this is the subtler point.
`softmax([s]) = 1.0` for *any* `s`: a softmax over one element is a constant function, so
`∂A/∂s = 0`. Position 0's attention logit is unlearnable. Verify it:

```
softmax([-5.0]) = 1.0     softmax([0.0]) = 1.0
softmax([ 5.0]) = 1.0     softmax([100.0]) = 1.0
```

Each row also sums to ~0 (`0.0012 − 0.0012 = 0`), which is the softmax Jacobian's shift-invariance:
adding a constant to every logit in a row changes nothing, so the gradient has no component along
the all-ones direction.

### 13.3 Gradient flows back into the encoder

```
dL/dMEMORY — per source token
   bank      [ 0.06641, -0.02519, -0.01944,  0.00131]   L2 = 0.073661
   approved  [ 0.01358, -0.00058, -0.02780, -0.01381]   L2 = 0.033864
   the       [ 0.01640, -0.01279,  0.00466, -0.00990]   L2 = 0.023521
   loan      [ 0.00073, -0.01412,  0.01547,  0.00495]   L2 = 0.021481
```

Non-zero — so **cross-attention is the channel through which the encoder trains.** There is no
separate encoder loss in a seq2seq model. The encoder learns entirely from gradient arriving
through `K_c` and `V_c`, and the whole thing is one differentiable graph from decoder logits back
to source embeddings.

`bank` receives 3.4× the gradient of `loan` — and `bank` is precisely the source token that
cross-attention head 0 weighted most heavily (`0.6316` at position 0, §8.2). **Attention weight
forward becomes gradient share backward.** A source token nobody attends to gets almost no
learning signal, which is why attention *is* the routing mechanism in both directions.

### 13.4 `W_o` splits the heads — both blocks

```
dL/dW_o  (masked self-attn)              dL/dWo_c  (cross-attn)
[[ 0.0688, -0.0591,  0.0429, -0.0526],   [[ 0.0089,  0.0037, -0.0247,  0.0121],
 [ 0.1367, -0.1125, -0.0436,  0.0194],    [ 0.0120,  0.0215, -0.0982,  0.0647],
 [ 0.0484, -0.0789,  0.0951, -0.0647],    [-0.0899,  0.0516,  0.0409, -0.0027],
 [ 0.1877, -0.1795,  0.0652, -0.0733]]    [ 0.1126, -0.0885,  0.0052, -0.0293]]
   rows 0-1 -> head 0,  rows 2-3 -> head 1  (same routing in both)
```

Two independent copies of 06b §14.1. Each attention block has its own `W_o`, and each is the sole
meeting point of its own two heads.

---

## 14. Weight update — verify

```
SGD, lr = 0.5,  W <- W - lr * dL/dW   (all eleven matrices, plus the memory)

loss before = 1.536187
loss after  = 1.090838      DECREASED by 0.445348   ✓
```

```
p(gold) before : [0.2181, 0.2381, 0.1919]
p(gold) after  : [0.4511, 0.2903, 0.2894]        all three rose

greedy before  : ['granted', '<eos>',   '<sos>' ]   0 / 3
greedy after   : ['loan',    'granted', 'granted']  2 / 3
gold           : ['loan',    'granted', '<eos>'  ]
```

One step, at an absurd learning rate, on a five-token vocabulary — position 0 went from wrong to
right and position 1 with it. Position 2 still prefers `granted` over `<eos>`, though `p(<eos>)`
rose from `0.1919` to `0.2894`. The block learns.

---

## 15. Why the KV cache exists

§11.2 recomputed the entire prefix at every step. Count what is actually re-derived:

```
step | prefix len | K,V rows built, no cache | with cache | self-attn dot products
  1  |     1      |            1             |     1      |          1
  2  |     2      |            2             |     1      |          2
  3  |     3      |            3             |     1      |          3
                      total 6                   total 3
```

At `L` steps that is `O(L²)` recomputed K/V rows against `O(L)` with a cache. **Every one of those
recomputations is bit-identical to the previous step's** — token 0's key never changes, because
causal masking means token 0's representation cannot depend on anything that comes after it. Cache
it. This is board 12.

Cross-attention gets the stronger version of the same argument:

```
K_c row 0 at L_tgt=1: [-0.0828, 1.3471, -1.6855, 0.4212]
K_c row 0 at L_tgt=2: [-0.0828, 1.3471, -1.6855, 0.4212]
K_c row 0 at L_tgt=3: [-0.0828, 1.3471, -1.6855, 0.4212]

cross K,V rows built over 3 steps:   no cache = 12      with cache = 4
```

**The cross-attention K and V do not depend on the decoder at all.** They are `MEMORY @ Wk_c` and
`MEMORY @ Wv_c` — computed once, before the first token is generated, and reused unchanged for
every step of a 500-token output. Only `A_cross` changes shape, growing one row per step.

This is why encoder-decoder inference is cheaper than its parameter count suggests: the encoder runs
exactly once.

---

## 16. Decoder-only vs encoder-decoder

Delete the cross-attention sub-layer from §8 and you have GPT.

| | encoder-decoder (this file, T5, BART) | decoder-only (GPT, Llama, Mistral) |
|---|---|---|
| Attention blocks / layer | 2 | **1** |
| Projection matrices / layer | 8 | **4** |
| Masking | causal on self, none on cross | causal, everywhere |
| Source is | encoded separately into a memory | **prepended to the same sequence** |
| Source attends to itself | bidirectionally | causally — a real loss of context |
| Attention matrix | `L_tgt×L_tgt` + `L_tgt×L_src` | one `(L_src+L_tgt)²` |
| Encoder runs | once per input | n/a |
| Good at | translation, summarisation, doc→text | everything else, at scale |

The decoder-only trade is real and it is not free: prompt tokens are processed causally, so token 3
of your prompt cannot see token 40. Encoder-decoder keeps the source bidirectional — which is why
**Donut, LayoutLM-style document models, and Whisper are still encoder-decoder**: the input is an
image or an audio spectrogram where "causal" is meaningless, and there is no reason to cripple it.

> Board 10 (T5/BART) is exactly this block. The Donut decoder in Phase 10 is this block with a
> vision encoder supplying `MEMORY`. Nothing new to learn there — only the encoder changes.

### 16.1 Where the parameters go

Per layer, `d_ff = 4·d_model`:

```
decoder-only     self-attn 4d²  +  FFN 8d²                     = 12d²
encoder-decoder  self-attn 4d²  +  cross-attn 4d²  +  FFN 8d²  = 16d²
```

Cross-attention is **+33% parameters per decoder layer**. In exchange the encoder side runs once and
stays bidirectional.

---

## 17. Quick reference

```
DECODER LAYER

 1. X   = E + PE                                  (L_tgt, d_model)   target shifted right
 2. Q,K,V = X @ Wq, X @ Wk, X @ Wv                 all from the DECODER
 3. S   = Q K^T / sqrt(d_head)                     (n_heads, L_tgt, L_tgt)
 4. S[i,j] = -inf  where j > i                     BEFORE softmax, never after
 5. A   = softmax(S);  O = A @ V;  MHA = concat @ W_o
 6. h1  = LayerNorm(X + MHA)
 7. Qc  = h1 @ Wq_c                                Q from the DECODER
    Kc  = MEM @ Wk_c ;  Vc = MEM @ Wv_c            K, V from the ENCODER
 8. Sc  = Qc Kc^T / sqrt(d_head)                   (n_heads, L_tgt, L_src)  NO MASK
 9. Ac  = softmax(Sc);  XA = concat @ Wo_c
10. h2  = LayerNorm(h1 + XA)
11. F   = ReLU(h2 @ W1) @ W2
12. out = LayerNorm(h2 + F)                        (L_tgt, d_model)  same shape in/out
13. logits = out @ W_vocab                         (L_tgt, V)
```

**The seven things to be able to say cold:**

1. A decoder layer is an encoder layer **plus one cross-attention block**, with a mask on the first
   attention. Three sub-layers, eight projection matrices.
2. Masking is `−inf` **before** softmax. Zeroing after leaves rows summing to `< 1` and silently
   shrinks early positions. Renormalising after is mathematically identical but loses float32
   precision and NaNs on large logits.
3. Cross-attention: **Q from the decoder, K and V from the encoder.** Matrix is `L_tgt × L_src`,
   rectangular, and carries **no mask** — the source is fully observed.
4. Teacher forcing and autoregressive decoding compute the **same function**. Row `t` of the batched
   pass equals the last row of the incremental pass — verified here to `0.00e+00`. The causal mask
   is the only reason that holds.
5. They differ in what is fed back: gold tokens vs the model's own. That gap is **exposure bias**,
   and one wrong token at step 1 moved this decoder's hidden state by `1.21`.
6. Cross-attention `K`/`V` are computed **once from the memory** and reused for every generated
   token. Self-attention `K`/`V` grow by one row per step. That asymmetry is the KV cache.
7. The encoder has no loss of its own — it trains **entirely** through `dL/dMEMORY`, which arrives
   through cross-attention, proportional to how much the decoder attended to each source token.

---

## See also

- [06b_transformer_encoder_multihead.md](06b_transformer_encoder_multihead.md) — the encoder that produced this file's memory
- [06_transformer_end_to_end.md](06_transformer_end_to_end.md) — `d_model=2`, single head, the original walkthrough
- [07_decoding_strategies.md](07_decoding_strategies.md) — what to do with §10's probability row: greedy, beam, top-k, nucleus
- [../../5.transformers/02_models/03_encoder_decoder.md](../../5.transformers/02_models/03_encoder_decoder.md) — the family this block belongs to: T5, BART, Donut
- [../../5.transformers/02_models/02_gpt_family.md](../../5.transformers/02_models/02_gpt_family.md) — this block with cross-attention deleted
- [../../5.transformers/02_models/08_modern_llm_architecture.md](../../5.transformers/02_models/08_modern_llm_architecture.md) — pre-LN, RMSNorm, SwiGLU, RoPE, GQA

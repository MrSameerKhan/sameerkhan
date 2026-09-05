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

Decoder input during training is `<sos> loan granted` — the target **shifted right** by one. Plain elementwise addition, every dimension:

```
<sos>   : [0.1,0.1,0.1,0.1] + [0.0000,1.0000,0.0000,1.0000]
        = [0.1+0.0000, 0.1+1.0000, 0.1+0.0000, 0.1+1.0000] = [0.1000, 1.1000, 0.1000, 1.1000]

loan    : [0.1,0.1,1.0,0.9] + [0.8415,0.5403,0.0100,1.0000]
        = [0.1+0.8415, 0.1+0.5403, 1.0+0.0100, 0.9+1.0000] = [0.9415, 0.6403, 1.0100, 1.9000]

granted : [0.3,0.2,0.4,0.3] + [0.9093,-0.4161,0.0200,0.9998]
        = [0.3+0.9093, 0.2-0.4161, 0.4+0.0200, 0.3+0.9998] = [1.2093, -0.2161, 0.4200, 1.2998]
```

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

**A structural shortcut worth noticing before the arithmetic starts.** Every one of these six
projection matrices (`Wq, Wk, Wv, Wq_c, Wk_c, Wv_c`) is either diagonal (`Wq`, `Wk_c`) or
**block-diagonal in 2×2 blocks** (`Wk, Wv, Wq_c, Wv_c` — dims 0-1 only ever mix with each other,
dims 2-3 only ever mix with each other, and the head split later is dims 0-1 → head 0, dims 2-3 →
head 1). That means every "4-term dot product" below is really at most a **2-term** sum — the other
two terms are multiplied by exactly `0.0` and can be skipped. `W_o`, `Wo_c`, `W1`, `W2`, `W_vocab`
have no such structure — those are genuine full-width sums, shown in full when we reach them.

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

**Q = X @ Wq.** `Wq = diag(1.2, 1.2, 1.2, 1.2)` — every off-diagonal entry is `0.0`, so this
"matrix multiply" is really just **multiply every entry by 1.2**, nothing else:
```
X[<sos>]  = [0.1000, 1.1000, 0.1000, 1.1000] × 1.2 = [0.1200, 1.3200, 0.1200, 1.3200]
X[loan]   = [0.9415, 0.6403, 1.0100, 1.9000] × 1.2 = [1.1298, 0.7684, 1.2120, 2.2800]
X[granted]= [1.2093,-0.2161, 0.4200, 1.2998] × 1.2 = [1.4512,-0.2593, 0.5040, 1.5598]
```
```
Q = X @ Wq
[[ 0.1200,  1.3200,  0.1200,  1.3200],
 [ 1.1298,  0.7684,  1.2120,  2.2799],
 [ 1.4512, -0.2594,  0.5040,  1.5598]]
```

**K = X @ Wk.** `Wk` is block-diagonal — column 0 only reads `Wk[0,0]=1.0` and `Wk[1,0]=0.2`
(both from dims 0-1), column 2 only reads `Wk[2,2]=1.0` and `Wk[3,2]=0.2` (both from dims 2-3).
Every cell below is a genuine **2-term** sum, not 4:
```
K[<sos>,0] = 0.1000×1.0 + 1.1000×0.2 = 0.1000+0.2200 = 0.3200
K[<sos>,1] = 0.1000×0.2 + 1.1000×1.0 = 0.0200+1.1000 = 1.1200
K[<sos>,2] = 0.1000×1.0 + 1.1000×0.2 = 0.1000+0.2200 = 0.3200
K[<sos>,3] = 0.1000×0.2 + 1.1000×1.0 = 0.0200+1.1000 = 1.1200

K[loan,0]  = 0.9415×1.0 + 0.6403×0.2 = 0.9415+0.1281 = 1.0696
K[loan,1]  = 0.9415×0.2 + 0.6403×1.0 = 0.1883+0.6403 = 0.8286
K[loan,2]  = 1.0100×1.0 + 1.9000×0.2 = 1.0100+0.3800 = 1.3900
K[loan,3]  = 1.0100×0.2 + 1.9000×1.0 = 0.2020+1.9000 = 2.1020

K[granted,0] = 1.2093×1.0 + (-0.2161)×0.2 = 1.2093-0.0432 = 1.1661
K[granted,1] = 1.2093×0.2 + (-0.2161)×1.0 = 0.2419-0.2161 = 0.0258
K[granted,2] = 0.4200×1.0 + 1.2998×0.2 = 0.4200+0.2600 = 0.6800
K[granted,3] = 0.4200×0.2 + 1.2998×1.0 = 0.0840+1.2998 = 1.3838
```
```
K = X @ Wk
[[ 0.3200,  1.1200,  0.3200,  1.1200],
 [ 1.0695,  0.8286,  1.3900,  2.1019],
 [ 1.1661,  0.0257,  0.6800,  1.3838]]
```

**V = X @ Wv.** Same block-diagonal shape as `Wk`, coefficients `0.9`/`0.1` instead of `1.0`/`0.2`:
```
V[<sos>,0] = 0.1000×0.9 + 1.1000×0.1 = 0.0900+0.1100 = 0.2000
V[<sos>,1] = 0.1000×0.1 + 1.1000×0.9 = 0.0100+0.9900 = 1.0000
V[<sos>,2] = 0.1000×0.9 + 1.1000×0.1 = 0.0900+0.1100 = 0.2000
V[<sos>,3] = 0.1000×0.1 + 1.1000×0.9 = 0.0100+0.9900 = 1.0000

V[loan,0]  = 0.9415×0.9 + 0.6403×0.1 = 0.8474+0.0640 = 0.9114
V[loan,1]  = 0.9415×0.1 + 0.6403×0.9 = 0.0942+0.5763 = 0.6704
V[loan,2]  = 1.0100×0.9 + 1.9000×0.1 = 0.9090+0.1900 = 1.0990
V[loan,3]  = 1.0100×0.1 + 1.9000×0.9 = 0.1010+1.7100 = 1.8110

V[granted,0] = 1.2093×0.9 + (-0.2161)×0.1 = 1.0884-0.0216 = 1.0668
V[granted,1] = 1.2093×0.1 + (-0.2161)×0.9 = 0.1209-0.1945 = -0.0736
V[granted,2] = 0.4200×0.9 + 1.2998×0.1 = 0.3780+0.1300 = 0.5080
V[granted,3] = 0.4200×0.1 + 1.2998×0.9 = 0.0420+1.1698 = 1.2118
```
```
V = X @ Wv
[[ 0.2000,  1.0000,  0.2000,  1.0000],
 [ 0.9114,  0.6704,  1.0990,  1.8110],
 [ 1.0668, -0.0736,  0.5080,  1.2118]]
```

Then the same reshape as 06b §6: `(3, 4) → (2 heads, 3 tokens, 2 dims)`. Columns 0-1 to head 0,
columns 2-3 to head 1. Still a **reshape, not a projection** — no arithmetic happens here, `Q`, `K`,
`V` just get sliced in half.

### 5.2 Raw scores — the future is computed, then thrown away

Head 0 reads columns 0-1 of Q, K: `Q₀ = [[0.12,1.32],[1.1298,0.7684],[1.4512,-0.2594]]`,
`K₀` identical shape. Every score is a 2-term dot product divided by `√d_head = √2 = 1.41421`:

```
S₀[<sos>,<sos>]   = (0.1200×0.3200 + 1.3200×1.1200) / 1.41421 = (0.0384+1.4784)/1.41421 = 1.0725
S₀[<sos>,loan]    = (0.1200×1.0695 + 1.3200×0.8286) / 1.41421 = (0.1283+1.0938)/1.41421 = 0.8641
S₀[<sos>,granted] = (0.1200×1.1661 + 1.3200×0.0257) / 1.41421 = (0.1399+0.0339)/1.41421 = 0.1229

S₀[loan,<sos>]    = (1.1298×0.3200 + 0.7684×1.1200) / 1.41421 = (0.3615+0.8606)/1.41421 = 0.8641
S₀[loan,loan]     = (1.1298×1.0695 + 0.7684×0.8286) / 1.41421 = (1.2083+0.6367)/1.41421 = 1.3046
S₀[loan,granted]  = (1.1298×1.1661 + 0.7684×0.0257) / 1.41421 = (1.3175+0.0197)/1.41421 = 0.9455

S₀[granted,<sos>]    = (1.4512×0.3200 + (-0.2594)×1.1200) / 1.41421 = (0.4644-0.2905)/1.41421 = 0.1229
S₀[granted,loan]     = (1.4512×1.0695 + (-0.2594)×0.8286) / 1.41421 = (1.5522-0.2150)/1.41421 = 0.9455
S₀[granted,granted]  = (1.4512×1.1661 + (-0.2594)×0.0257) / 1.41421 = (1.6926-0.0067)/1.41421 = 1.1918
```

```
scores = Q₀ @ K₀ᵀ / √2                    (head 0, before masking)

            <sos>    loan  granted
<sos>    [ 1.0725,  0.8641,  0.1229]     <- 0.8641 and 0.1229 are ILLEGAL
loan     [ 0.8641,  1.3046,  0.9455]     <- 0.9455 is ILLEGAL
granted  [ 0.1229,  0.9455,  1.1918]     <- all three legal
```

Notice `S₀[<sos>,loan] = S₀[loan,<sos>] = 0.8641` — the raw score matrix is **symmetric before
masking**, because it's literally `Q₀K₀ᵀ` computed from the same `Q=K` source (self-attention: every
position projects itself into both a query and a key). The mask is what breaks that symmetry — after
masking, position 0 can't see position 1's score even though position 1 can see position 0's.

Head 1 (columns 2-3) follows the identical recipe with `Q₁, K₁` instead — worth doing once yourself
to get `S₁[<sos>,<sos>]=1.0725` (same as head 0 — `<sos>`'s dims 2-3 happen to equal its dims 0-1
after PE) and `S₁[loan,loan]=4.5799` (much sharper — head 1 magnifies `loan`'s self-similarity far
more than head 0 does, which is exactly why the two heads disagree so strongly in §5.4).

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

**Row `<sos>`, both heads:** only one legal entry, so `softmax([1.0725]) = [1.0000]` no matter what
the number is — softmax over a single value is always 1 (proven properly in §13.2).

**Row `loan`, head 0** (legal entries `[0.8641, 1.3046]`):
```
exp(0.8641) = 2.3729
exp(1.3046) = 3.6860
sum = 2.3729 + 3.6860 = 6.0589

A₀[loan,<sos>] = 2.3729 / 6.0589 = 0.3917
A₀[loan,loan]  = 3.6860 / 6.0589 = 0.6084
```

**Row `granted`, head 0** (all three legal: `[0.1229, 0.9455, 1.1918]`):
```
exp(0.1229) = 1.1308
exp(0.9455) = 2.5743
exp(1.1918) = 3.2929
sum = 1.1308 + 2.5743 + 3.2929 = 6.9980

A₀[granted,<sos>]    = 1.1308 / 6.9980 = 0.1616
A₀[granted,loan]     = 2.5743 / 6.9980 = 0.3679
A₀[granted,granted]  = 3.2929 / 6.9980 = 0.4706
```

**Row `loan`, head 1** (legal entries `[2.0799, 4.5799]` — much further apart than head 0's, which
is exactly why this row ends up so much sharper):
```
exp(2.0799) = 8.0022
exp(4.5799) = 97.5099
sum = 8.0022 + 97.5099 = 105.5121

A₁[loan,<sos>] = 8.0022 / 105.5121 = 0.0759
A₁[loan,loan]  = 97.5099 / 105.5121 = 0.9241
```

**Row `granted`, head 1** (`[1.3493, 2.8136, 1.7685]`):
```
exp(1.3493) = 3.8546
exp(2.8136) = 16.6699
exp(1.7685) = 5.8622
sum = 3.8546 + 16.6699 + 5.8622 = 26.3867

A₁[granted,<sos>]    = 3.8546 / 26.3867 = 0.1461
A₁[granted,loan]     = 16.6699 / 26.3867 = 0.6318
A₁[granted,granted]  = 5.8622 / 26.3867 = 0.2222
```

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

**Computing `O₀` row by row** — each row is a weighted sum over `V₀ = [[0.20,1.00],[0.9114,0.6704],
[1.0668,-0.0736]]`, using that row's attention weights (zero-weight terms from masked positions
just don't appear — there's nothing to multiply by 0 and add):
```
O₀[<sos>]   = 1.0000×[0.2000,1.0000]                                            = [0.2000, 1.0000]

O₀[loan]    = 0.3917×[0.2000,1.0000] + 0.6084×[0.9114,0.6704]
  dim0: 0.3917×0.2000 + 0.6084×0.9114 = 0.0783+0.5546 = 0.6329
  dim1: 0.3917×1.0000 + 0.6084×0.6704 = 0.3917+0.4079 = 0.7996
            = [0.6329, 0.7996]

O₀[granted] = 0.1616×[0.2000,1.0000] + 0.3679×[0.9114,0.6704] + 0.4706×[1.0668,-0.0736]
  dim0: 0.1616×0.2000 + 0.3679×0.9114 + 0.4706×1.0668 = 0.0323+0.3353+0.5020 = 0.8696
  dim1: 0.1616×1.0000 + 0.3679×0.6704 + 0.4706×(-0.0736) = 0.1616+0.2466-0.0346 = 0.3736
            = [0.8696, 0.3736]
```

`O₁` follows the identical recipe against `V₁ = [[0.20,1.00],[1.0990,1.8110],[0.5080,1.2118]]` and
head 1's own `A₁` weights — worth reproducing yourself as a check: `O₁[loan]` should land on
`[1.0308, 1.7494]`.

**The triangle is the shape of the whole idea.** Row `t` has exactly `t+1` non-zero entries. Row 0
has one, and it is `1.0000` — position 0 has nowhere to look but itself, so `O₀` row 0 is `V₀`
row 0 unchanged (`[0.2000, 1.0000]` in both). Attention at the first position is the identity.

Head 0 and head 1 disagree about position `loan`: head 0 splits `0.3916 / 0.6084`, head 1 collapses
onto itself at `0.9241`. Two heads, two masks applied identically, two different answers — the 06b
story survives the mask intact.

### 5.5 Concat + W_o

`W_o` is dense (no zero structure like the projections above) — every cell here is a genuine 4-term
sum. `W_o = [[0.9,0.1,0.1,0.0],[0.1,0.9,0.0,0.1],[0.1,0.0,0.9,0.1],[0.0,0.1,0.1,0.9]]`:

```
concat[<sos>] = [0.2000, 1.0000, 0.2000, 1.0000]
  MHA[<sos>,0] = 0.2000×0.9 + 1.0000×0.1 + 0.2000×0.1 + 1.0000×0.0 = 0.18+0.10+0.02+0.00 = 0.3000
  MHA[<sos>,1] = 0.2000×0.1 + 1.0000×0.9 + 0.2000×0.0 + 1.0000×0.1 = 0.02+0.90+0.00+0.10 = 1.0200
  MHA[<sos>,2] = 0.2000×0.1 + 1.0000×0.0 + 0.2000×0.9 + 1.0000×0.1 = 0.02+0.00+0.18+0.10 = 0.3000
  MHA[<sos>,3] = 0.2000×0.0 + 1.0000×0.1 + 0.2000×0.1 + 1.0000×0.9 = 0.00+0.10+0.02+0.90 = 1.0200

concat[loan] = [0.6328, 0.7996, 1.0308, 1.7494]
  MHA[loan,0] = 0.6328×0.9 + 0.7996×0.1 + 1.0308×0.1 + 1.7494×0.0 = 0.5695+0.0800+0.1031+0.0000 = 0.7526
  MHA[loan,1] = 0.6328×0.1 + 0.7996×0.9 + 1.0308×0.0 + 1.7494×0.1 = 0.0633+0.7196+0.0000+0.1749 = 0.9578
  MHA[loan,2] = 0.6328×0.1 + 0.7996×0.0 + 1.0308×0.9 + 1.7494×0.1 = 0.0633+0.0000+0.9277+0.1749 = 1.1659
  MHA[loan,3] = 0.6328×0.0 + 0.7996×0.1 + 1.0308×0.1 + 1.7494×0.9 = 0.0000+0.0800+0.1031+1.5745 = 1.7576

concat[granted] = [0.8696, 0.3736, 0.8364, 1.5594]   (row 3 follows the same pattern as above)
  MHA[granted,0] = 0.8696×0.9+0.3736×0.1+0.8364×0.1+1.5594×0.0 = 0.7826+0.0374+0.0836+0.0000 = 0.9036
  MHA[granted,1] = 0.8696×0.1+0.3736×0.9+0.8364×0.0+1.5594×0.1 = 0.0870+0.3362+0.0000+0.1559 = 0.5791
  MHA[granted,2] = 0.8696×0.1+0.3736×0.0+0.8364×0.9+1.5594×0.1 = 0.0870+0.0000+0.7528+0.1559 = 0.9957
  MHA[granted,3] = 0.8696×0.0+0.3736×0.1+0.8364×0.1+1.5594×0.9 = 0.0000+0.0374+0.0836+1.4035 = 1.5245
```

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

**Residual — plain addition, `X + MHA`:**
```
R1[<sos>]    = [0.1000,1.1000,0.1000,1.1000] + [0.3000,1.0200,0.3000,1.0200] = [0.4000,2.1200,0.4000,2.1200]
R1[loan]     = [0.9415,0.6403,1.0100,1.9000] + [0.7526,0.9578,1.1659,1.7576] = [1.6941,1.5981,2.1759,3.6576]
R1[granted]  = [1.2093,-0.2161,0.4200,1.2998] + [0.9036,0.5791,0.9957,1.5245] = [2.1129,0.3630,1.4157,2.8243]
```

**LayerNorm, worked in full for every row.** Formula: `μ = mean(x)`, `σ² = mean((x−μ)²)`,
`LN(x) = (x−μ)/√(σ²+ε)` (`ε=1e-5`, negligible below).

```
Row <sos>: x=[0.4000,2.1200,0.4000,2.1200]
  μ = (0.4000+2.1200+0.4000+2.1200)/4 = 5.0400/4 = 1.2600
  (x-μ) = [-0.8600, 0.8600, -0.8600, 0.8600]
  σ² = [(-0.86)²+(0.86)²+(-0.86)²+(0.86)²]/4 = [0.7396×4]/4 = 0.7396
  σ = √0.7396 = 0.8600
  LN = [-0.8600/0.8600, 0.8600/0.8600, -0.8600/0.8600, 0.8600/0.8600] = [-1.0000, 1.0000, -1.0000, 1.0000]

Row loan: x=[1.6941,1.5981,2.1759,3.6576]
  μ = (1.6941+1.5981+2.1759+3.6576)/4 = 9.1257/4 = 2.2814
  (x-μ) = [-0.5873, -0.6833, -0.1055, 1.3762]
  σ² = [0.3450+0.4669+0.0111+1.8940]/4 = 2.7170/4 = 0.6792
  σ = √0.6792 = 0.8241
  LN = [-0.5873/0.8241, -0.6833/0.8241, -0.1055/0.8241, 1.3762/0.8241]
     = [-0.7126, -0.8291, -0.1280, 1.6698]

Row granted: x=[2.1129,0.3630,1.4157,2.8243]
  μ = (2.1129+0.3630+1.4157+2.8243)/4 = 6.7159/4 = 1.6790
  (x-μ) = [0.4339, -1.3160, -0.2633, 1.1453]
  σ² = [0.1883+1.7318+0.0693+1.3117]/4 = 3.3011/4 = 0.8253
  σ = √0.8253 = 0.9085
  LN = [0.4339/0.9085, -1.3160/0.9085, -0.2633/0.9085, 1.1453/0.9085]
     = [0.4776, -1.4485, -0.2898, 1.2607]
```

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

**Q_c = h1 @ Wq_c.** `Wq_c` is block-diagonal (`1.0`/`0.1`, same shape as `Wk` in §5.1) — 2-term
sums, using `h1` (not `X` — cross-attention's query comes from the *previous sublayer's output*):
```
h1[<sos>]=[-1.0,1.0,-1.0,1.0]:
  Qc[<sos>,0]=-1.0×1.0+1.0×0.1=-0.9   Qc[<sos>,1]=-1.0×0.1+1.0×1.0=0.9
  Qc[<sos>,2]=-1.0×1.0+1.0×0.1=-0.9   Qc[<sos>,3]=-1.0×0.1+1.0×1.0=0.9

h1[loan]=[-0.7126,-0.8291,-0.1280,1.6698]:
  Qc[loan,0]=-0.7126×1.0+(-0.8291)×0.1=-0.7126-0.0829=-0.7955
  Qc[loan,1]=-0.7126×0.1+(-0.8291)×1.0=-0.0713-0.8291=-0.9004
  Qc[loan,2]=-0.1280×1.0+1.6698×0.1=-0.1280+0.1670=0.0390
  Qc[loan,3]=-0.1280×0.1+1.6698×1.0=-0.0128+1.6698=1.6570

h1[granted]=[0.4776,-1.4485,-0.2898,1.2607]:
  Qc[granted,0]=0.4776×1.0+(-1.4485)×0.1=0.4776-0.1449=0.3327
  Qc[granted,1]=0.4776×0.1+(-1.4485)×1.0=0.0478-1.4485=-1.4007
  Qc[granted,2]=-0.2898×1.0+1.2607×0.1=-0.2898+0.1261=-0.1637
  Qc[granted,3]=-0.2898×0.1+1.2607×1.0=-0.0290+1.2607=1.2317
```

**K_c = MEMORY @ Wk_c.** `Wk_c = diag(1.1,1.1,1.1,1.1)` — trivial scale, every entry ×1.1:
```
bank     [-0.0753,1.2246,-1.5323,0.3829] × 1.1 = [-0.0828, 1.3471,-1.6855, 0.4212]
approved [ 0.2975,-0.2325,-1.4209,1.3559] × 1.1 = [ 0.3273,-0.2558,-1.5630, 1.4915]
the      [ 0.5105,-1.3907,-0.4019,1.2821] × 1.1 = [ 0.5616,-1.5298,-0.4421, 1.4103]
loan     [-0.3821,-1.3882, 0.4611,1.3093] × 1.1 = [-0.4203,-1.5270, 0.5072, 1.4402]
```

**V_c = MEMORY @ Wv_c.** `Wv_c` block-diagonal (`0.8`/`0.2`, same shape as `Wv` in §5.1), 2-term
sums:
```
bank:     0.1847=(-0.0753×0.8+1.2246×0.2)  0.9646=(-0.0753×0.2+1.2246×0.8)
         -1.1493=(-1.5323×0.8+0.3829×0.2) -0.0001=(-1.5323×0.2+0.3829×0.8)
approved: 0.1915=(0.2975×0.8-0.2325×0.2)  -0.1265=(0.2975×0.2-0.2325×0.8)
         -0.8655=(-1.4209×0.8+1.3559×0.2)  0.8005=(-1.4209×0.2+1.3559×0.8)
the:      0.1303=(0.5105×0.8-1.3907×0.2)  -1.0105=(0.5105×0.2-1.3907×0.8)
         -0.0651=(-0.4019×0.8+1.2821×0.2)  0.9453=(-0.4019×0.2+1.2821×0.8)
loan:    -0.5833=(-0.3821×0.8-1.3882×0.2) -1.1870=(-0.3821×0.2-1.3882×0.8)
          0.6308=(0.4611×0.8+1.3093×0.2)   1.1396=(0.4611×0.2+1.3093×0.8)
```

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

**All 12 cells of head 0's score matrix**, `Q_c₀ @ K_c₀ᵀ / √2` — `Q_c₀` and `K_c₀` are columns 0-1
of `Q_c`, `K_c` above:
```
Q_c₀: <sos>=[-0.9,0.9]  loan=[-0.7955,-0.9004]  granted=[0.3327,-1.4007]
K_c₀: bank=[-0.0828,1.3471]  approved=[0.3273,-0.2558]  the=[0.5616,-1.5298]  loan=[-0.4203,-1.5270]

S_c0[<sos>,bank]     = (-0.9×-0.0828 + 0.9×1.3471)/1.41421 = (0.0745+1.2124)/1.41421 = 0.9100
S_c0[<sos>,approved] = (-0.9×0.3273 + 0.9×-0.2558)/1.41421 = (-0.2946-0.2302)/1.41421 = -0.3711
S_c0[<sos>,the]      = (-0.9×0.5616 + 0.9×-1.5298)/1.41421 = (-0.5054-1.3768)/1.41421 = -1.3309
S_c0[<sos>,loan]     = (-0.9×-0.4203 + 0.9×-1.5270)/1.41421 = (0.3783-1.3743)/1.41421 = -0.7043

S_c0[loan,bank]     = (-0.7955×-0.0828 + -0.9004×1.3471)/1.41421 = (0.0659-1.2130)/1.41421 = -0.8112
S_c0[loan,approved] = (-0.7955×0.3273 + -0.9004×-0.2558)/1.41421 = (-0.2604+0.2304)/1.41421 = -0.0212
S_c0[loan,the]      = (-0.7955×0.5616 + -0.9004×-1.5298)/1.41421 = (-0.4468+1.3774)/1.41421 = 0.6581
S_c0[loan,loan]     = (-0.7955×-0.4203 + -0.9004×-1.5270)/1.41421 = (0.3344+1.3749)/1.41421 = 1.2088

S_c0[granted,bank]     = (0.3327×-0.0828 + -1.4007×1.3471)/1.41421 = (-0.0276-1.8871)/1.41421 = -1.3540
S_c0[granted,approved] = (0.3327×0.3273 + -1.4007×-0.2558)/1.41421 = (0.1089+0.3584)/1.41421 = 0.3304
S_c0[granted,the]      = (0.3327×0.5616 + -1.4007×-1.5298)/1.41421 = (0.1869+2.1430)/1.41421 = 1.6475
S_c0[granted,loan]     = (0.3327×-0.4203 + -1.4007×-1.5270)/1.41421 = (-0.1399+2.1393)/1.41421 = 1.4138
```

**Softmax, head 0, all 3 rows:**
```
Row <sos>: [0.9100,-0.3711,-1.3309,-0.7043]
  exp = [2.4843, 0.6900, 0.2643, 0.4944],  sum = 3.9330
  A_c0[<sos>] = [0.6317, 0.1755, 0.0672, 0.1257]

Row loan: [-0.8112,-0.0212,0.6581,1.2088]
  exp = [0.4443, 0.9790, 1.9311, 3.3496],  sum = 6.7040
  A_c0[loan] = [0.0663, 0.1460, 0.2881, 0.4997]

Row granted: [-1.3540,0.3304,1.6475,1.4138]
  exp = [0.2583, 1.3915, 5.1933, 4.1118],  sum = 10.9549
  A_c0[granted] = [0.0236, 0.1270, 0.4740, 0.3753]
```

**Weighted sum, head 0** (against `V_c₀ = [bank:[0.1847,0.9646], approved:[0.1915,-0.1265],
the:[0.1303,-1.0105], loan:[-0.5833,-1.1870]]`, a 4-term sum since there are 4 source positions):
```
O_c0[<sos>]:
  dim0: 0.6317×0.1847+0.1755×0.1915+0.0672×0.1303+0.1257×(-0.5833) = 0.1167+0.0336+0.0088-0.0733 = 0.0857
  dim1: 0.6317×0.9646+0.1755×(-0.1265)+0.0672×(-1.0105)+0.1257×(-1.1870) = 0.6093-0.0222-0.0679-0.1492 = 0.3700

O_c0[loan]:
  dim0: 0.0663×0.1847+0.1460×0.1915+0.2881×0.1303+0.4997×(-0.5833) = 0.0122+0.0280+0.0375-0.2915 = -0.2137
  dim1: 0.0663×0.9646+0.1460×(-0.1265)+0.2881×(-1.0105)+0.4997×(-1.1870) = 0.0640-0.0185-0.2911-0.5931 = -0.8387

O_c0[granted]:
  dim0: 0.0236×0.1847+0.1270×0.1915+0.4740×0.1303+0.3753×(-0.5833) = 0.0044+0.0243+0.0618-0.2189 = -0.1285
  dim1: 0.0236×0.9646+0.1270×(-0.1265)+0.4740×(-1.0105)+0.3753×(-1.1870) = 0.0228-0.0161-0.4790-0.4456 = -0.9180
```
All match `O_cross₀` below to the last shown digit. Head 1 runs the identical recipe against
`Q_c₁, K_c₁, V_c₁` (columns 2-3) — reproduce `S_c1[loan,approved]=1.7044` yourself as a check.

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

`Wo_c` is dense — full 4-term sums, exactly like `W_o` in §5.5:
```
concat_c[<sos>] = [0.0857, 0.3700, -0.5992, 0.6761]
  XA[<sos>,0] = 0.0857×1.0+0.3700×0.1+(-0.5992)×0.0+0.6761×0.1 = 0.0857+0.0370+0.0000+0.0676 = 0.1903
  XA[<sos>,1] = 0.0857×0.1+0.3700×1.0+(-0.5992)×0.1+0.6761×0.0 = 0.0086+0.3700-0.0599+0.0000 = 0.3186
  XA[<sos>,2] = 0.0857×0.0+0.3700×0.1+(-0.5992)×1.0+0.6761×0.1 = 0.0000+0.0370-0.5992+0.0676 = -0.4946
  XA[<sos>,3] = 0.0857×0.1+0.3700×0.0+(-0.5992)×0.1+0.6761×1.0 = 0.0086+0.0000-0.0599+0.6761 = 0.6247

concat_c[loan] = [-0.2137, -0.8386, -0.1940, 0.8770]
  XA[loan,0] = -0.2137×1.0+(-0.8386)×0.1+(-0.1940)×0.0+0.8770×0.1 = -0.2137-0.0839+0.0000+0.0877 = -0.2099
  XA[loan,1] = -0.2137×0.1+(-0.8386)×1.0+(-0.1940)×0.1+0.8770×0.0 = -0.0214-0.8386-0.0194+0.0000 = -0.8794
  XA[loan,2] = -0.2137×0.0+(-0.8386)×0.1+(-0.1940)×1.0+0.8770×0.1 = 0.0000-0.0839-0.1940+0.0877 = -0.1902
  XA[loan,3] = -0.2137×0.1+(-0.8386)×0.0+(-0.1940)×0.1+0.8770×1.0 = -0.0214+0.0000-0.0194+0.8770 = 0.8362

concat_c[granted] = [-0.1285, -0.9179, -0.3041, 0.8187]
  XA[granted,0] = -0.1285×1.0+(-0.9179)×0.1+(-0.3041)×0.0+0.8187×0.1 = -0.1285-0.0918+0.0000+0.0819 = -0.1384
  XA[granted,1] = -0.1285×0.1+(-0.9179)×1.0+(-0.3041)×0.1+0.8187×0.0 = -0.0129-0.9179-0.0304+0.0000 = -0.9612
  XA[granted,2] = -0.1285×0.0+(-0.9179)×0.1+(-0.3041)×1.0+0.8187×0.1 = 0.0000-0.0918-0.3041+0.0819 = -0.3140
  XA[granted,3] = -0.1285×0.1+(-0.9179)×0.0+(-0.3041)×0.1+0.8187×1.0 = -0.0129+0.0000-0.0304+0.8187 = 0.7754
```

```
concat = [O_cross₀ ‖ O_cross₁]           XA = concat @ Wo_c
[[ 0.0857,  0.3700, -0.5992,  0.6761],   [[ 0.1903,  0.3186, -0.4946,  0.6247],
 [-0.2137, -0.8386, -0.1940,  0.8770],    [-0.2099, -0.8794, -0.1901,  0.8362],
 [-0.1285, -0.9179, -0.3041,  0.8187]]    [-0.1384, -0.9611, -0.3140,  0.7754]]
```

---

## 9. Add & Norm 2

**Residual — `h1 + XA`:**
```
R2[<sos>]    = [-1.0000,1.0000,-1.0000,1.0000] + [0.1903,0.3186,-0.4946,0.6247] = [-0.8097,1.3186,-1.4946,1.6247]
R2[loan]     = [-0.7126,-0.8291,-0.1280,1.6698] + [-0.2099,-0.8794,-0.1902,0.8362] = [-0.9225,-1.7085,-0.3182,2.5060]
R2[granted]  = [0.4776,-1.4485,-0.2898,1.2607] + [-0.1384,-0.9612,-0.3140,0.7754] = [0.3392,-2.4097,-0.6038,2.0361]
```

**LayerNorm, same recipe as §7:**
```
Row <sos>: x=[-0.8097,1.3186,-1.4946,1.6247]
  μ = (-0.8097+1.3186-1.4946+1.6247)/4 = 0.6390/4 = 0.1598
  (x-μ) = [-0.9695, 1.1588, -1.6544, 1.4649]
  σ² = [0.9399+1.3428+2.7370+2.1459]/4 = 7.1656/4 = 1.7914
  σ = √1.7914 = 1.3384
  LN = [-0.9695/1.3384, 1.1588/1.3384, -1.6544/1.3384, 1.4649/1.3384] = [-0.7244, 0.8658, -1.2361, 1.0945]

Row loan: x=[-0.9225,-1.7085,-0.3182,2.5060]
  μ = (-0.9225-1.7085-0.3182+2.5060)/4 = -0.4432/4 = -0.1108
  (x-μ) = [-0.8117, -1.5977, -0.2074, 2.6168]
  σ² = [0.6589+2.5527+0.0430+6.8477]/4 = 10.1023/4 = 2.5256
  σ = √2.5256 = 1.5892
  LN = [-0.8117/1.5892, -1.5977/1.5892, -0.2074/1.5892, 2.6168/1.5892] = [-0.5108, -1.0053, -0.1305, 1.6466]

Row granted: x=[0.3392,-2.4097,-0.6038,2.0361]
  μ = (0.3392-2.4097-0.6038+2.0361)/4 = -0.6382/4 = -0.1596
  (x-μ) = [0.4988, -2.2501, -0.4442, 2.1957]
  σ² = [0.2488+5.0630+0.1973+4.8211]/4 = 10.3302/4 = 2.5826
  σ = √2.5826 = 1.6070
  LN = [0.4988/1.6070, -2.2501/1.6070, -0.4442/1.6070, 2.1957/1.6070] = [0.3104, -1.4002, -0.2764, 1.3663]
```

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

**Z = h2 @ W1.** `W1` is dense (4→8, no zero structure) — every one of the 24 cells is a genuine
4-term sum. Worked in full for row `<sos>` (`h2[<sos>]=[-0.7244,0.8658,-1.2361,1.0945]`), then the
same recipe stated compactly for the other two rows:
```
Z[<sos>,0]=-0.7244×0.30+0.8658×0.10+(-1.2361)×(-0.20)+1.0945×0.20 = -0.2173+0.0866+0.2472+0.2189=0.3354
Z[<sos>,1]=-0.7244×(-0.20)+0.8658×0.30+(-1.2361)×0.10+1.0945×0.40 =  0.1449+0.2597-0.1236+0.4378=0.7188
Z[<sos>,2]=-0.7244×0.10+0.8658×(-0.20)+(-1.2361)×0.40+1.0945×(-0.10)= -0.0724-0.1732-0.4944-0.1095=-0.8495
Z[<sos>,3]=-0.7244×0.40+0.8658×0.10+(-1.2361)×(-0.30)+1.0945×0.20 = -0.2898+0.0866+0.3708+0.2189=0.3865
Z[<sos>,4]=-0.7244×(-0.30)+0.8658×0.20+(-1.2361)×0.10+1.0945×(-0.20)= 0.2173+0.1732-0.1236-0.2189=0.0480
Z[<sos>,5]=-0.7244×0.20+0.8658×(-0.30)+(-1.2361)×0.20+1.0945×0.10 = -0.1449-0.2597-0.2472+0.1095=-0.5423
Z[<sos>,6]=-0.7244×0.10+0.8658×0.40+(-1.2361)×(-0.10)+1.0945×0.30 = -0.0724+0.3463+0.1236+0.3284=0.7259
Z[<sos>,7]=-0.7244×(-0.10)+0.8658×0.20+(-1.2361)×0.30+1.0945×(-0.20)= 0.0724+0.1732-0.3708-0.2189=-0.3441

h2[loan]=[-0.5108,-1.0053,-0.1305,1.6466]:
Z[loan,0..7] = [-0.5108×0.30-1.0053×0.10+0.1305×0.20+1.6466×0.20,   ...]  (same pattern, 4 terms each)
Z[loan]    = [0.1017, 0.4461, -0.0669, 0.0637, -0.3903, 0.3380, 0.0539, -0.5185]

h2[granted]=[0.3104,-1.4002,-0.2764,1.3663]:
Z[granted] = [0.2817, 0.0367, 0.0638, 0.3404, -0.6740, 0.5635, -0.0916, -0.6672]
```

**ReLU** — every negative entry becomes exactly `0.0`, positives pass through unchanged:
```
Hr[<sos>]    = [0.3354, 0.7188, 0.0000, 0.3865, 0.0480, 0.0000, 0.7259, 0.0000]   (Z<0 at idx 2,5,7)
Hr[loan]     = [0.1017, 0.4461, 0.0000, 0.0637, 0.0000, 0.3380, 0.0539, 0.0000]   (Z<0 at idx 2,4,7)
Hr[granted]  = [0.2817, 0.0367, 0.0638, 0.3404, 0.0000, 0.5635, 0.0000, 0.0000]   (Z<0 at idx 4,6,7)
```

**FFN_out = Hr @ W2.** `W2` is dense (8→4) — every cell is an 8-term sum, but the zeroed `Hr`
entries drop out (multiplying by 0 contributes nothing), so in practice each sum only has as many
live terms as `Hr` has nonzero entries (5 for `<sos>`, 5 for `loan`, 5 for `granted`):
```
FFN_out[<sos>,0] = 0.3354×0.20+0.7188×(-0.10)+0.3865×0.10+0.0480×(-0.20)+0.7259×0.10
                 = 0.0671-0.0719+0.0387-0.0096+0.0726 = 0.0969
FFN_out[<sos>,1] = 0.3354×(-0.10)+0.7188×0.30+0.3865×0.10+0.0480×0.30+0.7259×0.20
                 = -0.0335+0.2156+0.0387+0.0144+0.1452 = 0.3804
FFN_out[<sos>,2] = 0.3354×0.30+0.7188×0.10+0.3865×0.20+0.0480×0.10+0.7259×0.30
                 = 0.1006+0.0719+0.0773+0.0048+0.2178 = 0.4724
FFN_out[<sos>,3] = 0.3354×0.10+0.7188×0.20+0.3865×0.30+0.0480×(-0.10)+0.7259×(-0.20)
                 = 0.0335+0.1438+0.1160-0.0048-0.1452 = 0.1433

FFN_out[loan]    = [0.0889, 0.0733, 0.1716, 0.1415]     (same recipe, live terms at idx 0,1,3,5,6)
FFN_out[granted] = [0.2749, -0.0831, 0.2626, 0.2004]    (live terms at idx 0,1,2,3,5)
```

**Residual — `R3 = h2 + FFN_out`:**
```
R3[<sos>]    = [-0.7244,0.8658,-1.2361,1.0945] + [0.0969,0.3804,0.4724,0.1433] = [-0.6275, 1.2462,-0.7637, 1.2378]
R3[loan]     = [-0.5108,-1.0053,-0.1305,1.6466] + [0.0889,0.0733,0.1716,0.1415] = [-0.4219,-0.9320, 0.0411, 1.7881]
R3[granted]  = [0.3104,-1.4002,-0.2764,1.3663] + [0.2749,-0.0831,0.2626,0.2004] = [ 0.5853,-1.4833,-0.0138, 1.5667]
```

**LayerNorm, same recipe as §7 and §9:**
```
Row <sos>: μ=1.0928/4=0.2732, σ²=3.7637/4=0.9409, σ=0.9700
  OUTPUT[<sos>] = [(-0.6275-0.2732)/0.97, (1.2462-0.2732)/0.97, (-0.7637-0.2732)/0.97, (1.2378-0.2732)/0.97]
                = [-0.9286, 1.0031, -1.0690, 0.9944]

Row loan: μ=0.4753/4=0.1188, σ²=4.1892/4=1.0473, σ=1.0234
  OUTPUT[loan]  = [-0.5284, -1.0268, -0.0759, 1.6311]

Row granted: μ=0.6549/4=0.1637, σ²=4.8902/4=1.2226, σ=1.1057
  OUTPUT[granted] = [0.3813, -1.4895, -0.1605, 1.2689]
```

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

**logits = OUTPUT @ W_vocab.** `W_vocab` is dense (4→5) — 4-term sums, 15 cells total. Worked in
full for `<sos>` (`OUTPUT[<sos>]=[-0.9286,1.0031,-1.0689,0.9945]`), columns in vocab order
`<sos>,loan,granted,bank,<eos>`:
```
logit[<sos>,<sos>]   = -0.9286×0.5+1.0031×(-0.3)+(-1.0689)×0.2+0.9945×0.1 = -0.4643-0.3009-0.2138+0.0995=-0.8795
logit[<sos>,loan]    = -0.9286×(-0.2)+1.0031×0.4+(-1.0689)×0.6+0.9945×0.3 =  0.1857+0.4012-0.6413+0.2984= 0.2440
logit[<sos>,granted] = -0.9286×0.3+1.0031×0.2+(-1.0689)×(-0.3)+0.9945×0.5= -0.2786+0.2006+0.3207+0.4973= 0.7399
logit[<sos>,bank]    = -0.9286×0.1+1.0031×(-0.2)+(-1.0689)×0.1+0.9945×(-0.1)=-0.0929-0.2006-0.1069-0.0995=-0.4998
logit[<sos>,<eos>]   = -0.9286×(-0.1)+1.0031×0.1+(-1.0689)×0.2+0.9945×0.4 =  0.0929+0.1003-0.2138+0.3978= 0.3772

logits[loan]    = [0.1918, 0.1387, 0.4744, -0.0182, 0.5874]     (same recipe against OUTPUT[loan])
logits[granted] = [0.7323, -0.3878, 0.4991, 0.1931, 0.2883]     (against OUTPUT[granted])
```

**Softmax over the vocabulary, row `<sos>`** (the same mechanic as every softmax above, just 5-wide
now instead of 3 or 4):
```
exp([-0.8795, 0.2440, 0.7399, -0.4998, 0.3772]) = [0.4150, 1.2763, 2.0958, 0.6065, 1.4585]
sum = 0.4150+1.2763+2.0958+0.6065+1.4585 = 5.8521

probs[<sos>] = [0.4150/5.8521, 1.2763/5.8521, 2.0958/5.8521, 0.6065/5.8521, 1.4585/5.8521]
             = [0.0709, 0.2181, 0.3581, 0.1037, 0.2492]
```
Rows `loan` and `granted` follow the identical exp/sum/divide recipe against their own logits.

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
```

`log` here is the **natural log** (`ln`), not log base 10 — on a calculator that's the "ln" button,
not "log". Cross-entropy loss is always `-ln(p)`, because `ln` is what makes the gradient formula
`∂L/∂z = p − y` come out clean (that's used directly in §13.1).

```
ln(0.2181) = -1.5229     ln(0.2381) = -1.4354     ln(0.1919) = -1.6505

loss = -mean(ln p(gold)) = -[(-1.5229) + (-1.4354) + (-1.6505)] / 3
     = -(-4.6088) / 3 = 4.6088 / 3 = 1.5363   (≈ 1.536187 at full precision)
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
gradients describe the same computation. What follows derives every stated gradient below from
scratch, in the order it actually flows: loss → LM head → LN3 → FFN → LN2 → cross-attention →
LN1 → masked self-attention.

### 13.0 The chain, from the loss down to the FFN

**Step 1 — softmax + cross-entropy, combined.** The two operations chain into one famously clean
formula: `∂L/∂logits = (probs − one_hot(gold)) / N`, where `N=3` is the number of positions being
averaged over. This is *why* cross-entropy is paired with softmax rather than any other loss — the
messy softmax Jacobian and the `1/p` from `−ln(p)` cancel exactly. Row `<sos>` (gold = index 1,
`loan`):
```
probs[<sos>]     = [0.0709, 0.2181, 0.3581, 0.1037, 0.2492]
one_hot(loan)    = [0,      1,      0,      0,      0     ]
probs - one_hot  = [0.0709,-0.7819, 0.3581, 0.1037, 0.2492]
÷ N=3            = [0.0236,-0.2606, 0.1194, 0.0346, 0.0831]
```
This is `∂L/∂logits[<sos>]` — only the gold column changes sign (it's the only one being pushed
*down* rather than up). Rows `loan` and `granted` repeat this with their own gold index (2 and 4).

**Step 2 — LM head.** Standard linear-layer backward, same shape rule as every `dL/dW` in this
file: `dL/dW_vocab = OUTPUTᵀ @ dL/dlogits` (4×3 @ 3×5 → 4×5), and the gradient continuing backward
into the residual stream is `dL/dOUT = dL/dlogits @ W_vocabᵀ` (3×5 @ 5×4 → 3×4).

**Step 3 — LayerNorm 3 backward.** LayerNorm's backward formula (`d=4` features per row, same `d`
everywhere in this file) is heavier than a linear layer's because `μ` and `σ` both depend on every
element of the row:
```
xhat = (x-μ)/σ                                     (this is what LN's forward already computed)
dvar = Σ(dy·(x-μ)) × (-0.5)/σ³
dμ   = -Σ(dy/σ)                                    (the term coupling dvar to dμ is always exactly
                                                     zero, because mean(x-μ)=0 by construction)
dx   = dy/σ + dvar×2(x-μ)/d + dμ/d
```
Worked for row `<sos>`, with `dL/dOUT[<sos>]=[0.0949,-0.0861,-0.1674,0.0136]` and the forward
values `x=[-0.6275,1.2462,-0.7637,1.2378]`, `σ=0.9700`:
```
Σ(dy·(x-μ))  = 0.0949×(-0.9007)+(-0.0861)×0.9730+(-0.1674)×(-1.0369)+0.0136×0.9646
             = -0.0855-0.0838+0.1736+0.0131 = 0.0174
dvar = 0.0174 × (-0.5)/0.9700³ = 0.0174 × (-0.5)/0.9127 = -0.0095

Σ(dy/σ) = (0.0949-0.0861-0.1674+0.0136)/0.9700 = -0.1450/0.9700 = -0.1495
dμ = -(-0.1495) = 0.1495

dx[0] = 0.0949/0.97 + (-0.0095)×2×(-0.9007)/4 + 0.1495/4 = 0.0978+0.0043+0.0374 = 0.1395
```
(the remaining 3 entries of this row, and the other two rows, follow the identical four-line
recipe) — this is `dL/dR3`, and because `R3 = h2 + FFN_out` is a residual sum, **both branches
receive this exact gradient unchanged**: `dL/dh2 (direct) = dL/dFFN_out = dL/dR3`.

**Step 4 — FFN backward.** Standard linear-ReLU-linear backward:
```
dL/dW2      = Hrᵀ @ dL/dFFN_out                      (8×3 @ 3×4 → 8×4)
dL/dHr      = dL/dFFN_out @ W2ᵀ                       (3×4 @ 4×8 → 3×8)
dL/dZ       = dL/dHr ⊙ (Z>0)                          ReLU backward: the same mask as forward,
                                                       dead neurons (Z≤0) get exactly zero gradient
dL/dW1      = h2ᵀ @ dL/dZ                             (4×3 @ 3×8 → 4×8)
dL/dh2 (via FFN) = dL/dZ @ W1ᵀ                        (3×8 @ 8×4 → 3×4)
```
`dL/dh2 total = dL/dh2 (direct, from the residual) + dL/dh2 (via FFN)` — two paths converging on
the same tensor, added, exactly like every residual gradient in this file. This total is what feeds
LayerNorm 2's backward, which repeats Step 3's four-line recipe on `R2` instead of `R3`, producing
`dL/dh1 (direct) = dL/dXA = dL/dR2` — and that is where cross-attention's backward picks up.

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

**Cross-attention's backward runs first** in the actual chain (it's closer to the loss than
self-attention), so the recipe is introduced here and then reused for self-attention below.
`dL/dc_mat`-style softmax backward, generalized to a full attention matrix, is:
```
dL/dA = dL/dO @ Vᵀ                                  (undo the weighted sum)
g     = rowsum(A ⊙ dL/dA)                            (per-row scalar, the attention-weighted average)
dL/dS = A ⊙ (dL/dA − g)                              (softmax backward — same formula as 05/06)
dL/dQ = dL/dS @ K / √d_head        dL/dK = dL/dSᵀ @ Q / √d_head        dL/dV = Aᵀ @ dL/dO
```
Starting from `dL/dXA = dL/dR2` (Step 3 above), split into heads, then `dL/dconcat_c = dL/dXA @
Wo_cᵀ` and `dL/dWo_c = concat_cᵀ @ dL/dXA`. For head 0, using `dL/dO_c0[<sos>]` from that split and
`A_c0[<sos>]=[0.6317,0.1755,0.0672,0.1257]`:
```
dL/dA_c0[<sos>] = dL/dO_c0[<sos>] @ V_c0ᵀ            (a length-4 vector, one entry per source token)
g = 0.6317×dA[0] + 0.1755×dA[1] + 0.0672×dA[2] + 0.1257×dA[3]
dL/dS_c0[<sos>,j] = A_c0[<sos>,j] × (dA[j] − g)      for j = bank, approved, the, loan
```
This is exactly the number reported in §13's derivation of `dL/dWq_c`, `dL/dWk_c` below, and it is
**not zero anywhere** — cross-attention carries no mask, so unlike self-attention (next), every one
of the 12 entries of `dL/dS_c` (3 target rows × 4 source columns, per head) is a live gradient. Then
`dL/dQc = dL/dS_c @ Kc/√2`, `dL/dKc = dL/dS_cᵀ @ Qc/√2`, `dL/dVc = A_cᵀ @ dL/dOc`, and:
```
dL/dWq_c = h1ᵀ @ dL/dQc          dL/dWk_c = MEMORYᵀ @ dL/dKc          dL/dWv_c = MEMORYᵀ @ dL/dVc
dL/dh1 (via cross) = dL/dQc @ Wq_cᵀ
dL/dMEMORY         = dL/dKc @ Wk_cᵀ + dL/dVc @ Wv_cᵀ            ← §13.3's table, derived
```
`dL/dh1 total = dL/dh1 (direct, from residual 1) + dL/dh1 (via cross)` feeds LayerNorm 1's
backward exactly like Step 3, producing `dL/dX (direct) = dL/dMHA = dL/dR1`, which is where masked
self-attention's backward — the one place a mask actually changes the math — picks up.

**Masked self-attention runs the identical four formulas above**, with one addition: the softmax
was computed over `masked` scores (`-inf` in the upper triangle), so the *forward* `A` already has
exact zeros there — and `dL/dS = A ⊙ (dL/dA − g)` inherits that zero multiplicatively, **before any
mask needs to be reapplied on the backward pass.** Nobody codes a second masking step for the
gradient; it's already zero because `A` already is.

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

From `dL/dS` (both heads), the last leg of self-attention's backward is mechanical:
```
dL/dQ = dL/dS @ K / √2         dL/dK = dL/dSᵀ @ Q / √2         dL/dV = Aᵀ @ dL/dO
dL/dWq = Xᵀ @ dL/dQ            dL/dWk = Xᵀ @ dL/dK             dL/dWv = Xᵀ @ dL/dV
dL/dX (via self-attn) = dL/dQ @ Wqᵀ + dL/dK @ Wkᵀ + dL/dV @ Wvᵀ
```
`dL/dX total = dL/dX (direct, from residual 1) + dL/dX (via self-attn)` — and that total is the
end of the chain for this layer; it's what a stack of decoder layers would pass to the layer below.

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

Both are the same linear-layer backward as `dL/dW_vocab` in §13.0: `dL/dW_o = concatᵀ @ dL/dMHA`
and `dL/dWo_c = concat_cᵀ @ dL/dXA`, where `concat` is the `[O₀ ‖ O₁]` from §5.5 and `dL/dMHA =
dL/dR1` (the gradient arriving at the top of this sublayer, from §13.0's LayerNorm-1 backward).
Because `concat`'s first two columns are entirely `O₀` (head 0) and the last two are entirely `O₁`
(head 1), and matrix-multiply-transpose sums over rows independently per output column, **rows 0-1
of `dL/dW_o` only ever see head 0's output, rows 2-3 only ever see head 1's** — nothing mixes the
two heads back together at this step. That's why the split below is exact, not approximate:

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

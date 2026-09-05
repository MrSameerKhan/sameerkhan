# 06b — Transformer Encoder with Multi-Head Attention: End-to-End

> Companion to [06_transformer_end_to_end.md](06_transformer_end_to_end.md).
> That file runs `d_model = 2, n_heads = 1` — at that width **only one head is possible**
> (`d_head = 2/2 = 1` is degenerate). This file widens to `d_model = 4` so two real heads fit,
> and shows the one thing a single head cannot do.
>
> Same 4-token length as 06, so every shape lines up. New sentence.
> Every number below was computed and verified — the backward pass against `torch.autograd`.

---

## Table of Contents

1. What changes from 06
2. Setup — sentence, dimensions
3. Embeddings + Positional Encoding
4. Weight setup
5. Q, K, V — projections
6. Splitting into heads
7. Head 0 — the subject head
8. Head 1 — the object head
9. **Why one head cannot do this**
10. Concat + W_o
11. Residual + LayerNorm
12. FFN
13. Encoder output
14. Backward — where the gradients go
15. Weight update — verify the loss dropped
16. Scaling to real transformers
17. Quick reference

---

## 1. What changes from 06

| | 06 | **06b (this file)** |
|---|---|---|
| Sentence | `cat sat on mat` | `bank approved the loan` |
| Tokens `L` | 4 | 4 |
| `d_model` | 2 | **4** |
| `n_heads` | 1 | **2** |
| `d_head` | 2 | **2** |
| `√d_k` | 1.414 | **1.414** — unchanged |
| `d_ff` | 4 | 8 |
| PE frequency pairs | 1 (`i=0` only) | **2** (`i=0,1`) |
| `W_o` | identity (no-op) | **real 4×4 mixing matrix** |

Two things only become real at `d_model = 4`:

1. **`W_o` does something.** With one head it is the identity — a no-op. With two heads it is the
   only place the heads ever interact.
2. **Positional encoding has more than one frequency.** At `d=2` every position used the same
   `sin(pos)`/`cos(pos)` pair. At `d=4` there are two pairs oscillating at different rates.

`d_head` stays 2, so the scaling constant is still `√2 = 1.414` — the number you already know.

---

## 2. Setup

```
Sentence:  bank  approved  the  loan
Position:   0       1       2     3

d_model = 4        n_heads = 2        d_head = d_model / n_heads = 2
d_ff    = 8        L = 4              √d_k = √2 = 1.4142
```

**Why this sentence.** `bank` is ambiguous — financial institution or river edge. Only the right
context (`approved`, `loan`) resolves it. The encoder is bidirectional, so it can use that; a
causal decoder could not. And the query token `approved` needs two relationships at once:

- **who** approved → `bank` (subject)
- **what** was approved → `loan` (object)

That competition is the whole point of this file.

---

## 3. Embeddings + Positional Encoding

### 3.1 Vocabulary

Dimensions are given meaning by construction: **dims 0-1 carry "entity/who", dims 2-3 carry
"object/what".** Real embeddings are learned and not this tidy — this is a teaching device.

| Word | Index | Embedding | Character |
|------|-------|-----------|-----------|
| bank | 0 | `[1.0, 0.8, 0.1, 0.1]` | entity-heavy |
| approved | 1 | `[0.3, 0.2, 0.4, 0.3]` | balanced — the query |
| the | 2 | `[0.1, 0.1, 0.2, 0.2]` | function word |
| loan | 3 | `[0.1, 0.1, 1.0, 0.9]` | object-heavy |

### 3.2 Positional encoding — now with two frequencies

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

d = 4  ->  i = 0 and i = 1

i=0:  exponent 2i/d = 0.0   frequency = 1/10000^0.0   = 1.0000   (fast)
i=1:  exponent 2i/d = 0.5   frequency = 1/10000^0.5   = 0.0100   (slow)
```

```
        dim0     dim1     dim2     dim3
pos 0 [ 0.0000,  1.0000,  0.0000,  1.0000]
pos 1 [ 0.8415,  0.5403,  0.0100,  1.0000]
pos 2 [ 0.9093, -0.4161,  0.0200,  0.9998]
pos 3 [ 0.1411, -0.9900,  0.0300,  0.9996]
```

**This is what 06 could not show.** Dims 0-1 swing hard across positions (`0.00 → 0.84 → 0.91 → 0.14`);
dims 2-3 barely move (`0.000 → 0.010 → 0.020 → 0.030`). Fast dims separate *nearby* positions, slow
dims separate *distant* ones. At `d=512` there are 256 such pairs spanning the whole range — that is
how a transformer encodes position at every scale simultaneously.

### 3.3 X = E + PE

```
             dim0     dim1     dim2     dim3
bank      [ 1.0000,  1.8000,  0.1000,  1.1000]
approved  [ 1.1415,  0.7403,  0.4100,  1.3000]
the       [ 1.0093, -0.3161,  0.2200,  1.1998]
loan      [ 0.2411, -0.8900,  1.0300,  1.8996]
```

Shape `(4, 4)` = `(L, d_model)`.

---

## 4. Weight setup

All matrices are `d_model × d_model = 4×4`. **They are not split per head** — that is the point
of §6.

```
Wq (query)                      Wk (key)
[[1.2, 0.0, 0.0, 0.0],          [[1.0, 0.2, 0.0, 0.0],
 [0.0, 1.2, 0.0, 0.0],           [0.2, 1.0, 0.0, 0.0],
 [0.0, 0.0, 1.2, 0.0],           [0.0, 0.0, 1.0, 0.2],
 [0.0, 0.0, 0.0, 1.2]]           [0.0, 0.0, 0.2, 1.0]]

Wv (value)                      W_o (output mixing)
[[0.9, 0.1, 0.0, 0.0],          [[0.9, 0.1, 0.1, 0.0],
 [0.1, 0.9, 0.0, 0.0],           [0.1, 0.9, 0.0, 0.1],
 [0.0, 0.0, 0.9, 0.1],           [0.1, 0.0, 0.9, 0.1],
 [0.0, 0.0, 0.1, 0.9]]           [0.0, 0.1, 0.1, 0.9]]
```

`Wq/Wk/Wv` are block-diagonal here so each head's subspace stays clean and the demonstration is
legible. **Real transformers do not have this structure** — it emerges (partially, messily) from
training. `W_o` is deliberately *not* block-diagonal: its off-diagonal entries are what let head 0
and head 1 influence each other.

---

## 5. Q, K, V

Recall `X` from §3.3: `bank=[1.0000,1.8000,0.1000,1.1000]`, `approved=[1.1415,0.7403,0.4100,1.3000]`,
`the=[1.0093,-0.3161,0.2200,1.1998]`, `loan=[0.2411,-0.8900,1.0300,1.8996]`.

**Q = X @ Wq.** `Wq = diag(1.2,1.2,1.2,1.2)` (§4) — every off-diagonal entry is `0.0`, so this is
just **multiply every entry by 1.2**, nothing else:
```
Q[bank]     = [1.0000,1.8000,0.1000,1.1000] × 1.2 = [1.2000, 2.1600, 0.1200, 1.3200]
Q[approved] = [1.1415,0.7403,0.4100,1.3000] × 1.2 = [1.3698, 0.8884, 0.4920, 1.5600]
Q[the]      = [1.0093,-0.3161,0.2200,1.1998] × 1.2 = [1.2112,-0.3793, 0.2640, 1.4398]
Q[loan]     = [0.2411,-0.8900,1.0300,1.8996] × 1.2 = [0.2893,-1.0680, 1.2360, 2.2795]
```

**K = X @ Wk.** `Wk` is block-diagonal (dims 0-1 mix with each other via coefficients `1.0`/`0.2`;
dims 2-3 mix with each other the same way) — every cell is a genuine **2-term** sum, not 4:
```
K[bank,0] = 1.0000×1.0 + 1.8000×0.2 = 1.0000+0.3600 = 1.3600
K[bank,1] = 1.0000×0.2 + 1.8000×1.0 = 0.2000+1.8000 = 2.0000
K[bank,2] = 0.1000×1.0 + 1.1000×0.2 = 0.1000+0.2200 = 0.3200
K[bank,3] = 0.1000×0.2 + 1.1000×1.0 = 0.0200+1.1000 = 1.1200

K[approved,0] = 1.1415×1.0 + 0.7403×0.2 = 1.1415+0.1481 = 1.2896
K[approved,1] = 1.1415×0.2 + 0.7403×1.0 = 0.2283+0.7403 = 0.9686
K[approved,2] = 0.4100×1.0 + 1.3000×0.2 = 0.4100+0.2600 = 0.6700
K[approved,3] = 0.4100×0.2 + 1.3000×1.0 = 0.0820+1.3000 = 1.3820

K[the,0] = 1.0093×1.0 + (-0.3161)×0.2 = 1.0093-0.0632 = 0.9461
K[the,1] = 1.0093×0.2 + (-0.3161)×1.0 = 0.2019-0.3161 = -0.1142
K[the,2] = 0.2200×1.0 + 1.1998×0.2 = 0.2200+0.2400 = 0.4600
K[the,3] = 0.2200×0.2 + 1.1998×1.0 = 0.0440+1.1998 = 1.2438

K[loan,0] = 0.2411×1.0 + (-0.8900)×0.2 = 0.2411-0.1780 = 0.0631
K[loan,1] = 0.2411×0.2 + (-0.8900)×1.0 = 0.0482-0.8900 = -0.8418
K[loan,2] = 1.0300×1.0 + 1.8996×0.2 = 1.0300+0.3799 = 1.4099
K[loan,3] = 1.0300×0.2 + 1.8996×1.0 = 0.2060+1.8996 = 2.1056
```

**V = X @ Wv.** Same block-diagonal shape as `Wk`, coefficients `0.9`/`0.1` instead of `1.0`/`0.2`:
```
V[bank,0] = 1.0000×0.9 + 1.8000×0.1 = 0.9000+0.1800 = 1.0800
V[bank,1] = 1.0000×0.1 + 1.8000×0.9 = 0.1000+1.6200 = 1.7200
V[bank,2] = 0.1000×0.9 + 1.1000×0.1 = 0.0900+0.1100 = 0.2000
V[bank,3] = 0.1000×0.1 + 1.1000×0.9 = 0.0100+0.9900 = 1.0000

V[approved,0] = 1.1415×0.9 + 0.7403×0.1 = 1.0274+0.0740 = 1.1014
V[approved,1] = 1.1415×0.1 + 0.7403×0.9 = 0.1142+0.6663 = 0.7804
V[approved,2] = 0.4100×0.9 + 1.3000×0.1 = 0.3690+0.1300 = 0.4990
V[approved,3] = 0.4100×0.1 + 1.3000×0.9 = 0.0410+1.1700 = 1.2110

V[the,0] = 1.0093×0.9 + (-0.3161)×0.1 = 0.9084-0.0316 = 0.8768
V[the,1] = 1.0093×0.1 + (-0.3161)×0.9 = 0.1009-0.2845 = -0.1836
V[the,2] = 0.2200×0.9 + 1.1998×0.1 = 0.1980+0.1200 = 0.3180
V[the,3] = 0.2200×0.1 + 1.1998×0.9 = 0.0220+1.0798 = 1.1018

V[loan,0] = 0.2411×0.9 + (-0.8900)×0.1 = 0.2170-0.0890 = 0.1280
V[loan,1] = 0.2411×0.1 + (-0.8900)×0.9 = 0.0241-0.8010 = -0.7769
V[loan,2] = 1.0300×0.9 + 1.8996×0.1 = 0.9270+0.1900 = 1.1170
V[loan,3] = 1.0300×0.1 + 1.8996×0.9 = 0.1030+1.7096 = 1.8126
```

```
Q = X @ Wq                          K = X @ Wk
[[ 1.2000,  2.1600,  0.1200,  1.3200],   [[ 1.3600,  2.0000,  0.3200,  1.1200],
 [ 1.3698,  0.8884,  0.4920,  1.5599],    [ 1.2895,  0.9686,  0.6700,  1.3819],
 [ 1.2112, -0.3794,  0.2640,  1.4398],    [ 0.9461, -0.1143,  0.4600,  1.2438],
 [ 0.2893, -1.0680,  1.2360,  2.2795]]    [ 0.0631, -0.8418,  1.4099,  2.1055]]

V = X @ Wv
[[ 1.0800,  1.7200,  0.2000,  1.0000],
 [ 1.1014,  0.7804,  0.4990,  1.2110],
 [ 0.8768, -0.1836,  0.3180,  1.1018],
 [ 0.1280, -0.7769,  1.1170,  1.8126]]
```

Each is `(4, 4)`. **Full width — no reduction has happened yet.**

---

## 6. Splitting into heads

This is the step people expect to be a projection. **It is a reshape.**

```
Q  (4, 4)   ->   (2 heads, 4 tokens, 2 dims)

head 0 = columns 0-1        head 1 = columns 2-3
[[ 1.2000,  2.1600],        [[ 0.1200,  1.3200],
 [ 1.3698,  0.8884],         [ 0.4920,  1.5599],
 [ 1.2112, -0.3794],         [ 0.2640,  1.4398],
 [ 0.2893, -1.0680]]         [ 1.2360,  2.2795]]
```

There is **no `W_q_head0`**. There is one `W_q` of shape `4×4`, and its output is sliced. This is
why multi-head costs the same as single-head: identical matmuls, different view of the result.

**What multiplies:** the attention matrix. One head gives `1 × L × L`. Two heads give `2 × L × L`.
That factor is what Flash Attention attacks (board 12).

---

## 7. Head 0 — dims 0-1

```
scores = Q₀ @ K₀ᵀ / √2

              bank  approved     the    loan
bank      [ 4.2087,  2.5736,  0.6282, -1.2321]
approved  [ 2.5736,  1.8574,  0.8445, -0.4676]
the       [ 0.6282,  0.8445,  0.8409,  0.2799]
loan      [-1.2321, -0.4676,  0.2799,  0.6486]

A₀ = softmax(scores, dim=-1)

              bank  approved     the    loan
bank      [ 0.8149,  0.1589,  0.0227,  0.0035]
approved  [ 0.5835,  0.2851,  0.1035,  0.0279]   <- the row that matters
the       [ 0.2390,  0.2967,  0.2956,  0.1687]
loan      [ 0.0702,  0.1508,  0.3185,  0.4605]

O₀ = A₀ @ V₀
[[ 1.0754,  1.5187],
 [ 1.0385,  1.1854],
 [ 0.8657,  0.4573],
 [ 0.5801, -0.1777]]
```

**Row `approved`: `bank` = 0.5835, `loan` = 0.0279.** Head 0 found the **subject** and almost
entirely ignored the object. Every row sums to 1.0 — verify it.

---

## 8. Head 1 — dims 2-3

**All 16 cells of `Q₁ @ K₁ᵀ / √2`.** Head 1 reads columns 2-3 of `Q` and `K` (§5):
```
Q₁: bank=[0.1200,1.3200]  approved=[0.4920,1.5599]  the=[0.2640,1.4398]  loan=[1.2360,2.2795]
K₁: bank=[0.3200,1.1200]  approved=[0.6700,1.3819]  the=[0.4600,1.2438]  loan=[1.4099,2.1055]
```
Every cell is a 2-term dot product divided by `√d_head = √2 = 1.41421` — same recipe as head 0,
just against head 1's slice. Because this is self-attention (`Q` and `K` come from the same `X`),
the matrix is **symmetric before softmax**: `S₁[i,j] = S₁[j,i]`, so only the upper triangle (10
cells) needs computing — the lower triangle is a free copy:

```
S₁[bank,bank]      = (0.1200×0.3200 + 1.3200×1.1200)/1.41421 = (0.0384+1.4784)/1.41421 = 1.0725
S₁[bank,approved]  = (0.1200×0.6700 + 1.3200×1.3819)/1.41421 = (0.0804+1.8241)/1.41421 = 1.3467
S₁[bank,the]       = (0.1200×0.4600 + 1.3200×1.2438)/1.41421 = (0.0552+1.6418)/1.41421 = 1.2000
S₁[bank,loan]      = (0.1200×1.4099 + 1.3200×2.1055)/1.41421 = (0.1692+2.7793)/1.41421 = 2.0849

S₁[approved,approved] = (0.4920×0.6700 + 1.5599×1.3819)/1.41421 = (0.3296+2.1557)/1.41421 = 1.7574
S₁[approved,the]       = (0.4920×0.4600 + 1.5599×1.2438)/1.41421 = (0.2263+1.9402)/1.41421 = 1.5320
S₁[approved,loan]      = (0.4920×1.4099 + 1.5599×2.1055)/1.41421 = (0.6937+3.2843)/1.41421 = 2.8130

S₁[the,the]   = (0.2640×0.4600 + 1.4398×1.2438)/1.41421 = (0.1214+1.7909)/1.41421 = 1.3521
S₁[the,loan]  = (0.2640×1.4099 + 1.4398×2.1055)/1.41421 = (0.3723+3.0318)/1.41421 = 2.4068

S₁[loan,loan] = (1.2360×1.4099 + 2.2795×2.1055)/1.41421 = (1.7430+4.7995)/1.41421 = 4.6266

(lower triangle — free copies of the above, since S₁ is symmetric):
S₁[approved,bank]=S₁[bank,approved]=1.3467   S₁[the,bank]=S₁[bank,the]=1.2000
S₁[the,approved]=S₁[approved,the]=1.5320     S₁[loan,bank]=S₁[bank,loan]=2.0849
S₁[loan,approved]=S₁[approved,loan]=2.8130   S₁[loan,the]=S₁[the,loan]=2.4068
```

**Why the symmetry disappears after softmax** (worth having ready — it trips people up): softmax
normalizes **per row**, and different rows have different *other* entries competing for that row's
probability budget. `S₁[bank,approved]` and `S₁[approved,bank]` are the same raw number, `1.3467`,
but `A₁[bank,approved]=0.2121` while `A₁[approved,bank]=0.1243` (§8 below) — because row `bank`
divides by a different sum than row `approved` does. Symmetry is a property of the *raw scores
only*, and it's specific to self-attention (`Q=K` source); cross-attention (06c §8) is never
symmetric even before softmax, since `Q` and `K` come from different sequences entirely.

```
scores = Q₁ @ K₁ᵀ / √2

              bank  approved     the    loan
bank      [ 1.0725,  1.3467,  1.2000,  2.0849]
approved  [ 1.3467,  1.7574,  1.5320,  2.8130]
the       [ 1.2000,  1.5320,  1.3521,  2.4068]
loan      [ 2.0849,  2.8130,  2.4068,  4.6260]

A₁ = softmax(scores, dim=-1)

              bank  approved     the    loan
bank      [ 0.1612,  0.2121,  0.1831,  0.4436]
approved  [ 0.1243,  0.1874,  0.1496,  0.5386]   <- same query, different answer
the       [ 0.1449,  0.2020,  0.1687,  0.4844]
loan      [ 0.0583,  0.1208,  0.0805,  0.7404]

O₁ = A₁ @ V₁
[[ 0.6918,  1.4239],
 [ 0.7676,  1.4925],
 [ 0.7245,  1.4534],
 [ 0.9245,  1.6353]]
```

**Row `approved`: `loan` = 0.5386, `bank` = 0.1243.** Exactly inverted. Head 1 found the **object**.

Same token, same layer, same step — two different questions answered in parallel.

---

## 9. Why one head cannot do this

Run the identical input through **one** head of `d_head = 4` (so `√d_k = √4 = 2`):

```
query = 'approved'
                       bank  approved     the    loan
1 head,  d_head=4  [ 0.4049,  0.3262,  0.1359,  0.1330]
2 heads, head 0    [ 0.5835,  0.2851,  0.1035,  0.0279]
2 heads, head 1    [ 0.1243,  0.1874,  0.1496,  0.5386]
```

| | mass on `bank` | mass on `loan` |
|---|---|---|
| **1 head** | 0.4049 | **0.1330** |
| **head 0** | **0.5835** | 0.0279 |
| **head 1** | 0.1243 | **0.5386** |

**The single head gives `loan` only 0.1330.** The object relationship is nearly lost — the head
spent its probability mass on the subject and on itself, and what reaches `V` is a blur.

Two heads capture the object at **0.5386 — four times stronger** — while *also* capturing the
subject more sharply than the single head managed.

### The mechanism

Softmax is a **competition**. One head = one distribution per query = one budget of 1.0 to spend.
Two relationships must share it, and the weighted average `A @ V` smears them into one vector.

Two heads = two independent budgets. Nothing is shared, so nothing competes.

**This is why multi-head exists.** Not extra capacity — the parameter count is identical. It is an
escape from the single-softmax bottleneck.

> **Honest note.** These weights were *designed* so the separation is visible. With plausible random
> weights every entry lands near `1/L = 0.25` — logits near zero make softmax flat. Head
> specialisation is a property of **trained** models; the architecture only makes it *possible*.
> An untrained toy cannot demonstrate it, and claiming otherwise in an interview is a trap.

---

## 10. Concat + W_o

```
concat = [O₀ ‖ O₁]                shape (4, 4)

           └─ head 0 ─┘ └─ head 1 ─┘
bank     [ 1.0754, 1.5187, 0.6918, 1.4239]
approved [ 1.0385, 1.1854, 0.7676, 1.4925]
the      [ 0.8657, 0.4573, 0.7245, 1.4534]
loan     [ 0.5801,-0.1777, 0.9245, 1.6353]

MHA = concat @ W_o
[[ 1.1889,  1.6168,  0.8726,  1.5025],
 [ 1.1300,  1.3200,  0.9439,  1.5385],
 [ 0.8973,  0.6435,  0.8839,  1.4262],
 [ 0.5968,  0.0616,  1.0536,  1.5465]]
```

Up to the concat, head 0 and head 1 **have never interacted**. `W_o` is the first and only place
they mix — its off-diagonal entries let "the subject is `bank`" and "the object is `loan`" combine
into one representation.

**Delete `W_o` and you have two parallel monologues, never a conversation.** In 06 it was the
identity, so this step did nothing at all.

---

## 11. Residual + LayerNorm

```
R1 = X + MHA
[[ 2.1889,  3.4168,  0.9726,  2.6025],
 [ 2.2714,  2.0603,  1.3539,  2.8385],
 [ 1.9066,  0.3273,  1.1039,  2.6260],
 [ 0.8379, -0.8284,  2.0836,  3.4460]]

per-row mean: [2.2952, 2.1310, 1.4910, 1.3848]
per-row var : [0.7783, 0.2823, 0.7413, 2.4836]

h1 = LayerNorm(R1) = (R1 - mean) / √(var + 1e-5)
[[-0.1205,  1.2713, -1.4993,  0.3484],
 [ 0.2643, -0.1332, -1.4626,  1.3315],
 [ 0.4827, -1.3516, -0.4495,  1.3184],
 [-0.3470, -1.4044,  0.4434,  1.3079]]
```

LayerNorm normalises **across the 4 features of each token independently** — not across tokens.
Row `loan` had variance 2.4836 (widest spread); after normalisation every row has mean ≈ 0 and
variance ≈ 1. Tokens cannot leak into each other here, which is what makes it safe with padding.

---

## 12. FFN

```
d_ff = 8      FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

Z = h1 @ W1
[[ 0.1833, -0.3467,  0.3816, -0.5649, -0.3012,  0.4283, -0.4618,  0.2076],
 [-0.0805, -0.3993,  0.5324, -0.4519, -0.0801,  0.0668, -0.4124,  0.4390],
 [-0.2573, -0.1735,  0.3053, -0.0480,  0.1867, -0.3219, -0.0866,  0.3569],
 [-0.5158, -0.0768,  0.2076,  0.3082,  0.1154, -0.2558,  0.0983,  0.1825]]

ReLU(Z)
[[0.1833, 0.0000, 0.3816, 0.0000, 0.0000, 0.4283, 0.0000, 0.2076],
 [0.0000, 0.0000, 0.5324, 0.0000, 0.0000, 0.0668, 0.0000, 0.4390],
 [0.0000, 0.0000, 0.3053, 0.0000, 0.1867, 0.0000, 0.0000, 0.3569],
 [0.0000, 0.0000, 0.2076, 0.3082, 0.1154, 0.0000, 0.0983, 0.1825]]

FFN out = ReLU(Z) @ W2
[[0.3540, 0.2805, 0.2553, 0.3499],
 [0.3048, 0.1704, 0.3074, 0.2995],
 [0.2734, 0.1154, 0.2495, 0.2463],
 [0.2038, 0.2451, 0.2650, 0.2571]]
```

**ReLU zeroed 17 of 32 activations — 53%.** That sparsity is the FFN's non-linearity doing its job:
different tokens light up different hidden units.

**Division of labour:** attention *moves* information between positions (it is a weighted average —
it cannot compute anything new). The FFN *transforms* information within a position. It is the only
non-linear per-token computation in the block.

`d_ff = 8 = 2 × d_model` here to keep the arithmetic readable. **Real transformers use `4 × d_model`**
(BERT-base: 768 → 3072).

---

## 13. Encoder output

```
R2 = h1 + FFN_out
[[ 0.2336,  1.5519, -1.2440,  0.6983],
 [ 0.5690,  0.0372, -1.1552,  1.6310],
 [ 0.7561, -1.2361, -0.2000,  1.5647],
 [-0.1432, -1.1593,  0.7084,  1.5650]]

OUTPUT = LayerNorm(R2)
             dim0     dim1     dim2     dim3
bank     [-0.0753,  1.2246, -1.5323,  0.3829]
approved [ 0.2975, -0.2325, -1.4209,  1.3559]
the      [ 0.5105, -1.3907, -0.4019,  1.2821]
loan     [-0.3821, -1.3882,  0.4611,  1.3093]
```

Shape `(4, 4)` — **identical to the input**. That is what makes blocks stackable: `d_model` is a bus
running the depth of the network, unchanged from embedding to final layer. Stack 12 of these and you
have BERT-base's encoder.

These are **contextualised** representations. `bank` no longer carries its dictionary embedding; it
carries "bank, as used in a sentence about approving a loan."

---

## 14. Backward — where the gradients go

Loss: MSE against a target that pushes each token toward a role-specific vector.

```
loss = mean((OUTPUT - T)²) = 0.395052
```

### 14.1 The gradient that splits the heads

```
dL/dW_o
[[-0.1388, -0.0637,  0.0555,  0.1470],
 [-0.2457, -0.0459,  0.0750,  0.2166],
 [-0.0627, -0.0665,  0.0419,  0.0872],
 [-0.1318, -0.1232,  0.0782,  0.1768]]
      └── rows 0-1 ──┘└── rows 2-3 ──┘
        from HEAD 0      from HEAD 1
```

`W_o` is where the backward pass **splits**. Gradient arriving at the concatenated vector is routed
by row: rows 0-1 flow back into head 0's `Wq/Wk/Wv` slices, rows 2-3 into head 1's. The heads are
independent on the way forward *and* on the way back — they only ever meet inside `W_o`.

### 14.2 Gradient magnitudes — the useful observation

```
|dL/dWq| max = 0.026942
|dL/dWk| max = 0.049597
|dL/dWv| max = 0.229756     <- 8.5× larger than Wq
|dL/dWo| max = 0.245680     <- largest
|dL/dW1| max = 0.052943
|dL/dW2| max = 0.034362
```

**`W_v` and `W_o` get gradients an order of magnitude larger than `W_q`.**

Why: `V` and `W_o` sit on a **linear** path to the output — `A @ V @ W_o`. `Q` reaches the loss only
*through the softmax*, and softmax gradients are `A(1-A)`-shaped: they shrink toward zero as the
distribution sharpens. Head 0's `approved` row is already at 0.5835, so its Jacobian is small.

This is a real property, not an artefact of the toy: **the query/key path learns more slowly than
the value/output path**, and it is why attention logits are the part most sensitive to
initialisation and scaling.

---

## 15. Weight update — verify

```
SGD, lr = 0.5,  W ← W - lr · dL/dW   (all six matrices)

loss before = 0.395052
loss after  = 0.354209      DECREASED by 0.040843   ✓
```

The block learns. All numbers in §14–15 were computed with `torch.autograd`, and the torch forward
pass was checked against the numpy forward in §5–13 (`allclose`, atol 1e-6) — so the hand-written
arithmetic above and the autograd gradients describe the same computation.

---

## 16. Scaling to real transformers

| | this file | BERT-base | GPT-3 |
|---|---|---|---|
| `d_model` | 4 | 768 | 12288 |
| `n_heads` | 2 | 12 | 96 |
| `d_head` | 2 | **64** | 128 |
| `d_ff` | 8 (2×) | 3072 (**4×**) | 49152 (4×) |
| layers | 1 | 12 | 96 |
| `√d_k` | 1.414 | 8 | 11.3 |

**`d_head = 64` is remarkably stable** — the 2017 paper (512/8), BERT-base (768/12) and BERT-large
(1024/16) all land on 64. BERT-large scaled width *and* head count together to hold it there. Below
~32 a head cannot carry a useful subspace; above ~128 heads go redundant.

### Where the parameters live

Per BERT-base layer:

```
attention  4 × d_model²        = 4 × 768²        = 2,359,296   (33%)
FFN        2 × d_model × d_ff  = 2 × 768 × 3072  = 4,718,592   (67%)
                                                   ─────────
                                                   7,077,888

× 12 layers                                      =  85.1M
+ embeddings 30522 × 768                         =  23.8M
                                                    ──────
                                                    109M   ✓ ("110M")
```

**The FFN is two-thirds of every layer.** Attention is the part that is expensive at *inference*
(quadratic in sequence length); the FFN is the part that is heavy in *parameters*. Most people
assume the opposite.

The 33/67 split is scale-invariant: it is `4d²` vs `8d²` whenever `d_ff = 4·d_model`. This toy has
`d_ff = 2·d_model`, so its split is 50/50 — the one ratio here that does **not** match real models.

---

## 17. Quick reference

```
1.  X    = E + PE                                   (L, d_model)
2.  Q,K,V = X @ Wq, X @ Wk, X @ Wv                  (L, d_model)  full width
3.  split → (n_heads, L, d_head)                    RESHAPE, not a projection
4.  per head: S = Q K^T / √d_head                   (n_heads, L, L)
5.  per head: A = softmax(S, dim=-1)                rows sum to 1
6.  per head: O = A @ V                             (n_heads, L, d_head)
7.  concat  → (L, d_model);  MHA = concat @ W_o     heads mix HERE and nowhere else
8.  h1 = LayerNorm(X + MHA)                         residual, then norm across features
9.  F  = ReLU(h1 @ W1) @ W2                         (L, d_ff) → (L, d_model)
10. out = LayerNorm(h1 + F)                         (L, d_model) — same shape in as out
```

**The five things to be able to say cold:**

1. Splitting into heads is a **reshape**, not extra projections — that is why multi-head is free.
2. Scaling is `√d_head`, **never** `√d_model`.
3. One head = one softmax = one probability budget. Two relationships must compete for it.
4. `W_o` is the *only* place heads interact. Without it they are parallel monologues.
5. The FFN holds ~⅔ of the parameters; attention is the ⅔ of the *compute* at long context.

---

## See also

- [06c_transformer_decoder_cross_attention_end_to_end.md](06c_transformer_decoder_cross_attention_end_to_end.md) — the decoder that reads this file's output as its cross-attention memory
- [06_transformer_end_to_end.md](06_transformer_end_to_end.md) — `d_model=2`, single head, full backward
- [05_attention_end_to_end.md](05_attention_end_to_end.md) — attention before the transformer block
- [../../5.transformers/02_models/08_modern_llm_architecture.md](../../5.transformers/02_models/08_modern_llm_architecture.md) — what changed since: RMSNorm, SwiGLU, RoPE, GQA
- [../../5.transformers/02_models/05_bert_end_to_end.md](../../5.transformers/02_models/05_bert_end_to_end.md) — stacking 12 of these encoders

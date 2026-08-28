# 06 — GPT-1: End-to-End with Weight Tying

> **This file is GPT-1 only** (Radford et al., 2018, *Improving Language Understanding by
> Generative Pre-Training*). GPT-2 is a separate file:
> [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md). Nothing here is mixed between the two.
>
> Arc: [06b encoder](../../4.nlp/03_sequence_models/06b_transformer_encoder_multihead.md) →
> [06c decoder](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md) →
> [05 BERT](05_bert_end_to_end.md) → **GPT-1** → [06b GPT-2](06b_gpt2_end_to_end.md).
> Same dimensions throughout: `d_model=4`, `n_heads=2`, `d_head=2`, `√d_k=1.414`, `d_ff=8`,
> same `Wq/Wk/Wv/W_o`.
>
> Every number verified in numpy and re-checked against `torch.autograd`
> (forward `allclose` to 6.7e-16; loss `2.178202 → 1.660667`, perplexity `8.8304 → 5.2628`).

---

## GPT-1 in one box

```
12 layers · d_model 768 · 12 heads · d_head 64 · d_ff 3072
vocabulary   40,478        BPE, 40k merges
context         512
positions    LEARNED table (512 × 768)
activation   GELU
LayerNorm    POST-LN     — LayerNorm(x + sublayer(x))
LM head      TIED to the token embedding:  P(u) = softmax(h_n Eᵀ)
parameters   116,534,784                  (papers round this to "117M")
pretraining  BooksCorpus, ~7,000 unpublished books
downstream   task-specific FINE-TUNING, with an auxiliary LM loss
```

---

## Table of Contents

1. What GPT-1 is, against what came before
2. Setup — the shift
3. Input = token + learned position
4. Weight setup — and `W_lm = Eᵀ`
5. Causal multi-head attention
6. Post-LN: Add & Norm, FFN, output
7. The LM head — weight tying, hand-computed
8. Loss at every position · perplexity
9. Backward — the weight-tying gradient identity
10. Weight update
11. Sampling — greedy, temperature, top-k, top-p
12. Where GPT-1's 116.5M parameters live
13. How GPT-1 was actually used — fine-tuning, not prompting
14. GPT-1 vs BERT vs the full decoder
15. Quick reference

---

## 1. What GPT-1 is

| | 06c decoder | **GPT-1** |
|---|---|---|
| Masked self-attention | ✓ | ✓ — *unchanged* |
| **Cross-attention** | ✓ | **✗ deleted** |
| Encoder / memory | required | **none** |
| Projection matrices / layer | 8 | **4** |
| Position encoding | sinusoidal | **learned table (512 max)** |
| Activation | ReLU | **GELU** |
| LayerNorm | post-LN | **post-LN** — same |
| LM head | separate `W_vocab` | **tied: `W_lm = Eᵀ`** |

Against BERT the difference is one line: **BERT deletes the causal mask, GPT-1 keeps it.**

```
        x
        │
   ┌────┴──────────────────┐
   │  MASKED self-attn     │   causal, exactly as 06c §5
   └────┬──────────────────┘
      Add, then LayerNorm            <-- POST-LN
        │
   ╔════╪══════════════════╗
   ║  (cross-attention)    ║   <-- 06c had a block here. GPT-1 does not.
   ╚════╪══════════════════╝
        │
   ┌────┴──────────────────┐
   │  FFN (GELU)           │
   └────┬──────────────────┘
      Add, then LayerNorm            <-- POST-LN
        │
   logits = out @ Eᵀ
```

---

## 2. Setup — the shift

```
Sequence:  bank  approved  the  loan          L = 4,  V = 8

position:    0       1      2     3
input:     bank  approved  the  loan
target:  approved   the   loan  <eos>         <- input shifted LEFT by one
```

Every position predicts the **next** token. Position 0 sees only `bank`; position 3 sees the whole
sentence.

06c used this shift for teacher forcing, but there the decoder input was a *separate* target
sequence. Here **input and target are the same string, offset by one** — no labels, no pairs, no
annotation. That is why causal LM scales: any text is training data.

### 2.1 Vocabulary

`bank`, `approved`, `the`, `loan` reuse 06b's vectors unchanged.

| Token | Index | Embedding | Note |
|---|---|---|---|
| `<bos>` | 0 | `[0.05, 0.05, 0.05, 0.05]` | constant vector — see §7.2 |
| bank | 1 | `[1.00, 0.80, 0.10, 0.10]` | 06b |
| approved | 2 | `[0.30, 0.20, 0.40, 0.30]` | 06b |
| the | 3 | `[0.10, 0.10, 0.20, 0.20]` | 06b |
| loan | 4 | `[0.10, 0.10, 1.00, 0.90]` | 06b |
| granted | 5 | `[0.20, 0.10, 0.50, 0.40]` | |
| rejected | 6 | `[0.40, 0.30, 0.20, 0.10]` | |
| `<eos>` | 7 | `[0.15, 0.25, 0.10, 0.30]` | |

---

## 3. Input = token + learned position

**Two** embeddings — there are no segment embeddings in GPT.

```
E_token                                  E_position (LEARNED)
[[1.0000, 0.8000, 0.1000, 0.1000],       [[ 0.2000,  0.1000, -0.1000,  0.3000],
 [0.3000, 0.2000, 0.4000, 0.3000],        [-0.1000,  0.3000,  0.4000,  0.1000],
 [0.1000, 0.1000, 0.2000, 0.2000],        [ 0.3000, -0.2000,  0.2000,  0.4000],
 [0.1000, 0.1000, 1.0000, 0.9000]]        [ 0.1000,  0.4000,  0.3000, -0.2000]]

X = E_token + E_position
             dim0     dim1     dim2     dim3
bank     [ 1.2000,  0.9000,  0.0000,  0.4000]
approved [ 0.2000,  0.5000,  0.8000,  0.4000]
the      [ 0.4000, -0.1000,  0.4000,  0.6000]
loan     [ 0.2000,  0.5000,  1.3000,  0.7000]
```

A **learned** table of shape `(512, 768)`. Same consequence as BERT: a hard context limit at 512,
because there is no row 513. (06b used sinusoidal, which has no such limit; RoPE replaced both —
board 13.)

---

## 4. Weight setup

`Wq`, `Wk`, `Wv`, `W_o` identical to 06b. **There is no `Wq_c/Wk_c/Wv_c/Wo_c` — that is the deleted
cross-attention block.**

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
                                                               [ 0.30,-0.20, 0.20, 0.10],
                                                               [ 0.10, 0.20, 0.30,-0.20],
                                                               [ 0.20, 0.10,-0.20, 0.30]]
```

### The LM head is not a new matrix

The GPT-1 paper writes the output layer as `P(u) = softmax(h_n Wₑᵀ)` — `Wₑ` is the **token
embedding matrix**. So:

```
W_lm = Eᵀ            (4 × 8)

        <bos>    bank  approved     the    loan  granted  rejected   <eos>
dim0 [ 0.0500, 1.0000,  0.3000, 0.1000, 0.1000,  0.2000,   0.4000, 0.1500]
dim1 [ 0.0500, 0.8000,  0.2000, 0.1000, 0.1000,  0.1000,   0.3000, 0.2500]
dim2 [ 0.0500, 0.1000,  0.4000, 0.2000, 1.0000,  0.5000,   0.2000, 0.1000]
dim3 [ 0.0500, 0.1000,  0.3000, 0.2000, 0.9000,  0.4000,   0.1000, 0.3000]
```

Column `w` **is** the embedding of word `w`, so `logit(w) = out · E[w]` — "how much does my final
state point toward this word?"

---

## 5. Causal multi-head attention

```
Q = X @ Wq                              K = X @ Wk
[[ 1.4400,  1.0800,  0.0000,  0.4800],  [[ 1.3800,  1.1400,  0.0800,  0.4000],
 [ 0.2400,  0.6000,  0.9600,  0.4800],   [ 0.3000,  0.5400,  0.8800,  0.5600],
 [ 0.4800, -0.1200,  0.4800,  0.7200],   [ 0.3800, -0.0200,  0.5200,  0.6800],
 [ 0.2400,  0.6000,  1.5600,  0.8400]]   [ 0.3000,  0.5400,  1.4400,  0.9600]]

V = X @ Wv
[[ 1.1700,  0.9300,  0.0400,  0.3600],
 [ 0.2300,  0.4700,  0.7600,  0.4400],
 [ 0.3500, -0.0500,  0.4200,  0.5800],
 [ 0.2300,  0.4700,  1.2400,  0.7600]]
```

Reshape into heads, `−∞` above the diagonal **before** softmax — mechanics and the reason it is
`−∞` rather than zeroing afterwards are in
[06c §5–6](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md).

```
masked scores head 0                    masked scores head 1
[[2.2758,   -inf,   -inf,   -inf],      [[0.1358,   -inf,   -inf,   -inf],
 [0.7179, 0.2800,   -inf,   -inf],       [0.1901, 0.7874,   -inf,   -inf],
 [0.3717, 0.0560, 0.1307,   -inf],       [0.2308, 0.5838, 0.5227,   -inf],
 [0.7179, 0.2800, 0.0560, 0.2800]]       [0.3258, 1.3033, 0.9775, 2.1587]]

A₀ = softmax                            A₁ = softmax
          bank  apprv    the   loan               bank  apprv    the   loan
bank  [ 1.0000, 0.0000, 0.0000, 0.0000]   [ 1.0000, 0.0000, 0.0000, 0.0000]
apprv [ 0.6077, 0.3923, 0.0000, 0.0000]   [ 0.3549, 0.6451, 0.0000, 0.0000]
the   [ 0.3976, 0.2900, 0.3124, 0.0000]   [ 0.2658, 0.3783, 0.3559, 0.0000]
loan  [ 0.3563, 0.2300, 0.1838, 0.2300]   [ 0.0845, 0.2247, 0.1622, 0.5285]

O₀                                      O₁
[[1.1700, 0.9300],                      [[0.0400, 0.3600],
 [0.8013, 0.7496],                       [0.5044, 0.4116],
 [0.6412, 0.4904],                       [0.4476, 0.4686],
 [0.5870, 0.5383]]                       [0.8977, 0.6251]]

concat = [O₀ ‖ O₁]                      MHA = concat @ W_o
[[1.1700, 0.9300, 0.0400, 0.3600],      [[1.1500, 0.9900, 0.1890, 0.4210],
 [0.8013, 0.7496, 0.5044, 0.4116],       [0.8466, 0.7959, 0.5753, 0.4958],
 [0.6412, 0.4904, 0.4476, 0.4686],       [0.6709, 0.5524, 0.5138, 0.5155],
 [0.5870, 0.5383, 0.8977, 0.6251]]       [0.6719, 0.6057, 0.9291, 0.7062]]
```

The heads disagree usefully at row `approved`: head 0 leans back to `bank` (0.6077), head 1 leans
to itself (0.6451).

---

## 6. Post-LN: Add & Norm, FFN, output

**GPT-1 normalises *after* the residual add** — `LayerNorm(x + sublayer(x))`, the 2017 ordering.

```
R1 = X + MHA                            per-row mean: [1.3125, 1.1534, 0.8882, 1.4032]
[[2.3500, 1.8900, 0.1890, 0.8210],      per-row var : [0.7284, 0.0368, 0.0689, 0.2632]
 [1.0466, 1.2959, 1.3753, 0.8958],
 [1.0709, 0.4524, 0.9138, 1.1155],      h1 = LayerNorm(R1)
 [0.8719, 1.1057, 2.2291, 1.4062]]      [[ 1.2156,  0.6766, -1.3164, -0.5759],
                                         [-0.5567,  0.7425,  1.1562, -1.3420],
                                         [ 0.6961, -1.6599,  0.0978,  0.8660],
                                         [-1.0356, -0.5799,  1.6097,  0.0058]]

Z = h1 @ W1
[[ 0.5804, -0.4021, -0.4827,  0.8336, -0.2458, -0.2807,  0.3511, -0.2660],
 [-0.5924, -0.0871,  0.3925, -0.7637,  0.6995, -0.2371, -0.2769,  0.8194],
 [ 0.1965, -0.2810,  0.3541,  0.2563, -0.7042,  0.7434, -0.3443, -0.5454],
 [-0.6894,  0.1964,  0.6557, -0.9540,  0.3545,  0.2894, -0.4947,  0.4693]]

GELU(Z)                                 GELU(x) = 0.5x(1 + erf(x/√2))
[[ 0.4174, -0.1382, -0.1519,  0.6650, -0.0990, -0.1093,  0.2237, -0.1051],
 [-0.1640, -0.0405,  0.2562, -0.1699,  0.5302, -0.0963, -0.1082,  0.6504],
 [ 0.1135, -0.1094,  0.2261,  0.1541, -0.1695,  0.5734, -0.1258, -0.1597],
 [-0.1691,  0.1135,  0.4878, -0.1622,  0.2264,  0.1776, -0.1536,  0.3194]]

FFN out = GELU(Z) @ W2                  R2 = h1 + FFN_out
[[ 0.1066, -0.0207,  0.3160,  0.1211],  [[ 1.3222,  0.6559, -1.0004, -0.4548],
 [ 0.0154,  0.2602, -0.2416,  0.1043],   [-0.5413,  1.0027,  0.9145, -1.2377],
 [ 0.2783, -0.1902,  0.1233,  0.1098],   [ 0.9744, -1.8501,  0.2211,  0.9758],
 [ 0.1415,  0.1659, -0.1724,  0.1276]]   [-0.8941, -0.4139,  1.4373,  0.1333]]

per-row mean: [0.1308, 0.0346, 0.0803, 0.0657]
per-row var : [0.8294, 0.9155, 1.3369, 0.7593]

GPT-1 OUTPUT = LayerNorm(R2)
             dim0     dim1     dim2     dim3
bank     [ 1.3082,  0.5767, -1.2420, -0.6429]
approved [-0.6018,  1.0118,  0.9197, -1.3297]
the      [ 0.7733, -1.6695,  0.1218,  0.7745]
loan     [-1.1014, -0.5504,  1.5741,  0.0777]
```

**There is no final LayerNorm after the stack** — post-LN already ends each block with one. (GPT-2
adds one; see [06b](06b_gpt2_end_to_end.md).)

Post-LN is also why GPT-1 needed **learning-rate warmup** — 2000 steps of linear warmup followed by
cosine decay, in the paper's own recipe. Every residual add is renormalised, so the path from the
loss back to the embeddings is squeezed through 24 LayerNorms in a 12-layer model.

---

## 7. The LM head — weight tying

```
logits = OUTPUT @ Eᵀ                    (4, 4) @ (4, 8) -> (4, 8)

           <bos>     bank  approved      the     loan  granted rejected    <eos>
bank    [ 0.0000,  1.5811, -0.1819, -0.1885, -1.6321, -0.5588,  0.3836,  0.0233]
apprvd  [ 0.0000,  0.1667, -0.0092, -0.0410, -0.2361, -0.0912,  0.1138, -0.1443]
the     [ 0.0000, -0.4727,  0.1791,  0.0896,  0.7292,  0.3584, -0.0898, -0.0569]
loan    [ 0.0000, -1.3766,  0.2125,  0.1652,  1.4789,  0.5428, -0.2831, -0.1221]

probs = softmax(logits, dim=-1)
           <bos>     bank  approved      the     loan  granted rejected    <eos>
bank    [ 0.0928,  0.4508,  0.0773,  0.0768,  0.0181,  0.0530,  0.1361,  0.0949]
apprvd  [ 0.1279,  0.1511,  0.1267,  0.1227,  0.1010,  0.1167,  0.1433,  0.1107]
the     [ 0.1078,  0.0672,  0.1289,  0.1179,  0.2235,  0.1543,  0.0985,  0.1018]
loan    [ 0.0876,  0.0221,  0.1083,  0.1033,  0.3844,  0.1507,  0.0660,  0.0775]
```

### 7.1 Verify one logit by hand

Row `the`, column `loan`. `OUTPUT[2] = [0.7733, -1.6695, 0.1218, 0.7745]`, `E[loan] = [0.1, 0.1, 1.0, 0.9]`:

```
0.7733×0.1 + (-1.6695)×0.1 + 0.1218×1.0 + 0.7745×0.9
= 0.07733 - 0.16695 + 0.1218 + 0.69705
= 0.7292   ✓
```

### 7.2 The `<bos>` column is exactly zero, everywhere

Not rounding — **exactly** `0.0000` at all four positions. `E[<bos>] = [0.05, 0.05, 0.05, 0.05]` is
constant, and `OUTPUT` is LayerNormed so every row has mean 0:

```
out · [c, c, c, c] = c · Σ outᵢ = c × 0 = 0
```

**Under weight tying, a token whose embedding is constant across dimensions can never receive a
non-zero logit** — whatever the network computes. A real consequence of tying + LayerNorm, and one
reason tied models keep a learnable output bias. (BERT's head is a separate matrix, so nothing like
this happens there.)

---

## 8. Loss at every position

```
pos  input      gold       p(gold)    greedy     loss
 0   bank       approved   0.077334   bank       2.559624
 1   approved   the        0.122728   bank       2.097782
 2   the        loan       0.223512   loan       1.498288
 3   loan       <eos>      0.077528   loan       2.557114

mean loss   = 2.178202
perplexity  = exp(2.178202) = 8.830415
```

**All four positions contribute.** One 4-token sequence produces 4 training signals.

```
GPT-1 : loss on 4 of 4 positions  = 100%
BERT  : loss on 15% of positions            -> 6.67x more signal per sequence for GPT
```

### Perplexity

```
PPL = exp(cross-entropy) = 8.830415
```

Read it as **"as confused as if choosing uniformly among 8.83 tokens."** With `V = 8`, uniform gives
`PPL = 8` exactly — so this untrained model is slightly *worse* than random, which is correct for an
untrained model.

Perplexity is comparable only across models with the **same tokenizer**: a model with bigger tokens
predicts fewer, easier units and scores lower for free.

---

## 9. Backward — the weight-tying gradient identity

Torch forward matched numpy to **6.7e-16** (`allclose`, atol 1e-6).

### 9.1 Magnitudes

```
                  max |grad|        L2
  dL/dEMB          0.675427      1.154689    <- largest
  dL/dPOS          0.523971      0.812914
  dL/dW_o          0.347900      0.721807
  dL/dW_v          0.304313      0.575374
  dL/dW2           0.060422      0.106425
  dL/dW1           0.042805      0.081765
  dL/dW_q          0.041354      0.077066
  dL/dW_k          0.027651      0.046366    <- smallest, 11x below W_v
```

Same ordering as 06b, 06c and BERT: `W_v`/`W_o` sit on a linear path and learn fastest; `W_q`/`W_k`
reach the loss only through the softmax and learn slowest.

### 9.2 The identity

Run the same forward with an **untied** head — a separate `W_lm` initialised to the same values. The
loss is identical (`2.178202`), but the gradient splits in two:

```
max | dL/dEMB_tied  −  ( dL/dEMB_untied  +  (dL/dW_lm_untied)ᵀ ) |  =  0.000e+00
```

**Exactly zero.** Tying does not approximate the two paths — it *sums* them.

```
 token       in X   target   |input path|   |output path|   |tied total|
 <bos>          -        -       0.000000        0.053611      0.053611
 bank         yes        -       0.223847        0.208298      0.425382
 approved     yes      yes       0.376697        0.532752      0.269151
 the          yes      yes       0.669416        0.472782      0.843382
 loan         yes      yes       0.143886        0.461746      0.346800
 granted        -        -       0.000000        0.090471      0.090471
 rejected       -        -       0.000000        0.058324      0.058324
 <eos>          -      yes       0.000000        0.483621      0.483621
```

Three things fall out:

1. **Tokens that never appear in the input still get gradient.** `granted` and `rejected` are neither
   input nor target, but they sit in the softmax denominator, so the output path pushes their
   embeddings *away*. With an **untied** head their embedding gradient would be exactly zero —
   compare BERT §12.2, where `the` and `granted` were zero for precisely that reason.
2. **`<eos>` is a target but never an input** — it learns entirely through the output path.
3. **The two paths can cancel.** For `approved`, which is both context (pos 1) and target (pos 0):

```
   input path   [ 0.2567,  0.0298, -0.1614, -0.2215]      norm 0.376697
   output path  [-0.3257, -0.1697,  0.3622,  0.1332]      norm 0.532752
   sum (tied)   [-0.0690, -0.1399,  0.2008, -0.0883]      norm 0.269151
```

The tied gradient is **smaller than either path alone** and points in a different direction from
both. One vector is being pulled one way as *context* and the other way as *a thing to predict*.
That tension is the real cost of tying — usually worth paying, since it removes `V × d_model`
parameters (§12).

---

## 10. Weight update

```
SGD, lr = 0.5

loss  2.178202 -> 1.660667     DECREASED by 0.517535   ✓
PPL   8.8304   -> 5.2628
```

```
    lr        loss       ppl    p(gold) per position
     0    2.178202    8.8304    [0.0773, 0.1227, 0.2235, 0.0775]
   0.1    1.952535    7.0465    [0.0841, 0.1481, 0.3713, 0.0877]
   0.3    1.770919    5.8762    [0.1021, 0.1870, 0.4022, 0.1092]
   0.5    1.660667    5.2628    [0.1292, 0.2300, 0.3400, 0.1290]
   1.0    1.703509    5.4932    [0.0939, 0.3580, 0.2007, 0.1628]   <- overshoots
```

```
greedy before : [bank, bank, loan, loan]      1 / 4
greedy after  : [bank, the,  loan, loan]      2 / 4
gold          : [approved, the, loan, <eos>]
```

At `lr = 1.0` the loss rises again and position 2's `p(gold)` collapses from 0.40 to 0.20 while
position 1's doubles — positions compete for shared weights.

---

## 11. Sampling

Position 3's row — the model has seen `bank approved the loan` and must emit the next token.

```
vocab :   <bos>     bank  approved      the     loan  granted rejected    <eos>
logits: [ 0.0000, -1.3766,  0.2125,  0.1652,  1.4789,  0.5428, -0.2831, -0.1221]
probs : [ 0.0876,  0.0221,  0.1083,  0.1033,  0.3844,  0.1507,  0.0660,  0.0775]
```

### 11.1 Greedy

```
argmax -> loan   (p = 0.3844)
```

Deterministic. Feed it back and the context becomes `... loan loan`, favouring `loan` again — the
repetition loop every greedy decoder falls into.

### 11.2 Temperature — `p = softmax(logits / T)`

```
    T    <bos>    bank approved     the    loan granted rejctd   <eos>     max   entropy
  0.5   0.0363  0.0023   0.0555  0.0505  0.6989  0.1075 0.0206  0.0284  0.6989   1.1170
  0.7   0.0637  0.0089   0.0862  0.0806  0.5264  0.1382 0.0425  0.0535  0.5264   1.5337
  1.0   0.0876  0.0221   0.1083  0.1033  0.3844  0.1507 0.0660  0.0775  0.3844   1.8033
  1.5   0.1049  0.0419   0.1208  0.1171  0.2811  0.1506 0.0868  0.0967  0.2811   1.9559
  2.0   0.1122  0.0564   0.1247  0.1218  0.2349  0.1471 0.0974  0.1055  0.2349   2.0099
  inf   0.1250  0.1250   0.1250  0.1250  0.1250  0.1250 0.1250  0.1250  0.1250   2.0794
```

`T` divides the logits **before** softmax. `T < 1` sharpens, `T > 1` flattens toward uniform;
entropy climbs monotonically to `log 8 = 2.0794`. `T → 0` is greedy. **Temperature never changes the
ranking** — only how much mass the leader keeps.

### 11.3 Top-k (k = 3)

```
kept: loan (1.4789), granted (0.5428), approved (0.2125)
probs: [0.0000, 0.0000, 0.1684, 0.0000, 0.5974, 0.2343, 0.0000, 0.0000]     sum = 1.0
```

`k` is fixed regardless of how peaked the distribution is — too permissive when one token deserves
0.99, too strict when fifty are plausible.

### 11.4 Top-p / nucleus (p = 0.9)

```
rank    token      prob   cumulative
   0     loan    0.3844       0.3844
   1  granted    0.1507       0.5351
   2 approved    0.1083       0.6434
   3      the    0.1033       0.7468
   4    <bos>    0.0876       0.8344
   5    <eos>    0.0775       0.9119   <- cutoff (first to reach 0.9)
   6 rejected    0.0660       0.9779
   7     bank    0.0221       1.0000

kept: loan, granted, approved, the, <bos>, <eos>   (6 tokens)
probs: [0.0961, 0.0000, 0.1188, 0.1133, 0.4215, 0.1653, 0.0000, 0.0850]    sum = 1.0
```

**The nucleus is 6 tokens wide here because this distribution is flat.** On a confident step it would
be 1–2. That adaptivity is why top-p replaced top-k as the default.

In practice: **temperature ≈ 0.7–0.9 with top-p ≈ 0.9**, in that order. Full treatment in
[../../4.nlp/03_sequence_models/07_decoding_strategies.md](../../4.nlp/03_sequence_models/07_decoding_strategies.md) — board 11.

> Note: GPT-1 itself was not used as a *generator* in the paper — it was pretrained then fine-tuned
> for classification-style tasks (§13). Sampling is the mechanism the whole family uses, and it is
> placed here because this is where logits first appear.

---

## 12. Where GPT-1's 116.5M parameters live

Exact accounting, including biases and LayerNorm gains/biases:

```
token embeddings      40478 × 768            =  31,087,104
position embeddings     512 × 768            =     393,216
                                                ──────────
                                                 31,480,320   (27.0%)

per layer:
  ln_1                 2 × 768               =       1,536
  attn c_attn      768 × 2304 + 2304         =   1,771,776
  attn c_proj       768 × 768 + 768          =     590,592
  ln_2                 2 × 768               =       1,536
  mlp  c_fc        768 × 3072 + 3072         =   2,362,368
  mlp  c_proj      3072 × 768 + 768          =   2,360,064
                                                ──────────
                                                   7,087,872
× 12 layers                                   =  85,054,464   (73.0%)
                                                ──────────
TOTAL                                         = 116,534,784
```

Matches the released checkpoint (HuggingFace `openai-gpt`) exactly. Papers round this to **"117M"**.

**No final LayerNorm** in the total — post-LN has none.

**Weight tying saves 31,087,104 parameters.** An untied GPT-1 would be 147,621,888, so tying removes
**21.1%** of the model.

Within a layer, the FFN (`4,722,432`) is **two-thirds** and attention (`2,362,368`) one-third — the
same 33/67 split as 06b §16 and BERT §16.

---

## 13. How GPT-1 was actually used

This is the part people forget, and it is the cleanest GPT-1-vs-GPT-2 distinction after post-LN.

**GPT-1 was fine-tuned per task.** Pretrain the LM on BooksCorpus, then for each downstream task add
a linear head and train on labelled data — with an **auxiliary LM loss** kept alongside the task
loss:

```
L_total = L_task + λ · L_LM          (the paper uses λ = 0.5)
```

Keeping the LM objective during fine-tuning improved generalisation and sped up convergence.

**Structured inputs were handled by traversal-style input transformations**, not by prompting: the
task's structure was flattened into one token sequence with delimiters, so the architecture never
changed.

```
entailment      [start] premise  [delim] hypothesis [extract]
similarity      [start] text-1   [delim] text-2     [extract]   (both orders, summed)
multiple choice [start] context  [delim] answer-i   [extract]   (one pass per choice)
```

The final `[extract]` token's hidden state feeds the linear head — the same role `[CLS]` plays in
BERT.

**GPT-2 abandoned all of this** for zero-shot prompting. That shift, not the architecture, is what
the GPT-2 paper was actually about — see [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md) §12.

---

## 14. GPT-1 vs BERT vs the full decoder

| | BERT | **GPT-1** | 06c decoder |
|---|---|---|---|
| Attention | bidirectional | **causal** | causal |
| Cross-attention | ✗ | **✗** | ✓ |
| Input embeddings | token + segment + position | **token + position** | token + position |
| LayerNorm | post-LN | **post-LN** | post-LN |
| LM head | separate `W_mlm` | **tied `Eᵀ`** | separate `W_vocab` |
| Objective | MLM (+ NSP) | **next token** | next token, conditioned |
| Loss positions | 15% | **100%** | all target positions |
| Needs paired data | ✗ | **✗** | ✓ source + target |
| Can generate | ✗ | **✓** | ✓ |
| Projections / layer | 4 | **4** | 8 |

**The interview trap:** *"GPT is decoder-only — so it has cross-attention, right?"* No.
"Decoder-only" names the **mask**, not the block count. A GPT layer has masked self-attention and an
FFN, and that is all — the same four projection matrices as a BERT layer. The cross-attention block
that made 06c a *decoder* in the encoder-decoder sense is exactly what "decoder-only" discards.

---

## 15. Quick reference

```
GPT-1 LAYER  (post-LN)

 1. X   = E_token + E_position                  (L, d)   TWO tables, both learned
 2. Q,K,V = X @ Wq, X @ Wk, X @ Wv
 3. split → (n_heads, L, d_head)
 4. S = Q Kᵀ / √d_head ;  S[i,j] = -inf for j > i        BEFORE softmax
 5. A = softmax(S) ; O = A @ V ; MHA = concat @ W_o
 6. h1  = LayerNorm(X + MHA)                    <- LN AFTER the add
 7. F   = GELU(h1 @ W1) @ W2                    NO cross-attention block
 8. out = LayerNorm(h1 + F)                     <- LN AFTER the add; no final LN
 9. logits = out @ Eᵀ                           TIED to the embedding table
10. loss = cross_entropy(logits, next_token)    EVERY position
```

**The seven things to be able to say cold:**

1. GPT-1 is 06c's decoder **minus the cross-attention block**. "Decoder-only" names the mask, not the
   block count — 4 projection matrices per layer, same as BERT.
2. Input and target are **the same string offset by one**. No labels — that is why it scales.
3. **`W_lm = Eᵀ`**: `logit(w) = out · E[w]`. Saves `V × d` — 21.1% of GPT-1.
4. Tying **sums two gradient paths** into one tensor, verified to `0.000e+00`. For a token that is
   both context and target they can partly cancel: `0.376697` and `0.532752` becoming `0.269151`.
5. Under tying, a token with a **constant embedding gets logit exactly 0** forever — a LayerNormed
   state is orthogonal to the all-ones direction.
6. Loss at **100%** of positions vs BERT's 15% — 6.67× more signal per sequence.
7. GPT-1 is **post-LN and needs LR warmup** (2000 steps), and it was **fine-tuned per task** with an
   auxiliary LM loss (`λ = 0.5`) — not prompted.

---

## See also

- [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md) — GPT-2: pre-LN, `1/√N` init, byte-level BPE, zero-shot
- [06c_gpt3_end_to_end.md](06c_gpt3_end_to_end.md) — GPT-3: sparse attention, the 8-model ladder, in-context learning
- [05_bert_end_to_end.md](05_bert_end_to_end.md) — the same block with the mask removed
- [../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md) — the cross-attention block GPT deletes, and the KV cache
- [../../4.nlp/03_sequence_models/07_decoding_strategies.md](../../4.nlp/03_sequence_models/07_decoding_strategies.md) — §11 in full: beam search, repetition penalties
- [02_gpt_family.md](02_gpt_family.md) — GPT-1 → 2 → 3 → 4, and what scaled

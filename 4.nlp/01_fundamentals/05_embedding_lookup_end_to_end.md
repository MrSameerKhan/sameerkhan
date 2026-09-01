# 05 — Tokens to Vectors: The Embedding Matrix

> Board 7, second half. [04_tokenization_end_to_end.md](04_tokenization_end_to_end.md) takes text to
> **token IDs** — BPE merges, WordPiece, SentencePiece, byte-level BPE, fertility. That file's BPE
> walkthrough reproduces exactly and needs no correction.
>
> This file takes **token IDs to vectors**: the embedding matrix, weight tying, and why vocabulary
> size is an economic decision rather than a modelling one.

---

## 1. Where this sits

```
"bank approved the loan"
        │  tokenizer  (04_tokenization_end_to_end.md)
        ▼
   [1, 2, 3, 4]                     token IDs — integers, no meaning yet
        │  embedding lookup          <- THIS FILE
        ▼
   [[1.00, 0.80, 0.10, 0.10],       (L, d_model) — the residual stream at depth 0
    [0.30, 0.20, 0.40, 0.30],
    [0.10, 0.10, 0.20, 0.20],
    [0.10, 0.10, 1.00, 0.90]]
        │  + position, then the stack
        ▼
```

**The embedding matrix `E` is `(V, d_model)`.** Row `i` is the vector for token `i`. That is the
entire data structure — there is nothing else to it.

---

## Table of Contents

1. Where this sits
2. The lookup is a matmul you do not do
3. The embedding matrix across the arc
4. Weight tying
5. Embedding gradients are sparse
6. Vocabulary size is an economic decision
7. Token count ≠ word count
8. Quick reference

---

## 2. The lookup is a matmul you do not do

Formally, embedding lookup is a one-hot vector times the matrix:

```
one-hot(4) @ E     with E of shape (8, 4)

  [0, 0, 0, 0, 1, 0, 0, 0] @ E   ->   row 4 of E
```

```
as a matmul:  8 × 4 = 32 multiply-adds, 7 of the 8 rows multiplied by ZERO
as a gather:  1 row read, 0 FLOPs
```

At Llama 3's dimensions that matters: `V = 128,256`, `d = 4096`, so the honest matmul would be
**525,336,576 multiply-adds per token** to retrieve one row. Every implementation does an indexed
read instead.

**Why the one-hot framing still matters:** it is why the backward pass is a *scatter-add* into `E`
(§5), and it is why the output projection is the transpose of the same operation — which is what
makes weight tying (§4) coherent rather than a coincidence.

---

## 3. The embedding matrix across the arc

Token embedding table only (`V × d`), against each model's exact total:

```
  model            vocab   d_model            V × d    tied?        total emb   % of model
  BERT-base       30,522       768       23,440,896     tied       23,440,896       21.6%
  GPT-1           40,478       768       31,087,104     tied       31,087,104       26.7%
  GPT-2 small     50,257       768       38,597,376     tied       38,597,376       31.0%
  T5-Base         32,128       768       24,674,304     tied       24,674,304       11.1%
  GPT-3 175B      50,257    12,288      617,558,016     tied      617,558,016        0.4%
  Llama-3 8B     128,256     4,096      525,336,576   UNtied    1,050,673,152       13.1%
```

*(Token embeddings only. [06b §14](../../5.transformers/02_models/06b_gpt2_end_to_end.md) quotes
31.6% for GPT-2 small because it also counts the 786,432 position embeddings.)*

**The proportion collapses with scale.** GPT-2 small spends **31.0%** of itself on the embedding
table; GPT-3 175B spends **0.4%** — the same 50,257-row vocabulary against 96 layers of `d=12288`.

That single row explains a lot of design history:

- **Small models must tie.** At 31% of parameters, a second copy for the output head is unaffordable.
- **Large models need not care.** GPT-3 could untie for free and it would not show up.
- **T5-Base is lowest at 11.1%** because it has *two* stacks (encoder + decoder) amortising one
  embedding table.
- **Llama 3 pays 13.1% and unties anyway** — a 128k vocabulary *and* a separate head. It buys that
  back in fertility (§7). Llama 3.2's 1B and 3B do tie, because at that size the table would
  dominate again.

---

## 4. Weight tying

Set the output projection to the transpose of the input table:

```
input :  x = E[token_id]                 pick a row
output:  logits = h @ Eᵀ                 dot h against EVERY row
         logit(w) = h · E[w]             "how much does h point toward word w?"
```

**One matrix, two directions.** The saving:

```
GPT-1        tied  116,534,784      untied would be 147,621,888   tying saves 21.1%
GPT-2 small  tied  124,439,808      untied would be 163,037,184   tying saves 23.7%
Llama-3 8B  UNtied   8,030,261,248  tying would save 525,336,576  =  6.5%
```

**Llama 3 declines a 6.5% saving.** At 8B the extra 525M is affordable, and untying lets the input
and output roles specialise — a token's "what I mean as context" vector need not equal its "what I
look like as a prediction target" vector. §9.2 of
[06_gpt1_end_to_end.md](../../5.transformers/02_models/06_gpt1_end_to_end.md) shows those two roles
producing gradients that **partially cancel** when tied, which is precisely the tension untying
removes.

Three consequences of tying worth knowing:

1. **The gradient sums two paths** — input and output — verified exactly in
   [06 §9.2](../../5.transformers/02_models/06_gpt1_end_to_end.md).
2. **Tokens never seen in the batch still get gradient**, because they appear in the softmax
   denominator. Untied, they get exactly zero.
3. **A constant embedding row can never receive a non-zero logit**, since the hidden state is
   LayerNormed and therefore orthogonal to the all-ones direction
   ([06 §7.2](../../5.transformers/02_models/06_gpt1_end_to_end.md)).

T5 goes further and ties **three** ways — encoder input, decoder input, output head — which is why
it needs a `d_model^-0.5` rescale before the head
([07 §7](../../5.transformers/02_models/07_t5_end_to_end.md)).

---

## 5. Embedding gradients are sparse

Backward through a gather is a **scatter-add**: only the rows actually looked up receive gradient.

```
BERT §12.2 (untied head):   of 8 vocabulary rows, 3 received EXACTLY zero
                            — they appeared in neither the input nor the targets
```

At real scale, a batch touches perhaps a few thousand of 128,256 rows. Two practical consequences:

- **Embedding gradients are stored sparsely**, or the optimizer state for an unused row is skipped.
- **Rare tokens train slowly** — a token appearing once per million documents gets one update per
  million documents. This is the mechanism behind "glitch tokens": vocabulary entries that survive
  tokenizer training but almost never appear in the *model's* training data, leaving their
  embeddings near-random and their behaviour undefined.

---

## 6. Vocabulary size is an economic decision

```
larger vocabulary
  +  fewer tokens per document       -> less attention compute, smaller KV cache, lower bill
  +  better coverage of code, non-English, rare words
  −  more embedding parameters       (V × d, doubled if untied)
  −  a larger, slower final softmax over V
  −  rarer tokens per entry -> each row trains on less data
```

The arc's trajectory is one-directional:

```
BERT      30,522    WordPiece
GPT-1     40,478    BPE
T5        32,128    SentencePiece (+100 sentinels, padded to 32,128)
GPT-2     50,257    byte-level BPE  = 256 bytes + 50,000 merges + 1
Llama 2   32,000    SentencePiece
Llama 3  128,256    tiktoken BPE
```

**Llama 2 → Llama 3 quadrupled the vocabulary**, costing `4×` the embedding table (and doubly, since
Llama 3 unties). Meta judged the fertility win worth ~800M parameters
([08b §7](../../5.transformers/02_models/08b_llama3_end_to_end.md)).

**Byte-level BPE has a distinct property that is not about size:** with all 256 byte values in the
base vocabulary, **`<UNK>` becomes impossible** — any Unicode string decomposes into tokens the
model already has ([06b §12](../../5.transformers/02_models/06b_gpt2_end_to_end.md)). That is a
robustness guarantee, not a compression one.

---

## 7. Token count ≠ word count

The board's second killer question.

```
cost     = tokens × price_per_token
tokens   = words  × FERTILITY
```

**Fertility is a property of the tokenizer, not of the model's quality.** The same document, priced
at the same rate, costs different amounts through different tokenizers — because it becomes a
different number of tokens.

Three places this bites, all of them real:

1. **Billing.** A 4× larger vocabulary packs more characters per token, so the same text is fewer
   tokens. Comparing per-token prices across model families without comparing fertility is
   comparing nothing.
2. **Context limits.** "128k context" is 128k *tokens*. In a language with high fertility that is far
   less text than in English — the same 8k-token budget might hold a third as much Hindi or Thai as
   English. This is a genuine equity problem, not a curiosity.
3. **Latency.** Generation cost is per token ([04b](../../5.transformers/02_models/04b_attention_at_scale_end_to_end.md)),
   so a high-fertility tokenizer makes the *same answer* slower and dearer.

`04_tokenization_end_to_end.md` §"The Fertility Problem" measures this; the point here is that
fertility, embedding cost and serving cost are the same decision seen from three sides.

---

## 8. Quick reference

```
E is (V, d_model).  x = E[token_id]        a gather, not a matmul
tied head:  logits = h @ E^T               logit(w) = h . E[w]
backward :  scatter-add — only rows present in the batch get gradient

embedding share of the model:  GPT-2 small 31.0%   GPT-3 175B 0.4%
tying saves:  GPT-1 21.1%   GPT-2 23.7%   (Llama 3 declines 6.5% and unties)
cost = words x FERTILITY x price_per_token
```

**The seven things to be able to say cold:**

1. **The embedding matrix is `(V, d_model)` and lookup is a gather.** Written as `one-hot @ E` it
   would be `525,336,576` multiply-adds per token at Llama 3's size, almost all against zeros.
2. **The one-hot framing still matters** — it is why backward is a scatter-add, and why the tied
   output head is the same matrix transposed.
3. **Embedding share collapses with scale:** `31.0%` of GPT-2 small, `0.4%` of GPT-3 175B, on the
   *same* 50,257 vocabulary. Small models must tie; large ones need not.
4. **Tying saves ~21–24%** at GPT-1/GPT-2 scale. **Llama 3 declines a 6.5% saving** so the input and
   output roles can specialise — the roles whose gradients partly cancel when tied.
5. **Only rows present in the batch get gradient.** Rare tokens train slowly, which is the mechanism
   behind glitch tokens.
6. **Bigger vocabulary trades parameters for fertility** — and byte-level BPE separately makes
   `<UNK>` impossible, which is a robustness property, not a size one.
7. **`cost = words × fertility × price`.** Fertility belongs to the tokenizer, so per-token prices
   are not comparable across families — and "128k context" holds materially less text in a
   high-fertility language.

---

## See also

- [04_tokenization_end_to_end.md](04_tokenization_end_to_end.md) — BPE merges, WordPiece, SentencePiece, fertility (verified: its BPE walkthrough reproduces exactly)
- [03_tokenization.md](03_tokenization.md) — the concept file
- [../../5.transformers/02_models/06_gpt1_end_to_end.md](../../5.transformers/02_models/06_gpt1_end_to_end.md) — weight tying's gradient identity, verified to `0.000e+00`
- [../../5.transformers/02_models/06b_gpt2_end_to_end.md](../../5.transformers/02_models/06b_gpt2_end_to_end.md) — byte-level BPE, `256 + 50,000 + 1 = 50,257`
- [../../5.transformers/02_models/08b_llama3_end_to_end.md](../../5.transformers/02_models/08b_llama3_end_to_end.md) — the 128,256 vocabulary and why Llama 3 unties
- [../../5.transformers/02_models/07_t5_end_to_end.md](../../5.transformers/02_models/07_t5_end_to_end.md) — three-way tying and the `d_model^-0.5` rescale

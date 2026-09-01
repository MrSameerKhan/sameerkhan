# 07b — BART: Denoising Autoencoding

> **This file is BART only** (Lewis et al., 2020, *BART: Denoising Sequence-to-Sequence Pre-training
> for Natural Language Generation, Translation, and Comprehension*). T5 is a separate file:
> [07_t5_end_to_end.md](07_t5_end_to_end.md). Nothing here is mixed between them.
>
> **Read this first.** BART's *architecture* is the standard 2017 encoder-decoder with a
> BERT-style bidirectional encoder and a GPT-style causal decoder — hand-computed in
> [06c](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md) and
> [05](05_bert_end_to_end.md), and not repeated here. Where T5 changed the *block*
> (relative position bias, RMSNorm, no biases, no `√d_k`), **BART changed almost nothing**.
>
> What BART contributed is the **pretraining objective** — five corruption transformations, of which
> two won. That, and the T5 comparison, are what this file covers.

---

## BART in one box

```
architecture ENCODER-DECODER — bidirectional encoder + causal decoder + cross-attention
encoder      BERT-like (bidirectional, no mask)
decoder      GPT-like (causal) + cross-attention
vocabulary   50,265        GPT-2's byte-level BPE + a few extras
positions    LEARNED ABSOLUTE  (max 1024)   — like BERT/GPT-2, NOT relative
norm         LayerNorm, POST-norm, plus an extra LayerNorm on the embeddings
biases       YES — standard linear layers with bias
activation   GELU
attn scale   standard 1/sqrt(d_k)
embeddings   shared: encoder input, decoder input, output head
pretraining  DENOISING — corrupt the document, reconstruct the WHOLE original
sizes        BART-base 139,420,416   ·   BART-large 406,291,456
```

**Compare that box to T5's.** Almost every line differs, and the architecture line is the one that
does not.

---

## Table of Contents

1. The five corruption transformations
2. Text infilling — the one that mattered
3. **BART vs T5** — the comparison to have ready
4. Why the target is the whole document
5. Architecture — what BART did *not* change
6. Where BART's parameters live
7. Fine-tuning BART
8. Quick reference

---

## 1. The five corruption transformations

BART corrupts the input with an arbitrary noising function and trains the decoder to reconstruct the
original. The paper evaluates five:

```
original             A B C . D E .

token masking        A _ C . D E .        BERT's MLM: replace single tokens with [MASK]
token deletion       A C . D E .          delete tokens; model must infer WHERE
text infilling       A _ . D _ E .        replace a SPAN with ONE [MASK] (span may be length 0)
sentence permutation D E . A B C .        shuffle sentences
document rotation    C . D E . A B        rotate to start at a random token
```

Each teaches something different:

| Transformation | What the model must learn |
|---|---|
| token masking | what token is missing |
| token deletion | **that** something is missing, and where |
| text infilling | **how many** tokens are missing, and what they are |
| sentence permutation | document-level ordering |
| document rotation | where the document begins |

**The paper's finding: text infilling + sentence permutation performed best.** Token deletion and
rotation alone underperformed.

---

## 2. Text infilling — the one that mattered

```
original          bank approved the loan .
                       ^^^^^^^^  span of length 1 sampled

corrupted         bank [mask] the loan .
target            bank approved the loan .        <- the ENTIRE original
```

Span lengths are drawn from a **Poisson distribution with λ = 3**, so a span can be length 0 — which
means `[mask]` is *inserted* without deleting anything. Two consequences:

1. **The model cannot count masks to know the answer length.** One `[mask]` might stand for 0, 1, or
   7 tokens. BERT's MLM always had exactly one token per `[MASK]`, which is a large hint.
2. **Length prediction becomes part of the task.** That is precisely what a generation model needs
   and what MLM never teaches.

This is the closest thing BART has to T5's span corruption, and §3 is where they part company.

---

## 3. BART vs T5 — the comparison to have ready

Both are encoder-decoders pretrained by corrupting text. Almost everything else differs.

| | **BART** | **T5** |
|---|---|---|
| Corruption | text infilling + sentence permutation | span corruption |
| Mask token | one shared `[mask]` | **numbered sentinels** `<X>`, `<Y>`, `<Z>`… |
| **Target** | **the entire original document** | **only the dropped spans** |
| Target length | ≈ input length | ≈ 15% of input length |
| Positions | learned absolute (1024) | relative position bias |
| Norm | LayerNorm, **post**-norm | RMSNorm, **pre**-norm |
| Biases | yes | none |
| Activation | GELU | ReLU (v1.0) |
| Attention scale | `1/√d_k` | none |
| Vocabulary | 50,265 byte-level BPE | 32,128 SentencePiece |
| Task framing | task-specific heads for classification | **everything is text-to-text** |
| Best at | summarisation, generation | transfer across many tasks |

### 3.1 The single most important row

```
input      bank [mask] the loan .

BART target:   bank approved the loan .          <- reconstruct EVERYTHING
T5   target:   <X> approved </s>                 <- emit ONLY what was removed
```

**BART's decoder pays full sequence length on every example; T5's pays ~15%.** T5 gets far more
gradient signal per unit of decoder compute, which is a real efficiency argument in T5's favour.

**BART's counter-argument:** reconstructing the full document is exactly the shape of
summarisation and translation — read a document, emit a document. That is why BART-large-CNN
remained a strong summarisation baseline long after T5 existed, and why the two models split by
task rather than one dominating.

Numbered sentinels matter too: T5's `<X>`, `<Y>`, `<Z>` let the decoder tell *which* gap it is
filling, so multiple spans stay unambiguous in a single flat target. BART's one shared `[mask]`
has no such handle — but it does not need one, because the target is the whole document in order.

---

## 4. Why the target is the whole document

BART is a **denoising autoencoder**: corrupt `x` into `x̃`, and maximise `log P(x | x̃)`. The
"autoencoder" part is literal — the output space is the input space.

That makes fine-tuning trivially natural for seq2seq tasks:

```
pretraining     corrupted document  -> original document
summarisation   full article        -> summary
translation     source sentence     -> target sentence
```

All three are "document in, document out". There is no gap between the pretraining objective and the
downstream task — no sentinels to strip, no format to learn. **T5 closes the same gap differently**,
by reframing every task as text-to-text with a task prefix.

---

## 5. Architecture — what BART did *not* change

Explicitly, so you do not credit BART with things it inherited:

- **Bidirectional encoder** — BERT ([05_bert_end_to_end.md](05_bert_end_to_end.md))
- **Causal decoder + cross-attention** — the 2017 paper, hand-computed in [06c](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md)
- **Learned absolute positions** — BERT and GPT
- **GELU** — BERT and GPT
- **Byte-level BPE** — GPT-2 ([06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md)); BART reuses the tokenizer
- **Weight tying** — GPT-1 ([06_gpt1_end_to_end.md](06_gpt1_end_to_end.md))

The one small architectural addition: **an extra LayerNorm applied to the embeddings** at the input
of each stack (`layernorm_embedding`), which T5 and GPT-2 do not have.

**BART-large uses post-LN.** By 2019 GPT-2 had already moved to pre-LN
([06b §5](06b_gpt2_end_to_end.md)) and T5 was pre-norm; BART stayed post-norm. That is a genuine
difference, and it is why BART is more sensitive to learning rate and warmup than T5 of comparable
size.

> **The one-line summary interviewers want:** *"BART is BERT's encoder and GPT's decoder glued
> together with cross-attention, pretrained by corrupting a document and reconstructing all of it."*
> That sentence is accurate and complete at the architecture level.

---

## 6. Where BART's parameters live

```
model         d_model   d_ff   layers (enc+dec)          parameters   reported
BART-base         768   3072        6 + 6              139,420,416     139M
BART-large       1024   4096       12 + 12             406,291,456     406M
```

Both match the released checkpoints (`facebook/bart-base`, `facebook/bart-large`).

For scale against the siblings in this arc:

```
BART-base      139,420,416      6 enc + 6 dec
BART-large     406,291,456     12 enc + 12 dec
T5-Base        222,903,552     12 enc + 12 dec       (07_t5_end_to_end.md §11)
BERT-base      108,770,304     12 enc                (05_bert_end_to_end.md §16)
GPT-2 small    124,439,808     12 dec                (06b_gpt2_end_to_end.md §14)
```

**BART-large is roughly BERT-large's encoder plus GPT-2-medium's decoder**, which is close to how
the paper describes it: about 10% more parameters than BERT-large for the same `d_model` and depth.

---

## 7. Fine-tuning BART

Four patterns, and the classification one is the non-obvious one:

```
SEQ2SEQ (summarisation, translation)
  encoder <- source document      decoder <- generates the target       no new parameters

CLASSIFICATION
  feed the SAME document to encoder AND decoder,
  take the decoder's FINAL token hidden state, add a linear head
  ^ the final token is used because it is the only position that has attended to the whole document

TOKEN CLASSIFICATION (NER, span extraction)
  decoder's per-token hidden states -> per-token head

TRANSLATION into a new language
  replace the encoder's embedding layer with a fresh randomly-initialised encoder,
  train it in two stages while most of BART stays frozen
```

The classification trick is worth remembering: BART has no `[CLS]`. It uses the **last decoder
token** as the sentence representation, for the same reason GPT does — under a causal mask, only
the final position has seen everything. BERT's `[CLS]` works only because its encoder is
bidirectional ([05 §9](05_bert_end_to_end.md)).

---

## 8. Quick reference

```
BART = BERT encoder + GPT decoder + cross-attention
       post-LN · LayerNorm · biases · GELU · learned absolute positions · 1/sqrt(d_k)
       + an extra LayerNorm on the embeddings

pretraining:  x  --noise-->  x~   then maximise  log P(x | x~)
              best noise = text infilling (Poisson lambda=3 spans -> one [mask]) + sentence permutation
              target = the ENTIRE original document
```

**The seven things to be able to say cold:**

1. BART is a **denoising autoencoder**: corrupt the document, reconstruct **all** of it. The target
   is the whole original, not just the missing pieces.
2. **Text infilling + sentence permutation** was the winning combination of the five transformations.
3. Text infilling draws span lengths from **Poisson λ=3, including length 0** — so one `[mask]` may
   stand for zero tokens, and the model must predict *how many* are missing. MLM never teaches this.
4. **BART vs T5 in one line:** BART reconstructs everything with one shared `[mask]`; T5 emits only
   the dropped spans, each tagged with a numbered sentinel. T5's decoder does ~15% of the work.
5. Architecturally BART is **conservative** — learned absolute positions, LayerNorm, biases, GELU,
   `1/√d_k`. Everything T5 changed, BART kept. Its only addition is a LayerNorm on the embeddings.
6. **BART-large is post-LN**, when GPT-2 and T5 had already moved to pre-norm — hence more
   sensitivity to LR and warmup.
7. BART has **no `[CLS]`**. For classification it feeds the document to both stacks and reads the
   **final decoder token**, because under a causal mask that is the only position that has seen
   everything.

---

## See also

- [07_t5_end_to_end.md](07_t5_end_to_end.md) — T5: relative position bias, RMSNorm, span corruption, fully hand-computed
- [../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md) — the cross-attention and causal mask BART inherits
- [05_bert_end_to_end.md](05_bert_end_to_end.md) — BART's encoder half
- [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md) — BART's decoder half, and the tokenizer it borrows
- [03_encoder_decoder.md](03_encoder_decoder.md) — family overview: when to reach for BART vs Flan-T5
- [../../9.multimodal/05_donut_end_to_end.md](../../9.multimodal/05_donut_end_to_end.md) — Donut's decoder is BART's, with a Swin encoder supplying the memory

# 06c — GPT-3: Scale, Sparse Attention, In-Context Learning

> **This file is GPT-3 only** (Brown et al., 2020, *Language Models are Few-Shot Learners*).
> GPT-1: [06_gpt1_end_to_end.md](06_gpt1_end_to_end.md) · GPT-2:
> [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md). Nothing here is mixed between them.
>
> **Read this first.** The GPT-3 paper says, in its own words, that it uses *"the same model and
> architecture as GPT-2, including the modified initialization, pre-normalization, and reversible
> tokenization described therein, with the exception that we use alternating dense and locally
> banded sparse attention patterns in the layers of the transformer."*
>
> So the block-level forward pass **is** GPT-2's — pre-LN, `ln_f`, `1/√N` init, GELU, learned
> positions, tied head, byte-level BPE. All of it is hand-computed in
> [06b](06b_gpt2_end_to_end.md) and is not repeated here. This file covers **only what is
> actually different**: the sparse attention pattern, the scale, and in-context learning.

---

## GPT-3 in one box

```
sizes        8 models, 125M -> 175B      largest: 96 layers · d_model 12288 · 96 heads · d_head 128
vocabulary   50,257        byte-level BPE — UNCHANGED from GPT-2
context      2,048         (GPT-2: 1024)
positions    LEARNED table (2048 × d)
LayerNorm    pre-LN + ln_f            — unchanged from GPT-2
init         residual × 1/sqrt(N)     — unchanged from GPT-2
activation   GELU                     — unchanged
LM head      tied                     — unchanged
NEW          alternating DENSE and LOCALLY BANDED SPARSE attention layers
parameters   174,604,259,328 for the 175B model
training     300B tokens · 3.14e23 FLOPs
downstream   IN-CONTEXT LEARNING — zero / one / few-shot, no gradient updates
```

---

## Table of Contents

1. What actually changed from GPT-2
2. The eight models — and two errata in the paper's table
3. **Sparse attention** — masks, hand-computed
4. Why alternate dense and sparse
5. **In-context learning** — the mechanism, computed
6. Zero-shot vs one-shot vs few-shot
7. Why 175B — compute and scaling laws
8. The serving reality — weights and KV cache
9. What GPT-3 did *not* change
10. Quick reference

---

## 1. What actually changed from GPT-2

| | GPT-2 | **GPT-3** |
|---|---|---|
| Block structure | pre-LN, GELU, tied head | **identical** |
| `1/√N` residual init | ✓ | **identical** |
| Tokenizer | byte-level BPE, 50,257 | **identical** |
| Attention pattern | dense causal, every layer | **alternating dense / locally banded sparse** |
| Context | 1,024 | **2,048** |
| Largest model | 1.5B | **175B** |
| Use | zero-shot prompting | **few-shot in-context learning** |

**Three real changes: sparse attention, context length, and scale.** Everything else is GPT-2.

That is a short list, and it is the correct answer to *"what's new in GPT-3 architecturally?"* — the
paper's contribution was the demonstration that few-shot ability emerges from scale, not a new block
design. Saying "GPT-3 introduced a new architecture" is wrong.

---

## 2. The eight models

The paper trained eight sizes to measure how capability scales. Parameter counts below are computed
exactly (embeddings + positions + per-layer attention/FFN/LayerNorm, including all biases) with
`V = 50,257`, `ctx = 2048`, `d_ff = 4·d_model`:

| Model | layers | `d_model` | heads | `d_head` | computed params | paper |
|---|---|---|---|---|---|---|
| Small | 12 | 768 | 12 | 64 | 125,226,240 | 125M |
| Medium | 24 | 1024 | 16 | 64 | 355,871,744 | 350M |
| Large | 24 | 1536 | 16 | 96 | 760,300,032 | 760M |
| XL | 24 | 2048 | 24 | 128 | 1,315,723,264 | 1.3B |
| 2.7B | 32 | 2560 | 32 | 80 | 2,651,553,280 | 2.7B |
| 6.7B | 32 | 4096 | 32 | 128 | 6,658,404,352 | 6.7B |
| 13B | 40 | 5120 | 40 | 128 | 12,853,386,240 | 13.0B |
| **175B** | **96** | **12288** | **96** | **128** | **174,604,259,328** | 175.0B |

`d_head` is pinned at **64–128 across a 1400× range of model size** — the same stability noted in
06b §16. Width and head count scale together; the per-head subspace does not.

### 2.1 Two errata in the paper's Table 2.1 — worth knowing

`n_heads × d_head` should equal `d_model`. In two rows it does not:

```
  Small    768   12 ×  64 =   768   ok
  Medium  1024   16 ×  64 =  1024   ok
  Large   1536   16 ×  96 =  1536   ok
  XL      2048   24 × 128 =  3072   MISMATCH
  2.7B    2560   32 ×  80 =  2560   ok
  6.7B    4096   32 × 128 =  4096   ok
  13B     5140   40 × 128 =  5120   MISMATCH   (table lists d_model = 5140)
  175B   12288   96 × 128 = 12288   ok
```

- **13B**: `5140` is universally read as a typo for **5120**, which makes `40 × 128` consistent and
  makes the parameter count land on 13.0B. The table above uses 5120.
- **XL**: `2048` with `d_head = 128` implies **16** heads, not 24. The paper's head count is the
  suspect entry. Nothing downstream depends on it here, but do not quote `24 × 128` as if it works.

Knowing these is a cheap signal that you read the table rather than a summary of it.

---

## 3. Sparse attention — the one architectural change

GPT-3 alternates **dense causal** layers with **locally banded sparse** layers. A banded layer lets
each query see only itself and the previous `w − 1` positions.

Worked on 8 tokens — `bank approved the loan granted rejected the loan` — with band width `w = 4`:

```
dense causal                            locally banded causal (w = 4)
  1 . . . . . . .                         1 . . . . . . .
  1 1 . . . . . .                         1 1 . . . . . .
  1 1 1 . . . . .                         1 1 1 . . . . .
  1 1 1 1 . . . .                         1 1 1 1 . . . .
  1 1 1 1 1 . . .                         . 1 1 1 1 . . .
  1 1 1 1 1 1 . .                         . . 1 1 1 1 . .
  1 1 1 1 1 1 1 .                         . . . 1 1 1 1 .
  1 1 1 1 1 1 1 1                         . . . . 1 1 1 1

  36 allowed of 64                        26 allowed of 64   (72.2% of dense)
```

Both are still **causal** — the band only removes *old* keys, never future ones. Sparse attention
does not weaken the causal guarantee.

### 3.1 What it does to one attention row

Head 0, query at position 7 (the second `loan`):

```
tokens :  bank  approved   the    loan  granted rejected  the    loan
dense  : [0.1278, 0.1318, 0.1128, 0.1318, 0.1263, 0.1242, 0.1130, 0.1323]   sum 1.0
banded : [0.0000, 0.0000, 0.0000, 0.0000, 0.2548, 0.2505, 0.2278, 0.2668]   sum 1.0
```

The banded row is **structurally incapable** of placing mass on positions 0–3. The `0.5042` the
dense row spent there is redistributed across the four positions inside the band, roughly doubling
each. Softmax still normalises to 1 — the row is a valid distribution, just over a smaller support.

### 3.2 The complexity win

Dense causal attention costs `L(L+1)/2` score computations. Banded costs `Σ min(i+1, w)`:

```
       L        w        dense        banded    reduction
       8      256           36            36        1.0x     <- w >= L, no saving
    1024      256      524,800       229,504        2.3x
    2048      256    2,098,176       491,648        4.3x
    4096      256    8,390,656     1,015,936        8.3x
```

Dense grows as `O(L²)`, banded as `O(L·w)` — **linear** in sequence length once `L > w`. At GPT-3's
own 2048 context the saving is 4.3×, on the sparse half of the layers. The saving grows without
bound as context grows, which is why every long-context method since is some variant of this idea
(board 14).

---

## 4. Why *alternate* rather than go fully sparse

A banded layer moves information at most `w − 1` positions. Stacking them propagates:

```
  1 banded layer   ->  reach   3 positions   (w-1, with w=4)
  2 banded layers  ->  reach   6
  4 banded layers  ->  reach  12
  8 banded layers  ->  reach  24
```

Reach grows **linearly with depth** — so an all-sparse 96-layer model with a narrow band could not
connect the ends of a 2048-token context in a usable number of hops, and the paths that do exist are
long and lossy.

**One dense layer reaches the entire context in a single hop.** Alternating gives you both: the
sparse layers carry most of the sequence-length cost, and every other layer restores global reach in
one step. It is a compute/reach trade made at the layer level rather than inside the attention op.

> **Be careful how you state this.** The paper says *"alternating dense and locally banded sparse
> attention patterns... similar to the Sparse Transformer"* and does not publish the band width, the
> exact alternation, or which layers are which. Describe the *mechanism* and cite the Sparse
> Transformer (Child et al., 2019) for the patterns; do not invent specific hyperparameters for
> GPT-3. The `w = 4` above is a teaching value, not GPT-3's.

---

## 5. In-context learning — the mechanism

This is GPT-3's actual contribution, and the mechanism is simple enough to verify exactly.

**Claim:** the model adapts to a task from examples placed in the prompt, with **no gradient
updates and no weight changes**. The demonstration is that the weights are bit-identical between two
runs that produce different predictions.

```
zero-shot prompt :  the loan
one-shot prompt  :  the loan granted | the loan          demonstration: "the loan -> granted"
```

```
weight sha256 before : 91ebd8e0142013c367a86eebdc7a443c55789478
weight sha256 after  : 91ebd8e0142013c367a86eebdc7a443c55789478
identical            : True          <- ZERO gradient steps between the two runs
```

```
vocab:  <bos>  bank  approved  the  loan  granted  rejected  <eos>

ZERO-SHOT  final-position probs
  [0.0814, 0.0163, 0.0973, 0.0987, 0.4276, 0.1432, 0.0545, 0.0810]   p(granted) = 0.143182

ONE-SHOT   final-position probs
  [0.0798, 0.0158, 0.0965, 0.0973, 0.4341, 0.1435, 0.0532, 0.0799]   p(granted) = 0.143453

  delta p(granted)   = +0.000270
  max |prob change|  =  0.006545
```

**The prediction changed and no weight moved.** That is in-context learning, exactly — the prompt
enters through the *activations*, not the parameters. Everything the demonstration contributes
arrives through attention over the prompt tokens.

### 5.1 The honest limit of this toy

The shift is `+0.000270`. That is **the mechanism without the capability**, and the distinction
matters:

```
final-row attention, one-shot run
  head 0: [0.0610, 0.3529, 0.1743, 0.0392, 0.3727]
  head 1: [0.0497, 0.3346, 0.2246, 0.0398, 0.3513]
  tokens: [ the,    loan,  granted,  the,   loan ]
```

Position 2 is `granted` — the token that followed `the loan` the first time. A trained **induction
head** would put most of its mass there, copying the completion forward. This toy puts `0.1743` and
`0.2246` — barely above the `0.20` a flat row would give.

**Induction heads are learned, and they appear only with scale and training.** That is precisely the
GPT-3 paper's finding: few-shot performance is near-flat for the small models and rises sharply with
size. A one-block untrained model can show you the *plumbing* — prompt in, activations changed,
weights untouched — and cannot show you the behaviour. Claiming otherwise from a toy is the trap.

---

## 6. Zero-shot vs one-shot vs few-shot

All three are the same forward pass over a longer prompt. **None of them updates a weight.**

```
ZERO-SHOT   Translate English to French:
            cheese =>

ONE-SHOT    Translate English to French:
            sea otter => loutre de mer
            cheese =>

FEW-SHOT    Translate English to French:
            sea otter => loutre de mer
            peppermint => menthe poivree
            plush girafe => girafe peluche
            ... (K examples)
            cheese =>
```

| | examples | gradient steps | weights change |
|---|---|---|---|
| zero-shot | 0 | 0 | ✗ |
| one-shot | 1 | 0 | ✗ |
| few-shot | K = 10–100 | 0 | ✗ |
| **fine-tuning** | many | **many** | **✓** |

`K` is bounded by the context, not by the method — every example must fit in 2048 tokens alongside
the query:

```
  K =  20 examples of ~102 tokens each fills the 2048 context
  K =  50 examples of ~ 40 tokens each
  K = 100 examples of ~ 20 tokens each
```

That is the entire reason context length became a headline number after GPT-3: **context is the
few-shot budget.**

**The contrast with GPT-1** is the cleanest way to hold the family in mind:

```
GPT-1   pretrain -> fine-tune per task, with a task head and auxiliary LM loss
GPT-2   pretrain -> prompt, zero-shot, no head
GPT-3   pretrain -> prompt with K examples in the context, still no head, still no gradients
```

---

## 7. Why 175B — compute

The standard estimate for transformer training compute is:

```
C ≈ 6 · N · D          N = parameters, D = training tokens
                       (2 FLOPs per multiply-add forward, ×3 for forward+backward)
```

For GPT-3 175B, `D = 300B` tokens:

```
C = 6 × 174,604,259,328 × 300e9 = 3.1429e23 FLOPs
```

**The paper reports 3.14e23.** It matches only if you use the *computed* 174.6B rather than the
rounded "175B" — the rounded figure gives `3.15e23`. A small thing, but it confirms the formula and
the parameter count are both right.

The eight-model ladder exists to trace loss and few-shot accuracy against `N`. The finding —
smooth, predictable improvement across orders of magnitude — is the scaling-laws result
(Kaplan et al., 2020), and it is what justified spending that compute on one model. Board 17.

> **Chinchilla, two years later, showed GPT-3 was badly under-trained** for its size: 300B tokens
> for 175B parameters is far below the ~20 tokens/parameter that is compute-optimal. GPT-3 at
> compute-optimal would have been a much smaller model on much more data. Say this if scaling comes
> up — it is the single most important correction to the GPT-3 paper.

---

## 8. The serving reality

```
weights, fp16 :  175e9 × 2 bytes                       = 326.0 GiB
KV cache, fp16:  2 × 96 layers × 2048 tokens × 12288   =   9.00 GiB   per sequence at full context
```

Two consequences worth being able to state:

1. **The weights do not fit on one accelerator.** 326 GiB at fp16 needs ~4.1 × 80 GB A100s for
   weights alone — before activations, before any KV cache. GPT-3 inference is inherently
   multi-GPU, which is why tensor/pipeline parallelism became standard.
2. **9 GiB of KV cache per sequence** means batching is memory-bound, not compute-bound. Serve 8
   concurrent full-context sequences and the cache alone is 72 GiB — an entire A100. This is the
   pressure that produced MQA/GQA, PagedAttention and quantised caches (boards 12–13).

---

## 9. What GPT-3 did *not* change

Explicitly, so you do not credit it with things it inherited:

- **pre-LN and `ln_f`** — GPT-2 ([06b §5, §7](06b_gpt2_end_to_end.md))
- **`1/√N` residual init** — GPT-2 ([06b §11](06b_gpt2_end_to_end.md))
- **byte-level BPE, 50,257** — GPT-2 ([06b §12](06b_gpt2_end_to_end.md)), unchanged, not even resized
- **weight tying** — GPT-1 ([06 §7, §9.2](06_gpt1_end_to_end.md))
- **GELU, learned positions, loss at every position** — GPT-1
- **No cross-attention, no encoder** — GPT-1

And things GPT-3 did **not** introduce that people sometimes attribute to it: RLHF and instruction
tuning (that is InstructGPT, 2022 — board 20), chat formatting, RoPE, RMSNorm, SwiGLU, GQA, MoE.

---

## 10. Quick reference

```
GPT-3 = GPT-2's block, with:
  1. attention alternating DENSE  causal   (full L x L lower triangle)
                      and BANDED causal   (each query sees w previous positions)
  2. context 2048 (GPT-2: 1024)
  3. scale to 96 layers / d_model 12288 / 96 heads / d_head 128
  everything else identical to GPT-2
```

**The seven things to be able to say cold:**

1. GPT-3's architecture is **GPT-2's**, in the paper's own words. The only block-level change is
   **alternating dense and locally banded sparse attention**. It did not introduce a new block.
2. Banded attention is still **causal** — it removes old keys, never future ones. Cost goes
   `O(L²) → O(L·w)`; at L=4096, w=256 that is **8.3×** fewer scores.
3. **Alternate** rather than go fully sparse: a banded layer moves information `w−1` positions, so
   reach grows only linearly with depth. One dense layer restores global reach in a single hop.
4. **In-context learning changes activations, not parameters.** Verified here: identical weight
   hash, different prediction. Zero, one and few-shot are the same forward pass on a longer prompt.
5. **Few-shot ability emerges with scale.** A small or untrained model shows the plumbing and not the
   behaviour — the mechanism needs trained induction heads. `K` is bounded by the context, which is
   why context length became a headline number.
6. **175B = 174,604,259,328** exactly; `C = 6ND = 3.14e23` FLOPs on 300B tokens — and it reconciles
   only with the unrounded parameter count.
7. GPT-3 was **under-trained** by Chinchilla's rule (300B tokens for 175B params, against ~20:1
   compute-optimal). And GPT-3 ≠ InstructGPT: **no RLHF, no instruction tuning** in this paper.

---

## See also

- [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md) — the block GPT-3 reuses, hand-computed in full
- [06_gpt1_end_to_end.md](06_gpt1_end_to_end.md) — weight tying, sampling, the causal LM objective
- [04_efficient_transformers.md](04_efficient_transformers.md) — sparse attention in general, Flash Attention, KV cache
- [11_long_context_scaling.md](11_long_context_scaling.md) — what replaced banded attention: RoPE scaling, ALiBi, sliding window
- [../../4.nlp/03_sequence_models/08_scaling_laws_emergent.md](../../4.nlp/03_sequence_models/08_scaling_laws_emergent.md) — Kaplan, Chinchilla, and what "emergent" does and does not mean
- [../../6.llms/01_prompting.md](../../6.llms/01_prompting.md) — few-shot prompting as practice
- [02_gpt_family.md](02_gpt_family.md) — GPT-1 → 2 → 3 → 4 overview

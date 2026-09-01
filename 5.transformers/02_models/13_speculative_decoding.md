# 13 — Speculative Decoding

> Board 16, and the last of the architecture stage. Speculative decoding is **provably lossless** —
> it produces exactly the target model's distribution — and this file proves it three ways rather
> than deferring to the paper.
>
> Depends on [04b_attention_at_scale_end_to_end.md](04b_attention_at_scale_end_to_end.md): the whole
> technique only makes sense because decode is memory-bound.

---

## 1. Why it works at all

From board 12: generating one token reads **all** the weights and does ~1 FLOP per byte, against an
A100 ridge point of 153. The GPU is idle, waiting on HBM.

**So verifying `K` tokens costs almost the same as verifying 1.** The weight read is amortised;
the extra FLOPs are free because you were never compute-limited.

```
sequential decode   K tokens -> K target forward passes   (K weight reads)
speculative         K tokens -> 1 target forward pass     (1 weight read)
                               + K cheap draft passes
```

That asymmetry — **generation is sequential and memory-bound, verification is parallel and nearly
free** — is the entire opportunity. Speculative decoding does *more* total FLOPs and is *faster*.

---

## Table of Contents

1. Why it works at all
2. The algorithm
3. **The lossless proof** — verified three ways
4. Acceptance rate
5. Expected tokens ≠ speedup
6. Choosing K
7. Variants
8. When it fails
9. Quick reference

---

## 2. The algorithm

```
1. DRAFT   small model q proposes K tokens autoregressively   (K cheap passes)
2. VERIFY  large model p scores ALL K positions in ONE pass
3. ACCEPT  for t = 1..K, in order:
             accept x_t with probability  min(1, p(x_t)/q(x_t))
             on the first rejection, STOP and emit one token from the residual:
                            norm( max(0, p − q) )
4. If all K accepted, emit a free bonus token sampled from p
```

Two details people get wrong:

- **Rejection stops the round.** Tokens after a rejection are discarded even if they would have been
  accepted — the prefix they were conditioned on is now wrong.
- **A rejected round still emits one token.** It never stalls; worst case is one token per round,
  which is plain autoregressive decoding.

---

## 3. The lossless proof

Target `p` is the GPT-1 distribution from [06_gpt1_end_to_end.md §11](06_gpt1_end_to_end.md);
draft `q` is a different, cheaper model over the same vocabulary.

```
vocab:    <bos>     bank  approved      the     loan  granted rejected    <eos>
p     : [0.0876,  0.0221,  0.1083,  0.1033,  0.3844,  0.1507,  0.0660,  0.0775]
q     : [0.1503,  0.0286,  0.0795,  0.1623,  0.2704,  0.1078,  0.1094,  0.0917]
```

The draft **under**-weights three tokens (`q < p`): `approved`, `loan`, `granted`.

```
max(0, p − q) = [0, 0, 0.0288, 0, 0.1139, 0.0430, 0, 0]
Z = Σ max(0, p − q) = 0.185705

residual = max(0, p−q)/Z = [0, 0, 0.1552, 0, 0.6135, 0.2314, 0, 0]
```

**The residual is supported exactly on the tokens the draft under-weights.** That is the intuition:
rejection is the mechanism for injecting the probability mass the draft failed to propose.

### Proof 1 — elementwise algebra

```
P(emit x) = min(q(x), p(x))  +  P(reject) · residual(x)

and  P(reject) = 1 − Σ min(q,p) = Z,  so  P(reject)·residual(x) = max(0, p(x) − q(x))

therefore  P(emit x) = min(q,p) + max(0, p−q)

  where q ≥ p :  min = p,  max(0,p−q) = 0      ->  p
  where q < p :  min = q,  max(0,p−q) = p−q    ->  q + (p−q) = p
```

**Identically `p(x)`, in both cases.** Verified elementwise: `max |min(q,p) + max(0,p−q) − p| = 0.000e+00`.

### Proof 2 — the closed form, numerically

```
P(emit) = [0.0876, 0.0221, 0.1083, 0.1033, 0.3844, 0.1507, 0.0660, 0.0775]
target p= [0.0876, 0.0221, 0.1083, 0.1033, 0.3844, 0.1507, 0.0660, 0.0775]

max |P(emit) − p| = 0.000e+00
```

### Proof 3 — Monte Carlo, 10,000,000 draws

```
empirical: [0.0875, 0.0221, 0.1083, 0.1033, 0.3844, 0.1508, 0.0660, 0.0775]
target p : [0.0876, 0.0221, 0.1083, 0.1033, 0.3844, 0.1507, 0.0660, 0.0775]

max |empirical − p| = 8.64e-05        sampling noise ~ 1/√N = 3.16e-04
```

Within noise, as it must be.

> **What "lossless" does and does not mean.** The *distribution* is exact — speculative decoding is
> not an approximation and needs no quality evaluation. But it is **not bit-identical** to a given
> greedy run: it is sampling, so a specific seeded output differs. Under greedy decoding (`T=0`)
> it *is* bit-identical, because accept reduces to "did the draft pick the argmax".

---

## 4. Acceptance rate

```
α = Σ min(q(x), p(x))          the overlap between the two distributions

  predicted  0.814295
  observed   0.814360          (10M draws)
```

**The acceptance rate is exactly the overlap of the two distributions** — the total variation
distance subtracted from 1. That is why draft choice matters so much: a draft from the *same family
and tokenizer* as the target overlaps far more than a generically "good" small model.

It also explains a practical rule: **acceptance is higher on easy, predictable text** (boilerplate,
code, formatting) and lower on genuinely uncertain continuations. Speculative decoding speeds up
exactly the parts that were easy anyway.

---

## 5. Expected tokens ≠ speedup

Expected tokens accepted per round, with acceptance `α` and `K` speculated:

```
E(α, K) = (1 − α^(K+1)) / (1 − α)
```

```
    α    K=1      K=2      K=3      K=4      K=5      K=8
  0.3  1.3000   1.3900   1.4170   1.4251   1.4275   1.4285
  0.5  1.5000   1.7500   1.8750   1.9375   1.9688   1.9961
  0.7  1.7000   2.1900   2.5330   2.7731   2.9412   3.1988
  0.8  1.8000   2.4400   2.9520   3.3616   3.6893   4.3289
  0.9  1.9000   2.7100   3.4390   4.0951   4.6856   6.1258
```

**But `E` is not the speedup.** Each round also costs `K` draft forward passes. With `c` = draft
cost / target cost:

```
speedup = E(α, K) / (1 + K·c)
```

```
draft = 5% of target cost (c = 0.05)
    α    K=1      K=2      K=3      K=4      K=5      K=8
  0.5  1.4286   1.5909  *1.6304   1.6146   1.5750   1.4258
  0.7  1.6190   1.9909   2.2026   2.3109  *2.3529   2.2849
  0.8  1.7143   2.2182   2.5670   2.8013   2.9514  *3.0921
  0.9  1.8095   2.4636   2.9904   3.4126   3.7485  *4.3756

draft = 20% of target cost (c = 0.20)
    α    K=1      K=2      K=3      K=4      K=5      K=8
  0.5 *1.2500   1.2500   1.1719   1.0764   0.9844   0.7677
  0.7  1.4167   1.5643  *1.5831   1.5406   1.4706   1.2303
  0.8  1.5000   1.7429   1.8450  *1.8676   1.8446   1.6650
  0.9  1.5833   1.9357   2.1494   2.2751   2.3428  *2.3561
                                                (* = best K for that row)
```

**Read the bottom-left cell: `α=0.5, c=0.2, K=8` gives `0.7677` — a slowdown.** Speculating far
ahead with a mediocre draft and an expensive draft model is strictly worse than not speculating.

---

## 6. Choosing K

The tables above show the structure:

- **`E` saturates in `K`.** At `α = 0.5`, going from `K=5` to `K=8` buys `1.9688 → 1.9961` — you
  cannot exceed `1/(1−α) = 2` however far you speculate, because the chance of surviving `K`
  accepts decays geometrically.
- **Cost is linear in `K`.** So there is always a finite optimum, and it moves right as `α` rises
  and left as `c` rises.
- At `c = 0.05`: best `K` is 3 → 5 → 8 → 8 as `α` goes 0.5 → 0.9.
- At `c = 0.20`: best `K` is 1 → 3 → 4 → 8 over the same range.

**`K = 4–5` is the usual default** because it is near-optimal across the plausible middle of that
grid. Production systems increasingly adapt `K` online from the recent acceptance rate.

---

## 7. Variants

| Variant | Where the draft comes from | Notes |
|---|---|---|
| **Standard** | a smaller model of the same family | Llama-3-1B drafting for -70B; needs the **same tokenizer** |
| **Self-speculative** | the target's own early layers | no second model to host |
| **Medusa** | extra heads on the target predicting `t+1, t+2, …` | one model, tree-structured candidates |
| **EAGLE** | a light head over the target's *features*, not tokens | higher acceptance than Medusa |
| **Prompt lookup / n-gram** | copy from the prompt itself | **zero** draft model; superb for summarisation, RAG, code edit |

**Prompt lookup is the one to remember for practical work.** When the output is expected to quote
the input heavily — summarising a document, editing code, answering from retrieved context — the
"draft" is just the next few tokens after a matching n-gram in the prompt. No model, no memory, and
acceptance is high precisely because the task is extractive.

---

## 8. When it fails

- **Low acceptance.** A draft with a different tokenizer, or a much weaker one — the overlap `α`
  collapses and §5 turns negative.
- **Large batch.** At high batch the target is already compute-bound
  ([04b §0](04b_attention_at_scale_end_to_end.md)), so verification is no longer nearly free and the
  premise of §1 disappears. **Speculative decoding is a low-batch, latency-oriented technique.**
- **Draft model memory.** The draft occupies GPU memory that could have held KV cache — a real cost
  at scale, invisible in a latency benchmark.
- **High temperature.** Sampling at `T = 1.2` flattens `p`, `α` falls, speculation gets less useful
  exactly when generation is most exploratory.

---

## 9. Quick reference

```
draft  q proposes K tokens        target p verifies all K in ONE pass
accept x with prob min(1, p(x)/q(x))
reject -> emit from norm(max(0, p - q)) and stop the round
all accepted -> free bonus token from p

LOSSLESS   min(q,p) + max(0,p-q) = p           exactly, elementwise
alpha      = sum min(q,p)                       = distribution overlap
E(a,K)     = (1 - a^(K+1)) / (1 - a)            expected tokens per round
speedup    = E(a,K) / (1 + K*c)                 c = draft cost ratio
```

**The seven things to be able to say cold:**

1. **It works because decode is memory-bound.** Verifying K tokens costs almost the same as
   verifying one — the weight read is amortised. More total FLOPs, less wall time.
2. **It is provably lossless**, via `min(q,p) + max(0, p−q) = p`. Verified elementwise
   (`0.000e+00`) and by 10M-draw Monte Carlo (`8.64e-05`, inside `3.16e-04` noise).
3. **The residual `norm(max(0, p−q))` is the mechanism** — it is supported exactly on the tokens the
   draft under-weights, injecting the mass the draft failed to propose. Omit it and the output is
   biased toward the draft.
4. **Acceptance rate α = Σ min(q,p)** — literally the overlap of the two distributions. Here
   `0.814295` predicted, `0.814360` observed. Same-family drafts overlap more.
5. **Expected tokens is not speedup.** `E(α,K)/(1 + K·c)`. At `α=0.5, c=0.2, K=8` the "speedup" is
   `0.7677` — a **slowdown**.
6. **`E` saturates at `1/(1−α)`** however large `K` gets, while cost grows linearly — so an optimal
   `K` always exists. `K=4–5` is the usual default.
7. **It is a low-batch technique.** At high batch the target is already compute-bound and the free
   lunch is gone.

---

## See also

- [04b_attention_at_scale_end_to_end.md](04b_attention_at_scale_end_to_end.md) — why decode is memory-bound, the premise of §1
- [../../4.nlp/03_sequence_models/07_decoding_strategies.md](../../4.nlp/03_sequence_models/07_decoding_strategies.md) — board 11: the sampling this preserves exactly
- [06_gpt1_end_to_end.md](06_gpt1_end_to_end.md) — the target distribution used in §3
- [../../6.llms/05_vllm_internals.md](../../6.llms/05_vllm_internals.md) — speculative decoding in a real serving stack
- [12_constrained_decoding.md](12_constrained_decoding.md) — the other way to change what gets emitted

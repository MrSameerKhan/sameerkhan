# 07 — Decoding Strategies: End-to-End

> Board 11. Every number below is computed on **one complete 8-token distribution** — the real
> output of the GPT-1 walkthrough in
> [../../5.transformers/02_models/06_gpt1_end_to_end.md](../../5.transformers/02_models/06_gpt1_end_to_end.md) §11,
> position 3. Because the vocabulary is complete, **every probability row sums to exactly 1** and
> every claim here is checkable by hand.
>
> Beam search and the length penalty are run through the actual model, autoregressively.

---

## The distribution everything below uses

Context: `bank approved the loan`. The model must emit the next token.

```
vocab :   <bos>     bank  approved      the     loan  granted rejected    <eos>
logits: [ 0.0000, -1.3766,  0.2125,  0.1652,  1.4789,  0.5428, -0.2831, -0.1221]
probs : [ 0.0876,  0.0221,  0.1083,  0.1033,  0.3844,  0.1507,  0.0660,  0.0775]

sum = 1.000000        entropy = 1.803281 nats        uniform would be log 8 = 2.0794
```

---

## Table of Contents

1. The core problem
2. Greedy
3. Temperature
4. Top-k
5. Top-p (nucleus)
6. min-p
7. Typical sampling
8. Repetition penalty
9. **Beam search** — run for real
10. **Length penalty** — and the exact crossover
11. Which to use
12. Comparison table
13. Quick reference

---

## 1. The core problem

The model gives you a distribution over the vocabulary at every step. Decoding is the question of
what to *do* with it — and it is entirely separate from the model. **No decoding strategy changes a
single weight.** Same model, different decoder, different text.

Two failure modes bound the space:

```
too deterministic  ->  repetition, generic "safe" text, degenerate loops
too random         ->  incoherence, hallucinated tokens, drift
```

Everything below is a way of cutting between them.

---

## 2. Greedy

Take the argmax, every step.

```
argmax -> loan   (p = 0.3844)
```

Deterministic, zero-temperature, no randomness. Its failure mode is not theoretical — generate from
this very model and watch it:

```
greedy from 'bank'   ->  bank bank bank bank
```

**It locks on.** Feeding the argmax back makes the same token more likely next step, and the loop
closes. Compare three sampled runs at `T = 1.0` from the same start:

```
seed 0  ->  bank the  bank  <bos>
seed 1  ->  bank bank <eos> bank
seed 2  ->  bank bank bank  rejected
```

**Use greedy when there is one right answer** — classification, extraction, structured output — and
essentially never for open-ended text.

---

## 3. Temperature

Divide the logits **before** softmax: `p = softmax(logits / T)`.

```
    T    <bos>    bank approved     the    loan granted rejctd   <eos>     sum   entropy
 0.25   0.0026  0.0000   0.0061  0.0050  0.9611  0.0227 0.0008  0.0016  1.0000   0.2134
 0.50   0.0363  0.0023   0.0555  0.0505  0.6989  0.1075 0.0206  0.0284  1.0000   1.1170
 0.70   0.0637  0.0089   0.0862  0.0806  0.5264  0.1382 0.0425  0.0535  1.0000   1.5337
 1.00   0.0876  0.0221   0.1083  0.1033  0.3844  0.1507 0.0660  0.0775  1.0000   1.8033
 1.50   0.1049  0.0419   0.1208  0.1171  0.2811  0.1506 0.0868  0.0967  1.0000   1.9559
 2.00   0.1122  0.0564   0.1247  0.1218  0.2349  0.1471 0.0974  0.1055  1.0000   2.0099
```

Three things to read off this table:

- **Entropy moves monotonically** from `0.2134` to `2.0099`, ceiling `log 8 = 2.0794`.
- `T → 0` approaches greedy; at `T = 0.25` the leader already holds `0.9611`.
- **Temperature never changes the ranking.** `loan` is first at every `T`. It only changes how much
  mass the leader keeps. That is the key distinction from truncation methods (§4–§7), which
  *remove* tokens.

---

## 4. Top-k

Keep the `k` highest logits, set the rest to `−∞`, renormalise.

```
k=1  loan                                   [0.0000 … 1.0000 …]              = greedy
k=2  loan, granted                          loan 0.7183   granted 0.2817
k=3  loan, granted, approved                loan 0.5974   granted 0.2343   approved 0.1684
k=5  + the, <bos>                           loan 0.4607   granted 0.1807   approved 0.1298
                                            the  0.1238   <bos>   0.1050
```

**The weakness is that `k` is fixed while the distribution is not.** When one token deserves 0.99,
`k=50` admits 49 tokens that should be impossible. When fifty are genuinely plausible, `k=5` throws
away good ones. Top-p fixes exactly this.

---

## 5. Top-p (nucleus)

Sort descending, keep the shortest prefix whose cumulative mass reaches `p`.

```
rank  token      prob   cumulative
   0  loan     0.3844      0.3844
   1  granted  0.1507      0.5351     <- p=0.5 cutoff
   2  approved 0.1083      0.6434
   3  the      0.1033      0.7468     <- p=0.7 cutoff
   4  <bos>    0.0876      0.8344
   5  <eos>    0.0775      0.9119     <- p=0.9 cutoff
   6  rejected 0.0660      0.9779     <- p=0.95 cutoff
   7  bank     0.0221      1.0000
```

```
p=0.50   2 tokens   loan 0.7183  granted 0.2817
p=0.70   4 tokens   loan 0.5147  granted 0.2019  approved 0.1451  the 0.1384
p=0.90   6 tokens   loan 0.4215  granted 0.1653  approved 0.1188  the 0.1133  <bos> 0.0961  <eos> 0.0850
p=0.95   7 tokens   loan 0.3931  granted 0.1541  approved 0.1108  the 0.1057  <bos> 0.0896  <eos> 0.0793  rejected 0.0675
```

**The nucleus is 6 tokens wide at `p=0.9` here because this distribution is flat** (entropy 1.80 of
a possible 2.08). On a confident step it would be 1–2 tokens. That adaptivity is why top-p replaced
top-k as the default.

---

## 6. min-p

Keep every token with `p ≥ min_p × p_max`. Threshold scales with the *leader*, not with a rank or a
cumulative mass.

```
p_max = 0.3844

min_p=0.05   threshold 0.019218   ->  8 tokens  (everything survives)
min_p=0.10   threshold 0.038437   ->  7 tokens  (drops 'bank' at 0.0221)
min_p=0.20   threshold 0.076873   ->  6 tokens  (also drops 'rejected' at 0.0660)
```

**On a confident step min-p becomes very aggressive automatically.** If the leader held `0.95`, then
`min_p=0.1` would require `0.095` — cutting almost everything — while top-p at 0.9 would keep only
the leader anyway and top-k at 50 would keep 49 junk tokens. min-p is the 2023-era answer to
"one knob that behaves sensibly at both extremes".

---

## 7. Typical sampling

Keep tokens whose **surprisal `−log p` is closest to the distribution's entropy `H`**, then take
enough of them to reach mass `τ`.

```
H = 1.803281 nats

    token        p    -log p   |surprisal - H|
  granted   0.1507    1.8922            0.0889   <- most "typical"
 approved   0.1083    2.2226            0.4193
      the   0.1033    2.2698            0.4666
    <bos>   0.0876    2.4350            0.6317
    <eos>   0.0775    2.5571            0.7538
     loan   0.3844    0.9562            0.8471   <- the MODE, and it is atypical
 rejected   0.0660    2.7181            0.9148
     bank   0.0221    3.8116            2.0083   <- most surprising
```

```
tau=0.5   keeps 5:  granted, approved, the, <bos>, <eos>     <- 'loan' EXCLUDED
tau=0.9   keeps 6:  the above plus loan
```

**At `τ = 0.5` typical sampling drops `loan` — the most probable token in the distribution.**
No amount of top-k or top-p tuning can do that; both are prefix-of-sorted-order methods and always
keep the mode. Typical trims from *both* ends: too-predictable and too-surprising.

The motivation is information-theoretic — human text sits near the model's entropy rather than at
its mode, which is also why greedy reads flat and repetitive.

---

## 8. Repetition penalty

Divide the logit of any already-seen token by `θ` (or multiply, if the logit is negative — the sign
rule matters and is easy to get wrong).

```
if logit > 0:  logit /= theta          both cases push the logit DOWN
else:          logit *= theta
```

Context already contains `the` and `loan`:

```
theta=1.0  logits  [0.0000, -1.3766, 0.2125, 0.1652, 1.4789, 0.5428, -0.2831, -0.1221]
           probs   [0.0876,  0.0221, 0.1083, 0.1033, 0.3844, 0.1507,  0.0660,  0.0775]   p(loan) 0.3844

theta=1.1  logits  [0.0000, -1.3766, 0.2125, 0.1502, 1.3444, 0.5428, -0.2831, -0.1221]
           probs   [0.0922,  0.0233, 0.1140, 0.1071, 0.3537, 0.1587,  0.0695,  0.0816]   p(loan) 0.3537

theta=1.5  logits  [0.0000, -1.3766, 0.2125, 0.1101, 0.9859, 0.5428, -0.2831, -0.1221]
           probs   [0.1037,  0.0262, 0.1282, 0.1157, 0.2779, 0.1784,  0.0781,  0.0918]   p(loan) 0.2779
```

`p(loan)` falls `0.3844 → 0.3537 → 0.2779`, and every unseen token's probability rises to absorb it.

**Greedy still picks `loan` at all three settings.** A repetition penalty biases; it does not
forbid. To actually forbid, use `no_repeat_ngram_size`, which hard-masks any token completing a
repeated n-gram — effective, but it will also block legitimate repeats like a name recurring in a
summary.

---

## 9. Beam search

Keep the `B` best *sequences* by cumulative log-probability instead of the best single token. Run
for real from `bank`, `B = 3`, through the model:

```
step 1: 8 candidates -> keep top 3
   bank bank                          cum logprob  -0.796673    P = 0.450826
   bank rejected                      cum logprob  -1.994152    P = 0.136129
   bank <eos>                         cum logprob  -2.354425    P = 0.094948
   (best discarded: bank <bos>        -2.377757)

step 2: 24 candidates -> keep top 3
   bank bank bank                     cum logprob  -1.612540    P = 0.199381
   bank bank rejected                 cum logprob  -2.746881    P = 0.064128
   bank rejected bank                 cum logprob  -2.999969    P = 0.049789
   (best discarded: bank bank <bos>   -3.150799)

step 3: 24 candidates -> keep top 3
   bank bank bank bank                cum logprob  -2.486812    P = 0.083175
   bank bank bank rejected            cum logprob  -3.563370    P = 0.028343
   bank rejected bank bank            cum logprob  -3.907871    P = 0.020083
   (best discarded: bank bank rejected bank   -3.908375)
```

`B` beams × `V` tokens = `3 × 8 = 24` candidates per step, of which 3 survive. Cost is `B×` the
forward passes of greedy.

**Beam search is not a search for the most likely token — it is a search for the most likely
sequence,** and those differ. A token that is second-best now can lead to a far better continuation.
Greedy cannot recover from that; beam search can, up to `B` hypotheses deep.

**Where to use it:** translation, summarisation, ASR — tasks with a *correct* answer. `B = 4–6` is
typical; beyond ~10 gains vanish.

**Where not to:** open-ended generation. Beam search optimises likelihood, and the most likely text
is bland, hedged and repetitive. This is the well-documented result that high-likelihood text is not
high-quality text — and the reason nucleus sampling exists at all.

---

## 10. Length penalty — and the exact crossover

Cumulative log-probability **strictly decreases** with every token added, because every log-prob is
negative. The same prefix, extended:

```
 len  sequence                   cum logprob           P
   2  bank bank                    -0.796673    0.450826
   3  bank bank bank               -1.612540    0.199381
   4  bank bank bank bank          -2.486812    0.083175
```

**So beam search structurally prefers shorter sequences.** A beam that emits `<eos>` at length 2
beats one that keeps going, on raw score alone, regardless of quality. That is the bias the length
penalty exists to correct:

```
score = cum_logprob / len^alpha
```

```
  alpha       len 2       len 3       len 4   winner
  0.000   -0.796673   -1.612540   -2.486812   len 2
  0.500   -0.563333   -0.931000   -1.243406   len 2
  1.000   -0.398336   -0.537513   -0.621703   len 2
  1.500   -0.281666   -0.310333   -0.310852   len 2
  1.642   -0.255264   -0.265508   -0.255306   len 2   <- crossover
  2.000   -0.199168   -0.179171   -0.155426   len 4
```

**The crossover is exact:** len-4 overtakes len-2 when `2^alpha > 2.486812/0.796673 = 3.1215`, i.e.
`alpha > 1.6422`.

> **The sign trap.** Scores are **negative**, so dividing by a larger `len^alpha` makes them *less*
> negative. **`alpha > 0` rewards longer sequences.** People routinely state this backwards.
> HuggingFace's `length_penalty` defaults to `1.0`; tuned values usually sit in `0.6–2.0`.

---

## 11. Which to use

```
FACTUAL / EXTRACTION / CODE      greedy, or T=0.0
                                  one right answer; randomness is pure downside

TRANSLATION / SUMMARISATION       beam search B=4-6, length_penalty ~1.0,
                                  no_repeat_ngram_size=3

GENERAL CHAT / ASSISTANT          T=0.7, top_p=0.9        <- the everyday default
                                  apply temperature FIRST, then the nucleus cut

CREATIVE WRITING                  T=0.9-1.1, top_p=0.95, repetition_penalty~1.1

STRUCTURED OUTPUT (JSON, schema)  constrained decoding — mask invalid tokens at every
                                  step so malformed output is IMPOSSIBLE, not just unlikely
```

**Order matters:** temperature scales the logits, then truncation cuts the scaled distribution.
Reversing them changes the result.

---

## 12. Comparison table

| Strategy | Cuts by | Adapts to the distribution | Can drop the mode | Deterministic |
|---|---|---|---|---|
| Greedy | — | — | ✗ | ✓ |
| Temperature | nothing (rescales) | — | ✗ | ✗ |
| Top-k | fixed rank | ✗ | ✗ | ✗ |
| Top-p | cumulative mass | ✓ | ✗ | ✗ |
| min-p | fraction of `p_max` | ✓ | ✗ | ✗ |
| Typical | distance from entropy | ✓ | **✓** | ✗ |
| Beam search | sequence log-prob | — | ✗ | ✓ |

---

## 13. Quick reference

```
temperature   p = softmax(logits / T)              rescales; RANKING UNCHANGED
top-k         keep k highest logits                fixed count
top-p         keep shortest prefix with mass >= p  adaptive count
min-p         keep p_i >= min_p * p_max            adaptive, scales with confidence
typical       keep |-log p_i - H| smallest         can drop the MODE
rep. penalty  logit /= theta if >0 else *= theta   biases, does not forbid
beam search   keep B best SEQUENCES by sum log p   B x compute
length pen.   score = cum_logprob / len^alpha      alpha > 0 REWARDS length
```

**The seven things to be able to say cold:**

1. **Decoding changes no weights.** Same model, different decoder, different text.
2. **Temperature never changes the ranking** — only how much mass the leader keeps. Truncation
   methods remove tokens; temperature does not.
3. **Top-k is a fixed count, top-p is adaptive.** That is the whole reason top-p won.
4. **Typical sampling can drop the most probable token** — verified above, `loan` at `τ=0.5`.
   Top-k and top-p structurally cannot, because both keep a prefix of the sorted order.
5. **Beam search searches sequences, not tokens** — and it is wrong for open-ended text, because the
   most *likely* continuation is bland.
6. **Cumulative log-prob always falls as length grows**, so beam search prefers short outputs. The
   length penalty corrects it, and **`alpha > 0` rewards longer sequences** — the sign trips people.
7. **Everyday default: `T=0.7`, `top_p=0.9`, temperature applied first.** Greedy for anything with
   one correct answer.

---

## See also

- [../../5.transformers/02_models/06_gpt1_end_to_end.md](../../5.transformers/02_models/06_gpt1_end_to_end.md) — where this distribution comes from, and the LM head that produces it
- [06c_transformer_decoder_cross_attention_end_to_end.md](06c_transformer_decoder_cross_attention_end_to_end.md) — autoregressive generation, the KV cache, exposure bias
- [../../5.transformers/02_models/13_speculative_decoding.md](../../5.transformers/02_models/13_speculative_decoding.md) — board 16: making any of these faster without changing the output
- [../../5.transformers/02_models/12_constrained_decoding.md](../../5.transformers/02_models/12_constrained_decoding.md) — grammar/schema-constrained generation
- [08_scaling_laws_emergent.md](08_scaling_laws_emergent.md) — board 17

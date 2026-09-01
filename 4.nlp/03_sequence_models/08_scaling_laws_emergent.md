# 08 — Scaling Laws and Emergent Abilities

> Board 17. Pretraining objectives themselves live in
> [../../5.transformers/01_fundamentals/04_pretraining_objectives.md](../../5.transformers/01_fundamentals/04_pretraining_objectives.md);
> this file is **how much model, how much data, and what actually happens as both grow**.
>
> Instruction tuning, RLHF and DPO are boards 18 and 20 — see
> [../../6.llms/02_finetuning.md](../../6.llms/02_finetuning.md) and
> [../../6.llms/03_alignment.md](../../6.llms/03_alignment.md). They are not covered here.

---

## 1. The functional form

```
L(N, D)  =  E  +  A / N^α  +  B / D^β

  N = parameters          α = 0.34        A = 406.4
  D = training tokens     β = 0.28        B = 410.7
  E = 1.69                                (Hoffmann et al. 2022, Approach 3)
```

Three terms, and each means something distinct:

- **`E` is the irreducible loss** — the entropy of natural language itself. No amount of scale goes
  below it. At 1.69 nats, that is a perplexity floor of `e^1.69 ≈ 5.4`.
- **`A/N^α`** is what you lose by having a model too small to represent the function.
- **`B/D^β`** is what you lose by not having seen enough text.

**Both are power laws, so both have diminishing returns, and neither ever reaches zero.** Scaling
is a straight line on a log-log plot — which is exactly what makes it *predictable*, and why labs
can forecast a run's final loss before starting it.

> A note on symbols: `C` is **compute** (`C ≈ 6ND` FLOPs), never the irreducible term. Some write-ups
> reuse `C` for both, which makes the equations unreadable.

---

## Table of Contents

1. The functional form
2. Kaplan vs Chinchilla
3. Deriving the compute-optimal split
4. The 20:1 rule — and the caveat
5. Real models against the rule
6. **The killer question: 7B on 3T vs 70B on 300B**
7. Why everyone now over-trains
8. **Emergence — and the mirage argument**
9. Quick reference

---

## 2. Kaplan vs Chinchilla

```
Kaplan et al. 2020        N ∝ C^0.73     D ∝ C^0.27     -> "make the model bigger"
Chinchilla 2022           N ∝ C^0.46     D ∝ C^0.54     -> "scale both, roughly equally"
```

**Kaplan's exponents put nearly three times more of a new compute budget into parameters than into
data.** OpenAI followed that, and it gave GPT-3: 175B parameters on only 300B tokens.

DeepMind re-ran the sweep with a crucial fix — Kaplan had used a **fixed learning-rate schedule**
across model sizes, which systematically disadvantaged the smaller-model / more-data runs. With the
schedule tuned per run, the optimum moved sharply toward data.

**Chinchilla itself was the demonstration:** 70B parameters on 1.4T tokens beat GPT-3's 175B on
300B, on most benchmarks, at **2.5× fewer parameters**.

---

## 3. Deriving the compute-optimal split

Minimise `L(N, D)` subject to `C = 6ND`. Substituting `D = C/(6N)`:

```
L(N) = E + A·N^(−α) + B·(6N/C)^β

dL/dN = 0   ->   αA·N^(−α−1) = βB·6^β·N^(β−1)/C^β

              ->   N^(α+β) = αA·C^β / (βB·6^β)

              ->   N ∝ C^(β/(α+β)),   D ∝ C^(α/(α+β))
```

```
β/(α+β) = 0.28/0.62 = 0.4516        N ∝ C^0.4516
α/(α+β) = 0.34/0.62 = 0.5484        D ∝ C^0.5484
```

**The exponents are not `0.5` and `0.5` — they are `0.46` and `0.54`.** "Scale them equally" is the
rounded takeaway, not the fit. Data should grow *slightly faster* than parameters.

---

## 4. The 20:1 rule — and the caveat

The famous number: **roughly 20 training tokens per parameter**, from Chinchilla's own 70B / 1.4T
recommendation.

Chinchilla reached that through three approaches:

```
1. fix model size, vary tokens      a = 0.50, b = 0.50
2. IsoFLOP profiles                 a = 0.49, b = 0.51
3. parametric fit (the L above)     a = 0.46, b = 0.54
```

**Be aware of a real problem here.** Evaluating the *published Approach-3 parameters* at Chinchilla's
own compute budget (`C = 5.76e23`) gives:

```
  N_opt = 32.2B     D_opt = 2.98T     D/N = 93
  but the paper's own recommendation at that budget was 70B / 1.4T  ->  D/N = 20
```

The published parametric fit does not reproduce the paper's own headline. Besiroglu et al. (2024),
*"Chinchilla Scaling: A replication attempt"*, refit that data and found the original Approach-3 fit
describes it poorly.

**Practical position:** use **20:1** as the operational rule — it comes from Approaches 1 and 2,
which are directly empirical. Treat the parametric fit's absolute predictions with suspicion; its
*relative* comparisons (§6) still track what practitioners observe.

Knowing the replication exists is a strong signal in an interview. Quoting `A = 406.4` as gospel is
the opposite.

---

## 5. Real models against the rule

```
  model             N          D       D/N     vs 20:1
  GPT-3         175.0B      0.30T       1.7       0.1×     badly UNDER-trained
  Chinchilla     70.0B      1.40T      20.0       1.0×     the reference point
  Llama-2 7B      7.0B      2.00T     285.7      14.3×     deliberately over
  Llama-3 8B      8.0B     15.00T    1868.0      93.4×     extremely over
  Llama-3 70B    70.6B     15.00T     212.5      10.6×
```

**GPT-3 and Llama 3 are opposite errors — and only one of them was an error.** GPT-3 was
under-trained because Kaplan's law said to be. Llama 3 is over-trained *on purpose*, for the reason
in §7.

---

## 6. The killer question: 7B on 3T vs 70B on 300B

Both use the same compute:

```
  A:  N = 7B   D = 3T     C = 6ND = 1.260e23 FLOPs
  B:  N = 70B  D = 300B   C = 6ND = 1.260e23 FLOPs
```

Under the fitted surface:

```
  L(A) = 2.0045
  L(B) = 2.0246          A is better by 0.0202 nats
```

**A wins on loss at identical training cost** — and then wins again at inference, where it is **10×
cheaper to serve, forever**.

That is the complete answer: *the same compute buys a better model when spent on data rather than
parameters, and the resulting model is also the cheaper one to run.* GPT-3 sat at option B; every
open-weights model since sits far past option A.

> Take the `0.0202` as directional, not exact — §4's caveat applies to the absolute numbers. The
> *ordering* is what Chinchilla established empirically and what Llama demonstrated in production.

---

## 7. Why everyone now over-trains

Chinchilla answers *"what minimises loss for a fixed training budget?"* That is the wrong question
for anyone who will actually deploy the model.

```
TRAINING     paid once
INFERENCE    paid per request, forever
```

A model served billions of times should be **as small as possible for a given quality**, even if
reaching that quality costs far more training compute than "optimal". Training past the
compute-optimal point buys quality at a poor exchange rate — and it is still worth it, because every
parameter you avoid is paid back on every request for the model's whole life.

```
Chinchilla-optimal   minimises TRAINING compute
Llama 3              minimises INFERENCE cost, accepting 93× the "optimal" data
```

**Both statements are correct; they optimise different things.** Being able to say that is the
complete answer. Reciting "20 tokens per parameter" without it is the incomplete one.

---

## 8. Emergence — and the mirage argument

**The claim:** certain abilities (multi-step arithmetic, word unscrambling, chain-of-thought) are
absent in small models and appear *abruptly* past a scale threshold — not gradually.

**The counter-argument (Schaeffer, Miranda & Koyejo, 2023 — *"Are Emergent Abilities of Large
Language Models a Mirage?"*): the discontinuity is in the *metric*, not the model.**

Here is the mechanism, computed. Suppose per-token accuracy `p` improves perfectly smoothly with
scale. Exact-match accuracy on a `K`-token answer is `p^K` — and *that* is what gets plotted.

```
 model scale   per-token p      p^1       p^5      p^10       p^20
        1.0×          0.30    0.300    0.0024   0.00001   0.000000
        3.2×          0.45    0.450    0.0185   0.00034   0.000000
       10.0×          0.60    0.600    0.0778   0.00605   0.000037
       31.6×          0.70    0.700    0.1681   0.02825   0.000798
      100.0×          0.78    0.780    0.2887   0.08336   0.006949
      316.2×          0.85    0.850    0.4437   0.19687   0.038760
     1000.0×          0.90    0.900    0.5905   0.34868   0.121577
     3162.3×          0.94    0.940    0.7339   0.53862   0.290106
    10000.0×          0.97    0.970    0.8587   0.73742   0.543794
    31622.8×          0.99    0.990    0.9510   0.90438   0.817907
```

**Read the `p` column: it climbs smoothly, 0.30 → 0.99, with no jump anywhere.**

Now read `p^20`: `0.000798` → `0.038760` → `0.290106` → `0.817907`. Flat at zero for four orders of
magnitude, then a near-vertical rise. **On a chart that is indistinguishable from a phase
transition** — and it is generated by a perfectly smooth underlying improvement.

```
exact-match / multiple-choice accuracy   DISCONTINUOUS metric  -> emergence appears
token edit distance, Brier score, log-prob   CONTINUOUS metrics -> smooth curves
```

**The honest position** — and what to say when asked:

1. The *measured phenomenon* is real: on these benchmarks, with these metrics, capability does jump.
   That matters practically, because those metrics are often what users experience (a wrong step
   ruins the whole answer).
2. The *interpretation* is contested. There is no established evidence of a discontinuity in the
   underlying model; the sharpness is largely attributable to metric choice.
3. So: **"emergent" describes a plot, not a mechanism.** Treat claims that a specific capability
   will "emerge" at some future scale as speculation, not extrapolation — scaling laws predict
   *loss*, and loss is smooth.

---

## 9. Quick reference

```
L(N,D) = E + A/N^alpha + B/D^beta      E = irreducible entropy of text (1.69)
C ~ 6ND FLOPs                          (2 per MAC forward, x3 for fwd+bwd)

Kaplan 2020     N ~ C^0.73  D ~ C^0.27      -> bigger models
Chinchilla 2022 N ~ C^0.46  D ~ C^0.54      -> ~20 tokens/param
                (derived: beta/(alpha+beta) = 0.4516)

training compute-optimal  != inference-optimal
Llama 3 8B: 1,868 tokens/param = 93x past Chinchilla, deliberately

emergence: exact-match on K tokens = p^K. Smooth p, discontinuous-LOOKING p^K.
```

**The seven things to be able to say cold:**

1. **`L = E + A/N^α + B/D^β`.** `E ≈ 1.69` is the **irreducible** loss — the entropy of text. Scaling
   is a straight line on log-log, which is why final loss is forecastable before a run starts.
2. **Kaplan said `N ∝ C^0.73`; Chinchilla said `N ∝ C^0.46`.** The fix was tuning the LR schedule
   per run — Kaplan's fixed schedule penalised the small-model/more-data configurations.
3. **The exponents are 0.46/0.54, not 0.5/0.5.** Data should grow slightly *faster* than parameters.
4. **~20 tokens per parameter**, from Chinchilla's empirical approaches. Know that the published
   *parametric* fit does not reproduce it (it implies ~93:1 at their own budget) and that
   Besiroglu et al. 2024 found that fit poorly describes the data.
5. **7B on 3T beats 70B on 300B at identical compute** — better loss *and* 10× cheaper to serve.
   GPT-3 was the second option.
6. **Chinchilla optimises training compute; deployment optimises inference cost.** Llama 3 sits 93×
   past "optimal" on purpose. Both are right about different objectives.
7. **Emergence is a property of the metric.** `p^20` looks like a phase transition when `p` is
   perfectly smooth: `0.000798 → 0.038760 → 0.290106 → 0.817907`. Continuous metrics show no jump.
   "Emergent" describes a plot, not a mechanism.

---

## See also

- [../../5.transformers/01_fundamentals/04_pretraining_objectives.md](../../5.transformers/01_fundamentals/04_pretraining_objectives.md) — the objectives themselves: CLM, MLM, span corruption
- [../../5.transformers/02_models/06c_gpt3_end_to_end.md](../../5.transformers/02_models/06c_gpt3_end_to_end.md) — GPT-3's `3.14e23` FLOPs, and in-context learning
- [../../5.transformers/02_models/08b_llama3_end_to_end.md](../../5.transformers/02_models/08b_llama3_end_to_end.md) — Llama 3's 15T tokens, the deliberate over-training
- [../../6.llms/02_finetuning.md](../../6.llms/02_finetuning.md) — board 18: what happens after pretraining
- [../../6.llms/04_evaluation.md](../../6.llms/04_evaluation.md) — board 21: the metrics §8 is about

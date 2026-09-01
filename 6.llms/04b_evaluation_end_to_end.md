# 04b — Evaluation: Perplexity, Benchmarks, Judges, Contamination

> Board 21, and the last of the LLM track. Companion to [04_evaluation.md](04_evaluation.md), which
> covers BLEU/ROUGE/BERTScore, RAGAS and task-specific metrics well. **This file is the arithmetic**,
> and it fills a real gap: that file never mentions **perplexity** — which is the board's first
> killer question — and its `pass@k` example is garbled (§3).

---

## Table of Contents

1. **Why perplexity is a bad headline metric**
2. What perplexity is still good for
3. **pass@k — and a correction**
4. Multiple-choice floors
5. LLM-as-judge — the three biases
6. **Contamination detection**
7. Quick reference

---

## 1. Why perplexity is a bad headline metric

`PPL = exp(cross-entropy per token)`. Three independent problems.

### 1.1 It is tokenizer-dependent

The total cross-entropy of a document — in nats — is a property of the text and the model. The
*number of tokens* you divide by is a property of the tokenizer. Hold the first fixed at 1,400 nats
and vary the second:

```
  tokenizer          tokens   nats/token      PPL
  char-level          5,600       0.2500     1.28
  Llama 2 (32k)       1,000       1.4000     4.06
  GPT-2 (50k)           900       1.5556     4.74
  Llama 3 (128k)        780       1.7949     6.02
```

**Same document, same model quality, perplexity from 1.28 to 6.02.** A larger vocabulary packs more
characters per token, so each token is *individually harder* to predict — and perplexity goes up
while nothing got worse.

**Perplexity is comparable only across models sharing a tokenizer.** Comparing Llama 3's PPL to
GPT-2's is comparing nothing — this is the same fertility fact as
[05_embedding_lookup_end_to_end.md §7](../4.nlp/01_fundamentals/05_embedding_lookup_end_to_end.md),
seen from the metric side.

### 1.2 It measures prediction, not usefulness

**An SFT/RLHF model usually has *worse* perplexity on raw web text than the base model it came
from** — while being far more useful. Alignment deliberately moves probability mass toward helpful
continuations and away from the raw-text distribution perplexity is measured against.

If you rank a chat model and its base model by perplexity on Common Crawl, you will rank them
backwards.

### 1.3 It saturates

From board 17, `L = E + A/N^α + B/D^β` with `E = 1.69` irreducible — a perplexity floor of
`exp(1.69) = 5.42`. Near the floor, large capability differences correspond to tiny perplexity
differences, so the metric stops discriminating exactly where you most need it to.

---

## 2. What perplexity is still good for

It is not useless — it is a bad *headline*.

```
GOOD    same model, same tokenizer, tracking a training run
        detecting domain shift (PPL spikes on out-of-distribution data)
        comparing checkpoints of one model
        a cheap smoke test that fine-tuning did not break the model

BAD     ranking different model families
        anything a user experiences
        claiming quality after alignment
```

---

## 3. pass@k — and a correction

For code, one sample is noisy, so generate `n` and estimate the probability that at least one of `k`
would pass. The unbiased estimator (Chen et al., 2021):

```
pass@k  =  1  −  C(n−c, k) / C(n, k)          n generated, c correct
```

With `n = 20`, `c = 8`:

```
   k     pass@k
   1     0.4000
   2     0.6526
   5     0.9489
  10     0.9996
  20     1.0000
```

> **Correction to the companion file.** [04_evaluation.md](04_evaluation.md) states
> *"pass@1 (k=20, 8 correct): ≈ 0.48"* and *"pass@1 (k=20, 8 correct, n=10): ≈ 0.95"*. Both labels
> are incoherent — `pass@1` cannot have `k=20`, and the two lines give different values for the same
> name. The correct figures for `n=20, c=8` are **`pass@1 = 0.4000`** and **`pass@10 = 0.9996`**.

**Why an estimator at all?** For `k=1` it reduces to the sample mean `c/n = 0.40`. The estimator
matters for `k > 1`, where naively computing "did any of the first k pass?" is high-variance and
biased by which samples you happened to draw.

**pass@k rewards breadth, not reliability.** A model at `pass@10 = 0.9996` may still be at
`pass@1 = 0.40` — useless without a test harness to filter with. Quote `pass@1` for anything
user-facing.

---

## 4. Multiple-choice floors

A 4-option benchmark has a **25% floor** for any model that answers at all.

```
MMLU        4 options, n = 14,042    guessing 25.0%   SE 0.365%   95% CI ±0.72%
ARC         4 options, n =  1,172    guessing 25.0%              95% CI ±2.48%
HellaSwag   4 options, n = 10,042    guessing 25.0%              95% CI ±0.85%
```

```
  score 25.2%   z = 0.55    within noise of guessing
  score 25.7%   z = 1.92    within noise
  score 27.0%   z = 5.47    statistically above chance
```

**27% is statistically above chance** — but the useful range is 25–100%, so it is only
`(27−25)/75 = 2.7%` of the way up it. **Statistically real, practically meaningless.**

Two consequences:

- **Never report a raw multiple-choice score without its floor.** "30% on MMLU" sounds like a third
  correct; it is 6.7% of the usable range.
- **Small benchmarks are noisy.** ARC's ±2.48% CI means a 3-point difference between two models is
  not evidence of anything.

---

## 5. LLM-as-judge — the three biases

Using a strong model to score outputs is cheap and correlates reasonably with humans. It has three
systematic biases, all of which are correctable:

```
POSITION      the judge prefers the response shown FIRST
              -> fix: run both orders, keep only agreeing verdicts (or average)

VERBOSITY     the judge prefers longer answers, largely independent of quality
              -> fix: control for length, or report length alongside win-rate

SELF-PREFERENCE  a judge prefers text from its own family
              -> fix: never judge a model with itself; use a third family, or an ensemble
```

**Position bias is the one to always fix**, because it is free to fix and large. Running both orders
and discarding disagreements gives you a consistency rate as a side benefit — if the judge flips on
40% of pairs, its verdicts are close to noise and the eval is not measuring anything.

The deeper limitation: **a judge cannot reliably evaluate what it cannot itself do.** For a task at
the frontier of the judge's ability, its preferences reflect its own errors. Judges are for style,
instruction-following, and relative comparison — not for verifying correctness in a domain the judge
is weak in.

---

## 6. Contamination detection

The board's second killer question. Four methods, roughly in order of how much access they need:

### 6.1 The perturbation test — the one you can always run

If a model **memorised** a benchmark, it does much better on the original wording than on a
semantically identical paraphrase. A clean model is indifferent.

```
  model        original   paraphrased    gap    reading
  clean            0.62          0.61   0.01    consistent with clean
  clean-2          0.71          0.69   0.02    consistent with clean
  SUSPECT          0.89          0.63   0.26    contamination likely
  SUSPECT-2        0.94          0.58   0.36    contamination likely
```

**A large original-minus-paraphrase gap is the signature.** It needs no access to training data,
which is why it is the practical test for closed models.

### 6.2 The others

```
n-gram overlap    search the training corpus for test-set n-grams (13-gram is a common threshold)
                  -> requires training-data access; the standard lab-internal check

canary strings    benchmarks plant a unique random string; if a model can reproduce it,
                  the benchmark was in its training data. BIG-bench does this.

train/test gap    a gap far larger than model size explains suggests memorisation
                  -> weak evidence alone, useful in combination
```

### 6.3 Why this matters more than it used to

Benchmarks are published on the open web, and training corpora are scraped from the open web. **The
default assumption for any benchmark older than a model should be that it is contaminated**, unless
the lab documents decontamination. This is why held-out and continuously-refreshed evaluations
(LMSYS Arena, private eval sets, freshly written tasks) displaced static benchmarks for frontier
comparison.

**The practical rule:** for anything you actually ship, build a private eval set from your own data
and never publish it.

---

## 7. Quick reference

```
PPL = exp(CE per token)     tokenizer-dependent (1.28 to 6.02 for ONE document)
                            worse after alignment; floor exp(1.69) = 5.42
                            use it WITHIN a model, never ACROSS families

pass@k = 1 - C(n-c,k)/C(n,k)      n=20 c=8: pass@1 0.4000, pass@10 0.9996
MC floor = 1/options              MMLU 25%, CI +/-0.72% at n=14,042

judge biases   position (fix: both orders) · verbosity · self-preference
contamination  perturbation gap · n-gram overlap · canaries · train/test gap
```

**The seven things to be able to say cold:**

1. **Perplexity is tokenizer-dependent.** One fixed document gives PPL from **1.28 to 6.02**
   depending only on how it is tokenized. Never compare PPL across model families.
2. **Alignment makes perplexity worse** on raw text while making the model more useful — rank a chat
   model against its base by PPL and you rank them backwards.
3. **Perplexity saturates** at `exp(1.69) = 5.42`, so it stops discriminating exactly where
   capability differences get interesting.
4. **`pass@k = 1 − C(n−c,k)/C(n,k)`.** For `n=20, c=8`: `pass@1 = 0.4000`, `pass@10 = 0.9996`.
   pass@k rewards breadth; quote `pass@1` for anything user-facing.
5. **Multiple choice has a floor.** 4 options ⇒ 25%. A 27% MMLU score is statistically above chance
   (`z = 5.47`) but only **2.7% of the usable range**. Always report the floor.
6. **Judges have three biases** — position, verbosity, self-preference. Position is free to fix by
   running both orders, and the flip rate doubles as a validity check on the eval itself.
7. **Contamination: use the perturbation test.** A large original-vs-paraphrase gap (`0.26`, `0.36`
   above) is the signature, and it needs no training-data access. Assume any benchmark older than
   the model is contaminated unless decontamination is documented.

---

## See also

- [04_evaluation.md](04_evaluation.md) — BLEU/ROUGE/BERTScore, RAGAS, hallucination detection (and the `pass@k` correction in §3)
- [../4.nlp/03_sequence_models/08_scaling_laws_emergent.md](../4.nlp/03_sequence_models/08_scaling_laws_emergent.md) — board 17: the irreducible loss `E`, and why metric choice creates "emergence"
- [../4.nlp/01_fundamentals/05_embedding_lookup_end_to_end.md](../4.nlp/01_fundamentals/05_embedding_lookup_end_to_end.md) — fertility, the same fact behind §1.1
- [../5.transformers/02_models/06_gpt1_end_to_end.md](../5.transformers/02_models/06_gpt1_end_to_end.md) — perplexity computed from a real loss
- [03c_dpo_end_to_end.md](03c_dpo_end_to_end.md) — board 20: what you are trying to measure the effect of
- [../4.nlp/04_applications/06_generative_eval.md](../4.nlp/04_applications/06_generative_eval.md) — generative evaluation in the NLP track

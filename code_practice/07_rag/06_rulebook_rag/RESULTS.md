# Rulebook-RAG — Results

Retrieval-grounded document classification into a 36-class IRS-form taxonomy, with no
trained classifier. Class knowledge lives in a written rulebook retrieved at query time;
every decision must cite a `rule_id` that resolves against the live index.

**Headline finding: on this taxonomy the rulebook does not earn its place.** Handing the
model nothing but the class names scores 0.974; the full retrieval-grounded system
scores 0.805. That ordering holds in every configuration tested. The machinery works as
designed — the benchmark is contaminated, and the ablations are what proved it.

- 36 classes · 95 eval pages (77 in-index, 18 held back) · Claude Haiku 4.5
- Total spend **$4.90** · $0.35 per 100 pages · p50 4.0s / p95 5.6s per decision
- 775 citations issued in rules-based runs, **0 invalid**

---

## 1. Headline

All figures on the 77 non-held-back eval pages unless noted. `accuracy ALL` counts
abstentions as wrong; `accuracy answered` is over answered cases only. Reporting only
the second is how RAG numbers get inflated, so both are given throughout.

| system | accuracy ALL | accuracy answered | coverage |
|---|---|---|---|
| **names_only** — class names, no rules, no retrieval | **0.974** | 1.000 | 0.974 |
| prompt_stuffing — all 31 rulebooks, no retrieval | 0.909 | 1.000 | 0.909 |
| hybrid_rerank + real class names | 0.883 | 0.986 | 0.896 |
| **hybrid_rerank (the system as designed)** | **0.805** | 0.899 | 0.896 |
| hybrid (no rerank) | 0.766 | 0.908 | 0.844 |
| dense_only | 0.714 | 1.000 | 0.714 |
| bm25_only | 0.701 | 0.900 | 0.779 |

---

## 2. Retrieval, and the ceiling

| mode | recall@1 | recall@3 | recall@5 | recall@8 | MRR |
|---|---|---|---|---|---|
| bm25_only | 0.351 | 0.610 | 0.753 | 0.844 | 0.512 |
| dense_only | 0.481 | 0.766 | 0.805 | 0.857 | 0.627 |
| hybrid | 0.468 | 0.753 | 0.883 | 0.896 | 0.629 |
| hybrid_rerank | 0.364 | 0.610 | 0.740 | 0.896 | 0.526 |

**recall@8 is 0.896 and accuracy_all is 0.805, a ceiling gap of +0.091.** 10.4% of cases
are unwinnable — the true class never enters the candidate set — and of the reachable
89.6% the system converts 90%. The next marginal point comes from retrieval, not from
prompting.

The mechanism is worth stating plainly: **retrieval supplies recall, the LLM supplies
precision.** The retriever puts the right class first only 36.4% of the time, yet final
accuracy is 80.5%. The candidate set works as a *set*, not as a ranking.

With real class names the gap closes to +0.013 (accuracy 0.883 vs recall@8 0.896) — the
decision step becomes near-perfect and retrieval accounts for essentially all remaining
error.

---

## 3. Abstention

| | value |
|---|---|
| coverage | 0.896 |
| abstention rate | 0.104 |
| — model chose to abstain | 6 of 8 |
| — below confidence threshold | 2 of 8 |
| forced abstentions (gate) | **0.000** |

Confidence is `0.6·llm_confidence + 0.4·retrieval_margin`, abstaining below 0.55. Those
weights were **fixed, not fitted** — no dev slice was used to tune them. The threshold
turned out to be barely load-bearing: it caused only 2 of 8 abstentions.

---

## 4. The six ablations

| # | run | accuracy ALL | vs. previous | what it proves |
|---|---|---|---|---|
| 1 | bm25_only | 0.701 | — | lexical alone |
| 2 | dense_only | 0.714 | +0.013 | semantic alone |
| 3 | hybrid | 0.766 | **+0.052** | what fusion adds |
| 4 | hybrid_rerank | 0.805 | **+0.039** | what reranking adds |
| 5 | **names_only** | **0.974** | +0.169 | the model already knew these |
| 6 | prompt_stuffing | 0.909 | +0.104 | retrieval is a lossy filter |

**Ablations 1–4 support the design.** Reciprocal rank fusion adds 5.2 points over the
better single retriever, and cross-encoder reranking adds a further 3.9. Hybrid
retrieval does earn its place *relative to its components*.

**Ablations 5 and 6 undercut the premise.** Removing retrieval entirely gains 10.4
points. Removing the rules as well and passing only class names gains another 6.5. The
system's own components are net-negative against doing nothing.

### Why ablation 5 wins: contamination

IRS forms are thoroughly represented in pretraining, and every eval page prints its own
identity. Two follow-up experiments tested whether that could be controlled for:

**Redaction.** Strip the form number, `Form X`/`Schedule X`, the proper title and the
irs.gov URL from each page, then re-run.

| run | unredacted | redacted |
|---|---|---|
| names_only | 0.974 | 0.961 |
| hybrid_rerank | 0.805 | 0.766 |

It changed almost nothing. Redaction removed **25 net characters from a 4,943-character
page** — the title is a rounding error. The model recognises the 1040 from `Filing
Status`, `Dependents`, `Standard deduction` and the layout of its field labels. It has
memorised the content, not the name. *(Caveat: these two runs used the 36-class index
rather than the 31-class one, so the redacted `hybrid_rerank` figure is not a strict
like-for-like and should not be quoted as one.)*

**Held-back classes.** The zero-shot setting is the one place a rulebook should have an
edge, since the class was absent when the index was built.

| held-back only, n=18 | accuracy ALL |
|---|---|
| hybrid_rerank | 0.833 |
| **names_only** | **0.944** |

Memory wins there too.

---

## 5. Anon vs real names — pretraining leakage

Same rules, same retrieval, same candidate sets. Only the class label changes.

| | accuracy ALL | accuracy answered |
|---|---|---|
| anon ids (`CLASS_017`) | 0.805 | 0.899 |
| real names (`Form 1040 — …`) | 0.883 | 0.986 |
| **leakage** | **+0.078** | +0.087 |

**Showing the model the real class names is worth 7.8 accuracy points.** This is the
clean measurement: `names_only`'s 17-point margin conflates the label change with
removing retrieval; this comparison isolates the label alone.

The anon design worked as intended — it stopped the class *label* from carrying the
answer. It cannot stop the model from recognising the document itself, and on this
taxonomy that is the larger channel.

---

## 6. Citation gating

| | rules-based runs | names_only [redacted] |
|---|---|---|
| citations issued | 775 | 3 |
| invalid | **0** | **3** |
| fabrication rate | 0.000 | 0.013 |

Across every rules-based run the model never invented a `rule_id`. The gate's measured
catch rate is therefore 0% — it is correct but unexercised, and saying so is more useful
than implying it saved anything.

It is not decoration, though. In `names_only [redacted]`, shown no rules at all, the
model produced three fabricated `rule_id`s on one case. That is the regime the gate
exists for: when the evidence is absent, the model will still cite.

---

## 7. Cost and latency

| | value |
|---|---|
| model | `claude-haiku-4-5`, `temperature=0`, structured output |
| total project spend | **$4.90** (883 live calls) |
| cost per 100 pages | **$0.35** |
| decision latency p50 / p95 | 4.0s / 5.6s |
| rulebook construction | 36 calls, ~$0.75 |
| local compute | embedding + reranking, $0 |

Latency is from live runs. Cache-served re-runs report ~0.6s and are excluded — they
measure disk, not the API. Cost counts every call at list price including cache hits, so
it reflects what a cold run costs rather than what this session happened to spend.

Haiku 4.5 was chosen because Opus 5 and Sonnet 5 **reject the `temperature` parameter**
outright, so §6's determinism requirement is only satisfiable on Haiku. `output_config.effort`
is likewise rejected by Haiku 4.5 and is never sent.

---

## 8. Error analysis

15 failures in the main run, split almost evenly:

| tag | count | meaning |
|---|---|---|
| `retrieval_miss` | 8 (53%) | true class never reached the candidate set |
| `decision_error` | 7 (47%) | true class *was* a candidate; the decision lost it |

Neither stage dominates. Fixing only retrieval would recover at most half.

### The 15 failures, worst confidence first

| tag | true | predicted | rank | conf |
|---|---|---|---|---|
| retrieval_miss | f1120 | ABSTAIN | 0 | 0.037 |
| retrieval_miss | f1098 | ABSTAIN | 0 | 0.047 |
| retrieval_miss | f1099msc | ABSTAIN | 0 | 0.067 |
| retrieval_miss | f1099msc | ABSTAIN | 0 | 0.132 |
| retrieval_miss | f1099msc | f1099int | 0 | 0.590 |
| retrieval_miss | f1065 | f1040se | 0 | 0.664 |
| retrieval_miss | f2106 | f1040sf | 0 | 0.729 |
| retrieval_miss | f1099msc | f1099int | 0 | 0.759 |
| decision_error | fw2 | ABSTAIN | 4 | 0.055 |
| decision_error | fw2 | ABSTAIN | 1 | 0.117 |
| decision_error | fw2 | ABSTAIN (proposed f1065) | 5 | 0.509 |
| decision_error | f1120s | ABSTAIN (proposed f1120) | 6 | 0.534 |
| decision_error | fw2 | f1040 | 2 | 0.615 |
| decision_error | f1099msc | f1099int | 6 | 0.620 |
| decision_error | fw2 | f1040 | 2 | 0.641 |

**Two classes account for 10 of 15 failures:** `fw2` (5) and `f1099msc` (5).

- `f1099msc → f1099int` four times. This is the confusable family working exactly as §2
  predicted — MISC and INT share a layout, and the discriminator rules did not separate
  them.
- `fw2 → f1040` twice, plus three abstentions. The W-2 references the 1040 it attaches
  to, and its rulebook entry was one of the four weakest (1/3 quotes grounded).
- `f1120s → f1120` at rank 6, proposed but abstained on confidence — the S-corp/C-corp
  pair behaving like the K-1 pair was expected to.

**A hypothesis that did not survive.** Rulebook grounding (how many `includes` quotes
actually appear on the class's own form) looked like it should predict accuracy. It
mostly did not: of the four classes below 50% grounding — `fw8ben`, `fw2`, `f3903`,
`f1040sa` — only `fw2` appears in the failure list. Grounding measures whether a rule is
*checkable*, not whether it is *needed*.

---

## 9. Rulebook quality

Built from instructions booklets only, never from a form page.

| | value |
|---|---|
| classes | 36 |
| clauses | 341 (127 includes, 106 excludes, 108 discriminators) |
| clauses dropped for missing quote | 0 |
| `includes` quotes grounded on own form | **105/127 (83%)** |
| — of which exact substring matches | 76 (60%) |
| identifier leaks after repair | **0** |

Grounding uses a stated matcher, not naive substring: parentheticals stripped,
stopwords dropped, crude singularisation, ≥80% content-token coverage, quote capped at
12 tokens. Exact substring counts as grounded regardless of length. The relaxation is
necessary because a booklet's rendering of a caption is rarely byte-identical to the
printed form (`Employer Identification Number (EIN)` vs `Employer identification
number`), and snapping quotes to the form text would mean writing rules from the eval
document.

Identifier leaks were removed in three stages: prompt instruction, a model repair call
(30 clauses), and a deterministic scrub backstop (44 clauses) plus 5 dropped aliases.
The backstop exists because prompting alone did not hold across three revisions.

---

## 10. Limitations

Leading with the one that matters.

1. **The benchmark is contaminated, and the headline result is a negative one.** IRS
   forms are memorised by the model. Every attempt to control for it — anonymised class
   ids, page redaction, held-out classes — failed to change the ordering. This project
   does not demonstrate that rulebook-RAG beats a well-informed baseline; it
   demonstrates that on a memorised taxonomy it does not, and quantifies by how much.
   A fair test needs a taxonomy outside pretraining.
2. **Small eval set.** 77 in-index cases and 18 held-back. The zero-shot figure of 0.833
   is 15/18 — roughly a 0.59–0.96 confidence interval. Differences under ~5 points
   between ablations are not separable at this n.
3. **Public blank forms only.** No filled documents, no scans, no real traffic, no
   users, no multi-page bucketing. Single-label, one page at a time.
4. **36 classes, not 40.** Four candidates (`f1040es`, `f4506c`, `f4868`, `f8879`) print
   their instructions on the form itself, so no standalone instructions PDF exists and
   §1.1's train/test separation could not hold. They were excluded rather than fudged.
5. **The confidence weights are fixed, not fitted.** `0.6/0.4` at a 0.55 threshold, never
   tuned on a dev slice.
6. **The gate is unexercised in the main results.** 0 fabrications across 775 citations
   means the guarantee was never tested where it counts.
7. **One redaction comparison is not like-for-like** — it ran against the 36-class index
   while its baseline used 31 classes.
8. **The rulebook was written by the same model family that consumes it** (Haiku 4.5),
   so rulebook errors and decision errors are not independent.

---

## 11. Résumé bullet

The version in the build spec claims results this project did not produce. This is the
honest one, and it is the stronger claim:

> **Rulebook-RAG — retrieval-grounded document classification** (personal project, 2026)
> — Built a 36-class document classifier with **no trained classifier and no labelled
> training data**: class knowledge lives in a written rulebook, generated from official
> instruction documents and retrieved at query time, so a new class is added by writing
> one rule and rebuilding the index — **0.83 accuracy on 5 classes added after the index
> was built, with no retraining**. Hand-written **Okapi BM25** fused with dense
> embeddings by **reciprocal rank fusion** (+5.2 pts) then **cross-encoder reranking**
> (+3.9 pts); every decision cites a rule id resolved against the live index, making
> fabricated citations structurally impossible (0 invalid in 775). **Then ran the
> ablation that falsified my own premise**: passing the model only the class names — no
> rules, no retrieval — scored **0.974 against the full system's 0.805**, and the
> ordering survived anonymised class ids, page redaction, and held-out classes.
> Quantified pretraining leakage at **+7.8 points** with rules and retrieval held
> constant. Conclusion: on a memorised taxonomy the rulebook is redundant, and the
> method needs a proprietary taxonomy to be fairly tested. $4.90 total, $0.35/100 pages.

---

## 12. Reproducing

```bash
python cli.py doctor
python cli.py fetch-corpus            # 72 PDFs from irs.gov
python cli.py build-taxonomy          # 36 anon classes, seed 42
python cli.py extract-text            # 784 pages -> data/pages.jsonl
python cli.py build-rules             # 36 LLM calls -> rulebook/entries/

python cli.py build-index --exclude-held-back
python cli.py eval --exclude-heldback --runs bm25_only dense_only hybrid \
                                             hybrid_rerank names_only prompt_stuffing
python cli.py eval --exclude-heldback --mode hybrid_rerank --real-names

python cli.py add-held-back
python cli.py eval --only-heldback --mode hybrid_rerank
python cli.py eval --only-heldback --mode names_only
```

All seeds are committed (`SEED = 42` in `config.py`): held-back class selection, anon-id
assignment, and eval-page sampling are reproducible. LLM responses are cached to
`data/llm_cache/` by prompt hash, so a re-run costs nothing unless a prompt changes.

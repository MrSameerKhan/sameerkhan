# Session 06 — Rulebook-RAG

Status: `✅ Run` — built and executed end to end; every number below is from a real run,
captured in [RESULTS.md](RESULTS.md).

Build spec: [RULEBOOK_RAG_3DAY_BUILD.md](../../../RULEBOOK_RAG_3DAY_BUILD.md)
Long-term design (not built): [RULEBOOK_RAG_PROJECT_SPEC.md](../../../RULEBOOK_RAG_PROJECT_SPEC.md)
Earlier abandoned attempt: [06_rulebook_rag_draft_1/](../06_rulebook_rag_draft_1/)

---

## Use case

Classify a page into one of 36 document classes **without training a classifier**. Class
knowledge lives in a written rulebook generated from official instruction documents and
retrieved at query time; every decision cites a `rule_id` verified against the live index.

Theory cross-references — this file does not re-explain them:

- RAG patterns and pipeline depth: [../../../7.rag/](../../../7.rag/)
- Sentence embeddings, bi-encoder vs cross-encoder: [../../../4.nlp/02_embeddings/02_sentence_embeddings.md](../../../4.nlp/02_embeddings/02_sentence_embeddings.md)
- BERT-family encoders: [../../../5.transformers/02_models/01_bert_family.md](../../../5.transformers/02_models/01_bert_family.md)
- Decoder-only LLMs: [../../../5.transformers/02_models/02_gpt_family.md](../../../5.transformers/02_models/02_gpt_family.md)

---

## Headline result

**The ablation falsified the premise, and that is the finding.**

| | accuracy ALL |
|---|---|
| class names only — no rules, no retrieval | **0.974** |
| full retrieval-grounded system | 0.805 |

The ordering held under page redaction and on held-out classes. IRS forms are memorised
by the model, so this taxonomy cannot test the method. Full analysis in
[RESULTS.md](RESULTS.md).

What *did* hold: the retrieval ladder (fusion +5.2 pts, cross-encoder rerank +3.9 pts),
zero-shot class addition without retraining (0.833), 0 invalid citations in 775, and
pretraining leakage measured at +7.8 points.

---

## Scale

| | |
|---|---|
| classes | 36 (40 probed, 4 dropped — no standalone instructions PDF) |
| corpus | 72 PDFs, 784 pages, 23 MB |
| rulebook | 341 clauses, 83% of `includes` quotes grounded on their own form |
| eval set | 95 pages (77 in-index, 18 held-back), ≤5 per class, seed 42 |
| model | `claude-haiku-4-5`, `temperature=0`, structured output |
| cost | $4.90 total, $0.35 per 100 pages |
| latency | p50 4.0s, p95 5.6s per decision |

---

## Deviations from the spec, and why

| spec | built | reason |
|---|---|---|
| 40 classes | 36 | 4 forms print their instructions on the form itself, so §1.1's train/test separation cannot hold for them |
| `temperature = 0` | kept, on Haiku 4.5 | Opus 5 / Sonnet 5 **reject** `temperature` (400). Haiku 4.5 accepts it, so determinism is only reachable there — and it rejects `output_config.effort`, which is never sent |
| instructions truncated to ~4000 tokens | ~15K tokens | the 4000-token cap was a cost constraint; at Haiku pricing full-booklet sampling is ~$0.75 for all 36, and 3.5% booklet coverage was starving caption mining |
| head+tail truncation | stratified sampling | line-by-line captions live in the *middle* of a booklet, which head+tail discards |
| eval quotes matched exactly | exact **or** token-based | a booklet's rendering of a caption is rarely byte-identical to the printed form; snapping quotes to the form would mean writing rules from the eval document |
| few-shot exemplars in the prompt | none | every labelled page we own is in the eval set; exemplars would leak eval data into the decision path |
| — | added: redaction ablation | to test whether the rulebook carries the decision once the memorisation channel is removed. It did not — see RESULTS §4 |

---

## Files

```
config.py      every tunable, no magic numbers elsewhere
llm.py         Anthropic client + StubLLM + disk cache + real cost/latency counter
cli.py         command dispatch (10 commands)
fetch.py       sources.yaml -> data/pdfs/, url-hash cache, 404-tolerant
extract.py     taxonomy.json + PDFs -> data/pages.jsonl
rules.py       instructions -> rulebook entries + grounding/leak audit + repair + scrub
bm25.py        Okapi BM25 by hand; `python bm25.py` runs a known-answer self-test
index.py       one vector per rule clause, context-prefixed, + BM25 + manifest
retrieve.py    bm25 / dense / RRF / rerank, one mode switch
decide.py      one LLM call, citation gate, confidence blend
evaluate.py    metrics, six ablations, redaction ablation, RESULTS tables
README.md      what it is and how to run it
RESULTS.md     every number, error analysis, limitations, résumé bullet
```

---

## What I would do differently

1. **Choose the taxonomy against the contamination risk first.** The whole method rests
   on the model *not* already knowing the classes. IRS forms were picked for their
   confusable families — a good reason that turned out to be the wrong criterion. One
   `names_only` probe on 10 pages, before writing any code, would have caught it in five
   minutes and $0.05.
2. **Run ablations 5 and 6 first, not last.** They are the controls that decide whether
   the rest of the numbers mean anything. Building the full system before running them
   meant three days of work before learning the benchmark was unusable.
3. **Not over-tighten a rule the spec did not ask for.** I banned form titles from quotes
   on my own initiative; §5b's own example uses one. That cost an iteration.

## Next step

Re-run against a taxonomy outside pretraining — internal document types, a synthetic
taxonomy with invented class semantics, or a niche public corpus. Everything except
`sources.yaml` and `taxonomy.json` is reusable as-is, and `names_only` is already
implemented as the first check to run.

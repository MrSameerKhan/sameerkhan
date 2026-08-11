# Rulebook-RAG

Document classification into a 36-class taxonomy **without training a classifier**.
Class knowledge lives in a written rulebook that is retrieved at query time; every
decision must cite a `rule_id` that resolves against the live index, and a citation that
does not resolve voids the whole decision.

**Results, including the ablation that falsified the premise: [RESULTS.md](RESULTS.md).**

---

## The idea

A conventional classifier learns classes from labelled examples. This one reads them.

1. For each class, an LLM reads that form's **official instructions booklet** and writes
   a rulebook entry — `includes` / `excludes` / `discriminators`, each clause carrying a
   literal `quote` that should be printed on the form.
2. Every clause is embedded individually and indexed for BM25.
3. At query time a page retrieves candidate clauses, they roll up to 8 candidate
   classes, and one LLM call picks between them — citing the rule ids it used.
4. A gate checks every citation against the index. Any that does not resolve forces an
   abstention.

Adding a class means writing one rulebook entry and rebuilding the index. No retraining,
no labelled data.

## Two design decisions that make the evaluation honest

**Rules and eval data come from physically different documents.** The rulebook for a
class is written *only* from `i<slug>.pdf` (the instructions). The eval pages come *only*
from `f<slug>.pdf` (the blank form). They are separate publications about the same class,
so the evaluation cannot be circular and no eval data is LLM-generated. A build where
those two are ever the same file is invalid.

**Class ids are anonymised.** The index, the prompt and the decision use `CLASS_017`, not
"Schedule C". The real name lives in a mapping file that never enters a prompt, and the
id-to-class assignment is shuffled with a committed seed so it carries no ordering
signal. Running the eval twice — once anonymised, once with real names — measures how
much of the score is pretrained memory rather than retrieval. It came to **+7.8 points**.

## What it found

The system reaches **0.805 accuracy** (0.899 on answered cases, 89.6% coverage) with
zero invalid citations across 775.

Then ablation 5 handed the model nothing but the class names — no rules, no retrieval —
and it scored **0.974**. That ordering held under page redaction and on held-out classes.

IRS forms are memorised. On this taxonomy the rulebook is redundant, and the method needs
a taxonomy outside pretraining to be fairly tested. The full account, including the
retrieval ladder that *does* hold (fusion +5.2, reranking +3.9) and the error analysis,
is in [RESULTS.md](RESULTS.md).

## Setup

```bash
pip install -r requirements.txt
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env      # git-ignored
python cli.py doctor
```

`doctor` checks the Python version, imports, that torch is a CPU build, that both local
models load from cache, that the API key works with a live 10-token call, and that the
directory layout exists.

A full cold run costs about **$4.90** and takes roughly 90 minutes, most of it waiting on
the API. Responses are cached by prompt hash in `data/llm_cache/`, so re-running costs
nothing unless a prompt changes.

## Commands

| command | what it does |
|---|---|
| `doctor` | verify the environment |
| `fetch-corpus [--dry-run]` | `sources.yaml` → `data/pdfs/` (72 PDFs). `--dry-run` probes URLs only |
| `build-taxonomy [--show-mapping]` | `sources.yaml` → `taxonomy.json`, anon ids + held-back split |
| `extract-text` | PDFs → `data/pages.jsonl`, one record per page |
| `build-rules [--limit N] [--stub-llm]` | instructions → `rulebook/entries/CLASS_NNN.json` |
| `build-index [--exclude-held-back]` | rulebook → vectors + BM25 + manifest |
| `add-held-back` | rebuild the index with held-back classes included |
| `retrieve --page-id X [--mode M]` | retrieval only, no LLM, free |
| `classify --page-id X --verbose` | one page end to end: retrieval → prompt → decision → gate → confidence |
| `eval [--runs ...] [--only-heldback] [--real-names] [--redact-pages]` | the eval set and §7 metrics |

`--stub-llm` runs a schema-valid fake LLM: the whole pipeline executes for $0.

## Layout

```
config.py      every tunable; no magic numbers elsewhere
llm.py         Anthropic client + StubLLM + disk cache + real cost/latency counter
cli.py         command dispatch
fetch.py       sources.yaml -> data/pdfs/
extract.py     taxonomy.json + PDFs -> data/pages.jsonl
rules.py       instructions -> rulebook entries, with grounding + leak audit
bm25.py        Okapi BM25 by hand, ~40 lines, no library. `python bm25.py` self-tests
index.py       one vector per rule clause + BM25 + manifest
retrieve.py    bm25 / dense / RRF / rerank; the ablation switch
decide.py      one LLM call, the citation gate, the confidence blend
evaluate.py    metrics, six ablations, RESULTS tables
sources.yaml   36 classes x 2 URLs. A dead link is a yaml edit, never a code change
taxonomy.json  anon id <-> real name, held-back flags
```

`data/`, `index/`, `results/` and `.env` are git-ignored. PDFs are never committed.

## Notes on the model

Runs on `claude-haiku-4-5` at `temperature=0` with structured output.

Haiku is not a cost compromise here — it is the only current model that accepts
`temperature`. Opus 5 and Sonnet 5 reject the parameter outright, so the determinism the
design calls for is only reachable on Haiku 4.5. Haiku in turn rejects
`output_config.effort`, which is therefore never sent. Both facts are asserted in
`llm.py`.

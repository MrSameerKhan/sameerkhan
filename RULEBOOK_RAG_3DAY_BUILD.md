# Rulebook-RAG — 3-DAY BUILD (v1, personal project)

Replaces `RULEBOOK_RAG_PROJECT_SPEC.md` for the first build. That file stays as the
long-term design; **build this one.** self-contained. Photograph raw markdown, not preview.

**Where:** personal laptop. **Data:** public IRS PDFs only. **Cost:** ~$5.
**No ICE data, no ICE account, no compliance question.**

---

## 0. THE CLAIM THIS BUILD EARNS

> Classifies documents into a 40-class taxonomy with **no trained classifier**. Class
> knowledge lives in a written rulebook retrieved at query time. Every decision cites a
> rule id that is resolved against the live index — unresolvable → forced abstention.
> A class added *after* the index was built is handled with no retraining.

Three results, in order of value:

1. **zero-shot** — 5 of the 40 classes are held out of the index, added at the end, and
   scored. A trained baseline is 0% there by construction.
2. **hybrid retrieval earns its place** — bm25_only vs dense_only vs hybrid vs +rerank.
3. **citation gating** — how often the model cited a rule that does not exist, all of
   which the gate converted to abstentions.

---

## 1. TWO DESIGN MOVES THAT MAKE THIS HONEST

### 1.1 Rules from the INSTRUCTIONS pdf, eval pages from the FORM pdf

IRS publishes both, at predictable paths:

```
form:         https://www.irs.gov/pub/irs-pdf/f<slug>.pdf
instructions: https://www.irs.gov/pub/irs-pdf/i<slug>.pdf
```

- **rulebook entry for class X is written ONLY from `i<slug>.pdf`**
- **eval pages for class X come ONLY from `f<slug>.pdf`**

Two different documents. No overlap. So the eval is not circular, and you never have to
generate eval data with an LLM. This is free anti-circularity — do not break it.

Record on every rulebook entry: `source_doc: "i1040sc.pdf"`.
Record on every eval page: `source_doc: "f1040sc.pdf"`.
**A build where these two are ever the same file is invalid.**

### 1.2 Anonymised class ids

The model already knows IRS forms. If class ids read `irs_1040_schedule_c`, good scores
may just be pretrained memory, and an interviewer will say so.

So: the index, the prompt, and the decision all use opaque ids — `CLASS_017`. The
human-readable name is held in a mapping file that is **never put in a prompt**.
The rulebook text still describes the class; only the *name* is hidden.

Run the eval twice:

| run | class ids | what it tells you |
|---|---|---|
| `anon` | `CLASS_017` | **the honest number.** Rulebook is the only knowledge source |
| `real_names` | `irs_1040_schedule_c` | how much pretrained knowledge was helping |

Report `anon` as the headline. The gap between them is a finding, not an embarrassment:
it quantifies pretraining leakage, which almost no portfolio project measures at all.

This is also what makes the project transfer to a proprietary taxonomy. Say that.

---

## 2. THE 40 CLASSES

IRS slugs. Verify each returns a PDF before relying on it; drop any that 404.

```
f1040      f1040sa    f1040sb    f1040sc    f1040sd    f1040se    f1040sf    f1040sse
f1040es    f1040x     f1065      f1065sk1   f1120      f1120s     f1120ssk1  fw2
fw9        fw8ben     f1098      f1099msc   f1099int   f1099div   f4506c     f2106
f4562      f4797      f4868      f8879      f8825      f8582      f8606      f8889
fss4       f2441      f8949      f8863      f5695      f2555      f3903      f8995
```

Confusable families you get for free — this is why 40 IRS forms is a better taxonomy
than 40 random document types:

- 1040 schedules A/B/C/D/E/F/SE — same header, same layout, different purpose
- 1099-MISC / 1099-INT / 1099-DIV
- 1065 Schedule K-1 / 1120-S Schedule K-1 — near-identical forms, different entity
- f1040 / f1040es / f1040x

**Held back for zero-shot: pick 5 with a fixed seed, do not hand-pick.** Commit the list.

---

## 3. REPO LAYOUT — 12 files

```
rulebook-rag/
    README.md
    RESULTS.md
    requirements.txt
    config.py        paths, K=8, RRF_K=60, ABSTAIN_THRESHOLD, model ids. no magic
                     numbers anywhere else
    llm.py           Anthropic client + StubLLM + disk cache + cost counter
    cli.py           every command
    fetch.py         sources.yaml -> data/pdfs/     (form + instructions)
    extract.py       PyMuPDF -> data/pages.jsonl    (one record per page)
    rules.py         instructions text -> rulebook/entries/CLASS_NNN.json
    bm25.py          hand-written Okapi BM25, ~40 lines, NO library
    index.py         embed rule clauses + build bm25 + manifest.json
    retrieve.py      bm25 + dense + RRF + rerank; mode switch for ablations
    decide.py        one LLM call (structured output) + gate.py logic inline
    evaluate.py      metrics + ablations + RESULTS tables
    taxonomy.json    40 classes, anon id <-> real name, held_back flags
```

`data/`, `results/`, `.env` are git-ignored. Never commit PDFs.

---

## 4. DAY 0 — ENVIRONMENT (1 hour, evening before)

```bash
# Python 3.11 from python.org, tick "Add Python to PATH"
python --version

mkdir rulebook-rag && cd rulebook-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # PowerShell

pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU FIRST
pip install anthropic sentence-transformers numpy scikit-learn pymupdf pyyaml requests matplotlib

# cache the two local models once (needs network; offline after)
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('cached')"

$env:ANTHROPIC_API_KEY="sk-ant-..."     # needs API credits, ~$10 is plenty
```

**Build `cli doctor` first.** Checks: python version · all imports · torch is CPU ·
both models load from cache · API key present · one 10-token call succeeds.
One clear message beats a confusing traceback on day 3.

**Also build `--stub-llm` in `llm.py` on day 0.** A fake LLM returning schema-valid
canned responses. It lets the whole pipeline run for $0 and makes every later day
debuggable without spending money.

---

## 5. DAY 1 — DATA AND RULES (~6 hrs)

1. **`sources.yaml`** — 40 classes × 2 urls (form + instructions). URLs in yaml, never
   in code, so a dead link is a yaml edit.
2. **`fetch.py`** — download, cache by url hash, 1s delay, be polite to irs.gov.
   404 → log and continue. Never let one dead link stop the run.
3. **`extract.py`** — PyMuPDF, one record per page:
   `{page_id, text, class_id, source_doc, kind: form|instructions, n_chars}`.
   Flag any page under 200 chars as image-only and exclude it.
4. **`taxonomy.json`** — 40 entries: `{anon_id, real_name, slug, held_back}`.
   Pick the 5 held-back with `random.Random(42)`.
5. **`rules.py`** — per class, one LLM call. Input:
   - the **instructions** text for that class (truncate to ~4000 tokens, head+tail)
   - the **anon ids and real names of its 3 nearest siblings** ← this is the important
     one. A rule that does not discriminate is worthless, so the model must be told
     what it is discriminating against.
   - NOT the form pdf. NOT any eval page.

   Output per §6 schema. Every clause gets a stable `rule_id`.
6. **Eval set** — every page of every `f<slug>.pdf`. Hold them out; they are never used to write rules.
   ~100: 2-3 pages per class.

**Checkpoint before bed:** `cli classify --stub-llm` runs end to end on one page.

---

## 5b. RULEBOOK ENTRY SCHEMA

```json
{
    "class_id": "CLASS_017",
    "document_type": "tax_schedule",
    "source_doc": "i1040sc.pdf",
    "definition": "Schedule attached to Form 1040 reporting profit or loss from a sole
                   proprietorship business.",
    "includes": [
      {"rule_id": "CLASS_017.inc.1",
       "text": "contains the heading 'Profit or Loss From Business'",
       "quote": "Profit or Loss From Business"}
    ],
    "excludes": [
      {"rule_id": "CLASS_017.exc.1",
       "text": "Does not report rental or royalty income; that is CLASS_019",
       "quote": "rental real estate"}
    ],
    "discriminators": [
      {"rule_id": "CLASS_017.disc.1", "vs_class_id": "CLASS_019",
       "text": "Reports sole-proprietor business income; CLASS_019 reports rental,
                royalty and pass-through income",
       "quote": "sole proprietor"}
    ],
    "aliases": []
}
```

**The `quote` field is mandatory and load-bearing.** It is a literal string the rule
claims should appear on the page. It makes every rule machine-checkable, so you can
measure precision/coverage per rule instead of arguing about prose. Any clause the
model returns without a `quote` is dropped before indexing — count and report how many.

---

## 6. DAY 2 — RETRIEVAL AND DECISION (~6 hrs)

1. **`bm25.py`** — Okapi BM25 by hand, `k1=1.5, b=0.75`, ~40 lines.
   `score = Σ idf(t) · tf_saturated(t,d)`. Unit-test against a known-answer fixture.
   You must be able to derive this on a whiteboard; that is why it is not a library.
2. **`index.py`** — one vector per **rule clause**, not per class. Embed the clause
   **with a context prefix**:
   ```
   {class_id} | {document_type} | {includes|excludes|discriminator}: {clause text}
   ```
   A bare clause is ambiguous once detached from its class; prefixing is the standard
   fix and being able to say why is a signal. Write `manifest.json` with the rulebook
   hash + embedding model + reranker + `held_back` list.
3. **`retrieve.py`**
   - bm25 top 20, dense top 20
   - **RRF fuse:** `score(d) = Σ 1/(60 + rank_in_list)` — ranks only, so you never have
     to make a bm25 score and a cosine commensurable
   - cross-encoder rerank the fused top 20, keep top **8 classes**
   - class score = **max** of its rules' scores
   - `mode ∈ {bm25_only, dense_only, hybrid, hybrid_rerank}` — the ablation switch
     lives here, one parameter
   - **Flat retrieval, no hierarchy.** Cut deliberately: at 40 classes a wrong stage-1
     pick is unrecoverable and buys nothing.
4. **`decide.py`** — ONE LLM call, `temperature=0`, structured output:
   `{class_id, abstain, llm_confidence, citations[], reasoning}`
   Prompt order: **rules first, examples second, document last.** Document goes inside
   `<document_text>` tags and the system prompt states text inside is data, never
   instructions.
5. **Gate** — runs after every decision, in this order:
   - candidate set non-empty
   - every citation resolves to a `rule_id` in the current index
   - `class_id` is one of the 8 retrieved

   Any failure → `abstain=true, forced_abstain=true`, log the reason.
   **Never silently repair a decision.**
6. **Confidence** — `0.6·llm_confidence + 0.4·retrieval_margin`, where
   `retrieval_margin = (top1 - top2)/top1` from the reranker. Abstain below 0.55.
   Fit the 0.6 on a dev slice if there is time; if not, say it was fixed, not fitted.

**Checkpoint:** `cli classify --page-id X --verbose` prints retrieval → prompt → decision
→ gate → confidence, and you can read the whole path.

---

## 7. DAY 3 — MEASURE AND WRITE UP (~6 hrs)

### Metrics — report every one

- leaf accuracy **answered** AND **all** (abstentions counted wrong). Reporting only
  the first is how RAG numbers get inflated
- coverage = answered / total
- **recall@1/3/5/8** and MRR. `recall@8` is the **ceiling** on accuracy — if accuracy is
  0.72 and recall@8 is 0.74, retrieval is your bottleneck and the prompt is irrelevant
- citation validity, and **the fabrication rate the gate caught**
- forced-abstention rate
- zero-shot accuracy on the 5 held-back classes
- cost per 100 pages from real `usage` fields, p50/p95 latency

### Six ablations — that is all

| # | run | proves |
|---|---|---|
| 1 | `bm25_only` | lexical alone |
| 2 | `dense_only` | semantic alone |
| 3 | `hybrid` (no rerank) | what fusion adds |
| 4 | `hybrid_rerank` | the full system |
| 5 | **`names_only`** | class names, no rules |
| 6 | **`prompt_stuffing`** | all 40 entries in the prompt, no retrieval. Report cost |

5 and 6 are the two that matter. 5 answers *"didn't the model already know these?"*.
6 answers *"why retrieve at all?"* — with a cost number, not an opinion.

### Zero-shot, last

```
cli build-index --exclude-held-back      # 35 classes
cli eval                                 # main numbers
cli add-held-back                        # +5 classes, rebuild index. NO RETRAINING
cli eval --only-heldback                 # <- the headline
```

⚠ `--limit N` takes the *first* N cases and will silently give you the wrong number
here. Use an explicit `--only-heldback` filter.

⚠ Held-back classes must be absent from the index **and** from exemplar retrieval.
Assert it in code.

### Write `RESULTS.md`

order: headline table · retrieval (recall@k + the ceiling sentence) · abstention ·
ablations · anon vs real_names · cost/latency · **20 worst errors, each tagged
`retrieval_miss` or `decision_error`** · **LIMITATIONS**.

The error split is the most useful number in the project — it says whether to fix
retrieval or the prompt, and it proves you diagnosed rather than tuned blindly.

Limitations must name: public forms only · no real traffic or users · single-label ·
pages · 40 classes · no multi-page bucketing. Saying this makes the rest credible.

---

## 8. IF YOU RUN OUT OF TIME — drop in this order

1. the cross-encoder rerank (ablation 3 becomes the full system)
2. the fitted confidence weight — hardcode 0.6 and say so
3. per-rule precision/coverage measurement
4. `prompt_stuffing`

**Never drop:** the eval set · `names_only` · zero-shot · the citation gate ·
answered-vs-all accuracy. Those five *are* the project.

---

## 9. CUT FROM THE BIG SPEC — and why

| cut | why |
|---|---|
| 400 classes → 40 | 40 IRS forms are naturally confusable; 400 was 5 days of yaml |
| hierarchical 2-stage retrieval | unrecoverable stage-1 failure, no benefit at 40 |
| `term_stats.py` log-odds mining | rules come from instructions, not from mined examples |
| LLM-generated synthetic documents | the form pdfs are labeled by construction. Free |
| `defects.py` taxonomy defect report | interesting, not load-bearing |
| 12 ablations → 6 | the other 6 were decoration |
| 500 eval cases → ~100 | 100 is enough to separate 0.65 from 0.85 |
| 3-level hierarchy → flat | 40 classes needs one level |

---

## 10. RESUME BULLET — fill after `RESULTS.md` exists

> **Rulebook-RAG — retrieval-grounded document classification** (personal project, 2026)
> — Classifies documents into a 40-class taxonomy **with no trained classifier**: class
> knowledge lives in a written, versioned rulebook retrieved at query time rather than in
> model weights, so **a new class is added by writing one rule and rebuilding the index —
> no retraining, no labeled data**. Hand-written **Okapi BM25** fused with dense
> embeddings by **reciprocal rank fusion**, then **cross-encoder reranking**; every
> decision cites a specific rule id resolved against the live index, and an unresolvable
> citation **forces abstention, making fabricated citations structurally impossible**.
> Class ids are anonymised so the model cannot fall back on pretrained knowledge of the
> forms — measured explicitly (<FILL> names-only vs <FILL> full system). Results on
> <FILL> held-out pages: **<FILL> accuracy at <FILL> coverage**, recall@8 <FILL>,
> **<FILL> zero-shot accuracy on 5 classes added after the index was built**; retrieval
> beat prompt-stuffing all 40 entries by <FILL> at <FILL>× lower cost per page.

Nothing goes on the résumé until `RESULTS.md` holds the number. That rule exists because
it was broken twice already.

---

## 11. WHAT THIS BUILD LETS YOU ANSWER

Not theory — experience. These are the questions that separate built from read-about.

- *"what surprised you?"* → the anon-vs-real-names gap, or recall@8 capping accuracy
- *"what did you measure and what did you do about it?"* → the retrieval-miss vs
  decision-error split, and which one you fixed
- *"why hybrid?"* → bm25_only X, dense_only Y, fused Z. Form identifiers are the
  strongest signal for document identity and dense embeddings are weakest exactly there
- *"why not put it all in the prompt?"* → ablation 6, with the cost multiple
- *"how do you know the citations are real?"* → the gate, plus the fabrication rate it caught
- *"Didn't the model already know these forms?"* → `names_only`, and the anon design
- *"what's the weakest part?"* → public forms, no real traffic, 40 classes, single-label

---

**END — build this, not the 2,368-line file.**

# Resume Change Request — Draft 2 Feedback

**To:** Jaya Waldia, Resume Writing Team, Naukri
**From:** Sameer Khan
**Ref:** `#VR#260817TS43026084_43453892#`
**Re:** `CustCopy3.pdf`, received 28 Aug 2026

---

Thank you for draft 2. Several things are clearly better and I want to name them before the
requests: the F1 correction is right, the placeholder sections are gone, the embedded images
are gone, the palette is down to three colours, and the bottom margin no longer runs to the
page edge. Please keep all of that.

This round has three parts, deliberately separated so nothing gets lost:

- **Part A — three blocking items.** Content that draft 1 had and draft 2 removed.
- **Part B — items from round 1 that were not applied.** Mostly §4 of my first document.
- **Part C — layout.** Measurable, and separate from the wording.

Ready-to-paste replacement text is given wherever it exists, so this should be a copy job
rather than a rewrite.

---

## Part A — Blocking

### A1. The P1 RCA bullet was deleted, not revised

Draft 1 had this bullet at ICE. Draft 2 has no equivalent anywhere; the only trace is the
phrase "incident RCA" inside a Profile Summary line.

My round-1 feedback asked you to *strengthen* it by adding the outcome. It was removed instead.
It is the only evidence on the page of cross-system production debugging, which is the single
most-tested skill in senior ML interviews.

**Please add, as a Key Result Area bullet:**

> Lead root-cause analysis for P1 production incidents — including one involving 21,096
> reported failures across 260K pages — tracing five downstream services through CloudWatch
> Logs Insights, ruling out model and infrastructure causes, and isolating a platform
> validation defect where one dropped page could null a 530-page file; the analysis cleared
> the model and established the actual failure rate at 2.3%.

### A2. The Highlights / Key Result Areas split says everything twice

643 classes, 1.7M requests/day, 0.62 → 0.92 across 946,355 pages, ~20% cost reduction,
20 → 118 SageMaker instances and the 0.975 F1 each appear in **both** halves of the ICE entry.
Six Highlights bullets plus four KRA bullets carry roughly six distinct facts between them.

That duplication is what pushed out the RCA bullet and the TensorFlow migration. Net ICE
content fell from 336 words in draft 1 to 257 in draft 2, and about 40% of what remains is a
restatement.

**Please merge them into a single Key Result Areas list**, present tense, using the seven
bullets in A3. If the template requires a Highlights strip, cap it at three bullets that appear
nowhere else in the entry.

### A3. Replacement text for the whole ICE entry

Present tense for ongoing responsibilities; past tense only for completed one-off events, as
agreed in round 1.

**ICE DATA SERVICES | Senior ML Engineer | Hyderabad | Nov 2023 – Present**

> - Own the end-to-end ML lifecycle — model development, evaluation, optimization, deployment,
>   monitoring and production operations — for a mortgage document classification platform
>   spanning 643 classes and 2.8M+ training pages from 49 encrypted lender sources, serving
>   1.7M requests/day and 45M+ requests/month on AWS SageMaker at 1.8-second p95 latency and
>   99.99% success; document verification turnaround fell from approximately 4 days to 4–5 hours.
>
> - Develop and scale a five-branch multimodal classification architecture combining ResNet50,
>   CNN, BiLSTM and a multi-head-attention Transformer I wrote from scratch over frozen GloVe
>   embeddings; distribute training across 20 A10G GPUs using Horovod, resolving host-memory
>   OOM and distributed all-reduce failures, achieving 0.959–0.972 weighted F1 across five
>   lender benchmarks.
>
> - Proposed and built the OCR-free successor using Donut/Swin with 75M parameters fully
>   fine-tuned through PyTorch Lightning DDP across 20 L40S GPUs in BF16, reaching 0.975
>   weighted F1 on 528K+ pages against 0.974 from the incumbent OCR-based production ensemble;
>   exported to ONNX after benchmarking 9 SageMaker instance types, cutting hourly
>   infrastructure cost by approximately 20%.
>
> - Recover production accuracy regressions across 946,355 pages by fixing a train-to-benchmark
>   taxonomy mismatch that took F1 from 0.62 to 0.92; established that 64% of residual errors
>   were valid predictions discarded by a confidence threshold, leading to calibrated per-class
>   selective prediction across all 643 classes.
>
> - Lead root-cause analysis for P1 production incidents — including one involving 21,096
>   reported failures across 260K pages — tracing five downstream services through CloudWatch
>   Logs Insights, ruling out model and infrastructure causes, and isolating a platform
>   validation defect where one dropped page could null a 530-page file; the analysis cleared
>   the model and established the actual failure rate at 2.3%.
>
> - Manage production ML operations and releases through the Jenkins-to-SageMaker CI/CD
>   pipeline — cross-account artifact promotion, model registration, staging validation, change
>   control, rollback and deployment — delivering four consecutive model releases and scaling
>   production infrastructure from 20 to 118 SageMaker instances during demand surges without
>   loss of reliability.
>
> - Modernized the production serving stack by migrating model exports from TensorFlow 1.x to
>   2.x, improving maintainability and framework alignment while supporting reliable production
>   inference.

---

## Part B — Round-1 items not applied

### B1. Six of the seven phrases in §4 are still missing

Three are already folded into the A3 text above (`from scratch`, `Proposed and built`,
`530-page file`, `cleared the model`). The remaining three:

| Where | Please use |
|---|---|
| Document Deduplication project | "Ran exact retrieval inside label and page-count groups, using many small exact searches instead of one approximate index" — replaces "Optimized exact retrieval through label/page-count grouping" |
| Zyclyx, scanner-audit bullet | Append: "…mandated bank-wide through an official bank circular, fixing the problem at the source instead of in preprocessing." |
| Profile Summary, bullet 1 | "…9+ years in Document AI, applied to one problem across three roles." |

These are not stylistic preferences. Each is a question I want an interviewer to ask, and the
generalised version removes the hook.

### B2. "AI Solution Architecture & Technical Leadership" is still in Core Competencies

Round 1 asked for this to be removed regardless of what else changed. I have never held a lead
or manager title, and a claim of technical leadership will be tested in interview.

**Please delete it.**

### B3. The Domain Expertise block was not added

None of these three terms appear anywhere in `CustCopy3.pdf`. For Document AI roles, domain
match is frequently what shortlists a candidate.

**Please add, in the sidebar under Core Competencies:**

> **DOMAIN EXPERTISE**
> - Mortgage & Loan File Documents
> - Retail Banking — Arabic / English
> - Identity Verification & KYC

Deleting B2 plus two more competency lines makes room without adding height.

### B4. Profile Summary is still six bullets

Round 1 asked for three. Draft 2 compressed 261 words to 186 but kept all six. The first hard
number now arrives at word 111 — better than 158, still a long way into a six-second scan.

**Please use these three bullets:**

> - Senior Machine Learning Engineer with 9+ years in Document AI, applied to one problem across
>   three roles: a 643-class mortgage classification platform serving 1.7M requests/day at
>   99.99% success, built on 2.8M+ pages from 49 encrypted lender sources.
>
> - Deep Learning and Multimodal specialist across PyTorch, TensorFlow, Transformers, ResNet50,
>   BiLSTM, Swin/ViT, Donut, LoRA/QLoRA, PEFT, SFT and RLHF/DPO, delivering up to 0.975 weighted
>   F1 through distributed multi-GPU training on Horovod and PyTorch Lightning DDP.
>
> - LLM and RAG engineer building retrieval, hybrid search, cross-encoder reranking and
>   structured generation systems, and an end-to-end MLOps practitioner across SageMaker, MLflow,
>   ONNX, Docker, Jenkins CI/CD and CloudWatch — from experimentation through production
>   operations, incident RCA and release governance.

### B5. Two hedges from §6 survive

"Growing focus on" and "strong understanding of" were both removed — thank you. "Exposure"
remains twice:

- "with end-to-end **exposure** across model development, evaluation, optimization…"
- "with **exposure** to modern GenAI and Agentic AI architectures"

The second also re-imports the Agentic AI aspiration that the first edit removed. Both are
handled by the B4 rewrite; if B4 is not taken, please cut the two words in place.

### B6. "Supported" contradicts "Own"

Highlight 1 reads "**Supported** a mortgage document classification platform"; the KRA bullet
four lines below reads "**Own** the end-to-end ML lifecycle". Merging per A2 removes the
conflict. If the Highlights strip stays, the verb must be "Own".

---

## Part C — Layout

Measured from the PDF's own content stream. These are separate from the wording and should be
quick.

### C1. Work experience is split by the sidebar *(highest priority in this part)*

The page-1 text layer extracts as: ICE bullets → `CORE COMPETENCIES` → `TECHNICAL SKILLS`.
Page 2 then opens directly on `PERSISTENT SYSTEMS` with no heading above it. A parser that maps
content to section headings can file Persistent, Zyclyx and VissIndia under "Technical Skills".

In draft 1 all four employers sat contiguously under one `WORK EXPERIENCE` heading, which was
correct.

**Please either** move the ICE entry to page 2 so the employment history is contiguous, **or**
repeat `WORK EXPERIENCE (CONT.)` at the top of page 2.

### C2. Page 2 line spacing is below single

Page 2 is set at 8.6pt leading on 9pt type — a ratio of 0.96, tighter than the type size.
Page 1 is correct at 1.21.

**Please set a minimum of 1.15 line spacing across both pages.** If that overflows, the space
comes from the A2 duplication, not from further compression.

### C3. My current role is in the narrowest column

ICE runs down a 343pt column on page 1 while Persistent, Zyclyx and VissIndia get the full 549pt
width on page 2. **Please give the current role the full width.**

### C4. Side margins

33pt left and 28pt right put page 2 at 108 characters per line. **Please widen to 45–54pt** —
it shortens the measure and reduces the crammed look without cutting a word.

### C5. Heading scale

Section headings are 12pt against 9pt body. **Please raise to 13–14pt** in the navy already in
the palette. No extra vertical space, materially better scanning.

### C6. "Supervised Fine-Tuning" breaks across a line

In Technical Skills it extracts as `Supervised Fine-` / `Tuning (SFT)`, so a keyword search for
"fine-tuning" will not match it. Draft 1 had the same problem on "Model Fine-Tuning", which is
now fixed — this is the same bug in a new place. **Please reflow so the term sits on one line.**

### C7. Page 2 has no identity

**Please add a single header line** with name and phone number, in case the pages are separated
or printed.

---

## Part D — Two items not raised in round 1

1. **Gap, Mar 2021 – Aug 2021.** Zyclyx ends March, Persistent starts August. Neither draft
   accounts for the five months. Either a one-line note or year-only dates on that pair would
   close it — please advise what you normally recommend.

2. **7.1M vs 2.8M pages.** The deduplication project cites a 7.1M-page corpus while the profile
   cites 2.8M+ pages at ICE. Both are correct for different scopes (full mortgage corpus vs.
   training set), but on one page it reads as an inconsistency. Suggest: "7.1M-page production
   corpus" in the project bullet.

---

## Part E — What draft 2 got right, please keep

1. The F1 correction, and the exact figure 946,355.
2. Removal of Certifications, Date of Birth, Personal Details and the Career Timeline box.
3. The three-colour palette — black, navy, grey. The cyan and the white-on-fill text are gone.
4. Zero embedded images; draft 1 had eight on page 1.
5. Bottom margin moved off the page edge, 7pt to 22pt.
6. `CORE COMPETENCIES` now extracts adjacent to its own items — the draft-1 parsing bug is fixed.
7. The Technical Skills categorisation, unchanged from draft 1.
8. Present tense in the Key Result Areas bullets.

---

## Outstanding questions from round 1

1. Has this template been tested against non-Naukri ATS parsers — Workday, Greenhouse, Taleo?
   (Round-1 §7, unanswered. C1 makes it more pressing.)
2. Can you send the **editable `.docx`** with draft 3? Every Part C item is a few minutes in
   Word and a full round trip in PDF. (Round-1 §10.2, unanswered.)

Happy to walk through any of this on a call.

— Sameer

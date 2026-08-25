# Resume Comparison — Mine vs Naukri Draft 1

> Prepared 22 Aug 2026, for the Naukri sync on 23 Aug, 11:00 AM.
> **A** = `resume_sameer_khan_sr_ML.pdf` (mine) · **B** = `CustCopy2.pdf` (Naukri draft 1)

---

## Verdict

B is not bad work. It is a competent portal-optimised rewrite with **real keyword gains** — but it
ships with **three defects that must be fixed**, and it dilutes the specific phrases that make
me interviewable. Do not accept it as-is. Do not throw it away either.

**Recommended outcome: hybrid.** B's structure and keyword coverage, A's sentences.

---

## 1. BLOCKING DEFECTS in B — raise these first tomorrow

### 1.1 Unfilled template placeholders shipped in the deliverable

```
CERTIFICATIONS
  • Please mention if any          <-- template instruction, left in
PERSONAL DETAILS
  • Date of Birth: DD'MM'YY        <-- placeholder, left in
```

This is a QC failure on a paid engagement. Also: the file is named `CustCopy2.pdf`
("customer copy 2") — a template artifact, not a candidate filename. If this had gone
to a recruiter it would have read as careless. **Ask what their proofing step is.**

### 1.2 FACTUAL / LOGICAL ERROR in the taxonomy bullet

| | Text |
|---|---|
| **A (correct)** | "Fixed a train/benchmark taxonomy mismatch **that took F1 from 0.62 to 0.92**" |
| **B (broken)** | "identifying a train-to-benchmark taxonomy mismatch **that had reduced F1 from 0.62 to 0.92**" |

"Reduced F1 from 0.62 to 0.92" is incoherent — 0.62 → 0.92 is an *increase*. B's sentence says
the mismatch caused the improvement. Any ML interviewer catches this in one read, and the
damage is not the typo, it is that **the candidate apparently did not notice**. This is the
single most dangerous line in the document.

### 1.3 Tense error on the current role

ICE is `Nov 2023 – Present`, but every B bullet is past tense: "**Owned** the end-to-end ML
lifecycle", "**Managed** production ML operations". Reads as though I have left.
A uses present tense — "**Own** ICE's production mortgage document classifier end to end" —
which is correct and also stronger.

---

## 2. What each has that the other does not

| Element | A (mine) | B (Naukri) |
|---|---|---|
| Layout | Single column | Two column + sidebar |
| Objective / targeting statement | ✗ | ✓ (~55 words) |
| Profile summary | 4 lines, ~75 words | 6 bullets, ~270 words |
| Core Competencies keyword boxes | ✗ | ✓ (12 boxes) |
| Career Timeline graphic | ✗ | ✓ |
| Certifications section | ✗ | ✓ but EMPTY |
| Personal details / DOB | ✗ | ✓ (placeholder) |
| "One problem across three roles" thesis | ✓ | ✗ **dropped** |
| "Transformer **I wrote from scratch**" | ✓ | ✗ diluted |
| "**Proposed** and built the OCR-free successor" | ✓ | ✗ dropped |
| "RCA **that cleared the model**" | ✓ | ✗ dropped |
| "one dropped page could null a **530-page file**" | ✓ | ✗ dropped |
| Dedup design *reasoning* (why exact beats ANN) | ✓ | ✗ dropped |
| Dedup + signature verification surfaced on page 1 | ✗ | ✓ |
| Keyword coverage (GenAI, Agentic, Info Extraction…) | thinner | ✓ stronger |
| Explicit "Senior/Lead IC, not manager" signal | ✗ | ✓ |
| Total word count | ~1,100 | ~1,555 (+40%) |
| Words before the first hard number | ~30 | ~370 |

---

## 3. Line-level dilution — where B weakened A

| A (mine) | B (Naukri) | What was lost |
|---|---|---|
| "a multi-head-attention Transformer **I wrote from scratch** over frozen GloVe" | "**custom** multi-head-attention Transformer **components**" | Personal authorship → vague adjective. This is the line that earns the deep-dive interview. |
| "**Proposed and built** the OCR-free successor" | "**Developed** an OCR-free Donut/Swin model" | Initiative. "Proposed" is the seniority signal; "developed" is task execution. |
| "Led the P1 RCA **that cleared the model**" | "Led technical RCA for a P1 production regression" | The outcome. Without it, it is process, not result. |
| "an input validator was rejecting whole documents, so **one dropped page could null a 530-page file**" | "rejected complete documents when individual pages failed" | The concrete image. 530 is what a reader remembers. |
| "I ran searches inside label and page-count groups **so I could use many small exact searches instead of one approximate index**" | "Optimized exact retrieval through label/page-count grouping" | The engineering *judgement*. For a senior role the reasoning IS the signal; the mechanism alone is not. |
| "the bank issued it as a circular… **That fixed the problem at the source instead of in preprocessing**" | "established a standardized configuration subsequently mandated bank-wide" | The systems-thinking punchline. |
| "9 years in Document AI… **applied to one problem across three roles**" | (absent) | My single most differentiating sentence. Nine years of depth on one problem is rare; B replaced it with generic breadth claims. |

---

## 4. Where B genuinely wins — do not discard these

1. **Keyword coverage.** B adds Generative AI, Information Extraction, Model Optimization,
   Production Monitoring, Confidence Calibration, Image Preprocessing, Instruction Tuning.
   For portal + ATS keyword matching this is real value, and it is what I paid for.
2. **Dedup and signature verification promoted to page 1.** In A they are buried on page 2.
   B surfaces them in the profile summary. Correct call.
3. **Zyclyx made legible.** A crams classifier + detection + OCR + NER + signature verification
   into one dense bullet. B splits them. B is more skimmable here.
4. **Explicit IC targeting.** "Strong technical individual contributor with an
   architecture-oriented approach" pre-empts the "do you want to manage?" question.
5. **Scannability.** Bolded metrics land better in B's denser format than I expected.

---

## 5. Where my instinct ("overly written") is right — and where it is wrong

**Right, but the problem is not total length.** Both are 2 pages. The real problem is
**front-loading**: B spends ~370 words on objective + profile summary + competency boxes
*before* a single piece of evidence. A recruiter's first 8 seconds hit adjectives
("architecture-oriented", "end-to-end exposure") instead of "643 classes, 1.7M requests/day".
A hits the hard numbers in the first two lines.

**Wrong about the content volume itself.** B's *work experience* section is barely longer
than A's and is fine. The bloat is concentrated in the top-of-page marketing block and the
12 competency boxes, not in the body.

**Also flag — hedge language.** B writes "with **knowledge and practical exposure to**
LLM-based applications" and "with **growing focus on** Agentic AI". Senior hiring managers
read "exposure" and "growing focus" as "has not actually done this". A never hedges.
Either state it concretely or cut it.

---

## 6. Target: the hybrid

| Section | Take from | Note |
|---|---|---|
| Header / contact | B | Cleaner, includes full URLs |
| Objective statement | B, cut to 2 lines | Keep the IC signal, drop the adjectives |
| Profile summary | **A** | 4 lines. Restore "one problem across three roles". |
| Technical skills | B | Best keyword coverage. Keep. |
| Core competencies boxes | **CUT** or reduce to 6 | 12 unevidenced phrases is keyword stuffing |
| Career timeline graphic | **CUT** | Duplicates Work Experience; costs space |
| Work experience bullets | **A's sentences** | Restore every phrase in §3 |
| Zyclyx bullet split | B | B's structure, A's wording |
| VISSINDIA | **A** | B's 3 bullets are padding — bullets 1 and 3 say the same thing |
| Projects | **A** | Restore the dedup reasoning |
| Certifications | **CUT** | Empty section advertises a gap |
| Personal details / DOB | **CUT** | Liability for international roles, adds nothing domestically |
| Layout | Decide — see below | |

### Open decision: one column or two?

Two-column with a sidebar is a **known ATS parsing risk** — Workday / Greenhouse / Taleo can
interleave sidebar text into the body. Naukri's own parser will handle their format; a
target company's may not.

**Ask them directly tomorrow: "has this exact template been tested against Workday and
Greenhouse parsers, or only against Naukri's?"** Their answer tells me how much of the
₹10K was template-picking versus real ATS engineering.

Pragmatic option: keep **two versions** — B's two-column for the Naukri portal, A's
single-column for direct company applications.

---

## 7. Agenda for the 11:00 call

Open cooperative, not adversarial — I want draft 2 to be good, not to win an argument.

1. **Placeholders.** "Certifications and DOB still have template text. What's the proofing step?"
2. **The F1 line.** "This one is a factual error — 0.62 to 0.92 is an increase, not a reduction.
   Please use my original phrasing." *(Non-negotiable.)*
3. **Tense.** "ICE is my current role — bullets should be present tense."
4. **Restore the specific phrases.** Hand them §3 as a list. Frame it as:
   "These aren't stylistic. Each one is a question I want an interviewer to ask me."
5. **Front-loading.** "Can we cut the top block so hard numbers appear above the fold?"
6. **ATS question.** Workday / Greenhouse parser testing — see §6.
7. **Hedge words.** "Please remove 'exposure to' and 'growing focus on'."
8. **Ask for the editable source file** (.docx), not just PDF, so I can iterate without a round trip.

### One thing to be careful about

They are the paid vendor, but **I am the one who has to defend every line in an interview.**
Where we disagree on style, they can win. Where we disagree on a **technical claim**, I win —
because I am the one who will be asked "walk me through this."


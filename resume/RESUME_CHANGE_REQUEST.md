# Resume Change Request — Draft 1 Feedback

**To:** Jaya Waldia, Resume Writing Team, Naukri  
**From:** Sameer Khan  
**Ref:** `#VR#260817TS43026084_43453892#`  
**Re:** `CustCopy2.pdf` / `CustomerCopy.docx`, received 21 Aug 2026

**Attached:** `resume_sameer_khan_v2.pdf` — a reference layout I have prepared showing all of
the changes below already applied. It is a reference for the wording and ordering, not a
replacement for your template; please treat it as the target content.

---

Thank you for the draft — the structure is a clear improvement on what I sent you, and
several of your decisions are better than mine (listed in §8). I have gone through it line
by line and prepared this document so my feedback is specific and testable rather than a
matter of taste.

Requests are ranked. **§1 is a factual correction and is not negotiable.** §2–§6 are
strongly evidenced. §7 is a question, not a request.

---

## 1. Factual error — the F1 figure reads backwards *(must fix)*

**Your draft, ICE bullet 4:**

> "Improved model performance across 946K+ pages by identifying a train-to-benchmark
> taxonomy mismatch **that had reduced F1 from 0.62 to 0.92**"

"Reduced from 0.62 to 0.92" describes an *increase*. As written, the sentence says the
defect caused the improvement.

**What actually happened:** the taxonomy mismatch had depressed F1 **to** 0.62. After I
fixed it, F1 rose **to** 0.92.

**Please use:**

> "Recovered a production accuracy regression across 946,355 pages by fixing a
> train/benchmark taxonomy mismatch **that took F1 from 0.62 to 0.92**; further established
> that 64% of residual errors were valid predictions discarded by a confidence threshold,
> leading to calibrated per-class selective prediction across all 643 classes."

**Why this matters more than the wording:** every interviewer for these roles is an ML
practitioner. They will notice within one read. The risk is not the typo — it is that it
implies I did not check my own numbers.

---

## 2. My current role is written in the past tense

ICE Data Services is dated **Nov 2023 – Present**, but every bullet reads `Owned`,
`Managed`, `Modernized`, `Developed`. On a current role this reads as though I have left.

**Please convert the ICE bullets to present tense** — `Own`, `Manage`, `Lead` — and keep
past tense only for completed one-off events (the P1 RCA, the TF1→TF2 migration).

Prior roles stay in past tense, correctly.

---

## 3. Core Competencies — please remove, and use the space for domains

I want to be precise here, because your cover note gives a specific rationale for this
block: *"keywords mentioned in Core Competencies and across Organisational Experience have
greater chances of finding a match in Applicant Tracking Systems."*

I tested that claim against your own document. I extracted every content word from the 12
competency phrases and searched for each one in the rest of the resume.

| Competency phrase | Its keywords, counted elsewhere in the same resume |
|---|---|
| Document AI & Intelligent Document Processing | document×24, processing×9 — *only new word: "intelligent"* |
| Machine Learning & Multimodal Deep Learning | machine×4, learning×8, multimodal×5, deep×4 |
| Computer Vision & OCR-Free AI | computer×4, vision×5, ocr-free×3 |
| NLP & Information Extraction | nlp×1, information×2, extraction×4 |
| LLM Engineering & Generative AI | llm×5, engineering×4, generative×4 |
| RAG & Enterprise Retrieval Systems | rag×6, enterprise×2, retrieval×7, systems×3 |
| Transformer Architecture & Model Fine-Tuning | transformer×5, architecture×7, model×22, fine-tuning×1 |
| Distributed ML & Large-Scale Model Training | distributed×5, large-scale×2, model×22, training×9 |
| MLOps & Production AI Engineering | mlops×4, production×18, engineering×4 |
| AWS SageMaker & Cloud AI Engineering | aws×6, sagemaker×8, cloud×3, engineering×4 |
| Model Optimization, Inference & AI Reliability | model×22, optimization×4, inference×2, reliability×2 |
| AI Solution Architecture & Technical Leadership | solution×5, architecture×7, technical×3 — *only new word: "leadership"* |

**Result: 10 of the 12 phrases contribute no keyword that is not already in the document**,
usually several times over, and mostly in the Technical Skills block immediately below them.
The only two new words the block adds are **"intelligent"** and **"leadership"** — neither
of which a recruiter searches for.

### There is also an extraction problem

I pulled the text layer out of `CustCopy2.pdf` — this is what a parser receives:

```
CORE COMPETENCIES
EDUCATION
  B.Tech. - Computer Science & Engineering...
CERTIFICATIONS
PERSONAL DETAILS
CAREER TIMELINE
Document AI & Intelligent Document Processing
Machine Learning & Multimodal Deep Learning
...
```

The `CORE COMPETENCIES` heading is separated from its own items, which land after
`CAREER TIMELINE`. A keyword-matching ATS still finds the strings, but any parser that maps
content to sections files them under the wrong heading.

The same extraction also yields **`Model FineTuning`** — the hyphen is destroyed by the line
break inside the box. A search for "fine-tuning" or "fine tuning" will not match it.

### One competency I specifically want removed regardless

**"AI Solution Architecture & Technical Leadership."** I have never held a lead or manager
title. If a resume claims technical leadership, the interview will test it, and I would
rather compete on what I can evidence.

### What to put in that space instead

**Domain Expertise** — three items, none of which appear anywhere else in the document:

- Mortgage & Loan File Documents
- Retail Banking — Arabic / English
- Identity Verification & KYC

For Document AI roles, domain match is frequently what shortlists a candidate, and this is
genuinely absent from the current draft.

**If the block must stay for template reasons:** please trim it to 5–6 items, drop the
Technical Leadership entry, and build it as a **single-column table with cell shading rather
than text boxes or SmartArt** — drawing objects sit outside the document flow and many
parsers skip them entirely, whereas table cells are extracted reliably.

---

## 4. Please restore seven specific phrases from my original

These are not stylistic preferences. **Each one is a question I want an interviewer to ask
me**, because each is something I can talk about for ten minutes. Generalising them removes
the hook.

| My original | Your draft | What is lost |
|---|---|---|
| "a multi-head-attention Transformer **I wrote from scratch**" | "**custom** multi-head-attention Transformer **components**" | Personal authorship. This single phrase generates more technical interviews than anything else on the page. |
| "**Proposed and built** the OCR-free successor" | "**Developed** an OCR-free Donut/Swin model" | Initiative. "Proposed" says I identified the opportunity; "developed" says I was assigned it. For a senior IC role that distinction is the whole point. |
| "Led the P1 root-cause analysis **that cleared the model**" | "Led technical RCA for a P1 production regression" | The outcome. Without it the bullet describes activity, not result — and the result was that my model was exonerated. |
| "an input validator was rejecting whole documents, so **one dropped page could null a 530-page file**" | "rejected complete documents when individual pages failed" | The concrete image. "530-page file" is what a reader remembers an hour later. |
| "I ran the searches inside label and page-count groups **so I could use many small exact searches instead of one approximate index**" | "Optimized exact retrieval through label/page-count grouping" | The engineering judgement. At senior level the reasoning *is* the signal; the mechanism alone reads as a tool list. |
| "the bank issued it as a circular… **That fixed the problem at the source instead of in preprocessing**" | "established a standardized configuration subsequently mandated bank-wide" | The systems-thinking punchline — I fixed a data problem upstream instead of patching it in code. |
| "9 years in Document AI… **applied to one problem across three roles**" | *(absent)* | My single most differentiating sentence. Nine years of depth on one problem is rare; the draft replaces it with breadth claims that every candidate makes. |

---

## 5. Hard numbers appear too late

Counting words from the top of page 1 to the first concrete metric:

| Version | Words before the first hard number |
|---|---|
| Your draft | **~370** (Job Objective + 6 Profile Summary bullets + competency boxes) |
| My original | ~30 |

A recruiter's first pass is short. In the current draft it lands on *"architecture-oriented,"
"end-to-end exposure," "robust model architectures"* — phrasing that does not distinguish me
from any other applicant. **"643 classes at 1.7M requests/day with 99.99% success"** does.

**Request:** compress the Job Objective to two lines and the Profile Summary from six bullets
to three, so a hard number appears in the first two lines of the page.

I am not asking to shorten the resume overall — your Organisational Experience section is a
good length. The density is concentrated entirely in the top block.

---

## 6. Three hedging phrases to remove

| Current text | Problem |
|---|---|
| "with knowledge and **practical exposure to** LLM-based applications" | "Exposure" is read by senior hiring managers as *has not actually done this*. |
| "with **growing focus on** Agentic AI and modern GenAI architectures" | Signals aspiration, not capability — and there is no agentic work on the resume to support it. |
| "with a **strong understanding of** designing scalable ML pipelines" | "Understanding" is weaker than the bullets below it, which describe systems I actually built and run. |

**Request:** state these concretely or cut them. I would rather claim less and defend all of it.

---

## 7. Question, not a request — has this template been parser-tested?

The two-column layout with a left sidebar is visually strong. My only concern is that some
enterprise ATS platforms interleave sidebar text into the body when extracting.

**Has this specific template been tested against Workday, Greenhouse and Taleo parsers, or
against Naukri's own parser?** I am not assuming there is a problem — I would just like to
know, because most of my applications will go through company portals rather than the Naukri
portal.

If it has not been tested, I am happy to keep two versions: your designed layout for the
Naukri portal and recruiter email, and a single-column variant for strict ATS portals. That
is a good outcome, not a complaint.

---

## 8. The blue fields you asked me to complete

| Field | My answer |
|---|---|
| **Certifications** | I have none. **Please remove the section entirely** rather than leaving it blank — an empty heading draws attention to the gap. |
| **Date of Birth** | **Please remove this field.** Several of my target employers are US/UK-headquartered, where DOB on a resume is discouraged and occasionally triggers compliance filtering. It adds nothing for domestic applications either. |
| **Languages Known** | English, Hindi, Telugu, Urdu *(please confirm this list is correct before it goes in)* |
| **"over 9 years of experience"** | Correct. Feb 2017 – present. |

---

## 9. What I want kept from your draft

To be clear that this is not a rejection — these are improvements on what I sent you, and I
want them preserved:

1. **Technical Skills keyword coverage.** Your categorisation is better than mine, and you
   added terms I had missed — Information Extraction, Model Optimization, Production
   Monitoring, Confidence Calibration, Image Preprocessing, Instruction Tuning. Keep all of it.
2. **Deduplication and signature verification promoted to page 1.** In my version both were
   buried on page 2. You were right to surface them.
3. **The Zyclyx role split into separate bullets.** I had crammed classifier, detection, OCR,
   NER and signature verification into one dense bullet. Yours is far more readable.
4. **The individual-contributor targeting statement.** Useful — it pre-empts the "do you want
   to move into management?" conversation. I only want it shorter, not gone.
5. **Reverse-chronological structure and the overall two-page discipline.**

---

## 10. Questions for you

1. Has this template been tested against non-Naukri ATS parsers (§7)?
2. Can you send the **editable `.docx`** with each revision, so I can check wording without a
   round trip?
3. If the Core Competencies block is a fixed requirement of the service, can it be built as a
   shaded table rather than text boxes (§3)?
4. Is there a length constraint I should work within if I want the top block compressed (§5)?

---

## Appendix — how the evidence in §3 and §5 was produced

- **Keyword counts:** every content word was extracted from the 12 competency phrases,
  stop-words removed, then counted with word-boundary matching against the full text of
  `CustCopy2.pdf` excluding the competency block itself.
- **Extraction order and the `FineTuning` artifact:** taken from the PDF's own text layer —
  the same layer an ATS reads. Reproducible from the file you sent.
- **Word-position counts in §5:** counted from the first word of the document to the first
  numeric metric.

Happy to walk through any of this on a call.

— Sameer

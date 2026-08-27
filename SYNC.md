# SYNC — Cross-Machine Handoff

> **Update this file before every `git push`. Read it after every `git pull`.**
> This is the only file that communicates state between Mac and Windows.

---

## Last Session

| Field | Value |
|-------|-------|
| **Machine** | Mac |
| **Date** | 28 August 2026 |
| **What I did** | Resume work. Compared my resume vs Naukri's paid draft 1 (`CustCopy2.pdf`) — found 3 blocking defects in theirs (unfilled template placeholders, a factually inverted F1 bullet, past tense on a current role). Built the hybrid in two variants: **v2 designed** (two-column, sidebar, colour — the one to send) and **v1 plain** (single-column, ATS-safe fallback). Analysis in `resume/RESUME_COMPARISON.md`. |
| **Files changed** | `resume/RESUME_COMPARISON.md`, `resume/resume_hybrid_MASTER.md`, `resume/resume_hybrid_v1.html`, `resume/resume_sameer_khan_v2.html`, `resume/RESUME_CHANGE_REQUEST.md` (all new) |

> **Correction (25 Aug):** the blue placeholders in Naukri's draft (`Please mention if any`, `DD'MM'YY`) are **deliberate fill-in markers** — their cover email asks the customer to complete them. Earlier I called this a QC failure. It is not. Do not raise it.

> ⚠️ Reminder from 22 Aug: three commits shipped without updating this file. **Update it in the same commit as the work, not later.**

---

## Next Task

```
What:   Reply to Naukri with the change request + reference resume
Attach: resume/RESUME_CHANGE_REQUEST.pdf   (6 pages, 10 sections, evidence-backed)
        resume/resume_sameer_khan_v2.pdf   (2 pages, the target layout)
To:     resumeservice@naukri.com — DO NOT change the subject line
Subject must keep: #VR#260817TS43026084_43453892#
Before sending: confirm the Languages list in section 8 is correct

Then:   Drill 01 — open since 14 Aug, never attempted
File:   code_practice/11_interview_drills/01_multihead_attention.py
Goal:   13/13 PASS, 20 min, no reference
```

---

## ACTIVE ARC — Interview Prep (started 14 Aug 2026)

**Context:** self-assessed weak in all four areas: resume defence, LLM/GenAI theory, ML system design, behavioural/STAR. Format chosen: **coding drills as the spine**, other three folded in around them.

**Timeline (rev. 22 Aug):** nothing scheduled yet. The original "1–2 weeks" clock is off. Days below are an *order*, not dates — runway is intact, so depth over speed. But drills have slipped 8 days with zero attempted; theory is not a substitute for saying it out loud or typing it cold.

### ACTIVE TRACK — LLM Architecture (scope set 25 Aug)

Full tracker: **`code_practice/11_interview_drills/MASTERY_PLAN.md`**

**Scope: architecture only.** LoRA/QLoRA and RLHF/DPO are parked — adaptation and alignment, not
architecture. Theory already exists for every board; the gap is recall, not reading.

Each board passes 4 gates: **G1 draw · G2 hand-compute · G3 code in 20 min · G4 defend aloud.**
No board advances until all four pass.

| # | Board | G1 | Status |
|---|-------|----|--------|
| 1–5 | RNN → LSTM → GRU → Attention → Transformer **encoder** | — | ✅ drawn · G3 outstanding |
| 6 | **Transformer Decoder** ← NEXT | 25 | ⬜ |
| 6b | Encoder w/ multi-head, d_model=4 — **theory written 28 Aug**, board not drawn | 20 | 📄 |
| 7 | Tokenization → Embedding | 15 | ⬜ |
| 8 | BERT | 20 | ⬜ |
| 9 | GPT | 20 | ⬜ |
| 10 | T5 / BART *(= the Donut decoder)* | 25 | ⬜ |
| 11 | Modern LLM block — pre-LN, RMSNorm, SwiGLU, RoPE, GQA | 25 | ⬜ |
| 12 | Attention at scale — Flash, KV cache, PagedAttention | 25 | ⬜ |
| 13 | Long context — RoPE scaling, ALiBi, SWA | 20 | ⬜ |
| 14 | Mixture of Experts | 20 | ⬜ |
| 15 | Decoding + speculative | 20 | ⬜ |

Boards 1–5 live in `4.nlp/03_sequence_models/whiteboard/`.
Boards 6–15 go in **`5.transformers/whiteboard/`**, continuing the numbering.

**Done means:** given any model card (Llama / Mistral / DeepSeek / Qwen), draw the whole model
from memory — tokens → vectors, each block, what changed since 2017 and why, how generation runs,
where the memory goes.

Every drill is a literal "I hand-wrote / I built from scratch" claim on the resume — drilling them **is** resume defence.

### Resume — RESOLVED 24 Aug

Hybrid built. `resume/resume_hybrid_MASTER.md` is the **content SSOT** — edit that, then mirror into the HTML.

| File | Use |
|---|---|
| `resume_sameer_khan_v2.html` | **Primary.** Designed two-column w/ sidebar + metrics. Send this. |
| `resume_hybrid_v1.html` | Single-column ATS-safe fallback for strict Workday/Greenhouse portals |
| `resume_hybrid_MASTER.md` | Plain-text content source — hand this to Naukri |

**To make the PDF — scripted (preferred):**
```bash
cd resume
/private/tmp/.../scratchpad/pdfenv/bin/python make_pdf.py resume_sameer_khan_v2.html
```
`make_pdf.py` is committed in `resume/`. It needs a playwright venv:
```bash
python3 -m venv pdfenv && ./pdfenv/bin/pip install playwright pypdf
./pdfenv/bin/playwright install chromium
```
(the scratchpad venv is temporary — recreate it anywhere, incl. Windows.)

**Manual fallback:** open the .html in Safari → File → Print → tick **"Print backgrounds"**
→ A4, scale **100%** → PDF ▾ → Save as PDF. Without that checkbox the navy band and tinted
sidebar render white.

**Verified 25 Aug** by rendering and inspecting: 2 pages, page-1 main slack 19.7mm /
sidebar 12.9mm, page-2 slack 16.0mm. PDF text layer extracts main content BEFORE sidebar
(correct ATS reading order), 719 words selectable on page 1, **0 ligature glyphs** —
`font-variant-ligatures:none` is set so "first" does not extract as "ﬁrst".
Old files kept for reference only: `resume_sameer_khan_sr_ML.pdf`, `CustCopy2.pdf` (Naukri draft 1).

### Open Finding — 02_INTERVIEW_PACK.md is stale

Written against an older resume. Must be rewritten before any interview. Confirmed mismatches:

| Pack says | Final resume says |
|-----------|-------------------|
| "8 years", "94% accuracy", "60% RCA reduction" | 9 years; weighted F1 0.959–0.972 and 0.975; P1 RCA = 5 services / 21,096 pages / 2.3% true failure rate |
| RAG project = FAISS + Streamlit + `llama3.2:1b` | **Rulebook-RAG**: 36 classes, hand-written BM25 + RRF (+5.2 pts), cross-encoder (+3.9 pts), 0 invalid of 775 |
| "You have done QLoRA with Mistral-7B" | Not on the resume — **and Phase 09 is parked, so it cannot be defended. Do not claim this.** |

Zero coverage in the pack for: Rulebook-RAG, ONNX export, 643 classes, dedup pipeline, Union-Find, per-class thresholds, Textract, L40S. Scheduled for days 12–14.

---

## Machine Differences (matters for drills)

| | Mac (here) | Windows |
|---|---|---|
| **torch** | 2.12.0, CPU | 2.5.1 + cu121, GTX 1650 Ti |
| **Good for** | Drills 1–8 (all CPU, tiny tensors, seconds to run) | GPU sessions, Phase 05/10 |
| **Known block** | — | Phase 09 fine-tuning parked: torch 2.6 not on cu121, trl 1.6 meta-tensor bug |

All interview drills are **deliberately CPU-only and seed-fixed** — they run identically on both machines. No GPU needed, no environment drift.

---

## Active Learning Arc

| Layer | Current Focus | Status |
|-------|--------------|--------|
| Theory | All 11 folders complete · `4.nlp/03_sequence_models/` end-to-end arc added 22 Aug (RNN→LSTM→GRU→Attention→Transformer, hand-computed) | ✅ Done |
| Code Practice | Phase 05 Transformers (S1-S7 exist) → Phase 06-10 also exist | 🔧 Need to run sessions |
| Root Packs | 00_HUB, 01_CAREER, 02_INTERVIEW, 03_LEARNING | ✅ Done |

---

## Code Practice Phase Status

| Phase | Topic | Sessions | Confirmed Run | Notes |
|-------|-------|----------|--------------|-------|
| 05 | Transformers | S01-S07 | ✅ All 7 run | GPU, GTX 1650 Ti |
| 06 | LLMs | S01-S03 | ✅ All 3 run | OpenAI gpt-4o-mini; S01 also tested on Ollama |
| 07 | RAG | S01-S05 | ✅ All 5 run | faiss-cpu + rank-bm25 installed |
| 08 | Agents | S01-S04 | ✅ All 4 run | langchain-openai + langgraph installed |
| 09 | Fine-tuning | S01-S06 | ⏸ Parked | torch 2.6 not on cu121; trl 1.6 meta tensor bug with 2.5.1 |
| 10 | Document AI | S01-S04 | unknown | Check _details.md badges |

---

## Pending Cleanup

- [ ] Delete `5.transformers/02_models/04_efficient_transformers copy.md` (stale duplicate)
- [ ] Move scripts out of `junk/` → root or `scripts/`
- [ ] Update `progress.md` to reflect actual session state

---

## How to Update This File

After any work session, before `git push`, update only these blocks:

**Last Session** — overwrite machine, date, what you did, files changed.

**Next Task** — overwrite with exactly what to do next (be specific: file name, command, topic).

**Drill Schedule status column** — while the interview arc is active. Use `🔧 Built` → `✅ Passed` (record your time and score, e.g. `✅ 13/13 in 24 min`).

That's it. Keep everything else as-is until it changes.

**On the other machine:** `git pull`, open this file, read **Next Task**, go. Nothing else to reconstruct.

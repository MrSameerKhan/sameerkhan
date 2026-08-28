# SYNC — Cross-Machine Handoff

> **Update this file before every `git push`. Read it after every `git pull`.**
> This is the only file that communicates state between Mac and Windows.

---

## Last Session

| Field | Value |
|-------|-------|
| **Machine** | Mac |
| **Date** | 28 August 2026 |
| **What I did** | (1) `06c_transformer_decoder_end_to_end.md` written. (2) **Board track reordered** (12 before 13; decoding 15->11; parked half added as boards 17-21). (3) **`05_bert_end_to_end.md` rewritten** — old one had a softmax summing to 1.400, contradictory K/V tables, sinusoidal PE (BERT uses learned), no NSP. (4) **GPT SPLIT INTO TWO CLEAN FILES**: `06_gpt1_end_to_end.md` (post-LN, vocab 40478, ctx 512, 116,534,784 params, task fine-tuning w/ aux LM loss) and `06b_gpt2_end_to_end.md` (pre-LN + ln_f, 1/sqrt(N) residual init, byte-level BPE 50257 = 256+50000+1, ctx 1024, 124,439,808 params, zero-shot). (5b) `06c_gpt3_end_to_end.md` added — GPT-3's block IS GPT-2's (paper's own words), so the file covers only what differs: alternating dense/banded-sparse attention, ctx 2048, the 8-model ladder w/ the two Table 2.1 errata, in-context learning verified by identical weight hash across prompts, C=6ND reconciling to 3.14e23. Zero mixing — verified no file contains another's numbers. Param counts are EXACT incl. biases + LayerNorm and match the HF checkpoints. (5) Fixed stale citations of the OLD broken GPT numbers `(2.604, 3.101, 2.120)` in `04_pretraining_objectives.md` and `02b_finetuning_end_to_end.md`, and the old BERT `x_cls = [1.386, 2.019]`. |
| **Files changed** | `4.nlp/03_sequence_models/06c_...md` (new), `5.transformers/02_models/05_bert_end_to_end.md` (rewritten), `06_gpt_end_to_end.md` -> `06_gpt1_end_to_end.md` (renamed + rewritten), `06b_gpt2_end_to_end.md` (new), `5.transformers/README.md`, `01_fundamentals/04_pretraining_objectives.md`, `6.llms/02b_finetuning_end_to_end.md`, `7.rag/01b_rag_end_to_end.md`, `06b_transformer_encoder_multihead.md` + `07_t5_end_to_end.md` + `03_encoder_decoder.md` (cross-refs), `MASTERY_PLAN.md` (reordered), `SYNC.md` |

> **Correction (25 Aug):** the blue placeholders in Naukri's draft (`Please mention if any`, `DD'MM'YY`) are **deliberate fill-in markers** — their cover email asks the customer to complete them. Earlier I called this a QC failure. It is not. Do not raise it.

> ⚠️ Reminder from 22 Aug: three commits shipped without updating this file. **Update it in the same commit as the work, not later.**

---

## Next Task

```
What:   Board 6 — Transformer DECODER whiteboard  (file done, board not drawn)
Board:  5.transformers/whiteboard/6.decoder.jpg      (to draw, 25 min)
Source: 4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md

The board must show:
  - three sub-layers: masked self-attn -> cross-attn -> FFN, each + Add & Norm
  - the causal triangle, -inf marked BEFORE the softmax box
  - cross-attn as a RECTANGLE (3 x 4), K,V arrows in from the encoder, Q up from
    the decoder  <- this is the one people draw backwards
  - teacher forcing (one pass, all rows) vs autoregressive (L passes, last row only)
  - where the KV cache sits; cross-attn K,V built ONCE

Reading order after that (board numbers, REORDERED 28 Aug):
   7 tokenization -> 8 BERT -> 9 GPT -> 10 T5 -> 11 decoding -> 12 attention at
   scale -> 13 modern block -> 14 long ctx -> 15 MoE -> 16 speculative
   -> 17-21 pretraining / SFT / LoRA / RLHF / eval

Boards 8 and 9 theory rewritten and verified. Board 9 is now TWO files, no mixing:
    `05_bert_end_to_end.md`      (BERT)
    `06_gpt1_end_to_end.md`      (GPT-1: post-LN, 116,534,784 params, fine-tuned per task)
    `06b_gpt2_end_to_end.md`     (GPT-2: pre-LN + ln_f, 1/sqrt(N) init, 124,439,808, zero-shot)
    `06c_gpt3_end_to_end.md`     (GPT-3: sparse attn, 8-model ladder, in-context learning)
  All three were numerically broken before; every value is now audited + torch-checked.

Also open:
  Board 6b whiteboard — theory written, board not drawn (20 min).
  Drill 01 — built 14 Aug, NEVER ATTEMPTED. 20 min, no reference, target 13/13.
  code_practice/11_interview_drills/01_multihead_attention.py
  Drill 04 (cross-attention) — not built; board 6 is reached so it is due.
```


---

## ACTIVE ARC — Interview Prep (started 14 Aug 2026)

**Context:** self-assessed weak in all four areas: resume defence, LLM/GenAI theory, ML system design, behavioural/STAR. Format chosen: **coding drills as the spine**, other three folded in around them.

**Timeline (rev. 22 Aug):** nothing scheduled yet. The original "1–2 weeks" clock is off. Days below are an *order*, not dates — runway is intact, so depth over speed. But drills have slipped 8 days with zero attempted; theory is not a substitute for saying it out loud or typing it cold.

### ACTIVE TRACK — LLM Architecture (scope set 25 Aug)

Full tracker: **`code_practice/11_interview_drills/MASTERY_PLAN.md`**

**REORDERED 28 Aug.** Two moves + the parked half added as boards 17–21. Rationale lives in
MASTERY_PLAN.md → "Ordering logic". Theory already exists for every board; the gap is recall,
not reading.

Each board passes 4 gates: **G1 draw · G2 hand-compute · G3 code in 20 min · G4 defend aloud.**
No board advances until all four pass.

| # | Board | G1 | Status |
|---|-------|----|--------|
| 1–5 | RNN → LSTM → GRU → Attention → Transformer **encoder** | — | ✅ drawn · G3 outstanding |
| 6 | **Transformer Decoder** — theory written 28 Aug (`06c`), board not drawn ← NEXT | 25 | 📄 |
| 6b | Encoder w/ multi-head, d_model=4 — **theory written 28 Aug**, board not drawn | 20 | 📄 |
| 7 | Tokenization → Embedding | 15 | ⬜ |
| 8 | BERT | 20 | ⬜ |
| 9 | GPT | 20 | ⬜ |
| 10 | T5 / BART *(= the Donut decoder)* | 25 | ⬜ |
| 11 | **Decoding** — greedy/beam/top-k/top-p/temperature ← *moved up from 15* | 20 | ⬜ |
| 12 | **Attention at scale** — KV cache, Flash, PagedAttention ← *now before the modern block* | 25 | ⬜ |
| 13 | **Modern LLM block** — pre-LN, RMSNorm, SwiGLU, RoPE, GQA ← *was 11* | 25 | ⬜ |
| 14 | Long context — RoPE scaling, ALiBi, SWA | 20 | ⬜ |
| 15 | Mixture of Experts | 20 | ⬜ |
| 16 | **Speculative decoding** ← *split out of old 15* | 15 | ⬜ |
| 17 | **Pretraining + scaling laws** *(was out of scope)* | 20 | ⬜ |
| 18 | **SFT / instruction tuning** *(was out of scope)* | 20 | ⬜ |
| 19 | **LoRA / QLoRA** *(was out of scope)* | 20 | ⬜ |
| 20 | **RLHF → DPO** *(was out of scope)* | 25 | ⬜ |
| 21 | **Evaluation** *(was out of scope)* | 20 | ⬜ |

**Why the two moves:** GQA cannot be justified without KV-cache arithmetic, so 12 now precedes 13.
And board 10 leaves you holding a probability row with no way to pick a token — answering that at
board 15 was backwards.

Boards 1–5 live in `4.nlp/03_sequence_models/whiteboard/`.
Boards 6–21 go in **`5.transformers/whiteboard/`**, continuing the numbering.

**Done means:** given any model card (Llama / Mistral / DeepSeek / Qwen), draw the whole model
from memory — tokens → vectors, each block, what changed since 2017 and why, how generation runs,
where the memory goes.

Every drill is a literal "I hand-wrote / I built from scratch" claim on the resume — drilling them **is** resume defence.

### Naukri resume — draft 3 reviewed 28 Aug
> **Handled in a SEPARATE session.** Theory/transformer sessions: skip this block.

Files: `resume/CustCopy2.pdf` (draft 1) · `resume/CustCopy3.pdf` (draft 3) ·
`resume/RESUME_CHANGE_REQUEST.md`+`.pdf` (what was sent) · `resume/resume_sameer_khan_v2.*` (my layout).

Draft 3 is 219 words SHORTER than draft 1.

| Ask | Result |
|-----|--------|
| S1 fix the inverted F1 sentence | ✅ fixed, both instances, my exact wording |
| S5 front-load hard numbers | ✅ first number at word ~79 (was ~370) |
| S8 remove Certifications / DOB / Career Timeline | ✅ all gone |
| S2 present tense on ICE | ⚠️ KRA bullets yes; Highlights still past tense |
| S6 remove hedges | ⚠️ "growing focus"+"strong understanding" gone; "exposure" ×2 remains, 3 new hedges added |
| S3 remove Core Competencies | ❌ ignored — still 11 items incl. "AI Solution Architecture & Technical Leadership" |
| S4 restore 7 specific phrases | ❌ 0 of 7 |
| S7 ATS parser test / S10 send .docx | ⏳ no written answer yet |

**REGRESSION — they cut length by deleting evidence:**
- **entire P1 RCA bullet gone** (21,096 pages · five services · Java validation defect · 260K regression · 2.3%) ← strongest bullet on the page
- TF 1.x→2.x migration, change control, rollback — gone
- `0.974` gone, so "outperforming the ensemble" now has no number behind a 0.975-vs-0.974 margin

**Two things got worse:** first verb a recruiter reads is now *"**Supported** a mortgage document
classification platform"* (draft 1 said "Owned"; the KRA section two inches below says "Own" —
the resume contradicts itself).

**Round-3 asks, in priority order:**
1. Restore the P1 RCA bullet — same standing as the F1 fix had
2. "Supported" → "Own"
3. Put `0.974` back, reframed: *"matched the OCR-based production ensemble's 0.974 while removing OCR from the inference path"*
4. Core Competencies — asked once with evidence, ignored. Drop it and spend goodwill on #1.

Not yet written: the round-3 reply. Keep it SHORT (4 asks), lead with what they got right.
Reply to `resumeservice@naukri.com`, subject unchanged, must keep `#VR#260817TS43026084_43453892#`.

### Resume layout files — RESOLVED 24 Aug

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

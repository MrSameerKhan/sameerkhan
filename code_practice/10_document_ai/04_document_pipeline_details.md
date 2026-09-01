# Session 4 — Full ICE-Style Document Pipeline (Portfolio Project)
Status: `🔧 Code-built`

Theory: [../../../9.multimodal/02_document_ai.md](../../9.multimodal/02_document_ai.md) · [../../../9.multimodal/06_layoutlm_end_to_end.md](../../9.multimodal/06_layoutlm_end_to_end.md)

**Portfolio milestone:** pin this to GitHub, add to resume:
> "Built end-to-end document AI pipeline (ingest → OCR → classify → extract → QA) for mortgage document processing; production upgrade path through LayoutLMv3 → Donut → ColPali + RAG."

---

## Use Case

A mortgage application file arrives as a PDF. The pipeline automatically:
1. Converts pages to images
2. Runs OCR
3. Identifies document type
4. Extracts structured fields (loan amount, LTV, applicant name, decision)
5. Answers natural language questions from a loan officer

This is the same workflow ICE Data Services, Nanonets, and Docsumo run on millions of documents.

---

## Pipeline Architecture

```
PDF / Image
    │ Stage 1: Ingest (PyMuPDF)
    │   PDF → PIL images (one per page)
    │
    │ Stage 2: OCR (EasyOCR / Tesseract / Donut)
    │   Image → raw text
    │
    │ Stage 3: Classify (rule-based → LayoutLMv3 in prod)
    │   Text → doc_type (mortgage_application, invoice, identity...)
    │
    │ Stage 4: Extract (regex → LayoutLMv3 / Donut in prod)
    │   Text + doc_type → {loan_amount, applicant_name, decision, ...}
    │
    │ Stage 5: Answer (OpenAI RAG → ColPali in prod)
    │   Question + extracted_fields + raw_text → grounded answer
    ▼
PipelineResult (dataclass with all outputs + per-stage latency)
```

---

## Production Upgrade Path

| Stage | This Demo | Upgrade 1 | Upgrade 2 |
|-------|-----------|-----------|-----------|
| OCR | EasyOCR (general) | Azure Document Intelligence | Donut (OCR-free) |
| Classify | Keyword rules | LayoutLMv3 fine-tuned on doc types | Donut + task prompt |
| Extract | Regex patterns | LayoutLMv3 fine-tuned on field labels | Donut CORD-style |
| QA | GPT-4o-mini + text RAG | ColPali + visual RAG | Native VLM (Gemini Vision) |

---

## Expected Output

```
Building synthetic mortgage document...
  3 pages ready

Running Document AI Pipeline:
  ingest → OCR → classify → extract → answer

  [Ingest]   3 pages — 2.1ms
  [OCR]      1847 chars — 4823.4ms
  [Classify] mortgage_application (conf=1.0) — 0.3ms
  [Extract]  9 fields — 1.2ms

══════════════════════════════════════════════════════════════
PIPELINE RESULTS
══════════════════════════════════════════════════════════════

Document type: mortgage_application
Key signals:   ['mortgage', 'loan amount', 'ltv', 'fixed rate', 'applicant']

Extracted fields (9):
  applicant_name:  Sarah Mitchell
  loan_amount:     320,000
  property_value:  390,000
  monthly_payment: 1,788.43
  interest_rate:   4.61%
  term_years:      25
  ltv_ratio:       82.1%
  decision:        APPROVED
  application_ref: APP-2024-08471

Q&A:
  Q: What is the loan amount and monthly payment?
  A: The loan amount is £320,000 with a monthly payment of £1,788.43 at 4.61% fixed over 25 years.

  Q: Is this mortgage application approved?
  A: Yes, the application is APPROVED as of 15 November 2024, valid until 15 February 2025.

  Q: What are the early repayment charges?
  A: ERCs apply during the 5-year fixed period: 5% in year 1, declining to 1% in year 5. No ERC applies after the fixed period ends.

Latency breakdown:
  ingest_ms                      2.1ms
  ocr_ms                      4823.4ms   ← EasyOCR is slow; Donut is faster
  classify_ms                    0.3ms
  extract_ms                     1.2ms
  TOTAL (non-QA)              4827.0ms
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."   # optional — falls back to field lookup without it
KMP_DUPLICATE_LIB_OK=TRUE python 04_document_pipeline.py
```

EasyOCR first run: downloads models (~100 MB). MPS OCR: ~5 seconds per page.
QA step: ~$0.005 per run (5 questions × gpt-4o-mini).

**Without OpenAI key:** pipeline still runs — QA step falls back to rule-based field lookup.

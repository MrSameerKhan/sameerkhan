# Session 1 — LayoutLM v3: Key-Value Extraction
Status: `🔧 Code-built`

Theory: [../../../9.multimodal/06_layoutlm_end_to_end.md](../../9.multimodal/06_layoutlm_end_to_end.md)

---

## Use Case

Invoice / form key-value extraction: rule-based systems fail when "Total:" appears at different positions across vendor templates. LayoutLMv3 learns that a right-aligned bold number after a colon is always the value, regardless of absolute position — because it sees text + spatial coordinates + pixel patches together.

---

## What LayoutLMv3 Sees (3 Modalities Fused)

```
Token stream:  ["Invoice", "Number", ":", "INV-2024-001", "Date", ":", "Nov", ...]
Bbox stream:   [[10,20,80,35], [85,20,150,35], [155,20,165,35], [170,20,280,35], ...]
               ↑ normalised 0-1000. spatial position = layout signal
Image patches: 16×16 pixel blocks from the document scan
               ↑ captures font weight, ruling lines, checkboxes, stamps

Cross-attention across all 3 → entity classification per token
Output: B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER, B-HEADER, O
```

---

## Why Layout Matters for Extraction

```
"Amount" at position (50, 200)  — left column → QUESTION (label)
"$8,912" at position (450, 200) — right column → ANSWER (value)

Same words in a different document:
"Amount" at position (200, 600) — table header → HEADER
"$8,912" at position (200, 625) — table cell   → ANSWER

Without bbox: model confuses label vs value based on word order alone.
With bbox:    model knows right-aligned numbers are always values.
```

---

## FUNSD Dataset

| Stat | Value |
|------|-------|
| Documents | 199 annotated forms |
| Entity types | HEADER, QUESTION, ANSWER, OTHER |
| Label format | IOB2 (B- / I- prefix) |
| Download | `load_dataset("nielsr/funsd-layoutlmv3")` |

---

## Expected Output

```
Device: mps
Parameters: 125.4M total, 125.4M trainable

Training LayoutLMv3...
Epoch 1 | eval_f1: 0.6821
Epoch 3 | eval_f1: 0.7934
Epoch 5 | eval_f1: 0.8410

Saved to models/10_document_ai/layoutlmv3_funsd

── Extraction demo ──
Document has 187 words. Extracted key-value pairs:
  Invoice Number: INV-2024-00847
  Date: November 15, 2024
  Vendor: DataSync Solutions Ltd.
  API Integration Services: $4,800.00
  Data Migration Consulting: $1,900.00
  Subtotal: $7,750.00
  VAT: $1,162.50
  Total: $8,912.50
```

LayoutLMv3-base fine-tuned on FUNSD achieves ~80-88 F1 (vs BERT text-only ~60 F1).

---

## How to Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE python 01_layoutlm_extraction.py
```

First run: downloads `microsoft/layoutlmv3-base` (~500 MB) + FUNSD (~50 MB).
MPS training time: ~45–60 min for 5 epochs (image patches are memory-heavy).
`apply_ocr=False` — no Tesseract dependency, uses FUNSD's pre-computed OCR.

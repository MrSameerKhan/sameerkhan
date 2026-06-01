# Session 2 — Donut: OCR-Free Document Understanding
Status: `🔧 Code-built`

Theory: [../../../9.multimodal/05_donut_end_to_end.md](../../../9.multimodal/05_donut_end_to_end.md)

---

## Use Case

OCR errors compound: "Totai" instead of "Total", garbled amounts on poor scans. Donut reads the document image directly — no OCR step, no cascading errors, one model does everything.

---

## Architecture

```
Document image (pixels)
    │
    ViT Encoder (Swin Transformer)
    │   Splits image into 14×14 pixel patches
    │   Encodes spatial + visual features
    │   Output: (H/14 × W/14) patch embeddings
    │
    Bart-style Decoder
    │   Prompt: "<s_docvqa><s_question>What is total?</s_question><s_answer>"
    │   Auto-regressively generates answer tokens
    │   Output: "$8,912.50"
    │
    For structured extraction (CORD):
    │   Prompt: "<s_cord-v2>"
    │   Output: {"menu": [{"nm": "API Integration", "price": "$4800"}], "total": {"price": "$8912.50"}}
```

---

## Two Pre-trained Models Used

| Model | Task | Output |
|-------|------|--------|
| `donut-base-finetuned-docvqa` | Document VQA | Free-form answer text |
| `donut-base-finetuned-cord-v2` | Receipt/invoice extraction | Structured JSON |

Both run on MPS (≈200M params).

---

## Key Difference: Donut vs LayoutLM

| | LayoutLMv3 | Donut |
|-|-----------|-------|
| OCR dependency | Yes (bboxes required) | No |
| Input | Image + words + bboxes | Image only |
| Training data | Need OCR-aligned labels | Images + target JSON |
| Speed | Faster (no generation) | Slower (autoregressive) |
| Best for | High-precision extraction | OCR-error-prone docs, scans |

---

## Expected Output

```
── Document VQA (OCR-free) ──
  Q: What is the total amount due?
  A: $8,912.50

  Q: What is the invoice number?
  A: INV-2024-00847

  Q: What is the VAT rate?
  A: 15%

  Q: Is the mortgage application approved?
  A: APPROVED

── Structured extraction (image → JSON) ──
{
  "menu": [
    {"nm": "API Integration Services", "cnt": "4", "price": "$4,800.00"},
    {"nm": "Data Migration Consulting", "cnt": "2", "price": "$1,900.00"}
  ],
  "subtotal": {"subtotal_price": "$7,750.00"},
  "total": {"total_price": "$8,912.50"}
}
```

---

## How to Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE python 02_donut_parsing.py
```

First run: downloads two Donut models (~400 MB each).
MPS inference: ~3–8 seconds per document (autoregressive generation).
No Tesseract or EasyOCR needed — entirely OCR-free.

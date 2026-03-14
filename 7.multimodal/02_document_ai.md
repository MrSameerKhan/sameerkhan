# Document AI

## Quick Reference
| Model / Tool | Task | Key Strength |
|-------------|------|-------------|
| PaddleOCR / Tesseract | OCR (text extraction) | Fast, free, self-hosted |
| AWS Textract / Azure DI | OCR + layout | Production-grade, handles complex layouts |
| LayoutLM v3 | Document understanding | Layout-aware BERT |
| Donut | End-to-end doc parsing | No OCR needed, reads image directly |
| GPT-4V / Claude 3 Vision | Zero-shot extraction | No training needed |
| Nougat | Scientific PDF parsing | LaTeX/formulas |
| PaddleNLP UIE | Universal extraction | Chinese + English, configurable schema |

**Domain relevance:** Document AI = your core domain. This is the most important chapter for your work.

---

## Core Concepts

### Document AI Pipeline Overview

```
Raw Document (PDF / image scan)
         ↓
    [OCR Engine]              Extract text + bounding boxes
         ↓
    [Layout Analysis]         Detect tables, headers, sections
         ↓
    [Document Understanding]  Classify, extract, relate entities
         ↓
    [Post-processing]         Validate, normalize, structure
         ↓
    Structured Output (JSON)
```

**Two paradigms:**
```
Paradigm 1: OCR → NLP pipeline
  OCR extracts text → NLP model processes text
  ✓ Modular, each component improvable independently
  ✓ Works with text-native PDFs (no OCR needed)
  ✗ OCR errors propagate; loses spatial layout information

Paradigm 2: End-to-end vision model
  Image → model → structured output (no separate OCR)
  ✓ Preserves layout; no OCR error propagation
  ✓ Handles complex layouts (tables, multi-column)
  ✗ Needs more training data; slower; larger models
```

---

### OCR Engines

```python
# ─── PaddleOCR (best open-source) ────────────────────────────────────────
from paddleocr import PaddleOCR
import cv2
import numpy as np

ocr = PaddleOCR(
    use_angle_cls=True,    # detect rotated text
    lang='en',
    use_gpu=True,
    det_model_dir='./models/det',  # detection model
    rec_model_dir='./models/rec',  # recognition model
)

result = ocr.ocr('invoice.jpg', cls=True)
# result: [[[box_coords], (text, confidence)], ...]

for line in result[0]:
    box = line[0]        # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text = line[1][0]    # "Invoice Number: INV-2024-0432"
    conf = line[1][1]    # 0.98

# Extract with bounding boxes for layout-aware processing
def extract_with_layout(image_path):
    result = ocr.ocr(image_path, cls=True)
    words = []
    for line in result[0]:
        box, (text, conf) = line
        x_min = min(p[0] for p in box)
        y_min = min(p[1] for p in box)
        x_max = max(p[0] for p in box)
        y_max = max(p[1] for p in box)
        words.append({
            "text": text,
            "confidence": conf,
            "bbox": [x_min, y_min, x_max, y_max],  # [x0, y0, x1, y1]
        })
    return sorted(words, key=lambda w: (w['bbox'][1], w['bbox'][0]))  # reading order

# ─── Tesseract ────────────────────────────────────────────────────────────
import pytesseract
from PIL import Image

image = Image.open('document.png')

# Basic text extraction
text = pytesseract.image_to_string(image, lang='eng')

# Word-level with bounding boxes
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
words = [
    {
        "text": data['text'][i],
        "conf": data['conf'][i],
        "bbox": [data['left'][i], data['top'][i],
                 data['left'][i]+data['width'][i],
                 data['top'][i]+data['height'][i]]
    }
    for i in range(len(data['text']))
    if data['text'][i].strip() and int(data['conf'][i]) > 30
]

# ─── AWS Textract (production) ────────────────────────────────────────────
import boto3

textract = boto3.client('textract', region_name='us-east-1')

with open('invoice.pdf', 'rb') as f:
    response = textract.analyze_document(
        Document={'Bytes': f.read()},
        FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']
    )

# Extract key-value pairs (forms)
key_value_pairs = {}
for block in response['Blocks']:
    if block['BlockType'] == 'KEY_VALUE_SET' and 'KEY' in block.get('EntityTypes', []):
        key_text = get_text(block, response)
        value_block = get_value_block(block, response)
        value_text = get_text(value_block, response)
        key_value_pairs[key_text] = value_text

# Extract tables
for block in response['Blocks']:
    if block['BlockType'] == 'TABLE':
        table = extract_table(block, response)
```

---

### LayoutLM Family

**LayoutLM v3 — Layout-aware BERT:**
```
Key insight: document understanding requires text + position + image together

Inputs to LayoutLM v3:
  1. Text tokens (from OCR)
  2. Bounding box coordinates [x0, y0, x1, y1, width, height] for each token
  3. Image patches (from ResNet or ViT)

All three modalities fused via unified transformer self-attention.

Pre-training objectives:
  - MLM (Masked Language Modeling) on text tokens
  - MIM (Masked Image Modeling) on image patches
  - Word-patch alignment: predict which patches correspond to masked words
```

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=9   # IOB tags: B-DATE, I-DATE, B-AMOUNT, I-AMOUNT, ...
)

image = Image.open("invoice.jpg").convert("RGB")

# Processor handles OCR internally (uses Tesseract) or accepts pre-extracted words/boxes
encoding = processor(
    image,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512,
)

with torch.no_grad():
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze()

# Map predictions back to words
words = encoding.words[0]
boxes = encoding.boxes[0]
id2label = model.config.id2label

for word, pred in zip(words, predictions):
    label = id2label[pred.item()]
    if label != "O":  # non-outside
        print(f"{word:20s} → {label}")

# Fine-tuning LayoutLMv3
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./layoutlmv3-invoice",
    num_train_epochs=20,            # small dataset → more epochs
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

---

### Donut (Document Understanding Transformer)

**Architecture:**
```
No OCR needed — reads document image directly end-to-end.

Swin Transformer encoder: image → patch embeddings
BART decoder: patch embeddings → structured text

Training: "Pretrain then Fine-tune"
  Pretraining: SynthDoG dataset (500K synthetic documents)
    Task: read all text from image (document OCR pretraining)
  Fine-tuning: task-specific datasets with structured output

Output format: special token-based structured output
  <s_invoice>
    <s_invoice_number>INV-2024-0432</s_invoice_number>
    <s_date>2024-03-14</s_date>
    <s_total>1250.00</s_total>
    <s_vendor>Acme Corp</s_vendor>
  </s_invoice>
```

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import json
import re

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
).to("cuda")

image = Image.open("receipt.jpg")

# Prepare input
pixel_values = processor(image, return_tensors="pt").pixel_values.to("cuda")
task_prompt = "<s_cord-v2>"  # task-specific start token
decoder_input_ids = processor.tokenizer(
    task_prompt, add_special_tokens=False, return_tensors="pt"
).input_ids.to("cuda")

# Generate structured output
outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

sequence = processor.batch_decode(outputs.sequences)[0]
sequence = sequence.replace(processor.tokenizer.eos_token, "")
sequence = sequence.replace(processor.tokenizer.pad_token, "")
# Remove task prompt
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
parsed = processor.token2json(sequence)
print(parsed)
# {"menu": [{"nm": "Americano", "cnt": "1", "price": "4500"}, ...], "total": {"price": "4500"}}

# Fine-tuning Donut for custom document type
# Prepare SynthDoG-style dataset:
# {"image": PIL.Image, "ground_truth": '{"invoice_number": "INV-001", "date": "2024-01-01"}'}
```

---

### GPT-4V / Claude Vision for Document AI

```python
import anthropic
import base64
from pathlib import Path

client = anthropic.Anthropic()

def extract_document(image_path: str, schema: dict) -> dict:
    """Zero-shot document extraction using Claude Vision."""

    # Encode image
    image_data = base64.standard_b64encode(
        Path(image_path).read_bytes()
    ).decode("utf-8")

    media_type = "image/jpeg"  # or image/png, image/pdf

    schema_str = json.dumps(schema, indent=2)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": f"""Extract information from this document.
Return a JSON object matching this schema exactly:
{schema_str}

Rules:
- Use null for fields not found in the document
- Normalize dates to YYYY-MM-DD format
- Normalize amounts to float (e.g., "$1,250.00" → 1250.0)
- Return ONLY valid JSON, no explanation

JSON:"""
                    }
                ],
            }
        ],
    )

    return json.loads(response.content[0].text)

# Usage
schema = {
    "invoice_number": "string or null",
    "date": "YYYY-MM-DD or null",
    "vendor_name": "string or null",
    "total_amount": "float or null",
    "line_items": [{"description": "string", "quantity": "float", "unit_price": "float", "total": "float"}]
}

result = extract_document("invoice.jpg", schema)
```

---

### Table Extraction

```python
# ─── Approach 1: AWS Textract Table Extraction ────────────────────────────
def extract_tables_textract(response):
    """Convert Textract CELL blocks into pandas DataFrames."""
    import pandas as pd

    tables = []
    for block in response['Blocks']:
        if block['BlockType'] != 'TABLE':
            continue

        # Build cell map
        cells = {}
        for relationship in block.get('Relationships', []):
            if relationship['Type'] == 'CHILD':
                for cell_id in relationship['Ids']:
                    cell = next(b for b in response['Blocks'] if b['Id'] == cell_id)
                    if cell['BlockType'] == 'CELL':
                        row = cell['RowIndex'] - 1
                        col = cell['ColumnIndex'] - 1
                        text = get_cell_text(cell, response)
                        cells[(row, col)] = text

        if not cells:
            continue

        max_row = max(k[0] for k in cells) + 1
        max_col = max(k[1] for k in cells) + 1
        table = [[cells.get((r, c), '') for c in range(max_col)]
                 for r in range(max_row)]

        df = pd.DataFrame(table[1:], columns=table[0])
        tables.append(df)

    return tables

# ─── Approach 2: table-transformer (TATR) ────────────────────────────────
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection"
)

image = Image.open("document.png")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, threshold=0.9, target_sizes=target_sizes
)[0]

# Detected tables → crop → run OCR → parse rows/columns
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    if model.config.id2label[label.item()] == "table":
        x0, y0, x1, y1 = box.int().tolist()
        table_img = image.crop((x0, y0, x1, y1))
        # Run PaddleOCR on table_img → parse structure
```

---

### Post-Processing & Validation

```python
import re
from datetime import datetime
from pydantic import BaseModel, validator

class InvoiceData(BaseModel):
    invoice_number: str | None
    date: str | None          # YYYY-MM-DD
    vendor_name: str | None
    total_amount: float | None
    currency: str = "USD"

    @validator('date', pre=True)
    def normalize_date(cls, v):
        if v is None:
            return None
        # Try multiple date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%d %B %Y']
        for fmt in formats:
            try:
                return datetime.strptime(str(v).strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None  # couldn't parse

    @validator('total_amount', pre=True)
    def normalize_amount(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # Strip currency symbols, commas
        cleaned = re.sub(r'[^\d.]', '', str(v).replace(',', ''))
        try:
            return float(cleaned)
        except ValueError:
            return None

    @validator('invoice_number', pre=True)
    def normalize_invoice_number(cls, v):
        if v is None:
            return None
        # Extract alphanumeric + hyphens only
        return re.sub(r'[^\w\-]', '', str(v)).upper()

# Usage
raw = {"invoice_number": "INV 2024-0432", "date": "March 14, 2024",
       "vendor_name": "Acme Corp", "total_amount": "$1,250.00"}
validated = InvoiceData(**raw)
# InvoiceData(invoice_number='INV2024-0432', date='2024-03-14', total_amount=1250.0, ...)
```

---

### Production Document Pipeline

```python
import asyncio
from pathlib import Path
from typing import Optional
import logging

class DocumentExtractionPipeline:
    def __init__(self, ocr_engine="paddleocr", extraction_model="claude"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en') if ocr_engine == "paddleocr" else None
        self.client = anthropic.Anthropic()

    async def process(self, document_path: str, doc_type: str, schema: dict) -> dict:
        path = Path(document_path)

        # 1. Convert PDF to images if needed
        if path.suffix.lower() == '.pdf':
            images = pdf_to_images(path)  # returns list of PIL Images
        else:
            images = [Image.open(path)]

        # 2. OCR each page
        ocr_results = []
        for i, img in enumerate(images):
            ocr_text = self.run_ocr(img)
            ocr_results.append({"page": i+1, "text": ocr_text, "image": img})

        # 3. Extract structured data
        if len(images) == 1:
            raw_data = self.extract_vision(images[0], schema)
        else:
            # Multi-page: combine OCR text
            combined_text = "\n\n".join(r['text'] for r in ocr_results)
            raw_data = self.extract_text(combined_text, schema)

        # 4. Validate and normalize
        validated = self.validate(raw_data, doc_type)

        # 5. Confidence scoring
        validated['confidence'] = self.compute_confidence(raw_data, validated)

        return validated

    def run_ocr(self, image):
        result = self.ocr.ocr(np.array(image), cls=True)
        return " ".join(line[1][0] for line in result[0] if line[1][1] > 0.5)

    def extract_vision(self, image, schema):
        # Use Claude Vision for single-page documents
        return extract_document(image, schema, self.client)

    def compute_confidence(self, raw, validated):
        """Score based on how many fields were successfully extracted."""
        total_fields = len([k for k in validated if k != 'confidence'])
        filled_fields = sum(1 for k, v in validated.items() if v is not None and k != 'confidence')
        return round(filled_fields / max(total_fields, 1), 2)
```

---

## Gotchas

**OCR errors cascade.** "INV0432" extracted as "INV0432" (good), but "l" vs "1" and "O" vs "0" confusions are extremely common. Always normalize and run regex validation on extracted fields. Common fixes: `text.replace('O', '0').replace('l', '1')` for numeric-only fields.

**Multi-page PDFs.** Most models process one image at a time. For multi-page invoices, process each page separately and merge results (e.g., header fields from page 1, line items from all pages).

**Rotated/skewed documents.** PaddleOCR handles rotation via `use_angle_cls=True`. For severe skew (>15°), deskew first using Hough transform or `cv2.warpAffine`.

**Table parsing is hard.** Merged cells, headerless tables, multi-line cells all break naive row/column extraction. Use table-specific models (Textract, table-transformer) rather than line-by-line OCR.

**LLM extraction hallucination.** VLMs sometimes invent plausible but wrong values (e.g., inventing a vendor name). Always validate extracted values against known patterns. Add "use null if not clearly visible" to prompts.

**PDF text layer vs scanned.** Text-native PDFs don't need OCR — extract text directly with `pdfplumber` or `pypdf`. Only use OCR for scanned documents (images). Running OCR on text-native PDFs degrades quality.

```python
import pdfplumber

with pdfplumber.open("invoice.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()  # uses native text layer, no OCR needed
        tables = page.extract_tables()  # native table extraction
```

---

## Interview Q&A

**Q: What is the difference between LayoutLM and Donut for document understanding?**
A: LayoutLM is an OCR-dependent model — it takes OCR-extracted text with bounding boxes and optionally image patches as input, then applies a transformer to fuse layout + text + image. Donut is end-to-end — it takes the raw document image and generates structured text directly (no OCR step). LayoutLM advantages: faster inference (no vision encoder for text), better for text-heavy documents where OCR is reliable. Donut advantages: no OCR error propagation, better for complex layouts (tables, forms), handles handwriting. In production: LayoutLM for high-quality scans, Donut or GPT-4V for complex/noisy documents.

**Q: How would you handle OCR errors in a document extraction pipeline?**
A: Defense-in-depth: (1) Preprocessing — deskew, denoise, binarize images before OCR; (2) OCR confidence filtering — discard words below 50% confidence; (3) Post-processing normalization — regex patterns for known formats (invoice numbers, dates, amounts), character substitution for common confusions ('O'↔'0', 'l'↔'1', 'I'↔'1'); (4) Schema validation — Pydantic models reject or normalize malformed values; (5) Multi-OCR ensemble — run Tesseract and PaddleOCR, take agreement where confidence differs; (6) Domain vocabulary correction — spell-check against domain word list (vendor names, product catalog).

**Q: When would you use GPT-4V over a specialized document model like LayoutLM?**
A: GPT-4V/Claude Vision when: (1) zero-shot — no training data available, (2) high document variety — many different formats that would require separate fine-tuned models, (3) complex reasoning — understanding context beyond simple field extraction (e.g., "is this invoice overdue?"), (4) handwriting or unusual layouts. LayoutLM when: (1) high volume + cost sensitivity — LLM APIs are expensive at scale, (2) consistent document format — fine-tuned LayoutLM outperforms GPT-4V on specific formats, (3) strict latency — local inference is faster, (4) data privacy — no sending documents to external APIs.

---

## Connections
- **Vision-Language Models (6.multimodal/01):** LayoutLM, Donut, and GPT-4V are VLMs applied to documents
- **NLP Applications (3.nlp/applications/):** NER, IE techniques applied to OCR-extracted text
- **RAG (5.llms/04):** Document AI creates structured data → chunked → embedded → RAG retrieval
- **LLM Prompting (5.llms/01):** GPT-4V extraction quality depends heavily on prompt design
- **CV Fundamentals (2.computerVision/fundamentals/):** OCR preprocessing (deskew, binarize) uses CV techniques

## Key Takeaway
Document AI = OCR → layout analysis → structured extraction → validation. Two paradigms: OCR-dependent (LayoutLM — fast, accurate for known formats) vs end-to-end vision (Donut, GPT-4V — no OCR step, better for complex/varied layouts). Always validate extracted fields with Pydantic schemas and regex patterns. In production: PDF text layer when available (no OCR), PaddleOCR for scanned documents, AWS Textract for tables/forms, GPT-4V for complex reasoning or zero-shot extraction.

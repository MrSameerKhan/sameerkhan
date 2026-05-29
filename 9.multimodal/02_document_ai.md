# 02 Document AI

> **Domain relevance: Document AI = your core domain. This is the most important chapter for your work.**

## Quick Reference

| Model / Tool | Task | Key Strength |
|---|---|---|
| PaddleOCR / Tesseract | OCR (text extraction) | Fast, free, self-hosted |
| AWS Textract / Azure DI | OCR + layout | Production-grade, handles complex layouts |
| LayoutLM v3 | Document understanding | Layout-aware BERT |
| Donut | End-to-end doc parsing | No OCR needed, reads image directly |
| GPT-4V / Claude 3 Vision | Zero-shot extraction | No training needed |
| Nougat | Scientific PDF parsing | LaTeX/formulas |
| PaddleNLP UIE | Universal extraction | Chinese + English, configurable schema |
| ColPali / ColQwen (2024) | Multimodal retrieval over document images | Skip OCR — retrieve as images, ground at patch level |
| Native-VLM parsing (Claude 3.5 / GPT-4o / Qwen2.5-VL) | Zero-setup document parsing | Low-volume high-complexity docs |

---

## Core Concepts

### Document AI Pipeline Overview

```
Raw Document (PDF / image scan)
  ↓
[OCR Engine]          Extract text + bounding boxes
  ↓
[Layout Analysis]     Detect tables, headers, sections
  ↓
[Document Understanding]  Classify, extract, relate entities
  ↓
[Post-Processing]     Validate, normalize, structure
  ↓
Structured Output (JSON)
```

Two paradigms:
```
Paradigm 1: OCR + NLP pipeline
  OCR extracts text → NLP model processes text
  ✓ Modular, each component improvable independently
  ✓ Works with text-native PDFs (no OCR needed)
  ✗ OCR errors propagate, loses spatial layout information

Paradigm 2: End-to-end vision model
  Image → model → structured output (no separate OCR)
  ✓ Preserves layout; no OCR error propagation
  ✓ Handles complex layouts (tables, multi-column)
  ✗ Needs more training data; slower; larger models
```

---

## OCR Engines

```python
# PaddleOCR (best open-source)
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
    box  = line[0]   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    text = line[1][0]  # "Invoice Number: INV-2024-0432"
    conf = line[1][1]  # 0.98

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
            "bbox": [x_min, y_min, x_max, y_max]   # [x0, y0, x1, y1]
        })
    return sorted(words, key=lambda w: (w['bbox'][1], w['bbox'][0]))  # reading order

# Tesseract
import pytesseract
from PIL import Image

image = Image.open('document.png')
# Basic text extraction
text = pytesseract.image_to_string(image, lang='eng')
# Word-level with bounding boxes
data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
words = [
    {"text": data['text'][i],
     "bbox": (data['left'][i], data['top'][i],
              data['left'][i]+data['width'][i], data['top'][i]+data['height'][i])
    }
    for i in range(len(data['text'])) if data['text'][i].strip()
]
```

```python
# AWS Textract (production)
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
        key_text  = get_text(block, response)
        value_block = get_value_block(block, response)
        key_value_pairs[key_text] = get_text(value_block, response)

# Extract tables
for block in response['Blocks']:
    if block['BlockType'] == 'TABLE':
        table = extract_table(block, response)
```

---

## LayoutLM Family

Evolution: v1 → v2 → v3

| Version | Params | Image | Attention | Pre-training |
|---|---|---|---|---|
| v1 (2020) | 113M | ✗ None | Text+layout only | MLM + Layout Prediction |
| v2 (2021) | 200M | ✓ CNN features, cross-attention | Text+layout+image (separate streams) | MLM + MIM + ITM |
| v3 (2022) | 133M | ✓ ViT patches, unified | All modalities in one transformer | MLM + MIM + WPA |

**v3 is the default — use it unless you have a specific reason not to.**

### LayoutLM v3 Architecture

```
Key insight: text position on page carries semantic meaning.
"Total: $1,250" in the bottom-right corner = invoice total.
Same text in the middle = probably a line item.

Inputs:
1. Text tokens (from OCR) — up to 512 tokens
2. Bounding box per token: [x, y0, x1, y1, width, height]
   Normalized to [0, 1000]: x_norm = int(x_pixels / img_width × 1000)
   Example: word at pixel (320, 340) in 1280×1700 image
     → x0 = int(320/1280 × 1000) = 250
     → y0 = int(340/1700 × 1000) = 200
3. Image patches: 224×224 = 14×14 = 196 patches (16×16 each)

Embedding Layers:
  token_emb    = nn.Embedding(vocab_size, 768)
  x_emb        = nn.Embedding(1001, 128)   # x0, x1, width
  y_emb        = nn.Embedding(1001, 128)   # y0, y1, height
  layout_proj  = nn.Linear(128×5, 768)     # fuse all 6 bbox coords → 768
  patch_proj   = nn.Linear(768, 768)       # [196, d_hidden→3 = project]

All three streams → concatenated → single 12-layer transformer self-attention.

Pre-training objectives (133M params, IIT-CDIP dataset):
  MLM: mask 15% of text tokens → predict original token
       Forces model to use layout context for masked word prediction
  MIM: mask 40% of image patches → predict dVAE discrete tokens
       Forces model to learn visual features from surrounding patches
  WPA: Word-Patch Alignment — binary classification per token:
       Does this OCR text token spatially overlap with this image patch?
       Forces cross-modal spatial correspondence learning

Subword inheritance (critical detail):
  Word "Invoice" → tokenizer splits to ["In", "##vo", "##ice"]
  All 3 subwords inherit the same bbox as the parent word.
  This is why box encoding happens per-token, not per-word.
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
    if label != "O":   # non-outside
        print(f"{word:20s} {label}")

# Fine-tuning LayoutLMv3
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./LayoutLMv3-invoice",
    num_train_epochs=20,              # small dataset = more epochs
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

## Donut (Document Understanding Transformer)

```
Architecture:
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
    <s_date>2024-03-16</s_date>
    <s_total>$1,250.00</s_total>
    <s_vendor>Acme Corp</s_vendor>
  </s_invoice>
```

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json, re

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-cord-v2"
).to("cuda")

image = Image.open("receipt.jpg")

# Prepare input
pixel_values = processor(image, return_tensors="pt").pixel_values.to("cuda")
task_prompt = "<s_cord-v2>"   # task-specific start token
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
sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
parsed = processor.token2json(sequence)
print(parsed)
# {"menu": [{"name": "Americano", "cnt": "1", "price": "4500"}, ...], "total": {"price": "4500"}}

# Fine-tuning Donut for custom document type:
# Prepare SynthDoG-style dataset:
# {"image": PIL.Image, "ground_truth": '{"invoice_number": "INV-001", "date": "2026-01-01"}'}
```

---

## Nougat (Neural Optical Understanding for Academic Documents)

```
Problem: Scientific PDFs contain LaTeX equations, tables, figures, references.
Standard OCR produces garbage for math: "f(x)dx" + "JT(x)dx" (wrong).
Nougat reads the PDF page as an image and outputs valid Markdown + LaTeX.

Architecture: same as Donut (Swin encoder + mBART decoder), but trained on:
  - 8M+ academic paper pages from arXiv, PubMed
  - Ground truth: LaTeX source matched to compiled PDF pages
  - Output: Markdown with LaTeX math blocks

Donut vs Nougat:
  Donut  → business documents (invoices, receipts, forms) → structured JSON
  Nougat → academic PDFs → readable Markdown with equations
```

Output example:
```
Nougat output:
## 3.2 Self-Attention Mechanism
The attention function maps a query and set of key-value pairs to an output:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
| Layer | d_model | d_ff | Heads |
|-------|---------|------|-------|
| 1-6   | 512     | 2048 | 8     |
```

```python
from nougat import NougatModel
from nougat.utils.dataset import LazyDataset
import torch

model = NougatModel.from_pretrained("facebook/nougat-base").to("cuda")
model.eval()

from nougat.utils.checkpoint import get_checkpoint
from PIL import Image

# Convert PDF pages to images first
pages = pdf_to_images("paper.pdf")   # list of PIL Images
predictions = []
for page in pages:
    pixel_values = model.encoder.prepare_input(page, random_padding=False)
    pixel_values = pixel_values.unsqueeze(0).to("cuda")
    with torch.no_grad():
        outputs = model.inference(image_tensors=pixel_values)
    predictions.append(outputs["predictions"][0])

# Combine pages into single markdown
full_text = "\n\n".join(predictions)
```

When to use Nougat: ✓ Converting academic papers / technical reports to searchable text ✓ Building RAG systems over scientific literature ✓ Extracting equations, tables, algorithms from papers ✗ Not for business documents (invoices, contracts) — use LayoutLM/Donut ✗ Not for handwritten documents.

**Limitation:** Nougat can hallucinate when the PDF is low-quality or heavily scanned. Always check output on representative samples before deploying.

---

## GPT-4V / Claude Vision for Document AI

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

    media_type = "image/jpeg"   # or image/png, image/pdf
    schema_str = json.dumps(schema, indent=2)

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[{
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
        }]
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

## ColPali / ColQwen — Multimodal Retrieval Over Document Images (2024)

**The paradigm shift:** skip OCR entirely. Treat each PDF page as an image; encode patches with a vision-language model; retrieve and ground answers at the patch level. ColPali (Faysse et al., Sep 2024) and ColQwen (Dec 2024 — same idea on Qwen2-VL) are the canonical implementations.

**Why this matters for Document AI:** OCR is the longest-running source of error in document pipelines. ColPali sidesteps it entirely: naturally handles tables, charts, hand-drawn diagrams, scanned forms, math equations — anything visual that OCR mangles. Citations point to specific image regions (auditor-friendly, regulator-friendly). Beats OCR+BM25 and OCR+dense pipelines by ~10-20% on document retrieval benchmarks (ViDoRe, DocVQA).

How it works (late-interaction over patches):
```
1. Index time:
   For each PDF page (as image):
     - PaliGemma / Qwen2-VL encoder → patch embeddings: [n_patches × d]
     - n_patches = 1024 per page (e.g., 32×32 grid for 448×448 input)
     Store all patch vectors (NOT a single page-level pooled vector)

2. Query time:
   - Embed query as tokens: [n_query_tokens × d]
   - For each query token, find max similarity with any patch on any page
   - Sum the max-sims per token = ColBERT-style late-interaction score
   - Top-k pages = pages with highest aggregate score
   - The patches with high query-token similarity = the "evidence regions" (highlight-ready)
```

| Aspect | OCR + Text RAG | ColPali / ColQwen |
|---|---|---|
| Setup complexity | High (OCR + parser + chunker + embedder) | Low (one model, one pass) |
| Storage per page | Text + dense vector (~4B) | Patch embeddings (~1MB — 1000× larger) |
| Retrieval latency | Fast (BM25 + dense + rerank) | Slower (late-interaction is expensive) |
| Handles tables / charts / handwriting | Poor — OCR-bottlenecked | Strong — direct visual encoding |
| Citation precision | Page-level / chunk-level | Patch-level (visual highlight) |
| Cost at scale | Lower (text is small) | Higher (storage + compute) |
| Best for | High-volume text-native PDFs | Visual-heavy / scanned / mixed-modality |

```python
from colpali_engine.models import ColPali, ColPaliProcessor
import torch
from PIL import Image

model = ColPali.from_pretrained("vidore/colpali-v1.3").eval().to("cuda")
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")

# Index: encode pages
page_images = [Image.open(f"page_{i}.png") for i in range(N)]
with torch.no_grad():
    batch_imgs = processor.process_images(page_images).to("cuda")
    page_embeddings = model(**batch_imgs)   # [N, n_patches, d]

# Query: encode + retrieve
queries = ["What was Q3 revenue?"]
with torch.no_grad():
    batch_q = processor.process_queries(queries).to("cuda")
    query_emb = model(**batch_q)   # [1, n_query_tokens, d]

scores = processor.score_multi_vectors(query_emb, page_embeddings)  # [1, N]
top_pages = scores.topk(5).indices
```

When to use ColPali: ✓ Scanned documents where OCR struggles (handwritten forms, low-quality scans, multilingual) ✓ Visual content matters for retrieval (charts, diagrams, layout-cued meaning) ✗ High-volume text-native PDFs (text extraction is much faster + cheaper) ✗ Strict latency / storage budgets (patch embeddings are big).

---

## Native-VLM Document Parsing — The Simpler 2024 Alternative

For **low-volume / high-complexity** documents, the cleanest 2024-25 approach is also the simplest: skip the entire OCR + LayoutLM + post-processing pipeline and send the page image straight to a strong VLM with a Pydantic schema.

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field
from PIL import Image
import base64, io

client = instructor.from_anthropic(Anthropic())

class Invoice(BaseModel):
    invoice_number: str
    date: str = Field(description="YYYY-MM-DD")
    vendor: str
    total: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    line_items: list[dict]

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.standard_b64decode(buf.getvalue()).decode()

invoice: Invoice = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                          "data": to_b64(page_image)}},
            {"type": "text", "text": "Extract invoice fields. Be precise. Use ISO dates."}
        ],
    }]
)
# invoice is a validated Invoice instance — schema violations auto-retry
```

**Strengths:** zero training data needed; handles arbitrary layouts; Pydantic-validated output via Instructor; one library replaces the LayoutLM+CRF+post-processor stack. **Tradeoffs:** API cost (~$0.01-0.05 per page on Claude 3.5 Sonnet); latency (~2-5s); no offline / data-residency-restricted scenarios.

**Hybrid pattern in production:** high-volume / structured docs → LayoutLM fine-tuned; long-tail / new doc types → native-VLM parsing; visual retrieval over the whole corpus → ColPali. Most production teams in 2025 run all three for different document classes rather than picking one.

---

## Table Extraction

```python
# Approach 1: AWS Textract Table Extraction
def extract_tables_textract(response):
    """Convert Textract CELL blocks into pandas DataFrames."""
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

        max_row = max(r for r, c in cells) + 1
        max_col = max(c for r, c in cells) + 1
        table = [[cells.get((r, c), "") for c in range(max_col)] for r in range(max_row)]
        df = pd.DataFrame(table[1:], columns=table[0])
        tables.append(df)
    return tables

# Approach 2: table-transformer (TATR)
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
)
```

---

## Post-Processing & Validation

```python
import re
from datetime import datetime
from pydantic import BaseModel, validator

class InvoiceData(BaseModel):
    invoice_number: str | None
    date:    str | None   # YYYY-MM-DD
    vendor_name: str | None
    total_amount: float | None
    currency: str = "USD"

    @validator('date', pre=True)
    def normalize_date(cls, v):
        if v is None:
            return None
        # Try multiple date formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%d-%m-%Y',
                   '%d %b, %Y', '%b %d, %Y', '%B %d %Y']
        for fmt in formats:
            try:
                return datetime.strptime(str(v).strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None   # couldn't parse

    @validator('total_amount', pre=True)
    def normalize_amount(cls, v):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        # Strip currency symbols, commas
        cleaned = re.sub(r'[\$,]', '', str(v).replace(',', ''))
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
raw = {"invoice_number": "INV 2024-0432", "date": "March 16, 2026",
       "vendor_name": "Acme Corp", "total_amount": "$1,250.00"}
validated = InvoiceData(**raw)
# InvoiceData(invoice_number='INV2024-0432', date='2026-03-16', total_amount=1250.0, ...)
```

---

## Gotchas

**OCR errors cascade:** "INV0432" extracted as "INV0432" (good), but "1" vs "l" and "O" vs "0" confusions are extremely common. Always normalize and run regex validation on extracted fields. Common fixes: `text.replace('O', '0').replace('l', '1').replace('I', '1')` for numeric-only fields.

**Multi-page PDFs:** Most models process one page at a time. For multi-page invoices, process each page separately and merge results (e.g., header fields from page 1, line items from all pages).

**Rotated/skewed documents:** PaddleOCR handles rotation via `use_angle_cls=True`. For severe skew (>15°), deskew first with Hough transform or `cv2.warpAffine`.

**Table parsing is hard:** Merged cells, headers with no clear row/column structure. Use table-specific models (Textract, table-transformer) rather than line-by-line OCR.

**LLM extraction hallucination:** VLMs sometimes invent plausible but wrong values (e.g., inventing a vendor name). Always validate extracted values against known patterns. Add "use null if not clearly visible" to prompts.

**PDF text layer vs scanned:** Text-native PDFs don't need OCR — extract directly with `pdfplumber` or `pypdf`. Only run OCR on scanned documents (applying OCR to text-native PDFs degrades quality).

```python
import pdfplumber

with pdfplumber.open("invoice.pdf") as pdf:
    for page in pdf.pages:
        text   = page.extract_text()   # uses native text layer, no OCR needed
        tables = page.extract_tables()  # native table extraction
```

---

## Interview Q&A

**Q: What is the difference between LayoutLM and Donut for document understanding?**
A: LayoutLM is an OCR-dependent model — it takes OCR-extracted text with bounding boxes and optionally image patches as input, then applies a transformer to fuse layout + text + image. Donut is end-to-end — it takes the raw document image and generates structured text directly (no OCR step). LayoutLM advantages: faster inference (no vision encoder for text), better for text-heavy documents where OCR is reliable. Donut advantages: no OCR error propagation, better for complex layouts (tables, forms), handles handwriting. In production: LayoutLM for high-quality scans, Donut or GPT-4V for complex/noisy documents.

**Q: How do you handle OCR errors in a document extraction pipeline?**
A: Defense-in-depth: (1) Pre-processing — deskew, denoise, binarize images before OCR; (2) OCR quality — discard words below 50% confidence; (3) Post-processing normalization — regex patterns for known formats (invoice numbers, dates, amounts); character substitution for common confusions (O↔0, 1↔I, 1↔l); (4) Schema validation — Pydantic models reject or normalize malformed values; (5) Multi-OCR ensemble — run Tesseract and PaddleOCR, take agreement where confidence differs; (6) Domain vocabulary correction — spell-check against domain word list (vendor names, product catalog).

**Q: When would you use GPT-4V over a specialized document model like LayoutLM?**
A: GPT-4V/Claude Vision when: (1) zero-shot — no training data available; (2) complex reasoning — many different formats that would require separate fine-tuned models; (3) complex reasoning — understanding context beyond simple field extraction (e.g., "is this invoice overdue?"); (4) handwriting or unusual layouts. LayoutLM when: (1) high volume + cost sensitivity — LLM APIs are expensive at scale; (2) consistent document format — fine-tuned LayoutLM outperforms GPT-4V on specific formats; (3) strict latency — local inference is faster; (4) data privacy — no sending documents to external APIs.

---

## Connections

- Vision-Language Models (`6.multimodal/01`): LayoutLM, Donut, and GPT-4V are VLMs applied to documents
- NLP Applications (`3.nlp/applications/`): NER, IE techniques applied to OCR-extracted text
- RAG (`5.llms/04`): Document AI creates structured data → chunked → embedded → RAG retrieval
- LLM Prompting (`6.llms/01`): GPT-4V extraction quality depends heavily on prompt design
- CV Fundamentals (`2.computerVision/fundamentals/`): OCR preprocessing (deskew, binarize) uses CV techniques

---

## Key Takeaway

Document AI = OCR → layout analysis → structured extraction → validation. Two paradigms: OCR-dependent (LayoutLM — fast, accurate for known formats) vs end-to-end vision (Donut, GPT-4V — no OCR step, better for complex/varied layouts). Always validate extracted fields with Pydantic schemas and regex patterns. In production: PDF text layer when available (no OCR), PaddleOCR for scanned documents, AWS Textract for tables/forms, GPT-4V for complex reasoning or zero-shot extraction. 2024 additions: ColPali for visual-first retrieval; native-VLM parsing for low-volume high-complexity docs.

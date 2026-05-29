# 05 — Donut End-to-End: Worked Examples

> Complete dry-run of Donut from raw document image → encoder → decoder → structured output. No OCR. Every step traced with numbers.

---

## What Donut Actually Does

```
Problem: Extract structured fields from a document image.

LayoutLM approach: image → OCR → text+boxes → transformer → labels
  ✗ OCR errors cascade. Two-stage, complex pipeline.

Donut approach: image → encoder → decoder → JSON output directly
  ✓ No OCR step. End-to-end differentiable.
  ✓ OCR errors cannot happen — there is no OCR.

The key insight: treat document understanding as a
sequence-to-sequence task. Input = image. Output = structured text.

Architecture:
  Encoder: Swin Transformer (image → patch features)
  Decoder: BART-style autoregressive transformer (features → text tokens)
```

---

## Part 1: Architecture Deep Dive

### Swin Transformer Encoder

```
Input: document image, resized to 2560×1920 pixels (Donut uses large resolution)
  Smaller inference: 1280×960 also common

Step 1: Patch partition
  Split image into non-overlapping 4×4 patches
  Each patch: 4×4×3 = 48 values (RGB)
  Number of patches: (2560/4) × (1920/4) = 640 × 480 = 307,200 patches

Step 2: Linear embedding
  Each patch → linear projection → d=96 (Swin-B uses d=128)
  307,200 × 96 dimensional feature map

Step 3: 4 stages of Swin blocks with patch merging
  Stage 1: resolution/1, channels=1   → (640×480, 96)
  Stage 2: resolution/2, channels=2   → (320×240, 192)  + patch merging halves H,W
  Stage 3: resolution/4, channels=4   → (160×120, 384)
  Stage 4: resolution/8, channels=8   → (80×60,   768)

  Final output: 80×60 = 4,800 visual feature vectors, each ∈ R^768

Swin attention key property: Window-based attention
  Each token attends only to tokens in its 7×7 window (not all 307K)
  → O(n) complexity instead of O(n²)
  → Makes high-resolution document images tractable
```

### BART Decoder

```
Decoder: 4-layer transformer (Donut base)
Hidden size: 1024
Attention heads: 16
Vocab: 57,000+ tokens (added special task tokens like <s_invoice>)

Decoding is autoregressive:
  At each step t, decoder:
  1. Cross-attends to 4,800 encoder visual features
  2. Self-attends to previously generated tokens (causal mask)
  3. Projects to vocab size → softmax → next token probability
  4. Sample/argmax → next token

  Repeat until <eos> generated.

Input to decoder at step 0: task prompt token
  e.g., "<s_invoice>" for invoice extraction
        "<s_docvqa>" for document VQA
        "<s_cord-v2>" for receipt parsing
```

---

## Part 2: Forward Pass — Complete Dry Run

### Document

```
Invoice image: 2560×1920 pixels
Contents: invoice number INV-2026-0432, date 2026-03-14, total $1,250.00
```

### Encoder Pass

```
Image (2560×1920×3)
↓ patch partition (4×4)
(640×480, 48)
↓ linear embedding
(640×480, 96) = 307,200 patch tokens in R^96
↓ Swin Stage 1 (window attention, 96-dim)
(640×480, 96)
↓ patch merge (2×2 → 1, concat channels)
(320×240, 192)
↓ Swin Stage 2
(320×240, 192)
↓ patch merge
(160×120, 384)
↓ Swin Stage 3
(160×120, 384)
↓ patch merge
(80×60, 768)
↓ Swin Stage 4
(80×60, 768)
↓ flatten
4,800 visual feature vectors ∈ R^768

These 4,800 vectors are the encoder output.
Each vector represents a spatial region of the document.
The top-left region of the invoice maps to the first few vectors.
```

### Decoder Pass — Token by Token

```
Target output we want to generate:
  <s_invoice> <s_invoice_number>INV-2026-0432</s_invoice_number>
  <s_date>2026-03-14</s_date>
  <s_total>$1,250.00</s_total></s_invoice>

Step t=0:
  Input: <s_invoice>  (task prompt)
  Cross-attention: decoder token attends to all 4,800 encoder vectors
    → finds visual features corresponding to invoice fields
  Self-attention: only 1 token, trivial
  Output projection: logits over 57K vocab
  Top predictions: ["<s_invoice_number>" 0.82, "<s_date>": 0.04, ...]
  Selected: <s_invoice_number>  ← highest probability

Step t=1:
  Input: <s_invoice> <s_invoice_number>
  Cross-attention: attends to encoder features in top-right region
    (where invoice number typically appears)
  Output: "INV" → probability 0.79

Step t=2:
  Input: <s_invoice> <s_invoice_number> INV
  Output: "-" → probability 0.91

Step t=3:
  Input: ... INV -
  Output: "2026" → probability 0.87

  ...continues...

Step t=7:
  Output: </s_invoice_number>  ← closing tag

Step t=8:
  Output: <s_date>

Step t=9–11:
  Output: "2026", "-", "03", "-", "14"

Step t=12:
  Output: </s_date>

  ...etc...

Final generated sequence (30 steps):
  <s_invoice>
    <s_invoice_number>INV-2026-0432</s_invoice_number>
    <s_date>2026-03-14</s_date>
    <s_total>$1,250.00</s_total>
  </s_invoice>
```

### Parse to Dictionary

```python
import re, json

def token2json(token_sequence: str) -> dict:
    """Convert special-token XML-like output to Python dict."""
    result = {}

    # Match all field tags
    pattern = r"<s_([\w]+)>(.*?)</s_\1>"
    matches = re.findall(pattern, token_sequence, re.DOTALL)

    for field_name, value in matches:
        if field_name == "invoice":
            # Parse nested fields
            nested = token2json(value)
            result.update(nested)
        else:
            result[field_name] = value.strip()

    return result

output = """<s_invoice>
<s_invoice_number>INV-2026-0432</s_invoice_number>
<s_date>2026-03-14</s_date>
<s_total>$1,250.00</s_total>
</s_invoice>"""

parsed = token2json(output)
# {
#   "invoice_number": "INV-2026-0432",
#   "date": "2026-03-14",
#   "total": "$1,250.00"
# }
```

---

## Part 3: Training — Teacher Forcing + Loss

### Teacher Forcing

```
During training, we DON'T feed the model's own predictions as next input.
We feed the GROUND TRUTH token at each step.

Why? If model makes a mistake at step t, it would compound at t+1, t+2...
  Teacher forcing stabilizes training.

Ground truth sequence:
  <s_invoice> <s_invoice_number> INV - 2026 - 0432 </s_invoice_number>
  <s_date> 2026 - 03 - 14 </s_date>
  <s_total> $1 , 250 . 00 </s_total> </s_invoice> <eos>

Decoder input (shift right):
  t=0: <s_invoice>
  t=1: <s_invoice_number>
  t=2: INV
  ...

Decoder target (original sequence):
  t=0: <s_invoice_number>
  t=1: INV
  t=2: -
  ...

Loss: cross-entropy at each step, averaged over sequence length.
```

### Loss Computation — Dry Run

```
Sequence length: 30 tokens
Vocab size: 57,000

At each step t, decoder outputs logits ∈ R^57000
Cross-entropy loss at step t = -log(p_model(true_token_t))

Step t=0: target = <s_invoice_number>
  p(<s_invoice_number>) = 0.82 → loss = -log(0.82) = 0.198

Step t=1: target = "INV"
  p("INV") = 0.79 → loss = -log(0.79) = 0.236

Step t=2: target = "-"
  p("-") = 0.91 → loss = -log(0.91) = 0.094

Step t=3: target = "2026"
  p("2026") = 0.87 → loss = -log(0.87) = 0.139

Step t=4: target = "-"
  p("-") = 0.93 → loss = -log(0.93) = 0.073

Step t=5: target = "0432"
  p("0432") = 0.71 → loss = -log(0.71) = 0.344  ← harder, less common

...

Average over 30 steps:
  Total loss = sum of all step losses / 30
  Example: 0.198 + 0.236 + 0.094 + 0.139 + 0.073 + 0.344 + ... = 5.2 total
  Avg loss = 5.2 / 30 = 0.173 per token

After 1 epoch (100 documents): avg loss ≈ 1.2
After 10 epochs: avg loss ≈ 0.18
After 30 epochs: avg loss ≈ 0.08
```

---

## Part 4: Fine-Tuning for Custom Document Type

### Dataset Format

```
For each training document, you need:
{
  "image": PIL.Image,    # raw document image
  "ground_truth": str    # expected output in special token format
}

Ground truth format for invoice:
  <s_invoice><s_invoice_number>INV-2026-0432</s_invoice_number>
  <s_date>2026-03-14</s_date><s_vendor>Acme Corp</s_vendor>
  <s_total>$1250.00</s_total></s_invoice>

Minimum dataset size: 100–500 documents per document type
More complex layouts → need more data
```

### Adding Custom Tokens

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Add task-specific special tokens to the tokenizer
new_special_tokens = [
    "<s_invoice>", "</s_invoice>",
    "<s_invoice_number>", "</s_invoice_number>",
    "<s_date>", "</s_date>",
    "<s_vendor>", "</s_vendor>",
    "<s_total>", "</s_total>",
]
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": new_special_tokens}
)

# CRITICAL: resize model embeddings to match new vocab
model.decoder.resize_token_embeddings(len(processor.tokenizer))

# Set decoder start token (task prompt)
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
    "<s_invoice>"
)

model.config.pad_token_id = processor.tokenizer.pad_token_id
```

### Training Loop

```python
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch

# Hyperparameters
LR = 3e-5
EPOCHS = 10
BATCH_SIZE = 2          # large images → small batch
MAX_LENGTH = 768        # max decoder sequence length
IMAGE_SIZE = [2560, 1920]

optimizer = AdamW(model.parameters(), lr=LR)

model.train()
model.to("cuda")

for epoch in range(EPOCHS):
    total_loss = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to("cuda")  # (B, 3, H, W)
        labels = batch["labels"].to("cuda")              # (B, seq_len) token ids
        # -100 in labels means "ignore this position in loss" (padding)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss  # cross-entropy, averaged over non-padding tokens

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: avg loss = {total_loss/len(dataloader):.4f}")
```

### Preprocessing Function

```python
def preprocess(example, processor, max_length=768, image_size=[2560, 1920]):
    image = example["image"].convert("RGB")
    ground_truth = example["ground_truth"]

    # Process image
    pixel_values = processor(
        image,
        random_padding=True,       # augmentation: add random padding during train
        return_tensors="pt"
    ).pixel_values.squeeze()

    # Tokenize target sequence
    target_sequence = processor.tokenizer(
        ground_truth,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    labels = target_sequence.input_ids.squeeze()
    # Replace padding token id with -100 (ignored in loss)
    labels[labels == processor.tokenizer.pad_token_id] = -100

    return {"pixel_values": pixel_values, "labels": labels}
```

---

## Part 5: Inference

```python
def extract_document(image_path: str, processor, model, task_start_token: str) -> dict:
    image = Image.open(image_path).convert("RGB")

    # Encode image
    pixel_values = processor(image, return_tensors="pt").pixel_values.to("cuda")

    # Decoder start: task prompt
    decoder_input_ids = processor.tokenizer(
        task_start_token,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to("cuda")

    # Generate — autoregressive decoding
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,                                    # greedy (fast); use 4 for better accuracy
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode generated tokens to string
    sequence = processor.batch_decode(outputs.sequences)[0]

    # Remove special padding/eos
    sequence = sequence.replace(processor.tokenizer.eos_token, "")
    sequence = sequence.replace(processor.tokenizer.pad_token, "")

    # Remove task start token
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

    # Parse to dict
    return token2json(sequence)

# Usage
result = extract_document(
    "invoice.jpg",
    processor,
    model,
    task_start_token="<s_invoice>"
)
# {"invoice_number": "INV-2026-0432", "date": "2026-03-14", "total": "1250.00"}
```

### Beam Search vs Greedy

```
num_beams=1 (greedy):
  At each step, pick the single highest-probability token.
  Fast: 30 tokens × 30 decoder forward passes.
  Risk: locally optimal choices may not be globally optimal.

num_beams=4 (beam search):
  At each step, keep top-4 partial sequences.
  At the end, return highest-probability complete sequence.
  Slower: 4× more decoder passes.
  Better: typically 2-3% higher exact match accuracy.

For production at scale: greedy (num_beams=1) — the speed trade-off is usually worth it.
For high-stakes documents with small volume: beam search (num_beams=4).
```

---

## Part 6: Donut vs LayoutLM vs GPT-4V

| Metric              | Donut            | LayoutLM v3      | GPT-4V / Claude       |
|---|---|---|---|
| OCR dependency      | None             | Required         | None                  |
| Training data needed| 100–500 docs     | 100–500 docs     | 0 (zero-shot)         |
| Inference speed     | ~2s/doc          | ~0.5s/doc        | ~5s/doc (API latency) |
| Cost at scale       | Low (local)      | Low (local)      | High (per-token API)  |
| Handwriting         | Good             | Poor             | Good                  |
| Complex layouts     | Good             | Moderate         | Good                  |
| Custom fields       | Requires FT      | Requires FT      | Prompt engineering    |
| Max resolution      | High (2560px)    | 224px patches    | Variable (API)        |
| Interpretability    | Low              | Medium (tokens)  | Low                   |
| Best for            | Noisy, varied    | Clean, typed     | Zero-shot, complex reasoning |

---

## Part 7: Pre-training — SynthDoG

```
Donut is pre-trained on SynthDoG (Synthetic Document Generator):
  - 500K synthetic document images (auto-generated)
  - English, Korean, Japanese, Chinese
  - Random backgrounds, fonts, layouts

Pre-training task: read all text in the image (document OCR pre-training)
  Input: synthetic document image
  Output: all text content in reading order

This teaches the model:
  - How to read text from images
  - Layout awareness (reading order)
  - Visual-text correspondence

Then fine-tuned on task-specific datasets:
  CORD (receipts): 1K receipts → structured menu/total extraction
  DocVQA: 10K document + question + answer pairs
  RVL-CDIP: 400K docs → document classification

Key insight: SynthDoG pre-training means you need MUCH less real data for fine-tuning.
  SynthDoG (500K) + 100 real invoices = strong invoice extractor
  Without SynthDoG → need 10K+ real invoices for comparable results
```

---

## Part 8: Interview Questions

**Q: Why doesn't Donut need OCR? How does it "read" text?**

Donut uses a Swin Transformer encoder that processes the full document image at high resolution (2560×1920). Through pre-training on 500K synthetic documents with the task "output all text in the image," the encoder learns to represent text content spatially — effectively learning to "read" from visual features alone.

The decoder then generates text autoregressively by cross-attending to these visual features. It never explicitly runs character recognition (OCR) — instead it learns the visual patterns of characters and words from massive data.

Limitation: on low-quality scans or unusual fonts, Donut struggles more than OCR+LayoutLM because OCR has explicit character recognition training.

---

**Q: What is teacher forcing and why is it used?**

Teacher forcing: during training, feed the true (ground truth) token at step t as input for step t+1, regardless of what the model predicted at step t.

Without teacher forcing: a mistake at step 3 corrupts steps 4, 5, 6... → training becomes very slow, unstable (compounding errors).

With teacher forcing: each step trains independently on the correct prefix → stable gradients, fast convergence.

Downside: exposure bias — at inference, model never sees its own mistakes → can diverge at inference if it makes an early error. Mitigation: scheduled sampling (gradually replace GT with model predictions).

---

**Q: How would you improve Donut's performance on a specific document type?**

1. Domain-specific fine-tuning: 200–500 labeled documents of that type
2. Larger image resolution: 2560px × 1280px for text-dense documents (trade-off: 4× more memory/compute)
3. Beam search at inference (num_beams=4): 2-3% better exact match
4. Data augmentation:
   - Random rotation ±5° (handles slight skew)
   - Color jitter (handles scan quality variation)
   - Random crops with padding (partial document handling)
5. Longer max_length: if fields are long (line items, addresses)
6. Task prompt engineering: use more specific start token `"<s_construction_invoice>"` vs generic `"<s_invoice>"`

---

## Key Takeaway

```
Donut = Swin Transformer encoder + BART-style autoregressive decoder
No OCR. Image → encoder (4800 visual vectors) → decoder (generate token by token).

Training: teacher forcing + cross-entropy loss per token
  Avg loss epoch 1: ~1.2  → epoch 30: ~0.08
  Dataset: 100–500 labeled docs per document type
  Compute: 2-4 hours on A100 for 30 epochs

Inference: greedy or beam search decoding
  Output: XML-like special token sequence → parsed to dict via regex

Pre-training: SynthDoG 500K synthetic docs → "read all text"
  → fine-tune with 100 real docs → strong extractor

When to use Donut:
  Handwriting, noisy scans, complex varied layouts, no OCR infrastructure
When to use LayoutLM:
  Clean typed documents at scale, cost-sensitive, OCR already available
When to use GPT-4V:
  Zero-shot, complex reasoning, very low volume, high document variety
```

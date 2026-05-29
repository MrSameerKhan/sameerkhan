# 09 — VLM Hallucination: Patterns and Mitigation

> Why VLMs hallucinate AND how to detect / prevent it in production. The Document AI engineer's reliability toolkit.

---

## Table of Contents

1. Objective
2. Three types of VLM hallucination
3. Why VLMs hallucinate (mechanically)
4. Mitigation patterns
5. Detection — confidence and verification
6. Production design — Document AI specific
7. Interview questions (5)
8. Further reading

---

## 1. Objective

VLM hallucination is the #1 production risk for Document AI. The model says "the invoice total is $1,250.00" when the actual image says $1,520.00 — and there's no obvious way to detect this from the output text.

Senior interview Q: "How do you know your VLM-extracted invoice total is correct?"

---

## 2. Three Types of VLM Hallucination

### Object hallucination

The VLM says objects are present that aren't. "I see a cat on the table" — there's no cat. Most common: VLMs confabulate common objects (people, vehicles, furniture) that are plausible in the scene.

Measured by **POPE** (Polling-based Object Probing Evaluation, Li et al. 2023): ask the VLM yes/no questions about objects; check against ground-truth annotations.

### Attribute hallucination

Right object, wrong properties. "The car is red" — it's blue. "The text says $1,250" — it says $1,520.

**Document AI's biggest enemy.**

### Relation hallucination

Wrong relationship between objects. "The man is holding an umbrella" — the umbrella is on the ground next to him.

For Document AI, attribute hallucination dominates (numbers, dates, names misread). Relation hallucination matters when extracting tables (which row goes with which header).

---

## 3. Why VLMs Hallucinate (Mechanically)

Five contributing causes:

**1. Language prior dominance**

The LLM half of the VLM has strong priors from text-only pretraining. "Invoices usually have totals in the $100-$5000 range" — if the actual number is $50,000, the LLM may correct toward its prior.

**2. Vision encoder loss of resolution**

VLMs typically downsample to 336×336 or 448×448. Fine print and small digits become illegible. The model SHOULD say "I can't read it" but instead guesses based on context.

**3. Cross-modal alignment failure**

The projection layer maps vision tokens to LLM embedding space. Imperfect alignment → vision information is "blurry" to the LLM, which then defaults to text priors.

**4. Training data bias**

VLMs are trained on image-caption pairs from the web. Captions are noisy ("there's a beautiful sunset" when image shows a sunrise). Bias propagates.

**5. Sampling temperature**

At temperature > 0, the model can confidently produce wrong outputs. Setting temperature=0 reduces but doesn't eliminate hallucination.

---

## 4. Mitigation Patterns

### Pattern A — Constrained Decoding for Structured Output

Force the VLM to emit JSON matching a schema. The schema enforces valid values (e.g., dates must match `YYYY-MM-DD`, amounts must be `^\\\$\d+\.\d{2}$`). Eliminates format hallucination but not value hallucination.

### Pattern B — Verifier Model

Use a smaller, specialized model as a second pass.

```
image → VLM extractor → JSON
               ↓
       VLM_verifier(image, JSON) → confidence + corrections
```

The verifier asks: "Does the image actually contain the claimed values?" Can flag mismatches.

### Pattern C — Multi-Pass Extraction with Majority Vote

Run extraction 3-5 times at temperature > 0. Take the majority value for each field. If no majority, flag for human review.

Cost: 3-5× per extraction. Quality: noticeably reduces hallucination on numerical fields.

### Pattern D — OCR-Grounded Extraction

For documents specifically: use Tesseract / PaddleOCR to extract raw text first, THEN have the VLM extract structured fields from the OCR + image. The OCR gives a verifiable text baseline; the VLM does the structure.

```
image → OCR → raw_text
image, raw_text, schema → VLM → structured JSON
```

Reduces hallucination significantly for numerical fields. Adds OCR error mode (Tesseract can mis-read characters), but those errors are different and detectable.

### Pattern E — Crop and Re-Prompt

For high-stakes fields (totals, account numbers): CROP the image to just that region and re-prompt the VLM. The model can use full resolution on the cropped image. Much more accurate than asking about a field within a full-document context.

### Pattern F — Domain-Specific Fine-Tuning

Off-the-shelf VLMs are general. Fine-tune on (your_image, your_extracted_json) pairs for your specific document types. Reduces hallucination on the trained format dramatically.

PaliGemma 2 is explicitly designed for this kind of narrow fine-tuning.

---

## 5. Detection — Confidence and Verification

### Logit-Based Confidence

For each extracted field, compute the cumulative log-probability of the output tokens. Low confidence → flag for review.

```python
fields_with_confidence = []
for field_value, token_logprobs in extracted:
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    fields_with_confidence.append({
        "value": field_value,
        "confidence": exp(avg_logprob)
    })
```

Note: VLMs are often OVERCONFIDENT — outputs that look 99% confident might be wrong 10% of the time. Calibrate by collecting (output, correctness) pairs and fitting a calibration curve. In production: use the calibrated confidence to set routing thresholds (auto-process vs human review).

### Rule-Based Verification

Hard rules from the domain:
- Invoice line items sum to subtotal
- Subtotal + tax = total
- Account numbers match format `ACC-\d{4}`
- Dates parseable, in reasonable range

If a rule fires, flag.

### Cross-Model Consensus

Run two different VLMs (e.g., Qwen2.5-VL + Claude Sonnet) on the same image. Disagreement → flag.

Costly but highly effective for high-stakes extraction.

### Human-in-the-Loop

The most reliable for high-stakes use. Triage:
- High confidence + rules pass → auto-process
- Low confidence OR rule fail → human review queue
- Always 5% random sample to humans for QA

In financial document AI, ~5-20% of documents typically end up in human review. The goal is to make the auto-processable subset reliable enough.

---

## 6. Production Design — Document AI Specific

For the user's domain (mortgage docs, financial extraction at ICE):

```
┌─ INGEST ──────────────────────────────────────────────────────┐
│  Image / PDF → preprocessing (deskew, denoise, dpi correct)   │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─ EXTRACT (parallel paths) ────────────────────────────────────┐
│  Path A: Tesseract / PaddleOCR → raw text                     │
│  Path B: Donut or VLM → structured JSON                       │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─ VERIFY ──────────────────────────────────────────────────────┐
│  Rule checks:                                                 │
│   - Totals add up?                                            │
│   - Required fields present?                                  │
│   - Format valid (regex)?                                     │
│  Cross-check:                                                 │
│   - OCR raw text contains the extracted values?               │
│   - Log-probability above threshold?                          │
└───────────────────────────────────────────────────────────────┘
                              ↓
┌─ ROUTE ───────────────────────────────────────────────────────┐
│  if confidence > 0.95 AND rules pass:                         │
│    → auto-process                                             │
│  elif confidence > 0.7:                                       │
│    → spot-check or human-light review                         │
│  else:                                                        │
│    → full human review queue                                  │
└───────────────────────────────────────────────────────────────┘
```

This is exactly the pattern ICE's mortgage pipeline likely uses (or should). The user's "confidence-based routing" experience here is a strong differentiator.

---

## 7. Interview Questions (5)

**Q1: Why do VLMs hallucinate on document extraction?**

Five reasons: (1) language prior — the LLM half "corrects" toward common values, the LLM had model guesses anyway; (2) low resolution — small digits become illegible; (3) cross-modal alignment is imperfect; (4) training data is noisy (web captions); (5) sampling temperature. For document AI specifically, attribute hallucination on numerical fields (totals, dates) is the biggest risk.

---

**Q2: How do you detect hallucination in extracted invoice totals?**

Three layers: (1) Log-probability — low confidence → flag. (2) Rule-based — invoice subtotal + tax should CONTAIN the extracted total; line items should sum to subtotal. (3) Cross-grounding — OCR raw text should CONTAIN the extracted field as a string. If any of these fails, route to human review. In production: 5-20% of docs typically end up in human review queue.

---

**Q3: What's the OCR-grounded extraction pattern?**

Don't ask the VLM to read text from an image and structure it in one shot — that's where hallucination happens. Instead: (1) run OCR (Tesseract/PaddleOCR) for raw verifiable text. (2) Run VLM with the IMAGE + OCR TEXT + schema → structured JSON. (3) Verify each extracted field appears in the OCR raw text. The OCR ground-truth keeps the VLM honest.

---

**Q4: How would you reduce hallucination for high-stakes fields like account numbers?**

Crop-and-re-prompt: detect the field region (via VLM, layout model, or learned heuristic), crop to just that region, re-prompt the VLM at full resolution to enforce format (`ACC-\d{4}`). Verify with OCR. For truly critical fields (money movement), always require human approval regardless of confidence.

---

**Q5: What's the right confidence calibration for VLM outputs?**

Logit-based confidence (cumulative log-prob of output tokens) is a starting point, but VLMs are often OVERCONFIDENT — outputs that look 99% confident might be wrong 10% of the time. Calibrate by collecting (output, correctness) pairs and fitting a calibration curve. In production: use the calibrated confidence to set routing thresholds (auto-process vs human review).

---

## 8. Further Reading

- POPE (Li et al. 2023) — arXiv:2305.10355 — VLM object hallucination benchmark
- HallusionBench (Guan et al. 2023) — arXiv:2310.14566 — diagnostic VLM eval
- Visual Programming (Gupta & Kembhavi 2022) — arXiv:2211.11559 — verifier-style VLM
- Donut (Kim et al. 2022) — arXiv:2111.15664 — OCR-free extraction, less prone to language-prior bias
- PaliGemma 2 — fine-tuning recipe for narrow document tasks
- LayoutLM v3 — for layout-aware document extraction
- Tesseract + post-processing — classical OCR baseline

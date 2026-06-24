# Session 1 — BERT Text Classification
Status: `✅ Run`

---

## Use Case

Classify a sentence as **positive** or **negative** sentiment.
Fine-tune a pretrained BERT model on labeled sentences, then run inference on new text.

---

## Input Data Format

SST-2 dataset — each row is a sentence + binary label:

| sentence | label |
|----------|-------|
| `"hide new secretions from the parental units"` | 0 (negative) |
| `"contains no wit , only labored gags"` | 0 (negative) |
| `"that loves its characters and communicates something of the joy"` | 1 (positive) |

- `label 0` = negative, `label 1` = positive
- Downloaded automatically via `load_dataset("glue", "sst2")` on first run (~7 MB)
- Training uses 4,000 rows, validation uses 500 rows (subset for speed)

---

## Tokenizer Output (what BERT actually sees)

Input sentence: `"I loved this movie"`

```
input_ids:      [101, 1045, 3866, 2023, 3185, 102, 0, 0, ...]   shape: (128,)
attention_mask: [1,   1,    1,    1,    1,    1,   0, 0, ...]   shape: (128,)
```

- `101` = [CLS] token, `102` = [SEP] token, `0` = padding
- `attention_mask` tells BERT which positions are real (1) vs padding (0)

---

## Inference Input

Three custom sentences passed to `predict()` at the end of the script:

```python
"This movie was absolutely fantastic, I loved every second!"
"What a waste of time. Terrible acting, boring plot."
"It was okay, nothing special but not awful either."
```

---

## Expected Output

```
Device: mps

Epoch 1/3 | loss: 0.4821 | val_acc: 0.8760
Epoch 2/3 | loss: 0.3102 | val_acc: 0.9020
Epoch 3/3 | loss: 0.2341 | val_acc: 0.9100

Saved to ./bert_sst2

── Inference ──
  [positive 97.32%]  This movie was absolutely fantastic, I loved every second!
  [negative 98.61%]  What a waste of time. Terrible acting, boring plot.
  [negative 54.20%]  It was okay, nothing special but not awful either.
```

- Val accuracy on 4k subset: ~88–92% after 3 epochs
- Third sentence is intentionally ambiguous — lower confidence is correct behavior

---

## Actual Output (Windows, GTX 1650 Ti, 2026-06-17)

```
Device: cuda

Epoch 1/3 | loss: 0.4108 | val_acc: 0.8800
Epoch 2/3 | loss: 0.1871 | val_acc: 0.9180
Epoch 3/3 | loss: 0.0787 | val_acc: 0.9060

Saved to models/05_transformers/bert_classification

── Inference ──
  [positive 99.89%]  This movie was absolutely fantastic, I loved every second!
  [negative 99.75%]  What a waste of time. Terrible acting, boring plot.
  [negative 93.79%]  It was okay, nothing special but not awful either.
```

- GPU-Util peaked at 97%, VRAM 3431MiB / 4096MiB during training
- Loss dropped sharply: 0.41 → 0.19 → 0.08 (strong convergence on 4k samples)
- Confidence on ambiguous sentence improved to 93.79% (vs 54.20% on Mac MPS)

---

## How to Run

```powershell
python code_practice\05_transformers\01_bert_classification.py
```

First run downloads `bert-base-uncased` (~440 MB) — cached after that.
Training on GTX 1650 Ti CUDA: ~2–3 minutes for 3 epochs on 4k samples.
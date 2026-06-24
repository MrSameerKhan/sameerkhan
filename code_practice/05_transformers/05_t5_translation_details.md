# Session 5 — T5 Neural Machine Translation
Status: `✅ Run`

Theory: [../../../5.transformers/02_models/03_encoder_decoder_family.md](../../../5.transformers/02_models/03_encoder_decoder_family.md)

---

## Use Case

Multilingual document processor: Arabic or German financial documents → English for downstream processing. Any company with international clients, cross-border compliance docs, or multilingual customer communications.

---

## Key Concept — T5's Prefix-Based Task Specification

T5 (Text-To-Text Transfer Transformer) treats every NLP task as a text-to-text problem. The task is specified by prepending a natural language prefix to the input:

```
Task              Prefix                              Input → Output
──────────────────────────────────────────────────────────────────────────
Translation EN→DE "translate English to German: "    "Hello" → "Hallo"
Translation EN→FR "translate English to French: "    "Hello" → "Bonjour"
Summarization     "summarize: "                      article → summary
QA                "question: Q context: C"           → answer
Classification    "sst2 sentence: "                  text → "positive"
```

One model, one checkpoint, all tasks — just change the prefix.

---

## Tensor Shapes

```
Source (English with prefix):
  "translate English to German: The quarterly report shows..."
  input_ids:      shape (128,)
  attention_mask: shape (128,)

Target (German):
  "Der Quartalsbericht zeigt..."
  labels:         shape (128,) — target token ids; padding → -100

Encoder → contextual source representations (128, 512)
Decoder cross-attends → generates target tokens auto-regressively

Loss: cross-entropy on labels (same as BART)
```

**Why LR=3e-4?** T5 was pre-trained with Adafactor and higher learning rates than BERT. Using AdamW with 3e-4 (vs BERT's 2e-5) is standard for T5 fine-tuning.

---

## Extending to Arabic → English

Swap the model and remove the prefix:

```python
# In Config:
MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"   # MarianMT, same HuggingFace API
PREFIX     = ""                              # MarianMT infers direction from vocab

# Dataset: any Arabic-English parallel corpus
# load_dataset("Helsinki-NLP/opus-100", "ar-en") works the same way
```

The MarianMT models (Helsinki-NLP/) are specialized per language pair, smaller than T5, and often better for specific pairs. T5's advantage is single model for multiple language pairs.

---

## Input Data — opus_books (en-de)

| Field | Example |
|-------|---------|
| translation.en | "The quarterly report shows a 12% increase." |
| translation.de | "Der Quartalsbericht zeigt eine Steigerung von 12 Prozent." |

- ~51k sentence pairs (subset of European book translations)
- Script uses 4,000 training pairs
- `load_dataset("Helsinki-NLP/opus_books", "en-de")`

---

## BLEU Score

BLEU measures n-gram overlap between generated and reference translation. The script implements a simple unigram BLEU for demonstration. Production: use `sacrebleu` library for corpus-level BLEU.

```python
# Production BLEU:
from sacrebleu.metrics import BLEU
bleu = BLEU()
score = bleu.corpus_score(hypotheses, [references])
```

t5-small on 4k opus_books examples: expect BLEU ~8–15 (low — 4k is tiny).
Full fine-tune (50k pairs, 5 epochs): ~25–30 BLEU, competitive for domain-specific translation.

---

## Expected Output

```
Device: mps

Epoch 1/3 | loss: 2.1403
Epoch 2/3 | loss: 1.7821
Epoch 3/3 | loss: 1.5240

Saved to models/05_transformers/t5_translation

── Inference ──
  EN  : The quarterly report shows a 12% increase in revenue.
  DE  : Der Quartalsbericht zeigt eine Zunahme von 12% beim Umsatz.
  Ref : Der Quartalsbericht zeigt eine Umsatzsteigerung von 12 Prozent.
  BLEU-1: 0.583

  EN  : Please submit all documents before the deadline.
  DE  : Bitte alle Unterlagen vor dem Stichtag einreichen.
  Ref : Bitte reichen Sie alle Unterlagen vor Ablauf der Frist ein.
  BLEU-1: 0.500

  EN  : The loan application requires proof of income.
  DE  : Der Kreditantrag erfordert einen Einkommensnachweis.
  Ref : Der Kreditantrag erfordert einen Einkommensnachweis.
  BLEU-1: 1.000
```

---

## Actual Output (Windows, GTX 1650 Ti, 2026-06-17)

```
Device: cuda

Epoch 1/3 | loss: 2.2194
Epoch 2/3 | loss: 1.9881
Epoch 3/3 | loss: 1.8935

Saved to models/05_transformers/t5_translation

-- Inference --
  EN  : The quarterly report shows a 12% increase in revenue.
  DE  : Der vierteljährliche Bericht zeigt eine Erhöhung der Einnahmen um 12 %.
  Ref : Der Quartalsbericht zeigt eine Umsatzsteigerung von 12 Prozent.
  BLEU-1: 0.455

  EN  : Please submit all documents before the deadline.
  DE  : Bitte schicken Sie alle Dokumente vor Ablauf der Frist.
  Ref : Bitte reichen Sie alle Unterlagen vor Ablauf der Frist ein.
  BLEU-1: 0.667

  EN  : The loan application requires proof of income.
  DE  : Der Darlehensantrag erfordert einen Nachweis des Einkommens.
  Ref : Der Kreditantrag erfordert einen Einkommensnachweis.
  BLEU-1: 0.429
```

- Translations are grammatically correct German — model learned the task
- BLEU scores lower than expected (0.43–0.67 vs 0.5–1.0) — different word choices, not wrong
- Fix applied: dataset config `en-de` → `de-en` (opus_books key order)

---

## How to Run

```powershell
python code_practice\05_transformers\05_t5_translation.py
```

First run: downloads `t5-small` (~242 MB) + opus_books/de-en (~9 MB).
CUDA training: ~3–5 min for 3 epochs on 4k examples.

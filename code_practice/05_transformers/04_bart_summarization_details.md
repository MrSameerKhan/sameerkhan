# Session 4 — BART Abstractive Summarization
Status: `✅ Run`

Theory: [../../../5.transformers/02_models/03_encoder_decoder_family.md](../../../5.transformers/02_models/03_encoder_decoder_family.md)

---

## Use Case

Earnings report summarizer: a 50-page analyst report → 3 bullet points in under 2 seconds. Used by research analysts, fintech platforms, and news desks that process hundreds of reports daily.

Real deployment: `facebook/bart-large-cnn` (prod-quality) is already fine-tuned for this exact task. This session shows how to fine-tune `bart-base` from scratch on your own document style.

---

## Key Concept — Abstractive vs Extractive

| Type | How | Output |
|------|-----|--------|
| Extractive (BERT QA) | Select existing spans from source | "...exact words from input..." |
| Abstractive (BART) | Generate new tokens via encoder-decoder | "Revenue surged 12% driven by services growth" |

Abstractive can rephrase, merge sentences, and synthesize — more human-like but harder to train.

---

## How BART Works

BART was pre-trained by corrupting text (masking, deletion, permutation, infilling) and training the decoder to reconstruct the original. This denoising objective makes it excellent at text repair — and summarization is "repair a corrupted/compressed document."

```
Article (512 tokens)
    │
    Encoder → contextual representations (512, 768)
    │
    Decoder → auto-regressively generates summary
             (each step attends to encoder output via cross-attention)
    │
    Summary tokens → decode → text
```

---

## Tensor Shapes

```
Encoder input (article):
  input_ids:      shape (512,)
  attention_mask: shape (512,)

Decoder target (summary):
  labels:         shape (128,) — target token ids; padding positions → -100

Model internals:
  encoder_hidden_states: shape (B, 512, 768)
  decoder cross-attends to encoder at every step

Loss: cross-entropy on labels (ignoring -100 padding positions)

Generation (inference):
  model.generate() → beam search over decoder vocabulary at each step
  Returns: shape (1, T) where T = number of generated tokens
```

**Why labels have -100 for padding?** PyTorch's cross-entropy loss ignores positions with label -100 — prevents the model from learning to generate padding tokens.

---

## Input Data — CNN/DailyMail v3.0.0

| Field | Example |
|-------|---------|
| article | Full news article (~750 tokens avg) |
| highlights | 3–5 bullet-point summary sentences |

- ~300k examples total
- Script uses 2,000 train examples (fine-tuning demonstration)
- `load_dataset("cnn_dailymail", "3.0.0")`

---

## Generation Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `num_beams=4` | 4 beams | Explores 4 candidate sequences; more accurate than greedy |
| `early_stopping=True` | — | Stop beam search when all beams hit EOS |
| `no_repeat_ngram_size=3` | 3-gram | Prevents phrase repetition in output |
| `max_new_tokens=128` | 128 | Maximum summary length |

---

## Expected Output

```
Device: mps

Epoch 1/2 | loss: 2.8341
Epoch 2/2 | loss: 2.3107

Saved to models/05_transformers/bart_summarization

── Inference ──
  Generated : Apple Q3 revenue hit $94.9B, up 5%, beating estimates. Services
              reached a record $24.2B. China sales fell 6%. Q4 guidance below forecasts.
  Reference : Apple Q3 revenue $94.9B, up 5%, EPS $1.53 beat estimates. Services
              hit record $24.2B. China sales fell 6%. Q4 guidance slightly below forecasts.
  ROUGE-L   : 0.6240
```

Production ROUGE-L with `bart-large-cnn` on CNN/DM test set: ~40–44. Our baseline after 2 epochs on 2k examples will be lower (~25–30).

---

## Actual Output (Windows, GTX 1650 Ti, 2026-06-17)

```
Device: cuda

Epoch 1/2 | loss: 2.6505
Epoch 2/2 | loss: 2.0713

Saved to models/05_transformers/bart_summarization

-- Inference --
  Generated : Apple Inc. reported third-quarter earnings on Thursday that exceeded
              Wall Street expectations, driven by strong iPhone sales in emerging
              markets and continued growth in its services segment. The company
              guided for Q4 revenue between $89B and $90B, slightly below forecasts.
  Reference : Apple Q3 revenue $94.9B, up 5%, EPS $1.53 beat estimates. Services
              hit record $24.2B. China sales fell 6%. Q4 guidance slightly below forecasts.
  ROUGE-L   : 0.1471
```

- Low ROUGE-L (0.1471) expected — 2 epochs on 2k examples is a demonstration, not production
- Model is generating plausible but verbose summaries (not yet compressed to bullet style)
- Warnings: `as_target_tokenizer` deprecated → use `text_target` arg in future versions

---

## How to Run

```powershell
python code_practice\05_transformers\04_bart_summarization.py
```

First run: downloads `facebook/bart-base` (~558 MB) + CNN/DailyMail (~838 MB).
CUDA training: ~5–8 min for 2 epochs on 2k examples.

**Upgrade path:** swap `MODEL_NAME = "facebook/bart-large-cnn"` and skip fine-tuning — it already achieves state-of-the-art on CNN/DM out of the box.

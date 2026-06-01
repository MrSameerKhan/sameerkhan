# Session 2 — BERT Named Entity Recognition (NER)
Status: `🔧 Code-built`

---

## Use Case

Tag each word in a sentence with its named-entity type: **PER** (person), **ORG** (organisation), **LOC** (location), **MISC** (other), or **O** (no entity).
Fine-tune a pretrained BERT model on CoNLL-2003, then run token-level inference on new sentences.

---

## Input Data Format

CoNLL-2003 — each row is a tokenised sentence with BIO-format NER tags:

| tokens | ner_tags |
|--------|----------|
| `["EU", "rejects", "German", "call"]` | `[3, 0, 7, 0]` → B-ORG, O, B-MISC, O |
| `["Peter", "Blackburn"]` | `[1, 2]` → B-PER, I-PER |

- 9 classes: `O  B-PER  I-PER  B-ORG  I-ORG  B-LOC  I-LOC  B-MISC  I-MISC`
- Downloaded automatically via `load_dataset("conll2003")` on first run
- Training uses 4,000 sentences, validation uses 500 sentences (subset for speed)

---

## Tensor Shapes

```
Input word list:  ["EU", "rejects", "German", "call"]

After BertTokenizerFast (is_split_into_words=True):
  input_ids:      [101, 7270, 22961, 1528, 1840, 102, 0, ...]   shape: (128,)
  attention_mask: [1,   1,    1,     1,    1,    1,  0, ...]   shape: (128,)

Label alignment (first subword → real label, continuations → -100):
  labels:         [-100, 3, 0, 7, 0, -100, -100, ...]           shape: (128,)
  ↑ -100 on [CLS]/[SEP]/padding and subword continuations — ignored by cross-entropy loss

Model output:
  logits:  shape: (batch, 128, 9)   — one score per class per token position
  loss:    scalar — averaged over non-(-100) positions only
```

**Key difference from classification**: labels are per-token `(128,)`, not per-example `(1,)`.  
**Subword alignment rule**: first subword of each word → real label; continuation subwords (`##...`) → -100.  
**Why bert-base-cased**: capitalisation is a strong NER signal — `Apple` (company) vs `apple` (fruit).

---

## Inference Input

Two sentences passed to `predict()` at the end of the script:

```python
"Apple CEO Tim Cook announced new products in San Francisco on Tuesday ."
"Elon Musk 's company Tesla is based in Austin Texas ."
```

---

## Expected Output

```
Device: mps

Epoch 1/3 | loss: 0.3521 | val_f1: 0.7810
Epoch 2/3 | loss: 0.1803 | val_f1: 0.8230
Epoch 3/3 | loss: 0.1241 | val_f1: 0.8490

Saved to models/05_transformers/bert_ner

── Inference ──
  Apple                B-ORG
  Tim                  B-PER
  Cook                 I-PER
  San                  B-LOC
  Francisco            I-LOC

  Elon                 B-PER
  Musk                 I-PER
  Tesla                B-ORG
  Austin               B-LOC
  Texas                I-LOC
```

- Val F1 on 4k subset: ~78–86% after 3 epochs (full dataset reaches ~90–91 F1)
- `seqeval` F1 is **entity-span-level**: a partial match on a multi-token entity counts as wrong

---

## How to Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE python 02_bert_ner.py
```

First run downloads `bert-base-cased` (~400 MB) and CoNLL-2003 (~7 MB) — cached after that.  
Training on Apple Silicon MPS: ~5–8 minutes for 3 epochs on 4k samples.

**Extra dependency** (already in `environment.yml`): `seqeval`

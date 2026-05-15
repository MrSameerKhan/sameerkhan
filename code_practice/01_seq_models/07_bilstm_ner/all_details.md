# BiLSTM for NER — All Details

> Phase 1, Session 7 of the coding practice sequence. **First real ML engineering session:** train/val/test split, F1 metrics, padded batches, overfitting visible.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Why BiLSTM (and not just LSTM)](#2-why-bilstm-and-not-just-lstm)
3. [Architecture](#3-architecture)
4. [What's New vs Sessions 1-6](#4-whats-new-vs-sessions-1-6)
5. [Dataset & Preprocessing Pipeline](#5-dataset--preprocessing-pipeline)
6. [Train / Val / Test Split — Why and How](#6-train--val--test-split--why-and-how)
7. [Padded Batches & Masking](#7-padded-batches--masking)
8. [Metrics — F1 for NER](#8-metrics--f1-for-ner)
9. [Setup Required](#9-setup-required)
10. [How to Run](#10-how-to-run)
11. [Expected Outputs](#11-expected-outputs)
12. [Reading the Training Log](#12-reading-the-training-log)
13. [Files in This Folder](#13-files-in-this-folder)
14. [Next Steps](#14-next-steps)

---

## 1. Objective

Build a **BiLSTM token tagger** that learns to identify Named Entities in sentences:

```
Input:  "Sarah Chen works as Senior Analyst in Risk department."
Output: [PERSON: Sarah Chen] works as Senior Analyst in Risk department.
```

Entities in our data: **PERSON**, **PRODUCT**, **MONEY** (tagged in BIO scheme).

This is the **first session with a real ML pipeline:**
- Train / validation / test split
- Vocab built from train only (no data leakage)
- Padded batches with masked loss
- F1 score on validation, not just training loss
- Save best model by val F1
- Per-entity precision / recall / F1 on test

---

## 2. Why BiLSTM (and not just LSTM)

NER needs **both-side context** to tag tokens correctly:

```
Sentence: "Sarah Chen serves as Head of Loans"

To tag "Chen" as I-PERSON, you need:
  - LEFT context:  "Sarah" (it's part of a name that started)
  - RIGHT context: "serves" (the name ends here, not continues)

A forward-only LSTM sees only the past.
A bidirectional LSTM reads forward AND backward, concatenates → both directions.
```

Architecture:

```
                    → forward LSTM  ——→ h_f (forward hidden)
char_idx → Embed —                       ← concat → Linear → logits
                    → backward LSTM ——→ h_b (backward hidden)
```

Output dimension is `2 × hidden_size` (forward concat backward) → `Linear(2*hidden, num_labels)` projects to per-token tag scores.

---

## 3. Architecture

```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=64, hidden_size=64, pad_token_id=0):
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc     = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, x):
        emb    = self.embed(x)       # (batch, seq, embed_dim)
        out, _ = self.bilstm(emb)    # (batch, seq, 2*hidden)
        return self.fc(out)          # (batch, seq, num_labels)
```

~30 lines. The PyTorch idioms (`Embedding` with `padding_idx`, `bidirectional` flag) handle the complexity.

### Parameter count

With `vocab_size ~160`, `num_labels=8`, `embed_dim=64`, `hidden_size=64`:

- Embedding: 160 × 64 = 10,240
- BiLSTM: ~66,000 (4 gates × 2 directions × (in + hidden + bias))
- Linear: 128 × 8 + 8 = 1,032
- **Total: ~77,000 params**

---

## 4. What's New vs Sessions 1-6

| Concept | Sessions 1-6 | Session 7 |
|---|---|---|
| Task | Character LM | **Token NER tagging** |
| Vocab | All chars in corpus | **Word vocab built from train only** |
| OOV handling | Not needed | **`<UNK>` token** |
| Sequence length | Fixed (25 chars) | **Variable per example** |
| Padding | Not needed | **Dynamic per batch** |
| Loss masking | Not needed | **`ignore_index=PAD_LABEL_ID`** |
| Train / val / test | One corpus | **80 / 10 / 10 split** |
| Evaluation | Look at samples | **F1 on validation per epoch** |
| Best-model save | Last epoch | **Best validation F1** |
| Metrics | Loss only | **Precision, Recall, F1 per entity** |
| Direction | Forward only | **Bidirectional** |

7 new patterns in one session. This is the real-ML-engineering jump.

---

## 5. Dataset & Preprocessing Pipeline

Source: `shared_dataset.get_ner_data(n_total=200, seed=42)` — 200 synthetic Acme NER examples.

Each example: `(tokens, BIO_tags)` where tokens are words and tags follow BIO scheme.

### Preprocessing flow (in `data.py`)

```
get_ner_data(n_total=200)      ← 200 (tokens, tags) examples
    ↓
split_data(seed=42, 80/10/10)  ← train: 160, val: 20, test: 20
    ↓
build_vocab(train_only)        ← {<PAD>: 0, <UNK>: 1, "Sarah": 2, ...}
build_label_vocab(train_only)  ← {<PAD>: 0, "O": 1, "B-PERSON": 2, ...}
    ↓
NERDataset → DataLoader        ← collate_fn pads each batch dynamically
```

### Critical detail: vocab built from train ONLY

If we built vocab from all data, val/test words would all be "in vocab" and we couldn't see how the model handles OOV. That's data leakage. **Real ML always builds vocab from train and lets val/test handle the consequences.**

---

## 6. Train / Val / Test Split — Why and How

### Why three splits, not two?

- **Train** — model learns on this
- **Val** — chooses hyperparameters / best epoch (we save the best-val-F1 checkpoint)
- **Test** — final number you report. **Used ONCE, after all decisions are made.**

Using val data for both selection AND final reporting inflates your metrics.

### Why deterministic seed?

Same seed → same split. Reproducible results across runs. Without this, every run gives different splits → different F1 → can't compare changes.

### Ratio (80/10/10)

- 200 examples × 0.8 = 160 train
- 200 × 0.1 = 20 val
- 200 × 0.1 = 20 test

Small val/test sets mean F1 will be noisy. With 200 examples, this is the best we can do without harming train size.

---

## 7. Padded Batches & Masking

### The problem

Sentences have different lengths:
- "Sarah Chen serves as Head of Loans ." (8 tokens)
- "The Premium Checking earns 2.5 percent ." (7 tokens)
- "Premium ." (2 tokens)

You can't put them in a tensor with a single shape unless you **pad** to max length.

### The solution — `collate_fn` does dynamic padding

For each batch:
1. Find max length in this batch
2. Pad shorter sequences with `<PAD>` (token ID 0)
3. Pad shorter label sequences with `<PAD>` label ID (0)
4. Return `(tokens, labels, lengths)` — lengths tell us where each real sequence ends

### Why dynamic (not fixed) padding?

If we padded all sequences to the corpus max (say 20 tokens), most batches would be 90% pad → wasted compute. Dynamic padding (per-batch max) wastes much less.

### CRITICAL — masking in loss

```python
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)
```

Without `ignore_index`, the model gets penalized for predicting `<PAD>` positions, biasing it toward predicting the dominant tag. **This is THE classic NER bug.**

For evaluation, we slice predictions to true length before passing to seqeval:

```python
for i in range(len(lengths)):
    L = lengths[i].item()
    true_seqs.append([ix_to_label[labels[i, j]] for j in range(L)])
```

---

## 8. Metrics — F1 for NER

### Why not accuracy?

Most tokens are `O` (~75% in our data). A model that predicts `O` for everything gets 75% token accuracy and 0% useful. **Token accuracy is misleading for NER.**

### What we use: entity-level F1 (via `seqeval`)

- **Precision**: of predicted entities, how many are correct?
- **Recall**: of true entities, how many did we find?
- **F1**: 2 × P × R / (P + R) — harmonic mean
- An entity is "correct" only if its type AND span (start, end) match exactly

`seqeval.classification_report` gives per-entity numbers + macro/weighted averages.

### Example

```
              precision  recall  f1-score  support
    PERSON       0.95    0.92      0.93       12
   PRODUCT       0.87    0.91      0.89       11
     MONEY       1.00    0.83      0.91        6
 micro avg       0.92    0.89      0.91       29
 macro avg       0.94    0.89      0.91       29
```

MONEY has 100% precision (never falsely predicted) but only 83% recall (missed some). This is the kind of insight you'd never get from a single accuracy number.

---

## 9. Setup Required

One-time install:

```bash
pip install seqeval
```

This is a compliance-safe public package (used by HuggingFace, AllenNLP, all major NER pipelines). Pure Python, no external API calls.

If you skip it, training still works but uses token-level accuracy as a fallback (less meaningful).

---

## 10. How to Run

```bash
cd code_practice/01_seq_models/07_bilstm_ner
python train.py
```

Takes ~30-60s on CPU. Then:

```bash
python predict.py --text "Sarah Chen works as Senior Analyst in Risk department."
python predict.py --text "The Premium Checking has 2.5 percent rate."
python predict.py --text "Apply for Personal Loan with up to 50000 dollars."
```

---

## 11. Expected Outputs

### Training output (abridged)

```
Loading data...
Device:      cpu
Train:       160 examples
Val:          20 examples
Test:         20 examples
Vocab:       ~160 words (incl <PAD>, <UNK>)
Labels:      8 (incl <PAD>) ['<PAD>', 'B-MONEY', 'B-PERSON', 'B-PRODUCT',
                              'I-PERSON', 'I-PRODUCT', 'O']

Train label distribution (~1200 tokens):
  O              ~800  (~70%)
  B-PERSON        ~80  (~7%)
  I-PERSON        ~80  (~7%)
  B-PRODUCT       ~80  (~7%)
  ...

OOV (out-of-vocab) statistics:
  Val : ~150 tokens, ~10 OOV (~6%)
  Test: ~150 tokens, ~12 OOV (~8%)

Model params: ~77,000

=== Training (30 epochs) ===
Epoch | Train Loss | Val Loss | Val F1 | Best
    1 |     0.7234 |   0.5891 | 0.4123 | ★
    2 |     0.3845 |   0.4012 | 0.7234 | ★
  ...
   15 |     0.0123 |   0.1234 | 0.9123 | ★
   16 |     0.0099 |   0.1456 | 0.9089 |
  ...

Best val F1: 0.91 at epoch 15
```

### Prediction output

```
Input:
  Sarah Chen works as Senior Analyst in Risk department.

Raw tokens and tags:
  Sarah               B-PERSON ★
  Chen                I-PERSON ★
  works               O
  as                  O
  Senior              O
  Analyst             O
  in                  O
  Risk                O
  department          O
  .                   O

Highlighted:
  [PERSON: Sarah Chen] works as Senior Analyst in Risk department .
```

---

## ✅ Actual Run Results

*(MacBook M1 — device: mps)*

### Data pipeline stats

- 200 examples → 160 train / 20 val / 20 test
- Vocab: **176 words** (incl `<PAD>`, `<UNK>`)
- Labels: 7 (incl `<PAD>`) `['<PAD>', 'B-MONEY', 'B-PERSON', 'B-PRODUCT', 'I-PERSON', 'I-PRODUCT', 'O']`
- OOV: **0%** on both val and test
- Model params: **78,727**
- Class imbalance: 79% `O`, 8% B-MONEY, 4% I-PRODUCT, 3% each B-PRODUCT / B-PERSON / I-PERSON

### Training (30 epochs, 9.4s on MPS)

| Epoch | Train Loss | Val Loss | Val F1 |
|---|---|---|---|
| 1 | 1.8994 | 1.8035 | 0.0612 |
| 5 | 0.8950 | 0.7710 | 0.0000 |
| 14 | 0.4323 | 0.4260 | 0.4151 |
| 17 | 0.3219 | 0.3151 | 0.7097 |
| 21 | 0.2004 | 0.1957 | 0.8286 |
| 25 | 0.1199 | 0.1178 | 0.9487 |
| **29 ★** | **0.0732** | **0.0739** | **1.0000** |
| 30 | 0.0656 | 0.0661 | 1.0000 |

**Best val F1: 1.0000 at epoch 29 → checkpoint saved.**

### Test set results

- **Test F1: 0.9863**

### Per-entity classification report (test set)

```
              precision    recall  f1-score   support

       MONEY       1.00      1.00      1.00        19
      PERSON       1.00      0.90      0.95        10
     PRODUCT       1.00      1.00      1.00         8

   micro avg       1.00      0.97      0.99        37
   macro avg       1.00      0.97      0.98        37
weighted avg       1.00      0.97      0.99        37
```

Improvement over work laptop (Test F1: 0.9333 → **0.9863**). PERSON recall 0.90 — model missed one name span; MONEY and PRODUCT perfect.

### Predictions

**"Sarah Chen works in the Loans department."**
```
[PERSON: Sarah Chen] works in the Loans department .
```
Correct — both tokens tagged, right span.

**"The Premium Checking has 2.5 percent rate."**
```
The [PRODUCT: Premium Checking] has [MONEY: 2] .5 percent rate .
```

### The Bug — Tokenizer Splits Decimals Wrong

`2.5` → `["2", ".5"]` — model tags `2` as B-MONEY, `.5` as O. Root cause: tokenizer replaces all `.` with ` . `, splitting the decimal.

**The model is correct on the tokens it sees.** Real-world fix:
```python
# Don't split periods that are between digits
text = re.sub(r"(\d)\.(\d)", r"\1.\2", text)
```

**Interview takeaway:** "NER scored 98% F1 but mislabelled prices in production — tokenization split decimal points. ML metrics don't catch input preprocessing bugs."

### The 8 ML engineering patterns demonstrated

1. **Train/val/test split** — val F1 peaked at epoch 29, saved that checkpoint
2. **Vocab from train only** — 0% OOV on synthetic; would matter on real data
3. **Best-checkpoint saving** — epoch 30 model same F1 but epoch 29 was best
4. **Loss masking via `ignore_index`** — PAD positions excluded from loss
5. **Val/test gap** — Val 1.00 vs Test 0.9863 (reality check)
6. **F1 not accuracy** — all-O baseline would look 79% accurate
7. **Per-entity breakdown** — PERSON recall 0.90 flags where to improve
8. **Real bug discovery** — decimal tokenization found from prediction inspection

These all happened in one session because we built the pipeline properly.

---

## 12. Reading the Training Log

| Pattern in log | What it means |
|---|---|
| Train loss decreasing, val loss decreasing | Model is learning generalizable patterns |
| Train loss → 0, val loss flat | Model is reaching capacity |
| Train loss → 0, val loss **increasing** | **Overfitting** — model memorizing train, not generalizing |
| Val F1 plateaus | Model has learned what it can from this data + capacity |
| Best epoch is early (e.g. 5/30) | Model overfit fast; could reduce capacity or add regularization |
| Best epoch is at the end | Model still learning; train longer |

### What you'll likely see here

With 160 training examples and a 77K-param model, **the model has way more capacity than the data needs.** Expect:
- Train loss drops to near-0 fast
- Val F1 peaks around epoch 10-20 then plateaus or wobbles
- Some val examples will get OOV-related errors

This is **expected behavior on small datasets.** In real projects you'd:
- Reduce model size
- Add dropout / weight decay
- Use pre-trained embeddings
- Get more data

We'll address these incrementally in later sessions.

---

## 13. Files in This Folder

| File | Purpose |
|---|---|
| `data.py` | Vocab building, train/val/test split, NERDataset, padded DataLoader |
| `model.py` | BiLSTMTagger class + save/load helpers |
| `train.py` | Training loop with val tracking, best-checkpoint save, F1 metrics, error analysis |
| `predict.py` | CLI inference with pretty NER output (`[PERSON: ...]` formatting) |
| `all_details.md` | This document |
| `checkpoints/bilstm_ner.pt` | Best-val-F1 checkpoint (saved after training) |

---

## 14. Next Steps

### Session 8 — Bahdanau Attention on BiLSTM

On top of this BiLSTM, add **additive attention** — a mechanism that lets the model focus on specific tokens dynamically. This is the **direct mathematical bridge** to transformer attention.

What you'll learn:
- Attention as a weighted sum
- Query / key / value (in the original Bahdanau form, just q and h)
- Softmax-based weighting
- The same matrix shapes that show up in transformer attention

By the end of Session 8, transformer attention (Phase 2) will feel like a small refactor of attention you already know.

### Session 9 — Seq2Seq with attention

Encoder-decoder architecture. Encoder = BiLSTM. Decoder = LSTM with attention over encoder outputs. Used for translation, summarization, question answering.

After Session 9, Phase 1 is complete and Phase 2 transformers will feel natural.

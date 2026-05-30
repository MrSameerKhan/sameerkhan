# Session 3 — BERT Extractive Question Answering
Status: `🔧 Code-built`

Theory: [../../../5.transformers/02_models/01_bert_family.md](../../../5.transformers/02_models/01_bert_family.md)

---

## Use Case

Policy document QA: user asks "What is the early repayment charge?" — the model extracts the exact answer span from the uploaded document. No hallucination possible: every answer is a substring of the input context.

Real deployment: banks, insurance companies, legal teams — any domain with dense policy PDFs that users need to query without a search UI.

---

## Key Concept — Span Extraction

Unlike classification (one label per example) or NER (one label per token), QA predicts two positions: where the answer **starts** and where it **ends** in the context.

```
[CLS] question tokens [SEP] context tokens [SEP] [PAD...]
      ─── 0 ────────        ─── 1 ────────        padding
      token_type_ids=0      token_type_ids=1

Model output:
  start_logits: (384,) → argmax → start token position
  end_logits:   (384,) → argmax → end token position

Answer = context[offset_mapping[start][0] : offset_mapping[end][1]]
         ↑ character-level extraction using tokenizer's offset_mapping
```

---

## Tensor Shapes

```
Tokenizer input: question="What is the max LTV?" + context (384 tokens total)

  input_ids:       shape (384,)  — question + [SEP] + context + [SEP] + [PAD]
  attention_mask:  shape (384,)  — 1 for real tokens, 0 for padding
  token_type_ids:  shape (384,)  — 0=question, 1=context, 0=padding
  offset_mapping:  shape (384, 2) — (char_start, char_end) per token

  start_positions: scalar — ground truth start token index
  end_positions:   scalar — ground truth end token index

Model loss: cross_entropy(start_logits, start_pos) + cross_entropy(end_logits, end_pos)
            averaged, divided by 2
```

**Why MAX_LEN=384?** BERT's 512-token limit minus question overhead (~100 tokens max) leaves 384 for the context. SQuAD uses `truncation="only_second"` — question is never cut.

**Why offset_mapping?** Tokenization splits words into subwords (e.g., "repayment" → "repay" + "##ment"). offset_mapping maps each subword token back to its character span in the original string — needed to convert token indices to readable text.

---

## Input Data — SQuAD v1.1

| Field | Example |
|-------|---------|
| question | "What is the max LTV for first-time buyers?" |
| context | "...95% for first-time buyers under Help to Buy..." |
| answers.text | `["95% for first-time buyers"]` |
| answers.answer_start | `[42]` ← char offset into context |

- ~87k training examples, ~10k validation
- Script uses 4,000 train examples (subset for speed)
- Downloaded automatically via `load_dataset("squad")`

---

## Expected Output

```
Device: mps

Epoch 1/2 | loss: 1.4821
Epoch 2/2 | loss: 0.9103

Saved to models/05_transformers/bert_qa

── Inference ──
  Q: What is the maximum LTV for first-time buyers?
  A: 95% for first-time buyers under the Help to Buy scheme

  Q: How long is the fixed interest rate period?
  A: 5 years

  Q: What is the early repayment charge?
  A: 2%
```

Full SQuAD (87k examples, 3 epochs) → ~80 Exact Match / ~88 F1.

---

## How to Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE python 03_bert_qa.py
```

First run: downloads `bert-base-uncased` (~400 MB) + SQuAD (~30 MB).
MPS training time: ~15–20 min for 2 epochs on 4k examples (MAX_LEN=384 is 3× slower than 128).

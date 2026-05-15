# Session 9 — Seq2Seq with Bahdanau Attention (Phase 1 Finale)

> Phase 1, Session 9. Full encoder-decoder architecture — every key idea in transformers, in RNN form.

---

## Table of Contents

1. [Objective](#1-objective)
2. [What Seq2Seq Is](#2-what-seq2seq-is)
3. [Architecture](#3-architecture)
4. [Bahdanau Attention in Decoder](#4-bahdanau-attention-in-decoder)
5. [Task — QA Pair Generation](#5-task--qa-pair-generation)
6. [Training Details](#6-training-details)
7. [Teacher Forcing](#7-teacher-forcing)
8. [Bridge to Transformers](#8-bridge-to-transformers)
9. [How to Run](#9-how-to-run)
10. [Expected Outputs](#10-expected-outputs)
11. [Files in This Folder](#11-files-in-this-folder)
12. [✅ Actual Run Results](#-actual-run-results)
13. [Next Steps](#13-next-steps)

---

## 1. Objective

Build a **sequence-to-sequence model with Bahdanau attention** — encoder reads the input, decoder generates output one token at a time while attending to encoder outputs.

By the end of this session you should be able to:
- Implement encoder (BiLSTM) + decoder (LSTM + attention) architecture
- Understand teacher forcing during training vs free-running during inference
- Map this architecture to transformer encoder-decoder (same structure, different attention)

Task: given a question-like input sentence from Acme data, generate the corresponding answer.

---

## 2. What Seq2Seq Is

```
Input:  "What is the interest rate for Premium Checking?"
Output: "The interest rate for Premium Checking is 2.5 percent."

Encoder reads input → produces context vectors (one per token)
Decoder generates output → at each step, attends to encoder context
```

This is the architecture behind early neural machine translation, summarization, and QA systems.

---

## 3. Architecture

```
ENCODER (BiLSTM):
  input tokens → Embedding → BiLSTM → encoder_outputs [T_src, 2H]
                                     + final hidden h_n [2, B, H]

DECODER (LSTM + Attention):
  For each output step t:
    attention_context = Σ α_i · encoder_outputs[i]   ← attend over all encoder positions
    decoder_input     = [token_embed ; attention_context]
    h_dec, c_dec      = LSTM(decoder_input, h_dec, c_dec)
    logit             = Linear(h_dec) → vocab_size

ATTENTION (Bahdanau additive):
  score_i = v^T · tanh(W_enc·encoder_outputs[i] + W_dec·h_dec)
  α       = softmax(scores)
  context = Σ α_i · encoder_outputs[i]
```

---

## 4. Bahdanau Attention in Decoder

At each decoder step, attention re-queries the full encoder output:

```python
# h_dec: [B, H_dec]   current decoder hidden
# enc_out: [B, T, H_enc]   all encoder outputs

energy = tanh(enc_proj + dec_proj)   # [B, T, attn_dim]
scores = v(energy).squeeze(-1)       # [B, T]
alpha  = softmax(scores)             # [B, T]
context = (alpha.unsqueeze(1) @ enc_out).squeeze(1)   # [B, H_enc]
```

This is the same math as Session 8 — but applied **per decoder step** rather than once for pooling.

---

## 5. Task — QA Pair Generation

```python
get_qa_pairs()
→ [(question, answer), ...]
```

The Acme corpus has Q&A pairs about products, employees, and policies. We train the model to map questions to answers.

Dataset split: 80/10/10 train/val/test.

Vocabulary: built from train only (no data leakage). Special tokens: `<PAD>`, `<UNK>`, `<SOS>` (start of sequence), `<EOS>` (end of sequence).

---

## 6. Training Details

| Hyperparameter | Value |
|---|---|
| Encoder hidden | 128 (BiLSTM → 256 total) |
| Decoder hidden | 256 |
| Attention dim | 128 |
| Embed dim | 64 |
| Epochs | 50 |
| Optimizer | Adam, lr=1e-3 |
| Teacher forcing ratio | 0.5 |
| Gradient clipping | 5.0 |
| Batch size | 16 |

---

## 7. Teacher Forcing

During training, with probability `teacher_forcing_ratio`, feed the **ground-truth** token as the next decoder input instead of the model's own prediction:

```
teacher_forcing=True  → decoder_input[t+1] = target[t]   ← fast, stable training
teacher_forcing=False → decoder_input[t+1] = argmax(logit[t])  ← realistic, harder
```

Without teacher forcing, early training fails (decoder generates garbage → learns nothing).  
With only teacher forcing, the model becomes brittle (never sees its own errors).  
Mix of 50% is the standard practice.

---

## 8. Bridge to Transformers

| Seq2Seq (this session) | Transformer |
|---|---|
| Encoder: BiLSTM | Encoder: multi-head self-attention + FFN stacked |
| Decoder: LSTM + cross-attention | Decoder: masked self-attention + cross-attention + FFN |
| Bahdanau cross-attention | Scaled dot-product cross-attention |
| Sequential (step by step) | Parallel (all positions at once) |
| One attention head | Multiple attention heads |

The **cross-attention** in a transformer decoder is exactly the decoder attention here — queries come from the decoder, keys and values come from the encoder. The only differences are scoring function (dot-product vs additive) and parallelism.

---

## 9. How to Run

```bash
cd code_practice/01_seq_models/09_seq2seq
python train.py
python predict.py --question "What is the interest rate for Premium Checking?"
python predict.py --question "Who manages the Loans department?"
```

---

## 10. Expected Outputs

### Training

```
Epoch  10 | train loss 3.2145 | val loss 3.4821
Epoch  20 | train loss 2.8923 | val loss 3.1204
Epoch  50 | train loss 2.1045 | val loss 2.6831
```

### Inference

```
Question: What is the interest rate for Premium Checking?
Answer  : The interest rate for Premium Checking is 2.5 percent .
```

---

## 11. Files in This Folder

| File | Purpose |
|---|---|
| `data.py` | Tokenize, vocab with SOS/EOS, QA dataset, padded batches |
| `model.py` | `Encoder`, `BahdanauAttention`, `Decoder`, `Seq2Seq` wrapper |
| `train.py` | Teacher forcing loop, best model save by val loss |
| `predict.py` | Greedy decode with `--question` CLI arg |
| `all_details.md` | This document |
| `checkpoints/seq2seq.pt` | Saved best model |

---

## ✅ Actual Run Results

*(MacBook M1 — device: mps)*

### Training

```
Device: mps
Vocab size : 142
Train pairs: 44 | Val: 5 | Test: 7
Parameters : 1,042,446

Epoch   5 | train 4.0038 | val 4.1582 | 0.1s
Epoch  10 | train 3.0132 | val 3.4392 | 0.2s
Epoch  20 | train 1.7692 | val 2.7793 | 0.1s
Epoch  50 | train 0.5296 | val 3.2838 | 0.1s

Best val loss : 2.7458
Test loss     : 2.4262
```

### Inference

```
Question : Who is Sarah Chen?
Answer   : You is is our our in the the department .

Question : What is the interest rate for Premium Checking?
Answer   : The Student Checking earns earns percent annual annual .
```

### ML Engineering Lesson — Overfitting at Scale

**The numbers tell the full story:**

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 5 | 4.0038 | 4.1582 |
| 10 | 3.0132 | 3.4392 |
| 20 | 1.7692 | **2.7793** ← best val |
| 50 | 0.5296 | 3.2838 ↑ diverging |

The model hits its best generalization at epoch ~20 then overfits severely. By epoch 50, train loss is 0.53 while val loss climbs back to 3.28 — a 6× gap. Classic overfitting signature.

**Why it's expected:** 1,042,446 parameters trained on 44 examples. The model has ~23,700 parameters per training pair. It memorizes the training set perfectly but can't generalize.

**The repetition artifact:**

```
Answer: You is is our our in the the department .
Answer: The Student Checking earns earns percent annual annual .
```

Repetitive loops are characteristic of an overfit seq2seq model: it learned the surface n-gram patterns from training but not the underlying semantics. The decoder gets stuck repeating tokens that scored high in training.

**Real-world fixes (in order of impact):**

1. **More data** — 44 examples is unusable for a 1M param model. Need 10K+ pairs minimum
2. **Smaller model** — reduce `enc_hidden`, `dec_hidden`, `embed_dim` to match data size
3. **Dropout** — add `nn.Dropout` after embedding and before fc output
4. **Beam search** — replaces greedy decode; reduces repetition loops
5. **Coverage penalty** — explicitly penalize attending to the same source positions repeatedly

**What this session actually teaches (despite the bad outputs):**

The architecture is correct. The attention mechanism works. The encoder-decoder pipeline runs. The failure mode is data starvation — a real engineering problem, not a code problem. Recognizing this distinction is core ML engineering skill.

**Interview answer:** "The model trains fine — loss drops, gradients flow, attention weights are valid. The output quality fails because we have ~23K params per training example. I'd diagnose this by plotting train vs val loss curves, seeing the divergence at epoch 20, and concluding: more data, less model, or both."

---

## 13. Next Steps

### Phase 2 — Transformers

With this session complete, you've built every key idea in transformers from scratch:
- Attention mechanism (Session 8)
- Encoder-decoder with cross-attention (this session)

Phase 2 replaces:
- BiLSTM encoder → Transformer encoder (self-attention + FFN, stacked)
- LSTM decoder → Transformer decoder (masked self-attention + cross-attention)
- Sequential → Parallel (all positions computed simultaneously)

The math is the same. The architecture is the same shape. Only the building blocks change.

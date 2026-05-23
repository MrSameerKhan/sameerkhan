# BERT Fine-tuning — Deep Dive

> The full machinery: MLM + NSP pretraining math, fine-tuning heads, learning-rate schedules, layer-wise unfreezing. The patterns that actually move accuracy in production.

---

## Table of Contents

1. Objective
2. MLM pretraining math
3. NSP — used by BERT; dropped by RoBERTa
4. Fine-tuning heads for downstream tasks
5. Hyperparameters and LR schedule
6. Layer-wise unfreezing and discriminative LR
7. Failure modes
8. Interview questions (5)
9. Further reading

---

## 1. Objective

You used BERT in production at ICE (94% accuracy boost). Senior interviewer will ask:
- Why does BERT work? What does MLM actually learn?
- How do you fine-tune BERT? What learning rate? How many epochs?
- What's the fine-tuning head for classification / NER / QA?
- What goes wrong and how do you fix it?

This file is the reference for those questions.

---

## 2. MLM Pretraining Math

### The Training Task

Given a sequence of tokens `x_1, x_2, ..., x_n`, randomly select 15% of positions and:
- 80% of selected: replace with `[MASK]`
- 10% of selected: replace with a RANDOM token
- 10% of selected: keep ORIGINAL token

Then predict the original token at every masked position.

### The Loss

For each masked position i:

```
L_i = -log P(x_i_original | x_others)
```

Total loss is the mean over masked positions in the batch.

### Why the 80/10/10 Split?

If all masked positions were `[MASK]`, the model never sees its own correct tokens at those positions during pretraining — mismatch with downstream tasks where there's no `[MASK]` token.

10% random tokens force the model to consider the context even when the input token is wrong. 10% keep-original forces the model to NOT just default to "copy input" for non-masked positions.

This is one of those subtle paper details that matters in practice. RoBERTa kept it; ELECTRA replaced it with a different paradigm entirely.

### What MLM Actually Learns

The model learns to predict any token from its context. This forces:
- **Syntactic understanding** (predicting next word requires grammar)
- **Semantic knowledge** (predicting "Paris" requires knowing it's the capital of France)
- **Coreference** (predicting pronouns requires tracking entities)
- **World knowledge** (predicting common facts)

All without labels. **Self-supervised learning at its purest.**

---

## 3. NSP — Used by BERT, Dropped by RoBERTa

BERT's original design had a SECOND objective: Next Sentence Prediction (NSP).

```
Input: [CLS] sentence_A [SEP] sentence_B [SEP]
Task:  Is sentence_B the actual next sentence after A? (binary)
Half the time: real consecutive sentences
Half the time: sentence_B is a random sentence from the corpus
```

Use the `[CLS]` token's final hidden state to predict the binary label.

### Why Dropped

RoBERTa (Liu et al. 2019) showed NSP doesn't help much:
- NSP is too easy — the model learns it in a few thousand steps
- The 50/50 mix means the signal is "is this random?" — surface-level
- Better to use the parameters / compute on more MLM steps

ALBERT replaced NSP with Sentence Order Prediction (SOP) — distinguishing real next sentence from REVERSED order. Harder, more useful.

**Bottom line:** Original BERT: MLM + NSP. RoBERTa: just MLM (and trained longer with bigger batches). Most modern BERT-family models (DeBERTa, DistilBERT) follow RoBERTa.

---

## 4. Fine-Tuning Heads for Downstream Tasks

BERT comes pretrained. To use it on a task, you add a small head and fine-tune the whole thing (or just the head).

### Classification (Sentence-Level)

```
[CLS] token's final hidden state → Dropout → Linear → softmax over C classes
```

Train: cross-entropy loss. Whole BERT + linear head trains end-to-end.

### Token Classification (NER, POS)

```
Every token's final hidden state → Dropout → Linear → softmax over tag classes
```

BIO tagging: classes are O, B-PER, I-PER, B-LOC, I-LOC, ...

### Question Answering (Extractive)

```
Each token's final hidden state → two linear heads:
  - start_logit  (likelihood this token is the answer start)
  - end_logit    (likelihood this token is the answer end)

Loss: cross-entropy on start position + cross-entropy on end position
```

At inference: pick the (i, j) pair with highest `start_i + end_j` subject to j ≥ i.

### Sentence-Pair Tasks (NLI, Similarity)

```
Input: [CLS] sentence_A [SEP] sentence_B [SEP]
Use [CLS] for classification head (entailment / contradiction / neutral)
Or for regression (similarity score)
```

### Sentence Embeddings (Use as Feature Extractor)

- Take `[CLS]` (not great quality)
- Or **mean-pool over token embeddings** (better)
- Or use a **Sentence Transformer** (sentence-BERT) fine-tuned with contrastive loss for similarity

The last option is what production RAG systems use. Plain BERT `[CLS]` is poor for similarity; Sentence-BERT was a major fix.

---

## 5. Hyperparameters and LR Schedule

The BERT paper specifies a recipe. Modern best practice:

### Learning Rate

- **Base BERT:** 2e-5 to 5e-5
- **Large BERT:** 1e-5 to 3e-5 (smaller LR for larger model)
- **Below 1e-5:** usually too slow
- **Above 1e-4:** usually destabilizes pretrained features

### Schedule

- **Linear warmup** for the first 10% of steps (LR ramps from 0 to peak)
- **Linear decay** for the remaining 90% (LR drops to 0)
- Some variants: cosine decay, polynomial decay

### Batch Size

- 16-32 for sentence-level tasks (limited by GPU memory)
- Use gradient accumulation if you want larger effective batch

### Epochs

- **2-4 epochs** for most fine-tuning. More epochs → catastrophic forgetting of pretrained knowledge.
- If accuracy keeps improving at 4 epochs, your data might be very different from pretraining — keep going but watch eval set.

### Optimizer

- AdamW with weight decay 0.01
- ε = 1e-8 (Adam epsilon — small for numerical stability)
- β1=0.9, β2=0.999 (Adam defaults)

### Dropout

- 0.1 by default — works for most tasks
- 0.2-0.3 for very small datasets (more regularization)

### Maximum Sequence Length

- Always 512 unless you really need more
- Longer sequences are much slower (O(n²) attention)
- For long docs, use sliding windows or chunking

---

## 6. Layer-Wise Unfreezing and Discriminative LR

Sometimes you don't want to fine-tune ALL of BERT. Especially with small datasets — risk of overfitting / forgetting.

### Strategies in Order of Regularization

1. **Full fine-tune** — train every parameter. Default. Best on big datasets.

2. **Discriminative learning rate** — lower LR for earlier layers, higher LR for later (Howard & Ruder 2018):
   ```
   layer_0_LR   = base_LR × 0.95^N
   layer_1_LR   = base_LR × 0.95^(N-1)
   ...
   final_layer_LR = base_LR
   ```
   Idea: earlier layers learn general features (preserve them); later layers learn task-specific (let them change).

3. **Gradual unfreezing** — start with all of BERT frozen. Train only the head. Unfreeze the top layer, train. Unfreeze the next, train. Continue down. Slow but very stable on tiny datasets.

4. **Freeze backbone, train head** — minimal fine-tuning. Treats BERT as a feature extractor. Fast, low risk, often surprisingly good.

5. **LoRA / PEFT** — keep BERT frozen, add tiny adapter matrices. Modern alternative to discriminative LR with similar properties.

### Production Rule of Thumb

```
> 50K labeled examples  →  full fine-tune
  5K-50K               →  full fine-tune with discriminative LR
  500-5K               →  gradual unfreezing OR LoRA
  < 500                →  frozen BERT + classifier head, OR few-shot prompting with a generative LLM
```

---

## 7. Failure Modes

1. **Catastrophic forgetting** — train too many epochs or with too-high LR; BERT's pretrained representations get destroyed; model regresses to weaker performance. Fix: reduce LR, fewer epochs, discriminative LR.

2. **Train accuracy 99%, eval accuracy 60%** — overfitting on small data. Increase dropout, reduce epochs, or freeze layers.

3. **`[CLS]` embedding is bad for similarity** — vanilla BERT `[CLS]` is NOT trained for similarity. For RAG / retrieval, use mean pooling at minimum; ideally a Sentence-BERT variant fine-tuned with contrastive loss.

4. **Sequence length truncation losing answers** — QA where the answer is past position 512. Use sliding window with overlap or a long-context model variant (LongFormer, BigBird).

5. **Imbalanced classes** — BERT learns to predict the majority class. Use class weights in cross-entropy, or use focal loss, or oversample minority class.

6. **Batch effects** — small batch + BatchNorm-style layers (some BERT variants) → unstable training. Use batch ≥ 8, or switch to LayerNorm-only variant.

---

## 8. Interview Questions (5)

**Q1: Explain MLM. Why the 80/10/10 mask/random/keep split?**

For each masked position, predict the original token. 15% of tokens are selected; 80% replaced with [MASK], 10% with a random token, 10% kept as original. The split prevents the model from getting addicted to seeing [MASK] (mismatch at inference) — it must always consider context. RoBERTa shows MLM with these masking variants is THE driver of BERT's power.

**Q2: How do you fine-tune BERT for NER?**

Per token classification: each token's final hidden state → linear projection → softmax over BIO tag classes. Train with cross-entropy loss on all labeled tokens. Use sequence length 128-256 typically, batch 16-32, LR 3e-5, 3-5 epochs, linear warmup + decay. With small data (< 5K sentences), consider freezing the lower layers or using LoRA.

**Q3: Why did RoBERTa drop NSP?**

Analysis showed NSP doesn't significantly help downstream performance — the task is too easy (model learns to detect random sentences in a few thousand steps). RoBERTa removed NSP and used the saved compute / parameters for more MLM steps with bigger batches — significantly better downstream results.

**Q4: What's discriminative learning rate and when do you use it?**

Different LRs for different layers — usually exponentially decreasing toward earlier layers. Lower LR on early layers preserves the general features they encode; higher LR on later layers lets them adapt to the task. Especially useful for small datasets where you want to fine-tune without destroying pretrained knowledge.

**Q5: Why is `[CLS]` not great for sentence similarity?**

`[CLS]` is supervised during pretraining only for the NSP task (binary classification of sentence pairs). It doesn't learn to produce embeddings useful for similarity / retrieval. Sentence-BERT (Reimers & Gurevych 2019) explicitly fine-tunes BERT with contrastive / triplet loss on sentence pairs to produce good similarity embeddings. For RAG, always use a Sentence-BERT variant.

---

## 9. Further Reading

- BERT (Devlin et al. 2018) — arXiv:1810.04805 — the original
- RoBERTa (Liu et al. 2019) — arXiv:1907.11692 — robust pretraining recipe
- ALBERT (Lan et al. 2019) — arXiv:1909.11942 — parameter sharing + SOP
- DeBERTa (He et al. 2020) — arXiv:2006.03654 — disentangled attention
- Sentence-BERT (Reimers & Gurevych 2019) — arXiv:1908.10084
- ULMFiT (Howard & Ruder 2018) — arXiv:1801.06146 — discriminative LR origin
- HuggingFace BERT tutorial — huggingface.co/docs/transformers/model_doc/bert

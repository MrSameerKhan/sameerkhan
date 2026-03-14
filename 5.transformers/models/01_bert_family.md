# BERT Family (Encoder Models)

## Quick Reference
| Model | Key Innovation | Best For |
|-------|---------------|----------|
| BERT | MLM + NSP pretraining | Fine-tuning baseline |
| RoBERTa | Remove NSP, more data, dynamic masking | Better BERT in most tasks |
| DeBERTa | Disentangled attention, EMD | SOTA on GLUE/SuperGLUE tasks |
| DistilBERT | Knowledge distillation (40% smaller) | Inference-constrained production |
| ALBERT | Cross-layer param sharing, SOP | Low memory, still decent |
| ELECTRA | Replaced Token Detection | Efficient pretraining |

**When to use encoder models:** Classification, NER, extractive QA, embeddings, RE — tasks where full input context is available at inference time.

---

## Core Concepts

### BERT (Devlin et al., 2018)

**Architecture:**
```
BERT-base:  12 layers, 12 heads, d_model=768,  d_ff=3072  → 110M params
BERT-large: 24 layers, 16 heads, d_model=1024, d_ff=4096  → 340M params
Vocab: 30,522 WordPiece tokens
Max sequence: 512 tokens
```

**Pretraining objectives:**
```
1. Masked Language Modeling (MLM):
   - Randomly mask 15% of tokens
   - Of those: 80% → [MASK], 10% → random token, 10% → unchanged
   - Predict original token at masked positions
   - Forces bidirectional context understanding

   Input:  "The [MASK] sat on the mat"
   Target: predict "cat" at [MASK] position

   Why 80/10/10 split?
   - Pure masking: model sees [MASK] at fine-tune → train-test mismatch
   - Random replacement: forces model not to blindly trust input
   - Unchanged: forces model to have good representation for non-masked tokens

2. Next Sentence Prediction (NSP):
   - Input: [CLS] sentence_A [SEP] sentence_B [SEP]
   - 50% IsNext (consecutive sentences), 50% NotNext (random)
   - Binary classification on [CLS] token
   - Motivation: downstream tasks like QA need inter-sentence understanding
   - Later found: NSP hurts more than helps (RoBERTa ablation) — adds noise
```

**Input representation:**
```
Token: [CLS] The   cat  sat  [SEP] It   sat  [SEP]
Type:   0     0     0    0    0     1    1    1     ← segment embedding (A vs B)
Pos:    0     1     2    3    4     5    6    7     ← positional embedding

Final input = Token Embedding + Segment Embedding + Positional Embedding
```

**Fine-tuning patterns:**
```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# 1. Sequence Classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer("The movie was great!", return_tensors='pt',
                   padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
logits = outputs.logits  # [batch, num_labels]

# 2. Token Classification (NER)
from transformers import BertForTokenClassification
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9)  # IOB tags

# 3. Extractive QA (SQuAD style)
from transformers import BertForQuestionAnswering
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')
# Predicts start_logits and end_logits for answer span

# 4. Get sentence embeddings (mean pool last hidden state)
from transformers import BertModel
model = BertModel.from_pretrained('bert-base-uncased')

with torch.no_grad():
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state  # [batch, seq, 768]
    attention_mask = inputs['attention_mask']

    # Mean pooling (ignore padding tokens)
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_embeddings = torch.sum(last_hidden * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    sentence_embeddings = sum_embeddings / sum_mask  # [batch, 768]
```

---

### RoBERTa (Liu et al., 2019)

**Key changes from BERT:**
```
1. Remove NSP — NSP found to be harmful, train with full sentences only
2. Dynamic masking — generate new mask each epoch (BERT used static mask)
3. Larger batches — 256 → 2048 sequences per batch
4. More data — 160GB text vs 16GB in BERT (Books + CC-News + OpenWebText + Stories)
5. Longer training — 10× more steps (500K vs 1M but with 2K batch)
6. BPE tokenizer — 50K vocab (same as GPT-2) vs BERT's WordPiece 30K

Result: +2-3 points on GLUE, SQUAD vs BERT-large
Usage: drop-in replacement for BERT in most tasks
```

```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Usage is identical to BERT — just swap model name
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

# Note: RoBERTa doesn't use token_type_ids (no NSP, no segment A/B distinction)
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
# Don't pass token_type_ids — RoBERTa ignores them anyway
```

---

### DeBERTa (He et al., 2020) — Current SOTA for NLU

**Key innovations:**

**1. Disentangled Attention:**
```
BERT: single vector per token encodes both content and position
DeBERTa: separate content vector and position vector, combine during attention

Standard attention: Aᵢⱼ = qᵢᵀkⱼ  (content-to-content only)

Disentangled attention:
  Aᵢⱼ = Hᵢᵀ Hⱼ  +  Hᵢᵀ Pᵢⱼ  +  Pᵢⱼᵀ Hⱼ
         c2c         c2p          p2c
  Where H = content vector, P = relative position vector

c2p: "what does this content attend to in terms of position?"
p2c: "what position is most relevant to this content?"
→ More nuanced position-aware attention
```

**2. Enhanced Mask Decoder (EMD):**
```
During pretraining, use absolute position only in the final softmax layer
(the "decoding" step that predicts masked tokens), not in attention.
This forces attention to rely on relative positions, improving generalization.
```

**3. Virtual Adversarial Training (DeBERTa v3):**
```
Add gradient-based adversarial perturbations to embeddings during training
→ Regularization that significantly improves fine-tuning performance
```

```python
from transformers import DebertaV2ForSequenceClassification, DebertaV2Tokenizer

tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')
model = DebertaV2ForSequenceClassification.from_pretrained(
    'microsoft/deberta-v3-base', num_labels=3
)
# DeBERTa-v3-large is typically best for competition tasks
# Slower than RoBERTa but higher accuracy
```

---

### DistilBERT (Sanh et al., 2019)

**Knowledge Distillation:**
```
Teacher: BERT-base (110M params)
Student: DistilBERT (66M params, 40% smaller, 60% faster)

Training loss for student:
L = α · L_MLM + β · L_distill + γ · L_cos

L_distill = KL(softmax(T_teacher/T), softmax(T_student/T))
  where T = temperature (e.g., 4) — softer probability distributions
  → Student learns to match teacher's soft probability outputs

L_cos = cosine distance between hidden state representations

Architecture: remove 6 of 12 layers (keep every other), same d_model=768

Results:
  97% of BERT performance on GLUE
  40% fewer parameters
  60% faster inference
```

```python
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)
# Good for: mobile/edge deployment, real-time APIs, resource-constrained environments
```

---

### ELECTRA (Clark et al., 2020)

**Replaced Token Detection:**
```
Problem with MLM: only 15% of tokens contribute to loss → sample-inefficient

ELECTRA uses two models:
  Generator (small MLM): fills [MASK] positions with plausible tokens
  Discriminator (main model): for EVERY token, predict "original" or "replaced"

Example:
  Original: "The chef cooked the meal"
  Generator fills mask: "The chef ate the meal"  (plausible replacement)
  Discriminator: [orig] [orig] [replaced] [orig] [orig]

All tokens contribute to loss (not just 15%) → 4× more sample efficient
Same compute as BERT → significantly better performance

ELECTRA-base ≈ BERT-large with same compute
```

---

### Choosing BERT Variants

```python
# Decision guide
def choose_bert_variant(task, constraints):
    if constraints.get('latency') == 'strict':
        return 'distilbert-base-uncased'  # 60% faster

    if constraints.get('memory') == 'low':
        return 'albert-base-v2'  # cross-layer sharing

    if task in ['ner', 'classification', 'qa'] and constraints.get('accuracy') == 'high':
        return 'microsoft/deberta-v3-large'  # best accuracy

    if task in ['embeddings', 'semantic_similarity']:
        return 'sentence-transformers/all-mpnet-base-v2'  # SBERT

    # Default: best accuracy/speed tradeoff
    return 'roberta-base'  # remove NSP was the key fix
```

---

### Full Fine-Tuning Recipe

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np

# 1. Load model + tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# 2. Tokenize dataset
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512, padding='max_length')

dataset = Dataset.from_dict({'text': texts, 'label': labels})
tokenized = dataset.map(tokenize, batched=True)

# 3. Metrics
metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels, average='macro')

# 4. Training args
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.06,            # 6% of steps for warmup (standard)
    weight_decay=0.01,
    learning_rate=2e-5,           # 1e-5 to 5e-5 typical for BERT fine-tuning
    lr_scheduler_type="linear",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                    # mixed precision training
    report_to="wandb",
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)
trainer.train()
```

---

## Gotchas

**Max 512 tokens:** BERT and variants are pretrained with max 512 tokens. Truncation loses information. For long documents: sliding window with stride, hierarchical models, or switch to Longformer.

**Fine-tuning LR is critical:** Use 1e-5 to 5e-5. Higher LR → catastrophic forgetting of pretrained weights. Lower LR → underfitting. Always use linear decay with warmup (6% warmup ratio standard).

**CLS token ≠ sentence embedding:** The raw [CLS] token from BERT is NOT a good sentence embedding (it's trained for NSP, which was removed in RoBERTa). Use mean-pooling over last hidden states or use SBERT instead.

**token_type_ids:** RoBERTa doesn't use segment IDs. Passing `token_type_ids` to RoBERTa raises an error in strict mode or silently ignores them. BERT requires them.

**Subword tokenization affects NER:** `"London"` might tokenize to `["Lon", "##don"]`. Only label the first subword; ignore (or use -100 to mask loss) for continuation subwords.

```python
# NER: align labels with subword tokens
word_ids = encoding.word_ids()  # [None, 0, 0, 1, 2, 2, None]  (None=special, int=word index)
previous_word_idx = None
label_ids = []
for word_idx in word_ids:
    if word_idx is None:
        label_ids.append(-100)  # special tokens: ignore in loss
    elif word_idx != previous_word_idx:
        label_ids.append(labels[word_idx])  # first subword: use real label
    else:
        label_ids.append(-100)  # continuation subword: ignore
    previous_word_idx = word_idx
```

---

## Interview Q&A

**Q: Why is RoBERTa better than BERT despite using the same architecture?**
A: Three main changes: (1) Removing NSP — NSP was found to add noise since random-sentence pairs are easy to distinguish by topic, not discourse — model wastes capacity on this. (2) Dynamic masking — static masking means model sees same mask every epoch; dynamic generates new random masks per epoch, providing more diverse training signal. (3) More data and longer training. The architecture is identical; it's purely a training recipe improvement.

**Q: Explain BERT's 80/10/10 masking strategy. Why not just 100% [MASK]?**
A: If all masked tokens use [MASK], the model never sees [MASK] at fine-tune time (only real tokens) — this train-test mismatch hurts performance. The 10% random tokens force the model not to blindly trust input tokens; the 10% unchanged force good representations even for non-masked tokens. Practically: you see the token "cat" during fine-tuning and the model must still produce good representations for it, not just for [MASK] positions.

**Q: What is knowledge distillation and how does DistilBERT use it?**
A: Distillation trains a small "student" model to mimic a large "teacher" model. Instead of hard labels (one-hot), the student learns to match the teacher's soft output probabilities (at high temperature T). Soft probabilities carry richer information — e.g., for "The cat sat on the ___", the teacher might give: "mat" 60%, "floor" 20%, "couch" 15%. This full distribution is more informative than just the label "mat". DistilBERT additionally matches hidden state representations via cosine loss.

**Q: How do you handle long documents with BERT (>512 tokens)?**
A: Several strategies: (1) Truncate — works when answer/label is in first 512 tokens. (2) Sliding window with stride — split document into overlapping chunks, run model on each, aggregate (take max or mean of logits). (3) Hierarchical — encode sentences independently, then encode sentence representations. (4) Switch models — Longformer (4096 tokens, sliding window attention), BigBird (4096), or modern LLMs with large context windows.

---

## Connections
- **Attention Mechanism (fundamentals/01):** BERT stacks bidirectional self-attention
- **Transformer Architecture (fundamentals/02):** BERT = encoder-only transformer
- **Word Embeddings (NLP/embeddings/01):** BERT replaces static embeddings — contextual
- **NER and Tagging (NLP/applications/02):** BertForTokenClassification
- **Efficient Transformers (models/04):** DistilBERT, Longformer for long documents
- **RAG (LLMs):** BERT-family models often used as retrievers / bi-encoders

## Key Takeaway
BERT = bidirectional transformer encoder pretrained on MLM. RoBERTa fixed BERT's training recipe (remove NSP, dynamic masking, more data) — use RoBERTa as default. DeBERTa adds disentangled attention for best accuracy. DistilBERT when speed matters. Fine-tuning recipe: LR 2e-5, linear decay, 6% warmup, 3 epochs — this works for 90% of downstream tasks.

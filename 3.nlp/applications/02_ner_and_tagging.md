# NER and Sequence Tagging

## Quick Reference
| Task | Description | Output Format | Key Model |
|------|-------------|---------------|-----------|
| NER | Named entity recognition | IOB/BIOES tags | BERT-NER |
| POS Tagging | Part-of-speech labels | Per-token tags | BERT or spaCy |
| Chunking | Phrase boundary detection | IOB chunks | CRF/BERT |
| SRL | Semantic role labeling | Argument labels | AllenNLP |
| Entity Linking | Map span → KB entry | Entity ID | BLINK, REL |

**Core difference from text classification:** Token-level output, not document-level. Each token gets a label.

---

## Core Concepts

### IOB/BIOES Tagging Schemes

**IOB2 (most common):**
```
B-TYPE  = Beginning of entity of TYPE
I-TYPE  = Inside (continuation) of entity
O       = Outside (not an entity)

"Apple   was   founded   in   Cupertino   by   Steve   Jobs"
 B-ORG   O      O        O    B-LOC       O    B-PER   I-PER
```

**BIOES (more expressive):**
```
B = Beginning (multi-token entity)
I = Inside
O = Outside
E = End of entity
S = Single-token entity

"Steve   Jobs   visited   Paris"
 B-PER  E-PER    O       S-LOC
```

### Entity Types
```
Standard CoNLL-2003:  PER, ORG, LOC, MISC
OntoNotes 5.0:       18 types (PERSON, ORG, GPE, DATE, MONEY, PERCENT, ...)
Custom domains:      DRUG, DISEASE, GENE (biomedical)
                     INVOICE_NUM, VENDOR, DATE, AMOUNT (document AI)
```

### Span Extraction vs Token Tagging
```
Token tagging:  Assign label to each token → reconstruct spans (most common)
Span extraction: Predict (start, end, type) directly → avoids IOB alignment issues
                 Used in SQuAD-style reading comprehension, modern NER models
```

---

## Approaches

### 1. Rule-Based (Baseline)
```python
import re
import spacy

# spaCy matcher for custom entities
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Pattern: invoice number like "INV-12345" or "Invoice #12345"
pattern = [
    {"LOWER": {"IN": ["invoice", "inv"]}},
    {"IS_PUNCT": True, "OP": "?"},
    {"LIKE_NUM": True}
]
matcher.add("INVOICE_NUM", [pattern])

doc = nlp("Invoice #12345 from Acme Corp")
matches = matcher(doc)
for match_id, start, end in matches:
    print(doc[start:end].text)
```

### 2. CRF (Classical, Strong Baseline)
```python
import sklearn_crfsuite
from sklearn_crfsuite import metrics

def word2features(sent, i):
    word = sent[i][0]
    return {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],          # suffix
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.has_hyphen': '-' in word,
        'prefix-1': word[:1],
        'prefix-2': word[:2],
        # Context features
        'next_word': sent[i+1][0].lower() if i < len(sent)-1 else '<END>',
        'prev_word': sent[i-1][0].lower() if i > 0 else '<START>',
    }

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,           # L1 regularization
    c2=0.1,           # L2 regularization
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# Why CRF: models label transitions (B-PER → I-PER valid; B-PER → I-ORG invalid)
# The Viterbi algorithm finds globally optimal tag sequence
```

### 3. BERT-NER (Current Standard)
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
import torch

# Label scheme
label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",  # cased matters for NER (capitalization is a strong signal)
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# CRITICAL: Tokenization alignment issue
# "John Smith" → ["John", "Smith"] → ["John", "Sm", "##ith"]
# Labels:          B-PER  I-PER       B-PER   I-PER  -100 (ignored)

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,   # input is already tokenized into words
        max_length=512
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)         # [CLS], [SEP], [PAD] → ignored
            elif word_id != prev_word_id:
                aligned_labels.append(labels[word_id])  # first subword → real label
            else:
                aligned_labels.append(-100)         # subsequent subwords → ignored
            prev_word_id = word_id
        all_labels.append(aligned_labels)
    tokenized["labels"] = all_labels
    return tokenized
```

### Post-processing: Reconstruct Spans from Token Labels
```python
def extract_entities(tokens, predictions):
    """Convert IOB token labels back to entity spans."""
    entities = []
    current_entity = None

    for token, label in zip(tokens, predictions):
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'type': label[2:],
                'text': token,
                'tokens': [token]
            }
        elif label.startswith('I-') and current_entity:
            entity_type = label[2:]
            if entity_type == current_entity['type']:
                current_entity['tokens'].append(token)
                current_entity['text'] += ' ' + token
            else:
                # I-ORG after B-PER: treat as new entity
                entities.append(current_entity)
                current_entity = {'type': entity_type, 'text': token, 'tokens': [token]}
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None

    if current_entity:
        entities.append(current_entity)
    return entities
```

### seqeval: Standard NER Evaluation
```python
from seqeval.metrics import classification_report, f1_score

# seqeval expects IOB-formatted sequences
y_true = [['B-PER', 'I-PER', 'O', 'B-ORG'], ['O', 'B-LOC', 'O']]
y_pred = [['B-PER', 'I-PER', 'O', 'B-ORG'], ['O', 'B-LOC', 'O']]

print(classification_report(y_true, y_pred))
# Entity-level F1 (not token-level): entire span must match exactly
# PER: precision=1.00, recall=1.00, f1=1.00
```

---

## Document AI / Information Extraction NER

For document understanding (invoices, receipts, contracts), layout matters:

### LayoutLM / LayoutLMv3
```python
# LayoutLM: BERT + 2D position embeddings (x0, y0, x1, y1 from OCR bounding boxes)
# LayoutLMv3: also adds visual token embeddings from document images

from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base",
                                                  apply_ocr=True)  # auto-runs OCR
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list)
)

# Input: image + (optionally) pre-run OCR words and boxes
encoding = processor(image, words=words, boxes=boxes, return_tensors="pt")
outputs = model(**encoding)
```

### Donut (OCR-free document understanding)
```python
# Donut: end-to-end, no OCR step, generates structured output as JSON string
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")

pixel_values = processor(image, return_tensors="pt").pixel_values
output = model.generate(pixel_values, ...)
sequence = processor.decode(output[0], skip_special_tokens=True)
# → {"menu": [{"nm": "COFFEE", "cnt": "1", "price": "4.50"}], ...}
```

---

## Metrics

### Token-level vs Entity-level
```
Token-level F1: each token judged independently (lenient — partial matches count)
Entity-level F1: entire span must match type AND boundaries (seqeval default)

Example: True = "New York City" (B-LOC I-LOC I-LOC)
         Pred = "New York" (B-LOC I-LOC O)
Token-level: 2/3 correct
Entity-level: 0 correct (span doesn't fully match)

Always report entity-level F1 for NER benchmarks.
```

### Partial Match Scoring (for lenient evaluation)
```python
# For document extraction where partial matches have value:
# Use span overlap ratio instead of exact match
def span_overlap_f1(true_spans, pred_spans):
    """Character-level overlap F1 for lenient NER evaluation."""
    tp = sum(len(set(range(*t)) & set(range(*p)))
             for t in true_spans for p in pred_spans
             if t[2] == p[2])  # same type
    precision = tp / sum(end-start for start, end, _ in pred_spans + [1e-8])
    recall = tp / sum(end-start for start, end, _ in true_spans + [1e-8])
    return 2*precision*recall/(precision+recall+1e-8)
```

---

## When to Use What

| Scenario | Approach |
|----------|----------|
| Standard NER, enough data (>1K examples) | BERT-NER fine-tune |
| Domain-specific, limited data | Few-shot with spaCy + BERT or NuNER |
| Document layout matters (invoices, forms) | LayoutLMv3 |
| OCR-free end-to-end document extraction | Donut or Pix2Struct |
| Production, low latency | Distilled BERT-NER or spaCy transformer |
| Quick rules-based extraction | spaCy Matcher / regex |
| Label transitions matter, classical features | CRF on top of BiLSTM or BERT |

---

## Gotchas

**Cased vs uncased for NER:** Always use `bert-base-cased`. Capitalization is a critical NER signal — "apple" (fruit) vs "Apple" (company). Uncased destroys this.

**Subword alignment is not optional:** If you assign the word-level label to all subword tokens, your model trains incorrectly. Always use `-100` for non-first subwords.

**Sentence boundaries matter:** BERT processes sequences, not documents. Split at sentence boundaries before tokenization. A 512-token window arbitrarily split mid-sentence confuses the model.

**Nested entities:** Standard IOB can't represent nested entities ("Bank of New York" ORG contains "New York" LOC). Options: (1) span-based models, (2) multi-layer tagging, (3) just ignore nesting for most practical use cases.

**O class dominance:** In typical text, 80-90% of tokens are O. Macro F1 on O class inflates scores. Always report entity-level F1 from seqeval (which ignores O).

**Entity boundary errors are the most common failure:** Model predicts the right type but wrong start/end. Check if your longest entities have good recall — often the last few tokens are missed.

---

## Debugging Guide

**Low recall on long entity spans:**
- Increase max_length (default 128 is often too short for NER)
- Check if spans are truncated during tokenization
- Use stride/overlap for long documents

**Model predicts I-TYPE without B-TYPE:**
- IOB constraint violation — model doesn't understand tag structure
- Solution: add CRF head on top of BERT token logits
- Or post-process: convert invalid I-TYPE at sequence start to B-TYPE

**Poor performance on rare entity types:**
- Per-type F1 analysis first: `seqeval.metrics.classification_report`
- Augment with entity replacement (swap all PER entities with random names from gazetteer)
- Increase class weight for rare types in loss

**Training loss goes to 0 but eval F1 stays low:**
- Overfitting to training labels — add more dropout (0.2-0.3)
- Check for label noise — NER annotation is expensive and often inconsistent
- Use span-level training with stronger data augmentation

---

## Interview Q&A

**Q: Why is BERT-NER better than BiLSTM-CRF?**
A: BERT provides deep bidirectional contextual representations from pretraining on massive text. BiLSTM-CRF can only see what its fixed-size embeddings and training data provide. The CRF head is still useful on top of BERT to enforce label transition constraints, but BERT's contextual representations do most of the heavy lifting. In practice, BERT-NER without CRF often beats BiLSTM-CRF with CRF.

**Q: What is the subword alignment problem in BERT NER and how do you solve it?**
A: BERT uses WordPiece tokenization which splits words into subwords (e.g., "Washington" → ["Wash", "##ington"]). If a word has label B-PER, we can only assign one label per token. The solution: assign the word's label to the first subword, assign -100 (ignored in loss) to subsequent subwords. During inference, take the prediction for the first subword only.

**Q: How do you evaluate NER? What's the difference between token-level and entity-level F1?**
A: Token-level F1 computes metrics for each token independently — a partial span match partially counts. Entity-level F1 (from seqeval) requires both the span boundaries and entity type to match exactly for a true positive. Entity-level is stricter and the standard benchmark metric. In practice, token-level F1 can be 10-15 points higher than entity-level F1 for the same model.

**Q: You're building invoice field extraction. What approach would you take?**
A: This is a document AI problem where layout matters. (1) If we have OCR output (words + bounding boxes), use LayoutLMv3 which incorporates 2D positional embeddings alongside text. (2) If we want to avoid OCR errors and go end-to-end, use Donut which processes the raw document image and generates structured JSON. (3) For quick prototyping: regex + spatial rules on OCR output as baseline. Train the model on field-annotated invoices and evaluate with entity-level F1 per field type.

**Q: What is a CRF and why was it used in NER before BERT?**
A: CRF (Conditional Random Field) is a discriminative sequence model that models the conditional probability P(y|x) of a label sequence given observations. Unlike classifying each token independently, CRF uses the Viterbi algorithm to find the globally optimal tag sequence, incorporating label transition probabilities. This means it learns that B-PER → I-PER is valid but B-PER → I-ORG is not. Pre-BERT, BiLSTM-CRF was state-of-the-art because the BiLSTM captures context and the CRF enforces valid IOB transitions.

---

## Connections
- **Text Preprocessing (fundamentals/01):** Tokenization choices critically affect subword alignment in NER
- **Text Classification (applications/01):** Same BERT backbone, different output head (token vs sequence level)
- **Information Extraction (applications/03):** NER is the first step in most IE pipelines (entity → relation)
- **Transformers (transformers/01):** BERT encoder powers modern NER; positional embeddings matter
- **LayoutLM:** Extends BERT with 2D position embeddings for document AI

## Key Takeaway
NER is token classification. The pipeline: tokenize (carefully align subword labels with -100), fine-tune BERT with token classification head, evaluate with seqeval entity-level F1. The hard parts are subword alignment, handling long documents without truncating spans, and the domain gap (use cased models, domain-adapted tokenizers when possible). For document AI with layout, LayoutLMv3 is the current standard.

# Session 11 — Load Pretrained from HuggingFace

## Table of Contents
- [Objective](#objective)
- [Architecture & Math](#architecture--math)
- [Dataset Note](#dataset-note)
- [How to Run](#how-to-run)
- [Expected Output](#expected-output)
- [✅ Actual Run Results](#-actual-run-results)
- [Key Insights](#key-insights)
- [Next Steps](#next-steps)

---

## Objective

Load real pretrained BERT-base-uncased and GPT-2 from HuggingFace. Inspect every named parameter and its shape. Map those shapes to our hand-built MHA (Session 3), encoder block (Session 5), and mini transformer (Session 6). Run a forward pass on Acme Financial text and extract attention weights.

**Goals:**
- [ ] Load BERT-base-uncased and GPT-2 from HuggingFace without errors
- [ ] Print all named parameters with shapes — confirm total param count (~110M BERT, ~117M GPT-2)
- [ ] Map BERT HuggingFace layer names → our Session 3 MHA + Session 5 encoder block variable names
- [ ] Run forward pass on Acme Financial corpus text, extract attention weights (layer 0, head 0)
- [ ] Confirm numerically: head_dim = H / A = 768 / 12 = 64

---

## Architecture & Math

### BERT-base-uncased Config

| Hyperparameter | Value |
|---|---|
| Layers (L) | 12 |
| Hidden size (H) | 768 |
| Attention heads (A) | 12 |
| Head dim (H/A) | 64 |
| FFN intermediate | 3072 (= 4 × 768) |
| Vocab size | 30,522 |
| Max position | 512 |
| Total params | ~110M |

### GPT-2 (small) Config

| Hyperparameter | Value |
|---|---|
| Layers (L) | 12 |
| Hidden size (H) | 768 |
| Attention heads (A) | 12 |
| Head dim (H/A) | 64 |
| FFN intermediate | 3072 |
| Vocab size | 50,257 |
| Max position | 1,024 |
| Total params | ~117M |

### Attention Math (same as Session 3)

```
Q = X · W_Q    [seq_len, 768] · [768, 768] → [seq_len, 768]
K = X · W_K    [seq_len, 768] · [768, 768] → [seq_len, 768]
V = X · W_V    [seq_len, 768] · [768, 768] → [seq_len, 768]

Split into 12 heads → each head: [seq_len, 64]

Attention(Q, K, V) = softmax(Q · K^T / √64) · V
```

### Shape Mapping — BERT HuggingFace → Our Implementation

| Component | HuggingFace Parameter Name | Our Name (Session 3/5) | Shape |
|---|---|---|---|
| Q projection weight | `bert.encoder.layer.0.attention.self.query.weight` | `W_Q` | [768, 768] |
| Q projection bias | `bert.encoder.layer.0.attention.self.query.bias` | `b_Q` | [768] |
| K projection weight | `bert.encoder.layer.0.attention.self.key.weight` | `W_K` | [768, 768] |
| K projection bias | `bert.encoder.layer.0.attention.self.key.bias` | `b_K` | [768] |
| V projection weight | `bert.encoder.layer.0.attention.self.value.weight` | `W_V` | [768, 768] |
| V projection bias | `bert.encoder.layer.0.attention.self.value.bias` | `b_V` | [768] |
| Output projection weight | `bert.encoder.layer.0.attention.output.dense.weight` | `W_O` | [768, 768] |
| Output projection bias | `bert.encoder.layer.0.attention.output.dense.bias` | `b_O` | [768] |
| LayerNorm 1 weight | `bert.encoder.layer.0.attention.output.LayerNorm.weight` | `ln1_gamma` | [768] |
| LayerNorm 1 bias | `bert.encoder.layer.0.attention.output.LayerNorm.bias` | `ln1_beta` | [768] |
| FFN layer 1 weight | `bert.encoder.layer.0.intermediate.dense.weight` | `W_1` | [3072, 768] |
| FFN layer 1 bias | `bert.encoder.layer.0.intermediate.dense.bias` | `b_1` | [3072] |
| FFN layer 2 weight | `bert.encoder.layer.0.output.dense.weight` | `W_2` | [768, 3072] |
| FFN layer 2 bias | `bert.encoder.layer.0.output.dense.bias` | `b_2` | [768] |
| LayerNorm 2 weight | `bert.encoder.layer.0.output.LayerNorm.weight` | `ln2_gamma` | [768] |
| LayerNorm 2 bias | `bert.encoder.layer.0.output.LayerNorm.bias` | `ln2_beta` | [768] |
| Token embedding | `bert.embeddings.word_embeddings.weight` | `E` | [30522, 768] |
| Position embedding | `bert.embeddings.position_embeddings.weight` | `PE` | [512, 768] |

---

## Dataset Note

Uses the **Acme Financial Services** synthetic corpus from `code_practice/shared_dataset.py`.

```python
from shared_dataset import get_text_corpus
texts = get_text_corpus()   # list of strings — Acme loan/document sentences
```

A few Acme sentences are tokenised by BERT's own tokenizer and passed as the forward-pass input. This keeps the session self-contained and consistent with every other session.

---

## How to Run

```bash
# Step 1 — inspect all named params + shapes, print total param count
python model.py

# Step 2 — run forward pass on Acme text, extract + save attention weights
python train.py

# Step 3 — load saved attention weights, show top attending token pairs
python predict.py --text "Acme Financial Services loan approval document"
```

> **Note:** `train.py` here is not gradient-based training — it runs the pretrained model in eval mode to extract attention weights and saves them to `checkpoints/attention_layer0.pt`. No fine-tuning happens.

---

## Expected Output

**model.py**
```
=== BERT-base-uncased ===
Total params: 109,482,240

Layer 0 — Attention:
  query.weight        : torch.Size([768, 768])
  query.bias          : torch.Size([768])
  key.weight          : torch.Size([768, 768])
  key.bias            : torch.Size([768])
  value.weight        : torch.Size([768, 768])
  value.bias          : torch.Size([768])
  output.dense.weight : torch.Size([768, 768])
  output.dense.bias   : torch.Size([768])

Layer 0 — FFN:
  intermediate.dense.weight : torch.Size([3072, 768])
  intermediate.dense.bias   : torch.Size([3072])
  output.dense.weight       : torch.Size([768, 3072])
  output.dense.bias         : torch.Size([768])

Confirmed head_dim = 768 / 12 = 64 ✓

=== GPT-2 (small) ===
Total params: 124,439,808
...
```

**train.py**
```
Input text : "Acme Financial Services approved the mortgage document for client ID 4821."
Tokens     : ['[CLS]', 'acme', 'financial', 'services', 'approved', ...]
Token IDs  : [101, 15720, 3361, 2578, 4844, ...]
Input shape: torch.Size([1, 14])

Running BERT forward pass...
Last hidden state shape: torch.Size([1, 14, 768])
Pooler output shape    : torch.Size([1, 768])

Attention weights (layer 0, head 0): torch.Size([1, 12, 14, 14])
Saved → checkpoints/attention_layer0.pt
```

**predict.py**
```
Loaded attention weights from checkpoints/attention_layer0.pt

Top-5 attention pairs (layer 0, head 0):
  [CLS]       → approved      : 0.312
  mortgage    → document      : 0.287
  approved    → mortgage      : 0.241
  financial   → services      : 0.198
  client      → ID            : 0.175
```

---

## ✅ Actual Run Results

> Fill this section after running the scripts.

**model.py**
```
[paste output here]
```

**train.py**
```
[paste output here]
```

**predict.py**
```
[paste output here]
```

---

## Key Insights

> Fill this section after running.

- [ ] BERT Q/K/V weight shape [768, 768] matches our `W_Q`, `W_K`, `W_V` from Session 3 — confirms our scratch impl had the right dimensions
- [ ] FFN intermediate [3072, 768] = 4× expansion matches our Session 5 encoder block
- [ ] Head dim 64 = 768/12 — exactly what we hardcoded in Session 3
- [ ] Total param count visible: embeddings alone = 30522 × 768 = 23.4M — largest single block

---

## Next Steps

→ **Phase 3, Session 1** — First LLM call (3 providers): same prompt → OpenAI, Anthropic, Ollama (local)
`code_practice/03_prompting/01_first_call/`

With BERT/GPT-2 shapes fully mapped to our hand-built implementation, we now understand exactly what a production pretrained model looks like internally. Phase 3 shifts from building to using — calling LLMs via API and shaping their output.

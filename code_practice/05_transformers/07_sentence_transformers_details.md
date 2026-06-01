# Session 7 — Sentence Transformers: Semantic Document Search
Status: `🔧 Code-built`

Theory: [../../../4.nlp/02_embeddings/02_sentence_embeddings.md](../../../4.nlp/02_embeddings/02_sentence_embeddings.md)

---

## Use Case

Semantic document search: "Find all mortgage contracts similar to this one" — keyword search finds documents with the same words, semantic search finds documents with the same *meaning*. Bridge to RAG: this is the retrieval stage of every RAG pipeline.

---

## Key Concept — Bi-Encoder + Mean Pooling

BERT produces a token embedding for every position, but we need a single vector per sentence (for similarity comparison). Mean pooling averages all non-padding token embeddings:

```
BERT token outputs: shape (batch, 128, 768)
Attention mask:     shape (batch, 128)     — 0 on padding

Mean pool: sum(token_embs × mask_expanded) / sum(mask)
         = shape (batch, 768)              — one vector per sentence

L2 normalize → unit sphere → cosine sim = dot product
```

**Why not use [CLS]?** BERT's [CLS] was trained for classification, not similarity. Mean pooling over non-padding tokens gives better sentence representations. `all-MiniLM-L6-v2` was fine-tuned specifically with this pooling.

---

## Training — MultipleNegativesRankingLoss

Given a batch of (anchor, positive) pairs, treat every other positive in the batch as a negative (in-batch negatives). No need for explicit hard-negative mining.

```
Batch of B pairs:
  anchor_emb:   shape (B, 384)
  positive_emb: shape (B, 384)

scores = anchor_emb @ positive_emb.T    shape (B, B)
  scores[i][i] = similarity to own positive (should be HIGH)
  scores[i][j≠i] = similarity to other positives (should be LOW)

loss = cross_entropy(scores × 20, arange(B))
       ↑ temperature scaling (20 = 1/0.05) sharpens the distribution
```

Each (anchor, positive) pair provides B-1 free negatives. A batch of 32 gives 31 negatives per anchor — very efficient.

---

## Tensor Shapes

```
Anchor (SNLI premise):
  input_ids:  shape (128,)
  attn_mask:  shape (128,)
  → mean_pool → L2-norm → emb: shape (384,)

Positive (SNLI hypothesis — entailment):
  same shapes → emb: shape (384,)

Cosine similarity matrix: shape (B, B) after matmul of L2-normalised embeddings
Search: query_emb (1, 384) @ corpus_emb.T (384, N) → scores (N,) → topk indices
```

---

## Input Data — SNLI (Entailment pairs)

| Label | Meaning | Used as |
|-------|---------|---------|
| 0 = entailment | Hypothesis follows from premise | (anchor, positive) ✓ |
| 1 = neutral | Possibly related | Skip |
| 2 = contradiction | Opposite meaning | Skip (hard negative — advanced use) |

```
anchor:   "A man is playing guitar on a street corner."
positive: "Someone is performing music outdoors."   ← same meaning, different words
```

- ~550k entailment pairs total; script uses 4,000
- `load_dataset("snli")`

---

## Retrieval vs RAG Connection

```
This session:
  Query → encode → (1, 384)
  Corpus (8 docs) → encode → (8, 384)
  Cosine sim → top-2 → return docs

RAG pipeline (next phase):
  Same bi-encoder retrieval
  + pass retrieved chunks as context to LLM
  + LLM generates grounded answer
```

`07_sentence_transformers.py` is the retrieval component of session `code_practice/07_rag/01_basic_rag.py`.

---

## Expected Output

```
Device: mps

Epoch 1/2 | loss: 1.8420
Epoch 2/2 | loss: 1.4103

Saved to models/05_transformers/sentence_transformers

── Semantic Search ──

  Query: What is the LTV limit for first-time buyers?
    [0.891] First-time buyers are eligible for a 95% LTV mortgage under the Help to Buy scheme.
    [0.743] The maximum loan-to-value ratio is 90% for standard residential mortgages.

  Query: How much do I pay if I repay the loan early?
    [0.876] Early repayment charges of 2% apply within the initial fixed-rate period.
    [0.621] The fixed-rate period lasts five years before reverting to the variable rate.

  Query: What income documents do I need to provide?
    [0.882] All applications require proof of income for the past three months.
    [0.594] Documents must be submitted within 30 days of the initial application date.
```

---

## How to Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE python 07_sentence_transformers.py
```

First run: downloads `all-MiniLM-L6-v2` (~90 MB) + SNLI (~100 MB).
MPS training: ~10–15 min for 2 epochs on 4k examples.

**Production scale-up:** replace the numpy dot-product search with FAISS:
```python
import faiss
index = faiss.IndexFlatIP(384)      # inner product on L2-normalised vectors = cosine sim
index.add(corpus_emb.cpu().numpy())
scores, indices = index.search(query_emb.cpu().numpy(), k=5)
```
FAISS handles millions of vectors with sub-millisecond search.

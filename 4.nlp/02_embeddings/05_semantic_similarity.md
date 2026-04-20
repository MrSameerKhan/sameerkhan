# Semantic Similarity & Sentence Embeddings

> **Core use:** RAG retrieval, duplicate detection, question matching, clustering, cross-encoder reranking.

---

## The Task

```
Given two texts, measure how semantically similar they are.

"The invoice is overdue"  vs  "The bill has not been paid"  → high similarity
"The invoice is overdue"  vs  "I like pizza"               → low similarity

Applications:
  RAG:         query embedding ↔ document chunk embeddings → find relevant chunks
  Deduplication: find near-duplicate documents in corpus
  Paraphrase:  detect if two sentences express the same meaning
  Search:      semantic search vs keyword search
  Clustering:  group similar documents without labels
```

---

## Similarity Metrics

### Cosine Similarity (Most Common)

```
cos(u, v) = (u · v) / (||u|| × ||v||)

Range: [-1, 1]  (for L2-normalized vectors: always in [0, 1])
  cos = 1.0:  identical direction (perfect similarity)
  cos = 0.0:  orthogonal (unrelated)
  cos = -1.0: opposite direction (antonyms, if model supports it)

When to use: comparing embeddings of different lengths
             invariant to vector magnitude (only direction matters)
             standard for sentence embeddings

Example:
  u = [0.6, 0.8, 0.0]  ("invoice overdue")
  v = [0.5, 0.7, 0.5]  ("bill not paid")

  u·v = 0.6×0.5 + 0.8×0.7 + 0.0×0.5 = 0.30 + 0.56 + 0.00 = 0.86
  ||u|| = √(0.36+0.64+0.00) = 1.0
  ||v|| = √(0.25+0.49+0.25) = √0.99 ≈ 0.995

  cos(u,v) = 0.86 / (1.0 × 0.995) = 0.864  ← high similarity ✓
```

### Dot Product

```
u · v = Σᵢ uᵢvᵢ

Proportional to cosine × magnitude product
  → sensitive to vector norm (longer = higher score)
  → used when magnitude encodes confidence/relevance
  → faster than cosine (no normalization needed)
  → used in FAISS for fast ANN search

Note: if vectors are L2-normalized, dot product = cosine similarity
```

### Euclidean Distance

```
||u - v||₂ = √(Σᵢ (uᵢ - vᵢ)²)

Range: [0, ∞)
  0 = identical
  higher = more different

Less preferred for high-dim embeddings — suffers from curse of dimensionality
Use when: comparing short, fixed-length feature vectors (tabular ML)
          training metric learning models (triplet loss)
```

---

## Bi-Encoder (Fast, for Retrieval)

Encode query and document independently → compare embeddings.

```
query ──────► encoder ──► q_emb (768-dim)
                                    ↓
                           cosine_similarity → score
                                    ↑
doc   ──────► encoder ──► d_emb (768-dim)

Key properties:
  Documents pre-computed offline (index all docs once)
  At query time: only encode query (fast)
  Search: ANN lookup in embedding space → sub-millisecond for 1M docs

Trade-off:
  ✓ Fast (O(log N) ANN search)
  ✗ Lower accuracy than cross-encoder (query-doc interaction only via cosine)
```

### SBERT (Sentence-BERT)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# 22M params, 384-dim embeddings, fast

sentences = [
    "The invoice is overdue",
    "The bill has not been paid",
    "I enjoy eating pizza",
    "The payment is past due date",
]

embeddings = model.encode(sentences, normalize_embeddings=True)
# embeddings.shape: (4, 384)

# Pairwise cosine similarity matrix
similarity_matrix = embeddings @ embeddings.T
print(np.round(similarity_matrix, 3))
# [[1.000, 0.821, 0.102, 0.879],   ← "invoice overdue" vs all
#  [0.821, 1.000, 0.098, 0.912],   ← "bill not paid"
#  [0.102, 0.098, 1.000, 0.089],   ← "pizza" — unrelated to others
#  [0.879, 0.912, 0.089, 1.000]]   ← "payment past due"

# Query-document similarity (retrieval)
query_emb = model.encode(["Who has not paid?"], normalize_embeddings=True)
doc_embs  = embeddings[1:]  # exclude query itself

scores = (query_emb @ doc_embs.T).flatten()
ranking = np.argsort(scores)[::-1]
for i in ranking:
    print(f"{scores[i]:.3f}: {sentences[i+1]}")
# 0.891: The payment is past due date
# 0.834: The bill has not been paid
# 0.078: I enjoy eating pizza
```

### How SBERT Is Trained (Siamese Network + Triplet Loss)

```
Architecture:
  Two BERT encoders with SHARED weights (siamese)
  Each sentence goes through the same BERT model
  Use [CLS] or mean-pooled token embeddings as sentence embedding

Training data: sentence pairs with similarity labels (NLI, STS-B)

Objective 1 — Regression (STS tasks):
  L = MSE(cosine(emb₁, emb₂), gold_similarity)

Objective 2 — Triplet loss:
  Anchor: "The invoice is overdue"
  Positive: "The bill has not been paid" (similar)
  Negative: "I like pizza" (dissimilar)

  L = max(0, ||f(a)-f(p)||₂ - ||f(a)-f(n)||₂ + margin)
  margin = 1.0 (typical)

  Minimized when: d(anchor, positive) < d(anchor, negative) - 1
  → embeddings of similar pairs closer than dissimilar pairs by margin
```

---

## Cross-Encoder (Accurate, for Reranking)

Feed query and document TOGETHER through the model → single relevance score.

```
[CLS] query [SEP] document [SEP]
       ──────────────────────────► BERT ──► Linear ──► score

Key properties:
  Full attention over query + document → captures exact match, negation, etc.
  Cannot pre-compute → must run model for every (query, doc) pair at query time
  Much slower: 100 candidates × 50ms = 5 seconds (vs sub-millisecond for bi-encoder)
  Much more accurate: model sees both texts simultaneously

Trade-off:
  ✗ Slow (O(N) forward passes)
  ✓ Higher accuracy
```

```python
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query     = "invoice payment overdue"
documents = [
    "The bill has not been paid",
    "The payment is past due date",
    "I enjoy eating pizza",
    "Invoice INV-2024-0432 remains unpaid",
]

# Score all (query, doc) pairs
pairs  = [(query, doc) for doc in documents]
scores = cross_encoder.predict(pairs)

# Sort by score
ranked = sorted(zip(scores, documents), reverse=True)
for score, doc in ranked:
    print(f"{score:.3f}: {doc}")
# 9.21:  Invoice INV-2024-0432 remains unpaid
# 8.74:  The payment is past due date
# 7.82:  The bill has not been paid
# -3.41: I enjoy eating pizza
```

---

## Bi-Encoder + Cross-Encoder Pipeline (RAG Standard)

```
Stage 1 — Retrieval (bi-encoder, fast):
  Query → encode → ANN search → top-100 candidates

Stage 2 — Reranking (cross-encoder, accurate):
  (Query, candidate_1), ..., (Query, candidate_100) → cross-encoder → scores
  Re-sort by cross-encoder score → top-10

Stage 3 — Generation:
  Top-10 chunks → LLM context → generate answer

Why two stages?
  Cross-encoder on 1M docs: 1M × 50ms = 14 hours per query (impossible)
  Cross-encoder on 100 docs: 100 × 5ms = 500ms (acceptable)
  Bi-encoder narrows to 100; cross-encoder picks the best 10
```

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

bi_encoder    = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Offline: build FAISS index ──────────────────────────────────────────
documents = load_documents()  # list of text chunks
doc_embs  = bi_encoder.encode(documents, batch_size=64, normalize_embeddings=True)
doc_embs  = doc_embs.astype(np.float32)

index = faiss.IndexFlatIP(doc_embs.shape[1])  # inner product = cosine (normalized)
index.add(doc_embs)
faiss.write_index(index, "doc_index.faiss")

# ── Online: retrieve + rerank ────────────────────────────────────────────
def retrieve_and_rerank(query: str, top_k=10, retrieval_k=100):
    # Stage 1: bi-encoder retrieval
    q_emb = bi_encoder.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, retrieval_k)   # top-100
    candidates = [documents[i] for i in indices[0]]

    # Stage 2: cross-encoder reranking
    pairs        = [(query, doc) for doc in candidates]
    rerank_scores = cross_encoder.predict(pairs)

    # Sort by rerank score → top-k
    ranked = sorted(zip(rerank_scores, candidates), reverse=True)[:top_k]
    return [doc for _, doc in ranked]
```

---

## Sentence Embedding Models Comparison

| Model | Params | Dim | Speed | MTEB Score | Best for |
|-------|--------|-----|-------|-----------|----------|
| all-MiniLM-L6-v2 | 22M | 384 | Very fast | 56.3 | Quick retrieval, low latency |
| all-mpnet-base-v2 | 109M | 768 | Fast | 57.8 | General purpose |
| bge-large-en-v1.5 | 335M | 1024 | Medium | 64.2 | High accuracy retrieval |
| text-embedding-3-large (OpenAI) | — | 3072 | API | 64.6 | Best accuracy, cloud |
| e5-mistral-7b | 7B | 4096 | Slow | 66.6 | SOTA, expensive |

**MTEB** = Massive Text Embedding Benchmark — standard evaluation across 56 tasks.

---

## STS Benchmark (Standard Evaluation)

```
STS-B (Semantic Textual Similarity Benchmark):
  8,628 sentence pairs with human similarity scores 0–5
  Evaluated with Pearson and Spearman correlation

Example pairs:
  "A man is playing guitar"  vs "A man plays guitar"    → score 4.75 (near identical)
  "A cat is sleeping"        vs "A dog is running"      → score 0.80 (unrelated)
  "The stock fell"           vs "Shares declined"       → score 4.20 (paraphrase)

Model evaluation:
  Compute cosine_similarity(encode(s1), encode(s2)) for all pairs
  Compute Spearman correlation with human scores

State of the art: ~93 Spearman correlation (near human agreement ~95)
```

```python
from datasets import load_dataset
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-mpnet-base-v2")
sts = load_dataset("mteb/stsbenchmark-sts", split="test")

sentences1  = sts["sentence1"]
sentences2  = sts["sentence2"]
gold_scores = np.array(sts["score"]) / 5.0  # normalize to [0,1]

emb1 = model.encode(sentences1, normalize_embeddings=True)
emb2 = model.encode(sentences2, normalize_embeddings=True)

cosine_scores = (emb1 * emb2).sum(axis=1)  # dot product of normalized = cosine

spearman = spearmanr(gold_scores, cosine_scores).correlation
print(f"Spearman correlation: {spearman:.4f}")
# all-mpnet-base-v2: 0.8805
# SBERT original:    0.8654
# BERT [CLS] (naive): 0.2040  ← BERT without fine-tuning is poor for similarity
```

---

## Gotchas

**BERT [CLS] without fine-tuning is terrible for similarity.** Vanilla BERT [CLS] cosine similarity performs at ~0.20 Spearman (barely above random) because BERT training didn't optimize for semantic similarity. Always use a model fine-tuned specifically for STS tasks (SBERT, BGE, E5).

**Normalize before dot product.** If using dot product for similarity, L2-normalize first. Un-normalized embeddings: long vectors dominate (popular documents get higher scores regardless of relevance).

**Cross-encoder calibration.** Cross-encoder scores are logits, not probabilities — absolute values aren't meaningful. Only the ranking matters. Don't threshold cross-encoder scores (e.g., "only show results >5.0") — this is arbitrary and unstable across models.

**Domain mismatch.** General-purpose embeddings (trained on Wikipedia, web) may underperform on specialized domains (medical, legal, code). Fine-tune on in-domain data or use domain-specific models (e.g., BioSentBERT for biomedical).

---

## Interview Q&A

**Q: What is the difference between a bi-encoder and a cross-encoder?**
A: Bi-encoder: encode query and document separately → compare embeddings via cosine similarity. Fast — documents pre-computed offline, only query needs encoding at inference. Lower accuracy because query-document interaction only happens via dot product at the end. Cross-encoder: feed query + document concatenated through the model → single relevance score. Accurate — full attention across both texts captures exact matches, negation, context. Slow — must run a full forward pass per (query, doc) pair. Standard production pipeline: bi-encoder retrieves top-100, cross-encoder reranks to top-10.

**Q: Why is cosine similarity preferred over dot product for sentence embeddings?**
A: Cosine similarity is magnitude-invariant — it measures the angle between vectors, not their length. Dot product is proportional to cosine × magnitude, so longer vectors (potentially encoding more content) would always score higher regardless of relevance. For comparing semantic similarity, we want direction (semantic content), not magnitude. Most embedding models produce L2-normalized vectors, making dot product = cosine similarity anyway — but if you're unsure whether vectors are normalized, cosine is safer.

**Q: How would you build a semantic search system for 1M documents?**
A: (1) Offline: encode all 1M documents with a bi-encoder (e.g., all-mpnet), build a FAISS index (HNSW for ANN search). (2) Online: encode query → FAISS.search(query, k=100) → retrieve 100 candidates in sub-millisecond. (3) Rerank: run cross-encoder on (query, 100 candidates) → re-sort by score → return top-10. (4) Generate: pass top-10 to LLM as context. Total latency: ~200ms. Scale: FAISS handles 1M 768-dim vectors on 6GB RAM.

---

## Connections

- **RAG pipeline:** `6.llms/08_rag_end_to_end.md` — bi-encoder + FAISS + cross-encoder in full RAG
- **CLIP (image-text similarity):** `7.multimodal/04_clip_finetuning_end_to_end.md`
- **FAISS index types:** `9.system_design/03_search_and_rag_system.md`
- **Sentence embeddings end-to-end:** `4.nlp/02_embeddings/03_sentence_embeddings_end_to_end.md`

## Key Takeaway

Semantic similarity = encode text → compare embeddings via cosine similarity. Two model types: **bi-encoder** (encode separately, fast, use for retrieval) and **cross-encoder** (encode jointly, accurate, use for reranking). Standard pipeline: bi-encoder → FAISS top-100 → cross-encoder rerank → top-10. Use L2-normalized embeddings + dot product = cosine similarity for efficiency. BERT without STS fine-tuning is useless for similarity (Spearman 0.20). Use SBERT, BGE, or E5 models for production.

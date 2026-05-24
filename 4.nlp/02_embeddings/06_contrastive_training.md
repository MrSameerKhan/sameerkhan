# Contrastive Embedding Training

> How modern embedding models (BGE, E5, MiniLM, OpenAI ada) are trained. The technique behind RAG quality.

---

## Table of Contents

1. Objective
2. The contrastive setup
3. In-batch negatives
4. Hard negative mining
5. Temperature and similarity functions
6. Modern recipe — bi-encoder fine-tuning
7. Failure modes
8. Interview questions (5)
9. Further reading

---

## 1. Objective

Sentence embeddings used in RAG (Phase 5) are trained with contrastive loss. **The choice of negatives drives quality more than any other decision.**

Senior interview Q: "How would you train a custom embedding model for legal documents?" or "What's hard negative mining and why does it matter?"

---

## 2. The Contrastive Setup

The goal: produce a vector for each text such that:
- Texts that should be SIMILAR have HIGH cosine similarity
- Texts that should be DISSIMILAR have LOW cosine similarity

Training data: **triples** (anchor, positive, negatives) — anchor: a query; positive: a document that SHOULD match the query; negatives: documents that should NOT match.

For retrieval, "positive" might be the correct document for a query; "negatives" are random other documents.

### The InfoNCE loss

For a batch where each row has an anchor q_i, a positive p_i, and N negatives n_{i,j}:

```
sim(a, b) = cos(a, b) / τ                          (temperature-scaled cosine)
L_i = -log[ exp(sim(q_i, p_i)) / (exp(sim(q_i, p_i)) + Σ_j exp(sim(q_i, n_{i,j}))) ]
```

Translation: softmax classification — out of {p_i, n_{i,1}, ..., n_{i,N}}, classify the positive correctly.

### Why softmax classification works

Each anchor q_i competes for similarity against ALL the candidates. As training progresses:
- Similarity to the positive goes UP
- Similarity to each negative goes DOWN
- Both the encoder for q AND the encoder for p (often the same network for symmetric retrieval) get gradients.

---

## 3. In-batch Negatives

The cheap and effective trick: use OTHER positives in the batch as negatives for each anchor.

```
Batch of B (anchor, positive) pairs:
  (q_1, p_1), (q_2, p_2), ..., (q_B, p_B)

For q_i, treat p_j (j ≠ i) as a negative.
Each anchor competes against B-1 in-batch negatives + its own positive.
```

**Cost: free.** The other positives are already in the batch. No extra encoding needed.

### Why bigger batches help

B=32: each anchor has ~31 negatives. Easy contrastive task. B=512: each anchor has ~511 negatives. Harder task → better features.

This is why SimCLR and CLIP train with batches of 4096-32768.

### Limitation

In-batch negatives are RANDOM. For RAG, where queries and documents are domain-specific, random negatives are often "easy" — clearly unrelated. The model never learns to distinguish near-duplicates. This is where hard negatives come in.

---

## 4. Hard Negative Mining

A "hard negative" is a candidate that's NOT the correct match but the model CONFUSES with the positive — high similarity, wrong answer.

### Why hard negatives matter

With easy negatives: model achieves 99% recall@5 quickly but never improves at fine distinctions. With hard negatives: model is forced to learn what subtly distinguishes the correct doc from a near-duplicate.

### Mining strategies

**1. BM25-based hard negatives** For each (query, positive) pair, retrieve top-K with BM25. Skip the positive; the rest are hard negatives. Cheap to compute. BM25 gets the "vocabulary overlap" right but misses semantic match. Standard approach for first iteration of training.

**2. ANN-based hard negatives** After a first training round, use the current model to find top-K candidates for each query. Non-positives = hard negatives. More expensive (need to embed all docs). Catches semantically similar but wrong docs. Used in iterative training (E5, BGE).

**3. Cross-encoder labeled hard negatives** Use a stronger cross-encoder model to score candidate-positive pairs. The ones the cross-encoder ranks high but aren't truly positives → hard negatives. Highest quality. Most expensive (need a good cross-encoder). Used by Cohere, OpenAI for proprietary embedders.

### The iterative training loop

Modern retrieval models (BGE, E5) train iteratively:
1. Train v1 with in-batch + BM25 hard negatives
2. Use v1 to mine new hard negatives
3. Train v2 with the new hard negatives
4. Repeat 2-3 rounds

Each iteration squeezes another few percent out of recall.

---

## 5. Temperature and Similarity Functions

### Temperature τ

Controls how "sharp" the softmax is:
- τ = 0.05 (low) — strong contrast, slow training, high-quality features
- τ = 0.1 — typical for production
- τ = 1.0 (high) — soft, fast training, weaker features

Empirically, sentence-transformers use τ = 0.02-0.05; CLIP uses LEARNED temperature.

### Cosine vs dot product

For NORMALIZED embeddings (||e||=1), cosine = dot product. For unnormalized, dot has a bias toward larger-magnitude vectors.

**Always normalize before storing in a vector DB.** It makes cosine = dot = a fast operation.

### Margin-based losses (alternative to InfoNCE)

- **Triplet loss** (Schroff 2015): `max(0, d(a, p) - d(a, n) + margin)`. Predates InfoNCE. Used in FaceNet originally.
- **Multiple negatives ranking** (MNR, Reimers 2019): in-batch negatives with cross-entropy. Cleaner version of triplet.

InfoNCE is the modern default for sentence embeddings.

---

## 6. Modern Recipe — Bi-encoder Fine-tuning

For Sentence-BERT style training:

```python
# Input: (query, positive_doc) pairs
# Optional: explicit hard negatives per pair

# Architecture:
encoder    = "BERT-base or DistilBERT or similar"
encode_query = encoder + mean_pool + normalize
encode_doc   = encoder + mean_pool + normalize   # same encoder for symmetric

# Loss:
q_emb      = encode_query(query_batch)           # [B, d]
d_emb      = encode_doc(positive_batch)          # [B, d]
similarity  = q_emb @ d_emb.T / τ               # [B, B]
labels      = arange(B)                          # (diagonal = positives)
L           = cross_entropy(similarity, labels)
```

### Asymmetric variant (used by BGE, E5)

Query and document encoded with the SAME network but DIFFERENT prompt prefixes:

```python
encode_query(q) = encoder("Represent this sentence for searching relevant passages: " + q)
encode_doc(d)   = encoder(d)
```

This tiny change gives 2-5% recall lift on most retrieval benchmarks.

### Hyperparameters (rough)

- Batch size: 64-512 (bigger = better; limited by GPU)
- LR: 2e-5 (similar to BERT fine-tuning)
- Temperature: 0.05
- Epochs: 1-3 (with iterative hard-negative mining between)
- Mean pooling > CLS pooling for sentence retrieval

---

## 7. Failure Modes

**1. All negatives are too easy** — in-batch (random + tiny dataset) → model can't learn to distinguish near-duplicates. Add BM25+ANN hard negatives.

**2. False negatives in hard mining** — a "hard negative" sometimes IS actually correct. The retriever has produced a relevant document not in your training labels. Filter false negatives (e.g., via cross-encoder scoring above threshold = "actually positive, skip as negative").

**3. Forgetting general features** — heavy fine-tuning on a narrow domain → model loses ability to handle out-of-domain queries. Mix in a few general training examples (10-20%).

**4. Batch contains semantically equivalent positives** — two different queries in the batch happen to have the same answer document. They become each other's negatives. Mitigation: dedup batch by document.

**5. Asymmetric prompts inconsistency** — train with the asymmetric prompt prefix, deploy without it. Performance tanks. Always use the same prompts at inference as at training.

---

## 8. Interview Questions (5)

**Q1: Walk me through training a sentence embedding model.**
Bi-encoder: same BERT-style encoder for query and document; mean-pool, normalize. Loss: InfoNCE — softmax classification where the positive doc must score higher than negatives. In-batch negatives are free; add hard negatives via BM25 retrieval or iterative ANN mining. Temperature ~0.05, batch 256+, train 2-3 iterations with new hard negatives each round.

**Q2: What's hard negative mining and why does it matter?**
A hard negative is a candidate the model CONFUSES with the positive — high similarity, wrong answer. With only in-batch random negatives, the model never learns to distinguish near-duplicates. Hard negatives (mined via BM25 or the current model) force the model to make fine distinctions. Typical lift: 5-15% recall@5 on dense retrieval benchmarks.

**Q3: Why use mean pooling instead of [CLS] for sentence retrieval?**
[CLS] is pretrained for NSP (binary classification), not for similarity. Empirically, mean pooling over all token embeddings produces better similarity scores. Sentence-BERT showed this clearly — mean pool + contrastive fine-tuning beats raw [CLS] by 10-15% on STS benchmarks.

**Q4: Cosine vs dot product — when does it matter?**
For NORMALIZED embeddings (always normalize before storing!), cosine = dot product. The choice doesn't matter. For UNNORMALIZED, dot product favors larger-magnitude vectors — which makes scores depend on text length / specificity in unhelpful ways. Production rule: always L2-normalize then use dot (fast).

**Q5: How would you train a custom embedding model for legal documents?**
(1) Collect (query, relevant_clause) pairs from existing legal Q&A data or contracts. (2) Start from a strong general embedder (BGE-base). (3) Fine-tune with InfoNCE, in-batch negatives + BM25 hard negatives mined from your legal corpus. (4) Use asymmetric prompts ("Represent this legal clause for retrieval."). (5) Iterate hard negative mining with the v1 model. Expect 10-20% improvement on legal-domain recall over general BGE.

---

## 9. Further Reading

- Sentence-BERT (Reimers & Gurevych 2019) — arXiv:1908.10084
- DPR (Karpukhin et al. 2020) — arXiv:2004.04906 — dense retrieval for QA
- ANCE (Xiong et al. 2020) — arXiv:2007.00808 — iterative ANN hard negatives
- E5 (Wang et al. 2022) — arXiv:2212.03533 — text embedding via contrastive learning
- BGE (Chen et al. 2024) — arXiv:2402.03216 — strong open embedder family
- InfoNCE / CPC (Oord et al. 2018) — arXiv:1807.03748 — the loss origin
- MTEB leaderboard — huggingface.co/spaces/mteb/leaderboard — evaluation benchmark

---

## Code Practice — Wired by Phase 6

- `code_practice/05_rag/02_embeddings/` — MiniLM vs BGE comparison

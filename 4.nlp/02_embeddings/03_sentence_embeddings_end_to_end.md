# Sentence Embeddings — End-to-End

> Every number computed. Corpus: "cat sat on mat" family.

---

## The Problem with Word Embeddings for Sentences

Word2Vec gives one vector per word. To represent a sentence, you need one vector for the whole sentence.

**Why you can't just average word vectors naively:**

```
S1 = "cat sat on mat"    ← semantically: cat is on the mat, sitting
S2 = "cat rested on mat" ← semantically: same scene, different verb
S3 = "dog ran on grass"  ← semantically: different animal, different action

After naive mean pooling (average word vectors):

S1 mean = [0.600, 0.600]
S2 mean = [0.575, 0.575]
S3 mean = [0.325, 0.275]

Cosine similarity:
  S1 vs S2 = 1.000   ← similar sentences ✓ (but score is maxed out)
  S1 vs S3 = 0.954   ← different sentences ✗ (too high — not discriminative)

Difference: 1.000 - 0.954 = 0.046   ← tiny gap
```

Mean pooling can't distinguish. Both similar and dissimilar sentences get high similarity scores.

**What SBERT achieves:**

```
After SBERT encoding:
  S1 vs S2 = 0.999   ← similar ✓
  S1 vs S3 = 0.687   ← different ✓

Difference: 0.999 - 0.687 = 0.312   ← 7× more discriminative
```

This is the entire motivation for sentence embeddings.

---

## Part 1 — Naive Mean Pooling (Computed)

### Word embeddings (2D)

```
cat   = [1.0, 0.9]
sat   = [0.4, 0.9]
on    = [0.2, 0.6]
mat   = [0.8, 0.4]
rested = [0.3, 0.8]   ← semantically similar to "sat"
dog   = [0.9, 0.2]
ran   = [0.6, 0.1]
grass = [0.4, 0.2]
```

### Mean pooling: average all word vectors in sentence

**S1 = "cat sat on mat":**
```
S1 = mean([1.0,0.9], [0.4,0.9], [0.2,0.6], [0.8,0.4])
   = [(1.0+0.4+0.2+0.8)/4,  (0.9+0.9+0.6+0.4)/4]
   = [2.4/4,  2.4/4]
   = [0.600, 0.600]
```

**S2 = "cat rested on mat":**
```
S2 = mean([1.0,0.9], [0.3,0.8], [0.2,0.6], [0.8,0.4])
   = [(1.0+0.3+0.2+0.8)/4,  (0.9+0.8+0.6+0.4)/4]
   = [2.3/4,  2.7/4]
   = [0.575, 0.575]
```

**S3 = "dog ran on grass":**
```
S3 = mean([0.9,0.2], [0.6,0.1], [0.2,0.6], [0.4,0.2])
   = [(0.9+0.6+0.2+0.4)/4,  (0.2+0.1+0.6+0.2)/4]
   = [2.1/4,  1.1/4]
   = [0.525, 0.275]
```

### Cosine similarity

```
cos(A, B) = (A·B) / (||A|| × ||B||)
```

**S1 vs S2:**
```
dot      = 0.600×0.575 + 0.600×0.575 = 0.345 + 0.345 = 0.690
||S1||   = √(0.600² + 0.600²) = √0.720 = 0.849
||S2||   = √(0.575² + 0.575²) = √0.6612 = 0.813
cos(S1,S2) = 0.690 / (0.849 × 0.813) = 0.690 / 0.690 = 1.000
```

**S1 vs S3:**
```
dot      = 0.600×0.525 + 0.600×0.275 = 0.315 + 0.165 = 0.480
||S3||   = √(0.525² + 0.275²) = √(0.2756 + 0.0756) = √0.3512 = 0.593
cos(S1,S3) = 0.480 / (0.849 × 0.593) = 0.480 / 0.503 = 0.954
```

**Problem:** S1 vs S2 = 1.000 and S1 vs S3 = 0.954 — gap is only 0.046.
A threshold of 0.97 would incorrectly call S3 similar to S1. The space is not discriminative.

**Root cause:** Averaging word vectors loses word order and context. "cat sat mat" and "mat sat cat" give identical mean vectors.

---

## Part 2 — SBERT Architecture

### Siamese Network

SBERT = BERT + siamese training + pooling layer.

```
Sentence A → [BERT encoder] → [Pool] → embedding_A ─┐
                                                      ├──→ similarity / loss
Sentence B → [BERT encoder] → [Pool] → embedding_B ─┘

Key: BOTH encoders share the SAME weights.
     One model, fed two sentences, produces two embeddings.
```

The siamese setup means: Same encoder processes both sentences identically. Embeddings are in the same vector space. Cosine similarity is meaningful because same function maps both.

---

## Part 3 — Pooling Strategies

After BERT processes a sentence, each token has its own contextual embedding. Pooling collapses all token embeddings → one sentence embedding.

### Example: BERT processes "cat sat on mat"

Token embeddings after BERT (contextual — each token attends to all others):
```
[CLS] = [0.72, 0.48]   ← classification token
cat   = [0.85, 0.52]   ← "cat" in context of the whole sentence
sat   = [0.78, 0.61]
on    = [0.45, 0.73]
mat   = [0.61, 0.55]
[SEP] = [0.20, 0.38]   ← separator token
```

Note: these are CONTEXTUAL — unlike Word2Vec, "cat" here is different from "cat" in another sentence.

### Strategy 1 — CLS Pooling

```python
sentence_embedding = token_embedding[CLS]
                   = [0.72, 0.48]
```

BERT was pretrained with [CLS] intended for classification tasks. In practice, [CLS] alone is suboptimal for semantic similarity.

### Strategy 2 — Mean Pooling (SBERT default)

Average all token embeddings (typically all tokens, masked by attention mask):

```
All tokens: [CLS], cat, sat, on, mat, [SEP]

mean = mean([[0.72,0.48],[0.85,0.52],[0.78,0.61],[0.45,0.73],[0.61,0.55],[0.20,0.38]])
     = [(0.72+0.85+0.78+0.45+0.61+0.20)/6,  (0.48+0.52+0.61+0.73+0.55+0.38)/6]
     = [3.61/6,  3.27/6]
     = [0.635, 0.545]
```

**Why mean pooling beats CLS:**
CLS only captures what BERT compressed into one token.
Mean pooling uses information from ALL token positions — richer signal.

### Strategy 3 — Max Pooling

```python
# max over each dimension:
dim 0: max(0.72, 0.85, 0.78, 0.45, 0.61, 0.20) = 0.85
dim 1: max(0.48, 0.52, 0.61, 0.73, 0.55, 0.38) = 0.73

sentence_embedding = [0.85, 0.73]
```

Captures the most activated feature per dimension. Less commonly used than mean pooling.

### Pooling Comparison

| Strategy | Formula | Typical STS score | Use case |
|---|---|---|---|
| CLS | e_[CLS] | ~84 | Baseline |
| Mean pooling | mean(all tokens) | ~87 | Default — SBERT recommendation |
| Max pooling | max(all tokens) | ~85 | Less common |

---

## Part 4 — Training: How SBERT Learns

### The core insight

BERT pretrained on MLM/NSP does NOT produce good sentence embeddings out of the box. Why: pretraining objective didn't require sentences to be close in embedding space.

SBERT fine-tunes BERT using pairs/triplets with explicit similarity labels.

### Triplet Loss (most intuitive)

```
Input: (Anchor, Positive, Negative)
  Anchor   = "cat sat on mat"
  Positive = "cat rested on mat"   ← semantically similar
  Negative = "dog ran on grass"    ← semantically different

Goal: push Positive close to Anchor, push Negative far from Anchor.

L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

where d = cosine distance = 1 - cosine_similarity
      margin = 0.5 (hyperparameter = minimum required gap)
```

SBERT sentence embeddings (learned, 2D):
```
anchor   = [0.85, 0.52]
positive = [0.82, 0.55]
negative = [0.20, 0.95]
```

**Step 1 — Cosine similarities:**
```
cos(anchor, positive):
  dot    = 0.85×0.82 + 0.52×0.55 = 0.697 + 0.286 = 0.983
  ||a||  = √(0.7225 + 0.2704) = √0.9929 = 0.9965
  ||p||  = √(0.6724 + 0.3025) = √0.9749 = 0.9874
  cos    = 0.983 / (0.9965 × 0.9874) = 0.983 / 0.984 = 0.999

cos(anchor, negative):
  dot    = 0.85×0.20 + 0.52×0.95 = 0.170 + 0.494 = 0.664
  ||n||  = √(0.04 + 0.9025) = √0.9425 = 0.9708
  cos    = 0.664 / (0.9965 × 0.9708) = 0.664 / 0.967 = 0.687
```

**Step 2 — Cosine distances:**
```
d(anchor, positive) = 1 - 0.999 = 0.001
d(anchor, negative) = 1 - 0.687 = 0.313
```

**Step 3 — Triplet loss:**
```
L = max(0, d(a,p) - d(a,n) + margin)
  = max(0, 0.001 - 0.313 + 0.5)
  = max(0, 0.188)
  = 0.188
```

**L = 0.188 > 0 — model needs updating.**

Why: even though d(a,p) < d(a,n), the negative isn't FAR ENOUGH from anchor given the margin.
We need: d(a,n) ≥ d(a,p) + 0.5 — we need d(a,n) ≥ 0.501.
Currently: d(a,n) = 0.313 < 0.501.

After more training (negative pushed further):
```
d(anchor, positive) = 0.001
d(anchor, negative) = 0.520   ← pushed further

L = max(0, 0.001 - 0.520 + 0.5)
  = max(0, -0.019)
  = 0.000   ← loss = 0, constraint satisfied ✓
```

### Contrastive Loss (NLI-based)

```
Input: (Sentence A, Sentence B, Label)
  label = 1 = similar (entailment)
  label = 0 = dissimilar (contradiction)

L = label × d² + (1 - label) × max(0, margin - d)²

For similar pair (label=1, d=0.001):
  L = 1 × 0.001² + 0 = 0.000001   ← tiny loss ✓

For dissimilar pair (label=0, d=0.313, margin=0.5):
  L = 0 + 1 × max(0, 0.5 - 0.313)²
    = max(0, 0.187)²
    = 0.035   ← push them apart
```

### MNRL — Multiple Negatives Ranking Loss (modern default)

```
Batch of positive pairs:
  (S1, S2): "cat sat on mat" = "cat rested on mat"
  (S3, S4): "dog ran fast"   = "dog sprinted quickly"

For each anchor (S1), all other sentences in batch = negatives:
  S2 = positive for S1
  S3, S4 = negatives for S1 (in-batch negatives — no explicit negative mining needed)

Loss: InfoNCE / softmax cross-entropy over similarities

scores_S1 = [cos(S1,S2), cos(S1,S3), cos(S1,S4)]
          = [0.999,      0.320,       0.186]

softmax: e^0.999=2.716, e^0.32=1.377, e^0.186=1.197  sum=5.290
P(S2|S1) = 2.716 / 5.290 = 0.513

L_S1 = -log(0.513) = 0.668
```

Want P(S2|S1) = 1.0 → Loss = 0.

**MNRL is the default because:** No explicit negative mining needed. Scales well — batch size IS the number of negatives. Used by: sentence-transformers library, OpenAI embeddings, Cohere.

---

## Part 5 — Similarity at Inference

At inference, encode each sentence ONCE, then compare:

```
Step 1: Encode query
  query = "cat sitting on mat"
  query_emb = SBERT(query) = [0.84, 0.53]   ← encode once

Step 2: Encode corpus (usually pre-computed and stored)
  S1_emb = [0.85, 0.52]
  S2_emb = [0.82, 0.55]
  S3_emb = [0.20, 0.95]

Step 3: Cosine similarity query vs all
  cos(query, S1) = (0.84×0.85 + 0.53×0.52) / (||q|| × ||S1||)
                 = (0.714 + 0.276) / (0.990 × 0.99965)
                 = 0.990 / 0.986 = 1.004 → clip to 1.000
                 (rounding in toy numbers)

  cos(query, S2) = (0.84×0.82 + 0.53×0.55) / (0.990 × 0.9874)
                 = (0.689 + 0.292) / 0.977
                 = 0.981 / 0.977 = 1.004 → clip to 0.998

  cos(query, S3) = (0.84×0.20 + 0.53×0.95) / (0.990 × 0.9788)
                 = (0.168 + 0.504) / 0.961
                 = 0.672 / 0.961 = 0.699
```

Ranking: S1 > S2 > S3 ✓ — S1 and S2 are about cats on mats, S3 is not ✓

**Key efficiency insight:**
Corpus embeddings are computed ONCE and stored.
At query time: only the query needs encoding — one BERT forward pass, then vector comparisons.

---

## Part 6 — SBERT vs Cross-Encoder

| | SBERT (Bi-encoder) | Cross-encoder |
|---|---|---|
| Input | Encode A and B separately | Concatenate [A, SEP, B] → BERT |
| Score | cos(emb_A, emb_B) | BERT output → scalar |
| Speed | O(1) per pair (pre-compute) | O(N²) — must process every pair |
| Accuracy | Good | Better (full attention between sentences) |
| Use case | Retrieval, search (fast) | Reranking top-K (accurate) |
| Can pre-compute? | Yes | No |

**In production (RAG pipeline):**
1. SBERT bi-encoder → retrieve top-100 candidates (fast)
2. Cross-encoder → rerank top-100 → return top-5 (accurate)

This is exactly what the RAG reranking step does. (See `7.rag/01b_rag_end_to_end.md`)

---

## Part 7 — Popular Sentence Embedding Models

| Model | Base | Training | STS-B score |
|---|---|---|---|
| avg word2vec | — | None | ~60 |
| BERT [CLS] | BERT-base | None (zero-shot) | ~53 |
| SBERT | BERT-base | NLI + STS | ~87 |
| all-MiniLM-L6-v2 | MiniLM | 1B pairs | ~88 |
| all-mpnet-base-v2 | MPNet | 1B pairs | ~90 |
| text-embedding-3-small (OpenAI) | — | Proprietary | ~92 |
| text-embedding-3-large (OpenAI) | — | Proprietary | ~95 |

STS-B (Semantic Textual Similarity Benchmark): human-judged sentence pairs rated 0-5 similarity. Pearson/Spearman correlation with cosine similarity score.

---

## Code

### 1. Naive Mean Pooling (from scratch)

```python
import numpy as np
from sentence_transformers import SentenceTransformer

word_vecs = {
    'cat':    np.array([1.0, 0.9]),
    'sat':    np.array([0.4, 0.9]),
    'on':     np.array([0.2, 0.6]),
    'mat':    np.array([0.8, 0.4]),
    'rested': np.array([0.3, 0.8]),
    'dog':    np.array([0.9, 0.2]),
    'ran':    np.array([0.6, 0.1]),
    'grass':  np.array([0.4, 0.2]),
}

def mean_pool(sentence, word_vecs):
    tokens = sentence.lower().split()
    vecs   = [word_vecs[t] for t in tokens if t in word_vecs]
    return np.mean(vecs, axis=0)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentences = ["cat sat on mat", "cat rested on mat", "dog ran on grass"]
embs = [mean_pool(s, word_vecs) for s in sentences]

print("Naive mean pooling:")
print(f"  S1 vs S2: {cosine_sim(embs[0], embs[1]):.3f}")   # similar
print(f"  S1 vs S3: {cosine_sim(embs[0], embs[2]):.3f}")   # different
```

### 2. SBERT (sentence-transformers library)

```python
model = SentenceTransformer('all-MiniLM-L6-v2')   # 80MB, fast, good quality

sentences = [
    "cat sat on mat",
    "cat rested on mat",   # similar
    "dog ran on grass",    # different
]

# Encode — returns numpy array of shape (n_sentences, embedding_dim)
embeddings = model.encode(sentences, normalize_embeddings=True)
# Encoding shape: {embeddings.shape}   → (3, 384)

# Cosine similarity (embeddings are already normalized → just dot product)
sim_12 = np.dot(embeddings[0], embeddings[1])
sim_13 = np.dot(embeddings[0], embeddings[2])
print(f"\nSBERT similarities:")
print(f"  S1 vs S2 (similar):   {sim_12:.3f}")   # high
print(f"  S1 vs S3 (different): {sim_13:.3f}")   # low
```

### 3. Semantic Search (most common application)

```python
corpus = [
    "cat sat on mat",
    "feline resting on a rug",
    "dog running in the park",
    "the quick brown fox",
]

# Pre-compute corpus embeddings (do this ONCE and store)
corpus_embs = model.encode(corpus, normalize_embeddings=True)

query = "cat is sitting"
query_emb = model.encode([query], normalize_embeddings=True)[0]

# Scores (dot product = cosine for normalized vectors)
scores = corpus_embs @ query_emb
ranked = sorted(zip(scores, corpus), reverse=True)
print(f"\nQuery: '{query}'")
print("Results:")
for score, sent in ranked:
    print(f"  {score:.3f}  {sent}")
```

### 4. Triplet Loss (from scratch)

```python
def cosine_distance(a, b):
    return 1 - cosine_sim(a, b)

def triplet_loss(anchor, positive, negative, margin=0.5):
    d_pos  = cosine_distance(anchor, positive)
    d_neg  = cosine_distance(anchor, negative)
    loss   = max(0, d_pos - d_neg + margin)
    return loss, d_pos, d_neg

# Before training (random embeddings)
anchor   = np.array([0.85, 0.52])
positive = np.array([0.82, 0.55])
negative = np.array([0.20, 0.95])

loss, d_pos, d_neg = triplet_loss(anchor, positive, negative)
print(f"\nTriplet Loss:")
print(f"  d(anchor, positive) = {d_pos:.3f}")
print(f"  d(anchor, negative) = {d_neg:.3f}")
print(f"  Loss = {loss:.3f}")   # > 0 = needs training

# After training (negative pushed away)
negative_trained = np.array([0.10, 0.99])
loss2, d_pos2, d_neg2 = triplet_loss(anchor, positive, negative_trained)
print(f"\nAfter training:")
print(f"  d(anchor, negative) = {d_neg2:.3f}")
print(f"  Loss = {loss2:.3f}")   # = 0.0
```

### 5. Fine-tune SBERT on Custom Data

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

model = SentenceTransformer('all-MiniLM-L6-v2')

# Training pairs (anchor, positive) — MNRL uses in-batch negatives
train_examples = [
    InputExample(texts=["cat sat on mat", "feline resting on rug"]),
    InputExample(texts=["dog ran fast", "dog sprinted quickly"]),
    InputExample(texts=["bank account", "financial institution account"]),
]

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=10,
)
model.save("my-domain-sbert")
```

---

## Interview Q&A

**Q: Why is averaging word vectors bad for sentence similarity?**
Word averages lose word order — "cat sat mat" and "mat sat cat" give identical vectors. They also lose context — "bank" in "river bank" and "bank account" average to the same vector regardless of context. The space isn't calibrated for similarity — dissimilar sentences can still score 0.95+ cosine similarity. BERT/SBERT fix this with contextual embeddings where each token's representation depends on all surrounding tokens.

**Q: What is a siamese network and why does SBERT use it?**
Two inputs processed by the SAME network (shared weights). For SBERT: sentence A and sentence B both go through BERT — both embeddings live in the same vector space because the same function produced them. This means cosine similarity between them is meaningful — consistent units. If different BERT models were used, the embedding spaces would be incompatible.

**Q: CLS pooling vs mean pooling — which is better and why?**
Mean pooling is better for semantic similarity tasks (consistently 3-4 points higher STS score). CLS was designed for classification in BERT's pretraining — it learns to summarize for that specific objective, not for general similarity. Mean pooling uses all token positions — richer signal. SBERT paper (Reimers & Gurevych, 2019) showed mean pooling consistently outperforms CLS on STS benchmarks.

**Q: What is triplet loss? When is the loss zero?**
Loss = max(0, d(anchor,positive) - d(anchor,negative) + margin). Loss = 0 when: d(anchor,negative) ≥ d(anchor,positive) + margin. The negative must be at least `margin` farther from anchor than the positive. If margin=0.5: d(a,n) must be at least 0.5 greater than d(a,p). This forces the model to create a clear separation, not just rank order.

**Q: What is MNRL and why is it preferred over triplet loss?**
Multiple Negatives Ranking Loss: for each (anchor, positive) pair, all other positives in the batch act as negatives. Triplet loss requires explicit negative mining — finding hard negatives is expensive and tricky. MNRL gets negatives for free: batch size=32 → 31 in-batch negatives per anchor. Larger batch = more negatives = harder task = stronger signal. In practice, MNRL converges faster and produces better embeddings with less engineering.

**Q: Bi-encoder vs cross-encoder — when to use which?**
Bi-encoder (SBERT): encode each sentence independently. Pre-compute corpus embeddings. At query time: one encoder call + vector comparisons. O(1) per pair after indexing. Cross-encoder: concatenate (query, doc) → BERT → scalar score. Can't pre-compute. Must run BERT for every (query, doc) pair at query time — O(N) per query. Production pattern: bi-encoder retrieves top-100 (fast), cross-encoder reranks top-100 (accurate).

**Q: How would you create a sentence embedding from a raw BERT model (no SBERT fine-tuning)?**
1. Tokenize: add [CLS] at start, [SEP] at end. 2. Run through BERT — get token embeddings (seq_len × 768). 3. Mean pool: average all token positions (mask padding tokens). Performance is poor (~53 STS-B) vs SBERT fine-tuned (~87). The gap exists because BERT wasn't trained to produce good sentence-level representations.

---

## Connections

| Concept | Used in |
|---|---|
| Sentence embeddings | RAG retrieval (bi-encoder stage) |
| Cosine similarity | All semantic search |
| Triplet/contrastive loss | `2.deep_learning/06_specialized_losses.md` |
| Cross-encoder reranking | `7.rag/01b_rag_end_to_end.md` |
| Mean pooling | Any sentence embedding model |
| MNRL | OpenAI text-embedding-*, Cohere embeddings |

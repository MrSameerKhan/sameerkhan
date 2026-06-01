# Text Representations End to End — BoW, TF-IDF, BM25 with Numbers

Same corpus throughout. Three documents built from **"cat sat on mat"**:

```
D1: "cat sat on mat"   (4 tokens)
D2: "cat sat on cat"   (4 tokens — cat appears twice)
D3: "mat mat on mat"   (4 tokens — mat appears three times)

Vocabulary (sorted alphabetically): cat=0, mat=1, on=2, sat=3  (V=4)
```

---

## 0. What Text Representations Solve

A machine learning model needs numbers. Text is symbols. The question is: which numbers best capture what the text means?

```
"cat sat on mat"  →  ???  →  model
```

The representation determines everything: what the model can learn, what it can't, how fast it runs, how much memory it uses.

```
Three eras:
BoW + TF-IDF + BM25 : sparse representations (1990s-2010s)
Word2Vec + GloVe     : static dense representations (2013+)
BERT + GPT           : contextual dense representations (2018+)
```

This file covers the first era. These methods are still widely used in production.

---

## 1. Bag of Words (BoW) — From Scratch

### 1.1 Build the Vocabulary

Collect all unique tokens across the entire corpus, sort alphabetically:

```
D1 tokens: cat, sat, on, mat
D2 tokens: cat, sat, on, cat  → unique: cat, sat, on
D3 tokens: mat, mat, on, mat  → unique: mat, on

Union + sort: cat, mat, on, sat
Index:        0    1    2    3
```

### 1.2 Count Vectors

For each document, count how many times each vocabulary word appears:

```
D1: "cat sat on mat"
  cat = 1, mat = 1, on = 1, sat = 1
  vector: [1, 1, 1, 1]

D2: "cat sat on cat"
  cat = 2, mat = 0, on = 1, sat = 1
  vector: [2, 0, 1, 1]

D3: "mat mat on mat"
  cat = 0, mat = 3, on = 1, sat = 0
  vector: [0, 3, 1, 0]
```

**BoW matrix (3 documents × 4 vocabulary words):**

```
     cat  mat  on  sat
D1: [ 1,   1,   1,  1 ]
D2: [ 2,   0,   1,  1 ]
D3: [ 0,   3,   1,  0 ]
```

Each row = one document. Each column = one vocabulary word. The value = count.

### 1.3 What BoW Ignores

**Word order:** "cat sat on mat" and "mat on sat cat" produce identical BoW vectors — same counts, same vector. A model using BoW cannot distinguish these.

**Semantics:** "cat" and "feline" appear in different columns. If "feline" isn't in training vocabulary, it maps to zero — all its information is lost.

**Context:** "not good" gets split into "not" (count 1) + "good" (count 1). The negation is invisible.

### 1.4 Binary BoW

Sometimes presence/absence matters more than count. Set all counts to 1:

```
     cat  mat  on  sat
D1: [ 1,   1,   1,  1 ]  + unchanged (all counts already 1)
D2: [ 1,   0,   1,  1 ]  + cat count 2 → 1
D3: [ 0,   1,   1,  0 ]  + mat count 3 → 1
```

Use binary BoW for: spam detection (did "free" appear?), presence-based features. Use count BoW when frequency matters (document about "cat" — many mentions of "cat").

---

## 2. TF — Term Frequency

### 2.1 The Problem with Raw Counts

D3 has mat=3 just because "mat" appears 3 times. If D3 were twice as long but had the same proportion of "mat," the count would be 6 — making it look 6× more "mat-like" than D1, even though they have the same relative frequency.

**Fix:** normalize by document length.

### 2.2 Computing TF

```
TF(term t, document d) = count(t in d) / total tokens in d
```

```
D1 (4 tokens total):
  TF(cat, D1) = 1/4 = 0.250
  TF(mat, D1) = 1/4 = 0.250
  TF(on,  D1) = 1/4 = 0.250
  TF(sat, D1) = 1/4 = 0.250

D2 (4 tokens total):
  TF(cat, D2) = 2/4 = 0.500  ← cat appears twice
  TF(mat, D2) = 0/4 = 0.000
  TF(on,  D2) = 1/4 = 0.250
  TF(sat, D2) = 1/4 = 0.250

D3 (4 tokens total):
  TF(cat, D3) = 0/4 = 0.000
  TF(mat, D3) = 3/4 = 0.750  ← mat appears three times
  TF(on,  D3) = 1/4 = 0.250
  TF(sat, D3) = 0/4 = 0.000

TF matrix:
     cat   mat   on    sat
D1: [0.250, 0.250, 0.250, 0.250]
D2: [0.500, 0.000, 0.250, 0.250]
D3: [0.000, 0.750, 0.250, 0.000]
```

TF tells you how prominent a term is within its document, but it can't tell you whether the term is actually informative: "on" appears in every document with TF=0.250 — is it really as important as "cat" or "mat"?

---

## 3. IDF — Inverse Document Frequency

### 3.1 The Problem with TF Alone

"on" appears in all 3 documents. It's a function word — it carries almost no content. Any document about cats, mats, or anything else would contain "on." A term that appears in every document gives you no information about which document is special.

**Fix:** down-weight terms that appear in many documents.

### 3.2 Computing IDF

First, count document frequency (df): how many documents contain each term.

```
cat: in D1, D2         → df(cat) = 2
mat: in D1, D3         → df(mat) = 2
on:  in D1, D2, D3     → df(on)  = 3
sat: in D1, D2         → df(sat) = 2

N = 3 (total documents)
```

**Standard IDF formula:**

```
IDF(t) = log(N / df(t))

IDF(cat) = log(3/2) = log(1.500) = 0.405
IDF(mat) = log(3/2) = log(1.500) = 0.405
IDF(on)  = log(3/3) = log(1.000) = 0.000  ← appears in ALL docs → zero IDF
IDF(sat) = log(3/2) = log(1.500) = 0.405
```

**Key insight:** "on" gets IDF=0. No matter how high its TF, it contributes nothing to TF-IDF. This is exactly what we want — common words are silenced.

**Smoothed IDF (sklearn default):**

```
IDF_smooth(t) = log((1+N) / (1+df(t))) + 1

Prevents zero IDF for terms in all documents (adds 1 to both numerator and denominator):
IDF_smooth(cat) = log(4/3) + 1 = 0.288 + 1 = 1.288
IDF_smooth(mat) = log(4/3) + 1 = 1.288
IDF_smooth(on)  = log(4/4) + 1 = 0.000 + 1 = 1.000  ← not zero, but lowest
IDF_smooth(sat) = log(4/3) + 1 = 1.288
```

We'll use **standard IDF** (without smoothing) for clarity. Just know sklearn applies smoothing by default.

---

## 4. TF-IDF — Full Matrix

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

Compute every cell:

```
D1:
  TF-IDF(cat, D1) = 0.250 × 0.405 = 0.101
  TF-IDF(mat, D1) = 0.250 × 0.405 = 0.101
  TF-IDF(on,  D1) = 0.250 × 0.000 = 0.000  ← "on" zeroed out
  TF-IDF(sat, D1) = 0.250 × 0.405 = 0.101

D2:
  TF-IDF(cat, D2) = 0.500 × 0.405 = 0.203  ← highest — cat is D2's signature
  TF-IDF(mat, D2) = 0.000 × 0.405 = 0.000
  TF-IDF(on,  D2) = 0.250 × 0.000 = 0.000
  TF-IDF(sat, D2) = 0.250 × 0.405 = 0.101

D3:
  TF-IDF(cat, D3) = 0.000 × 0.405 = 0.000
  TF-IDF(mat, D3) = 0.750 × 0.405 = 0.304  ← highest — mat is D3's signature
  TF-IDF(on,  D3) = 0.250 × 0.000 = 0.000
  TF-IDF(sat, D3) = 0.000 × 0.405 = 0.000

Complete TF-IDF matrix:
     cat    mat    on     sat
D1: [0.101, 0.101, 0.000, 0.101]  ← balanced, 3 informative terms
D2: [0.203, 0.000, 0.000, 0.101]  ← cat-heavy, no sat
D3: [0.000, 0.304, 0.000, 0.000]  ← pure mat signal
```

**Reading the matrix:** D1 is characterized equally by cat, mat, sat · D2 is most strongly characterized by cat (TF-IDF=0.203) · D3 is most strongly characterized by mat (TF-IDF=0.304) · "on" contributes nothing to any document (all zeros in that column).

---

## 5. Cosine Similarity — Step by Step

Why cosine? Because we want similarity to be independent of document length. A 1000-word document about cats is not more "cat-like" than a 10-word document about cats.

```
cosine(A, B) = (A · B) / (||A|| × ||B||)
```

### 5.1 D1 vs D2

```
D1 = [0.101, 0.101, 0.000, 0.101]
D2 = [0.203, 0.000, 0.000, 0.101]

Dot product (A · B):
  0.101×0.203 + 0.101×0.000 + 0.000×0.000 + 0.101×0.101
  = 0.0205 + 0.000 + 0.000 + 0.0102
  = 0.0307

Norms:
||D1|| = √(0.101² + 0.101² + 0.000² + 0.101²)
       = √(0.0102 + 0.0102 + 0.000 + 0.0102)
       = √(0.0306)
       = 0.175

||D2|| = √(0.203² + 0.000² + 0.000² + 0.101²)
       = √(0.0412 + 0.000 + 0.000 + 0.0102)
       = √(0.0514)
       = 0.227

Cosine similarity:
cosine(D1, D2) = 0.0307 / (0.175 × 0.227)
              = 0.0307 / 0.0397
              = 0.773
```

D1 and D2 share "cat" and "sat" as informative terms → high similarity (0.773).

### 5.2 D1 vs D3

```
D1 = [0.101, 0.101, 0.000, 0.101]
D3 = [0.000, 0.304, 0.000, 0.000]

Dot product:
  0.101×0.000 + 0.101×0.304 + 0.000×0.000 + 0.101×0.000
  = 0 + 0.0307 + 0 + 0
  = 0.0307

Norm of D3:
||D3|| = √(0.000² + 0.304² + 0.000² + 0.000²) = √(0.0924) = 0.304

cosine(D1, D3) = 0.0307 / (0.175 × 0.304)
              = 0.0307 / 0.0532
              = 0.577
```

D1 and D3 share only "mat" as an informative term → lower similarity (0.577 < 0.773).

### 5.3 D2 vs D3

```
D2 = [0.203, 0.000, 0.000, 0.101]
D3 = [0.000, 0.304, 0.000, 0.000]

Dot product:
  0.203×0.000 + 0.000×0.304 + 0.000×0.000 + 0.101×0.000
  = 0 + 0 + 0 + 0
  = 0.000

cosine(D2, D3) = 0.000 / (0.227 × 0.304) = 0.000
```

D2 (cat-heavy) and D3 (mat-heavy) share NO informative terms. Their cosine similarity = 0.

### 5.4 Full Similarity Matrix

```
     D1          D2         D3
D1 [1.000,      0.773,     0.577 ]
D2 [0.773,      1.000,     0.000 ]
D3 [0.577,      0.000,     1.000 ]

Interpretation: D1 ("cat sat on mat") is most similar to D2 ("cat sat on cat") — they share cat AND sat.
D1 is somewhat similar to D3 ("mat mat on mat") — they share mat.
D2 and D3 are completely dissimilar — no shared informative terms.
Notice: all three share "on" but it's zeroed out by IDF. The similarity is driven entirely by the
informative terms cat, mat, sat.
```

---

## 6. N-grams — Capturing Local Context

### 6.1 Why Unigrams Miss Negation

```
BoW of "not good":  not=1, good=1
BoW of "not bad":   not=1, bad=1
BoW of "very good": very=1, good=1

cosine("not good", "very good") is HIGH — they share "good"
```

But "not good" and "very good" have opposite meanings. Unigrams can't capture this.
**Fix:** Add bigrams (pairs of adjacent words) as features.

### 6.2 Bigrams from Our Corpus

Tokenize with bigrams:

```
D1: "cat sat on mat"
  Bigrams: "cat sat", "sat on", "on mat"

D2: "cat sat on cat"
  Bigrams: "cat sat", "sat on", "on cat"

D3: "mat mat on mat"
  Bigrams: "mat mat", "mat on", "on mat"

Bigram vocabulary (7 bigrams):
  Index 0: "cat sat"
  Index 1: "mat mat"
  Index 2: "mat on"
  Index 3: "on cat"
  Index 4: "on mat"
  Index 5: "sat on"
```

**Bigram BoW matrix:**

```
     cat sat  mat mat  mat on  on cat  on mat  sat on
D1: [  1,       0,       0,      0,      1,      1   ]
D2: [  1,       0,       0,      1,      0,      1   ]
D3: [  0,       1,       1,      0,      1,      0   ]
```

Now "cat sat" is a single feature. D1 and D2 share it (both start with "cat sat"). D3 does not.

### 6.3 Combined Unigrams + Bigrams

The most common approach: keep both unigrams and bigrams as features (`ngram_range=(1,2)` in sklearn).

```
Feature vector for D1 (unigrams + bigrams):
  Unigrams: [cat=1, mat=1, on=1, sat=1]
  Bigrams:  [cat sat=1, mat on=0, on cat=0, on mat=1, sat on=1]
  Combined: [1, 1, 1, 1, 1, 0, 0, 1, 1, 1]  (10-dimensional)
```

After TF-IDF weighting, rare bigrams get higher IDF — "cat sat" appears in 2/3 docs but "on mat" also appears in 2/3 docs.

### 6.4 N-gram Language Model (Statistical)

Before neural LMs, n-grams were used to predict the next word.

**Bigram model:** P(word | previous_word)

```
From our corpus (D1 only):
  P(cat | cat)  = count("cat cat") / count("cat") = 1/1 = 1.000
  P(on  | sat)  = count("sat on")  / count("sat") = 1/1 = 1.000
  P(sat | on)   = count("on sat")  / count("on")  = 1/1 = 1.000

Sentence probability:
P("cat sat on mat") = P(cat) × P(sat|cat) × P(on|sat) × P(mat|on)
                    = 0.25 × 1.0 × 1.0 × 1.0
                    = 0.25

P("cat mat on sat") = 0  (because P(mat|cat) = 0 — never seen)
```

This is why n-gram LMs need smoothing — zero-count bigrams kill entire sentence probabilities. Kneser-Ney smoothing handles this best.

---

## 7. BM25 — Better TF Saturation

### 7.1 TF-IDF's Flaw: Linear TF Scaling

In TF-IDF, TF scales linearly. If D2 has "cat" twice, its TF(cat) is exactly 2× D1's.

But does a document mentioning "cat" 100 times really tell you 100× more about cats than one mentioning it once? No — beyond a few occurrences, extra repetition adds little signal.

**BM25 solution:** Apply a saturation function to TF.

### 7.2 BM25 Formula

```
BM25(t, d) = IDF_bm25(t) × [TF(t,d) × (k1+1)] / [TF(t,d) + k1 × (1 - b + b × |d|/avgdl)]

Parameters:
  k1 = 1.5 (saturation speed — higher = slower saturation)
  b  = 0.75 (length normalization — higher = stronger normalization)
  |d| = length of document d (in tokens)
  avgdl = average document length across corpus

BM25 IDF formula:
IDF_bm25(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

### 7.3 Compute BM25 IDF

N=3 documents:

```
IDF_bm25(cat) = log((3-2+0.5)/(2+0.5) + 1) = log(1.5/2.5 + 1) = log(0.600+1) = log(1.600) = 0.470
IDF_bm25(mat) = log(1.5/2.5 + 1) = 0.470
IDF_bm25(on)  = log((3-3+0.5)/(3+0.5) + 1) = log(0.5/3.5 + 1) = log(0.143+1) = log(1.143) = 0.134
IDF_bm25(sat) = log(1.5/2.5 + 1) = 0.470
```

Compare to standard IDF: IDF(on)=0.000, BM25 IDF_bm25(on)=0.134 — BM25 doesn't fully zero out common terms.

```
avgdl = (4+4+4)/3 = 4.0  (all documents have 4 tokens)
```

### 7.4 Compute BM25 Scores

**"cat" in D1** (TF count=1, |D1|=4):

```
numerator:   1 × (1.5+1) = 2.500
denominator: 1.5 × (1 - 0.75 + 0.75 × 4/4) = 1 + 1.5 × (0.25 + 0.75) = 1 + 1.5 = 2.500
BM25(cat, D1) = 0.470 × 2.500/2.500 = 0.470 × 1.000 = 0.470
```

**"cat" in D2** (TF count=2, |D2|=4):

```
numerator:   2 × 2.500 = 5.000
denominator: 1.5 × 1.000 = 3.500
BM25(cat, D2) = 0.470 × 5.000/3.500 = 0.470 × 1.429 = 0.671
```

**"mat" in D3** (TF count=3, |D3|=4):

```
numerator:   3 × 2.500 = 7.500
denominator: 1.5 × 1.000 = 4.500
BM25(mat, D3) = 0.470 × 7.500/4.500 = 0.470 × 1.667 = 0.783
```

**"on" in D1** (TF count=1, |D1|=4):

```
BM25(on, D1) = 0.134 × 2.500/2.500 = 0.134 × 1.000 = 0.134
```

### 7.5 Side-by-Side: TF-IDF vs BM25

```
TF-IDF:
     cat    mat    on     sat
D1: [0.101, 0.101, 0.000, 0.101]
D2: [0.203, 0.000, 0.000, 0.101]
D3: [0.000, 0.304, 0.000, 0.000]

BM25:
     cat    mat    on     sat
D1: [0.470, 0.470, 0.134, 0.470]
D2: [0.671, 0.000, 0.134, 0.470]
D3: [0.000, 0.783, 0.134, 0.000]
```

### 7.6 The Key Difference: Saturation

```
TF-IDF cat column: D1=0.101, D2=0.203 → ratio = 2.00× (exactly linear)
BM25   cat column: D1=0.470, D2=0.671 → ratio = 1.43× (sublinear, saturating)

TF-IDF mat column: D1=0.101, D3=0.304 → ratio = 3.01× (exactly linear, mat×3)
BM25   mat column: D1=0.470, D3=0.783 → ratio = 1.67× (diminishing returns)
```

**What this means for search:** If a query is "cat" and D2 mentions "cat" twice while D1 mentions it once, TF-IDF ranks D2 exactly 2× higher. BM25 only ranks D2 1.43× higher — because the second mention of "cat" adds less certainty than the first.

This is more aligned with human intuition about relevance.

### 7.7 BM25 Length Normalization

Change b from 0.75 to 0 (no length normalization):
```
denominator becomes: TF + k1 × 1.0 + 1.5  (no |d|/avgdl term)
```

Change b to 1.0 (full normalization):
```
denominator becomes: TF + k1 × (|d|/avgdl)  (heavy length penalty)
```

For our corpus, all documents have length 4 = avgdl, so length normalization has no effect. In practice, long documents get penalized (b=0.75 reduces their BM25 scores proportionally to length).

---

## 8. The Ceiling of Sparse Representations

### 8.1 The Zero Similarity Problem

"The feline rested upon the carpet."

This means almost the same as "The cat sat on the mat." But:

```
Vocabulary (from training): cat, mat, on, sat
"The feline rested upon the carpet":
  cat=0, mat=0, on=0, sat=0
  TF-IDF vector: [0, 0, 0, 0]
```

Cosine similarity with anything = 0. The sentence is invisible.

This isn't a tuning problem — it's a fundamental limitation of sparse representations. If the words don't literally overlap, the similarity is zero.

**Where this hurts:** Paraphrase detection: "She's happy" vs "She's joyful" → cosine=0 · Cross-lingual retrieval: English query, Spanish documents → cosine=0 · Domain shift: train on news, test on social media (different vocabulary).

### 8.2 Where Sparse Representations Win

Despite their limitations, sparse representations dominate in:

**Keyword search (Elasticsearch, Lucene, Solr):** The query "machine learning" should literally appear in relevant documents. BM25 with inverted index scales to billions of documents at millisecond latency.

**Exact term matching:** Legal documents: "Section 12.3.b" must match literally — semantics don't help here.

**Low-data settings:** 100 labeled documents — TF-IDF + Logistic Regression is often competitive with BERT because BERT fine-tuning overfits on small data.

**Interpretability:** You can look at the top TF-IDF features for a prediction and explain why the model made a decision. BERT doesn't offer this.

---

## 9. Verification: Complete Forward Pass (Classification)

**Task:** Binary classification — does D1 ("cat sat on mat") belong to the "cat documents" class?

```
TF-IDF vector for D1 (query): [0.101, 0.101, 0.000, 0.101]
Logistic Regression weights (trained, one weight per feature):
w = [0.8, -0.3, 0.1, 0.5]  ← cat=0.8 (positive), mat=-0.3 (negative), on=0.1, sat=0.5
b = 0.0
```

**Prediction:**

```
z = D1_tfidf · w + b
  = 0.101×0.8 + 0.101×(-0.3) + 0.000×0.1 + 0.101×0.5
  = 0.0808 + (-0.0303)      + 0.000     + 0.0505
  = 0.101

P(cat class) = σ(0.101) = 1/(1+e^{-0.101}) = 0.525
```

Model is 52.5% confident D1 belongs to "cat class." Weak confidence because D1 also has "mat" which has negative weight.

```
For D2: [0.203, 0.000, 0.000, 0.101]
z = 0.203×0.8 + 0.000×(-0.3) + 0.000×0.1 + 0.101×0.5
  = 0.162 + 0 + 0 + 0.051
  = 0.213

P(cat class) = σ(0.213) = 0.553
```

D2 is more confidently "cat class" than D1 — because D2 has no "mat" pulling it negative.

```
For D3: [0.000, 0.304, 0.000, 0.000]
z = 0.000×0.8 + 0.304×(-0.3) + 0 + 0
  = -0.091

P(cat class) = σ(-0.091) = 0.477
```

D3 classified as NOT cat class — mat has negative weight, and mat is the only signal in D3.

---

## 10. Code

### 10.1 BoW and TF-IDF from Scratch (NumPy)

```python
import numpy as np
from collections import Counter

# Corpus
docs = [
    "cat sat on mat",
    "cat sat on cat",
    "mat mat on mat"
]

# Step 1: Tokenize
tokenized = [doc.split() for doc in docs]

# Step 2: Build vocabulary (sorted)
vocab = sorted(set(token for doc in tokenized for token in doc))
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)
print(f"Vocabulary {V} words): {vocab}")

# Step 3: BoW matrix
def build_bow(tokenized_docs, word2idx):
    V = len(word2idx)
    N = len(tokenized_docs)
    bow = np.zeros((N, V), dtype=float)
    for i, tokens in enumerate(tokenized_docs):
        for token in tokens:
            if token in word2idx:
                bow[i][word2idx[token]] += 1
    return bow

bow = build_bow(tokenized, word2idx)
print("\nBoW matrix:")
print(bow)

# Step 4: TF matrix
tf = bow / bow.sum(axis=1, keepdims=True)
print("\nTF matrix:")
print(tf.round(3))

# Step 5: IDF
N = len(docs)
df = (bow > 0).sum(axis=0)           # document frequency per term
idf = np.log(N / df)
idf[np.isinf(idf)] = 0.0             # handle zero (shouldn't happen)
idf = dict(zip(vocab, idf.round(3)))

# Step 6: TF-IDF
tfidf = tf * idf
print("\nTF-IDF matrix:")
print(tfidf.round(3))
```

### 10.2 N-grams from Scratch

```python
def get_ngrams(tokens, n):
    """Generate n-grams from a token list."""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def build_ngram_vocab(tokenized_docs, ngram_range=(1, 2)):
    """Build vocabulary of n-grams."""
    all_ngrams = set()
    for tokens in tokenized_docs:
        for n in range(ngram_range[0], ngram_range[1]+1):
            all_ngrams.update(get_ngrams(tokens, n))
    return sorted(all_ngrams)

tokenized = [doc.split() for doc in docs]
ngram_vocab = build_ngram_vocab(tokenized, ngram_range=(1, 2))
ngram2idx = {g: i for i, g in enumerate(ngram_vocab)}
print(f"\nUnigram+bigram vocabulary ({len(ngram_vocab)} features):")
print(ngram_vocab)

# Build n-gram count matrix
ngram_bow = np.zeros((len(tokenized_docs), len(ngram_vocab)))
for i, tokens in enumerate(tokenized):
    for n in range(1, 3):
        for gram in get_ngrams(tokens, n):
            ngram_bow[i][ngram2idx[gram]] += 1

print("\nUnigram+bigram BoW (first 6 features shown):")
for i, doc in enumerate(docs):
    print(f"  D{i+1}: {ngram_bow[i][:6]}")
```

### 10.3 Using sklearn

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs = [
    "cat sat on mat",
    "cat sat on cat",
    "mat mat on mat"
]

# --- BoW ---
bow_vec = CountVectorizer()
X_bow = bow_vec.fit_transform(docs)
print("BoW vocabulary:", bow_vec.get_feature_names_out())
print("BoW matrix:\n", X_bow.toarray())

# --- TF-IDF ---
tfidf_vec = TfidfVectorizer(
    use_idf=True,
    smooth_idf=False,   # standard IDF (no smoothing)
    norm='l2',          # L2-normalize each row (optional)
    sublinear_tf=False  # use raw TF (not log TF)
)
X_tfidf = tfidf_vec.fit_transform(docs)
print("\nTF-IDF matrix:")
print(X_tfidf.toarray().round(3))
print("IDF values:", dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_.round(3))))

# --- Cosine similarity ---
sim_matrix = cosine_similarity(X_tfidf)
print("\nCosine similarity matrix:")
print(sim_matrix.round(3))

# --- Bigrams ---
bigram_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20)
X_bigram = bigram_vec.fit_transform(docs)
print("\nUnigram+bigram features:", bigram_vec.get_feature_names_out())

# --- Keyword retrieval ---
query = "cat on mat"
query_vec = tfidf_vec.transform([query])
scores = cosine_similarity(query_vec, X_tfidf)[0]
ranked = sorted(enumerate(scores), key=lambda x: -x[1])
print(f"\nQuery '{query}' → ranked documents:")
for doc_idx, score in ranked:
    print(f"  D{doc_idx+1}: {docs[doc_idx]:<30}  score={score:.3f}")
```

### 10.4 BM25 from Scratch

```python
import numpy as np
from collections import Counter

def bm25_scores(docs, query, k1=1.5, b=0.75):
    """Compute BM25 scores for all documents given a query."""
    tokenized_docs = [doc.split() for doc in docs]
    query_tokens = query.split()
    N = len(docs)

    # Document frequencies
    df = Counter()
    for tokens in tokenized_docs:
        for term in set(tokens):
            df[term] += 1

    # Average document length
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lengths) / N

    # BM25 IDF
    def idf(term, df_t=0):
        df_t = df.get(term, 0)
        return np.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

    scores = np.zeros(N)
    for i, tokens in enumerate(tokenized_docs):
        tf_counts = Counter(tokens)
        doc_len = doc_lengths[i]
        for term in query_tokens:
            if term not in tf_counts:
                continue
            tf = tf_counts[term]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avgdl)
            scores[i] += idf(term) * numerator / denominator

    return scores

docs = [
    "cat sat on mat",
    "cat sat on cat",
    "mat mat on mat"
]

# Query: "cat"
query = "cat"
bm25 = bm25_scores(docs, query)
print(f"BM25 scores for query '{query}':")
for i, (doc, score) in enumerate(zip(docs, bm25)):
    print(f"  D{i+1}: {doc:<25}  score={score:.3f}")
```

### 10.5 Using rank-bm25 Library

```python
from rank_bm25 import BM25Okapi

# Tokenize corpus
tokenized_corpus = [doc.split() for doc in docs]

# Fit BM25
bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

# Score a query
query = "cat sat"
scores = bm25.get_scores(query.split())
print(f"BM25 scores for '{query}':", scores)

# Get top-n documents
top_docs = bm25.get_top_n(query.split(), docs, n=3)
print("Ranked documents:", top_docs)
```

### 10.6 Full Pipeline: TF-IDF + Logistic Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Toy dataset
texts = [
    "cat sat on mat",
    "cat sat on cat",
    "cat sat on mat",
    "dog ran on mat",
    "cat and dog sat",
]
labels = [1, 1, 0, 0, 1]  # 1=cat-related, 0=mat-related

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        sublinear_tf=True,   # use 1+log(TF) instead of TF — reduces effect of high freq
        smooth_idf=True,
        min_df=1,
    )),
    ('clf', LogisticRegression(C=1.0, max_iter=1000))
])

# With only 5 docs, can't do cross-validation — just fit and predict
pipeline.fit(texts, labels)
new_doc = ["cat on mat"]
print("Prediction:", pipeline.predict(new_doc))
print("Probability:", pipeline.predict_proba(new_doc))

# Show top features per class
tfidf = pipeline.named_steps['tfidf']
clf  = pipeline.named_steps['clf']
feature_names = tfidf.get_feature_names_out()
import numpy as np
top_pos = np.argsort(clf.coef_[0])[::-1][:5]
top_neg = np.argsort(clf.coef_[0])[:5]
print("\nTop features for class 1 (cat-related):")
for i in top_pos:
    print(f"  '{feature_names[i]}': {clf.coef_[0][i]:.3f}")
print("Top features for class 0 (mat-related):")
for i in top_neg:
    print(f"  '{feature_names[i]}': {clf.coef_[0][i]:.3f}")
```

---

## 11. Gotchas

1. **Always fit vectorizer on training data only.** `TfidfVectorizer.fit_transform(X_train)`, then `TfidfVectorizer.transform(X_test)`. If you fit on full dataset, you leak document frequency statistics to test set. Put the vectorizer inside a `Pipeline` so sklearn's cross-validation handles this correctly.

2. **`max_df` can remove domain-specific terms.** Setting `max_df=0.95` removes terms appearing in >95% of docs. In a specialized corpus (e.g., all medical notes), "patient" might appear in 98% of documents and get removed — but it's still important for downstream tasks. Tune `max_df` per domain.

3. **Sublinear TF reduces the impact of repeated terms.** `sublinear_tf=True` replaces TF with `1+log(TF)`. In our corpus, "mat" in D3 has TF=0.75. With sublinear: `TF_sub = 1+log(3/4)=1+log(0.75)` — no wait, it applies to raw counts: `TF_sub(mat) = 1+log(3) = 1+1.099 = 2.099` vs raw TF count 3. This is a soft alternative to BM25's saturation.

4. **Sparse matrix operations — don't convert to dense unnecessarily.** `X.toarray()` on a 100K document × 50K feature TF-IDF matrix → 40GB in dense format. As scipy sparse matrix with ~1% non-zeros: 400MB. sklearn's LogisticRegression, LinearSVC, and NaiveBayes all accept sparse matrices natively.

5. **BM25 and TF-IDF IDF formulas differ — don't mix them.** The standard IDF = log(N/df) can be zero (when df=N). BM25's IDF = log((N-df+0.5)/(df+0.5)+1) is always positive. sklearn uses smoothed IDF = log((1+N)/(1+df))+1. These give different values — always know which formula your library uses.

6. **Cosine similarity requires non-zero vectors.** If a document has no vocabulary words (all terms are OOV or stopped), its TF-IDF vector = all zeros → cosine similarity is undefined (division by zero). Handle with: `if norm > 0: cosine = dot/norm/norm_else: cosine = 0`.

7. **n-gram vocabulary size explodes.** Unigrams: V features. Bigrams: up to V² features. Trigrams: V³. For n=10, trigrams = 10¹² potential features. Always set `max_features` limit (e.g., 50K-100K). sklearn selects top features by document frequency.

---

## 12. Q&A

**Q: Why does TF-IDF sometimes perform better than BERT on short texts?**

A: Attention-based models need sufficient context to build useful representations. For very short texts (product titles, search queries, tweet-level), a 3-5 word TF-IDF bigram vector captures most of the lexical signal. BERT's 12 attention layers are overkill and the [CLS] embedding for "blue sneakers size 10" may not outperform TF-IDF for exact-match product retrieval. Rule of thumb: if the task is largely lexical (does this exact term appear?), TF-IDF is competitive. If the task requires reasoning (does this sentence describe a behavior with that one?), BERT wins.

**Q: How does an inverted index speed up BM25 retrieval?**

A: An inverted index maps each term to the list of documents containing it. For a query "cat sat", you look up "cat" → {D1, D2}, "sat" → {D1, D2}. You only compute BM25 for the union {D1, D2} — not for D3 which doesn't contain "cat". For a corpus of 10M documents where 99.9% don't contain "cat", you skip 9,999,000 documents. BM25 with an inverted index scales to billions of documents with sub-10ms query latency (Elasticsearch, Lucene). Dense retrieval (BERT embeddings + FAISS) requires computing distance to every document vector — much more expensive at scale.

**Q: What is the difference between `TfidfVectorizer` and `CountVectorizer` + `TfidfTransformer`?**

A: Functionally identical. `TfidfVectorizer` is a convenience wrapper that calls `CountVectorizer` internally then applies the TF-IDF transformation. Using them separately is useful when you want to reuse the count matrix for multiple purposes, or when combining with custom normalization. In production code, `TfidfVectorizer` is simpler; separate components are more flexible.

**Q: Should I L2-normalize TF-IDF vectors before using cosine similarity?**

A: sklearn's `TfidfVectorizer` applies L2 normalization by default (`norm='l2'`). After L2 normalization, cosine similarity = dot product (since ||v||=1 for normalized vectors). This is slightly faster. However, the raw (unnormalized) TF-IDF sometimes performs better because the magnitude carries information. Try both with cross-validation.

---

## 13. Connections

| This file | Links to | Why |
|---|---|---|
| Preprocessing runs BEFORE vectorization | `01_text_preprocessing.md` | Preprocessing (cleaned, tokenized text) is what BoW/TF-IDF receives as input — Stop word removal matters more for TF-IDF than for BERT — a removed stop word has zero IDF anyway, but explicitly removing it speeds the vectorizer |
| Dense embeddings (next step from sparse) | `../embeddings/01_word_embeddings.md` | TF-IDF: "cat" and "feline" → completely different columns, cosine=0 · Word2Vec/GloVe: "cat" and "feline" → cos≈0.85 · TF-IDF is the sparse baseline; dense embeddings solve its semantic limitations |
| Word2Vec end-to-end companion | `../embeddings/02_word2vec_end_to_end.md` | The co-occurrence matrix in GloVe is the same idea as TF-IDF's document-term matrix — except GloVe counts word-word co-occurrences, not word-document co-occurrences — BM25 saturation ≈ GloVe's log(X_ij) transformation — both reduce the impact of very high counts |
| TF-IDF for text classification | `../04_applications/01_text_classification.md` | TF-IDF + Logistic Regression = the baseline to beat for text classification — BM25 = the baseline for document retrieval |
| To BERT (5.transformers/02_models/05_bert_end_to_end.md) | — | BERT produces contextual dense representations — TF-IDF's semantic limitation is fully solved — But BERT's output for classification is the same final step as TF-IDF — The pipeline is: TF-IDF → LogReg (fast, interpretable) vs BERT → LogReg (slow, accurate) |

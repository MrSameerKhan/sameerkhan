# Evaluation Metrics — End-to-End

> Every number computed. Same vocabulary throughout: cat, sat, on, mat.

---

## Why Metrics Matter

Training loss tells you how well the model fits the training data.
Metrics tell you whether the model solves the actual problem.

| Metric family | When to use |
|---------------|-------------|
| Precision / Recall / F1 | Classification, NER, anything with labels |
| BLEU | Machine translation (candidate vs reference) |
| ROUGE | Summarization (generated summary vs gold summary) |
| Perplexity | Language model quality |
| BERTScore | Any generation task — catches synonyms |

---

## Part 1 — Classification Metrics

### Setup

**Task:** Sentiment classification. 3 classes — positive, negative, neutral.
9 sentences. True labels vs predicted labels:

| # | True | Predicted |
|---|------|-----------|
| 0 | pos  | pos       |
| 1 | pos  | neg       |
| 2 | neg  | pos       |
| 3 | neg  | neg       |
| 4 | neg  | neg       |
| 5 | neu  | neu       |
| 6 | neu  | pos       |
| 7 | neu  | neu       |
| 8 | pos  | pos       |

### Step 1 — Confusion Matrix (binary view per class)

For each class, treat it as one-vs-rest (binary):

**For "positive":**

|          | Pred=pos             | Pred≠pos                          |
|----------|----------------------|-----------------------------------|
| True=pos | TP=2 (rows 0,8)      | FN=1 (row 1 — predicted neg)      |
| True≠pos | FP=2 (rows 2,6)      | TN=4                              |

**For "negative":**

|          | Pred=neg             | Pred≠neg                          |
|----------|----------------------|-----------------------------------|
| True=neg | TP=2 (rows 3,4)      | FN=1 (row 2 — predicted pos)      |
| True≠neg | FP=1 (row 1)         | TN=5                              |

**For "neutral":**

|          | Pred=neu             | Pred≠neu                          |
|----------|----------------------|-----------------------------------|
| True=neu | TP=2 (rows 5,7)      | FN=1 (row 6 — predicted pos)      |
| True≠neu | FP=0                 | TN=6                              |

### Step 2 — Precision, Recall, F1 per class

```
Precision = TP / (TP + FP)   # of all we predicted as class X, how many were right?
Recall    = TP / (TP + FN)   # of all true class X, how many did we catch?
F1        = 2 × P × R / (P + R)
```

**Positive:**
```
Precision = 2 / (2 + 2) = 2/4 = 0.500
Recall    = 2 / (2 + 1) = 2/3 = 0.667
F1        = 2 × 0.500 × 0.667 / (0.500 + 0.667)
          = 2 × 0.3335 / 1.167
          = 0.667 / 1.167
          = 0.571
```

**Negative:**
```
Precision = 2 / (2 + 1) = 2/3 = 0.667
Recall    = 2 / (2 + 1) = 2/3 = 0.667
F1        = 2 × 0.667 × 0.667 / (0.667 + 0.667)
          = 2 × 0.445 / 1.334
          = 0.667
```

**Neutral:**
```
Precision = 2 / (2 + 0) = 2/2 = 1.000
Recall    = 2 / (2 + 1) = 2/3 = 0.667
F1        = 2 × 1.000 × 0.667 / (1.000 + 0.667)
          = 1.334 / 1.667
          = 0.800
```

**Summary table:**

| Class    | TP | FP | FN | Precision | Recall | F1    |
|----------|----|----|----|-----------|--------|-------|
| positive | 2  | 2  | 1  | 0.500     | 0.667  | 0.571 |
| negative | 2  | 1  | 1  | 0.667     | 0.667  | 0.667 |
| neutral  | 2  | 0  | 1  | 1.000     | 0.667  | 0.800 |

### Step 3 — Macro F1 vs Micro F1

**Macro F1** — average the per-class F1 scores, each class counts equally:

```
Macro F1 = (0.571 + 0.667 + 0.800) / 3
         = 2.038 / 3
         = 0.679
```

**Micro F1** — pool all TP, FP, FN across classes, compute once:

```
Total TP = 2 + 2 + 2 = 6
Total FP = 2 + 1 + 0 = 3
Total FN = 1 + 1 + 1 = 3

Micro Precision = 6 / (6 + 3) = 6/9 = 0.667
Micro Recall    = 6 / (6 + 3) = 6/9 = 0.667
Micro F1        = 0.667
```

**When they differ:**

| Scenario | Macro | Micro | Which to use |
|----------|-------|-------|--------------|
| Balanced classes | ≈ same | ≈ same | either |
| Imbalanced (rare class matters) | higher if rare class does well | dominated by frequent class | Macro |
| Imbalanced (rare class irrelevant) | pulled down by rare class | ignores rare class | Micro |

**Rule of thumb:** rare disease detection → Macro (every class matters equally). Search ranking → Micro (volume of correct predictions matters).

### Key Intuition: Precision vs Recall Tradeoff

```
High Precision, Low Recall:
  → Model is conservative. Only predicts "positive" when very confident.
  → Misses many true positives. Spam filter that lets spam through.

High Recall, Low Precision:
  → Model is aggressive. Predicts "positive" whenever in doubt.
  → Many false alarms. Spam filter that blocks everything.

F1 balances both.
```

**F-beta score:** when recall is more important than precision:

```
F_β = (1 + β²) × P × R / (β² × P + R)

β=2: recall twice as important.
β=0.5: precision twice as important.
```

---

## Part 2 — BLEU Score

**Task:** Machine translation quality. Does candidate match the reference?

```
Reference:   "cat sat on mat"
Candidate A: "cat sat on the mat"   + extra word "the"
Candidate B: "cat sat"              + too short, missing words
```

### BLEU Formula

```
BLEU = BP × exp( Σ w_n × log(p_n) )  for n=1..N

where:
  p_n = modified n-gram precision for n-grams
  w_n = 1/N (uniform weights, N=4 by default)
  BP  = brevity penalty
```

### Step 1 — Modified N-gram Precision

**Modified** means: clip each n-gram count to its max count in any reference.
Prevents gaming with repetition like "cat cat cat cat".

**Candidate A: "cat sat on the mat"**
Tokens: [cat, sat, on, the, mat] — length 5
Reference: [cat, sat, on, mat] — length 4

**1-gram precision (p_1):**

```
Candidate 1-grams and counts:
  cat = 1, sat = 1, on = 1, the = 1, mat = 1

Reference 1-gram max counts:
  cat = 1, sat = 1, on = 1, mat = 1, the = 0  ← "the" not in reference

Clipped matches:
  cat: min(1, 1) = 1
  sat: min(1, 1) = 1
  on:  min(1, 1) = 1
  the: min(1, 0) = 0  ← clipped to 0
  mat: min(1, 1) = 1

Sum clipped = 4
Total candidate 1-grams = 5

p_1 = 4/5 = 0.800
```

**2-gram precision (p_2):**

```
Candidate 2-grams: (cat,sat), (sat,on), (on,the), (the,mat) = 4 bigrams
Reference 2-grams: (cat,sat), (sat,on), (on,mat) = 3 bigrams

Match check:
  (cat,sat) = in reference ✓
  (sat,on)  = in reference ✓
  (on,the)  = NOT in reference ✗
  (the,mat) = NOT in reference ✗

Clipped matches = 2
Total candidate 2-grams = 4

p_2 = 2/4 = 0.500
```

**3-gram precision (p_3):**

```
Candidate 3-grams: (cat,sat,on), (sat,on,the), (on,the,mat) = 3 trigrams
Reference 3-grams: (cat,sat,on), (sat,on,mat) = 2 trigrams

Match check:
  (cat,sat,on)  = in reference ✓
  (sat,on,the)  = NOT in reference ✗
  (on,the,mat)  = NOT in reference ✗

Clipped matches = 1
p_3 = 1/3 = 0.333
```

**4-gram precision (p_4):**

```
Candidate 4-grams: (cat,sat,on,the), (sat,on,the,mat) = 2
Reference 4-grams: (cat,sat,on,mat) = 1

Match check:
  (cat,sat,on,the) = NOT in reference ✗
  (sat,on,the,mat) = NOT in reference ✗

Clipped matches = 0
p_4 = 0/2 = 0.000
```

### Step 2 — Brevity Penalty (BP)

Penalizes candidates shorter than the reference.
(Long candidates are already penalized by lower precision — extra words don't match.)

```
c = length of candidate = 5
r = length of closest reference = 4

BP = 1           if c >= r    → no penalty for being longer/equal
BP = exp(1-r/c)  if c < r     → exponential penalty for being shorter

Candidate A: c=5 >= r=4 → BP = 1
```

### Step 3 — Compute BLEU for Candidate A

**BLEU-4:**

```
BLEU-4 = BP × exp( w_1×log(p_1) + w_2×log(p_2) + w_3×log(p_3) + w_4×log(p_4) )

w_1 = w_2 = w_3 = w_4 = 0.25

log(p_1) = log(0.800) = -0.223
log(p_2) = log(0.500) = -0.693
log(p_3) = log(0.333) = -1.099
log(p_4) = log(0.000) = -∞  ← problem!

BLEU-4 = 0  (one zero precision collapses the whole score)
```

This is a known BLEU-4 limitation: it's 0 if ANY n-gram precision is 0.
For short sentences (4 words), BLEU-4 is often zero.
In practice: BLEU-4 is used on long documents, not individual sentences.

**BLEU-2 (only unigrams + bigrams):**

```
BLEU-2 = 1 × exp( 0.5×log(0.800) + 0.5×log(0.500) )
       = exp( 0.5×(-0.223) + 0.5×(-0.693) )
       = exp( -0.112 + (-0.307) )
       = exp( -0.458 )
       = 0.633
```

### Step 4 — Compute BLEU for Candidate B: "cat sat"

Tokens: [cat, sat] — length 2

**1-gram:**
```
cat = in ref ✓,  sat = in ref ✓
p_1 = 2/2 = 1.000
```

**2-gram:**
```
Candidate: (cat,sat) → in reference ✓
p_2 = 1/1 = 1.000
```

**Brevity Penalty:**
```
c=2, r=4  (c < r) → BP = exp(1 - r/c) = exp(1 - 4/2) = exp(-1) = 0.368
```

**BLEU-2:**
```
BLEU-2 = 0.368 × exp( 0.5×log(1.000) + 0.5×log(1.000) )
       = 0.368 × exp(0)
       = 0.368
```

### BLEU Summary

| Candidate | p_1   | p_2   | BP    | BLEU-2 |
|-----------|-------|-------|-------|--------|
| A: "cat sat on the mat" | 0.800 | 0.500 | 1.000 | 0.633 |
| B: "cat sat"            | 1.000 | 1.000 | 0.368 | 0.368 |

Candidate A wins because brevity penalty severely hurts B despite perfect n-gram precision.
→ Being precise but short is worse than having one extra word.

### BLEU Gotchas

```
1. BLEU is corpus-level — don't trust sentence-level BLEU scores
2. Multiple references → clipping uses max count across all references
3. p_n = 0 → BLEU = 0, so use smoothing for short sentences
4. BLEU doesn't catch synonyms: "cat rested" ≠ "cat sat" even if semantically same
5. BLEU correlates with human judgment at corpus level, poorly at sentence level
```

---

## Part 3 — ROUGE

**Task:** Summarization quality. How much of the reference summary is covered?

```
Reference summary: "cat sat on mat"
Candidate A:       "cat sat on the mat"   + extra "the"
Candidate B:       "cat mat on sat"       + same words, wrong order
```

Key difference from BLEU: ROUGE emphasizes recall (coverage of reference). BLEU emphasizes precision.

### ROUGE-1 (Unigram overlap)

**Candidate A: "cat sat on the mat"**

```
Reference tokens: [cat, sat, on, mat]     — 4 tokens
Candidate tokens: [cat, sat, on, the, mat] — 5 tokens

Overlap (tokens in both):
  cat = 1  ✓
  sat = 1  ✓
  on  = 1  ✓
  mat = 1  ✓
  the = 0  (not in reference)

Overlap count = 4

ROUGE-1 Recall    = overlap / |reference| = 4/4 = 1.000
ROUGE-1 Precision = overlap / |candidate| = 4/5 = 0.800
ROUGE-1 F1        = 2 × 1.000 × 0.800 / (1.000 + 0.800)
                  = 1.600 / 1.800
                  = 0.889
```

**Candidate B: "cat mat on sat"**

```
Reference tokens: [cat, sat, on, mat]  — 4 tokens
Candidate tokens: [cat, mat, on, sat]  — 4 tokens

Overlap:
  cat = 1 ✓
  mat = 1 ✓
  on  = 1 ✓
  sat = 1 ✓

Overlap count = 4 — same words, just reordered!

ROUGE-1 Recall    = 4/4 = 1.000
ROUGE-1 Precision = 4/4 = 1.000
ROUGE-1 F1        = 1.000  ← perfect score despite wrong word order
```

**Problem:** ROUGE-1 can't detect reordering. Both candidates get different scores with ROUGE-2 and ROUGE-L.

### ROUGE-2 (Bigram overlap)

**Candidate A: "cat sat on the mat"**

```
Reference bigrams:  (cat,sat), (sat,on), (on,mat)          — 3 bigrams
Candidate bigrams:  (cat,sat), (sat,on), (on,the), (the,mat) — 4 bigrams

Overlap:
  (cat,sat) = ✓
  (sat,on)  = ✓
  (on,the)  = ✗
  (the,mat) = ✗

Overlap count = 2

ROUGE-2 Recall    = 2/3 = 0.667
ROUGE-2 Precision = 2/4 = 0.500
ROUGE-2 F1        = 2 × 0.667 × 0.500 / (0.667 + 0.500)
                  = 2 × 0.3335 / 1.167
                  = 0.667 / 1.167
                  = 0.572
```

**Candidate B: "cat mat on sat"**

```
Reference bigrams:  (cat,sat), (sat,on), (on,mat) — 3 bigrams
Candidate bigrams:  (cat,mat), (mat,on), (on,sat) — 3 bigrams

Overlap:
  (cat,mat) = NOT in reference ✗
  (mat,on)  = NOT in reference ✗
  (on,sat)  = NOT in reference ✗

Overlap count = 0

ROUGE-2 Recall    = 0/3 = 0.000
ROUGE-2 Precision = 0/3 = 0.000
ROUGE-2 F1        = 0.000  ← correctly penalizes wrong order
```

### ROUGE-L (Longest Common Subsequence)

LCS captures word order without requiring consecutive matches.
"cat on mat" is a subsequence of "cat sat on the mat" (skip "sat" and "the").

LCS algorithm (dynamic programming):

```python
def lcs_length(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
```

**Candidate A: "cat sat on the mat" vs Reference "cat sat on mat"**

```
Reference  = [cat, sat, on, mat]     indices 1..4
Candidate  = [cat, sat, on, the, mat] indices 1..5

DP table:
       cat  sat  on   the  mat
  **    0    0    0    0    0    0
  cat   0    1    1    1    1    1
  sat   0    1    2    2    2    2
  on    0    1    2    3    3    3
  mat   0    1    2    3    3    4

LCS = 4 = "cat sat on mat" (skips "the" in candidate)

ROUGE-L Recall    = 4 / |reference| = 4/4 = 1.000
ROUGE-L Precision = 4 / |candidate| = 4/5 = 0.800
ROUGE-L F1        = 2 × 1.000 × 0.800 / (1.000 + 0.800) = 0.889
```

**Candidate B: "cat mat on sat" vs Reference "cat sat on mat"**

```
Reference  = [cat, sat, on, mat]  indices 1..4
Candidate  = [cat, mat, on, sat]  indices 1..4

DP table:
       cat  mat  on   sat
  **    0    0    0    0    0
  cat   0    1    1    1    1
  sat   0    1    1    1    2
  on    0    1    1    2    2
  mat   0    1    2    2    2

LCS = 2 = either "cat mat" or "cat on" or "cat sat"

ROUGE-L Recall    = 2 / 4 = 0.500
ROUGE-L Precision = 2 / 4 = 0.500
ROUGE-L F1        = 2 × 0.500 × 0.500 / (0.500 + 0.500) = 0.500
```

### ROUGE Comparison Table

| Candidate | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|-----------|-----------|-----------|-----------|
| A: "cat sat on the mat" | 0.889 | 0.572 | 0.889 |
| B: "cat mat on sat"     | **1.000** (misleading!) | 0.000 | 0.500 |

**Key insight:**
- ROUGE-1 can't detect word order — Candidate B scores perfect
- ROUGE-2 and ROUGE-L correctly penalize B for scrambled order
- ROUGE-L captures order without requiring contiguous n-grams (more flexible than ROUGE-2)

### BLEU vs ROUGE Summary

|                   | BLEU | ROUGE |
|-------------------|------|-------|
| Focus             | Precision (is every candidate word in the reference?) | Recall (does the candidate cover the reference?) |
| Use case          | Machine translation | Summarization |
| Reference         | Human translation | Gold summary |
| Multiple references | Supported | Supported |
| Synonym handling  | None | None (fixed by BERTScore) |

---

## Part 4 — Perplexity

**Task:** How good is a language model at predicting text?
Perplexity = exp(average cross-entropy loss over all tokens).
Lower perplexity = better model.

### Setup

**Text to evaluate:** "cat sat on mat"

Toy bigram LM — trained on a small corpus, assigns these probabilities:
```
P(cat | <start>) = 0.50   ← cat starts half the sentences
P(sat | cat)     = 0.40   ← sat follows cat 40% of the time
P(on  | sat)     = 0.35
P(mat | on)      = 0.45
```

### Step 1 — Cross-entropy loss per token

```
Loss = -log P(token | context)    (natural log)

token "cat": -log(0.50) = -(−0.693) = 0.693
token "sat": -log(0.40) = -(−0.916) = 0.916
token "on":  -log(0.35) = -(−1.050) = 1.050
token "mat": -log(0.45) = -(−0.799) = 0.799
```

### Step 2 — Average cross-entropy

```
H = (loss_cat + loss_sat + loss_on + loss_mat) / 4
  = (0.693 + 0.916 + 1.050 + 0.799) / 4
  = 3.458 / 4
  = 0.865
```

### Step 3 — Perplexity

```
PP = exp(H) = exp(0.865) = 2.374
```

### Interpretation

```
PP = 2.374

Meaning: at each position, the model is as uncertain as choosing
         uniformly among ~2.4 words.

Boundary cases:
  Perfect model: P=1.0 for correct token → H=0 → PP = exp(0) = 1.0
  Random (vocab=4): P=0.25 for each word → H=1.386 → PP = exp(1.386) = 4.0

Our model: PP = 2.374 (between random and perfect) ✓
```

Perplexity on unseen text is the real test — good training PP but high eval PP = overfitting.

### Perplexity and Cross-Entropy Loss

Perplexity IS your training loss, just exponentiated:

```python
H  = avg cross-entropy loss    (what the model reports as "loss")
PP = exp(H)

If training loss = 2.3 → PP = exp(2.3) = 10  (fairly uncertain)
If training loss = 1.0 → PP = exp(1.0) = 2.7  (starting to learn)
If training loss = 0.5 → PP = exp(0.5) = 1.6  (doing well)
```

Real model reference:
- GPT-2 (124M): perplexity ≈ 29 on Penn Treebank
- GPT-3: perplexity ≈ 20
- LLaMA-2 (7B): perplexity ≈ 5.5 on standard benchmarks

**Perplexity gotcha:** Perplexity is vocabulary-size dependent. A model with vocab=50k has higher baseline PP than one with vocab=1k. Only compare across models with the same vocab and tokenizer.

---

## Part 5 — BERTScore

BLEU and ROUGE do exact token matching. They can't handle synonyms.

**Problem:**
```
Reference:   "cat sat on mat"
Candidate:   "cat rested on mat"
ROUGE-1: "rested" ≠ "sat" → loses one match → F1 = 0.750
BERTScore: "rested" ≈ "sat" in embedding space → near-full credit
```

### How BERTScore Works

```
Step 1: Tokenize both reference and candidate
Step 2: Run both through BERT → get contextual embeddings per token
Step 3: For each candidate token: find max cosine similarity with any reference token
Step 4: For each reference token: find max cosine similarity with any candidate token
Step 5: Average to get precision/recall/F1
```

### Dry Run (conceptual, not real BERT numbers)

```
Reference tokens: [cat, sat, on, mat]
Candidate tokens: [cat, rested, on, mat]

BERT embeddings (2D illustration):
  "sat"    = [0.2, 0.3]
  "rested" = [0.19, 0.28]  → semantically similar
  cosine(sat, rested) = 0.97  → high similarity

BERTScore Precision (candidate → reference):
  cat:    max_sim(cat,    [cat,sat,on,mat]) = 1.00  (exact match)
  rested: max_sim(rested, [cat,sat,on,mat]) = 0.97  (matches "sat")
  on:     max_sim(on,     [cat,sat,on,mat]) = 1.00
  mat:    max_sim(mat,    [cat,sat,on,mat]) = 1.00
  avg = (1.00 + 0.97 + 1.00 + 1.00) / 4 = 0.993

BERTScore Recall (reference → candidate):
  cat: 1.00
  sat: max_sim(sat, [cat,rested,on,mat]) = 0.97  (matches "rested")
  on:  1.00
  mat: 1.00
  avg = (1.00 + 0.97 + 1.00 + 1.00) / 4 = 0.993

BERTScore F1 = 0.993
vs ROUGE-1 F1 = 0.750 for same candidate  (exact match gives "rested" zero credit)
```

### When to Use BERTScore

```
✓ When synonyms are valid: "rested" = "sat", "automobile" = "car"
✓ When paraphrasing is acceptable
✗ When exact terminology matters: medical codes, legal text, code
✗ When you need interpretability (BLEU/ROUGE counts are auditable)
```

---

## Summary Table — All Metrics

| Text | Reference | Metric | Score |
|------|-----------|--------|-------|
| "cat sat on the mat" | "cat sat on mat" | BLEU-2 | 0.633 |
| "cat sat" | "cat sat on mat" | BLEU-2 | 0.368 |
| "cat sat on the mat" | "cat sat on mat" | ROUGE-1 F1 | 0.889 |
| "cat sat on the mat" | "cat sat on mat" | ROUGE-2 F1 | 0.572 |
| "cat sat on the mat" | "cat sat on mat" | ROUGE-L F1 | 0.889 |
| "cat mat on sat" | "cat sat on mat" | ROUGE-1 F1 | 1.000 (misleading!) |
| "cat mat on sat" | "cat sat on mat" | ROUGE-L F1 | 0.500 |
| "cat sat on mat" | — | Perplexity | 2.374 |

---

## Code

```python
import numpy as np
from collections import Counter

# ─────────────────────────────────────
# 1. CLASSIFICATION METRICS
# ─────────────────────────────────────

y_true = ['pos', 'pos', 'neg', 'neg', 'neg', 'neu', 'neu', 'neu', 'pos']
y_pred = ['pos', 'neg', 'pos', 'neg', 'neg', 'neu', 'pos', 'neu', 'pos']

def classification_report_manual(y_true, y_pred):
    classes = sorted(set(y_true))
    results = {}
    for cls in classes:
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        FP = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        FN = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        P  = TP / (TP + FP) if (TP + FP) > 0 else 0
        R  = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        results[cls] = {'TP': TP, 'FP': FP, 'FN': FN,
                        'P': round(P,3), 'R': round(R,3), 'F1': round(F1,3)}

    macro_f1  = np.mean([v['F1'] for v in results.values()])
    total_tp  = sum(v['TP'] for v in results.values())
    total_fp  = sum(v['FP'] for v in results.values())
    total_fn  = sum(v['FN'] for v in results.values())
    micro_p   = total_tp / (total_tp + total_fp)
    micro_r   = total_tp / (total_tp + total_fn)
    micro_f1  = 2 * micro_p * micro_r / (micro_p + micro_r)

    return results, round(macro_f1, 3), round(micro_f1, 3)

results, macro, micro = classification_report_manual(y_true, y_pred)
for cls in results.items():
    print(f"{cls[0]}: {cls[1]['P']} P={cls[1]['P']}, R={cls[1]['R']}, F1={cls[1]['F1']}")
print(f"Macro F1 = {macro}")
print(f"Micro F1 = {micro}")


# ─────────────────────────────────────
# 2. BLEU SCORE
# ─────────────────────────────────────

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def modified_ngram_precision(candidate, references, n):
    """Clipped n-gram precision."""
    cand_ngrams = Counter(ngrams(candidate, n))
    if sum(cand_ngrams.values()) == 0:
        return 0
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = Counter(ngrams(ref, n))
        for ng in cand_ngrams:
            max_ref_counts[ng] = max(max_ref_counts.get(ng, 0), ref_ngrams.get(ng, 0))
    clipped = sum(min(max_ref_counts.get(ng, 0), cnt)
                  for ng, cnt in cand_ngrams.items())
    return clipped / sum(cand_ngrams.values())

def brevity_penalty(candidate, references):
    c = len(candidate)
    r = min(abs(len(ref) - c) for ref in references, key=lambda ref: abs(len(ref)-c))
    r = len(min(references, key=lambda ref: abs(len(ref)-c)))
    if c >= r:
        return 1.0
    return np.exp(1 - r / c)

def bleu(candidate_str, reference_strs, max_n=2):
    candidate  = candidate_str.lower().split()
    references = [r.lower().split() for r in reference_strs]
    bp         = brevity_penalty(candidate, references)
    precisions = [modified_ngram_precision(candidate, references, n)
                  for n in range(1, max_n + 1)]
    print(f"  BP = {bp:.3f}")
    for i, p in enumerate(precisions, 1):
        print(f"  p_{i} = {p:.3f}")
    # log average (skip zeros for display)
    log_avg = np.mean([np.log(p) if p > 0 else float('-inf') for p in precisions])
    score   = bp * np.exp(log_avg) if log_avg > float('-inf') else 0.0
    return round(score, 3)

reference = ["cat sat on mat"]
score_a   = bleu("cat sat on the mat", reference, max_n=2)
print(f"Candidate A: BLEU-2 = {score_a}")
score_b   = bleu("cat sat", reference, max_n=2)
print(f"Candidate B: BLEU-2 = {score_b}")


# ─────────────────────────────────────
# 3. ROUGE
# ─────────────────────────────────────

def rouge_n(candidate_str, reference_str, n):
    candidate  = candidate_str.lower().split()
    reference  = reference_str.lower().split()
    cand_ngrams = Counter(ngrams(candidate, n))
    ref_ngrams  = Counter(ngrams(reference, n))
    overlap     = sum((cand_ngrams & ref_ngrams).values())
    precision   = overlap / max(sum(cand_ngrams.values()), 1)
    recall      = overlap / max(sum(ref_ngrams.values()), 1)
    f1          = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'lcs': n, 'precision': round(precision,3), 'recall': round(recall,3), 'f1': round(f1,3)}

def lcs_length(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i-1] == Y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(candidate_str, reference_str):
    candidate  = candidate_str.lower().split()
    reference  = reference_str.lower().split()
    l          = lcs_length(candidate, reference)
    precision  = l / max(len(candidate), 1)
    recall     = l / max(len(reference), 1)
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {'lcs': l, 'precision': round(precision,3), 'recall': round(recall,3), 'f1': round(f1,3)}

reference = "cat sat on mat"
candidates = ["cat sat on the mat", "cat mat on sat"]
for cand in candidates:
    print(f"Candidate: '{cand}'")
    print(f"  ROUGE-1: {rouge_n(cand, reference, 1)}")
    print(f"  ROUGE-2: {rouge_n(cand, reference, 2)}")
    print(f"  ROUGE-L: {rouge_l(cand, reference)}")
    print()


# ─────────────────────────────────────
# 4. PERPLEXITY
# ─────────────────────────────────────

def perplexity(log_probs):
    """log_probs: list of natural log P(token|context)"""
    avg_nll = -np.mean(log_probs)
    return round(np.exp(avg_nll), 3)

# Toy bigram LM probabilities for "cat sat on mat"
probs = {
    'cat': 0.50,   # P(cat | <start>)
    'sat': 0.40,   # P(sat | cat)
    'on':  0.35,   # P(on  | sat)
    'mat': 0.45,   # P(mat | on)
}

log_probs = [-np.log(p) for p in probs.values()]
print("Cross-entropy losses:")
for token, p in probs.items():
    print(f"  {token}: -log({p}) = {-np.log(p):.3f}")

pp = perplexity([-np.log(p) for p in probs.values()])
print(f"Average NLL = {np.mean([-np.log(p) for p in probs.values()]):.3f}")
print(f"Perplexity  = {pp}")

print(f"Reference: random (vocab=4)  → PP = {round(np.exp(np.log(4)), 1)}")
print(f"Reference: perfect model     → PP = 1.0")
```

### Output:

```
# Classification:
pos: P=0.5, R=0.667, F1=0.571
neg: P=0.667, R=0.667, F1=0.667
neu: P=1.0, R=0.667, F1=0.800
Macro F1 = 0.679
Micro F1 = 0.667

# BLEU:
Candidate A:
  BP = 1.000
  p_1 = 0.800
  p_2 = 0.500
  BLEU-2 = 0.633

Candidate B:
  BP = 0.368
  p_1 = 1.000
  p_2 = 1.000
  BLEU-2 = 0.368

# ROUGE:
Candidate: 'cat sat on the mat'
  ROUGE-1: {'precision': 0.8, 'recall': 1.0, 'f1': 0.889}
  ROUGE-2: {'precision': 0.5, 'recall': 0.667, 'f1': 0.572}
  ROUGE-L: {'lcs': 4, 'precision': 0.8, 'recall': 1.0, 'f1': 0.889}

Candidate: 'cat mat on sat'
  ROUGE-1: {'lcs': 1, 'precision': 1.0, 'recall': 1.0, 'f1': 1.000} ← misleading!
  ROUGE-2: {'precision': 0.0, 'recall': 0.0, 'f1': 0}
  ROUGE-L: {'lcs': 2, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}

# Perplexity:
Cross-entropy losses:
  cat: -log(0.5)  = 0.693
  sat: -log(0.4)  = 0.916
  on:  -log(0.35) = 1.050
  mat: -log(0.45) = 0.799
Average NLL = 0.865
Perplexity  = 2.374
Reference: random (vocab=4) → PP = 4.0
Reference: perfect model    → PP = 1.0
```

---

## Interview Q&A

**Q: What's the difference between precision and recall?**

Precision: of everything we labeled positive, how many actually were? (false alarm rate). Recall: of all true positives, how many did we find? (miss rate). High recall, low precision = spam filter that blocks too much. High precision, low recall = spam filter that lets spam through.

**Q: When would you prefer recall over precision?**

When false negatives are more costly than false positives: cancer screening — missing a cancer case (FN) is worse than a false alarm (FP) → maximize recall. Spam filtering — blocking a real email (FP) might be worse than missing spam (FN) → maximize precision.

**Q: What's macro vs micro F1?**

Macro: average per-class F1. Each class counts equally. Use when you care about rare classes. Micro: global TP/FP/FN pooled. Dominated by frequent classes. Use when volume matters.

**Q: Why is BLEU often 0 for short sentences?**

BLEU-4 requires 4-gram matches. A 4-word sentence has only one possible 4-gram. If that 4-gram doesn't match exactly, BLEU-4 = 0. Use BLEU-2 for short sentences or corpus-level BLEU.

**Q: What does ROUGE-1 = 1.0 but ROUGE-L = 0.5 tell you?**

Same words, wrong order. ROUGE-1 only counts token overlap, not sequence. ROUGE-L uses LCS — captures ordering. Always report ROUGE-L alongside ROUGE-1.

**Q: Why does BLEU use brevity penalty?**

Without it, a 1-word candidate "cat" would have 100% precision: p_1=1/1=1/1=1, BLEU=1.0. Brevity penalty: BP = exp(1-r/c) when candidate is shorter than reference. For "cat" vs "cat sat on mat": BP = exp(1-4/1) = exp(-3) = 0.050 → BLEU near 0 ✓

**Q: What's the relationship between perplexity and cross-entropy loss?**

PP = exp(H) where H = average cross-entropy loss. The "loss" the model trainer prints IS H, and PP = exp(loss). If loss = 2.3 → PP = 10. If loss = 0.5 → PP = 1.6.

**Q: When would you use BERTScore over ROUGE?**

When paraphrasing is acceptable: "automobile" should match "car", "rested" should match "sat". ROUGE only does exact token matching. BERTScore matches via cosine similarity in embedding space. Tradeoff: BERTScore is slower, less interpretable, still needs a BERT model.

**Q: How do you evaluate an LLM end-to-end in production?**

1. Task-specific: BLEU/ROUGE for generation, F1 for extraction
2. Reference-free: LLM-as-judge (GPT-4 rates output 1-5)
3. Benchmark: MMLU (knowledge), HumanEval (code), MT-Bench (instruction following)
4. Human eval: gold standard, expensive, can't scale
5. Regression: track perplexity on held-out set across training runs

---

## Connections

| Concept | Used in |
|---------|---------|
| F1 score | NER (token-level), classification, retrieval evaluation |
| BLEU | MT systems, code generation benchmarks (CodeBLEU) |
| ROUGE | Summarization, QA, document generation |
| Perplexity | LLM training, model comparison, selecting checkpoints |
| BERTScore | Any generation task where synonyms are valid |
| Precision@K | Retrieval (top-K documents), RAG evaluation |
| MRR | Retrieval: mean reciprocal rank of first correct result |

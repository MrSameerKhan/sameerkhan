# McDonald's BarRaiser — Morning Cheat Sheet (April 14, 11 AM IST)

> Read this in 20 minutes. Then close it and do 2 mock answers out loud.

---

## 0. BarRaiser Rules (Read First)

```
1. Use "I" not "we" — every sentence
2. Every result needs a NUMBER (F1 from 0.71→0.89, latency 40min→2min)
3. Pause 5 seconds before answering: "Let me think of the best example..."
4. End EVERY answer with: "What I learned was..."
5. 90 seconds per answer. Shorter = shallow. Longer = rambling.

STAR timing:
  Situation  15s  →  Task  10s  →  Action  60s  →  Result  15s
```

---

## 1. Probability & Statistics

### Core Formulas

```
Mean      = Σx / n
Variance  = Σ(x−μ)² / (N−1)    ← N−1 = Bessel's correction (sample)
Std Dev   = √Variance
IQR       = Q3 − Q1
Outlier   = x < Q1 − 1.5·IQR   OR   x > Q3 + 1.5·IQR

Pearson r = Cov(X,Y) / (σx·σy)   ∈ [−1, 1]
Spearman  = Pearson on RANKS — use when data has outliers or is ordinal
```

### Normal Distribution

```
68% within μ ± 1σ
95% within μ ± 2σ
99.7% within μ ± 3σ
Z-score = (X − μ) / σ
```

### Central Limit Theorem

```
x̄ ~ N(μ, σ²/n)  as n → ∞

Even if original population is non-normal,
the SAMPLE MEAN becomes normal for large n.
→ Foundation for all confidence intervals and hypothesis tests.
```

### Hypothesis Testing

```
H₀ = null hypothesis (nothing is happening)
H₁ = alternative hypothesis (your claim)

p-value = P(observing data this extreme | H₀ is true)

If p < α (0.05) → reject H₀ → result is statistically significant
If p ≥ α       → fail to reject H₀ → insufficient evidence

Type I error  = False Positive = rejecting H₀ when it's true  (α)
Type II error = False Negative = failing to reject H₀ when it's false (β)
Power = 1 − β = probability of detecting a real effect
```

### Bayes' Theorem

```
P(A|B) = P(B|A) · P(A) / P(B)

Example: Disease test
  P(disease) = 0.01,  P(positive|disease) = 0.99,  P(positive|no disease) = 0.05
  P(disease|positive) = (0.99 × 0.01) / (0.99×0.01 + 0.05×0.99) = 0.0099/0.0594 = 16.7%
  → Most positives are false positives when prevalence is low!
```

### Key Distributions

| Distribution | When it appears |
|---|---|
| Normal N(μ,σ²) | Feature noise, CLT, weight init |
| Bernoulli(p) | Binary outcome (click/no-click) |
| Binomial B(n,p) | Count of successes in n trials |
| Poisson(λ) | Events per unit time (requests/sec) |
| Exponential(λ) | Time between events |
| Beta(α,β) | Prior over probabilities |

---

## 2. ML Fundamentals

### Bias-Variance Tradeoff

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias    = underfitting → model too simple → train AND test error high
High Variance = overfitting → model too complex → train good, test bad

Fix high bias:    more features, more complex model, less regularization
Fix high variance: more data, regularization (L1/L2), dropout, early stopping
```

### Regularization

```
L1 (Lasso): loss + λ·Σ|wᵢ|     → forces some weights to exactly 0 → feature selection
L2 (Ridge): loss + λ·Σwᵢ²      → shrinks weights smoothly → no sparsity
ElasticNet: α·L1 + (1−α)·L2    → combine both

λ large → stronger regularization → more bias, less variance
```

### Model Evaluation Metrics

```
Classification:
  Precision  = TP / (TP + FP)       → cost of false alarm is high
  Recall     = TP / (TP + FN)       → cost of missing is high
  F1         = 2·P·R / (P+R)        → imbalanced classes
  ROC-AUC    = ranking quality, threshold-independent
  PR-AUC     = better than ROC-AUC for imbalanced data

Regression:
  MAE  = mean |y − ŷ|               → robust to outliers
  RMSE = √mean(y−ŷ)²               → penalizes large errors
  R²   = 1 − SS_res/SS_tot         → proportion of variance explained

Fraud example (1000 tx, 10 fraud):
  TP=8, FP=50, FN=2, TN=940
  Accuracy=94.8% ← USELESS, misleading
  Precision=13.8%,  Recall=80%,  F1=23.7% ← use these
```

### Cross-Validation

```
k-Fold: split data into k parts, train on k−1, test on 1, rotate
  → k=5 or k=10 standard
  → Stratified k-Fold: preserve class ratio in each fold (imbalanced data)

Train/Val/Test: 70/15/15 or 80/10/10
  Never touch test set until final evaluation — only one shot
```

### Key Algorithms — One Line Each

```
Linear Regression   → learns weights via MSE loss, closed form or gradient descent
Logistic Regression → sigmoid(wx+b) → probability, trained with cross-entropy
Decision Tree       → splits on information gain (entropy) or Gini impurity
Random Forest       → ensemble of trees, bagging + feature subsampling, reduces variance
XGBoost/GBDT        → boosting: each tree fixes previous tree's residuals
SVM                 → maximum margin classifier, kernel trick for non-linear
k-NN                → no training, classify by majority vote of k nearest neighbors
k-Means             → cluster by minimizing within-cluster variance, iterative
```

### Gradient Descent

```
w ← w − η · ∂L/∂w

SGD:       one sample per update  → noisy but fast, good for large datasets
Mini-batch: 32-256 samples       → balance of noise and efficiency (standard)
Adam:      adaptive learning rate per parameter, momentum + RMSprop

Learning rate:
  Too high → loss oscillates or diverges
  Too low  → converges slowly or gets stuck in local minima
  Use: learning rate scheduler (warmup + cosine decay)
```

---

## 3. ML System Design — 8-Step Framework

```
1. CLARIFY      (2 min): scale? latency? data available? labels? constraints?
2. METRICS      (2 min): offline metric + online business metric, justify link
3. ARCHITECTURE (5 min): draw boxes, show data flow
4. DATA         (5 min): sources, labeling, preprocessing, splits
5. MODELING     (5 min): baseline first, then improve
6. SERVING      (5 min): batch vs real-time, latency budget, scaling
7. MONITORING   (3 min): data drift, model drift, alerting
8. TRADE-OFFS   (rest):  what you'd do differently with more time/data/budget
```

### Questions to Ask Before Designing

```
"How many requests per second?"
"Is this real-time (< 100ms) or batch?"
"How much labeled training data is available?"
"What's the cost of a false positive vs false negative?"
"Any compliance constraints (GDPR, PII)?"
```

### Standard Architecture Pattern

```
Client → API Gateway → Feature Service → Model Server → Response

             ↓                   ↓
        Feature Store      Model Registry
             ↓                   ↓
         Data Lake         Experiment Tracker
```

### Monitoring Checklist

```
□ Data drift: distribution shift in features (PSI, KS-test)
□ Concept drift: relationship between X and y changes
□ Model metrics: P95 latency, error rate, prediction confidence histogram
□ Business metrics: CTR, conversion, revenue — these are the ground truth
□ Retraining trigger: drift detected OR scheduled (weekly/monthly)
```

---

## 4. Coding Mindset

```
When you get a coding problem:
  1. Clarify: "Can I assume sorted input? Any constraints on space?"
  2. Brute force first: "The naive solution is O(n²), but let me think..."
  3. Optimize: "I can reduce this to O(n log n) using a heap / sliding window / hash map"
  4. Write clean code with meaningful variable names
  5. Test with edge cases: empty input, single element, duplicates, negatives

Complexity to know:
  Array lookup = O(1), Hash map = O(1), Binary search = O(log n)
  Sorting = O(n log n), BFS/DFS = O(V+E), DP = depends on state space
```

---

## 5. Your 5 Stories — Quick Reference

```
Story 1 — Technical achievement:
  RAG pipeline, 200K docs, BGE-large + Chroma + BM25 + cross-encoder
  Search time: 40 min → 2 min. Recall@10 = 87%. 300 queries/day.

Story 2 — Failure:
  NLP model deployment, underestimated load (10→200 users)
  4-hour outage. Fixed with dynamic batching + horizontal scaling.
  Changed: load test at 3× expected QPS before every deploy.

Story 3 — Conflict / difficult person:
  Data engineer blocked by competing priorities.
  Took schema work myself (−40% their effort), formalized blocker in tracker.
  Delivered 3 days later, set up weekly sync.

Story 4 — Learned quickly:
  LangGraph in 2 weeks. Read → rebuild from scratch → PR merged by day 10.
  Rule: rebuild something real without looking at code — gaps appear fast.

Story 5 — Incomplete information:
  Production incident, 3 hours to decide. Chose rollback over patch.
  "Prefer the reversible action under time pressure."
```

---

## 6. Morning Schedule

```
10:00 AM — Read this cheat sheet (20 min)
10:20 AM — Say Story 1 and Story 2 out loud, time them (10 min)
10:30 AM — Deep breath. Water. Quiet room. Camera on. Good lighting.
10:50 AM — Join 10 min early. Test audio/video.
11:00 AM — Interview starts.

First words: "Let me think of the best example for that..." — never rush.
Last words of every answer: "What I learned from that was..."
```

# Semi-Supervised Learning

> **When:** You have a small labeled set + large unlabeled set. Common in real-world ML — labeling is expensive.

## The Setup

```
Typical scenario:
  Labeled data:   L = 500 examples  (expensive — human annotators)
  Unlabeled data: U = 50,000 examples (cheap — collected from production)

Supervised learning uses only L → model underfits, high variance
Semi-supervised uses L + U → leverages structure in U to improve

Why unlabeled data helps:
  Unlabeled data reveals the input distribution P(X)
  Key assumption: data points that are "close" in feature space
    should have the same label
```

## Core Assumptions

```
1. Smoothness assumption:
   If x1 ≈ x2 (close in input space) → y1 ≈ y2 (same label)
   "Similar inputs → similar outputs"

2. Cluster assumption:
   Data forms clusters; points in the same cluster share a label
   Decision boundary should NOT pass through dense regions

3. Manifold assumption:
   High-dim data lies on a low-dim manifold
   Labels vary smoothly along the manifold
```

---

## Self-Training (Pseudo-Labeling)

**Simplest approach.** Use model predictions on unlabeled data as "pseudo labels", then retrain.

### Algorithm

```
Step 1: Train model f on labeled set L
Step 2: Predict on unlabeled set U → get probabilities p = f(x) for all x ∈ U
Step 3: Select high-confidence predictions:
          U_pseudo = {(x, argmax p(x)) : max p(x) > threshold τ}
Step 4: Retrain on L ∪ U_pseudo
Step 5: Repeat until convergence (or fixed iterations)
```

### Code

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def self_training(X_labeled, y_labeled, X_unlabeled, X_test, y_test,
                  threshold=0.90, max_iter=10):

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_labeled, y_labeled)
    print(f"Iter 0 (labeled only): test acc = {accuracy_score(y_test, model.predict(X_test)):.3f}")

    X_train = X_labeled.copy()
    y_train = y_labeled.copy()
    X_pool  = X_unlabeled.copy()

    for iteration in range(1, max_iter + 1):
        if len(X_pool) == 0:
            break

        probs         = model.predict_proba(X_pool)
        confidence    = probs.max(axis=1)
        pseudo_labels = probs.argmax(axis=1)

        # Select high-confidence samples
        mask       = confidence >= threshold
        n_selected = mask.sum()

        if n_selected == 0:
            print(f"Iter {iteration}: no confident predictions → stopping")
            break

        # Add to training set, remove from pool
        X_train = np.vstack([X_train, X_pool[mask]])
        y_train = np.concatenate([y_train, pseudo_labels[mask]])
        X_pool  = X_pool[~mask]

        # Retrain
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Iter {iteration}: added {n_selected} pseudo-labels, "
              f"pool={len(X_pool)}, test acc={acc:.3f}")

    return model

# Dry run results:
# Iter 0 (labeled only): test acc = 0.812
# Iter 1: added 1,243 pseudo-labels, pool=48,757, test acc = 0.851
# Iter 2: added 892 pseudo-labels,   pool=47,865, test acc = 0.864
# Iter 3: added 614 pseudo-labels,   pool=47,251, test acc = 0.871
# Iter 4: added 207 pseudo-labels,   pool=46,964, test acc = 0.873
# Iter 5: no confident predictions → stopping
# Final: +6.1% accuracy using 2,956 pseudo-labeled samples
```

### Key hyperparameter — threshold τ

```
τ = 0.99: very strict → few pseudo-labels → slow but high precision
τ = 0.70: loose → many pseudo-labels → fast but noisy labels hurt model
τ = 0.90: good default

Confidence distribution (dry run):
  max(p(x)) > 0.99:  420 samples  (very confident)
  max(p(x)) > 0.95: 1,820 samples
  max(p(x)) > 0.90: 3,241 samples  ← good threshold
  max(p(x)) > 0.80: 8,100 samples  (too many, includes noisy)
```

---

## Label Propagation

Graph-based method. Build a similarity graph over labeled + unlabeled data. Propagate labels through the graph.

### Algorithm

```
1. Build graph: nodes = all data points, edges = similarity w(xi, xj)
   w(xi, xj) = exp(-||xi - xj||² / 2σ²)   ← Gaussian kernel

2. Construct transition matrix T:
   Tij = w(xi, xj) / Σk w(xi, xk)          ← row-normalize

3. Propagate: F = α·T·F + (1-α)·Y0
   α = propagation factor (0.9 typical)
   Y0 = initial label matrix (1-hot for labeled, 0 for unlabeled)
   Iterate until convergence

4. Assign labels: argmax F for each unlabeled node
```

```python
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
import numpy as np

# -1 marks unlabeled samples
y_mixed = np.array([0, 1, 0, 1, -1, -1, -1, -1, -1, -1])
#                   ↑ labeled ↑       ↑——— unlabeled ———↑

# LabelPropagation: hard clamp on labeled nodes
model_lp = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
model_lp.fit(X_all, y_mixed)

# LabelSpreading: soft clamp (allows label noise), more robust
model_ls = LabelSpreading(kernel='rbf', gamma=20, alpha=0.2, max_iter=1000)
model_ls.fit(X_all, y_mixed)

# Get propagated labels for unlabeled samples
unlabeled_mask = (y_mixed == -1)
propagated_labels = model_ls.transduction_[unlabeled_mask]
```

**Best for:** low-dimensional data with clear cluster structure. Fails for high-dimensional data (curse of dimensionality — distances meaningless).

---

## Co-Training

**Requires:** two independent views/feature sets for the same example.

```
Classic example: web page classification
  View 1: page text content (bag of words)
  View 2: anchor text of hyperlinks pointing to the page

Algorithm:
  1. Train classifier f1 on View 1 using labeled set L
  2. Train classifier f2 on View 2 using labeled set L
  3. Each classifier labels examples from U that it's most confident about
  4. Add those pseudo-labeled examples to the other classifier's training set
  5. Repeat

Why it works: f1 and f2 are independent → each provides new information
              that the other doesn't already know

Example in NLP:
  View 1: BERT on input text
  View 2: structural features (font size, position, document metadata)
  → classifier trained on the other helps bootstrap structural classifier and vice versa
```

```python
def co_training(X1_l, X2_l, y_l, X1_u, X2_u, n_iter=30, pool_size=75, k=10):
    """
    X1, X2: two independent feature views
    k: examples to add per iteration per classifier
    """
    from sklearn.naive_bayes import MultinomialNB

    clf1 = MultinomialNB().fit(X1_l, y_l)
    clf2 = MultinomialNB().fit(X2_l, y_l)

    X1_pool, X2_pool = X1_u[:pool_size], X2_u[:pool_size]
    X1_rest, X2_rest = X1_u[pool_size:], X2_u[pool_size:]

    for _ in range(n_iter):
        # clf1 labels most confident from pool → add to clf2's training
        p1     = clf1.predict_proba(X1_pool)
        top_k1 = p1.max(axis=1).argsort()[::-1][:k]
        x_1_l  = np.vstack([X1_l, X1_pool[top_k1]])
        y_1_2  = np.append(y_l, p1[top_k1].argmax(axis=1))

        # clf2 labels most confident from pool → add to clf1's training
        p2     = clf2.predict_proba(X2_pool)
        top_k2 = p2.max(axis=1).argsort()[::-1][:k]
        x_2_l  = np.vstack([X2_l, X2_pool[top_k2]])
        y_1_1  = np.append(y_l, p2[top_k2].argmax(axis=1))

        # Replenish pool from rest
        X1_pool = np.vstack([X1_pool, X1_rest[2*k:]]);  X1_rest = X1_rest[2*k:]
        X2_pool = np.vstack([X2_pool, X2_rest[2*k:]]);  X2_rest = X2_rest[2*k:]

        clf1.fit(X1_l, y_1_1)
        clf2.fit(X2_l, y_1_2)

    return clf1, clf2
```

---

## Consistency Regularization (Deep Learning)

Modern semi-supervised deep learning. Key idea: a model should produce the same output for a sample and its augmented version.

```
Loss = L_supervised + λ · L_consistency

L_supervised = cross_entropy(f(x_labeled), y)
L_consistency = ||f(x_unlabeled) - f(augment(x_unlabeled))||²
              or KL(f(x) || f(augment(x)))

Intuition: if "cat" image → label=cat, then
           "slightly blurred cat" → should also be cat (with same confidence)
```

### FixMatch (2020) — SOTA Simple SSL

```
Strong augmentation (RandAugment): random color jitter, rotation, cutout
Weak augmentation:  random flip + crop only

Algorithm for each unlabeled batch:
  1. Weak-augment  x → x_weak
  2. Predict: p = f(x_weak)
  3. If max(p) > τ (e.g., 0.95):
       pseudo_label = argmax(p)
       Strong-augment x → x_strong
       Loss += cross_entropy(f(x_strong), pseudo_label)

Key: pseudo labels from WEAK augmentation, loss on STRONG augmentation
  → forces model to be invariant to strong perturbations
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Augmentation policies
weak_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
])

strong_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomErasing(p=0.5),
])

def fixmatch_loss(model, x_labeled, y_labeled, x_unlabeled,
                  threshold=0.95, lambda_u=1.0):

    batch_size = x_labeled.size(0)

    # --- Supervised loss ---
    logits_labeled = model(x_labeled)
    loss_supervised = F.cross_entropy(logits_labeled, y_labeled)

    # --- Unsupervised loss ---
    with torch.no_grad():
        # Weak augmentation → pseudo label
        x_weak        = weak_aug(x_unlabeled)
        probs         = F.softmax(model(x_weak), dim=-1)
        max_probs, pseudo_labels = probs.max(dim=-1)
        mask = max_probs >= threshold  # only use high-confidence

    # Strong augmentation → prediction
    x_strong      = strong_aug(x_unlabeled)
    logits_strong = model(x_strong)
    loss_unsup    = (F.cross_entropy(logits_strong, pseudo_labels, reduction='none') * mask).mean()

    total_loss = loss_supervised + lambda_u * loss_unsup
    return total_loss, mask.float().mean().item()  # return mask ratio for monitoring
```

---

## The Modern Alternative — Foundation Model + Few-Shot / Fine-Tune

In 2025, "I have 500 labels and 50K unlabeled examples" is often solved by **skipping semi-supervised entirely** and instead:

```
1. Take a pretrained foundation model (BERT / sentence-transformer / CLIP / LLM)
2. Either:
   (a) Few-shot prompt the LLM with 5-20 labeled examples
   (b) Fine-tune the foundation model on the 500 labels (LoRA / PEFT)
   (c) Embed all data with foundation model + train a small classifier head on the 500 labels
3. Use the 50K unlabeled set ONLY for selection (active learning) or pseudo-labeling
   as a top-up
```

**Why this dominates classical SSL:** Foundation model has already done implicit "semi-supervised" learning on internet-scale unlabeled data — 500 labels + pretrained BERT often beats 500 labels + 50K unlabeled + FixMatch. Massively simpler — no consistency loss, no co-training views, no graph kernel parameters.

**When classical SSL still wins:** Tabular domains where no relevant foundation model exists (specialized sensor data, proprietary medical features). Very high volume of cheap unlabeled data where the relative signal vs cost favors pseudo-labeling. Privacy-constrained settings where you can't send data to a pretrained model.

**Senior interview answer:** "I'd reach for SSL when (a) no relevant pretrained model exists for my domain, or (b) my unlabeled set is so much larger than the labeled set that pseudo-labeling becomes cheaper than fine-tuning. Otherwise — foundation model + fine-tune is usually the first thing to try."

---

## Active Learning — Pick Which Points to Label Next

Complementary to SSL. Instead of labeling random samples, label the ones the current model is **most uncertain** about. Same labeling budget — much better model.

```
Common acquisition functions:
  Least confidence:   pick x with smallest max p(y|x)
  Margin:             pick x with smallest p(y1) - p(y2)  (top-2 closest)
  Entropy:            pick x with highest H[p(y|x)]
  BALD (Bayesian):    max disagreement across MC dropout samples
  Embedding-based:    pick x far from any labeled point (cluster-based diversity)
```

```python
# Simple least-confidence loop
def active_learning_round(model, X_pool, batch_size=20):
    probs      = model.predict_proba(X_pool)
    confidence = probs.max(axis=1)
    # k samples with lowest confidence → send to human labelers
    return np.argsort(confidence)[:batch_size]
```

**Combine with SSL:** use active learning to choose what to label, then use SSL on the still-unlabeled remainder. Modal AI / Snorkel / Cleanlab productionize this pattern.

**Reference tools:** modAL (sklearn-style AL framework), Cleanlab (label-error detection + AL), Snorkel (weak supervision + labeling functions).

---

## Comparison

| Method | Data requirement | Scalability | Best use case |
|--------|-----------------|-------------|--------------|
| Self-training | Any model | High | Tabular, NLP, images |
| Label propagation | Low-dim features | Medium | Small datasets, graph data |
| Co-training | Two independent views | Medium | Multi-view data |
| FixMatch / consistency | Deep model | High | Images, NLP with augmentation |
| BERT fine-tuning (few-shot) | Pre-trained LLM | Very high | NLP with < 100 labels |

---

## Gotchas

**Confirmation bias in self-training.** If the initial model is wrong, pseudo-labels reinforce errors — model becomes overconfident in the wrong direction. Fix: lower threshold; use ensemble for pseudo-labels, or reset model before retraining.

**Distribution mismatch.** If unlabeled data comes from a different distribution than labeled data, semi-supervised methods can hurt. Always check unlabeled data quality before using it.

**Co-training view independence.** If the two views are correlated (not independent), co-training degrades to self-training. True independence is rare in practice — co-training works best when views provide genuinely different information.

**FixMatch threshold τ.** Too high (0.99): few pseudo-labels, slow improvement. Too low (0.80): many noisy pseudo-labels, model degrades. Start at 0.95, tune based on fraction of masked samples (monitor `mask.mean()` — target 30-60% samples above threshold).

---

## Interview Q&A

**Q: When would you use semi-supervised learning?**
A: When labeling is expensive or slow but unlabeled data is abundant. Common in: medical imaging (radiologist labeling costs $5/hour, but unlabeled scans are cheap), document classification (1000 labeled docs, 100K unlabeled), NLP (few human-labeled examples, large corpus). Also useful when a model trained on labeled data is deployed — production data can be used as unlabeled set.

**Q: Explain pseudo-labeling and its main risk.**
A: Pseudo-labeling uses a trained model to assign labels to unlabeled data, then retrains on labeled + pseudo-labeled data. The main risk is confirmation bias — if the initial model makes systematic errors, pseudo-labels encode those errors, and retraining amplifies them. The model becomes more confident in wrong predictions. Mitigation: high confidence threshold (0.90+), use ensemble predictions, verify a random sample of pseudo-labels manually.

**Q: What is consistency regularization?**
A: Consistency regularization adds a loss term that penalizes different predictions for the same sample under different augmentations. The intuition: a robust model should be invariant to small perturbations — "cat" should be "cat" whether slightly blurred or flipped. FixMatch (SOTA) generates pseudo-labels from weakly-augmented images, then trains on strongly-augmented versions of the same images. This forces the model to maintain consistent predictions across aggressive transformations, leveraging unlabeled data without needing ground truth.

---

## Connections

- Deep semi-supervised: `../../2.deep learning/02_architectures/08_semi_supervised.md`
- Self-supervised pre-training: `../../5.transformers/01_fundamentals/04_pretraining_objectives.md`
- Self-supervised vision (SimCLR/MoCo/BYOL/DINO/MAE): `../../3.computerVision/02_applications/05_self_supervised_vision.md`
- LLM-based pseudo-labeling: `../../6.llms/` — use a strong LLM to label data for a smaller student model (distillation as labeling)
- Contrastive embeddings (alternative path to use unlabeled data): `../../4.nlp/02_embeddings/06_contrastive_training.md`
- Data augmentation: `../../3.computerVision/` for vision pipelines

---

## Key Takeaway

Semi-supervised learning bridges labeled (small) + unlabeled (large) data. Three main families: **self-training** (pseudo-label confident predictions, retrain iteratively), **graph-based** (label propagation through similarity graph), **consistency regularization** (model predictions should be stable under augmentation — FixMatch is SOTA). Key risk across all methods: confirmation bias from noisy pseudo-labels. Always use high-confidence threshold and monitor pseudo-label quality.

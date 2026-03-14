# 04 — Generalization

---

## Quick Reference *(30-sec scan)*

- **Overfitting**: low train loss, high val loss → model memorized noise
- **Bias-variance**: underfitting = high bias, overfitting = high variance
- **L2 (weight decay)**: shrinks all weights — default regularization, use wd=0.01
- **Dropout**: randomly disables neurons — prevents co-adaptation, acts as ensemble
- **Batch size**: 32–256 default; large batches converge to sharp minima → worse generalization
- **Data split**: train/val/test — test set touched ONCE; data leakage is silent and lethal
- **Metric selection**: accuracy is misleading on imbalanced data — use F1 or AUC-ROC

---

## 1. Overfitting & Underfitting

| | Train Loss | Val Loss | Cause | Fix |
|--|-----------|---------|-------|-----|
| **Underfitting** | High | High | Model too simple | More capacity, more epochs |
| **Good fit** | Low | Low | Balanced | — |
| **Overfitting** | Very low | High | Model memorized noise | Regularize, more data |

### Loss Curve Patterns

```
Underfitting:          Good Fit:           Overfitting:
Loss                   Loss                Loss
│──── train            │\                  │\
│──── val              │ \──── val          │ \
│                      │  \─── train        │  \──────── train
└──── Epochs           └──── Epochs         │   \   /─── val rises
                                            └──── Epochs
```

### Bias-Variance Tradeoff

| | Bias | Variance |
|--|------|----------|
| Underfitting | High | Low |
| Good fit | Low | Low |
| Overfitting | Low | High |

**Bias** = error from wrong assumptions (model too simple).
**Variance** = sensitivity to training data fluctuations (model too complex).

Reducing one typically increases the other. Goal: minimize both simultaneously through regularization + data + architecture choices.

---

## 2. Regularization

### L2 Regularization (Weight Decay)

$$L_{total} = L_{data} + \lambda \sum W^2$$

During backprop: $\frac{\partial L}{\partial W} = \frac{\partial L_{data}}{\partial W} + \lambda W$

Gently pulls all weights toward zero → simpler decision boundaries → better generalization.

**Always use AdamW's weight decay, not L2 in the loss function.** (See `02_training_loop.md` → AdamW section.)

### L1 Regularization (Lasso)

$$L_{total} = L_{data} + \lambda \sum |W|$$

Encourages **exact zeros** — many weights become 0 → implicit feature selection.

| | L2 | L1 |
|--|----|----|
| Effect | Shrinks all weights | Zeros out small weights |
| Result | Dense, small weights | Sparse weights |
| Use case | General regularization | Feature selection |

### Dropout

Randomly disables neurons during each training step.

```
Training step (p=0.5):
  Layer activations: [0.8, 0.3, 0.5, 0.9]
  Random mask:       [1,   0,   1,   0  ]
  After dropout:     [0.8, 0.0, 0.5, 0.0]
  After scaling:     [1.6, 0.0, 1.0, 0.0]  ← divide by (1-p) = inverted dropout
```

**Why it works**: forces each neuron to be useful independently (can't rely on neighbors). Equivalent to training an ensemble of `2^n` sub-networks — inference approximates their average.

**Dropout is OFF at inference** — use `model.eval()`.

| Layer type | Typical dropout rate |
|-----------|---------------------|
| Fully connected (hidden) | 0.5 |
| CNN layers | 0.2–0.3 |
| Transformer attention/FFN | 0.1 |
| Output layer | Never |

**Don't use dropout + BatchNorm together** — they conflict (BN statistics are disrupted by dropout's random zeroing).

### Early Stopping

Stop training when validation loss stops improving.

```
Epoch 5:  val_loss = 0.42  ← best
Epoch 6:  val_loss = 0.44
Epoch 7:  val_loss = 0.45  ← patience=2, stop here
→ Load weights from epoch 5
```

**Patience** (how many epochs to wait) is itself a hyperparameter. Typical: 5–15 epochs.

### Data Augmentation

Artificially expand training data → model sees more variation → better generalization.

| Domain | Augmentations |
|--------|--------------|
| Images | Flip, rotate, crop, color jitter, cutout, mixup |
| Text | Synonym replacement, back-translation, random deletion |
| Audio | Time stretch, pitch shift, background noise |
| Documents | Rotation, noise, resolution changes, font variation |

---

## 3. Training Dynamics — Epochs, Batch Size, Iterations

$$\text{Iterations per epoch} = \frac{\text{Total samples}}{\text{Batch size}}$$

Example: 10,000 samples, batch=32 → 312 iterations per epoch.

### Gradient Descent Variants

| Type | Batch Size | Behavior |
|------|------------|---------|
| SGD (true) | 1 | Noisy, fast, escapes local minima |
| **Mini-batch** | 32–256 | Standard — balance of speed + stability + GPU |
| Full-batch | All data | Stable but needs all data in memory |

### Batch Size Effect on Generalization

Large batches (>512): deterministic gradients → converge to **sharp minima** → worse generalization (Keskar et al., 2017).
Small batches: noisy gradients → converge to **flat minima** → better generalization.

**Practical rule**: Start with 32. Use 64/128 if GPU allows. If test accuracy suffers, reduce batch size.

If you must use large batches (distributed training): compensate with **linear LR scaling** (`lr = base_lr × batch_size / 32`) + warmup.

### Always shuffle training data between epochs. Never shuffle val/test.

---

## 4. Data Splits

| Split | Purpose | When Touched |
|-------|---------|-------------|
| **Train** | Model learns from this | Every iteration |
| **Validation** | Tune hyperparameters, detect overfitting | After each epoch |
| **Test** | Final honest evaluation | Once — at the very end |

### Split Ratios

| Dataset Size | Train | Val | Test |
|-------------|-------|-----|------|
| Small (<10K) | 70% | 15% | 15% |
| Medium (10K–1M) | 80% | 10% | 10% |
| Large (>1M) | 98% | 1% | 1% |

### The Golden Rule

**Test set is never used to make any training decision.** Using it to select hyperparameters turns it into a second val set → reported accuracy is overly optimistic.

### Cross-Validation

When dataset is small, a single split is noisy. K-fold trains K models, each using a different fold as validation.

```
5-fold:
Fold 1: [Val][Tr ][Tr ][Tr ][Tr ]
Fold 2: [Tr ][Val][Tr ][Tr ][Tr ]
Fold 3: [Tr ][Tr ][Val][Tr ][Tr ]
...
Average val scores → stable estimate of generalization
```

Expensive but reliable for small datasets.

### Data Leakage — Common Mistakes

| Mistake | Why it's leakage | Fix |
|---------|-----------------|-----|
| Fit scaler on full dataset | Test data influenced train scaling | Fit scaler on train only, transform val/test |
| Use test set to pick model | Test set becomes val set | Only compare on val; test once at end |
| Don't shuffle before split | Ordered data → biased splits | Always shuffle before splitting |
| Target encoding before split | Target stats include test rows | Compute encoding on train fold only |

---

## 5. Evaluation Metrics

### Classification — Confusion Matrix

```
                 Predicted
                 Positive  Negative
Actual Positive    TP        FN
       Negative    FP        TN
```

| Metric | Formula | Use when |
|--------|---------|---------|
| **Accuracy** | `(TP+TN)/Total` | Balanced classes only |
| **Precision** | `TP/(TP+FP)` | FP is costly (spam filter) |
| **Recall** | `TP/(TP+FN)` | FN is costly (cancer detection) |
| **F1** | `2×P×R/(P+R)` | Imbalanced classes, both errors matter |
| **AUC-ROC** | Area under ROC curve | Comparing models, imbalanced data |

**Precision vs Recall tradeoff**: lowering classification threshold → more positives predicted → recall ↑, precision ↓. Choose threshold based on cost of each error type.

**AUC interpretation**: probability that model ranks a random positive higher than a random negative. AUC=1.0 → perfect, AUC=0.5 → random.

**Multi-class metrics:**
- **Macro avg**: each class weighted equally — use when all classes equally important
- **Weighted avg**: weighted by class support — use when class distribution reflects real world

### Regression Metrics

| Metric | Formula | Use when |
|--------|---------|---------|
| **MAE** | `mean(|y - ŷ|)` | Outliers present, interpretable |
| **MSE** | `mean((y - ŷ)²)` | Large errors should be penalized more |
| **RMSE** | `sqrt(MSE)` | Same unit as target, penalizes large errors |
| **R²** | `1 - SS_res/SS_tot` | Proportion of variance explained |

---

## 6. Hyperparameter Tuning

**Parameters**: learned by gradient descent (W, b)
**Hyperparameters**: set before training (LR, batch size, dropout rate, number of layers)

### Priority Order for Tuning

```
1. Learning rate          ← biggest impact
2. Batch size
3. Regularization (wd, dropout)
4. Architecture (layers, neurons)
5. Everything else
```

### Search Strategies

| Strategy | How | When to use |
|----------|-----|------------|
| **Manual** | Change one HP at a time based on intuition | Always start here |
| **Grid search** | Try all combinations in a grid | ≤2 HPs, small search space |
| **Random search** | Sample randomly from distributions | 3+ HPs, faster than grid |
| **Bayesian** | Build surrogate model, sample intelligently | Each run is expensive |

**Random > Grid** for high-dimensional spaces. If one HP matters more than others, grid wastes runs by fixing the unimportant ones. Random explores all dimensions.

**Tools**: Optuna, Ray Tune, Keras Tuner, W&B Sweeps.

### Good Starting Defaults

| Hyperparameter | Starting Value |
|----------------|---------------|
| LR | 1e-3 (Adam) / 0.1 (SGD) |
| Batch size | 32 |
| Weight decay | 0.01 |
| Dropout (FC) | 0.5 |
| Dropout (CNN) | 0.2 |
| Early stopping patience | 5–10 |
| Initialization | He (ReLU), Xavier (tanh) |

---

## When to Use What

| Situation | Tool |
|-----------|------|
| General regularization | AdamW with weight_decay=0.01 |
| Large FC layers overfitting | Dropout 0.5 |
| Feature selection needed | L1 regularization |
| Training time unknown | Early stopping |
| Small dataset | Cross-validation |
| Imbalanced classes | Stratified split + F1/AUC metric |
| Fast HP search | Random search (50 trials) |
| Expensive runs | Bayesian optimization (Optuna) |

---

## Gotchas

**1. Accuracy on imbalanced data is meaningless**
A model predicting all "healthy" on 990/10 cancer data gets 99% accuracy and detects zero cancer. Always check class distribution before choosing a metric.

**2. Data leakage is silent — it always inflates your results**
Preprocessing before splitting is the most common form. Normalizing on the full dataset, computing category statistics on full dataset — these are all leakage. Always split first, then fit any statistics on train only.

**3. Large batch size harms generalization**
Not a hardware problem — a fundamentals one. Large batches compute near-exact gradients → converge to sharp minima that overfit. Small batches inject useful noise. If you scale batch size for speed, also scale LR and add warmup.

**4. Test set touched more than once = invalidated**
Each time you look at test results and change something, the test set has leaked. In published research, this causes benchmark overfitting at the community level. In production, this causes overconfident models.

**5. val_loss improving ≠ generalization improving**
Validation metrics can plateau or even slightly worsen while training loss improves significantly. Monitor the gap (overfit ratio = val_loss/train_loss), not just val_loss alone.

---

## Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Train loss low, val loss high | Overfitting | Add dropout, increase weight decay, get more data |
| Both losses high and flat | Underfitting | Increase model capacity, train longer |
| Val loss improves then plateaus | LR too low or model at capacity | LR schedule, try larger model |
| Metrics look great but production fails | Data leakage | Audit entire preprocessing pipeline |
| Model performs well on test but poorly live | Distribution shift | Check live data vs test data distribution |
| Precision high, recall low | Threshold too high | Lower classification threshold |
| Recall high, precision low | Threshold too low | Raise threshold, or check for class imbalance |

---

## Code Reference

```python
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Data splitting with stratification
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# Dropout layer
dropout = nn.Dropout(p=0.5)  # training: randomly zeros p fraction; eval: identity

# L2 via AdamW weight_decay (correct way)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Early stopping (manual implementation)
best_val_loss = float('inf')
patience_counter = 0
patience = 10

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(torch.load('best_model.pt'))

# Classification report (multi-class metrics)
y_pred = model(X_test).argmax(dim=1).numpy()
print(classification_report(y_test, y_pred, target_names=class_names))

# AUC-ROC (binary)
y_prob = torch.sigmoid(model(X_test)).numpy()
auc = roc_auc_score(y_test, y_prob)

# Hyperparameter tuning with Optuna
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    wd = trial.suggest_float('wd', 1e-4, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    # build model, train, return val_loss
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

## Interview Q&A

**Q: What is the bias-variance tradeoff and how do you control it?**
> Bias = error from wrong model assumptions (underfitting). Variance = error from model sensitivity to training data fluctuations (overfitting). They trade off — a more complex model reduces bias but increases variance. You control it through: model capacity (architecture), regularization (L1/L2/dropout), dataset size, and early stopping. The goal is to find the sweet spot where both are minimized.

**Q: Why is accuracy a bad metric for imbalanced classification?**
> A model predicting only the majority class gets high accuracy while having zero predictive power for the minority class. In document processing (e.g., fraud detection, rare document types), the minority class is often what matters most. Use precision, recall, F1, or AUC-ROC which account for false positives and false negatives explicitly.

**Q: How does dropout act as an ensemble method?**
> With N neurons and dropout rate p, each forward pass uses a randomly sampled sub-network. There are `2^N` possible sub-networks. Each training step trains a different one. At inference with all neurons active, the output approximates averaging all `2^N` sub-network predictions. Ensembles consistently outperform single models — dropout gets this effect at the cost of a single model.

**Q: What is data leakage and give a real example?**
> Data leakage occurs when information from validation/test data influences training decisions, making results optimistically biased. A common example: you compute mean/std for normalization on the full dataset, then split. The normalizer has seen test data statistics. Now your model (indirectly) has access to test data distributions, inflating reported accuracy. Fix: split first, then fit any statistics on train only.

**Q: Why does large batch size often hurt generalization?**
> Large batches compute near-exact gradient estimates (low noise) → gradient descent converges toward sharp minima. Sharp minima are regions where a small perturbation to weights causes large increases in loss — the model overfits to the training distribution. Small batches inject noise → gradient descent wanders more → finds flat minima that generalize across the distribution. This is not just a theory — Keskar et al. 2017 showed this empirically on multiple benchmarks.

---

## Connections

- **Builds on**: `02_training_loop.md` — optimizer choice and LR schedule affect overfitting
- **Builds on**: `03_training_stability.md` — dropout conflicts with BatchNorm; normalization also regularizes
- **Relevant in**: `05_modern_components.md` — attention dropout, embedding dropout used in transformers
- **Relevant in**: CV, NLP domains — augmentation strategies are domain-specific
- **If your model overfits**: audit in order → weight decay → dropout → data augmentation → model size → more data

---

## Key Takeaway

```
Overfit:     train loss ↓, val loss ↑ → regularize (L2 + dropout + augmentation)
Underfit:    both losses high → more capacity or more training
Batch size:  small = noisy = flat minima = better generalization
Data split:  train/val/test; test touched ONCE; leakage = silent inflation
Metrics:     always match metric to task and class balance
HP tuning:   LR first, random search > grid search, Optuna for expensive runs
```
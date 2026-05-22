# 04 — Model Evaluation

## Quick Reference

| Metric | Task | When to Use |
|--------|------|-------------|
| Accuracy | Classification | Only when classes balanced |
| Precision | Classification | Cost of false positive is high (spam, fraud alerts) |
| Recall | Classification | Cost of false negative is high (cancer screening, fraud detection) |
| F1 | Classification | Imbalanced classes, need P/R balance |
| ROC-AUC | Classification | Ranking quality; threshold-independent |
| PR-AUC | Classification | Imbalanced classes (more informative than ROC-AUC) |
| MAE | Regression | Robust to outliers; interpretable in original units |
| RMSE | Regression | Penalizes large errors more; same units as target |
| MAPE | Regression | Percentage error (scale-independent) |
| R² | Regression | Proportion of variance explained |

---

## 1. Classification Metrics

### Confusion Matrix

```
                   Predicted Positive   Predicted Negative
Actual Positive:         TP                   FN
Actual Negative:         FP                   TN

Precision   = TP / (TP + FP)          of all predicted positives, how many were actually positive?
Recall      = TP / (TP + FN)          of all actual positives, how many did we catch?
Accuracy    = (TP + TN) / Total       overall correct rate
F1          = 2·P·R / (P+R)          harmonic mean of precision and recall
Specificity = TN / (TN + FP)         true negative rate
```

Numeric example (fraud detection, 1000 transactions, 10 frauds):
```
TP=8, FP=50, FN=2, TN=940

Accuracy  = (8+940)/1000 = 94.8%  ← looks great, but misleading!
Precision = 8/(8+50)    = 13.8%  ← of all fraud alerts, only 13.8% are real
Recall    = 8/(8+2)     = 80%    ← we caught 80% of actual fraud
F1        = 2×0.138×0.8/(0.138+0.8) = 23.7%
```

Accuracy is useless here. Use Precision/Recall/F1 for imbalanced problems.

### Precision-Recall Tradeoff

```
Lower threshold → more positives predicted → higher Recall, lower Precision
Higher threshold → fewer positives predicted → lower Recall, higher Precision

Threshold 0.5 is arbitrary — tune based on business cost of FP vs FN.
```

```python
from sklearn.metrics import precision_recall_curve, classification_report
import matplotlib.pyplot as plt

probs = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1],    label='Recall')
plt.xlabel('Threshold'); plt.legend()
plt.show()

# Choose threshold based on a business requirement
# e.g., need recall >= 0.9 → find minimum threshold that achieves this
target_recall = 0.9
idx = np.argmax(recalls >= target_recall)
optimal_threshold = thresholds[idx]
```

### ROC-AUC vs PR-AUC

**ROC Curve:** TPR (Recall) vs FPR at all thresholds.
```
AUC = 0.5: random classifier
AUC = 1.0: perfect classifier
AUC = 0.7-0.8: decent;  0.9+: excellent
```

**Problem with ROC-AUC for imbalanced data:** When negatives dominate, FPR = FP/(FP+TN) stays small even with many FPs → ROC-AUC looks good while model is practically useless.

**PR-AUC (Average Precision):** Precision vs Recall at all thresholds. Better for imbalanced data — doesn't involve TN in calculation, focuses entirely on positive class performance.

```python
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

auc = roc_auc_score(y_test, probs)         # ROC-AUC
ap  = average_precision_score(y_test, probs)  # PR-AUC

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, probs)
plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
plt.plot([0,1], [0,1], '--', color='grey')
plt.xlabel('FPR'); plt.ylabel('TPR')
```

### F-beta Score

When recall is more important than precision:

```
F_β = (1+β²) · P·R / (β²·P + R)

β=1:   F1   (equal weight)
β=2:   F2   (recall 2× more important than precision) — use for disease detection
β=0.5: F0.5 (precision 2× more important) — use for spam filter (don't block legitimate emails)
```

```python
from sklearn.metrics import fbeta_score
F2 = fbeta_score(y_test, y_pred, beta=2)
```

### Multiclass Metrics

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
# Shows precision, recall, F1 per class + macro/weighted averages

# Macro average:    mean of per-class metric (treats all classes equally)
# Weighted average: mean weighted by class support (accounts for imbalance)
# Micro average:    aggregate TP/FP/FN across all classes (same as accuracy for single-label)
```

---

## 2. Regression Metrics

| Metric | Formula | Properties |
|--------|---------|-----------|
| MAE | (1/n)Σ\|y−ŷ\| | Robust to outliers; interpretable |
| MSE | (1/n)Σ(y−ŷ)² | Penalizes large errors more; not same units |
| RMSE | √MSE | Same units as target; sensitive to outliers |
| MAPE | (1/n)Σ\|y−ŷ\|/y×100% | Scale-independent %; breaks when y≈0 |
| R² | 1 − SS_res/SS_tot | 0-1: fraction of variance explained |
| Adjusted R² | 1 − (1-R²)(n-1)/(n-k-1) | Penalizes adding irrelevant features |

**Choosing MAE vs RMSE:**
- MAE: business cases care equally about all errors → house price prediction (off by $10K is bad whether on $200K or $500K house)
- RMSE: large errors are catastrophically wrong → safety-critical systems, demand forecasting (being off by 1000 units is much worse than being off by 10)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}, MAPE: {mape:.1f}%")
```

**Negative R²?** Your model is worse than predicting the mean. Something is very wrong.

---

## 3. Cross-Validation

### Why CV?

Single train/test split → results depend on which samples ended up in test. CV averages over multiple splits → more reliable estimate of generalization.

### K-Fold CV

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

# Standard K-Fold (regression)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f"CV RMSE: {np.sqrt(-cv_scores.mean()):.3f} ± {np.sqrt(-cv_scores).std():.3f}")

# Stratified K-Fold (Classification — preserves class ratio in each fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

Always use **StratifiedKFold** for classification — regular KFold can put all positives in one fold.

### CV Strategies

| Strategy | When to Use |
|----------|-------------|
| K-Fold (k=5 or 10) | Standard; large enough dataset |
| Stratified K-Fold | Classification, especially imbalanced |
| Leave-One-Out (LOO) | Very small datasets (n < 50) |
| Time Series Split | Time-ordered data — train on past, test on future |
| Group K-Fold | Samples from same group must not split across folds |

```python
# Time Series Split (Critical for Sequential Data)
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    # train on past, validate on future — no lookahead

# Group K-Fold (Avoid Leakage Across Groups)
from sklearn.model_selection import GroupKFold
# Example: multiple samples per patient — don't let same patient appear in train AND test
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=patient_ids):
    ...
```

---

## 4. Data Leakage

The #1 reason ML models fail in production. **Leakage = test data information leaks into training.**

### Types of Leakage

**Feature leakage (most common):** Feature contains information that wouldn't be available at prediction time.
```
Example: predicting hospital readmission
  Leaky feature: "number of days in hospital on current admission"
  → not known until the patient is discharged, which is AFTER the prediction is needed

Example: predicting fraud
  Leaky feature: "account_suspended_flag"
  → only set AFTER fraud is detected, not before
```

**Temporal leakage:** Using future data to predict past.
```
Wrong: random 80/20 train/test split on time-series data
  → test set contains data from before some training samples
Right: train on data before date X, test on data after date X
```

**Preprocessing leakage:** Fitting transformers on the full dataset before splitting.
```python
# Wrong:
scaler.fit(X_all)                          # sees test distribution
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Right:
scaler.fit(X_train)                        # only train
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

**Target leakage in CV:**
```python
# Wrong:
X_selected = SelectKBest().fit_transform(X, y)  # uses full y
cross_val_score(model, X_selected, y)            # leaks test y into feature selection

# Right: put feature selection INSIDE the cross-validation loop (use Pipeline)
pipeline = Pipeline([('selector', SelectKBest()), ('model', model)])
cross_val_score(pipeline, X, y)                  # selection re-done per fold
```

### Detecting Leakage — Red Flags
- Suspiciously high CV score (>0.99 AUC on a hard problem)
- Feature correlation with target > 0.9
- Model performance degrades sharply at deployment
- A feature you didn't expect to be useful is the #1 predictor

---

## 5. Hyperparameter Tuning

### Grid Search vs Random Search vs Bayesian

| Method | Pros | Cons | When |
|--------|------|------|------|
| GridSearchCV | Exhaustive, reproducible | Exponential time with parameters | Few parameters, small grid |
| RandomizedSearchCV | Much faster, often finds optimum | Not exhaustive | Many parameters, wide ranges |
| Bayesian Optimization | Most efficient, learns from past trials | More setup | Expensive models, large search space |

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform

# Grid Search
param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'min_samples_leaf': [1, 5]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_, grid_search.best_score_)

# Randomized Search (faster)
param_dist = {'n_estimators': randint(100, 500), 'max_depth': randint(3, 15),
              'min_samples_leaf': randint(1, 20), 'max_features': uniform(0.3, 0.7)}
random_search = RandomizedSearchCV(RandomForestClassifier(), param_dist,
                n_iter=50, cv=5, scoring='roc_auc', n_jobs=-1)
random_search.fit(X_train, y_train)
```

### Bayesian Optimization (Optuna — recommended)

```python
import optuna

def objective(trial):
    params = {
        'n_estimators':  trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':     trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'subsample':     trial.suggest_float('subsample', 0.5, 1.0),
    }
    model = XGBClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
```

---

## 6. Model Calibration

A well-calibrated model: when it predicts 80% probability, the event happens 80% of the time.

### Why Calibration Matters

```
Use case: fraud risk scoring → send to manual review if P(fraud) > 0.3
If model is uncalibrated: predicted 0.3 may actually correspond to P(fraud) = 0.1
→ wrong threshold → too many false positives in manual review

Random Forests:   tend to be well-calibrated
SVM:              poor calibration (decision boundary, not probability)
Naive Bayes:      often overconfident
Neural networks:  often overconfident (need temperature scaling)
```

### Reliability Diagram

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0,1], [0,1], '--', color='grey', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
```

### Calibration Methods

```python
# Platt scaling (logistic regression on top of model output)
calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)

# Isotonic regression (more flexible, needs more data)
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
```

---

## 6.5. Conformal Prediction (Distribution-Free Uncertainty)

Calibration tells you "when the model says 0.8, the event happens ~80% of the time" — at the population level. **Conformal prediction** gives a stronger, per-prediction guarantee with no distributional assumptions:

> Given a target error rate α (e.g., 10%), conformal prediction returns a prediction SET (classification) or prediction INTERVAL (regression) that contains the true label with probability ≥ 1−α, on any new exchangeable test point.

**Why this matters now (2024-2026 hot topic):** Works for ANY model — XGBoost, deep nets, even a black-box LLM call. No distributional assumption (unlike CLT-based intervals). Used in safety-critical ML (medical, autonomous driving), LLM hallucination filters ("if the prediction set has > 1 element, defer to human").

### The Recipe (split conformal — simplest variant)

1. Split data into train, calibration, test sets.
2. Train model on train set.
3. Compute "nonconformity score" s_i on calibration set:
   - Regression: `s_i = |y_i − ŷ_i|`
   - Classification: `s_i = 1 − ŷ(y_i, x_i)` (1 minus predicted prob of true class)
4. Set `q` = (1 − α) quantile of the calibration scores (with finite-sample correction).
5. For a new test point:
   - Regression: prediction interval = `[ŷ − q, ŷ + q]`
   - Classification: prediction set = `{y : ŷ(y|x) ≥ 1 − q}`

### Numeric Example (Regression)

```
α = 0.10  (we want 90% coverage)
Calibration set residuals: |y − ŷ| for 200 samples
Sorted, the (1−α)(n+1)/n = 0.9045 quantile is 4.2

For a new test point with ŷ = 23.5:
  90% prediction interval = [23.5 − 4.2, 23.5 + 4.2] = [19.3, 27.7]

Guarantee: in expectation, 90% of such intervals on future exchangeable test points
will contain the true y.
```

```python
from mapie.regression import MapieRegressor  # MAPIE = sklearn-style conformal
mapie = MapieRegressor(estimator=base_model, cv="prefit")
mapie.fit(X_cal, y_cal)
y_pred, y_intervals = mapie.predict(X_test, alpha=0.1)
# y_intervals[:, 0] = lower bound,  y_intervals[:, 1] = upper bound
```

**Senior interview answer:** "For uncertainty quantification I'd use bootstrap CI on aggregate metrics, calibration for per-class probabilities, and conformal prediction when I need a per-prediction guarantee — for example, 'flag the prediction for human review if the conformal set has more than one class.' Conformal is model-agnostic and finite-sample valid, which makes it the right tool for safety-critical or high-stakes per-decision settings."

---

## 6.6. Fairness Metrics (Beyond Accuracy)

When a model affects people (hiring, credit, healthcare), aggregate accuracy can hide group-level harm. Three core fairness definitions, all in conflict (impossibility theorem):

| Metric | Definition | When it matters |
|--------|-----------|----------------|
| Demographic parity | P(ŷ=1 \| A=a) equal across groups | Outcome rate should be group-independent (hiring quotas) |
| Equal opportunity | TPR equal across groups | Equally good at catching positives across groups (medical screening) |
| Equalized odds | TPR AND FPR equal across groups | Strongest — equal error profile across groups |
| Calibration within groups | Predicted prob matches actual within each group | Score interpretability across groups |

```python
# fairlearn — sklearn-compatible fairness
from fairlearn.metrics import MetricFrame, true_positive_rate, false_positive_rate
from fairlearn.metrics import demographic_parity_difference

mf = MetricFrame(metrics={"tpr": true_positive_rate, "fpr": false_positive_rate},
                 y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
print(mf.by_group)
print("DP gap:", demographic_parity_difference(y_test, y_pred, sensitive_features=A_test))
```

**Impossibility theorem (Kleinberg / Chouldechova):** when base rates differ across groups, you cannot simultaneously satisfy demographic parity, equalized odds, AND calibration. You have to pick which one matters for your use case and defend the choice. The senior interview move — not "I'd be fair," but "given different base rates, I'd prioritize equal opportunity because [domain reasoning], accepting that DP will not be exact."

---

## 7. Learning Curves (Diagnosing Bias vs Variance)

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='roc_auc',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1),   label='Validation')
plt.xlabel('Training size'); plt.ylabel('AUC'); plt.legend()
```

### Interpretation

```
High train, low val (large gap) = HIGH VARIANCE (overfitting)
  Fix: more data, regularization, simpler model, dropout

Both low (small gap) = HIGH BIAS (underfitting)
  Fix: more features, complex model, less regularization

Both high and converging = GOOD MODEL
  Adding more data won't help much (val already converged to train)
```

---

## 8. When to Use What Metric

| Business Problem | Metric | Reason |
|-----------------|--------|--------|
| Cancer screening | Recall (sensitivity) | Miss no actual cases — FN is catastrophic |
| Spam filter | Precision | Don't block legitimate emails — FP is unacceptable |
| Fraud detection | PR-AUC or F2 | Imbalanced; recall weighted higher than precision |
| Click prediction (ranking) | ROC-AUC | Ranking quality across all thresholds |
| House price | RMSE (if outliers ok) or MAE | Both common; RMSE penalizes big misses |
| Demand forecasting | MAPE or RMSE | MAPE gives % error, scale-independent |
| Search ranking | NDCG, MRR | Order matters, not just binary correct/wrong |
| Document OCR accuracy | Character Error Rate (CER) | Edit distance at character level |
| LLM-generated text | Faithfulness, answer relevance, context recall (RAGAS) + LLM-as-judge | No single ground truth; see `../../11.system_design/11_llm_evaluation_systems.md` |
| RAG pipeline | MRR@k for retrieval + RAGAS for generation | Decouple retrieval and generation eval |
| Per-prediction uncertainty | Conformal prediction set/interval | Distribution-free, finite-sample valid (§6.5) |
| Group fairness | Demographic parity / equal opportunity / equalized odds | When model affects people (§6.6) |

---

## 9. Gotchas

**Accuracy is almost always the wrong metric.** For any imbalanced problem (fraud, disease, anomaly), accuracy misleads. A model predicting all-negative achieves 99% accuracy on 1% fraud data. Always check class distribution before choosing accuracy.

**CV score ≠ production performance.** CV estimates generalization from the training distribution. If production data has distribution shift (different time period, different demographics), CV overestimates performance. Monitor model performance in production.

**Never tune on test set.** Each time you check test results and make a decision, you're implicitly using the test set for selection → test set becomes part of training. Use validation set for tuning, test set only once at the end.

**SMOTE inside CV, not outside.**
```python
# Wrong — SMOTE on full dataset before CV
X_res, y_res = SMOTE().fit_resample(X, y)
cross_val_score(model, X_res, y_res)  # test folds contain synthetic samples

# Right — SMOTE inside CV pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
pipeline = ImbPipeline([('smote', SMOTE()), ('model', model)])
cross_val_score(pipeline, X, y)  # SMOTE only applied to train fold each time
```

**Nested CV for unbiased hyperparameter tuning.** Standard CV + GridSearch with same CV → optimistic bias because the same folds used for tuning influence the performance estimate. Nested CV has two loops: inner CV selects best hyperparameters, outer CV evaluates the final model. This gives an unbiased performance estimate. Practical consideration: it's expensive (k_outer × k_inner × n_configs model fits). Used in academic papers for unbiased comparison; in industry, a separate validation set serves the same purpose.

---

## 10. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| CV AUC = 0.99+ on hard problem | Data leakage | Check features for future information; verify CV pipeline |
| Production AUC much lower than CV | Distribution shift or leakage | Analyze feature drift; retrain periodically |
| Train AUC >> Val AUC (large gap) | Overfitting | Regularization; more data; simpler model |
| Both train and val AUC low | Underfitting | More features; complex model; feature engineering |
| Precision high, Recall low | Threshold too high | Lower decision threshold |
| Recall high, Precision low | Threshold too low | Raise decision threshold |
| RMSE >> MAE | Large outliers in predictions | Check for extreme predictions; robust loss (Huber) |
| Calibration curve far from diagonal | Uncalibrated model | Apply Platt scaling or isotonic regression |

---

## 11. Interview Q&A (Senior Level)

**Q: When would you prefer PR-AUC over ROC-AUC?**
For imbalanced datasets where the positive class is rare. ROC-AUC uses FPR = FP/(FP+TN) — the denominator (TN) is huge for imbalanced data, so FPR stays small even with many false positives. This makes ROC-AUC look optimistic. PR-AUC uses Precision = TP/(TP+FP) — directly measures how many of your predicted positives are correct, without TN. For fraud (1% positive), a model might get ROC-AUC=0.95 but PR-AUC=0.3, which is much more honest about real-world usefulness.

**Q: You deploy a model. Two weeks later, performance drops. What do you check?**
(1) **Data drift** — check if feature distributions have shifted using PSI (Population Stability Index) or KS test between training and current production data. (2) **Label drift** — is the target distribution changing? (3) **Upstream pipeline** — did any data source change, or start sending nulls? (4) **Seasonal effects** — is the drop expected due to seasonality not captured in training? (5) **New patterns** — fraud strategies, user behavior changes. Then: retrain with recent data, monitor with early warning alerts on feature PSI > 0.2.

**Q: How do you handle the test set properly in a competition vs production?**
In competition: you have a fixed test set, one submission — use it once at the very end. Never base decisions on test performance. Tune on cross-validation. In production: the "test set" concept is replaced by a holdout set (fixed, recent period) used once before deployment, then continuous monitoring with new data as the ground truth arrives. The difference: production has no "final answer" — the model must be monitored and retrained as the world changes.

**Q: What is nested cross-validation and when is it necessary?**
Standard approach: use k-fold CV to both tune hyperparameters and estimate model performance → optimistic bias because the same folds used for tuning influence the performance estimate. Nested CV has two loops: inner CV selects best hyperparameters, outer CV evaluates the final model. This gives an unbiased performance estimate. Practical consideration: it's expensive (k_outer × k_inner × n_configs model fits). Used in academic papers for unbiased comparison; in industry, usually a separate validation set serves the same purpose.

---

## 12. Connections

| This file | Links to | Why |
|-----------|---------|-----|
| Imbalanced classes | `03_feature_engineering.md` | SMOTE, class weights — handle before evaluation |
| Bias-variance tradeoff | `01_statistics_foundations.md` | Statistical foundation of learning curves |
| Bootstrap CI for metrics | `01c_statistics_end_to_end.md` | Paired bootstrap for "is model B better?" |
| Leakage from preprocessing | `03_feature_engineering.md` | Scaling/encoding inside CV pipeline |
| Threshold tuning | `../02_algorithms/01_linear_models.md` | Logistic regression outputs probabilities to threshold |
| Hyperparameter spaces | `../02_algorithms/02_tree_models.md` | XGBoost/LightGBM typical hyperparameter ranges |
| Generative / LLM evaluation | `../../11.system_design/11_llm_evaluation_systems.md` | RAGAS, LLM-as-judge, golden set design |
| Production drift monitoring | `../../10.mlops/11_llm_observability_tools.md` | What "deployed perf drop" actually looks like in tooling |

---

## Key Takeaway

**The evaluation pipeline order:**
1. Split (train/val/test) — respecting time order if temporal data
2. EDA on train only
3. Feature engineering + preprocessing (fit on train, apply to val/test)
4. CV with stratification — always inside the pipeline
5. Hyperparameter tuning on CV score
6. Final evaluation on test set — once, at the very end

**Most common mistakes, ranked:**
1. Data leakage (preprocessing before split, or leaky features)
2. Using accuracy for imbalanced problems
3. Not using stratified CV for classification
4. Tuning on test set (looking at test results more than once)
5. Not using calibration when model outputs probabilities to business users

# 01 — Linear Models (Linear Regression, Logistic Regression, SVM)

## Quick Reference

| Model | Task | Key Assumption | Regularization |
|-------|------|---------------|----------------|
| Linear Regression | Regression | Linear relationship, normal residuals | Ridge (L2), Lasso (L1), ElasticNet |
| Logistic Regression | Binary/Multiclass classification | Linear decision boundary in feature space | L1, L2 |
| Ridge Regression | Regression | Linear + all features relevant | L2 (shrinks all coefficients) |
| Lasso Regression | Regression | Linear + sparse features | L1 (zeros out irrelevant features) |
| SVM (linear) | Classification | Margin maximization | C parameter |
| SVM (kernel) | Non-linear classification | Data separable in higher-dim space | C + γ (kernel width) |

---

## 1. Linear Regression

### Model

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = Xw

Loss (OLS): MSE = (1/n) Σ(yᵢ − ŷᵢ)²

Closed-form solution: w = (X'X)⁻¹X'y
O(n·d² + d³) — expensive for large d; use gradient descent instead
```

### Assumptions (OLS is BLUE when these hold)

1. **Linearity:** E[y|x] = Xw (relationship is linear)
2. **Independence:** observations are independent
3. **Homoscedasticity:** Var(ε) = σ² (constant variance of residuals)
4. **No multicollinearity:** features are not perfectly correlated
5. **Normal residuals:** ε ~ N(0, σ²) (needed for valid inference, not prediction)

### Checking Assumptions

```python
import matplotlib.pyplot as plt
import numpy as np

# Residual plot (check linearity + homoscedasticity)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red')
plt.xlabel('Predicted'); plt.ylabel('Residuals')
# Want: random scatter around 0, no pattern, constant spread

# Q-Q plot (check normality of residuals)
from scipy import stats
stats.probplot(residuals, dist='norm', plot=plt)

# VIF (check multicollinearity)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({'Feature': X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
# VIF > 10 = serious multicollinearity → drop or combine features
```

### When Linear Regression Fails

```
Nonlinear relationship  → use polynomial features or tree models
Heteroscedasticity      → use log(y) or weighted regression
Multicollinearity       → use Ridge or drop correlated features
Too many features       → use Lasso
Outliers in y           → use Huber loss or remove outliers
```

---

## 2. Regularization (Ridge, Lasso, ElasticNet)

### Why Regularize?

Without regularization, OLS minimizes training error → overfits (high variance). Regularization adds penalty for large coefficients → bias-variance tradeoff.

### Ridge Regression (L2)

```
Loss = MSE + λ Σwᵢ²

Effect: shrinks all coefficients toward 0, but never exactly 0
Best when: all features are somewhat relevant, multicollinearity present
```

### Lasso Regression (L1)

```
Loss = MSE + λ Σ|wᵢ|

Effect: drives irrelevant feature coefficients to exactly 0 = automatic feature selection
Best when: many irrelevant features, need interpretable sparse model
```

### ElasticNet (L1 + L2)

```
Loss = MSE + λ₁Σ|wᵢ| + λ₂Σwᵢ²

Effect: combines Lasso sparsity with Ridge grouping of correlated features
Best when: many features, some correlated, some irrelevant
```

### Geometric Intuition

```
Ridge: constraint region is a circle → optimal solution tangent to circle → rarely hits axis → non-zero
Lasso: constraint region is a diamond → optimal solution often hits corner → exactly zero
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV

# Choose λ with cross-validation
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {ridge_cv.alpha_}")

lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {lasso_cv.alpha_}")
print(f"Non-zero features: {(lasso_cv.coef_ != 0).sum()}")

# ElasticNet
from sklearn.linear_model import ElasticNetCV
en_cv = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5)
en_cv.fit(X_train_scaled, y_train)
```

---

## 3. Logistic Regression

### Model

```
P(y=1|x) = σ(w·x + b) = 1 / (1 + e^(-(w·x + b)))

Log-odds (logit): log(P/(1-P)) = w·x + b   ← Linear in features
Decision boundary: w·x + b = 0             ← a hyperplane

Loss: Binary Cross-Entropy
L = -(1/n) Σ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

No closed-form solution → optimize with gradient descent (LBFGS, Newton-CG, SGD)
```

### Interpreting Coefficients

```
Coefficient wⱼ = change in log-odds per unit increase in xⱼ
Odds ratio = exp(wⱼ) = multiplicative effect on odds

Example: w_age = 0.05 → exp(0.05) = 1.051 → each extra year increases odds by 5.1%
```

### Multiclass Extensions

```
One-vs-Rest (OvR):         train N binary classifiers, predict class with highest score
Multinomial (Softmax):     single classifier with softmax output — proper multiclass

sklearn: multi_class='ovr' (default) or multi_class='multinomial'
```

```python
from sklearn.linear_model import LogisticRegression

# Standard logistic regression with L2
lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Coefficients interpretation
coef_df = pd.DataFrame({'feature': X.columns, 'coef': lr.coef_[0],
                        'odds_ratio': np.exp(lr.coef_[0])})
print(coef_df.sort_values('coef', ascending=False))

# C is inverse of regularization strength: high C = less regularization
# Common search: C ∈ [0.001, 0.01, 0.1, 1, 10, 100]
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
from sklearn.model_selection import GridSearchCV
lr_cv = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, scoring='roc_auc')
lr_cv.fit(X_train_scaled, y_train)
```

---

## 4. Support Vector Machine (SVM)

### Core Idea

Find the hyperplane that **maximizes the margin** between classes. Margin = distance from hyperplane to nearest points (support vectors).

```
Decision boundary: w·x + b = 0
Margin: 2/||w||

Optimization:
  Minimize ||w||²/2   (maximize margin)
  Subject to: yᵢ(w·xᵢ + b) ≥ 1 for all i

With soft margin (C parameter):
  Minimize ||w||²/2 + C Σξᵢ
  ξᵢ = slack variable (how much misclassification is allowed)
  C large = hard margin (low bias, high variance)
  C small = soft margin (high bias, low variance)
```

### Kernel Trick

Map data to higher-dimensional space without explicitly computing coordinates — use kernel function K(xᵢ, xⱼ) = φ(xᵢ)·φ(xⱼ).

| Kernel | Formula | When to Use |
|--------|---------|-------------|
| Linear | xᵢ·xⱼ | Linearly separable data; large sparse features (text) |
| RBF (Gaussian) | exp(−γ\|\|xᵢ−xⱼ\|\|²) | Most common; general nonlinear; small-medium datasets |
| Polynomial | (γxᵢ·xⱼ + r)^d | Feature interactions matter; image data |
| Sigmoid | tanh(γxᵢ·xⱼ + r) | Sometimes for NLP (less common) |

```python
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

# Classification
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)

# Hyperparameter tuning (C and gamma interact)
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
svm_cv = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
svm_cv.fit(X_train_scaled, y_train)
```

### SVM vs Logistic Regression

| Aspect | SVM | Logistic Regression |
|--------|-----|---------------------|
| Loss | Hinge loss (margin-based) | Log loss (probabilistic) |
| Output | Decision function (no probability by default) | Calibrated probabilities |
| Kernel | Yes — nonlinear classification | No (linear only; use NN for nonlinear) |
| Speed | Slow for large n (O(n²-n³)) | Fast; scales to large datasets |
| Interpretability | Low (support vectors only) | High (coefficients = odds ratios) |
| Best for | Small-medium, high-dimensional (text, images) | Large datasets, need probabilities |

---

## 4.5. Beyond OLS — GLM and Quantile Regression

Linear regression assumes Gaussian errors. Real targets aren't always Gaussian — count data, skewed positive data, or "predict the 95th percentile" tasks need extensions.

### Generalized Linear Models (GLMs)

A GLM is a linear model with: (1) a **link function** g connecting Xw to the mean of the target, and (2) a non-Gaussian distribution from the exponential family.

| Family | Link | Use case |
|--------|------|----------|
| Gaussian | identity | Standard OLS |
| Binomial | logit | Logistic regression (already covered) |
| Poisson | log | Count data — clicks/day, defects/unit |
| Negative Binomial | log | Overdispersed counts (variance > mean) |
| Gamma | log | Positive skewed continuous — claim amounts, latencies |
| Tweedie | log | Insurance pure premium (mix of 0s + positives) |

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Poisson regression for count target
poisson_model = smf.glm(formula="num_events ~ feature1 + feature2",
                         data=df, family=sm.families.Poisson()).fit()
print(poisson_model.summary())

# scikit-learn alternative
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor
pr = PoissonRegressor(alpha=1.0).fit(X_train, y_train)
```

Interview tip: "If asked to predict counts (orders/day), I'd start with Poisson regression, not OLS. OLS on counts can predict negative values and assumes constant variance — Poisson is the natural choice and gives proper rate predictions."

### Quantile Regression

OLS predicts the conditional **mean** E[y|x]. Quantile regression predicts a specific conditional **percentile** — useful when you care about the tails, not the average.

```
Loss (pinball loss for quantile τ ∈ (0,1)):
  L_τ(y, ŷ) = max(τ·(y − ŷ), (τ-1)·(y − ŷ))

τ = 0.5:          median regression (robust to outliers, equivalent to MAE)
τ = 0.05 / 0.95:  lower / upper bound of 90% prediction interval
```

```python
from sklearn.linear_model import QuantileRegressor

qr_50 = QuantileRegressor(quantile=0.5, alpha=0).fit(X_train, y_train)
# 90% prediction interval via two models
qr_lo = QuantileRegressor(quantile=0.05, alpha=0).fit(X_train, y_train)
qr_hi = QuantileRegressor(quantile=0.95, alpha=0).fit(X_train, y_train)
```

Use cases:
- **Pricing / demand forecasting:** predict 10th / 50th / 90th percentile of demand — set safety stock from upper bound, not the mean
- **Latency SLOs:** optimize p95 latency directly, not average latency
- **Robust regression:** median regression (τ=0.5) handles outliers far better than OLS

Modern alternative: LightGBM/XGBoost support `objective='quantile'` natively — see `02_tree_models.md`

---

## 5. When to Use What

| Scenario | Model | Why |
|----------|-------|-----|
| Baseline for any tabular task | Logistic Regression | Fast, interpretable, often competitive |
| Many irrelevant features | Lasso | Auto-zeros irrelevant coefficients |
| All features potentially relevant | Ridge | Distributes weight across correlated features |
| Correlated features + some irrelevant | ElasticNet | Best of both |
| Nonlinear boundary, small dataset | SVM + RBF kernel | Powerful for small-medium n |
| High-dimensional sparse features (text) | SVM linear or LogReg + L1 | Efficient for sparse; large d |
| Need coefficient interpretability | Logistic/Linear Regression | Odds ratios, feature direction |
| Low latency inference | Linear or Logistic | Single dot product at inference |
| Imbalanced classes | LogReg with `class_weight='balanced'` | Built-in support |

---

## 6. Gotchas

**Scale features before any linear model, SVM, or KNN.** Coefficients and regularization penalties are scale-sensitive. A feature with range [0, 1000] gets penalized differently than one with range [0, 1] under the same λ. Always StandardScale before fitting.

**High C in SVM = overfitting; Low C = underfitting.** Opposite intuition from regularization: C is the inverse of regularization strength in SVM. High C → allow fewer margin violations → smaller, tighter margin → overfit.

**Logistic Regression is NOT suitable for non-linear boundaries.** It finds a hyperplane. For nonlinear data: add polynomial features, use kernelized SVM, or use tree-based models.

**Lasso behaves unexpectedly with correlated features.** If features X1 and X2 are highly correlated, Lasso arbitrarily zeros one and keeps the other. Which one it keeps is sensitive to data perturbation. Use ElasticNet when correlated features are present.

**max_iter in LogisticRegression.** Default is 100 — often too low. Always set max_iter=1000+ or until convergence warning disappears. Convergence failure means coefficients are unreliable.

**SVM doesn't scale well.** Training is O(n²) to O(n³) in samples. Beyond 10K-50K samples: SVM becomes slow. Use LinearSVC (faster) or LogisticRegression for large datasets.

---

## 7. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Linear model coefficients very large | Features not scaled | Apply StandardScaler |
| Logistic Regression ConvergenceWarning | max_iter too low | Increase max_iter; try solver='lbfgs' |
| Ridge doesn't zero any features | L2 never zeros exactly | Use Lasso instead |
| Lasso zeros too many features | λ (α) too high | Reduce α; use LassoCV |
| SVM very slow on large dataset | O(n²) complexity | Use LinearSVC or LogisticRegression |
| SVM poor performance | Kernel/C/γ not tuned | Grid search C + gamma; always scale features |
| Coefficients don't match intuition | Multicollinearity | Check VIF; use Ridge or drop correlated features |

---

## 8. Code Reference

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import numpy as np

# Full pipeline with Scaling (prevents leakage)
logreg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000))
])

ridge_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(kernel='rbf', C=10, gamma='scale', probability=True))
])

# CV evaluation
for name, pipe in [('LogReg', logreg_pipe), ('Ridge', ridge_pipe), ('SVM', svm_pipe)]:
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## 9. Interview Q&A (Senior Level)

**Q: Why does L1 regularization produce sparse solutions while L2 doesn't?**
Geometric interpretation: L1 constraint region is a diamond (corners at axes), L2 is a circle. The optimal solution is where the loss function contour first touches the constraint region. For L1, this intersection almost always occurs at a corner (where some coordinates = 0) because the diamond has flat sides leading to corners. For L2, the smooth circular boundary means the intersection can occur anywhere — rarely exactly at an axis. Algebraically: the L1 subgradient at wⱼ=0 includes the entire interval [-λ, λ], so the subgradient optimality condition holds at exactly zero. For L2, the gradient at zero is very soft — it can only drive wⱼ toward zero, never exactly there.

**Q: When would you choose SVM over Logistic Regression for a classification task?**
SVM with kernel is preferred when: (1) data is not linearly separable and you need nonlinear boundaries but don't want the overhead of a neural network, (2) dataset is small-medium (< 50K samples) where SVM's quadratic complexity is manageable, (3) high-dimensional feature spaces (e.g., text TF-IDF) where kernel SVM can be very effective. Logistic Regression is preferred when: (1) you need calibrated probabilities (SVM outputs uncalibrated decision scores), (2) dataset is large (> 100K samples), (3) you need coefficient interpretability (odds ratios), (4) online learning (logistic regression adapts easily with SGD). In practice, for large datasets, XGBoost usually beats both.

**Q: What happens to linear regression coefficients when features are highly correlated?**
Multicollinearity inflates coefficient variance — small changes in the data lead to large changes in coefficients. Geometrically, the X'X matrix becomes near-singular → (X'X)⁻¹ has huge values → coefficients are unstable. Practically: two correlated features might get coefficients of +100 and -99, canceling each out — economically meaningless but algebraically valid. Ridge regression addresses this by adding λI to X'X before inverting, making it well-conditioned. VIF > 10 is the rule of thumb for "serious multicollinearity."

---

## 10. Connections

| This file | Links to | Why |
|-----------|---------|-----|
| Scaling requirement | `../01_fundamentals/03_feature_engineering.md` | StandardScaler, RobustScaler |
| Regularization and bias-variance | `../01_fundamentals/01_statistics_foundations.md` | Bias-variance tradeoff |
| Logistic Regression as DL head | `../../2.deep learning/02_architectures/01_mlp.md` | Final layer of classifier is logistic regression |
| Cross-entropy loss derivation | `../../2.deep learning/01_fundamentals/01_foundations.md` | LogReg uses same CE loss as DL |
| SVM kernel vs attention | `../../2.deep learning/02_architectures/04_transformer.md` | Both compute similarity in feature space |
| Quantile / Poisson alternatives | `02_tree_models.md` | Same objectives in LightGBM/XGBoost |
| Calibration of LogReg outputs | `../01_fundamentals/04_model_evaluation.md` | Platt scaling, isotonic, conformal |

---

## Key Takeaway

**Linear models are your baseline — always train them first.** They're fast, interpretable, and often competitive. If they perform well, don't overcomplicate.

**Regularization selection:**
- Ridge: correlated features, keep all
- Lasso: feature selection, sparse model
- ElasticNet: correlated + sparse

**SVM:** powerful for small datasets and nonlinear boundaries, but slow and unscalable. Replaced by XGBoost or neural networks for most production tasks.

**Golden rule:** scale first, always. Linear models and SVMs are completely scale-sensitive. Unscaled features → regularization doesn't work. SVM kernel distances are meaningless.

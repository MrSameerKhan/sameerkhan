# 01 — Statistics Foundations

## Quick Reference

| Concept | Formula | When it matters |
|---------|---------|----------------|
| Mean | Σx / n | Central tendency, sensitive to outliers |
| Median | Middle value | Robust central tendency for skewed data |
| Variance | Σ(x-μ)² / n | Spread; used in PCA, regularization |
| Std Dev | √Variance | Same units as data; interpret spread |
| Covariance | Σ(x-μ)(y-μ) / n | Direction of linear relationship |
| Pearson r | Cov(X,Y) / (σx·σy) | Normalized correlation [-1, 1] |
| p-value | P(data \| H₀ true) | Reject H₀ if p < α (typically 0.05) |
| Central Limit Theorem | X̄ ~ N(μ, σ²/n) as n→∞ | Foundation of inference on sample means |

---

## 1. Descriptive Statistics

### Measures of Central Tendency

```
Mean = (1 + 2 + 3 + 4 + 100) / 5 = 22  ← pulled by outlier
Median = 3                               ← robust
Mode = most frequent value (categorical)
```

**Trimmed mean:** discard top/bottom k% before averaging — robust to extreme outliers. Used in finance, benchmarks.

### Measures of Spread

```
Variance (population): σ² = Σ(xᵢ - μ)² / N
Variance (sample):     s² = Σ(xᵢ - x̄)² / (N-1)  ← Bessel's correction (unbiased)
Standard deviation: s = √s²
IQR: Q3 - Q1  ← robust to outliers, used in boxplots
MAD (Median Absolute Deviation): median(|xᵢ - median(x)|) ← most robust spread estimate
```

**Why N-1 (Bessel's correction)?** Sample mean is computed from the same data — degrees of freedom reduced by 1 — dividing by N underestimates population variance.

### Percentiles and Quartiles

```
Q1 = 25th percentile
Q2 = 50th percentile = median
Q3 = 75th percentile
IQR = Q3 - Q1

Outlier rule (Tukey): x < Q1 - 1.5·IQR  or  x > Q3 + 1.5·IQR
```

---

## 2. Probability Distributions

### Key Distributions and When They Appear in ML

| Distribution | Parameters | Use Case |
|-------------|------------|----------|
| Normal N(μ,σ²) | mean, std | Feature assumption in linear models, noise |
| Bernoulli | p | Binary outcome (click/no-click) |
| Binomial B(n,p) | n trials, p | Count of successes in n binary trials |
| Poisson λ | rate | Count data (events per unit time) |
| Uniform U(a,b) | min, max | Prior in Bayesian; random initialization |
| Exponential | λ | Time between events (inter-arrival) |
| Beta | α, β | Prior for probabilities (Bayesian) |
| Categorical | p₁,...,pK | Multiclass output distribution |

### Normal Distribution Properties

```
68% of data within μ ± 1σ
95% of data within μ ± 2σ
99.7% of data within μ ± 3σ

Standard normal: Z = (X - μ) / σ ~ N(0, 1)
```

**In ML context** — Features should be roughly normal for linear models (not required for trees) — Residuals should be normal for valid linear regression inference — Weight initialization assumes normal distribution (Xavier/He)

### Log-Normal Distribution

If log(X) ~ Normal, then X ~ Log-Normal. Appears naturally in: income, document lengths, word frequencies, time-to-event data. Fix: log-transform the feature before modeling.

---

## 3. Correlation and Covariance

### Pearson Correlation

```
r = Cov(X,Y) / (σx · σy)  ∈ [-1, 1]

r =  1: perfect positive linear relationship
r = -1: perfect negative linear relationship
r =  0: no linear relationship (may still have nonlinear)
```

**Numeric example:**
```
X = [1, 2, 3, 4, 5]
Y = [2, 4, 6, 8, 10]   (Y = 2X)
r = 1.0 (perfect positive)
```

### Spearman Correlation

Rank-based correlation — handles monotonic nonlinear relationships and outliers.

```
ρ = Pearson(rank(X), rank(Y))
```

Use Spearman when: data has outliers, ordinal variables, or non-normal distributions.

**Correlation ≠ Causation**

Classic example: ice cream sales and drowning rates are correlated (both driven by summer). In ML: correlated features don't both cause the target — multicollinearity inflates linear model variance.

---

## 4. Hypothesis Testing

### Framework

```
H₀ (null hypothesis):   no effect, no difference (status quo)
Hₐ (alternative):       there IS an effect/difference

α (significance level):  0.05 (5% false positive rate we accept)
p-value:                 P(observing this data or more extreme | H₀ is true)

Decision:
p ≤ α → reject H₀ (result is "statistically significant")
p > α → fail to reject H₀ (not enough evidence)
```

**p-value misconceptions:** p-value is NOT the probability H₀ is true · p-value is NOT the probability results are due to chance · Small p-value does NOT mean large practical effect.

### Common Tests

| Test | When to Use | What it Tests |
|------|------------|---------------|
| t-test (1 sample) | Is mean = target value? | Sample mean vs hypothesized value |
| t-test (2 sample) | Do two groups have same mean? | A/B test for continuous metric |
| Paired t-test | Same subjects, two conditions | Before/after treatment |
| Chi-squared test | Are two categorical vars independent? | Feature-target association |
| ANOVA | Are 3+ group means equal? | Multi-group continuous comparison |
| Mann-Whitney U | Non-parametric 2-group comparison | When normality can't be assumed |

### Type I and II Errors

```
                H₀ True           H₀ False
Reject H₀:     Type I Error (α)   Correct (Power = 1-β)
Fail to reject: Correct            Type II Error (β)

α = false positive rate (reject H₀ when it's actually true)
β = false negative rate (fail to reject H₀ when it's actually false)
Power = 1 - β = probability of detecting a real effect
```

**In ML context** — Precision/Recall tradeoff = Type I/II error tradeoff — A/B testing: α=0.05 means 1 in 20 "significant" results is a false positive by chance

### Statistical vs Practical Significance

With large n, tiny meaningless differences become "statistically significant." Always check **effect size** (Cohen's d, η²) alongside p-value.

```
Cohen's d = (μ₁ - μ₂) / pooled_σ
Small: 0.2,  Medium: 0.5,  Large: 0.8
```

---

## 5. Central Limit Theorem (CLT)

**Statement:** Regardless of the population distribution, the distribution of sample means approaches Normal as sample size n → ∞.

```
X̄ ~ N(μ, σ²/n)  for large n (n ≥ 30 rule of thumb)

Standard error of the mean: SE = σ / √n
```

**Why it matters in ML** — Foundation for confidence intervals and hypothesis tests on model metrics — Why we can use t-tests for A/B testing even if the raw metric isn't normal — Explains why averaging predictions (ensemble) reduces variance by √n

### Confidence Intervals

```
95% CI for mean: X̄ ± 1.96 · (σ/√n)

Interpretation: if we repeated sampling 100 times, ~95 of those intervals
would contain the true population mean.
(NOT: 95% probability that THIS interval contains the mean)
```

### Bootstrap (Distribution-Free CIs)

When the CLT-based formula X̄ ± 1.96·(σ/√n) doesn't apply — small n, skewed metric, or you want a CI for something non-trivial like the median, AUC, or 95th percentile — resample with replacement and read percentiles off the resampling distribution.

```
1. Sample n rows from your data WITH replacement (a "bootstrap sample")
2. Compute the statistic of interest (mean, AUC, median, ...)
3. Repeat B = 1000+ times → get a distribution of the statistic
4. 95% CI = [2.5th percentile, 97.5th percentile] of that distribution
```

Why it works: the bootstrap distribution approximates the sampling distribution of the statistic. No normality assumption required.

In ML standard tool for reporting CIs on accuracy / F1 / AUC of a single trained model when you only have one test set. Pair with **paired bootstrap** to compare two models on the same test set (resample rows together, compute Δmetric, get a CI on the difference). Worked example in `01c_statistics_end_to_end.md`

---

## 6. Bayes' Theorem

```
P(A|B) = P(B|A) · P(A) / P(B)

Posterior = Likelihood × Prior / Evidence
```

### Classic ML example (spam filter):

```
P(spam | "free") = P("free" | spam) · P(spam) / P("free")

P(spam) = 0.3        (prior: 30% of emails are spam)
P("free"|spam) = 0.9 (90% of spam contains "free")
P("free") = 0.3·0.9 + 0.7·0.1 = 0.34

P(spam|"free") = 0.9 · 0.3 / 0.34 = 0.79
```

**Why it matters in ML** — Foundation of Naive Bayes classifier — Bayesian inference: update beliefs with evidence — MAP estimation: Maximum A Posteriori (like regularization — adds prior on weights)

---

## 7. Information Theory Basics

### Entropy

```
H(X) = -Σ p(x) · log₂(p(x))  [bits]

Uniform distribution = maximum entropy (maximum uncertainty)
One-hot distribution = zero entropy (no uncertainty)
```

**Binary entropy:**
```
H(p) = -p·log₂(p) - (1-p)·log₂(1-p)
H(0.5) = 1.0 bit   (maximum uncertainty)
H(0.9) = 0.47 bit  (mostly certain)
```

### KL Divergence

```
KL(P || Q) = Σ P(x) · log(P(x) / Q(x))

Measures how much Q diverges from P.
Not symmetric: KL(P||Q) ≠ KL(Q||P)
KL = 0 when P = Q
```

### Cross-Entropy (your loss function)

```
H(P, Q) = H(P) + KL(P || Q) = -Σ P(x) · log Q(x)

In classification: P = true labels (one-hot), Q = predicted probabilities
CE = -log(q_true_class) = -log(model's probability for correct class)
```

Minimizing cross-entropy loss ≡ minimizing KL divergence between predicted and true distribution.

---

## 8. When to Use What

| Situation | Tool | Notes |
|-----------|------|-------|
| Summarize continuous feature | Mean + std (normal) or Median + IQR (skewed) | Check skewness first |
| Test A/B experiment | Two-sample t-test (or Mann-Whitney if non-normal) | Check sample size for power |
| Check feature-target association (categorical) | Chi-squared test | Expected cell count ≥ 5 |
| Check feature-target association (continuous) | Pearson or Spearman correlation | Spearman if outliers present |
| Report model metric uncertainty | Confidence interval (bootstrap or CLT) | Always report CI, not just point estimate |
| Prior knowledge on parameters | Bayesian inference | When data is limited |

---

## 9. Gotchas

**Correlation doesn't imply causation — but collinear features hurt linear models.** Two highly correlated features → inflated coefficient variance → unstable predictions. Check VIF (Variance Inflation Factor). VIF > 10 → multicollinearity problem.

**p-hacking / multiple comparisons.** Run 20 tests at α=0.05 → expect 1 false positive by chance. Use Bonferroni correction (α/n tests) or FDR (Benjamini-Hochberg) when testing many hypotheses simultaneously.

**Sample size matters for CLT.** n ≥ 30 is a rough rule. Heavy-tailed distributions (Pareto, log-normal) need larger n. Skewed distributions → CLT kicks in slower.

**Outliers distort mean and Pearson correlation.** One extreme value can shift both. Always plot before computing. Use Median + Spearman for robustness.

**Statistical significance ≠ practical significance.** With n=1M, a 0.001 difference in conversion rate is "significant" but worthless for business. Always compute effect size.

**Normal distribution assumption in linear models.** Linear regression doesn't require features to be normal — it requires **residuals** to be normal for valid inference (confidence intervals, p-values on coefficients). For prediction only, normality of residuals matters less.

---

## 10. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Feature mean ≠ median (large gap) | Skewed distribution / outliers | Log-transform or use robust statistics |
| Pearson r = 0 but clear relationship visible | Non-linear relationship | Use Spearman or mutual information |
| A/B test always significant | p-hacking / peeking at results | Fix sample size before experiment; use sequential testing |
| Linear model coefficients unstable across runs | Multicollinearity | Check VIF; drop or combine correlated features |
| Chi-squared test warning about expected counts | Small sample size | Use Fisher's exact test instead |
| Confidence interval includes 0 | Effect not significant at chosen α | Report CI alongside p-value; increase sample size |

---

## 11. Code Reference

```python
import numpy as np
import pandas as pd
from scipy import stats

# Descriptive statistics
x = np.array([1, 2, 3, 4, 100])
print(f"Mean: {x.mean():.2f}, Median: {np.median(x):.2f}, IQR: {np.percentile(x,75)-np.percentile(x,25):.2f}")

# Correlation
df = pd.DataFrame({'x': [1,2,3,4,5], 'y': [2,4,5,4,5]})
print(df.corr(method='pearson'))   # linear correlation
print(df.corr(method='spearman'))  # rank correlation

# Two-sample t-test (A/B test)
group_a = np.random.normal(10, 2, 1000)
group_b = np.random.normal(10.5, 2, 1000)
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t={t_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
cohens_d = (group_a.mean() - group_b.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")

# Chi-squared test (categorical feature vs target)
observed = np.array([[50, 30], [20, 100]])
chi2, p, dof, expected = stats.chi2_contingency(observed)
print(f"chi2={chi2:.3f}, p={p:.4f}")

# Bootstrap confidence interval for any metric
def bootstrap_ci(data, statistic=np.mean, n_boot=1000, ci=95):
    boot_stats = [statistic(np.random.choice(data, len(data))) for _ in range(n_boot)]
    lo = np.percentile(boot_stats, (100-ci)/2)
    hi = np.percentile(boot_stats, 100-(100-ci)/2)
    return lo, hi

lo, hi = bootstrap_ci(group_a)
print(f"95% CI for mean: [{lo:.3f}, {hi:.3f}]")
```

---

## 12. Interview Q&A (Senior Level)

**Q: What's the difference between a confidence interval and a credible interval?** A: Confidence interval (frequentist): if we repeated the experiment infinite times, 95% of computed intervals would contain the true parameter. It says nothing about the probability that THIS specific interval contains the parameter. Credible interval (Bayesian): given the data and prior, there's a 95% probability that the parameter lies in this range. The Bayesian interpretation is more intuitive but requires a prior. In practice, with large n and uninformative priors, they converge to the same interval.

**Q: You run an A/B test. p=0.04. Should you ship the feature?** A: Not automatically. Questions to ask: (1) What's the effect size — is this a meaningful improvement or just statistically detectable? (2) Did you pre-register the sample size before starting, or did you peek and stop early? Peeking inflates false positive rate. (3) Was this one of many tests run simultaneously? Multiple comparisons require Bonferroni or FDR correction. (4) Are there segment effects — positive overall but negative for a user subset? Ship if: pre-registered sample size reached, effect size is meaningful, no signs of p-hacking, consistent across key segments.

**Q: When would you use non-parametric tests instead of t-tests?** A: When the normality assumption is violated and sample size is too small for CLT to kick in (n < 30), or when you have ordinal data (not truly continuous), or when there are extreme outliers you can't remove. Mann-Whitney U (2 groups), Kruskal-Wallis (3+ groups), Wilcoxon signed-rank (paired). Tradeoff: less statistical power than parametric tests when assumptions ARE met, but robust when they're not.

**Q: Explain the bias-variance tradeoff in statistical terms.** A: For an estimator θ̂ of true θ: MSE = Bias² + Variance. Bias = E[θ̂] − θ (systematic error — model consistently under/over-estimates). Variance = E[(θ̂ − E[θ̂])²] (sensitivity to training data fluctuations). Regularization (L1/L2) reduces variance by shrinking coefficients toward zero, accepting some bias. Ensemble methods (bagging) reduce variance by averaging. Boosting reduces bias by iteratively correcting errors. The optimal model minimizes the sum, not either term alone.

---

## 13. Connections

| This file | Links to | Why |
|-----------|---------|-----|
| Hypothesis testing | `04_model_evaluation.md` | Statistical tests for model comparison |
| Cross-entropy | `../../2.deep learning/01_fundamentals/01_foundations.md` | CE loss derivation from KL divergence |
| Bayes' theorem | `../02_algorithms/04_probabilistic.md` | Foundation of Naive Bayes and Bayesian models |
| Correlation / VIF | `03_feature_engineering.md` | Feature selection — remove correlated features |
| CLT / confidence intervals | `04_model_evaluation.md` | Reporting metric uncertainty with CI |
| EM algorithm | `../02_algorithms/08_expectation_maximization.md` | Latent variable models, GMM, missing data |
| Gaussian processes | `../02_algorithms/09_gaussian_processes.md` | Bayesian regression with calibrated uncertainty |
| Bootstrap worked example | `01c_statistics_end_to_end.md` | Numeric bootstrap CI walkthrough |
| Conformal prediction | `04_model_evaluation.md` | Distribution-free uncertainty quantification |

---

## Key Takeaway

**Descriptive:** mean/median (central), std/IQR (spread), correlation (relationship). **Inferential:** hypothesis test → p-value → reject/fail-to-reject H₀ → always check effect size too. **CLT:** sample means → normal → foundation of probabilistic ML.

The most common ML interview trap: confusing statistical significance with practical significance. Always pair p-value with effect size.

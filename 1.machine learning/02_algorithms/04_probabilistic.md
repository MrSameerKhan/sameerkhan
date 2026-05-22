# 04 — Probabilistic Models (Naive Bayes, GMM, Bayesian Inference)

## Quick Reference

| Model | Core Assumption | Best For |
|-------|----------------|----------|
| Naive Bayes (Gaussian) | Features independent given class; continuous features normal | Fast baseline, continuous features |
| Naive Bayes (Multinomial) | Features are counts/frequencies | Text classification (TF, count vectors) |
| Naive Bayes (Bernoulli) | Features are binary | Binary word presence for text |
| GMM (EM) | Data is mixture of Gaussians | Soft clustering, density estimation |
| Bayesian Ridge | Gaussian prior on weights | Uncertainty quantification in regression |
| Bayesian Optimization | Gaussian Process prior on objective | Hyperparameter tuning, expensive black-box |

---

## 1. Naive Bayes

### Bayes' Theorem for Classification

```
P(y|x1,...,xn) = P(y) * P(x1,...,xn|y)

Naive assumption: features are conditionally independent given y
  P(x1,...,xn|y) = ∏ P(xi|y)

Therefore:
  P(y|x) ∝ P(y) * ∏ P(xi|y)

Prediction: ŷ = argmax_y P(y) * ∏ P(xi|y)
```

In practice: use log probabilities (avoid numerical underflow from multiplying small numbers)

```python
ŷ = argmax_y [log P(y) + Σ log P(xi|y)]
```

### Gaussian Naive Bayes

Assumes each feature follows a Gaussian distribution within each class:

```
P(xi|y=k) = N(xi; μik, σik²)
```

During training: estimate μik = mean of feature i for class k, σik² = variance of feature i for class k

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB(var_smoothing=1e-9)   # add small variance to avoid zero variance
gnb.fit(X_train, y_train)

# Access learned parameters
print(gnb.theta_)       # [n_classes, n_features] - class-conditional means
print(gnb.var_)         # [n_classes, n_features] - class-conditional variances
print(gnb.class_prior_) # P(y=k) for each class

probs = gnb.predict_proba(X_test)  # [n_samples, n_classes]
```

### Naive Bayes — Dry Run (Spam Classification)

**Setup:** Vocabulary = {free, money, meeting, tomorrow}. Two training documents per class.

```
Training corpus:
SPAM: "free money"     → {free:1, money:1, meeting:0, tomorrow:0}
SPAM: "free free money" → {free:2, money:1, meeting:0, tomorrow:0}
HAM:  "meeting tomorrow"→ {free:0, money:0, meeting:1, tomorrow:1}
      "tomorrow meeting"→ {free:0, money:0, meeting:1, tomorrow:1}

Count totals (with Laplace α=1):
P(spam) = 2/4 = 0.5,  P(ham) = 2/4 = 0.5

Word counts in SPAM (raw): free=3, money=2, meeting=0, tomorrow=0 → total=5
Word counts in HAM  (raw): free=0, money=0, meeting=2, tomorrow=2 → total=4

P(free|spam)    = (3+1)/(5+4) = 4/9 = 0.444
P(money|spam)   = (2+1)/(5+4) = 3/9 = 0.333
P(meeting|spam) = (0+1)/(5+4) = 1/9 = 0.111
P(tomorrow|spam)= (0+1)/(5+4) = 1/9 = 0.111

P(free|ham)     = (0+1)/(4+4) = 1/8 = 0.125
P(money|ham)    = (0+1)/(4+4) = 1/8 = 0.125
P(meeting|ham)  = (2+1)/(4+4) = 3/8 = 0.375
P(tomorrow|ham) = (2+1)/(4+4) = 3/8 = 0.375

Classify test message: "free money tomorrow"

  log P(spam|x) = log(0.5) + log(0.444) + log(0.333) + log(0.111)
                = -0.693 + (-0.811) + (-1.099) + (-2.198)
                = -4.801

  log P(ham|x)  = log(0.5) + log(0.125) + log(0.125) + log(0.375)
                = -0.693 + (-2.079) + (-2.079) + (-0.981)
                = -5.832

-4.801 > -5.832 → predict SPAM
```

Note: "tomorrow" shifted ham toward spam slightly, but "free money" (high P(word|spam)) dominated.

### Multinomial Naive Bayes (Text Classification)

Assumes features are non-negative integer counts (word frequencies):
```
P(xi|y=k) = θik^xi   where θik = count(word i in class k) / total words in class k
```

Laplace smoothing (α): add α to all counts to avoid zero probability for unseen words
```
θik = (count(word i in class k) + α) / (total words in class k + α * vocab_size)
```

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

# Text classification pipeline
text_clf = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', MultinomialNB(alpha=1.0))   # alpha = Laplace smoothing
])

text_clf.fit(X_train_text, y_train)
```

### Bernoulli Naive Bayes (Binary Features)

Assumes features are binary (word present/absent, not count):
```
P(xi=1|y=k) = pik
P(xi=0|y=k) = 1 - pik
```

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# Binary word presence (not count)
binarize_vec = CountVectorizer(binary=True)
X_binary = binarize_vec.fit_transform(texts)

bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_binary, y_train)
```

### Complement Naive Bayes (Imbalanced Text)

Better than MultinomialNB for imbalanced classes — trains on complement of each class:

```python
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB(alpha=1.0)
cnb.fit(X_train_tfidf, y_train)
```

### Naive Bayes — When It Fails and Why

The "naive" assumption (feature independence) is almost always wrong in practice. Despite this, NB often works well because:
- Correct posterior ordering even if calibration is off
- Very sample-efficient — estimates fewer parameters
- Robust to irrelevant features (they contribute equally to all classes)

**Fails when:**
- Features are highly correlated (e.g., "New" and "York" in text — NB double-counts)
- Feature-class relationship is nonlinear
- Need well-calibrated probabilities (NB probabilities are often overconfident)

### When to Use Naive Bayes

```
✓ Text classification (spam, sentiment, topic) — fast, effective baseline
✓ Very small datasets — needs fewer samples than discriminative models
✓ Real-time inference — scoring is just sum of log-probabilities
✓ Multi-label classification — each class independently
✓ Missing values — simply skip missing features in predict

✗ High-accuracy requirement with sufficient data → use XGBoost or DL
✗ Correlated features → calibration is poor
✗ Non-text tabular data → usually worse than LightGBM
```

---

## 2. Gaussian Mixture Models (GMM) + EM Algorithm

### GMM Setup

```
Data assumed to come from K Gaussian distributions (components):
  p(x) = Σk πk * N(x; μk, Σk)

πk = mixing coefficient (weight of component k), Σπk=1
μk = mean of component k
Σk = covariance matrix of component k
```

### EM Algorithm (Expectation-Maximization)

GMM has latent variables (which component each point came from) — can't directly maximize likelihood → use EM.

```
E-step (Expectation): Compute soft assignments (responsibilities)
  r_ik = P(z=k|xi, Σk) = πk * N(xi; μk, Σk) / Σj πj * N(xi; μj, Σj)

M-step (Maximization): Update parameters using weighted data
  Nk = Σi r_ik              (effective number of points in component k)
  πk = Nk / n               (update mixing coefficient)
  μk = (1/Nk) Σi r_ik * xi  (weighted mean)
  Σk = (1/Nk) Σi r_ik(xi-μk)(xi-μk)^T  (weighted covariance)

Repeat E-step and M-step until log-likelihood converges (always increases)
```

**EM properties:** Guaranteed to converge (likelihood never decreases). Converges to local maximum (not necessarily global) — run multiple initializations. K-Means is a special case of EM with hard assignments (responsibilities ∈ {0,1}) and spherical equal covariances.

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# Model selection: find best K using BIC (lower = better)
bics = []
for k in range(1, 11):
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                          n_init=5, random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))

best_k = np.argmin(bics) + 1
print(f"Best K (BIC): {best_k}")

# Fit final model
gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                      n_init=10, random_state=42)
gmm.fit(X_scaled)

# Soft assignments
probs  = gmm.predict_proba(X_scaled)  # [n_samples, K] - P(component k | xi)
labels = gmm.predict(X_scaled)         # hard assignment = argmax

# Log-likelihood of new data (use for anomaly detection)
log_likelihood = gmm.score_samples(X_test)
# Low log-likelihood = anomalous point (doesn't fit any Gaussian well)
```

### Covariance Type Selection

```python
# 'full': each component has its own full covariance matrix (most flexible)
# 'tied': all components share same covariance matrix (fewer parameters)
# 'diag': each component has diagonal covariance (assumes feature independence)
# 'spherical': each component has a single variance (like K-Means with soft assignments)

# Rule: start with 'full'; if overfitting, try 'tied' or 'diag'
```

---

## 3. Bayesian Linear Regression

### Frequentist vs Bayesian Regression

```
Frequentist (OLS): find single best weights w* that minimize MSE
  No uncertainty on w — point estimate

Bayesian: treat weights as random variables with prior distribution
  Prior: P(w) = N(0, α⁻¹I)  (Gaussian prior — equivalent to Ridge)
  Likelihood: P(y|X,w) = N(Xw, β⁻¹I)
  Posterior: P(w|X,y) ∝ P(y|X,w) * P(w) — still Gaussian for Gaussian prior+likelihood

Prediction: P(y*|x*, y) = ∫ P(y*|x*,w) P(w|X,y) dw
  = Gaussian predictive distribution with mean AND uncertainty estimate
```

**Why Bayesian regression?** Returns uncertainty on predictions (confidence intervals, not just point estimates). Naturally handles small datasets (prior regularizes). MAP estimate with Gaussian prior = Ridge regression.

```python
from sklearn.linear_model import BayesianRidge

# Automatically infers regularization from data (no manual alpha tuning)
br = BayesianRidge(compute_score=True)
br.fit(X_train, y_train)

# Get uncertainty estimates
y_pred, y_std = br.predict(X_test, return_std=True)

# 95% confidence interval
ci_lower = y_pred - 1.96 * y_std
ci_upper = y_pred + 1.96 * y_std
```

---

## 4. Bayesian Optimization (for Hyperparameter Tuning)

### Why Better Than Random/Grid Search?

- Random search: blind to past results — might try the same bad region twice
- Bayesian BO: fits a surrogate model (Gaussian Process) on past (params → metric) observations, uses acquisition function to choose next point to query intelligently

```
Surrogate model: GP models the objective function's distribution
  After each trial: update posterior belief about objective

Acquisition function: balance exploration vs exploitation
  Expected Improvement (EI): choose next point with max expected improvement over best so far
  Upper Confidence Bound (UCB): choose point where UCB of GP is highest
```

```python
# Optuna (most practical for ML)
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
    }

    model = XGBClassifier(**params, random_state=42)
    cv_score = cross_val_score(model, X_train, y_train, cv=5,
                               scoring='roc_auc').mean()
    return cv_score

# Optuna uses Tree-structured Parzen Estimator (TPE), a form of Bayesian optimization
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100, timeout=3600)

print(f"Best AUC: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualization
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

---

## 5. Hidden Markov Models (HMM)

### What is an HMM?

An HMM models a sequence where: **Hidden states** are unobservable (e.g., Part-of-Speech tags: NOUN, VERB, DET). **Observations** are what we see (e.g., words: "the", "cat", "runs"). We want to infer the hidden state sequence given the observations.

**5 components:**
```
S = {s1, s2, ..., sN}    Hidden states (e.g., {NOUN, VERB, DET})
O = {o1, o2, ..., oT}    Observation symbols (e.g., vocabulary)
A = [aij]                Transition probabilities: P(state j | state i)
B = [bk(ot)]             Emission probabilities: P(observation ot | state k)
π = [πi]                 Initial state probabilities: P(state i at t=0)
```

### Dry Run — POS Tagging

**Setup:** 3 states (DET, NOUN, VERB), small vocabulary.

```
States: DET, NOUN, VERB
Observations: "the", "dog", "runs"

Initial probabilities (π):
  P(DET) = 0.6,  P(NOUN) = 0.3,  P(VERB) = 0.1

Transition probabilities (A):
  From\To   DET   NOUN  VERB
  DET       0.0   0.8   0.2
  NOUN      0.1   0.2   0.7
  VERB      0.5   0.4   0.1

Emission probabilities (B):
  State\Word  "the"  "dog"  "runs"
  DET         0.8    0.1    0.1
  NOUN        0.1    0.7    0.2
  VERB        0.05   0.05   0.9
```

### Viterbi Algorithm — find the most likely hidden state sequence

```
Step t=0, word = "the":
  δ(DET) = π(DET) * B(DET,"the") = 0.6 * 0.8 = 0.480
  δ(NOUN)= π(NOUN)* B(NOUN,"the")= 0.3 * 0.1 = 0.030
  δ(VERB)= π(VERB)* B(VERB,"the")= 0.1 * 0.05= 0.005

Step t=1, word = "dog":
  δ(DET) = max[δ(DET)*A(DET,DET), δ(NOUN)*A(NOUN,DET), δ(VERB)*A(VERB,DET)]
           * B(DET,"dog")
         = max(0.480*0, 0.030*0.1, 0.005*0.5) * 0.1 = 0.003 * 0.1 = 0.0003
  δ(NOUN)= max(0.480*0.8, 0.030*0.2, 0.005*0.4) * B(NOUN,"dog")
         = max(0.384, 0.006, 0.002) * 0.7 = 0.384 * 0.7 = 0.2688
           (came from DET)
  δ(VERB)= max(0.480*0.2, 0.030*0.7, 0.005*0.1) * B(VERB,"dog")
         = max(0.096, 0.021, 0.0005) * 0.05 = 0.096 * 0.05 = 0.0048
           (came from DET)

Step t=2, word = "runs":
  δ(VERB)= max[δ(DET)*A(DET,VERB), δ(NOUN)*A(NOUN,VERB), δ(VERB)*A(VERB,VERB)]
           * B(VERB,"runs")
         = max(0.0003*0.2, 0.2688*0.7, 0.0048*0.1) * 0.9
         = max(0.00006, 0.18816, 0.00048) * 0.9 = 0.18816 * 0.9 = 0.16934
           (came from NOUN)

Backtrack: t=2: VERB → t=1: NOUN → t=0: DET
Best path: DET → NOUN → VERB = {"the"=DET, "dog"=NOUN, "runs"=VERB}
```

### Three Core HMM Problems

| Problem | Algorithm |
|---------|-----------|
| 1. Evaluation (likelihood): P(O|λ) | Forward algorithm |
| 2. Decoding (most likely path): argmax_S P(S|O,λ) | Viterbi algorithm |
| 3. Learning (parameter est.): argmax_λ P(O|λ) | Baum-Welch (EM for HMMs) |

### Forward Algorithm (Evaluation)

```python
import numpy as np

def forward(obs_seq, A, B, pi):
    """
    obs_seq: list of observation indices
    A: [n_states, n_states] transition matrix
    B: [n_states, n_obs] emission matrix
    pi: [n_states] initial probabilities
    Returns: total probability P(obs_seq | model)
    """
    T = len(obs_seq)
    N = len(pi)
    alpha = np.zeros((T, N))

    # Initialization
    alpha[0] = pi * B[:, obs_seq[0]]

    # Recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs_seq[t]]

    return alpha.sum(axis=1)[-1]  # P(obs_seq)
```

### Viterbi Algorithm (Decoding)

```python
def viterbi(obs_seq, A, B, pi):
    T = len(obs_seq)
    N = len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)   # backpointers

    delta[0] = pi * B[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(N):
            scores = delta[t-1] * A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = np.max(scores) * B[j, obs_seq[t]]

    # Backtrack
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path
```

### HMM Use Cases in NLP

```
POS tagging:         States=POS tags, Obs=words
Named entity recog:  States=BIO tags (B-PER, I-PER, O), Obs=words
Speech recognition:  States=phonemes, Obs=acoustic features
Spell correction:    States=intended chars, Obs=typed chars (edit distance model)
```

**Modern note:** Transformers (BERT with CRF head) have largely replaced HMMs for sequence labeling tasks, but HMMs are still used in speech recognition and are a core interview topic for understanding probabilistic sequence models.

---

## 6. Probabilistic Programming (Modern Bayesian)

The Bayesian recipes above assume conjugate priors. For arbitrary models — hierarchical / non-conjugate / latent variable — you need general Bayesian inference. Two practical tools:

| Tool | Inference engine | Best for |
|------|-----------------|----------|
| PyMC | NUTS (No-U-Turn HMC + ADV) | General-purpose Bayesian modeling, hierarchical models |
| NumPyro | NUTS on JAX (10-100x faster on GPU) | Same models, much faster — use for n > 10K rows |
| Stan | NUTS (C++) | Production-grade, language-agnostic, mature |

```python
# NumPyro — Bayesian logistic regression with hierarchical pooling
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

def model(X, y=None, n_groups=5):
    # Global mean of coefficients
    mu = numpyro.sample("mu", dist.Normal(0., 1.).expand([X.shape[1]]))
    # Per-group coefficients (partial pooling)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.))
    beta  = numpyro.sample("beta",
            dist.Normal(mu, sigma).expand([n_groups, X.shape[1]]))
    # Likelihood
    logits = (X * beta[group_idx]).sum(-1)
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=y)

mcmc = MCMC(NUTS(model), num_warmup=1000, num_samples=2000)
mcmc.run(rng_key, X, y)
mcmc.print_summary()
```

**Why this matters:** Uncertainty quantification: every parameter has a posterior distribution, not a point estimate. **Small-data regimes:** priors let you encode domain knowledge when n is too small for ML. **A/B testing at scale:** Beta-Binomial / hierarchical models capture group-level uncertainty. **Interpretability:** posterior distributions tell stakeholders "we're 90% sure the effect is between X and Y."

**When NOT to use:** large n, no need for uncertainty — boosted trees / NNs are 10-100x faster.

**Modern alternative:** variational inference (ADVI) trades exactness for speed — fits a parameterized distribution to the posterior by minimizing KL divergence. Used inside neural Bayesian networks.

For Gaussian Processes specifically — the most common non-conjugate Bayesian model in ML — see `09_gaussian_processes.md` for full treatment.

---

## 7. Calibration and Probability Estimation

### When Naive Bayes Needs Calibration

NB probabilities are often overconfident (pushed toward 0 and 1). Calibrate:

```python
from sklearn.calibration import CalibratedClassifierCV

# Naive Bayes with Platt scaling
calibrated_nb = CalibratedClassifierCV(MultinomialNB(), method='sigmoid', cv=5)
calibrated_nb.fit(X_train, y_train)

# Check calibration
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test,
                       calibrated_nb.predict_proba(X_test)[:,1],
                       n_bins=10)
```

---

## 8. When to Use What

| Scenario | Model | Why |
|----------|-------|-----|
| Text spam/sentiment, fast baseline | Multinomial NB | Very fast, works well for text counts |
| Very small dataset (< 1K samples) | Gaussian NB or Bayesian Ridge | Data-efficient; prior regularizes |
| Soft cluster assignments | GMM | Know cluster membership probabilities |
| Best K for GMM | BIC criterion | Penalizes model complexity |
| Hyperparameter optimization (expensive model) | Bayesian Optimization (Optuna) | More efficient than random search |
| Uncertainty quantification in regression | Bayesian Ridge | Returns prediction + std |
| Real-time text classification | Bernoulli / Multinomial NB | Near-zero inference cost |
| Anomaly detection with density | GMM log-likelihood | Flag points with low p(x) |

---

## 9. Gotchas

**Naive Bayes zero-probability problem.** If a word appears in class A but not class B in training data → P(word|class B) = 0 → model never predicts class B for any document containing that word. Fix: Laplace smoothing (alpha=1.0 in sklearn adds 1 to all word counts).

**GMM sensitive to initialization.** EM converges to local maxima. Always use `n_init=10`+ to run multiple initializations and keep the best log-likelihood.

**GMM covariance singularity.** If a component gets very few points → covariance matrix is singular → EM breaks. Fix: `reg_covar` parameter (sklearn default `1e-6`) adds a small regularization to diagonal.

**Bayesian Optimization needs sufficient trials to work.** With < 10-20 trials, the GP surrogate doesn't have enough signal. Use at least 50-100 trials. For fast models (linear), just use `RandomizedSearchCV`.

**Optuna objective must be deterministic (same params → same result) for best performance.** If your cross-validation shuffles differently each time, set `random_state` in both the model and the CV.

**NB probability interpretation is unreliable.** While NB often gets predictions right, the probabilities are systematically miscalibrated. For decision thresholding or cost-sensitive decisions, always calibrate NB with Platt scaling.

---

## 10. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Naive Bayes log-probabilities = -inf | Zero probability (unseen word/value) | Increase alpha (smoothing); check training coverage |
| GMM convergence warning | Too many components or too few data | Reduce n_components; increase n_init |
| GMM single cluster captures all data | Bad initialization; K too small | Increase K; use k-means initialization |
| Bayesian Optimization not improving | Too few trials or wrong search space | Increase n_trials; check param bounds (not too wide) |
| NB predictions all one class | Class prior dominates | Check class_prior; balance classes; use uniform prior |
| BayesianRidge much worse than Ridge | Prior assumption wrong | Stick with regular Ridge; tune alpha manually |

---

## 11. Interview Q&A (Senior Level)

**Q: Naive Bayes assumes feature independence — but it works well for text. Why?**
A: The independence assumption is clearly violated in text ("New" and "York" co-occur). But NB doesn't need perfect probability estimates — it needs the correct argmax (correct class ranked highest). Even with wrong absolute probabilities, NB often gets the ranking right because: (1) the dominant signal in most classification tasks is the feature-class correlation, not the feature-feature correlations, (2) the violations tend to be symmetric across classes (if "New" and "York" are correlated, they're correlated for all classes), so the ranking is preserved. Empirically, NB is a strong baseline for text despite the violated assumption.

**Q: What is the EM algorithm and when does it apply?**
A: EM is used when the likelihood function has latent (unobserved) variables that make direct maximization intractable. E-step: compute the expected value of the log-likelihood under the current parameter estimates (soft assignments of latent variables). M-step: maximize this expected log-likelihood with respect to parameters (update parameters given soft assignments). Repeat until convergence. Applies to: GMM (latent: which Gaussian generated each point), hidden Markov models (latent: hidden state sequence), k-means (hard-EM variant). Guaranteed to converge (likelihood never decreases) but to a local maximum — multiple initializations required.

**Q: Explain the three problems in HMMs and how each is solved.**
A: (1) **Evaluation** — given model parameters and an observation sequence, compute P(O|λ). Solved by the Forward algorithm: DP where α(t,j) = probability of being in state j at time t having seen observations o1...ot. O(T*N²) complexity. (2) **Decoding** — find most likely hidden state sequence given observations. Solved by Viterbi: same DP structure as Forward but takes max instead of sum over previous states, with backpointers to reconstruct the path. (3) **Learning** — given only observations, estimate A, B, π. Solved by Baum-Welch, which is EM applied to HMMs: E-step computes forward-backward probabilities (soft state assignments), M-step updates A, B, π using those soft assignments. Baum-Welch converges to a local maximum of the likelihood.

**Q: When would you use Bayesian Optimization over Random Search for hyperparameter tuning?**
A: Bayesian Optimization when: (1) each trial is expensive (training a large neural net takes hours), (2) you have a budget of < 100 trials, (3) the objective function is smooth (good params have nearby good params). Random search when: (1) trials are cheap (linear model, small dataset), (2) you can afford > 100 trials, (3) parallelism is more valuable than intelligence (random search parallelizes perfectly, BO is sequential). Practical rule: use Optuna (Bayesian) for any model that takes > 5 minutes to train. For quick models, 100 random trials often beats 50 Bayesian trials.

---

## 12. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| Bayes' Theorem foundation | `../01_fundamentals/01_statistics_foundations.md` | Full derivation and intuition |
| Bayesian A/B (Beta-Binomial) | `../01_fundamentals/01c_statistics_end_to_end.md` | Worked Monte Carlo example |
| GMM for clustering | `03_unsupervised.md` | GMM as soft clustering covered there |
| EM algorithm (deep dive) | `08_expectation_maximization.md` | Full derivation, missing data, mixtures |
| Gaussian Processes (deep dive) | `09_gaussian_processes.md` | Non-conjugate Bayesian regression |
| NB for text | `../../4.nlp/01_fundamentals/` | Text classification baseline |
| Bayesian Optimization (Optuna) | `../01_fundamentals/04_model_evaluation.md` | Hyperparameter tuning strategies |
| GMM log-likelihood for anomaly | `03_unsupervised.md` | Density-based anomaly detection |
| Calibration | `../01_fundamentals/04_model_evaluation.md` | Calibration curve, Platt scaling |
| Conformal prediction | `../01_fundamentals/04_model_evaluation.md#05-conformal-prediction-distribution-free-uncertainty` | Distribution-free alternative to Bayesian uncertainty |

---

## Key Takeaway

**Naive Bayes:** your fastest text classification baseline. MultinomialNB for TF/count features, BernoulliNB for binary presence, GaussianNB for continuous. Always add smoothing (alpha>0).

**GMM:** K-Means with soft assignments and elliptical clusters. Use BIC to choose K. Log-likelihood on new data = density-based anomaly score.

**EM Algorithm:** the general framework behind GMM and many probabilistic models — alternate between assigning soft cluster memberships (E-step) and updating parameters (M-step).

**Bayesian Optimization:** use Optuna when hyperparameter trials are expensive. 50 intelligent trials often beats 500 random trials.

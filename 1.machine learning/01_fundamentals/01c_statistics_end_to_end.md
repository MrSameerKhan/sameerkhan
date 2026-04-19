# 01c — Statistics End-to-End: Worked Examples

> Every concept traced with concrete numbers. Read this, not the theory file, when preparing for interviews.

---

## Part 1: Distributions — Dry Run

### Scenario
You work at a food delivery company. You have data about:
- Delivery times (continuous, roughly bell-shaped)
- Whether a customer re-orders (binary)
- Number of orders per day from a restaurant (count data)

**Which distribution for each?**

---

### Normal Distribution — Delivery Time

```
Problem: Delivery times have μ = 35 min, σ = 8 min.
         What fraction of deliveries take > 51 minutes?

Step 1: Convert to Z-score
  Z = (X − μ) / σ = (51 − 35) / 8 = 16 / 8 = 2.0

Step 2: Interpret using 68/95/99.7 rule
  95% of data is within μ ± 2σ → 5% is outside → 2.5% is above μ + 2σ

Answer: ~2.5% of deliveries take > 51 minutes.

Step 3: If interviewer asks "what's P(delivery < 27 min)?"
  Z = (27 − 35) / 8 = −1.0
  P(Z < −1) = 1 − P(Z < 1) = 1 − 0.841 = 0.159
  Answer: ~16% of deliveries take < 27 minutes.
```

**Key insight**: Z-score tells you how many standard deviations from the mean. Memorize:
```
Z = 1.0  →  84th percentile (P below = 84%)
Z = 1.65 →  95th percentile
Z = 1.96 →  97.5th percentile  ← used in 95% confidence intervals
Z = 2.0  →  97.7th percentile
Z = 2.58 →  99.5th percentile  ← used in 99% confidence intervals
```

---

### Bernoulli + Binomial — Re-order Probability

```
Problem: Each customer has p = 0.30 probability of re-ordering.
         You have n = 10 customers. What's the probability exactly 3 re-order?

Distribution: Binomial B(n=10, p=0.30)

P(X = k) = C(n,k) · pᵏ · (1−p)^(n−k)

P(X = 3) = C(10,3) · (0.30)³ · (0.70)⁷
          = 120 · 0.027 · 0.0824
          = 120 · 0.00222
          = 0.267

Answer: 26.7% chance exactly 3 of 10 customers re-order.

Mean of Binomial: μ = n·p = 10 × 0.30 = 3.0
Variance:         σ² = n·p·(1−p) = 10 × 0.30 × 0.70 = 2.1
Std dev:          σ = √2.1 = 1.45
```

**Interviewer follow-up**: "What's the probability AT LEAST 3 re-order?"
```
P(X ≥ 3) = 1 − P(X < 3) = 1 − [P(X=0) + P(X=1) + P(X=2)]
         = 1 − [0.028 + 0.121 + 0.233]
         = 1 − 0.382
         = 0.618

Answer: 61.8% chance at least 3 re-order.
```

---

### Poisson — Orders Per Day

```
Problem: A restaurant averages λ = 40 orders/day.
         What's the probability of receiving exactly 45 orders tomorrow?

Distribution: Poisson(λ = 40)

P(X = k) = e^(−λ) · λᵏ / k!

For large λ, Poisson ≈ Normal with μ = λ, σ² = λ, σ = √λ

Approximation:
  Z = (45 − 40) / √40 = 5 / 6.32 = 0.79
  P(X = 45) ≈ height of Normal at Z = 0.79 ≈ 0.067

Answer: ~7% chance of exactly 45 orders.

Key properties of Poisson:
  Mean = Variance = λ    ← if mean ≠ variance, Poisson doesn't fit well
  Use when: counting events in fixed time/space, events are independent
  Examples: server requests per second, customer arrivals per hour, defects per unit
```

---

### Exponential — Time Between Events

```
Problem: Orders arrive at rate λ = 40/day. On average, how long between orders?

Exponential distribution models time between events.
  Mean inter-arrival time = 1/λ = 1/40 day = 36 minutes

P(wait > t) = e^(−λt)
P(next order arrives within 1 hour = 1/24 day):
  = 1 − e^(−40 × 1/24)
  = 1 − e^(−1.67)
  = 1 − 0.188
  = 0.812

Answer: 81.2% chance the next order arrives within 1 hour.

Memoryless property: P(wait > s+t | wait > s) = P(wait > t)
  → The fact that you've already waited s minutes tells you nothing about future wait.
  → Only continuous distribution with this property.
```

---

### Beta — Prior over Probabilities

```
Problem: You believe a model's precision is around 0.7 but you're uncertain.
         How do you encode this as a prior?

Beta(α, β) lives in [0,1] → perfect for probabilities, proportions.
  Mean = α / (α + β)
  Variance = αβ / [(α+β)²(α+β+1)]

If you think precision ≈ 0.7 with moderate confidence:
  Set α = 7, β = 3
  Mean = 7/(7+3) = 0.7  ✓
  Variance = 21/(100·11) = 0.019  → std ≈ 0.14  (moderate uncertainty)

If you're very confident:
  Set α = 70, β = 30
  Mean = 0.7, but std ≈ 0.046  (tight around 0.7)

Used in: Bayesian A/B testing, Thompson sampling for multi-armed bandits,
         anywhere you need a prior over a probability.
```

---

## Part 2: P-Values — Complete Dry Run

### Scenario: A/B Test on New Recommendation Algorithm

```
Context: Current algorithm has CTR = 8%.
         New algorithm tested on 1,000 users → 92 clicks out of 1,000 = 9.2% CTR.
         Is this improvement real or just noise?
```

**Step 1: State the hypotheses**
```
H₀: new algorithm CTR = old algorithm CTR (p_new = p_old = 0.08)
H₁: new algorithm CTR > old algorithm CTR (one-tailed test)

Why one-tailed? We only care if new > old, not new ≠ old.
```

**Step 2: Compute the test statistic**
```
Under H₀, the observed proportion follows:
  p̂ = 92/1000 = 0.092
  SE = √(p₀(1−p₀)/n) = √(0.08 × 0.92 / 1000) = √0.0000736 = 0.00858

Z = (p̂ − p₀) / SE = (0.092 − 0.08) / 0.00858 = 0.012 / 0.00858 = 1.40
```

**Step 3: Find the p-value**
```
Z = 1.40 → P(Z > 1.40) = 1 − 0.919 = 0.081

p-value = 0.081
```

**Step 4: Make the decision**
```
α = 0.05 (our threshold)
p-value = 0.081 > 0.05

Decision: FAIL to reject H₀.
Conclusion: The improvement from 8% to 9.2% is NOT statistically significant.
            We cannot claim the new algorithm is better.
```

**Step 5: Explain it to a non-statistician**
```
"If the algorithms were identical, we'd see a difference this large or bigger
 about 8% of the time just by chance. Since 8% > 5% (our threshold),
 we don't have strong enough evidence to conclude the new algorithm is better.
 We need more data or a larger effect to be confident."
```

**What sample size would give significance?**
```
For Z = 1.645 (one-tailed α=0.05) with effect size (0.092 − 0.08) = 0.012:
  n = [Z · SE_per_obs]² / effect² ... (use power analysis)

Quick rule: double the sample to n=2000, retest.
```

---

### Common P-Value Interview Traps

**Trap 1: "p-value is probability H₀ is true"** — WRONG
```
Correct: p-value = P(data this extreme | H₀ is true)
         NOT:     P(H₀ is true | data)

The distinction matters. H₀ is either true or false — it doesn't have a probability
(in frequentist statistics). The p-value is about the data, not the hypothesis.
```

**Trap 2: "p < 0.05 means the result is important"** — WRONG
```
Statistical significance ≠ practical significance.

Example: Test whether a new feature improves CTR.
  n = 10,000,000 users
  Old CTR = 5.000%, New CTR = 5.001%
  Z = very large → p < 0.0001 → highly significant!
  
  But 0.001% CTR improvement on 10M users = 100 extra clicks/day.
  Business impact: negligible. Not worth engineering cost.

Always report effect size alongside p-value:
  Cohen's d = (μ₁ − μ₂) / pooled_σ
  d = 0.2: small,  d = 0.5: medium,  d = 0.8: large
```

**Trap 3: Multiple comparisons / p-hacking**
```
If you test 20 features and declare significance at α=0.05:
  Expected false positives = 20 × 0.05 = 1 feature

Fix: Bonferroni correction → use α' = α / m = 0.05 / 20 = 0.0025 per test
     Or: Benjamini-Hochberg (FDR control) — less conservative
```

**Trap 4: "Fail to reject = proof H₀ is true"** — WRONG
```
"Fail to reject" just means insufficient evidence.
Could be: true null, OR small sample size, OR wrong test.

Example: drug test with n=10 patients shows p=0.3.
  Does NOT mean drug doesn't work.
  Means: with only 10 patients, we can't detect the effect reliably.
```

---

## Part 3: Hypothesis Tests — Which One When

### T-test Full Example

```
Problem: You trained two models. On a test set of 50 samples each:
  Model A: mean accuracy = 0.845, std = 0.04
  Model B: mean accuracy = 0.862, std = 0.05

Is Model B significantly better? Use two-sample t-test.

Step 1: State hypotheses
  H₀: μ_A = μ_B  (no difference)
  H₁: μ_A ≠ μ_B  (two-tailed)

Step 2: Compute t-statistic
  SE = √(s_A²/n_A + s_B²/n_B)
     = √(0.0016/50 + 0.0025/50)
     = √(0.000032 + 0.00005)
     = √0.000082
     = 0.00906

  t = (x̄_B − x̄_A) / SE = (0.862 − 0.845) / 0.00906 = 0.017 / 0.00906 = 1.876

Step 3: Degrees of freedom ≈ n_A + n_B − 2 = 98

Step 4: p-value
  For t=1.876 with df=98 (two-tailed): p ≈ 0.064

Step 5: Decision
  p = 0.064 > 0.05 → fail to reject H₀
  The difference is not statistically significant at α=0.05.

Practical take: marginal result. Consider running with more samples.
```

### Chi-Squared Test Full Example

```
Problem: Is user platform (mobile/desktop) associated with purchase behavior?

Data:
              Purchased   Not Purchased   Total
  Mobile:        120           480          600
  Desktop:        80           320          400
  Total:         200           800         1000

Step 1: Compute expected counts (if independent)
  E[mobile, purchased]    = (600 × 200) / 1000 = 120
  E[mobile, not purchased] = (600 × 800) / 1000 = 480
  E[desktop, purchased]   = (400 × 200) / 1000 = 80
  E[desktop, not purchased] = (400 × 800) / 1000 = 320

Step 2: Chi-squared statistic
  χ² = Σ (Observed − Expected)² / Expected
     = (120−120)²/120 + (480−480)²/480 + (80−80)²/80 + (320−320)²/320
     = 0 + 0 + 0 + 0
     = 0.0

p-value = 1.0 → completely fails to reject independence

Interpretation: Platform and purchase rate are INDEPENDENT.
  Mobile rate = 120/600 = 20%, Desktop rate = 80/400 = 20% — identical!

When chi-squared IS significant:
  χ² > 3.84 (df=1, α=0.05) → significant association
  Tells you there IS an association — doesn't tell you direction or size.
  Follow up with Cramér's V for effect size.
```

---

## Part 4: Central Limit Theorem — Why It's Everywhere

### Dry Run

```
Scenario: Customer spending per order is NOT normally distributed.
  It's heavily right-skewed: most orders are small, a few are very large.
  True distribution: mean = $24, std = $18 (non-normal, skewed)

Question: You sample 100 orders. What's the distribution of the sample mean?

CLT answer:
  x̄ ~ N(μ, σ²/n) = N(24, 18²/100) = N(24, 3.24)
  Standard error = σ/√n = 18/√100 = 18/10 = 1.8

So even though individual orders are skewed, the sample mean of 100 orders
is approximately Normal with mean $24 and std $1.80.

P(sample mean > $27) = P(Z > (27−24)/1.8) = P(Z > 1.67) = 1 − 0.953 = 4.7%
```

**Why this matters in ML:**
```
1. A/B testing: even if conversion rate is Bernoulli, sample mean conversion
   is approximately Normal → we can use Z-tests on large samples.

2. Confidence intervals: x̄ ± 1.96 · (σ/√n) is valid regardless of distribution shape
   (for n ≥ 30 rule of thumb).

3. Ensemble averaging: averaging predictions of k models reduces variance by factor k.
   Var(mean of k models) = Var(single model) / k  ← CLT logic.

4. Batch gradient descent: gradient estimated from mini-batch is a noisy estimate
   of the true gradient. With larger batch size, this estimate is more normal and
   less noisy → more stable training. (But: less regularization effect from noise.)
```

---

## Part 5: Distributions in ML — Complete Connection Map

```
Distribution  → Where it appears in ML
─────────────────────────────────────────────────────────────────────────
Normal        → Linear model assumptions, weight init (Xavier uses N(0, 2/n)),
                noise modeling, PCA (Gaussian assumption), GMM components

Bernoulli     → Binary classification output (sigmoid), logistic regression

Categorical   → Multiclass output (softmax), token probabilities in LLMs

Binomial      → Aggregated binary outcomes, click count modeling

Poisson       → Count regression (predict number of events),
                queuing theory for API load, loss for count targets

Exponential   → Survival analysis, time-to-churn, time-to-failure

Log-Normal    → Salary prediction, document length, click-through time,
                any positive skewed quantity (log-transform first)

Beta          → Bayesian priors over probabilities, Thompson sampling,
                uncertainty over model confidence

Dirichlet     → Generalization of Beta to multiple classes,
                topic modeling (LDA prior), Bayesian multiclass

Uniform       → Random feature initialization, prior when you know nothing

Chi-squared   → Goodness of fit tests, chi-squared feature selection,
                confidence interval for variance

t-distribution → Small sample statistics (when σ unknown), heavier tails
                 than Normal → robust to outliers
```

---

## Part 6: How to Answer Distribution Questions in Interviews

### The 4-Step Framework

```
Step 1: Identify the variable type
  Binary outcome? → Bernoulli/Binomial
  Count of events? → Poisson
  Time between events? → Exponential
  Continuous, symmetric? → Normal
  Continuous, right-skewed, positive? → Log-Normal
  Probability itself? → Beta

Step 2: State the parameters
  What is μ, σ, λ, p, n? Give concrete numbers.

Step 3: Give a key property
  Poisson: mean = variance. Exponential: memoryless. Normal: 68/95/99.7.

Step 4: Connect to ML
  "In practice I'd check this by [plotting histogram, Q-Q plot, or running KS test]
   and then [log-transform if log-normal, use Poisson regression if count, etc.]"
```

### Quick Code: Fit and Test a Distribution

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = np.array([...])   # your observed data

# 1. Plot histogram
plt.hist(data, bins=50, density=True, alpha=0.7, label='data')

# 2. Fit Normal
mu, sigma = stats.norm.fit(data)
x = np.linspace(data.min(), data.max(), 100)
plt.plot(x, stats.norm.pdf(x, mu, sigma), label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
plt.legend(); plt.show()

# 3. Q-Q plot (if points on straight line → Normal)
stats.probplot(data, dist="norm", plot=plt)
plt.title("Q-Q Plot"); plt.show()

# 4. KS test: H₀ = data comes from Normal
stat, p = stats.kstest(data, 'norm', args=(mu, sigma))
print(f"KS test p-value: {p:.4f}")
# p > 0.05 → fail to reject Normal fit

# 5. Test for Poisson (mean ≈ variance?)
print(f"Mean: {np.mean(data):.2f}, Variance: {np.var(data):.2f}")
# If close → Poisson might fit
```

---

## Key Takeaway

```
Distributions:
  Normal  → continuous, symmetric, CLT, Z-score, 68/95/99.7
  Poisson → counts, mean=variance=λ
  Exponential → time between events, memoryless, mean=1/λ
  Binomial → successes in n trials, mean=np
  Beta → prior over probabilities [0,1]

P-values:
  p-value = P(data this extreme | H₀ true)
  p < α → reject H₀ → statistically significant
  p ≥ α → fail to reject (NOT proof H₀ is true)
  Traps: p-hacking, n is huge so everything significant, fail to reject ≠ accept H₀

CLT:
  x̄ ~ N(μ, σ²/n) regardless of original distribution, for n ≥ 30
  SE = σ/√n → larger sample = tighter estimate of mean

Interview move: always connect formula → concrete numbers → ML application
```

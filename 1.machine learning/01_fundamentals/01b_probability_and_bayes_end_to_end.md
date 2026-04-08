# Probability & Bayes — End-to-End

Every number computed. Built for BarRaiser "Probability & Statistics" round.

---

## What This Covers

| Topic | Interview question it answers |
|---|---|
| Conditional probability | "What is P(A given B)? Walk me through." |
| Bayes theorem | "How does a spam filter work mathematically?" |
| Joint & marginal probability | "What's the difference between joint and marginal?" |
| Independence | "When are two events independent?" |
| Gradient descent step | "Show me one weight update in logistic regression." |
| Information gain | "How does a decision tree choose which feature to split on?" |

---

## Part 1 — Conditional Probability

### Setup: Email dataset

```
100 emails total:
  60 are spam, 40 are not spam
  Of the 60 spam:    45 contain the word "free", 15 do not
  Of the 40 not spam: 5 contain the word "free", 35 do not
```

**Contingency table:**

|  | Contains "free" | No "free" | Total |
|---|---|---|---|
| Spam     | 45 | 15 | 60 |
| Not spam |  5 | 35 | 40 |
| **Total** | **50** | **50** | **100** |

---

### Step 1 — Marginal probabilities

```
P(spam)     = 60/100 = 0.60
P(not spam) = 40/100 = 0.40

P("free")     = 50/100 = 0.50
P(no "free")  = 50/100 = 0.50
```

Marginal = probability of one event, ignoring the other variable.

---

### Step 2 — Joint probabilities

```
P(spam AND "free")     = 45/100 = 0.45
P(spam AND no "free")  = 15/100 = 0.15
P(not spam AND "free") =  5/100 = 0.05
P(not spam AND no "free") = 35/100 = 0.35

Check: 0.45 + 0.15 + 0.05 + 0.35 = 1.00 ✓
```

Joint = probability of BOTH events happening simultaneously.

---

### Step 3 — Conditional probabilities

```
P(A | B) = P(A AND B) / P(B)
```

**P("free" | spam)** — given an email IS spam, how likely is "free"?
```
P("free" | spam) = P(spam AND "free") / P(spam)
                 = 0.45 / 0.60
                 = 0.75
```
75% of spam emails contain "free".

**P(spam | "free")** — given email contains "free", how likely is spam?
```
P(spam | "free") = P(spam AND "free") / P("free")
                 = 0.45 / 0.50
                 = 0.90
```
90% of emails containing "free" are spam.

**These are NOT the same number.**  
P("free" | spam) = 0.75 ≠ P(spam | "free") = 0.90  
Confusing these is called the **base rate fallacy**.

---

### Step 4 — Independence check

Two events A and B are independent if:
```
P(A AND B) = P(A) × P(B)
```

Are "spam" and "free" independent in our dataset?
```
P(spam) × P("free") = 0.60 × 0.50 = 0.30

P(spam AND "free") = 0.45

0.45 ≠ 0.30  →  NOT independent
```

Knowing the email contains "free" changes the probability it's spam (from 60% to 90%) — they are dependent.

---

## Part 2 — Bayes Theorem

### The formula

```
P(A | B) = P(B | A) × P(A) / P(B)

Posterior = Likelihood × Prior / Evidence
```

**Why we need it:** We often know P("free" | spam) from training data, but we want P(spam | "free") at inference time. Bayes flips the conditioning.

---

### Full spam filter computation

**Given a NEW email containing "free" — is it spam?**

```
Prior:      P(spam) = 0.60         ← 60% of all emails are spam
Likelihood: P("free" | spam) = 0.75    ← 75% of spam has "free"
Evidence:   P("free") = 0.50       ← 50% of all emails have "free"
```

**Apply Bayes:**
```
P(spam | "free") = P("free" | spam) × P(spam) / P("free")
                 = 0.75 × 0.60 / 0.50
                 = 0.45 / 0.50
                 = 0.90
```

90% probability the email is spam. ✓ (matches our direct calculation above)

---

### Alternative: compute Evidence from components

Often P("free") is not directly available. Compute it using the **law of total probability:**

```
P("free") = P("free" | spam) × P(spam) + P("free" | not spam) × P(not spam)
          = 0.75 × 0.60 + 0.125 × 0.40
          = 0.45 + 0.05
          = 0.50   ✓

where: P("free" | not spam) = 5/40 = 0.125
```

---

### Two-word update (chain of evidence)

New email contains BOTH "free" AND "winner". Naive Bayes assumes independence between words:

```
P("winner" | spam)     = 0.60   (60% of spam has "winner")
P("winner" | not spam) = 0.10   (10% of not spam has "winner")

Step 1 — Update with "free" (already computed):
  P(spam | "free")     = 0.90
  P(not spam | "free") = 0.10

Step 2 — Use this as new prior, update with "winner":
  Numerator:
    P("winner" | spam) × P(spam | "free") = 0.60 × 0.90 = 0.540

  Denominator (total probability):
    P("winner" | spam) × P(spam | "free") +
    P("winner" | not spam) × P(not spam | "free")
    = 0.60 × 0.90 + 0.10 × 0.10
    = 0.540 + 0.010
    = 0.550

  P(spam | "free", "winner") = 0.540 / 0.550 = 0.982
```

Adding "winner" pushes spam probability from 90% → 98.2%. Each word is a new piece of evidence.

---

### Sensitivity to prior (what if spam rate changes?)

Suppose we change the deployment context: only 20% of emails are spam.

```
Prior: P(spam) = 0.20
P("free" | spam) = 0.75,  P("free" | not spam) = 0.125

Evidence:
  P("free") = 0.75×0.20 + 0.125×0.80 = 0.150 + 0.100 = 0.250

Posterior:
  P(spam | "free") = 0.75 × 0.20 / 0.250 = 0.150 / 0.250 = 0.600
```

With only 20% prior spam rate: email with "free" → 60% spam probability.  
With 60% prior spam rate: email with "free" → 90% spam probability.

**Same evidence, same likelihood — different posterior because prior changed.**  
This is why base rate (prior) matters. Ignoring it is the base rate fallacy.

---

## Part 3 — Information Gain (Decision Tree Split)

### Setup

**Dataset: 10 emails, predict spam/not spam based on "free" feature**

| ID | "free" | Label |
|----|--------|-------|
| 1  | Yes    | Spam  |
| 2  | Yes    | Spam  |
| 3  | Yes    | Spam  |
| 4  | Yes    | Spam  |
| 5  | Yes    | Not Spam |
| 6  | No     | Spam  |
| 7  | No     | Not Spam |
| 8  | No     | Not Spam |
| 9  | No     | Not Spam |
| 10 | No     | Not Spam |

---

### Step 1 — Entropy of the full dataset

```
Entropy H(S) = -Σ p_i × log₂(p_i)

In S: 5 spam, 5 not spam → p_spam = 0.5, p_not = 0.5

H(S) = -(0.5 × log₂(0.5)) - (0.5 × log₂(0.5))
      = -(0.5 × (-1)) - (0.5 × (-1))
      = 0.5 + 0.5
      = 1.0 bit   ← maximum uncertainty (50/50 split)
```

---

### Step 2 — Entropy after split on "free"

**Left branch: "free" = Yes (5 emails: 4 spam, 1 not spam)**
```
p_spam = 4/5 = 0.80,  p_not = 1/5 = 0.20

H(left) = -(0.80 × log₂(0.80)) - (0.20 × log₂(0.20))
         = -(0.80 × (-0.322)) - (0.20 × (-2.322))
         = 0.258 + 0.464
         = 0.722 bits
```

**Right branch: "free" = No (5 emails: 1 spam, 4 not spam)**
```
p_spam = 1/5 = 0.20,  p_not = 4/5 = 0.80

H(right) = -(0.20 × log₂(0.20)) - (0.80 × log₂(0.80))
          = 0.464 + 0.258
          = 0.722 bits
```

**Weighted entropy after split:**
```
H(split) = (5/10) × H(left) + (5/10) × H(right)
          = 0.5 × 0.722 + 0.5 × 0.722
          = 0.722 bits
```

---

### Step 3 — Information Gain

```
IG("free") = H(S) - H(split)
           = 1.0 - 0.722
           = 0.278 bits
```

Splitting on "free" reduces uncertainty by 0.278 bits.  
The decision tree picks the feature with the **highest** information gain.

---

### Perfect split example (for comparison)

If "free" perfectly separated spam from not spam (5 spam all say Yes, 5 not spam all say No):

```
H(left)  = -(1.0 × log₂(1.0)) = 0.0 bits   ← pure node
H(right) = -(1.0 × log₂(1.0)) = 0.0 bits   ← pure node

H(split) = 0.5×0 + 0.5×0 = 0.0
IG = 1.0 - 0.0 = 1.0 bit   ← maximum possible gain
```

Pure split = maximum information gain = best possible feature.

---

## Part 4 — One Gradient Descent Step (Logistic Regression)

### Setup

**Task:** Predict spam (1) or not spam (0) from one feature x = word count.

```
Training sample: x = 3.0 (email has 3 "suspicious" words), y = 1 (spam)
Initial weights:  w = 0.5, b = 0.0
Learning rate:    η = 0.1
```

---

### Step 1 — Forward pass

```
Linear output:
  z = w × x + b = 0.5 × 3.0 + 0.0 = 1.5

Sigmoid activation:
  ŷ = σ(z) = 1 / (1 + e^{-z})
           = 1 / (1 + e^{-1.5})
           = 1 / (1 + 0.223)
           = 1 / 1.223
           = 0.818

Prediction: 81.8% probability of spam.
True label: y = 1 (spam) ✓ — model is in the right direction
```

---

### Step 2 — Loss (Binary Cross-Entropy)

```
L = -(y × log(ŷ) + (1-y) × log(1-ŷ))
  = -(1 × log(0.818) + 0 × log(0.182))
  = -(log(0.818))
  = -(-0.201)
  = 0.201
```

---

### Step 3 — Gradients

```
Key result: ∂L/∂z = ŷ - y   ← elegant simplification of sigmoid + CE

∂L/∂z = 0.818 - 1.0 = -0.182   ← negative means z should increase

∂L/∂w = ∂L/∂z × x = -0.182 × 3.0 = -0.546
∂L/∂b = ∂L/∂z × 1 = -0.182
```

---

### Step 4 — Weight update

```
w_new = w - η × ∂L/∂w = 0.5 - 0.1 × (-0.546) = 0.5 + 0.055 = 0.555
b_new = b - η × ∂L/∂b = 0.0 - 0.1 × (-0.182) = 0.0 + 0.018 = 0.018
```

---

### Step 5 — Verify (forward pass with new weights)

```
z_new = 0.555 × 3.0 + 0.018 = 1.665 + 0.018 = 1.683

ŷ_new = 1 / (1 + e^{-1.683})
       = 1 / (1 + 0.186)
       = 1 / 1.186
       = 0.843

L_new = -log(0.843) = 0.171
```

Loss decreased: 0.201 → 0.171 ✓  
Prediction improved: 81.8% → 84.3% spam probability ✓

---

## Summary Table

| Concept | Formula | Our result |
|---|---|---|
| Conditional P | P(A\|B) = P(A∩B) / P(B) | P(spam\|"free") = 0.90 |
| Bayes | P(A\|B) = P(B\|A)×P(A) / P(B) | 0.75×0.60/0.50 = 0.90 |
| Entropy | -Σ p×log₂(p) | H(S) = 1.0 bit |
| Information Gain | H(S) - H(after split) | IG("free") = 0.278 bits |
| Gradient (sigmoid+CE) | ŷ - y | -0.182 |
| Weight update | w - η×∂L/∂w | 0.5 → 0.555 |

---

## Interview Q&A

**Q: What is the difference between joint and conditional probability?**
> Joint P(A∩B): probability both A and B occur. From our table: P(spam AND "free") = 45/100 = 0.45.  
> Conditional P(A|B): probability of A given B has already occurred. P(spam|"free") = 0.45/0.50 = 0.90.  
> Conditional focuses the sample space — we restrict to emails that contain "free" (50), then ask how many are spam (45). 45/50 = 0.90.

**Q: Explain Bayes theorem in plain English.**
> We want P(spam|"free") — probability email is spam given it contains "free".  
> We know P("free"|spam) from training data (75% of spam has "free").  
> Bayes lets us flip this: multiply by the prior P(spam) and divide by how common "free" is overall.  
> It's a principled way to update our belief (prior) with new evidence (likelihood).

**Q: What is the base rate fallacy?**
> Ignoring the prior P(spam) when computing posterior probability.  
> Example: a disease affects 1% of people (prior = 0.01). A test is 99% accurate.  
> If you test positive, naive thinking says "99% chance I have it." But Bayes says:  
> P(disease|positive) = 0.99×0.01 / (0.99×0.01 + 0.01×0.99) = 0.0099/0.0198 = 0.50  
> Only 50%! Because the disease is so rare, half the positives are false positives.  
> In ML: a fraud model with 99% accuracy on 0.1% fraud rate is probably just predicting "not fraud" always.

**Q: What does entropy measure and why do decision trees use it?**
> Entropy measures impurity/uncertainty in a set of labels.  
> H=0: all one class (pure, no uncertainty).  
> H=1: 50/50 split (maximum uncertainty for binary).  
> Decision trees split on the feature that maximally reduces entropy (information gain).  
> More gain = the feature separates classes more cleanly.

**Q: Why is the gradient of cross-entropy loss with sigmoid just (ŷ - y)?**
> The sigmoid and log-loss gradients cancel each other beautifully.  
> ∂CE/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)  
> ∂ŷ/∂z = ŷ(1-ŷ)  (sigmoid derivative)  
> ∂CE/∂z = ∂CE/∂ŷ × ∂ŷ/∂z = [-y/ŷ + (1-y)/(1-ŷ)] × ŷ(1-ŷ) = ŷ - y  
> This is why logistic regression is computationally clean — the gradient is just the prediction error.

**Q: Two events have zero correlation. Does that mean they're independent?**
> No. Correlation only measures LINEAR relationships. Two events can be strongly dependent in a nonlinear way and still have zero Pearson correlation.  
> Example: X ~ Uniform(-1,1), Y = X². Corr(X,Y)=0 but Y is completely determined by X.  
> Independence requires P(A∩B) = P(A)×P(B) for ALL values, not just linear summary statistics.

**Q: What's the difference between Gini impurity and entropy in decision trees?**
> Both measure node impurity. Both reach 0 at pure nodes.  
> Gini: 1 - Σ p_i² (computationally cheaper — no log)  
> Entropy: -Σ p_i log₂(p_i) (information-theoretic, slightly prefers balanced splits)  
> In practice: results are nearly identical. Gini is faster. sklearn uses Gini by default.  
> For interviews: they're interchangeable — pick one and explain it correctly.

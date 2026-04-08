# Word2Vec End to End — Skip-gram, Negative Sampling, GloVe with Numbers

Same corpus throughout: **"cat sat on mat"**
Vocabulary: cat=0, sat=1, on=2, mat=3 (V=4)
Embedding dimension: d=2

---

## 0. The Core Question

Why not just use one-hot vectors?

```
One-hot (V=4):
  cat = [1, 0, 0, 0]
  sat = [0, 1, 0, 0]
  on  = [0, 0, 1, 0]
  mat = [0, 0, 0, 1]

cosine(cat, mat) = 0.0   ← zero, as if completely unrelated
cosine(cat, sat) = 0.0   ← also zero — no sense of "both are short words near each other"
```

Every word is equidistant from every other word. There's no semantic structure — "cat" and "feline" are as different as "cat" and "democracy."

Word2Vec fixes this. The embedding for each word is a dense vector learned so that **words that appear in similar contexts get similar vectors**.

After training on a large corpus:
```
cosine(cat, dog)    ≈ 0.85   ← both are domestic animals
cosine(cat, feline) ≈ 0.92   ← near-synonyms
cosine(cat, bank)   ≈ 0.10   ← unrelated
```

The geometry of the embedding space encodes meaning.

---

## 1. The Two Embedding Matrices

Word2Vec maintains **two separate embedding matrices**:

```
W_in  (V × d = 4 × 2):  "input embeddings"  — one row per word (used when word is center)
W_out (V × d = 4 × 2):  "output embeddings" — one row per word (used when word is context)
```

**Initial values (randomly initialized):**

```
W_in:                    W_out:
         dim_0  dim_1             dim_0  dim_1
cat:    [0.3,   0.5  ]   cat:    [0.2,   0.5  ]
sat:    [0.1,   0.2  ]   sat:    [0.1,   0.3  ]
on:     [0.4,   0.1  ]   on:     [0.3,   0.2  ]
mat:    [0.2,   0.3  ]   mat:    [0.4,   0.1  ]
```

After training:
- Keep W_in as the word embeddings (or average W_in and W_out — both are valid)
- Discard W_out (it was just a scaffold for the training objective)

---

## 2. Building Training Pairs

Word2Vec Skip-gram: given a center word, predict its surrounding context words.

**Corpus:** "cat sat on mat"
**Window size:** 1 (look 1 word left and right of center)

Sliding through the corpus:

```
Position 0 — center: cat
  Left context: (none)
  Right context: sat
  → Pair: (cat → sat)

Position 1 — center: sat
  Left context: cat
  Right context: on
  → Pairs: (sat → cat), (sat → on)

Position 2 — center: on
  Left context: sat
  Right context: mat
  → Pairs: (on → sat), (on → mat)

Position 3 — center: mat
  Left context: on
  Right context: (none)
  → Pair: (mat → on)
```

**All training pairs:**
```
(cat, sat)
(sat, cat), (sat, on)
(on, sat),  (on, mat)
(mat, on)
```

6 pairs total. For a real corpus (billions of tokens), this same process generates hundreds of billions of (center, context) pairs.

---

## 3. Skip-gram: Full Softmax — Forward Pass

**Training pair: center=sat(1), context=cat(0)**

**Step 1: Look up center word's input embedding**
```
v_sat = W_in[1] = [0.1, 0.2]
```

**Step 2: Compute score against EVERY word's output embedding**
```
score(sat → cat) = v_sat · W_out[0] = 0.1×0.2 + 0.2×0.5 = 0.02 + 0.10 = 0.12
score(sat → sat) = v_sat · W_out[1] = 0.1×0.1 + 0.2×0.3 = 0.01 + 0.06 = 0.07
score(sat → on)  = v_sat · W_out[2] = 0.1×0.3 + 0.2×0.2 = 0.03 + 0.04 = 0.07
score(sat → mat) = v_sat · W_out[3] = 0.1×0.4 + 0.2×0.1 = 0.04 + 0.02 = 0.06
```

**Step 3: Softmax to get probabilities**
```
e^0.12 = 1.127
e^0.07 = 1.073
e^0.07 = 1.073
e^0.06 = 1.062
sum = 4.335

P(cat | sat) = 1.127 / 4.335 = 0.260  ← target word
P(sat | sat) = 1.073 / 4.335 = 0.247
P(on  | sat) = 1.073 / 4.335 = 0.247
P(mat | sat) = 1.062 / 4.335 = 0.245
```

The model knows nothing yet — probabilities are nearly uniform (0.25 each). That's correct for random initialization.

**Step 4: Loss**
```
L = -log P(cat | sat) = -log(0.260) = 1.347
```

Compare to random baseline: -log(1/4) = -log(0.25) = 1.386. Our model is marginally better than random (barely — because scores differ by only 0.01-0.06).

---

## 4. Skip-gram: Full Softmax — Backward Pass

**Gradient of loss w.r.t. scores:**
```
∂L/∂scores = [P(cat)-1, P(sat), P(on), P(mat)]
           = [0.260-1, 0.247, 0.247, 0.245]
           = [-0.740,  0.247, 0.247, 0.245]
```

The target word (cat) gets a large negative gradient — we want its score to increase. All other words get small positive gradients — we want their scores to decrease.

**Gradient w.r.t. output embeddings:**
Each output embedding row j gets gradient = (∂L/∂score_j) × v_sat

```
∂L/∂W_out[0] = -0.740 × [0.1, 0.2] = [-0.074, -0.148]   ← cat_out: large negative (push toward v_sat)
∂L/∂W_out[1] =  0.247 × [0.1, 0.2] = [ 0.025,  0.049]   ← sat_out: push away
∂L/∂W_out[2] =  0.247 × [0.1, 0.2] = [ 0.025,  0.049]   ← on_out:  push away
∂L/∂W_out[3] =  0.245 × [0.1, 0.2] = [ 0.025,  0.049]   ← mat_out: push away
```

**Gradient w.r.t. center word's input embedding (v_sat):**
```
∂L/∂v_sat = Σ_j (∂L/∂score_j × W_out[j])
           = (-0.740)×[0.2,0.5] + (0.247)×[0.1,0.3] + (0.247)×[0.3,0.2] + (0.245)×[0.4,0.1]

           = [-0.148, -0.370]    (cat_out pulls v_sat toward it)
           + [ 0.025,  0.074]    (sat_out pushes away)
           + [ 0.074,  0.049]    (on_out pushes away)
           + [ 0.098,  0.025]    (mat_out pushes away)

           = [0.049, -0.222]
```

Intuition: v_sat is being pulled toward W_out[0] (cat's output embedding) and pushed away from all others.

**Update (lr = 0.1):**
```
W_in[1]_new   = [0.1, 0.2] - 0.1×[0.049, -0.222] = [0.095, 0.222]   ← v_sat
W_out[0]_new  = [0.2, 0.5] - 0.1×[-0.074, -0.148] = [0.207, 0.515]  ← cat_out
W_out[1]_new  = [0.1, 0.3] - 0.1×[0.025, 0.049]   = [0.097, 0.295]  ← sat_out
W_out[2]_new  = [0.3, 0.2] - 0.1×[0.025, 0.049]   = [0.297, 0.195]  ← on_out
W_out[3]_new  = [0.4, 0.1] - 0.1×[0.025, 0.049]   = [0.397, 0.095]  ← mat_out
```

---

## 5. Verify: Did Loss Decrease?

**New scores with updated embeddings:**
```
score(sat → cat) = [0.095,0.222]·[0.207,0.515] = 0.020+0.114 = 0.134   (was 0.12 ↑)
score(sat → sat) = [0.095,0.222]·[0.097,0.295] = 0.009+0.066 = 0.075   (was 0.07 ↑)
score(sat → on)  = [0.095,0.222]·[0.297,0.195] = 0.028+0.043 = 0.071   (was 0.07 ~)
score(sat → mat) = [0.095,0.222]·[0.397,0.095] = 0.038+0.021 = 0.059   (was 0.06 ↓)
```

**New probabilities:**
```
e^0.134 = 1.143
e^0.075 = 1.078
e^0.071 = 1.074
e^0.059 = 1.061
sum = 4.356

P(cat | sat)_new = 1.143 / 4.356 = 0.262   (was 0.260 ↑)
```

**New loss:**
```
L_new = -log(0.262) = 1.340   (was 1.347 ↓) ✓
```

The probability of the correct context word (cat) increased from 0.260 → 0.262, and loss decreased from 1.347 → 1.340.

After millions of (center, context) pairs, v_sat accumulates updates from every word that appeared near "sat" — v_cat, v_on, v_mat (in our tiny corpus) and thousands more in a real corpus. The final v_sat ends up in a region of embedding space surrounded by words that co-occur with "sat."

---

## 6. The Problem with Full Softmax

Computing softmax over all V words requires:
- V dot products per training step
- V = 50,000 (GPT-2), V = 30,000 (BERT)

For each of billions of training pairs: 50,000 dot products × 2 dimensions = expensive.

More concretely: with V=50K and d=300:
- One softmax step: 50,000 × 300 multiplications = 15M operations
- For 1 billion training pairs: 15 × 10^15 operations — infeasible

**Solution: Negative Sampling.**

---

## 7. Negative Sampling — The Real Training Objective

### 7.1 The Idea

Instead of "predict the correct word from all V words" (multiclass classification), reframe as:

**"Given (center, word) pair, is this word a real context word or a random noise word?"** (binary classification)

For each real pair (sat, cat):
1. The positive example: (sat, cat) — real co-occurrence → label=1
2. Sample k random negative words: say (sat, on) and (sat, mat) → label=0

Now train a binary classifier: σ(v_center · u_word) = probability it's a real pair.

**Why valid?** If u_cat gets pulled toward v_sat (because they co-occur) and u_on gets pushed away from v_sat (negative sample), then the learned embeddings still capture co-occurrence statistics — just via binary classification instead of multinomial.

### 7.2 Loss Function

For center=sat, positive=cat, negatives={on, mat} (k=2):
```
L = -log σ(v_sat · u_cat) - log σ(-v_sat · u_on) - log σ(-v_sat · u_mat)
```

- First term: maximize probability that (sat, cat) is real → push v_sat · u_cat UP
- Other terms: maximize probability that (sat, on) and (sat, mat) are noise → push those scores DOWN

### 7.3 Dry-Run: Forward Pass

Using original embeddings:
```
v_sat  = W_in[1]  = [0.1, 0.2]    ← center
u_cat  = W_out[0] = [0.2, 0.5]    ← positive
u_on   = W_out[2] = [0.3, 0.2]    ← negative 1
u_mat  = W_out[3] = [0.4, 0.1]    ← negative 2
```

**Dot product scores:**
```
s_pos  = v_sat · u_cat = 0.1×0.2 + 0.2×0.5 = 0.12
s_neg1 = v_sat · u_on  = 0.1×0.3 + 0.2×0.2 = 0.07
s_neg2 = v_sat · u_mat = 0.1×0.4 + 0.2×0.1 = 0.06
```

**Sigmoid probabilities:**
```
σ(s_pos)  = σ(0.12)  = 1/(1+e^{-0.12}) = 1/(1+0.887) = 0.530
σ(-s_neg1) = σ(-0.07) = 1/(1+e^{0.07})  = 1/(1+1.073) = 0.482
σ(-s_neg2) = σ(-0.06) = 1/(1+e^{0.06})  = 1/(1+1.062) = 0.485
```

**Loss:**
```
L = -log(0.530) - log(0.482) - log(0.485)
  = 0.635 + 0.730 + 0.724
  = 2.089
```

Notice this is HIGHER than full softmax (1.347) — because negative sampling also penalizes non-context words (0 and -∞ in full softmax is free). But the KEY difference: we only touched 3 output embeddings instead of V=4 (or V=50,000 in a real model).

**Efficiency gain:**
```
Full softmax: V updates per step                 = 4 here, 50,000 in practice
Negative sampling: (1 + k) updates per step      = 3 here, 6-20 in practice
```

For V=50,000, k=5: 50,000 → 6 updates. 8,000× faster.

### 7.4 Backward Pass

**Gradients of loss w.r.t. scores:**
```
∂L/∂s_pos  = σ(s_pos)  - 1 = 0.530 - 1 = -0.470   ← positive: push score up
∂L/∂s_neg1 = σ(s_neg1)    = σ(0.07) = 0.518         ← negative: push score down
∂L/∂s_neg2 = σ(s_neg2)    = σ(0.06) = 0.515         ← negative: push score down
```

**Gradients w.r.t. output embeddings (only 3 touched):**
```
∂L/∂u_cat = ∂L/∂s_pos  × v_sat = -0.470 × [0.1, 0.2] = [-0.047, -0.094]
∂L/∂u_on  = ∂L/∂s_neg1 × v_sat =  0.518 × [0.1, 0.2] = [ 0.052,  0.104]
∂L/∂u_mat = ∂L/∂s_neg2 × v_sat =  0.515 × [0.1, 0.2] = [ 0.052,  0.103]
```

W_out[1] (sat_out) is NOT touched — it wasn't sampled as a negative. This is the efficiency of negative sampling.

**Gradient w.r.t. v_sat:**
```
∂L/∂v_sat = ∂L/∂s_pos × u_cat + ∂L/∂s_neg1 × u_on + ∂L/∂s_neg2 × u_mat

= (-0.470)×[0.2, 0.5] + (0.518)×[0.3, 0.2] + (0.515)×[0.4, 0.1]
= [-0.094, -0.235]     + [0.155,  0.104]     + [0.206,  0.052]
= [0.267, -0.079]
```

**Update (lr=0.1):**
```
W_in[1]_new  = [0.1, 0.2]   - 0.1×[0.267, -0.079] = [0.073, 0.208]   ← v_sat
W_out[0]_new = [0.2, 0.5]   - 0.1×[-0.047,-0.094]  = [0.205, 0.509]   ← u_cat: moved toward v_sat
W_out[2]_new = [0.3, 0.2]   - 0.1×[0.052, 0.104]   = [0.295, 0.190]   ← u_on: moved away
W_out[3]_new = [0.4, 0.1]   - 0.1×[0.052, 0.103]   = [0.395, 0.090]   ← u_mat: moved away
```

### 7.5 Verify

**New scores:**
```
s_pos_new  = [0.073,0.208]·[0.205,0.509] = 0.015+0.106 = 0.121   (was 0.12)
s_neg1_new = [0.073,0.208]·[0.295,0.190] = 0.022+0.040 = 0.062   (was 0.07 ↓ ✓)
s_neg2_new = [0.073,0.208]·[0.395,0.090] = 0.029+0.019 = 0.048   (was 0.06 ↓ ✓)
```

**New loss:**
```
σ(0.121) = 0.530, σ(-0.062) = 0.485, σ(-0.048) = 0.488
L_new = -log(0.530) - log(0.485) - log(0.488)
      = 0.635 + 0.724 + 0.717
      = 2.076   (was 2.089 ↓) ✓
```

Negative word scores decreased. Positive score approximately held. Loss dropped.

### 7.6 Noise Distribution

Negatives aren't sampled uniformly — common words appear too often, rare words not enough. Word2Vec samples from a **unigram distribution raised to the 3/4 power**:

```
P_noise(w) = f(w)^0.75 / Σ f(v)^0.75

Where f(w) = word frequency fraction in corpus.
```

The 0.75 exponent compresses the distribution: very frequent words get sampled less often, rare words more often. This prevents "the" and "a" from dominating the negatives.

For our 4-word corpus with equal frequency (each appears 1-2 times):
- All words have approximately equal noise probability = 1/4 = 0.25

---

## 8. CBOW — The Other Architecture

CBOW (Continuous Bag of Words): predict the **center word** from its context. The opposite of Skip-gram.

**Training pair: context={cat, on}, center=sat**

**Step 1: Average the context embeddings**
```
v_context = (W_in[0] + W_in[2]) / 2
          = ([0.3, 0.5] + [0.4, 0.1]) / 2
          = [0.7, 0.6] / 2
          = [0.35, 0.30]
```

**Step 2: Score against all output embeddings**
```
score(sat) = v_context · W_out[1] = 0.35×0.1 + 0.30×0.3 = 0.035+0.090 = 0.125   ← target
score(cat) = v_context · W_out[0] = 0.35×0.2 + 0.30×0.5 = 0.070+0.150 = 0.220
score(on)  = v_context · W_out[2] = 0.35×0.3 + 0.30×0.2 = 0.105+0.060 = 0.165
score(mat) = v_context · W_out[3] = 0.35×0.4 + 0.30×0.1 = 0.140+0.030 = 0.170
```

**Softmax:**
```
e^0.125 = 1.133, e^0.220 = 1.246, e^0.165 = 1.179, e^0.170 = 1.185
sum = 4.743

P(sat | cat,on) = 1.133/4.743 = 0.239   ← target
```

**Loss:**
```
L_cbow = -log(0.239) = 1.430
```

**Backward:** The gradient flows back to BOTH context word embeddings (W_in[cat] and W_in[on]) equally — they each receive the same gradient (divided by 2 from the average).

### CBOW vs Skip-gram

| | CBOW | Skip-gram |
|---|---|---|
| Task | Context → Center | Center → Context |
| Training signal | 1 pair per center position | k pairs per center position (k=window×2) |
| Speed | Faster (1 softmax per position) | Slower (k softmaxes per position) |
| Common words | Better (averages context → stable gradient) | Worse |
| Rare words | Worse (rare centers averaged away) | Better (each occurrence creates its own gradient) |
| Typical use | Large corpora, common words | Rare words, analogy tasks |

**Word2Vec recommendation:** Skip-gram with negative sampling (SGNS) is the standard for NLP tasks.

---

## 9. GloVe — Global Vectors

### 9.1 The Co-occurrence Matrix

While Word2Vec uses a sliding window at training time, GloVe precomputes a global co-occurrence matrix X.

**Corpus:** "cat sat on mat", window=1

Count every (word_i, word_j) co-occurrence:
```
cat-sat: 1 (cat appears at pos 0, sat appears at pos 1, distance=1)
sat-cat: 1
sat-on:  1
on-sat:  1
on-mat:  1
mat-on:  1
```

**Co-occurrence matrix X** (4×4):
```
     cat  sat  on   mat
cat [  0,   1,   0,   0 ]
sat [  1,   0,   1,   0 ]
on  [  0,   1,   0,   1 ]
mat [  0,   0,   1,   0 ]
```

X[sat][on] = 1 (sat and on co-occur once with window=1)

In a real corpus, X[the][of] might be 1,000,000 (they co-occur all the time).

### 9.2 The GloVe Objective

Train word vectors w_i and context vectors w̃_j (plus biases b_i, b̃_j) to satisfy:

```
w_i · w̃_j + b_i + b̃_j ≈ log(X_ij)     for all (i,j) where X_ij > 0
```

Why log(X_ij)? Because co-occurrence counts span many orders of magnitude (1 to 1M). Log compresses this.

Full weighted loss:
```
J = Σ_{i,j: X_ij > 0} f(X_ij) × (w_i · w̃_j + b_i + b̃_j - log X_ij)²
```

Where f(X_ij) = min(1, (X_ij / x_max)^α) is a weighting function (rare pairs have lower weight, very frequent pairs capped at 1).

For our tiny corpus with x_max=100, all X_ij=1: f(1) = (1/100)^0.75 ≈ 0.032

### 9.3 Dry-Run: One GloVe Update

**Pair: (sat, on), X_sat,on = 1, log(1) = 0.0**

Initial word vectors and biases:
```
w_sat  = [0.1,  0.2],  b_sat  = 0.0
w̃_on   = [0.3,  0.2],  b̃_on   = 0.0
```

**Prediction:**
```
pred = w_sat · w̃_on + b_sat + b̃_on
     = (0.1×0.3 + 0.2×0.2) + 0 + 0
     = 0.03 + 0.04
     = 0.07
```

**Target:** log(X_sat,on) = log(1) = 0.0

**Residual:** pred - target = 0.07 - 0.0 = 0.07

**Weighting:** f(1) = 1.0 (simplified for our tiny corpus)

**Loss for this pair:** J_pair = f(1) × (0.07)² = 0.0049

**Gradients:**
```
∂J_pair/∂w_sat  = 2 × f(1) × (pred - target) × w̃_on
                = 2 × 1.0  × 0.07 × [0.3, 0.2]
                = [0.042, 0.028]

∂J_pair/∂w̃_on   = 2 × f(1) × (pred - target) × w_sat
                = 2 × 1.0  × 0.07 × [0.1, 0.2]
                = [0.014, 0.028]

∂J_pair/∂b_sat  = 2 × f(1) × (pred - target) = 0.14
∂J_pair/∂b̃_on   = 2 × f(1) × (pred - target) = 0.14
```

**Update (lr=0.1):**
```
w_sat_new = [0.1, 0.2] - 0.1×[0.042, 0.028] = [0.096, 0.197]
w̃_on_new  = [0.3, 0.2] - 0.1×[0.014, 0.028] = [0.299, 0.197]
b_sat_new = 0.0 - 0.1×0.14 = -0.014
b̃_on_new  = 0.0 - 0.1×0.14 = -0.014
```

**Verify — new prediction:**
```
pred_new = [0.096,0.197]·[0.299,0.197] + (-0.014) + (-0.014)
         = (0.029 + 0.039) + (-0.028)
         = 0.068 - 0.028
         = 0.040

Residual_new = 0.040 - 0.0 = 0.040   (was 0.07, decreased ↓ ✓)
```

GloVe also processes (on, sat): symmetric pair. After many passes over all non-zero X_ij entries, the dot products stabilize at the log co-occurrence counts.

### 9.4 GloVe Final Embedding

After convergence, the word vector is typically w_i + w̃_i (sum of input and context vector for the same word). GloVe found this performs better than w_i alone.

---

## 10. FastText — Handling OOV with Character N-grams

### 10.1 The Problem

Word2Vec and GloVe: one embedding per word. If a word wasn't seen in training, it gets UNK.

Real-world OOV examples:
- "ChatGPT" (new proper noun) → not in training vocab → UNK
- "unbelievably" → maybe training had "unbelievable" but not this form
- Morphologically rich languages: Turkish "kitaplarda" (books, plural, locative) — if not seen as a unit, OOV

### 10.2 FastText Representation

Instead of one embedding per word, represent a word as the **sum of its character n-gram embeddings**.

For "cat" with n=3:
```
Add boundary markers:  <cat>
Character 3-grams:     <ca,  cat,  at>
Plus the full word:    <cat>

e_cat = e_<ca + e_cat + e_at> + e_<cat>
```

Where each character n-gram (e.g., "<ca") has its own embedding learned during training.

**For "cats" (not in training vocab):**
```
<cats>
3-grams: <ca, cat, ats, ts>
         ↑↑↑
         These are shared with "cat" — so "cats" inherits partial representation

e_cats = e_<ca + e_cat + e_ats + e_ts> + e_<cats>
```

"cats" benefits from the n-grams it shares with known words.

**For "sat" — shared n-gram with "cat":**
```
"cat": 3-grams include "at>"
"sat": 3-grams include "at>"
         ↑
    Shared n-gram — "cat" and "sat" both benefit from learning about "at>" context
```

### 10.3 OOV Representation

For a completely new word "catfish":
```
3-grams: <ca, cat, atf, tfi, fis, ish, sh>
         ↑↑↑
         "<ca", "cat", "at" shared with "cat" in training
```

FastText computes an embedding even for words never seen in training by combining known n-gram embeddings.

Word2Vec: "catfish" → UNK → [0,0,...,0]
FastText: "catfish" → sum of n-gram embeddings → a meaningful vector

---

## 11. The Analogy Trick — Why It Works

The famous result: **king − man + woman ≈ queen**

### 11.1 Why the Geometry Emerges

During Word2Vec training:
- "king" and "queen" appear in similar contexts: "the ___ sat on the throne", "crowned the ___"
- "man" and "woman" appear in similar contexts: "a ___ walked", "the ___ said"

The difference vector: king_vec - man_vec encodes what's different between king and man. That difference = royalty (roughly: the dimensions where "king" >> "man").

Adding that difference to woman_vec: woman_vec + (king_vec - man_vec) gives a vector in "female + royalty" space = queen.

### 11.2 Toy Example with Numbers

After training on a large corpus, suppose word vectors (in 2D, where dim_0 = "royalty", dim_1 = "gender"):

```
king  = [0.8, 0.5]   ← high royalty, mid gender
queen = [0.8, 0.3]   ← high royalty, lower gender (convention: 0.5=male, 0.3=female in this space)
man   = [0.1, 0.5]   ← low royalty, mid gender
woman = [0.1, 0.3]   ← low royalty, lower gender
```

**Arithmetic:**
```
king - man + woman = [0.8-0.1+0.1, 0.5-0.5+0.3]
                   = [0.8, 0.3]
                   = queen ✓
```

The vector lands exactly on "queen" because in this idealized 2D space, the royalty and gender dimensions are cleanly separated.

### 11.3 Why It's Approximate in Practice

Real embeddings are 300-dimensional and trained on noisy internet text. The analogy doesn't hold perfectly because:
- Many dimensions mix royalty + gender + other signals
- "queen" also means "Queen (band)" in some contexts
- Training data has historical gender biases (models reflect corpus statistics)

Typical accuracy on standard analogy tasks:
- Word2Vec (Google News, 300d): ~65% on semantic analogies
- GloVe (Common Crawl, 300d): ~75% on semantic analogies

### 11.4 Analogy Arithmetic Step by Step

```python
# After training, find the word closest to king - man + woman:

target = king_vec - man_vec + woman_vec

# For each word v in vocabulary:
#   similarity = cosine(target, v)
# Return argmax
```

The result is the word whose embedding is geometrically closest to the target point. It's pure vector arithmetic — no special mechanism needed.

---

## 12. Static vs Contextual: Why Word2Vec Has a Ceiling

Word2Vec gives one fixed vector per word regardless of context.

**Problem — polysemy:**
```
"I went to the river bank."
"I deposited money at the bank."

Word2Vec: bank = [same vector in both sentences]
```

The single "bank" embedding is a compromise between the river meaning and financial meaning — good at neither.

**BERT's solution:** Contextual embeddings
```
"river bank" → bank_vec influenced by "river" via attention → [0.3, 0.8, ...]  (aquatic region)
"money bank"  → bank_vec influenced by "money" via attention → [0.7, 0.2, ...]  (financial institution)
```

Two completely different vectors for the same word token.

**Where Word2Vec still wins:**
- Speed: Word2Vec inference = 1 lookup. BERT = 12-layer forward pass.
- Memory: 50K × 300 = 15M params. BERT = 110M params.
- Training: Word2Vec trains in hours on a laptop. BERT needs weeks on TPUs.
- Low-resource settings: small corpora, edge devices

**Rule of thumb:**
- Semantic similarity, recommendation, fast retrieval → Word2Vec/GloVe
- NLU tasks (classification, NER, QA) → BERT

---

## 13. Embedding Space Properties

After training on a large corpus, the following properties emerge:

### Cosine Similarity
```
Similar words cluster together:
  cosine(cat, dog)    ≈ 0.85   ← both domestic animals
  cosine(cat, feline) ≈ 0.92   ← synonym
  cosine(cat, bank)   ≈ 0.10   ← unrelated

From our toy corpus (cat+mat co-occur via "on"):
  cosine(cat_vec, mat_vec) > cosine(cat_vec, sat_vec)  ← if trained long enough
```

### Clustering
Words naturally cluster by semantic category:
```
Animals:  [cat, dog, bird, fish] → close together
Verbs:    [sat, ran, walked]     → different cluster
Locations: [mat, floor, rug]     → another cluster
```

### Syntactic regularity
```
plural: king → kings ≈ queen → queens
tense:  sat → sit   ≈ ran → run
```

These hold because syntactic variants appear in similar contexts.

---

## 14. Code

### 14.1 Skip-gram from Scratch (NumPy)

```python
import numpy as np

# Vocabulary
vocab = ['cat', 'sat', 'on', 'mat']
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)  # 4
d = 2           # embedding dim

# Random initialization
np.random.seed(42)
W_in  = np.random.randn(V, d) * 0.1   # input embeddings
W_out = np.random.randn(V, d) * 0.1   # output embeddings

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def skipgram_negative_sampling(center_idx, context_idx, neg_indices, lr=0.01):
    """One training step with negative sampling."""
    v_c = W_in[center_idx].copy()          # center embedding

    total_loss = 0.0
    grad_v_c = np.zeros(d)

    # Positive example
    u_pos = W_out[context_idx]
    score_pos = np.dot(v_c, u_pos)
    sig_pos = sigmoid(score_pos)
    total_loss -= np.log(sig_pos + 1e-8)

    grad_u_pos = (sig_pos - 1) * v_c      # gradient w.r.t. output embed
    grad_v_c  += (sig_pos - 1) * u_pos    # accumulate gradient for center

    # Negative examples
    for neg_idx in neg_indices:
        u_neg = W_out[neg_idx]
        score_neg = np.dot(v_c, u_neg)
        sig_neg = sigmoid(score_neg)       # want this to be low
        total_loss -= np.log(1 - sig_neg + 1e-8)

        grad_u_neg = sig_neg * v_c         # gradient w.r.t. negative output embed
        grad_v_c  += sig_neg * u_neg       # accumulate gradient for center

        W_out[neg_idx] -= lr * grad_u_neg

    # Apply updates
    W_out[context_idx] -= lr * grad_u_pos
    W_in[center_idx]   -= lr * grad_v_c

    return total_loss

# Build training pairs from "cat sat on mat", window=1
corpus = [0, 1, 2, 3]  # cat sat on mat
window = 1
pairs = []
for center_pos, center_idx in enumerate(corpus):
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        context_pos = center_pos + offset
        if 0 <= context_pos < len(corpus):
            pairs.append((center_idx, corpus[context_pos]))

print(f"Training pairs: {[(vocab[c], vocab[ctx]) for c,ctx in pairs]}")

# Training loop
np.random.seed(0)
for epoch in range(500):
    total_loss = 0.0
    np.random.shuffle(pairs)
    for center_idx, context_idx in pairs:
        # Sample 2 negative words (exclude positive)
        neg_indices = []
        while len(neg_indices) < 2:
            neg = np.random.randint(V)
            if neg != context_idx:
                neg_indices.append(neg)

        loss = skipgram_negative_sampling(center_idx, context_idx, neg_indices)
        total_loss += loss

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

print("\nLearned embeddings (W_in):")
for i, word in enumerate(vocab):
    print(f"  {word}: {W_in[i]}")

print("\nSimilarities after training:")
for (a, b) in [('cat','mat'), ('cat','sat'), ('sat','on')]:
    sim = cosine_sim(W_in[word2idx[a]], W_in[word2idx[b]])
    print(f"  cosine({a}, {b}) = {sim:.4f}")
```

### 14.2 GloVe from Scratch (NumPy)

```python
import numpy as np
from collections import defaultdict

# Build co-occurrence matrix
corpus = ['cat', 'sat', 'on', 'mat']
vocab = list(set(corpus))
word2idx = {w: i for i, w in enumerate(sorted(vocab))}
V = len(vocab)
window = 1

# Count co-occurrences
X = np.zeros((V, V))
for i, center in enumerate(corpus):
    ci = word2idx[center]
    for j in range(max(0, i-window), min(len(corpus), i+window+1)):
        if i != j:
            cj = word2idx[corpus[j]]
            X[ci][cj] += 1

print("Co-occurrence matrix X:")
print(X)

# GloVe parameters
d = 2
x_max = 100
alpha = 0.75
lr = 0.05

np.random.seed(42)
W = np.random.randn(V, d) * 0.1       # word vectors
W_ctx = np.random.randn(V, d) * 0.1  # context vectors
b = np.zeros(V)                        # word biases
b_ctx = np.zeros(V)                    # context biases

def weight_func(x):
    return min(1.0, (x / x_max) ** alpha) if x > 0 else 0.0

# Training
for epoch in range(1000):
    total_loss = 0.0
    for i in range(V):
        for j in range(V):
            if X[i][j] == 0:
                continue
            f = weight_func(X[i][j])
            target = np.log(X[i][j])
            pred = np.dot(W[i], W_ctx[j]) + b[i] + b_ctx[j]
            residual = pred - target
            total_loss += f * residual**2

            # Gradients
            dW_i     = 2 * f * residual * W_ctx[j]
            dW_ctx_j = 2 * f * residual * W[i]
            db_i     = 2 * f * residual
            db_ctx_j = 2 * f * residual

            W[i]     -= lr * dW_i
            W_ctx[j] -= lr * dW_ctx_j
            b[i]     -= lr * db_i
            b_ctx[j] -= lr * db_ctx_j

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}: loss = {total_loss:.6f}")

# Final embeddings = word vector + context vector
embeddings = W + W_ctx
print("\nGloVe embeddings:")
for word, idx in word2idx.items():
    print(f"  {word}: {embeddings[idx]}")
```

### 14.3 Using gensim

```python
from gensim.models import Word2Vec, KeyedVectors
import numpy as np

# Train on toy corpus
sentences = [
    ['cat', 'sat', 'on', 'mat'],
    ['the', 'cat', 'sat', 'on', 'the', 'mat'],
    ['a', 'cat', 'sat', 'on', 'a', 'mat'],
]

# Skip-gram with negative sampling
model = Word2Vec(
    sentences=sentences,
    vector_size=50,    # embedding dimension (use 300 for production)
    window=2,          # context window size
    min_count=1,       # include words with freq >= 1
    sg=1,              # 1=skip-gram, 0=CBOW
    negative=5,        # number of negative samples
    epochs=100,
    seed=42
)

# Word vectors
print(model.wv['cat'])           # numpy array of size 50
print(model.wv.similarity('cat', 'mat'))    # cosine similarity
print(model.wv.most_similar('cat', topn=3)) # nearest neighbors

# Analogy: sat - cat + mat = ?
result = model.wv.most_similar(positive=['sat', 'mat'], negative=['cat'], topn=3)
print(result)

# Load pretrained Word2Vec (Google News, 300d)
# (requires downloading the file separately)
# pretrained = KeyedVectors.load_word2vec_format('GoogleNews-vectors-300.bin.gz', binary=True)
# pretrained.most_similar(positive=['king','woman'], negative=['man'], topn=5)
# → [('queen', 0.71), ...]

# Load pretrained GloVe (with gensim)
# from gensim.scripts.glove2word2vec import glove2word2vec
# glove2word2vec('glove.6B.100d.txt', 'glove.word2vec.txt')
# glove = KeyedVectors.load_word2vec_format('glove.word2vec.txt')
```

### 14.4 FastText

```python
from gensim.models import FastText

# FastText automatically handles OOV via character n-grams
model = FastText(
    sentences=sentences,
    vector_size=50,
    window=2,
    min_count=1,
    sg=1,              # skip-gram
    min_n=3,           # min char n-gram size
    max_n=6,           # max char n-gram size
    epochs=100,
    seed=42
)

# OOV word: "catfish" was never in training data
print(model.wv['catfish'])           # Still works! Built from char n-grams
print(model.wv.similarity('cat', 'catfish'))   # Similar because of shared n-grams

# vs Word2Vec
w2v = Word2Vec(sentences, vector_size=50, min_count=1, seed=42)
try:
    w2v.wv['catfish']  # KeyError — not in vocabulary
except KeyError:
    print("Word2Vec: catfish is OOV")
```

### 14.5 Using Pretrained GloVe as Feature Input

```python
import numpy as np
import torch
import torch.nn as nn

# Build embedding layer from pretrained GloVe
def load_glove_embeddings(glove_path, vocab, embed_dim=100):
    """Load GloVe vectors for words in vocab."""
    embeddings = np.random.randn(len(vocab), embed_dim) * 0.01
    word2idx = {w: i for i, w in enumerate(vocab)}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in word2idx:
                vec = np.array(parts[1:], dtype=np.float32)
                embeddings[word2idx[word]] = vec
    return embeddings

# Use in a model
vocab = ['cat', 'sat', 'on', 'mat', '<PAD>', '<UNK>']
embed_dim = 100

# glove_embeddings = load_glove_embeddings('glove.6B.100d.txt', vocab, embed_dim)

# For demo: random init
glove_embeddings = np.random.randn(len(vocab), embed_dim) * 0.1

embedding_layer = nn.Embedding(len(vocab), embed_dim)
embedding_layer.weight = nn.Parameter(torch.tensor(glove_embeddings, dtype=torch.float32))
embedding_layer.weight.requires_grad = True   # True: fine-tune; False: freeze

# Input sequence: cat sat on mat → indices [0, 1, 2, 3]
input_ids = torch.tensor([[0, 1, 2, 3]])
embedded = embedding_layer(input_ids)   # shape: [1, 4, 100]
print(embedded.shape)
```

### 14.6 Cosine Similarity and Nearest Neighbors

```python
import numpy as np

def cosine_similarity_matrix(embeddings):
    """Compute all pairwise cosine similarities."""
    # Normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    return normalized @ normalized.T

def nearest_neighbors(word, embeddings, vocab, word2idx, topn=3):
    idx = word2idx[word]
    sim_matrix = cosine_similarity_matrix(embeddings)
    similarities = sim_matrix[idx]
    # Sort descending, skip self
    ranked = np.argsort(similarities)[::-1]
    neighbors = [(vocab[i], similarities[i]) for i in ranked if i != idx][:topn]
    return neighbors

# Example with our toy embeddings
vocab = ['cat', 'sat', 'on', 'mat']
word2idx = {w: i for i, w in enumerate(vocab)}

embeddings = np.array([
    [0.3, 0.5],   # cat
    [0.1, 0.2],   # sat
    [0.4, 0.1],   # on
    [0.2, 0.3],   # mat
])

print(nearest_neighbors('cat', embeddings, vocab, word2idx))
# After training, 'mat' should be nearest (both appear near 'on')
```

---

## 15. Gotchas

**1. Two separate matrices W_in and W_out — which one do you use?**
Common choices: (a) W_in only (gensim default), (b) average of W_in and W_out (GloVe convention), (c) concatenate both → 2d-dimensional vectors. All are used. W_in is most common. Use whichever gives better validation performance.

**2. Negative sampling favors common words — check your noise distribution.**
With uniform sampling, "the", "a", "of" dominate negatives. Always use the unigram^0.75 distribution. gensim does this automatically; NumPy implementations often forget it.

**3. Word2Vec is sensitive to window size.**
Small window (2-5): captures syntactic similarity (words with same grammatical role).
Large window (10+): captures semantic similarity (words in same topic).
Typical recommendation: window=5 for semantic, window=2 for syntactic tasks.

**4. Subword tokenizers break Word2Vec's lookup.**
Word2Vec has one vector per word. If your downstream model uses BPE/WordPiece, "running" might be split to ["run", "##ning"]. Word2Vec can't provide a vector for "##ning". Align tokenization between your embedding model and downstream model.

**5. The analogy accuracy numbers are misleading.**
65% on the Google analogy task sounds great, but the task has 19,544 examples and many are trivially solvable (capital-country pairs, currency-country pairs). For semantic analogies specifically, accuracy is lower. Don't overfit to analogy benchmarks.

**6. Pretrained embeddings have demographic biases baked in.**
Word2Vec trained on Google News: doctor → more similar to man, nurse → more similar to woman. These biases come from corpus statistics. For sensitive applications, use debiased embeddings or test for bias explicitly.

**7. GloVe's co-occurrence matrix is huge for large corpora.**
For V=400K (GloVe's actual vocab), the full X matrix is 400K × 400K = 160 billion entries. In practice, only non-zero entries are stored (sparse matrix). Even so, it's tens of GB for a large corpus. Word2Vec avoids this by streaming through the corpus.

**8. min_count matters more than you think.**
Words appearing < min_count times have noisy embeddings (not enough context to learn from). gensim default is min_count=5. For a small domain corpus, you might need min_count=2 or 1.

---

## 16. Q&A

**Q: Why does negative sampling work? Doesn't it train an easier objective than full softmax?**  
A: Yes, it's easier — binary classification instead of V-way classification. But the gradient directions are the same: positive pairs pull embeddings together, negative pairs push them apart. The mathematical justification: negative sampling approximates the NCE (Noise Contrastive Estimation) objective, which is a consistent estimator of the softmax objective as k→∞ (more negatives → tighter approximation). In practice, k=5-20 gives embeddings nearly as good as full softmax at a fraction of the cost.

**Q: What's the difference between Word2Vec and a language model like GPT?**  
A: Word2Vec discards the prediction model — the LM head (W_out) is thrown away after training. Only the embeddings (W_in) are kept. GPT keeps everything: the entire transformer is the output. Word2Vec learns shallow (one matrix lookup + one dot product) representations. GPT learns deep, context-dependent representations through 12-96 layers of attention. Word2Vec is a byproduct of training a fake prediction task; GPT is the actual language model.

**Q: When would you fine-tune Word2Vec embeddings in a downstream model?**  
A: Almost always set `requires_grad=True` for the embedding layer in fine-tuning — letting gradients flow into the embeddings improves performance. The exception: very small training datasets (<1000 examples) where fine-tuning risks overfitting embeddings to training noise. In that case, freeze the embeddings. Practical heuristic: freeze for < 1K examples, fine-tune for > 10K.

**Q: Why does GloVe tend to outperform Word2Vec on analogy tasks?**  
A: GloVe directly minimizes a loss tied to log co-occurrence ratios. The ratio log(P(cat|river)/P(cat|ice)) encodes relationships more cleanly than the sliding-window approximation in Word2Vec. GloVe uses global statistics; Word2Vec sees each context window independently. For analogy tasks that require global co-occurrence structure, GloVe has a structural advantage. For most downstream NLP tasks, the difference is small (both are dominated by BERT).

**Q: Can you use Word2Vec embeddings to initialize BERT?**  
A: No — BERT's tokenizer uses WordPiece subwords ("unhappiness" → ["un","##happiness"]), while Word2Vec has one vector per whole word. The vocabularies don't align. You can initialize BERT's embedding layer by mapping whole words to their Word2Vec vectors (for tokens that are whole words), but this rarely helps — BERT's pretrained embeddings are already far superior to Word2Vec.

---

## 17. Connections

**To text representations (`fundamentals/02_text_representations.md`):**
- Word2Vec is the bridge from sparse (TF-IDF) to dense representations
- Same corpus "cat sat on mat" used in TF-IDF examples
- Cosine similarity: same formula, completely different result (0 for TF-IDF, 0.85+ for Word2Vec)

**To RNN end-to-end (`sequence_models/02_rnn_end_to_end.md`):**
- The embedding matrix lookup is step 0 in every RNN forward pass
- W_in is the same matrix used in the RNN's input layer
- The same cat=[1.0,0.5], sat=[0.2,0.3] etc. are pretrained Word2Vec vectors (in our toy setting they're manually set, but in production they'd come from Word2Vec training)

**To transformer end-to-end (`sequence_models/06_transformer_end_to_end.md`):**
- Transformers have their own embedding matrix (learned end-to-end via MLM/CLM)
- Positional encodings are added ON TOP of the token embeddings — the same lookup mechanism
- Weight tying: W_lm = token_embeds.T — this is exactly the W_in/W_out duality from Word2Vec

**To BERT (`5.transformers/models/05_bert_end_to_end.md`):**
- BERT's token embeddings = contextual replacement for Word2Vec
- MLM trains BERT's embeddings the same way Word2Vec trains W_in: by making words that appear in similar contexts have similar representations — but with full bidirectional attention instead of a single dot product

**To tokenization (`5.transformers/fundamentals/03_tokenization.md`):**
- Word2Vec: one vector per word-level token → limited by vocab size
- BPE/WordPiece: subword tokens → eliminates OOV, but Word2Vec can't be used directly
- FastText bridges this: character n-grams handle OOV at the word level, without needing subword tokenization

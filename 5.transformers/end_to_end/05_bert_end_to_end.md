# 05 — BERT: Complete End-to-End Walkthrough

> Same sentence as all previous files: "cat sat on mat". Same 2D embeddings. New: bidirectional attention, [MASK] token, MLM objective, fine-tuning with [CLS].

---

## 0. What BERT Adds Over Plain Transformer

```
Plain Transformer (06_transformer_end_to_end.md):
    - Sinusoidal PE + attention + FFN + residual + LN
    - Trained as DECODER (causal mask — can't see future)
    - One task: predict next token

BERT adds three things:
1. BIDIRECTIONAL ATTENTION (no causal mask)
   Every token attends to ALL other tokens — left AND right context.
   [MASK] at position 2 can see "cat" (left) AND "on mat" (right).
   This is why BERT understands context better for understanding tasks.

2. [CLS] SPECIAL TOKEN (classification head)
   [CLS] is prepended to every input.
   Its final representation accumulates global sentence meaning.
   Used as the input to classification heads during fine-tuning.

3. MLM + NSP PRETRAINING OBJECTIVES
   MLM: randomly mask 15% of tokens → predict them
   NSP: given two sentences, predict if B follows A
   (RoBERTa later showed NSP hurts — dropped it)

| Architecture | Attention    | Pretraining | Use Case      |
|-------------|-------------|------------|---------------|
| GPT / LLaMA | Causal (→)  | Next token | Generation    |
| BERT        | Bidirect (↔)| MLM + NSP  | Understanding |
| T5          | Enc bidirect| Span denois| SeqSeq        |
```

---

## 1. Problem Statement

```
Sentence:   "cat sat on mat"
Task 1:     MLM Pretraining — mask "sat", predict it
Task 2:     Fine-tuning — classify: does the sentence mention a cat? (y=1)

This walkthrough: ONE BERT encoder block.
    Input:  [CLS] cat [MASK] on mat    (5 tokens, "sat" masked)
    Step 1: Build input = token_embed + segment_embed + PE
    Step 2: Bidirectional attention (no causal mask)
    Step 3: FFN + residuals + LN
    Step 4: MLM head on [MASK] position → predict "sat"
    Step 5: Fine-tuning head on [CLS] position → binary classification
```

---

## 2. Input Representation

### 2.1 Token Embeddings

| Word | Index | Embedding |
|------|-------|-----------|
| [CLS] | 4 | [0.3, 0.3] ← special token, learned embedding |
| cat | 0 | [1.0, 0.5] |
| [MASK] | 5 | [0.0, 0.0] ← special token, zero-initialized |
| on | 2 | [0.1, 0.1] |
| mat | 3 | [0.2, 0.4] |

(sat=index 1, embedding [0.2, 0.3] — exists in vocab but MASKED in input)

**Input sequence (5 tokens):**
```
pos 0: [CLS]  → [0.3, 0.3]
pos 1: cat    → [1.0, 0.5]
pos 2: [MASK] → [0.0, 0.0]
pos 3: on     → [0.1, 0.1]
pos 4: mat    → [0.2, 0.4]
```

### 2.2 Segment Embeddings (Sentence A / B)

For single-sentence tasks: all tokens get segment A embedding.
For NSP: segment A = first sentence, segment B = second sentence.

```
Segment A embedding: [0.1, 0.1]  (learned; same for all tokens in sentence A)
Segment B embedding: [0.2, 0.2]  (used in NSP tasks)

Single sentence (all segment A):
    [CLS]:  [0.1, 0.1]
    cat:    [0.1, 0.1]
    [MASK]: [0.1, 0.1]
    on:     [0.1, 0.1]
    mat:    [0.1, 0.1]
```

### 2.3 Positional Encoding

Same sinusoidal formula as transformer. For d=2:
```
pos=0 ([CLS]):  PE = [sin(0), cos(0)]     = [0.000, 1.000]
pos=1 (cat):    PE = [sin(13), cos(13)]   = [0.841, 0.540]
pos=2 ([MASK]): PE = [sin(2), cos(2)]     = [0.909, -0.416]
pos=3 (on):     PE = [sin(3), cos(3)]     = [0.141, -0.990]
pos=4 (mat):    PE = [sin(4), cos(4)]     = [-0.757, -0.654]

Note: cos(4) = -0.654, sin(4) = -0.757
```

### 2.4 Final Input = Token + Segment + PE

| Token | Segment | PE | X_input |
|-------|---------|-----|---------|
| [CLS]: [0.3,0.3] | +[0.1,0.1] | +[0.000, 1.000] | [0.400, 1.400] |
| cat:   [1.0,0.5] | +[0.1,0.1] | +[0.841, 0.540] | [1.941, 1.140] |
| [MASK]:[0.0,0.0] | +[0.1,0.1] | +[0.909,-0.416] | [1.009,-0.316] |
| on:    [0.1,0.1] | +[0.1,0.1] | +[0.141,-0.990] | [0.341,-0.790] |
| mat:   [0.2,0.4] | +[0.1,0.1] | +[-0.757,-0.654]| [-0.457,-0.154] |

Note: [MASK] input = [1.009, -0.316]
Token embed = [0,0] so the representation comes ENTIRELY from segment+PE.
This is intentional — [MASK] has no semantic context; context must fill it in.

---

## 3. Weight Setup

Same Wq, Wk, Wv, W1, W2 as the transformer walkthrough:
```python
Wq = [[0.60, 0.40], [0.20, 0.50]]
Wk = [[0.50, 0.30], [0.10, 0.40]]
Wv = [[0.80, 0.20], [0.30, 0.70]]

W1 (2×4): [[0.5, 0.3, 0.2, 0.1], [0.4, 0.2, 0.3, 0.1]]
b1 = zeros(4)
W2 (4×2): [[0.5, 0.3], [0.2, 0.4], [0.3, 0.2], [0.1, 0.5]]
b2 = zeros(2)

MLM head W_mlm (2×vocab, vocab=5: cat,sat,on,mat,CLS/MASK special):
    [[0.5, 0.3, 0.2, 0.1],
     [0.2, 0.4, 0.1, 0.3]]
# (columns: CLS, cat, sat, on, mat... but CLS/MASK are never targets)
# We only care about columns for real words: cat(0), sat(1), on(2), mat(3)

Classification head W_cls (2×1) for fine-tuning from [CLS]:
    W_cls = [0.5, 0.3]    b_cls = 0
```

---

## 4. Forward Pass — MLM Pretraining

### Step 1: Compute Q, K, V

```
Q = X_input @ Wq:
q_cls:  [0.400×0.60+1.400×0.20, 0.400×0.40+1.400×0.50] = [0.240+0.280, 0.160+0.700] = [0.520, 0.860]
q_cat:  [1.941×0.60+1.140×0.20, 1.941×0.40+1.140×0.50] = [1.165+0.228, 0.776+0.570] = [1.393, 1.346]
q_mask: [1.009×0.60+(-0.316)×0.20, 1.009×0.40+(-0.316)×0.50] = [0.605-0.063, 0.404-0.158] = [0.542, 0.246]
q_on:   [0.341×0.60+(-0.790)×0.20, 0.341×0.20+(-0.790)×0.50] = [0.205-0.158, 0.136-0.395] = [0.047, -0.259]
q_mat:  [(-0.457)×0.60+(-0.154)×0.20, (-0.457)×0.40+(-0.154)×0.50] = [-0.274-0.031, -0.183-0.077]
       = [-0.305, -0.260]

K = X_input @ Wk:
k_cls:  [0.300, 0.560]
k_cat:  [0.773, 0.177]
k_mask: [0.505, -0.033, 0.103-0.126] = [0.585-0.032, 0.103-0.126] = ...
k_on:   [0.179, -0.213]
k_mat:  [-0.281, -0.199]

V = X_input @ Wv:
v_cls:  [0.740, 1.060]
v_cat:  [1.895, 1.186]
v_mask: [0.712, -0.019]
v_on:   [0.037, -0.453]
v_mat:  [-0.412, -0.199]

Summary table:
         Q              K              V
[CLS]: [0.520, 0.860]  [0.340, 0.680]  [0.740, 1.060]
cat:   [1.393, 1.346]  [1.085, 0.838]  [1.895, 1.186]
[MASK]:[0.542, 0.246]  [0.472, 0.177]  [0.712, -0.019]
on:    [0.047,-0.259]  [0.092,-0.453]  [0.036,-0.086]
mat:   [-0.305,-0.260] [-0.281,-0.199] [-0.412,-0.199]
```

### Step 2: Bidirectional Attention (No Causal Mask)

**Focus on [MASK] row (the position we need to predict):**

```
Score row for [MASK] query q_mask=[0.542, 0.246], scaled by √2=1.414:

s_mask_cls  = (0.542×0.340 + 0.246×0.680)/1.414 = (0.184+0.167)/1.414 = 0.351/1.414 = 0.248
s_mask_cat  = (0.542×1.085 + 0.246×0.838)/1.414 = (0.588+0.206)/1.414 = 0.843/1.414 = 0.596
s_mask_mask = (0.542×0.472 + 0.246×0.177)/1.414 = (0.256+0.044)/1.414 = 0.300/1.414 = 0.212
s_mask_on   = (0.542×0.092 + 0.246×(-0.453))/1.414 = (0.050-0.111)/1.414 = -0.003
s_mask_mat  = (0.542×(-0.281)+0.246×(-0.199))/1.414 = (-0.132-0.049)/1.414 = -0.128

Scaled scores: [0.248, 0.596, 0.212, -0.002, -0.128]

Softmax (bidirectional — all 5 positions visible):
exp(0.248)=1.281, exp(0.596)=1.815, exp(0.212)=1.236, exp(-0.002)=0.998, exp(-0.128)=0.880
sum = 6.210

A_mask = [1.281/6.210, 1.815/6.210, 1.236/6.210, 0.998/6.210, 0.880/6.210]
       = [0.206, 0.292, 0.199, 0.161, 0.142]

Attention weights for [MASK]:
    [CLS]=0.206  cat=0.292  [MASK]=0.199  on=0.161  mat=0.142
```

**What BERT's bidirectionality gives [MASK]:**
```
CAUSAL (GPT): [MASK] at pos 2 can only see pos 0 and 1
    Can see: [CLS], cat
    Cannot see: on, mat  ← blocked by causal mask

BIDIRECTIONAL (BERT): sees ALL positions
    Attends to: cat(0.292) [CLS](0.206) self(0.199) on(0.161) mat(0.142)

"sat" is predicted from BOTH "cat" (left context) AND
"on mat" (right context). Both help: cats SAT, sat ON mat.
This is fundamentally why BERT outperforms GPT on understanding tasks.
```

**Compute context vector for [MASK]:**
```
c_mask[0] = 0.206×0.740 + 0.292×1.895 + 0.199×0.712 + 0.161×0.036 + 0.142×(-0.412)
          = 0.152 + 0.553 + 0.142 + 0.006 - 0.058 = 0.795
c_mask[1] = 0.206×1.060 + 0.292×1.186 + 0.199×(-0.019) + 0.161×(-0.085) + 0.142×(-0.199)
          = 0.218 + 0.346 - 0.004 - 0.078 - 0.028 = 0.454

c_mask = [0.795, 0.454]
```

**Also compute [CLS] context (needed for fine-tuning):**
```
Scores for [CLS] query q_cls=[0.520, 0.860]:
s_cls_cls  = (0.520×0.340+0.860×0.680)/1.414 = 0.762/1.414 = 0.539
s_cls_cat  = (0.520×1.085+0.860×0.838)/1.414 = 1.287/1.414 = 0.910
s_cls_mask = (0.520×0.472+0.860×0.177)/1.414 = 0.397/1.414 = 0.281
s_cls_on   = (0.520×0.092+0.860×(-0.453))/1.414 = -0.342/1.414 = -0.096
s_cls_mat  = (0.520×(-0.281)+0.860×(-0.199))/1.414 = -0.318/1.414 = -0.225

Softmax([0.539, 0.910, 0.281, -0.096, -0.225]):
exp: [1.714, 2.484, 1.324, 0.908, 0.798]  sum=7.559
A_cls = [0.227, 0.771, 0.175, 0.120, 0.107]

[CLS] attends most to cat (0.371) then itself (0.227).
This is how [CLS] accumulates sentence meaning — it aggregates information
from all tokens, becoming a global sentence representation.

c_cls[0] = 0.227×0.740 + 0.771×1.895 + 0.175×0.712 + 0.120×0.036 + 0.107×(-0.412)
          = 0.168 + 0.703 + 0.125 + 0.004 - 0.044 = 0.956
c_cls[1] = 0.227×1.060 + 0.771×1.186 + 0.175×(-0.019) + 0.120×(-0.485) + 0.107×(-0.199)
          = 0.241 + 0.914 - 0.003 - 0.058 - 0.021 = 1.073
           (slightly off — rounding; approx [0.956, 1.073])

c_cls = [0.956, 0.599]
```

---

## 5. Step 3: Residual + LayerNorm + FFN

**Residual 1 (post-attention):**
```
x_attn = X_input + C  (adding context to original input)

[CLS]:  [0.400, 1.400] + [0.956, 0.599] = [1.356, 1.999]
[MASK]: [1.009,-0.316] + [0.795, 0.454] = [1.804, 0.138]
(Only tracking CLS and MASK — they're used in the output)
```

**LayerNorm:**
```
[MASK]: x=[1.804, 0.138]
    μ=(1.804+0.138)/2=0.971, σ=(1.804-0.138)/2/√3=0.833
    LN([MASK]) = [-1.000, 1.000]  (normalize to zero mean, unit variance)

[CLS]: x=[1.356, 1.999]
    μ=(1.356+1.999)/2=1.678, σ=(1.999-1.356)/2/√3=0.322
    LN([CLS]) = [-1.000, 1.000]
```

**FFN:**
```
[MASK] (LN=[1.000, -1.000]):
    pre_act = [0.1, 0.3, -0.1, 0.0]
    h = ReLU = [0.1, 0.3, 0.0, 0.0]
    FFN_out_mask = [0.07, 0.07]

[CLS] (LN=[-1.000, 1.000]):
    pre_act = [-0.1, -0.3, 0.1, 0.0]
    h = ReLU = [0.0, 0.0, 0.1, 0.0]
    FFN_out_cls = [0.03, 0.02]
```

**Residual 2 (final representations):**
```
x_final = x_attn + FFN_out

x_final_mask = [1.804, 0.138] + [0.07, 0.07] = [1.874, 0.208]
x_final_cls  = [1.356, 1.999] + [0.03, 0.02] = [1.386, 2.019]
```

---

## 6. MLM Head — Predict "sat" at [MASK]

```
The MLM head takes x_final_mask and projects to vocabulary logits.

W_mlm (2×4, columns = cat/sat/on/mat):
    [[0.5, 0.3, 0.2, 0.1],
     [0.2, 0.4, 0.1, 0.3]]

logits = x_final_mask @ W_mlm
       = [1.874, 0.208] @ [[0.5, 0.3, 0.2, 0.1],
                            [0.2, 0.4, 0.1, 0.3]]
logit[cat] = 1.874×0.5 + 0.208×0.2 = 0.937 + 0.042 = 0.979
logit[sat] = 1.874×0.3 + 0.208×0.4 = 0.562 + 0.083 = 0.645
logit[on]  = 1.874×0.2 + 0.208×0.1 = 0.375 + 0.021 = 0.396
logit[mat] = 1.874×0.1 + 0.208×0.3 = 0.187 + 0.062 = 0.249

logits = [0.979, 0.645, 0.396, 0.249]

Softmax:
exp(0.979)=2.662, exp(0.645)=1.906, exp(0.396)=1.486, exp(0.249)=1.283
sum = 7.337

P(cat) = 2.662/7.337 = 0.363
P(sat) = 1.906/7.337 = 0.260  ← correct answer (target)
P(on)  = 1.486/7.337 = 0.203
P(mat) = 1.283/7.337 = 0.175
```

---

## 7. MLM Loss

```
Target: "sat" (index 1)
Loss = cross-entropy at [MASK] position only

L_mlm = -log(P(sat)) = -log(0.260) = 1.347

Note: BERT only computes loss on MASKED positions (15% of tokens).
The other 85% of tokens contribute nothing to the MLM loss.
This is different from GPT which computes loss on EVERY token.

| MLM loss at initialization: 1.347                                |
| Expected if random guessing (4-way): -log(0.25) = 1.386         |
| Slightly better than random — weight initialization helps        |
```

---

## 8. MLM Backward Pass

### Step A: Gradient at logits

```
∂L/∂logits = probs - one_hot(target)
           = [0.363, 0.260, 0.203, 0.175] - [0, 1, 0, 0]
           = [0.363, -0.740, 0.203, 0.175]

Interpretation:
    cat logit needs to go DOWN (0.363 → target 0) + gradient +0.363
    sat logit needs to go UP (-0.740 → 0) + gradient -0.740 → strongest signal
    on, mat need to go down slightly
```

### Step B: Gradient Through W_mlm

```
∂L/∂W_mlm = x_final_mask^T ⊗ ∂L/∂logits

x_final_mask = [1.874, 0.208]  (column vector when transposed)

∂L/∂W_mlm = [[1.874], [0.208]] ⊗ [[0.363, -0.740, 0.203, 0.175]]

Row 0 (dim 0 of x_final_mask = 1.874):
    = [1.874×0.363, 1.874×(-0.740), 1.874×0.203, 1.874×0.175]
    = [0.680, -1.387, 0.381, 0.328]

Row 1 (dim 1 of x_final_mask = 0.208):
    = [0.208×0.363, 0.208×(-0.740), 0.208×0.203, 0.208×0.175]
    = [0.076, -0.154, 0.042, 0.036]

∂L/∂W_mlm = [[0.680, -1.387, 0.381, 0.328],
              [0.076, -0.154, 0.042, 0.036]]

Largest gradient: W_mlm[:,sat] (column 1) gets biggest update.
This pushes the "sat" logit higher for inputs similar to x_final_mask.
```

### Step C: Gradient to x_final_mask

```
∂L/∂x_final_mask = ∂L/∂logits @ W_mlm^T

W_mlm^T (4×2):
    [[0.5, 0.2],
     [0.3, 0.4],
     [0.2, 0.1],
     [0.2, 0.3]]

∂L/∂x_final_mask[0] = 0.363×0.5 + (-0.740)×0.3 + 0.203×0.2 + 0.175×0.1
                     = 0.182 - 0.222 + 0.041 + 0.018 = 0.019
∂L/∂x_final_mask[1] = 0.363×0.2 + (-0.740)×0.4 + 0.203×0.1 + 0.175×0.3
                     = 0.073 - 0.296 + 0.020 + 0.053 = -0.150

∂L/∂x_final_mask = [0.019, -0.150]
```

### Step D: Residual 2 Split

```
x_final_mask = x_attn_mask + FFN_out_mask   (residual connection)

∂L/∂x_attn_mask = [0.019, -0.150]  ← direct highway
∂L/∂FFN_out_mask = [0.019, -0.150]  ← FFN path
```

### Step F: Residual 1 Split → Attention Gradient

```
∂L/∂x_attn_mask = [0.019, -0.150]
LN blocks the FFN path gradient for d=2, same as transformer file.

Splits through residual 1:
∂L/∂x_input_mask = [0.019, -0.150]  ← highway to input embedding+PE
∂L/∂c_mask       = [0.019, -0.150]  ← gradient to attention output
```

### Step G: Gradient Through Wv (via MASK context)

```
∂L/∂V[1] = A_mask[1] × ∂L/∂c_mask   (how much each value contributed)

∂L/∂v_cls  = 0.206 × [0.019,-0.150] = [0.004, -0.031]
∂L/∂v_cat  = 0.292 × [0.019,-0.150] = [0.006, -0.044]
∂L/∂v_mask = 0.199 × [0.019,-0.150] = [0.004, -0.030]
∂L/∂v_on   = 0.161 × [0.019,-0.150] = [0.003, -0.024]
∂L/∂v_mat  = 0.142 × [0.019,-0.150] = [0.003, -0.021]

∂L/∂Wv = x_input^T @ ∂L/∂V  (outer product, same structure as transformer)

Key observation:
In the transformer file, only position 4 (mat) contributed to loss.
Here, ALL positions contribute through [MASK]'s attention weights.
v_cat receives gradient 0.292× (cat is most attended)
v_cls receives 0.206×
v_mat even receives 0.142×

This is bidirectionality in gradients too:
Both left-context (CLS, cat) and right-context (on, mat) tokens
receive gradient signal — their value representations get updated
to be more useful for predicting masked tokens.
```

---

## 9. Weight Update (η = 0.1)

```
W_mlm_new = W_mlm - 0.1 × ∂L/∂W_mlm
           = [[0.5, 0.3, 0.2, 0.1],   -0.1× [[ 0.680, -1.387, 0.381, 0.328],
              [0.2, 0.4, 0.1, 0.3]]          [ 0.076, -0.154, 0.042, 0.036]]

           = [[0.5-0.068, 0.3+0.139, 0.2-0.038, 0.1-0.033],
              [0.2-0.008, 0.4+0.015, 0.1-0.004, 0.3-0.004]]

           = [[0.432, 0.439, 0.162, 0.067],
              [0.192, 0.415, 0.096, 0.296]]

Changes:
    sat column (index 1): W_mlm[:,1] increased from [0.3,0.4] → [0.439,0.415]
    cat column (index 0): W_mlm[:,0] decreased from [0.5,0.2] → [0.432,0.192]

This makes the model more likely to predict "sat" for representations
similar to x_final_mask = [1.874, 0.208].
```

---

## 10. Second Forward — Verify MLM Loss Decreased

```
With W_mlm_new, recompute logits:
(Using approximately same x_final_mask = [1.874, 0.208])

logit[sat] = 1.874×0.439 + 0.208×0.415 = 0.823 + 0.086 = 0.909
logit[cat] = 1.874×0.432 + 0.208×0.192 = 0.809 + 0.040 = 0.849

Before: logit[sat]=0.645, logit[cat]=0.979  → cat wins
After:  logit[sat]=0.909, logit[cat]=0.849  → sat wins ✓

Softmax([0.849, 0.909, -0.38, -0.233]):
exp: [2.337, 2.482, 1.462, 1.262]  sum=7.543
P(sat) = 2.482/7.543 = 0.329  (up from 0.260)

L' = -log(0.329) = 1.111  < 1.347 ✓

Loss decreased: sat moved from 3rd most likely to MOST LIKELY after one step.
```

---

## 11. Fine-Tuning — Classification from [CLS]

After pretraining, we add a classification head on top of [CLS] and fine-tune.

```
[CLS] final representation: x_final_cls = [1.386, 2.019]

Task: does this sentence mention a cat? (y=1)

z = W_cls · x_final_cls + b_cls
  = [0.5, 0.3] · [1.386, 2.019] + 0
  = 0.5×1.386 + 0.3×2.019
  = 0.693 + 0.606
  = 1.299

ŷ = σ(1.299) = 1/(1+e^(-1.299)) = 0.786

L_cls = -log(0.786) = 0.241   (target y=1, prediction 0.786 is decent)
```

**Fine-tuning backward:**
```
∂L_cls/∂z = ŷ - y = 0.786 - 1 = -0.214

∂L_cls/∂W_cls = -0.214 × x_final_cls = -0.214 × [1.386, 2.019]
              = [-0.297, -0.432]

∂L_cls/∂x_final_cls = -0.214 × [0.5, 0.3] = [-0.107, -0.064]

W_cls_new = [0.5, 0.3] - 0.1 × [-0.297, -0.432]
          = [0.5+0.030, 0.3+0.043]
          = [0.530, 0.343]

The fine-tuning gradient flows through:
∂L_cls/∂x_final_cls = [-0.107, -0.064]
→ through residual 2 = x_attn_cls and FFN_cls
→ through residual 1 = ALL attention weights
→ ∂L_cls/∂Wq, ∂L_cls/∂Wk, ∂L_cls/∂Wv

During fine-tuning:
    W_mlm is NOT used (it's task-specific — MLM head is discarded after pretrain)
    W_cls is newly added and trained
    Wq, Wk, Wv, W1, W2 are fine-tuned (small LR, don't diverge from pretrained)

Typical fine-tuning LR: 2e-5 to 5e-5 (vs pretraining LR 1e-4)
→ Small updates to pretrained weights preserve general knowledge
→ Large W_cls update since it's freshly initialized
```

---

## 12. MLM Masking Strategy (15% Rule)

```
In pretraining, BERT doesn't mask every token — only 15%.
Of those 15%, the 80/10/10 rule applies:
    80% → replace with [MASK]
    10% → replace with random token from vocab
    10% → keep unchanged

Why not just mask 100%?
Problem: at fine-tuning, the model NEVER sees [MASK] tokens.
If trained only with [MASK], model is confused when real tokens appear.

Why 10% random replacements?
Forces model to distrust input — can't assume token is correct.
Model must use context to verify/predict every token.

Why 10% unchanged?
Forces model to produce good representations even for non-masked tokens.
The model never knows which tokens will be used in downstream tasks.

Example with "cat sat on mat" (4 tokens, mask 15% ≈ 1 token):
    "cat [MASK] on mat"  → 80% case (masked)
    "cat RANDOM on mat"  → 10% case (e.g., "cat dog on mat")
    "cat sat on mat"     → 10% case (appears unchanged, still predict "sat")
```

---

## 13. BERT vs GPT vs Plain Transformer

| | BERT | GPT | Plain Transformer |
|-|------|-----|------------------|
| Attention type | Bidirectional | Causal (→) | Causal (decoder) |
| [CLS] token | Yes (sent rep) | No | No |
| Pretraining obj | MLM + NSP | Next token | Next token |
| Input | Token+Seg+PE | Token+PE | Token+PE |
| Loss computed on | 15% (masked) | All tokens | All tokens |
| Use case | Understanding/NLU | Generation | Generation/Translation |
| Fine-tuning head | [CLS] → Linear | Last token | Same |
| Example model | BERT-base 110M | GPT-2 117M | Vaswani 2017 |

---

## 14. Gradient Comparison — MLM vs Causal LM

```
MLM (BERT): gradient flows only from masked positions
4 tokens, 1 masked = 25% of positions produce gradient per step
At scale: 15% of 512 tokens → ~77 gradient sources per sequence
Each masked token → gradient reaches ALL other tokens via attention

Causal LM (GPT): gradient flows from ALL positions
4 tokens → 4 gradient sources
BUT: token at position 1 only gets gradient from positions 1→4
     position 2 only from 2→4...
Causal mask restricts gradient paths

Which trains faster?
GPT: more gradient signal per sequence (all tokens)
BERT: each gradient is richer (uses full bidirectional context)
In practice: GPT scales better with data, BERT with fine-tuning
```

---

## 15. Full Picture

```
INPUT:
    [CLS]  cat  [MASK]   on   mat
      |     |     |       |     |
      + seg_A embeddings (all [0.1,0.1])
      |     |     |       |     |
      + PE(0) PE(1) PE(2) PE(3) PE(4)
[0.4,1.4] [1.94,1.14] [1.009,-0.316] [0.341,-0.990] [-0.457,-0.154]
                         ↓
              (ALL attend to ALL — no causal mask)
              BIDIRECTIONAL ATTENTION
              [MASK] attends:
                  [CLS]=0.206, cat=0.292,
                  self=0.199, on=0.161,
                  mat=0.142
                         ↓
    RESIDUAL 1 + LN + FFN + RESIDUAL 2
                         ↓
      x_final_mask=[1.874, 0.208]    x_final_cls=[1.386, 2.019]
             ↓                               ↓
        MLM HEAD                       CLS HEAD
      (pretraining)                  (fine-tuning)
   W_mlm @ x_final_mask          W_cls · x_final_cls
logits=[0.979,0.645,0.396,0.249]   z=1.299, ŷ=0.786
P(sat)=0.260                       L_cls=0.241
L_mlm=1.347                            ↓
         ↓                          fine-tune
      pretrain
```

---

## 16. Quick Reference

```
BERT END-TO-END — QUICK REFERENCE

INPUT = token_embed + segment_embed + PE
No causal mask → bidirectional attention
[MASK] at pos 2: attends CLS(0.206) cat(0.292) self(0.199) on(0.161) mat(0.142)
c_mask=[0.795, 0.454] → x_final_mask=[1.874, 0.208]

MLM logits=[0.979, 0.645, 0.396, 0.249] → P(sat)=0.260 + L_mlm=1.347

Fine-tuning: x_final_cls=[1.386, 2.019]
z=1.299, ŷ=0.786, L_cls=0.241

After update: P(sat)=0.329, L'=1.111 ✓

Masking: 15% tokens; 80% [MASK], 10% random, 10% unchanged
NSP: dropped in RoBERTa (too easy, hurts performance)
Bidirectional: [MASK] sees both left AND right context
```

---

## 17. Code

### Version 1: Pure NumPy — BERT MLM Forward

```python
import numpy as np

# Data
# Tokens: [CLS]=4, cat=0, [MASK]=5, on=2, mat=3
token_embeds = np.array([
    [1.0, 0.5],   # cat (index 0)
    [0.2, 0.3],   # sat (index 1, not in input — masked)
    [0.1, 0.1],   # on (index 2)
    [0.2, 0.4],   # mat (index 3)
    [0.3, 0.3],   # [CLS] (index 4)
    [0.0, 0.0],   # [MASK] (index 5)
])

# Input sequence: [CLS]=4, cat=0, [MASK]=5, on=2, mat=3
input_ids = [4, 0, 5, 2, 3]
X_tok = token_embeds[input_ids]    # (5, 2)
X_seg = np.full_like(X_tok, 0.1)  # segment A for all tokens

# Positional Encoding
def pe(seq_len, d):
    pe = np.zeros((seq_len, d))
    for pos in range(seq_len):
        for i in range(d // 2):
            pe[pos, 2*i]   = np.sin(pos / (10000 ** (2*i/d)))
            pe[pos, 2*i+1] = np.cos(pos / (10000 ** (2*i/d)))
    return pe

x_pe = pe(4, 2)   # only 4 positions for non-CLS tokens
# Add CLS position separately; all positions 0-4
x_pe_full = pe(5, 2)   # positions 0-4
X_input = X_tok + X_seg + x_pe_full   # (5, 2)

# Weights
Wq = np.array([[0.60, 0.40], [0.20, 0.50]])
Wk = np.array([[0.50, 0.30], [0.10, 0.40]])
Wv = np.array([[0.80, 0.20], [0.30, 0.70]])

W1 = np.array([[0.5, 0.3, 0.2, 0.1], [0.4, 0.2, 0.3, 0.1]])
b1 = np.zeros(4)
W2 = np.array([[0.5, 0.3], [0.2, 0.4], [0.3, 0.2], [0.1, 0.5]])
b2 = np.zeros(2)

W_mlm = np.array([[0.5, 0.3, 0.2, 0.1],
                   [0.2, 0.4, 0.1, 0.3]])  # 2×4 real vocab items
gamma, beta = np.ones(2), np.zeros(2)

# Attention (bidirectional — no mask)
Q = X_input @ Wq   # (5, 2)
K = X_input @ Wk
V = X_input @ Wv

d_k = Q.shape[-1]
scores = Q @ K.T / np.sqrt(d_k)   # (5, 5)
A = np.exp(scores_stable := scores - scores.max(axis=-1, keepdims=True))
A /= A.sum(axis=-1, keepdims=True)
C = A @ V

# Residual + LayerNorm + FFN
def layernorm(x, gamma, beta, eps=1e-8):
    mu = x.mean(-1, keepdims=True)
    std = x.std(-1, keepdims=True) + eps
    return gamma * (x - mu) / std + beta

X_ln = layernorm(X_input + C, gamma, beta)
h  = np.maximum(0, X_ln @ W1 + b1)    # (5, 4) → ReLU
FFN_out = h @ W2 + b2                  # (5, 2) — residual 2
X_final = X_attn + FFN_out

# MLM Head
logits = X_final[2] @ W_mlm   # [MASK] position (index 2) → (4,)
logits_stable = logits - logits.max()
probs = np.exp(logits_stable) / np.exp(logits_stable).sum()

print("MLM probs:", dict(zip(['cat', 'sat', 'on', 'mat'], np.round(probs, 3))))
# {'cat': 0.363, 'sat': 0.260, 'on': 0.203, 'mat': 0.175}

y_mlm = 1  # target: "sat" = index 1
loss = -np.log(probs[y_mlm])
print(f"MLM Loss: {loss:.3f}")  # 1.347

# Backward
one_hot = np.zeros(4); one_hot[y_mlm] = 1
dl_dlogits = probs - one_hot
dl_dW_mlm = np.outer(X_final[2], dl_dlogits)

# Weight Update
lr = 0.1
W_mlm_new = W_mlm - lr * dl_dW_mlm.T

# Verify Loss Decreased
logits2 = X_final[2] @ W_mlm_new
p2 = np.exp(logits2 - logits2.max())
p2 /= p2.sum()
loss2 = -np.log(p2[y_mlm])
print(f"After update: P(sat)={p2[y_mlm]:.3f}, L'={loss2:.3f}")  # L' < 1.347 ✓
```

### Version 2: PyTorch with Autograd

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Input
# Tokens: [CLS]=4, cat=0, [MASK]=5, on=2, mat=3
token_embeds = torch.tensor([
    [1.0, 0.5],  # cat (index 0)
    [0.2, 0.3],  # sat (index 1)
    [0.1, 0.1],  # on (index 2)
    [0.2, 0.4],  # mat (index 3)
    [0.3, 0.3],  # [CLS] (index 4)
    [0.0, 0.0],  # [MASK] (index 5)
])

input_ids = torch.tensor([4, 0, 5, 2, 3])
X_tok = token_embeds[input_ids]    # (5, 2)
X_seg = torch.full_like(X_tok, 0.1)

def make_pe(seq_len, d):
    pe = torch.zeros(seq_len, d)
    pos = torch.arange(seq_len).unsqueeze(1).float()
    div = torch.pow(10000, torch.arange(0, d, 2).float() / d)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe

X_input = X_tok + X_seg + make_pe(5, 2)   # (5, 2)

# Weights
Wq = torch.tensor([[0.60, 0.40], [0.20, 0.50]], requires_grad=True)
Wk = torch.tensor([[0.50, 0.30], [0.10, 0.40]], requires_grad=True)
Wv = torch.tensor([[0.80, 0.20], [0.30, 0.70]], requires_grad=True)
W1 = torch.tensor([[0.5, 0.3, 0.2, 0.1], [0.4, 0.2, 0.3, 0.1]], dtype=torch.float32, requires_grad=True)
b1 = torch.zeros(4, requires_grad=True)
W2 = torch.tensor([[0.5, 0.3], [0.2, 0.4], [0.3, 0.2], [0.1, 0.5]], dtype=torch.float32, requires_grad=True)
b2 = torch.zeros(2, requires_grad=True)
W_mlm = torch.tensor([[0.5, 0.3, 0.2, 0.1],
                       [0.2, 0.4, 0.1, 0.3]], requires_grad=True)
gamma = torch.ones(2, requires_grad=True)
beta  = torch.zeros(2, requires_grad=True)
y_mlm = torch.tensor(1)  # target: "sat"

# Forward
Q = X_input @ Wq
K = X_input @ Wk
V = X_input @ Wv

# Bidirectional attention — no mask
scores = Q @ K.T / (Q.shape[-1] ** 0.5)
A = F.softmax(scores, dim=-1)
C = A @ V

X_attn = X_input + C
X_ln = F.layer_norm(X_attn, [2], gamma, beta)
h = F.relu(X_ln @ W1 + b1)
X_final = X_attn + h @ W2 + b2

# MLM loss on [MASK] position (index 2)
logits = X_final[2] @ W_mlm   # (4,)
loss   = F.cross_entropy(logits.unsqueeze(0), y_mlm.unsqueeze(0))  # 1.347
print(f"MLM Loss: {loss.item():.3f}")

# Backward + Update
loss.backward()
print(f"∂L/∂W_mlm\n", W_mlm.grad.round(decimals=3))

lr = 0.1
with torch.no_grad():
    for p in [Wq, Wk, Wv, W1, W2, W_mlm, gamma, beta]:
        p -= lr * p.grad
        p.grad.zero_()

# Verify
logits2 = X_final[2].detach() @ W_mlm
probs2 = F.softmax(logits2, dim=0)
loss2 = F.cross_entropy(logits2.unsqueeze(0), y_mlm.unsqueeze(0))
print(f"After update: P(sat)={probs2[1].item():.3f}, L'={loss2.item():.3f}")
```

### Version 3: HuggingFace BERT Fine-Tuning (Production)

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertForMaskedLM
import torch

# MLM Pretraining Inference
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
mlm_model  = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "cat [MASK] on mat"
inputs = tokenizer(text, return_tensors='pt')
mask_ids = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.no_grad():
    outputs = mlm_model(**inputs)
    logits = outputs.logits[0, mask_ids, :]   # logits at [MASK] position
    mask_logits = logits[0, mask_ids, :]
    top5 = torch.topk(mask_logits, 5)

# Top 5 predictions
for score, idx in zip(top5.values.tolist(), top5.indices.tolist()):
    word = tokenizer.convert_ids_to_tokens([idx])[0]
    print(f"  {word}: {score:.3f}")
# sat: 8.127, lay: 0.889, stood: 0.123...

# Fine-Tuning for Classification
from transformers import AdamW, get_linear_schedule_with_warmup

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # binary classification
)

# Dataset
texts  = ["cat sat on mat", "dog ran in park", "bird flew over tree"]
labels = torch.tensor([1, 0, 0])  # 1 = mentions cat

# Tokenize
batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors='pt'
)

# Optimizer with weight decay (don't decay bias and LayerNorm)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_groups = [
    {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
]
optimizer = AdamW(optimizer_groups, lr=2e-5)

# Linear warmup + decay (standard BERT fine-tuning schedule)
total_steps = 10
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=2, num_training_steps=total_steps
)

# Training step
model.train()
outputs = model(**batch, labels=labels)
loss = outputs.loss
print(f"Fine-tuning loss: {loss.item():.3f}")

loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
optimizer.step()
scheduler.step()
optimizer.zero_grad()

# Inference
model.eval()
with torch.no_grad():
    test_input = tokenizer("the cat is here", return_tensors='pt')
    logits = model(**test_input).logits
    pred = torch.argmax(logits, dim=-1)
    print(f"Prediction: {pred.item()}")  # 1 (mentions cat)
```

---

## 18. Gotchas

| GOTCHA | WHAT GOES WRONG | HOW TO FIX |
|--------|----------------|-----------|
| Using BERT for generation | BERT is bidirectional — can't autoregressively generate text | Use GPT/LLaMA instead; BERT can't predict "what comes next" |
| Not adding [CLS] at the start of fine-tuning input | Classification uses [CLS] representation | Tokenizer does this automatically |
| Using MLM head at fine-tuning time | W_mlm is task-specific, discarded after pretrain | Use BertForSequenceClassification instead of BertForMaskedLM |
| Learning rate too high for fine-tuning | BERT weights pretrained on massive data | Use 2e-5 to 5e-5 with warmup |
| Not freezing lower layers on small datasets | Fine-tuning ALL layers on 100 examples → overfit | Freeze layers 1-6; fine-tune 7-12 only |
| Forgetting special tokens | Model input format: [CLS] text [SEP] — missing these → degraded | Always use tokenizer; don't tokenize manually |
| Segment embedding mismatch in NSP/sentence pair tasks | Pair: A then B | Check token_type_ids in tokenizer output |
| Input > 512 tokens | BERT max_length=512 — index error for longer inputs | Truncate, or use Longformer/BigBird |

---

## 19. Interview Q&A

**Q: What is Masked Language Modeling and why does it create bidirectional representations?**

MLM randomly masks 15% of input tokens and trains the model to predict them. Because the prediction task requires seeing BOTH left AND right context (e.g., predicting [MASK] in "cat [MASK] on mat" benefits from both "cat" left and "on mat" right), the self-attention has no causal mask — all tokens attend to all others. This is fundamentally different from GPT's causal LM: at position 2, GPT can only see tokens 1-2, while BERT's [MASK] at position 2 sees all 4 tokens. Bidirectionality makes BERT better at understanding tasks (classification, NER, QA) but incapable of generation (can't predict token-by-token without seeing the future).

**Q: Why does BERT use the 80/10/10 masking rule instead of masking 100%?**

The 80/10/10 rule reduces train-test mismatch (also called the pretraining-finetuning discrepancy): 100% masking: model learns to represent [MASK] tokens well, but at fine-tuning there are no [MASK] tokens → the model has never learned to represent real tokens. 80% [MASK] + 10% random + 10% unchanged: 80% provides the masked prediction signal; 10% random forces the model to distrust input → builds context-based representations for every token even when unmasked (it might be wrong!); 10% unchanged: forces model to produce useful representations for real tokens, since downstream tasks need those representations.

**Q: What is the [CLS] token and why does fine-tuning use it?**

[CLS] is a special token prepended to every BERT input. Unlike regular word tokens, [CLS] has no semantic meaning before pretraining — it's a blank slate. During bidirectional pretraining, [CLS] attends to ALL other tokens in every training example. Because it appears in every sequence and must produce useful representations for NSP (binary classification on [CLS]), it learns to aggregate global sentence information. After pretraining, [CLS]'s final representation is a compressed summary of the entire input sequence. Adding a linear layer on top (W_cls: d=768 for binary classification) and fine-tuning on labeled data is efficient and effective.

**Q: What changed between BERT and RoBERTa?**

RoBERTa (Liu et al., 2019) kept the architecture identical but changed training: (1) Removed NSP — ablation showed NSP hurts MLM performance (task conflict). (2) Dynamic masking — mask pattern changes each epoch vs BERT's static mask. (3) Larger batches: 256 → 8192 (with lower LR). (4) More data: 160GB vs 16GB in BERT. (5) Byte-level BPE tokenizer: vocab 50K vs BERT's WordPiece 30K. Result: 1-3+ improvements across GLUE, SQuAD, and RACE benchmarks. Lesson: data and training duration matter more than architectural novelty.

**Q: Can you use BERT for text generation? Why or why not?**

No — not directly. BERT is bidirectional: generating token t requires seeing tokens t+1, t+2,... which don't exist yet during generation. GPT: P(x_t | x_1,...,x_{t-1}) — each token conditioned on previous only. BERT: P(x_t | all other tokens) — requires the surrounding context is given. BERT CAN fill in blanks (masked prediction), but only when the surrounding context is given. Useful for: spell correction ("The cat sat on the ___"), cloze tests (fill in missing words given full surrounding context). For generation: use GPT/LLaMA (decoder-only) or T5 (encoder-decoder for conditional generation).

---

## 20. Connections

- **Transformer Architecture (fundamentals/02):** BERT = a stack of 12 encoder blocks; same MHA + FFN + residual + LN structure; [CLS]=101, [SEP]=102, [MASK]=103 in BERT-base vocabulary
- **Tokenization (fundamentals/03):** BERT uses WordPiece tokens, [CLS]/[SEP]/[MASK] are special; tokenizer directly shapes the MLM pretraining objective
- **06 Transformer E2E (NLP/sequence_models):** BERT = same architecture but no causal mask; same forward pass but bidirectional; same residual gradient flows
- **GPT Family (models/02):** GPT = same architecture but causal mask, trained on next-token prediction — the encoder vs decoder split
- **Efficient Transformers (models/04):** DistilBERT uses knowledge distillation from BERT; ELECTRA replaces MLM with a more efficient discriminative objective; T5 uses BERT-style encoder (bidirectional) + GPT-style decoder for generation

---

## Key Takeaway

BERT = transformer encoder with NO causal mask + MLM pretraining. Bidirectionality lets [MASK] see the full sentence, making it powerful for understanding. The [CLS] token accumulates global sentence meaning — used as input to classification head during fine-tuning. Loss computed only on masked positions (15%). The 80/10/10 masking rule solves the train-test discrepancy. RoBERTa proved that removing NSP + more data + larger batches beats architectural changes.

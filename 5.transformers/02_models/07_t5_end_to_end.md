# T5 End-to-End — Encoder-Decoder with Cross-Attention

Every number computed. Corpus: "cat sat on mat".

---

## What is T5

T5 (Text-to-Text Transfer Transformer) reframes every NLP task as seq2seq:

```
Translation:   "translate English to French: cat sat on mat"  →  "le chat s'est assis"
Summarization: "summarize: ..."                               →  shorter text
Classification: "sentiment: cat sat on mat"                  →  "positive"
Span fill:     "cat <X> mat"                                 →  "<X> sat on </s>"
```

Same model, same loss (cross-entropy on decoder output) for every task.

**Contrast with BERT and GPT:**

| Model | Architecture | Pretraining | Use case |
|---|---|---|---|
| BERT | Encoder only | MLM, bidirectional | Classification, NER, QA |
| GPT | Decoder only | CLM, causal | Generation, completion |
| T5 | Encoder + Decoder | Span corruption | Translation, summarization, any seq2seq |

---

## Part 1 — Span Corruption (T5 Pretraining Objective)

### The task

```
Original sentence:    "cat sat on mat"

Step 1 — Sample spans to corrupt:
  Span selected:      "sat on"  (positions 1-2)

Step 2 — Replace each span with sentinel token:
  Corrupted input:    "cat <X> mat"     ← encoder input
  Target:             "<X> sat on </s>" ← decoder target

Step 3 — Train decoder to reproduce corrupted spans
```

Multiple spans in longer text:
```
Original: "the cat sat on the mat near the door"
Corrupt:  "the cat <X> the mat <Y> the door"
Target:   "<X> sat on <Y> near </s>"

One sentinel per masked span.
```

### Why better than BERT's MLM

| | BERT MLM | T5 Span Corruption |
|---|---|---|
| Mask rate | 15% of tokens | ~15% of tokens |
| What's masked | Individual tokens | Contiguous spans (avg 3 tokens) |
| What's predicted | Each masked token independently | Full span as a sequence |
| Training signal | 15% of input positions | Decoder trains on full output sequence |
| Forces | Predict single token given context | Predict sequence given corrupted input |

Span corruption forces the model to learn to generate coherent text segments, not just fill in single blanks.

---

## Part 2 — Architecture Overview

```
ENCODER                          DECODER
─────────────────────────────    ─────────────────────────────────────────
"cat  <X>  mat"                  "<s>  <X>  sat  on"  (teacher-forced input)
  │    │    │                       │    │    │    │
[embed]   [embed]                [embed]   [embed]
  │    │    │                       │    │    │    │
[Encoder Self-Attention]         [Decoder Self-Attention]
 (bidirectional — no mask)        (CAUSAL — lower triangle mask)
  │    │    │                       │    │    │    │
[FFN]                            [Cross-Attention] ←── K, V from Encoder output
  │    │    │                     Q from Decoder
[repeat × N_layers]                │    │    │    │
  │    │    │                    [FFN]
H_enc[0] H_enc[1] H_enc[2]        │    │    │    │
(encoded  (encoded  (encoded     [repeat × N_layers]
  cat)      <X>)      mat)         │    │    │    │
     │         │         │       [logits over vocab]
     └─────────┴─────────┘         │    │    │    │
       K, V for cross-attn       <X>   sat   on  </s>  ← targets
```

### Three types of attention in T5

| Type | Q source | K source | V source | Mask | Purpose |
|---|---|---|---|---|---|
| Encoder self-attn | encoder | encoder | encoder | None (bidirectional) | Encoder understands input |
| Decoder self-attn | decoder | decoder | decoder | Causal (lower-tri) | Decoder reads own history |
| **Cross-attention** | **decoder** | **encoder** | **encoder** | **None** | **Decoder reads encoder context** |

Cross-attention is the new concept. Everything else is identical to BERT (encoder) or GPT (decoder self-attention).

---

## Part 3 — Encoder Forward Pass

### Input

```
Encoder input: "cat <X> mat"

Token embeddings (d=2):
  cat  = [1.0, 0.5]   (position 0)
  <X>  = [0.3, 0.8]   (position 1)
  mat  = [0.8, 0.4]   (position 2)

X_enc = [[1.0, 0.5],
         [0.3, 0.8],
         [0.8, 0.4]]
```

### Encoder Self-Attention (bidirectional — all tokens see all tokens)

Using W_Q = W_K = W_V = I (identity):

**Attention scores = X_enc × X_enc^T / sqrt(d_k):**

```
Raw scores:
         cat    <X>    mat
cat  [ 1.25,  0.70,  1.00 ]   ← 1.0×1.0 + 0.5×0.5, 1.0×0.3 + 0.5×0.8, 1.0×0.8 + 0.5×0.4
<X>  [ 0.70,  0.73,  0.56 ]
mat  [ 1.00,  0.56,  0.80 ]

Divide by sqrt(2) = 1.414:
         cat    <X>    mat
cat  [ 0.884, 0.495, 0.707 ]
<X>  [ 0.495, 0.516, 0.396 ]
mat  [ 0.707, 0.396, 0.566 ]

NO MASK in encoder — every token attends to all others. This is bidirectional.
```

**Softmax per row:**

```
cat row [0.884, 0.495, 0.707]:
  e^0.884=2.421, e^0.495=1.641, e^0.707=2.028   sum=6.090
  α_cat = [0.398, 0.269, 0.333]

<X> row [0.495, 0.516, 0.396]:
  e^0.495=1.641, e^0.516=1.675, e^0.396=1.486   sum=4.802
  α_X   = [0.342, 0.349, 0.309]  ← <X> attends roughly equally (it's a blank)

mat row [0.707, 0.396, 0.566]:
  e^0.707=2.028, e^0.396=1.486, e^0.566=1.761   sum=5.275
  α_mat = [0.384, 0.282, 0.334]
```

**Encoder output H_enc = α × V (V = X_enc):**

```
h_cat = 0.398×[1.0,0.5] + 0.269×[0.3,0.8] + 0.333×[0.8,0.4]
      = [0.398, 0.199] + [0.081, 0.215] + [0.266, 0.133]
      = [0.745, 0.547]

h_X   = 0.342×[1.0,0.5] + 0.349×[0.3,0.8] + 0.309×[0.8,0.4]
      = [0.342, 0.171] + [0.105, 0.279] + [0.247, 0.124]
      = [0.694, 0.574]

h_mat = 0.384×[1.0,0.5] + 0.282×[0.3,0.8] + 0.334×[0.8,0.4]
      = [0.384, 0.192] + [0.085, 0.226] + [0.267, 0.134]
      = [0.736, 0.552]
```

**Encoder output (these become K and V for cross-attention):**

```
H_enc = [[0.745, 0.547],   ← position 0: "cat"  (enriched with some <X> and mat context)
         [0.694, 0.574],   ← position 1: "<X>"  (enriched with cat and mat — it's a blank)
         [0.736, 0.552]]   ← position 2: "mat"  (enriched with some cat and <X> context)
```

Note: h_X captures information from both "cat" and "mat" — the encoder has processed what context surrounds the missing span. This information is what the decoder will attend to via cross-attention.

After N encoder layers (each: self-attention + FFN + residual + norm), H_enc is a rich contextual representation of the entire input.

---

## Part 4 — Decoder Forward Pass (predicting "sat")

### Setup

```
Full decoder sequence (teacher forcing):
  Input:   [<s>,   <X>,  sat,  on ]   ← shifted right
  Target:  [<X>,   sat,  on,   </s>]  ← what we predict at each step

We trace step 1 (decoder at position 1, predicting "sat" from context [<s>, <X>]).

Token embeddings:
  <s>  = [0.1, 0.1]   (start token)
  <X>  = [0.3, 0.8]   (sentinel — decoder has seen it, knows it needs to generate a span)
```

### Step 1 — Decoder Self-Attention (causal)

At position 1, decoder sees [<s>, <X>]:

```
X_dec = [[0.1, 0.1],   ← <s>   (position 0)
         [0.3, 0.8]]   ← <X>   (position 1)

Q = K = V = X_dec (W=I)

Raw scores (2×2):
         <s>    <X>
<s>  [ 0.02,  0.11 ]   ← 0.1×0.3 + 0.1×0.8
<X>  [ 0.11,  0.73 ]

Scaled by sqrt(2) = 1.414:
         <s>    <X>
<s>  [ 0.014,  0.078 ]
<X>  [ 0.078,  0.516 ]

Apply CAUSAL MASK (upper triangle → -∞):
         <s>    <X>
<s>  [ 0.014,   -∞  ]   ← <s> cannot look at <X> (future)
<X>  [ 0.078,  0.516]   ← <X> can see <s> and itself ✓
```

**Softmax for <X> row [0.078, 0.516]:**
```
e^0.078 = 1.081, e^0.516 = 1.675   sum = 2.756
α_X_dec = [0.392, 0.608]   ← <X> attends to itself more than to <s>
```

**Decoder self-attention output for position 1:**
```
h_dec_self = 0.392 × [0.1, 0.1] + 0.608 × [0.3, 0.8]
           = [0.039, 0.039] + [0.182, 0.486]
           = [0.221, 0.525]
```

This h_dec_self = [0.221, 0.525] is the query for cross-attention. It represents "I have seen <s> and <X> — now I need to look at the encoder to know what sat/on etc. to generate."

---

## Part 5 — Cross-Attention (The New Concept)

### What makes cross-attention different

```
Self-attention:   Q, K, V all come from the SAME sequence
                  (encoder reads itself; decoder reads itself)

Cross-attention:  Q  comes from the DECODER (where are we in the generation?)
                  K  comes from the ENCODER (what was the input?)
                  V  comes from the ENCODER (what do we read from the input?)
```

The decoder asks: "given my current state, which encoder positions are relevant?"

### Step-by-step computation

**Query (from decoder self-attention output):**
```
Q_cross = h_dec_self = [0.221, 0.525]   ← "I've seen <s> <X>, what encoder position helps?"
```

**Keys (from encoder output H_enc):**
```
K_enc = H_enc = [[0.745, 0.547],   ← K for "cat"
                 [0.694, 0.574],   ← K for "<X>"
                 [0.736, 0.552]]   ← K for "mat"
```

**Attention scores: Q_cross · K_enc^T**

```
score[cat] = Q · K_cat = 0.221×0.745 + 0.525×0.547
           = 0.165     + 0.287
           = 0.452

score[<X>] = Q · K_X   = 0.221×0.694 + 0.525×0.574
           = 0.153     + 0.301
           = 0.454   ← highest (slightly)

score[mat] = Q · K_mat = 0.221×0.736 + 0.525×0.552
           = 0.163     + 0.290
           = 0.453
```

Scale by sqrt(d_k) = sqrt(2) = 1.414:
```
[0.452/1.414, 0.454/1.414, 0.453/1.414] = [0.320, 0.321, 0.320]
```

**Softmax:**
```
e^0.320 = 1.377, e^0.321 = 1.379, e^0.320 = 1.377   sum = 4.133
α_cross = [0.333, 0.334, 0.333]   ← nearly uniform
```

**Note on uniformity:** In this toy example, all encoder positions look similar because the embeddings are close and W=I. In a real T5:
- W_K and W_Q are learned and differentiate encoder positions
- The <X> sentinel gets a very different representation from content words
- Cross-attention learns to focus on the most relevant encoder positions

Let me use projected values to show a meaningful attention pattern:

**With learned projections (pedagogically motivated):**

```
After N encoder layers and training, the encoder learns to make <X> have
a distinctive representation (it's a sentinel — "fill in this blank").
Let's say post-projection:

K_cat (projected) = [0.6, 0.2]
K_X   (projected) = [0.3, 0.9]   ← <X> has high second dimension
K_mat (projected) = [0.5, 0.1]

Q_cross (projected) = [0.4, 0.8]  ← decoder query, aligned with <X>

Scores:
  score[cat] = 0.4×0.6 + 0.8×0.2 = 0.24 + 0.16 = 0.40
  score[<X>] = 0.4×0.3 + 0.8×0.9 = 0.12 + 0.72 = 0.84   ← HIGHEST
  score[mat] = 0.4×0.5 + 0.8×0.1 = 0.20 + 0.08 = 0.28

Scaled by sqrt(2) = 1.414:
  [0.283, 0.594, 0.198]

Softmax:
  e^0.283=1.327, e^0.594=1.811, e^0.198=1.219   sum=4.357
  α_cross = [0.305, 0.416, 0.280]

Interpretation: decoder attends to <X> most (0.416),
               then cat (0.305), then mat (0.280).
The <X> position in the encoder holds information about the
surrounding context ("cat ___ mat") — decoder correctly reads it.
```

**Cross-attention output (V = H_enc):**
```
V_enc = H_enc = [[0.745, 0.547],   ← V for "cat"
                 [0.694, 0.574],   ← V for "<X>"
                 [0.736, 0.552]]   ← V for "mat"

h_cross = 0.305×[0.745, 0.547] + 0.416×[0.694, 0.574] + 0.280×[0.736, 0.552]
        = [0.227, 0.167] + [0.289, 0.239] + [0.206, 0.155]
        = [0.722, 0.561]
```

This h_cross = [0.722, 0.561] is a weighted blend of encoder positions, dominated by the <X> encoder state. It represents "the context of what needs to be filled in" — which the decoder will use to generate "sat".

---

## Part 6 — Self-Attention vs Cross-Attention: Side-by-Side

```python
# Self-attention (encoder or decoder causal)
def self_attention(X, mask=None):
    Q = X @ W_Q          # Q from same sequence
    K = X @ W_K          # K from same sequence
    V = X @ W_V          # V from same sequence
    scores = Q @ K.T / sqrt(d_k)
    if mask is not None:
        scores[mask] = -inf
    alpha = softmax(scores, dim=-1)
    return alpha @ V

# Cross-attention (decoder reads encoder)
def cross_attention(X_dec, H_enc):
    Q = X_dec @ W_Q      # Q from DECODER
    K = H_enc @ W_K      # K from ENCODER
    V = H_enc @ W_V      # V from ENCODER
    scores = Q @ K.T / sqrt(d_k)
    # No mask — decoder can see all encoder positions
    alpha = softmax(scores, dim=-1)
    return alpha @ V
```

**Mathematical difference:**

| | Q | K | V | Score Q·K^T |
|---|---|---|---|---|
| Encoder self-attn | x_i×W_Q | x_j×W_K | x_j×W_V | how much encoder token i should read from token j |
| Decoder self-attn | y_i×W_Q | y_j×W_K | y_j×W_V | how much decoder position i reads from position j (j≤i) |
| Cross-attention | y_i×W_Q | h_j×W_K | h_j×W_V | how much decoder position i reads from encoder position j |

Where x = encoder input, y = decoder input, h = encoder hidden state.

**Shapes:**

```
Self-attention scores: (T_x, T_x) — source length × source length
Cross-attention scores: (T_dec, T_enc) — decoder length × encoder length

For "cat <X> mat" → "<s> <X> sat on":
  Encoder self-attn: (3, 3)
  Decoder self-attn: (4, 4) with causal mask
  Cross-attention:   (4, 3) — 4 decoder positions × 3 encoder positions
```

The cross-attention matrix (4×3) answers: "which encoder positions does each decoder position attend to?"

```
                cat   <X>   mat
<s>  predicts <X>:  [0.25, 0.50, 0.25]  ← heavily attends to <X>
<X>  predicts sat:  [0.30, 0.42, 0.28]  ← mostly <X>, some context
sat  predicts on:   [0.20, 0.45, 0.35]  ← still mostly <X>
on   predicts </s>: [0.28, 0.40, 0.32]  ← still attends to <X>
```

All decoder positions attend most to the <X> encoder position because that's where the "fill in the blank" signal lives.

---

## Part 7 — Prediction and Loss

### From cross-attention output to prediction

```
h_cross = [0.722, 0.561]   (cross-attention output at position 1)
        + h_dec_self (residual connection)

→ FFN (d→4d→d)
→ Projection W_lm ∈ ℝ^{d × vocab_size}   ← maps to vocabulary

Vocab: {cat=0, sat=1, on=2, mat=3, <X>=4, </s>=5}  (size=6)
```

### Loss at each decoder step

**Step 0: predict <X> (from input <s>)**
```
Logits = [0.2, 0.4, 0.3, 0.1, 2.0, 0.1]
         cat  sat   on  mat  <X>  </s>

e^0.2=1.221, e^0.4=1.492, e^0.3=1.350, e^0.1=1.105, e^2.0=7.389, e^0.1=1.105
sum = 13.662
P(<X>) = 7.389 / 13.662 = 0.541
L_0 = -log(0.541) = 0.614
```

**Step 1: predict sat (from <s>, <X>)**
```
Logits = [0.3, 2.1, 0.8, 0.2, 0.5, 0.1]
         cat  sat   on  mat  <X>  </s>

e^0.3=1.350, e^2.1=8.166, e^0.8=2.226, e^0.2=1.221, e^0.5=1.649, e^0.1=1.105
sum = 15.717
P(sat) = 8.166 / 15.717 = 0.520
L_1 = -log(0.520) = 0.654
```

**Step 2: predict on (from <s>, <X>, sat)**
```
Logits = [0.2, 0.6, 1.8, 0.1, 0.4, 0.3]
         cat  sat   on  mat  <X>  </s>

e^0.2=1.221, e^0.6=1.822, e^1.8=6.050, e^0.1=1.105, e^0.4=1.492, e^0.3=1.350
sum = 13.040
P(on) = 6.050 / 13.040 = 0.464
L_2 = -log(0.464) = 0.768
```

**Step 3: predict </s> (from <s>, <X>, sat, on)**
```
Logits = [0.1, 0.3, 0.4, 0.2, 0.1, 1.9]
         cat  sat   on  mat  <X>  </s>

e^0.1=1.105, e^0.3=1.350, e^0.4=1.492, e^0.2=1.221, e^0.1=1.105, e^1.9=6.686
sum = 12.959
P(</s>) = 6.686 / 12.959 = 0.516
L_3 = -log(0.516) = 0.663
```

**Total loss (average over decoder steps):**
```
L = (L_0 + L_1 + L_2 + L_3) / 4
  = (0.614 + 0.654 + 0.768 + 0.663) / 4
  = 2.699 / 4
  = 0.675
```

**Backpropagation updates:**
- W_lm (output projection): learns to map h_cross to correct vocab logits
- W_Q_cross, W_K_cross, W_V_cross: learns to focus on relevant encoder positions
- W_K_enc, W_V_enc in encoder: learns representations useful for cross-attention
- Entire encoder: gets gradients through cross-attention K, V

---

## Part 8 — Inference (Autoregressive)

At inference, no teacher forcing — decoder generates one token at a time:

```
Encode once:
  "cat <X> mat" → H_enc = [[0.745,0.547], [0.694,0.574], [0.736,0.552]]
  H_enc is fixed. Cross-attention K, V cache computed from H_enc.

Decode step 0:
  Input: [<s>]
  Cross-attend to H_enc
  Predict: <X> (argmax or sample)
  
Decode step 1:
  Input: [<s>, <X>]
  Cross-attend to H_enc (same cached K,V)
  Predict: sat

Decode step 2:
  Input: [<s>, <X>, sat]
  Cross-attend to H_enc
  Predict: on

Decode step 3:
  Input: [<s>, <X>, sat, on]
  Cross-attend to H_enc
  Predict: </s>  ← stop

Output: "<X> sat on </s>"
Extracted span: "sat on"
Final answer: "cat sat on mat"  ← restored original
```

**Why encode once and reuse:** The encoder output H_enc doesn't change during decoding (encoder is not autoregressive). This is the key efficiency advantage of encoder-decoder over decoder-only for tasks where the full input is known upfront.

---

## Part 9 — T5 Specifics vs BERT/GPT

### Position embeddings

T5 uses **relative position bias** — different from everything else:

```
GPT-2:  learned absolute PE — add PE_0, PE_1, ..., PE_T to token embeddings
BERT:   learned absolute PE — same as GPT-2
LLaMA: RoPE — rotate Q, K by angle proportional to position
T5:     relative position bias — learned scalar added to attention logits based on (i-j) bucket

T5 attention score:
  score[i,j] = Q_i · K_j / sqrt(d_k) + b(i-j)
                                         ↑
                              learned bias from relative position bucket
                              Same b for same relative distance, regardless of absolute pos.
```

Relative position bias:
- b(0) for attending to self
- b(1) for attending 1 position back
- b(-1) for attending 1 position forward (encoder only)
- Positions beyond 128 share the same bucket (log-spaced buckets)

**Advantage:** like RoPE, generalizes to longer sequences than seen during training.

### Architecture variants

T5-Base:
```
Encoder: 12 layers, d=768, 12 heads
Decoder: 12 layers, d=768, 12 heads
FFN: 2048
Params: 250M
```

T5-Large:
```
Encoder/Decoder: 24 layers, d=1024, 16 heads
Params: 780M
```

T5-11B:
```
Encoder/Decoder: 24 layers, d=1024, 128 heads
Params: 11B
```

---

## Part 10 — BERT vs GPT vs T5 Comparison

| | BERT | GPT | T5 |
|---|---|---|---|
| Architecture | Encoder only | Decoder only | Encoder + Decoder |
| Attention | Bidirectional | Causal (masked) | Encoder: bidirectional, Decoder: causal + cross |
| Pretraining | MLM + NSP | CLM (next token) | Span corruption |
| Input/Output | 1 sequence → labels | sequence → next token | sequence → sequence |
| Handles | Classification, token-level | Open-ended generation | Any seq2seq task |
| Context | Full input visible | Left-to-right only | Encoder sees full input |
| Weakness | Can't generate | Can't use full context | More complex, more parameters |

**When to use what:**

```
BERT → when you have labeled data and need to classify/extract
        ("Is this email spam?", "What entities are in this text?")

GPT  → when you need to generate or complete text
        ("Continue this story...", "Answer this question:")

T5   → when input→output are both sequences and different length
        ("Translate...", "Summarize...", "Answer based on this passage:")
```

### The cross-attention bottleneck

In encoder-decoder models, the decoder can ONLY access encoder information through cross-attention. If the encoder fails to capture relevant information, the decoder cannot compensate.

This is why for long documents:
- Encoder-decoder: may lose info if encoder compresses too aggressively
- Decoder-only (GPT with full context): keeps all tokens in attention directly

Modern trend: large decoder-only models (GPT-4, LLaMA-3) with long contexts often outperform encoder-decoder for translation/summarization — no information bottleneck, same architecture for everything.

---

## Code

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (seq_q, d_k)
    K: (seq_k, d_k)
    V: (seq_k, d_v)
    mask: (seq_q, seq_k) — True where we should mask (set to -inf)
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)        # (seq_q, seq_k)
    if mask is not None:
        scores[mask] = -1e9
    alpha = softmax(scores, axis=-1)        # (seq_q, seq_k)
    return alpha @ V, alpha                 # (seq_q, d_v), weights

# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────

d = 2
vocab = {'cat': 0, '<X>': 1, 'mat': 2, '<s>': 3, 'sat': 4, 'on': 5, '</s>': 6}
embeddings = np.array([
    [1.0, 0.5],   # cat
    [0.3, 0.8],   # <X>
    [0.8, 0.4],   # mat
    [0.1, 0.1],   # <s>
    [0.4, 0.9],   # sat
    [0.2, 0.6],   # on
    [0.0, 0.0],   # </s>
])

def embed(tokens):
    return np.array([embeddings[vocab[t]] for t in tokens])

# ─────────────────────────────────────────────
# ENCODER — Bidirectional Self-Attention
# ─────────────────────────────────────────────

encoder_input = ['cat', '<X>', 'mat']
X_enc = embed(encoder_input)   # (3, 2)

# W = I for simplicity (in real T5: learned W_Q, W_K, W_V)
W_Q = W_K = W_V = np.eye(d)

Q_enc = X_enc @ W_Q   # (3, 2)
K_enc = X_enc @ W_K   # (3, 2)
V_enc = X_enc @ W_V   # (3, 2)

# No mask in encoder
H_enc, attn_enc = scaled_dot_product_attention(Q_enc, K_enc, V_enc, mask=None)
print("Encoder attention weights:")
print(np.round(attn_enc, 3))
print("Encoder output H_enc:")
print(np.round(H_enc, 3))

# ─────────────────────────────────────────────
# DECODER — Step 1: predict "sat" from [<s>, <X>]
# ─────────────────────────────────────────────

decoder_input = ['<s>', '<X>']   # teacher-forced input so far
X_dec = embed(decoder_input)     # (2, 2)

# Step 1a: Causal self-attention
seq_len = len(decoder_input)
causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)  # upper triangle
print("\nCausal mask:")
print(causal_mask)

Q_dec = X_dec @ W_Q
K_dec = X_dec @ W_K
V_dec = X_dec @ W_V

H_dec_self, attn_dec_self = scaled_dot_product_attention(Q_dec, K_dec, V_dec, mask=causal_mask)
print("\nDecoder self-attention output:")
print(np.round(H_dec_self, 3))

# ─────────────────────────────────────────────
# CROSS-ATTENTION — Decoder reads Encoder
# ─────────────────────────────────────────────

# Q from decoder, K and V from encoder
Q_cross = H_dec_self @ W_Q    # (2, 2) — decoder positions as queries
K_cross = H_enc @ W_K         # (3, 2) — encoder positions as keys
V_cross = H_enc @ W_V         # (3, 2) — encoder positions as values

H_cross, attn_cross = scaled_dot_product_attention(Q_cross, K_cross, V_cross, mask=None)
print("\nCross-attention weights (decoder_pos × encoder_pos):")
print("Columns: cat, <X>, mat")
for i, dec_tok in enumerate(decoder_input):
    print(f"  {dec_tok:6s} → {np.round(attn_cross[i], 3)}")

print("\nCross-attention output:")
print(np.round(H_cross, 3))

# ─────────────────────────────────────────────
# OUTPUT — Logits and Loss
# ─────────────────────────────────────────────

vocab_size = len(vocab)
np.random.seed(0)
W_lm = np.random.randn(d, vocab_size) * 0.5   # output projection

# Logits for each decoder position
logits = H_cross @ W_lm   # (2, vocab_size)

# At position 1 (predicting "sat"), compute loss
target_token = 'sat'
target_idx = vocab[target_token]

logit_step1 = logits[1]
probs_step1 = softmax(logit_step1)
loss_step1 = -np.log(probs_step1[target_idx] + 1e-9)

print(f"\nPredicting '{target_token}':")
print(f"  Logits: {np.round(logit_step1, 3)}")
print(f"  Probs:  {np.round(probs_step1, 3)}")
print(f"  P(sat)= {probs_step1[target_idx]:.3f}")
print(f"  Loss  = {loss_step1:.3f}")

# ─────────────────────────────────────────────
# FULL T5-STYLE FORWARD PASS (modular)
# ─────────────────────────────────────────────

class T5Block:
    """One encoder or decoder block (simplified, no FFN)."""

    def __init__(self, d):
        self.W_Q = np.eye(d)
        self.W_K = np.eye(d)
        self.W_V = np.eye(d)

    def encode(self, X):
        """Bidirectional self-attention (no mask)."""
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V
        out, _ = scaled_dot_product_attention(Q, K, V)
        return X + out   # residual

    def decode(self, Y, H_enc):
        """Causal self-attention + cross-attention."""
        seq = Y.shape[0]
        mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)

        # Self-attention (causal)
        Q = Y @ self.W_Q
        K = Y @ self.W_K
        V = Y @ self.W_V
        self_out, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
        Y = Y + self_out   # residual

        # Cross-attention
        Q_c = Y @ self.W_Q
        K_c = H_enc @ self.W_K
        V_c = H_enc @ self.W_V
        cross_out, cross_weights = scaled_dot_product_attention(Q_c, K_c, V_c)
        Y = Y + cross_out   # residual

        return Y, cross_weights

block = T5Block(d)

# Encode
H = X_enc.copy()
for _ in range(1):   # N encoder layers
    H = block.encode(H)
print("\nEncoder output (after 1 layer):")
print(np.round(H, 3))

# Decode
Y = embed(['<s>', '<X>'])
for _ in range(1):   # N decoder layers
    Y, cw = block.decode(Y, H)
print("\nDecoder output (after 1 layer):")
print(np.round(Y, 3))
print("Cross-attention weights:")
print(np.round(cw, 3))
```

---

## Interview Q&A

**Q: What is the key architectural difference between T5 and GPT?**
> T5 has both an encoder AND a decoder connected by cross-attention.  
> GPT is decoder-only — each token attends to previous tokens only (causal).  
> In T5, the encoder processes the full input with bidirectional attention first, then the decoder generates output while attending to encoder states via cross-attention.  
> For translation/summarization, T5 can see the entire source before generating — GPT cannot.

**Q: Explain cross-attention. Where do Q, K, V come from?**
> Cross-attention sits between self-attention and FFN in each decoder layer.  
> Q = decoder's current state (what we're generating so far)  
> K = encoder output (the encoded input)  
> V = encoder output (what we read from the input)  
> The decoder uses its Q to ask "which encoder positions are relevant right now?" — the answer (attention weights × V) is a summary of encoder context relevant to the current generation step.

**Q: What's the difference between encoder self-attention and decoder self-attention?**
> Encoder self-attention: every token can attend to every other token (bidirectional). No mask.  
> Decoder self-attention: each token can only attend to itself and previous tokens (causal). Upper-triangular mask sets future positions to -∞ before softmax.  
> Encoder sees the whole input → can build richer representations.  
> Decoder is autoregressive → can't peek at future tokens during generation.

**Q: How does cross-attention differ from self-attention mathematically?**
> Self-attention: Q = X@W_Q, K = X@W_K, V = X@W_V — all from same X.  
> Cross-attention: Q = Y@W_Q from decoder, K = H@W_K, V = H@W_V from encoder.  
> Score matrix shape: self-attn is (T,T); cross-attn is (T_dec, T_enc).  
> The attention weights in cross-attention answer: "which encoder position does each decoder step look at?"

**Q: What is span corruption in T5?**
> Instead of masking 15% of individual tokens (BERT's MLM), T5 selects contiguous spans and replaces each span with a sentinel token like <X>.  
> Input: "cat <X> mat" (original: "cat sat on mat")  
> Target: "<X> sat on </s>"  
> The decoder must reproduce all masked spans as a sequence. This forces the model to generate coherent multi-token continuations, which is more similar to the seq2seq tasks T5 is fine-tuned on.

**Q: Why use an encoder-decoder instead of a decoder-only model for translation?**
> Encoder sees the full source sentence with bidirectional attention → better source understanding.  
> In decoder-only: source is prepended to target and all attention is causal → source tokens on the right can't attend to source tokens on the left after the source ends.  
> However, modern large decoder-only models (GPT-4, LLaMA-3) close this gap by using very long contexts and scale.  
> The architectural advantage of encoder-decoder is clearer at smaller model sizes.

**Q: What is the information bottleneck in encoder-decoder?**
> The decoder can only access encoder information through cross-attention.  
> The encoder output H_enc (shape: T_enc × d) must encode everything the decoder might need.  
> If T_enc is short or d is small, information is lost.  
> In decoder-only models, the decoder attends directly to every source token — no bottleneck.  
> This is why large decoder-only LLMs often outperform encoder-decoder for tasks where both are applicable.

**Q: During inference, is the encoder run once or multiple times?**
> Once. The encoder processes the full input and produces H_enc.  
> H_enc is cached — cross-attention K, V = H_enc @ W_K and H_enc @ W_V are computed once.  
> The decoder generates tokens autoregressively, but at each step it cross-attends to the same fixed H_enc.  
> This is more efficient than decoder-only (where the KV cache grows with both source and generated tokens).

---

## Connections

| Concept | Built on | Used in |
|---|---|---|
| Cross-attention | Scaled dot-product attention (4.nlp/05) | T5, BART, mT5, Whisper, DALL-E, vision transformers |
| Encoder self-attention | Identical to BERT (5.transformers/05) | T5 encoder, BERT, RoBERTa |
| Decoder causal self-attention | Identical to GPT (5.transformers/06) | T5 decoder, GPT, LLaMA |
| Span corruption | MLM concept from BERT | T5 pretraining, UL2 |
| Teacher forcing | CLM training from GPT | Any seq2seq training |
| Autoregressive decoding | GPT inference | T5, BART, any generative model |

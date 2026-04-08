# 01 — Word Embeddings (Word2Vec, GloVe, FastText)

## Quick Reference

| Model | Type | OOV? | Context? | Best For |
|-------|------|------|---------|---------|
| Word2Vec (CBOW) | Static | ❌ | Local window | Fast training, semantic similarity |
| Word2Vec (Skip-gram) | Static | ❌ | Local window | Rare words, analogy tasks |
| GloVe | Static | ❌ | Global co-occurrence | Linear structure, analogies |
| FastText | Static | ✅ subword | Local window | Morphology, noisy/OOV text |
| BERT | Contextual | ✅ | Full sequence | SOTA — context-dependent meaning |

**Key difference vs BERT:** Word2Vec/GloVe/FastText give one fixed vector per word regardless of context. BERT gives different vector per occurrence based on context.

---

## 1. Why Dense Embeddings?

```
Sparse (TF-IDF):
  "cat" → [0, 0, 1, 0, 0, ..., 0]   (50K-dim, one-hot style)
  "feline" → [0, 0, 0, 0, 1, ..., 0]
  cosine("cat", "feline") = 0.0      ← no semantic relationship

Dense (Word2Vec):
  "cat"    → [0.23, -0.45, 0.78, ..., 0.12]   (300-dim)
  "feline" → [0.21, -0.43, 0.80, ..., 0.10]   (close!)
  cosine("cat", "feline") ≈ 0.85    ← semantically similar

Dense embeddings capture:
  - Synonyms: doctor ≈ physician
  - Analogies: king - man + woman ≈ queen
  - Semantic fields: [cat, dog, bird] cluster together
  - Syntactic roles: [quickly, slowly, rapidly] cluster together
```

---

## 1.5 The Embedding Matrix — Concrete Example

Two sentences, tiny 3-dimensional embeddings so you can see every number.

```
Sentences:
  S1: "I love cats"
  S2: "cats love fish"

Vocabulary (sorted, 0-indexed):
  0 → "I"
  1 → "cats"
  2 → "fish"
  3 → "love"

Embedding matrix W  [shape: 4 words × 3 dims]
         dim_0   dim_1   dim_2
  "I"   [ 0.21,  0.45, -0.12 ]   ← row 0
  "cats"[ 0.89, -0.32,  0.76 ]   ← row 1
  "fish"[ 0.83, -0.28,  0.71 ]   ← row 2
  "love"[ 0.05,  0.92,  0.10 ]   ← row 3
```

**Lookup = just index the row.** No multiplication needed.

```
"cats" has index 1  →  W[1]  =  [0.89, -0.32, 0.76]
"love" has index 3  →  W[3]  =  [0.05,  0.92, 0.10]
"fish" has index 2  →  W[2]  =  [0.83, -0.28, 0.71]
```

**Sentence S2 "cats love fish" → sequence of 3 vectors:**

```
  Token:   cats            love            fish
  Index:    1               3               2
            ↓               ↓               ↓
         [0.89,-0.32,0.76] [0.05,0.92,0.10] [0.83,-0.28,0.71]

  Shape passed to model: [seq_len=3, embed_dim=3]
```

**Why "cats" ≈ "fish" (cosine ≈ 0.99 here)?**
Both appear in the same context (after "love", near each other). During training,
their rows get pulled toward the same direction. "love" lives in a different
region (high dim_1, low dim_0) — it plays a syntactic role (verb), not a noun.

```python
import numpy as np

# The embedding matrix — rows = words, cols = dimensions
W = np.array([
    [ 0.21,  0.45, -0.12],  # "I"
    [ 0.89, -0.32,  0.76],  # "cats"
    [ 0.83, -0.28,  0.71],  # "fish"
    [ 0.05,  0.92,  0.10],  # "love"
])

vocab = {"I": 0, "cats": 1, "fish": 2, "love": 3}

# Lookup: just index the matrix
def embed(word):
    return W[vocab[word]]

# Sentence → matrix of embeddings
def sentence_to_matrix(sentence):
    tokens = sentence.lower().split()
    return np.stack([embed(t) for t in tokens])

s2 = sentence_to_matrix("cats love fish")
print(s2.shape)   # (3, 3)  ← [seq_len, embed_dim]
print(s2)
# [[ 0.89 -0.32  0.76]
#  [ 0.05  0.92  0.10]
#  [ 0.83 -0.28  0.71]]

# cats vs fish similarity
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine(embed("cats"), embed("fish")))   # 0.999  ← very similar
print(cosine(embed("cats"), embed("love")))   # -0.14  ← unrelated
```

**In PyTorch — exactly the same logic, GPU-accelerated:**

```python
import torch
import torch.nn as nn

# vocab_size=4, embed_dim=3
embedding = nn.Embedding(num_embeddings=4, embedding_dim=3)

# Preload our hand-crafted weights
embedding.weight = nn.Parameter(torch.tensor(W, dtype=torch.float32))

# S2: "cats love fish" → token indices [1, 3, 2]
s2_indices = torch.tensor([1, 3, 2])
output = embedding(s2_indices)
print(output.shape)   # torch.Size([3, 3])
# tensor([[ 0.8900, -0.3200,  0.7600],
#         [ 0.0500,  0.9200,  0.1000],
#         [ 0.8300, -0.2800,  0.7100]])
```

**Key insight:** The embedding matrix is just a lookup table. Index in → row out.
Training (Word2Vec / GloVe / fine-tuning) is simply adjusting these rows so
semantically similar words end up with similar row vectors.

---

## 2. Word2Vec

### Core Idea
Train a neural network to predict words from context (or context from words). The learned weight matrix = word embeddings. The prediction task is just a vehicle — the embeddings are the real output.

### Two Architectures

**CBOW (Continuous Bag of Words):**
```
Task: predict the CENTER word from CONTEXT words

Context: [The, ___, sat, on, the, mat]
Target:  predict "cat"

Architecture:
  context words → average their one-hot vectors
  → single hidden layer (embedding matrix W)
  → output layer (predict center word)
  → softmax over vocabulary

Training signal: maximize P("cat" | "The", "sat", "on", "the", "mat")

Faster to train, slightly worse on rare words
```

**Skip-gram:**
```
Task: predict CONTEXT words from the CENTER word

Center: "cat"
Task: predict ["The", "sat", "on", "the", "mat"]

Architecture:
  center word → embedding lookup
  → predict each context word independently
  → one softmax per context word

Slower but better for rare words and analogies
```

### Negative Sampling (Critical Trick)
```
Problem: softmax over 100K+ vocabulary at every step = expensive
Solution: Instead of "predict the correct word from all words",
          reframe as binary: "is this word a real context word or noise?"

Positive: (cat, sat) → label 1   [real co-occurrence]
Negative: (cat, xyz) → label 0   [random noise words sampled by frequency^0.75]

Use binary cross-entropy for each pair.
Sample k=5-20 negatives per positive → 5-20× speedup
```

### Training
```
Window size: typically 5 (±5 words around center)
Dimensions: 100-300 (300 for most tasks, 100 for speed)
Epochs: 5-10 passes over corpus
Min count: ignore words appearing < 5 times
Negative samples: 5-20

Corpus size: tens of billions of words for quality (Google News = 100B tokens)
```

```python
from gensim.models import Word2Vec

# Train from scratch
sentences = [["I", "love", "NLP"], ["NLP", "and", "AI", "are", "related"]]

model = Word2Vec(
    sentences=sentences,
    vector_size=300,      # embedding dimension
    window=5,             # context window size
    min_count=1,          # min word frequency
    workers=4,            # parallel training
    sg=1,                 # 1=Skip-gram, 0=CBOW
    negative=10,          # negative samples
    epochs=10
)

# Save / load
model.save('word2vec.model')
model = Word2Vec.load('word2vec.model')

# Use the embeddings
word_vectors = model.wv

# Similarity
print(word_vectors.similarity('cat', 'dog'))      # ~0.75
print(word_vectors.most_similar('king', topn=5))   # [queen, monarch, ...]

# Analogies: king - man + woman = ?
result = word_vectors.most_similar(
    positive=['king', 'woman'], negative=['man'], topn=1
)
# [('queen', 0.89)]

# Load pretrained (Google News, 3M words, 300d)
from gensim.models import KeyedVectors
wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
```

### Word2Vec Properties
```
Linear structure in embedding space:
  king - man + woman ≈ queen
  Paris - France + Italy ≈ Rome
  bigger - big + small ≈ smaller

Semantic clusters:
  animals: [cat, dog, bird, fish] clustered
  countries: [France, Germany, Italy] clustered
  verbs: [run, walk, jump] clustered

These emerge from training objective — never explicitly taught
```

---

## 3. GloVe (Global Vectors)

### Core Idea
Word2Vec uses local context windows — only sees nearby word pairs. GloVe uses **global co-occurrence statistics** from the entire corpus.

```
Co-occurrence matrix X:
  X[i][j] = how many times word i appears near word j in entire corpus

"ice" with "cold": X[ice][cold] = 5000  (high — related)
"ice" with "steam": X[ice][steam] = 50   (low — less related)
"ice" with "water": X[ice][water] = 1000 (medium)

Ratio X[ice][k] / X[steam][k]:
  k="cold": 8.9 → much more related to ice than steam
  k="hot":  0.085 → much more related to steam than ice
  k="water": 1.36 → roughly equally related to both

GloVe trains embeddings such that wᵢ · wⱼ ≈ log(Xᵢⱼ)
The dot product between embeddings should reflect log co-occurrence count
```

### GloVe Loss Function
```
J = Σᵢⱼ f(Xᵢⱼ) · (wᵢ · w̃ⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ)²

f(Xᵢⱼ) = weighting function:
  f(x) = (x/x_max)^α  if x < x_max, else 1
  Downweights very frequent co-occurrences (stop words)
  x_max = 100, α = 0.75 (standard)

wᵢ = word vector, w̃ⱼ = context vector, bᵢ, b̃ⱼ = biases
```

```python
import numpy as np

# Load pretrained GloVe (download: https://nlp.stanford.edu/projects/glove/)
def load_glove(path, dim=100):
    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

glove = load_glove('glove.6B.100d.txt', dim=100)
print(glove['king'].shape)  # (100,)

# Cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_sim(glove['cat'], glove['dog']))    # ~0.82
print(cosine_sim(glove['cat'], glove['table']))  # ~0.25

# Analogy: king - man + woman
analogy = glove['king'] - glove['man'] + glove['woman']
# Find nearest word to analogy vector (excluding the 3 query words)
```

### GloVe Pretrained Variants
```
glove.6B.50d    — Wikipedia + Gigaword, 6B tokens, 50 dimensions
glove.6B.100d   — Same corpus, 100 dimensions
glove.6B.300d   — Same corpus, 300 dimensions
glove.42B.300d  — Common Crawl, 42B tokens, 300 dimensions (larger, better)
glove.840B.300d — Common Crawl, 840B tokens, 300 dimensions (largest)
```

### GloVe vs Word2Vec
```
GloVe:
  ✅ Uses global statistics → captures corpus-wide relationships
  ✅ Faster to train (one pass over co-occurrence matrix)
  ✅ Competitive or better on analogy tasks
  ❌ Memory: co-occurrence matrix can be huge

Word2Vec:
  ✅ Online learning (doesn't need full corpus upfront)
  ✅ Easy to update with new data (continue training)
  ✅ Generally similar quality to GloVe
  ❌ Only local window context

In practice: similar quality. GloVe pretrained vectors widely used. Word2Vec preferred when training on custom domain.
```

---

## 4. FastText

### The OOV Problem
```
Word2Vec / GloVe:
  "running" → embedding vector ✅
  "runnning" (typo) → OOV → zero vector or [UNK] ❌
  "ChatGPT" (new word) → OOV ❌
  "tokenization" (rare) → OOV ❌

Real-world NLP: OCR errors, social media slang, technical jargon, new words
→ OOV is a serious problem for static word-level embeddings
```

### FastText Solution: Character N-grams
```
Every word represented as sum of its character n-gram embeddings

"running" (n=3):
  <ru, run, unn, nni, nin, ing, ng>  (< and > mark word boundaries)
  Embedding("running") = sum of embeddings of all its character n-grams

OOV word "runnning":
  <ru, run, unn, unn, nni, nin, ing, ng>
  Shares most n-grams with "running" → gets similar embedding ✅

"ChatGPT":
  <Ch, Cha, hat, atG, tGP, GPT, PT>
  New word but meaningful n-gram embeddings ✅

Morphologically rich languages (Finnish, Turkish):
  "running", "runner", "ran", "runs" all share n-grams → naturally related ✅
```

```python
from gensim.models import FastText

# Train FastText
model = FastText(
    sentences=sentences,
    vector_size=300,
    window=5,
    min_count=1,
    workers=4,
    sg=1,          # Skip-gram
    min_n=3,       # minimum n-gram size
    max_n=6,       # maximum n-gram size
    epochs=10
)

# Handles OOV words!
print(model.wv['running'])       # known word
print(model.wv['runnning'])      # typo — still works via shared n-grams!
print(model.wv['chatgpt'])       # new word — still works

model.save('fasttext.model')

# Load pretrained FastText (Facebook's pretrained vectors)
from gensim.models.fasttext import load_facebook_model
model = load_facebook_model('cc.en.300.bin')  # Common Crawl, 300d
```

### FastText Pretrained
```
Facebook Research released pretrained FastText for 157 languages:
  cc.en.300.bin      — English Common Crawl, 300d
  cc.{lang}.300.bin  — Other languages

Quality: competitive with GloVe for English, significantly better for morphologically rich languages.
```

### When FastText > Word2Vec / GloVe
```
✅ OCR output (character errors) → typos handled via n-gram overlap
✅ Social media text (slang, abbreviations)
✅ Domain-specific jargon (medical, legal, technical)
✅ Morphologically rich languages (German compounds, Turkish suffixes)
✅ Any text with frequent OOV words

❌ When vocabulary is fixed and known → GloVe/Word2Vec sufficient
❌ When character n-grams carry no signal (random OCR errors with no phonetic basis)
```

---

## 4.5 Word2Vec vs GloVe vs FastText — Concrete Comparison

Same 2 sentences throughout so you can see exactly what each model does differently.

```
Corpus:
  S1: "I love cats"
  S2: "cats love fish"

Vocabulary: {"I":0, "love":1, "cats":2, "fish":3}
```

---

### Step 1 — Word2Vec: sees training PAIRS from a sliding window

Skip-gram, window=1: for each center word, emit (center → context) pairs.

```
S1: "I love cats"
  center=I     → context: [love]        → pair (I, love)
  center=love  → context: [I, cats]     → pairs (love, I), (love, cats)
  center=cats  → context: [love]        → pair (cats, love)

S2: "cats love fish"
  center=cats  → context: [love]        → pair (cats, love)   ← same pair again!
  center=love  → context: [cats, fish]  → pairs (love, cats), (love, fish)
  center=fish  → context: [love]        → pair (fish, love)

All training pairs:
  (I, love)      ×1
  (love, I)      ×1
  (love, cats)   ×2   ← seen twice → strong pull
  (cats, love)   ×2   ← seen twice → strong pull
  (love, fish)   ×1
  (fish, love)   ×1
```

Word2Vec only ever sees **local pairs**. It never knows that `cats` and `fish`
directly co-occur — it only sees each of them next to `love`. They become similar
*indirectly* (both pulled toward `love`).

---

### Step 2 — GloVe: sees a CO-OCCURRENCE MATRIX over the full corpus

GloVe counts every (word_i, word_j) pair within the window across the *entire*
corpus first, then trains on those counts.

```
Co-occurrence matrix X  (window=1, counted over both sentences):

         I     love  cats  fish
  I    [ 0,    1,    0,    0  ]
  love [ 1,    0,    2,    1  ]   ← love appears next to cats TWICE
  cats [ 0,    2,    0,    0  ]
  fish [ 0,    1,    0,    0  ]
```

How each cell gets its value — window=1 means only **immediate neighbors** count:

```
S1: "I    love  cats"
     pos0  pos1  pos2

  "I"    neighbors: [love]        distance to cats = 2 → NOT counted
  "love" neighbors: [I, cats]
  "cats" neighbors: [love]

S2: "cats  love  fish"
     pos0   pos1  pos2

  "cats" neighbors: [love]
  "love" neighbors: [cats, fish]
  "fish" neighbors: [love]

Pair          S1   S2   Total   → cell
──────────────────────────────────────
(I,    love)   1    0     1     X[I][love]    = 1
(love, cats)   1    1     2     X[love][cats] = 2
(love, fish)   0    1     1     X[love][fish] = 1
(I,    cats)   0    0     0     X[I][cats]    = 0  ← distance=2 in S1, never adjacent
(cats, fish)   0    0     0     X[cats][fish] = 0  ← distance=2 in S2, never adjacent
```

```
X[love][cats] = 2   (S1 contributes 1, S2 contributes 1)
X[love][fish] = 1   (only S2)
X[cats][fish] = 0   (never directly adjacent)
```

GloVe's objective:  `w_i · w_j  ≈  log( X[i][j] )`

```
w_love · w_cats ≈ log(2) = 0.69   ← stronger signal
w_love · w_fish ≈ log(1) = 0.00   ← weaker signal
w_cats · w_fish ≈ log(0) = -∞     ← no direct signal at all
```

**Key difference from Word2Vec:** GloVe uses the count matrix as a global
summary. Word2Vec stochastically samples pairs online. Both produce similar
final vectors for this example, but GloVe's training is one-pass over the matrix.

---

### Step 3 — FastText: breaks words into character n-grams

Word2Vec and GloVe treat words as atomic units. FastText decomposes each word.

```
"cats"  (n=3, with boundary markers < >):
  character n-grams: <ca, cat, ats, ts>  +  the whole-word token <cats>

  Embedding("cats") = E(<ca>) + E(<cat>) + E(<ats>) + E(<ts>) + E(<cats>)
                      ↑ sum of all n-gram vectors

"fish"  character n-grams: <fi, fis, ish, sh>  +  <fish>
"love"  character n-grams: <lo, lov, ove, ve>  +  <love>
```

**OOV word at inference — "catss" (typo):**

```
Word2Vec / GloVe:
  "catss" → not in vocab → [UNK] → zero vector  ❌ no information

FastText:
  "catss" n-grams: <ca, cat, ats, tss, ss>  +  <catss>
                    ↑        ↑    ↑
              shared with "cats"!

  Embedding("catss") = E(<ca>) + E(<cat>) + E(<ats>) + E(<tss>) + E(<ss>) + E(<catss>)
                       ≈ very close to Embedding("cats")  ✅ meaningful vector
```

**OOV word "fishing" (unseen form):**

```
FastText n-grams: <fi, fis, ish, shi, hin, ing, ng>
                   ↑    ↑    ↑
              shares <fi, fis, ish> with "fish"

Embedding("fishing") ≈ similar to Embedding("fish")  ✅
```

---

### Side-by-Side Summary

```python
import numpy as np

vocab     = {"I": 0, "love": 1, "cats": 2, "fish": 3}
sentences = ["I love cats", "cats love fish"]

# ── Word2Vec: what it learned from (training pairs only) ──────────────────────
# Pairs seen:  (love,cats)×2, (cats,love)×2, (love,fish)×1, (fish,love)×1
# Result: "cats" and "fish" end up similar because both co-occur with "love"
w2v = {
    "I":    np.array([ 0.10,  0.80, -0.20]),
    "love": np.array([ 0.05,  0.92,  0.10]),
    "cats": np.array([ 0.89, -0.32,  0.76]),
    "fish": np.array([ 0.83, -0.28,  0.71]),   # ← similar to cats
}

# ── GloVe: learned from co-occurrence counts ─────────────────────────────────
# X[love][cats]=2, X[love][fish]=1  →  love-cats bond is stronger
glove = {
    "I":    np.array([ 0.12,  0.78, -0.18]),
    "love": np.array([ 0.04,  0.93,  0.09]),
    "cats": np.array([ 0.90, -0.31,  0.77]),
    "fish": np.array([ 0.70, -0.18,  0.60]),   # ← slightly further from cats
}                                               #   because X[love][fish] < X[love][cats]

# ── FastText: sum of n-gram vectors ──────────────────────────────────────────
ft_ngrams = {                   # each n-gram has its own trained vector
    "<ca":  np.array([ 0.40, -0.10,  0.35]),
    "cat":  np.array([ 0.35, -0.12,  0.32]),
    "ats":  np.array([ 0.30, -0.15,  0.30]),
    "ts>":  np.array([ 0.20, -0.08,  0.18]),
    "<cats>": np.array([ 0.04,  0.03,  0.01]),  # whole-word residual
}
cats_embedding = sum(ft_ngrams.values())        # [1.29, -0.42, 1.16]

# OOV word "catss" — shares most n-grams with "cats"
ft_catss = {
    "<ca":  np.array([ 0.40, -0.10,  0.35]),   # shared ✅
    "cat":  np.array([ 0.35, -0.12,  0.32]),   # shared ✅
    "ats":  np.array([ 0.30, -0.15,  0.30]),   # shared ✅
    "tss":  np.array([ 0.02,  0.01,  0.02]),   # new (small random)
    "ss>":  np.array([ 0.01,  0.00,  0.01]),   # new (small random)
}
catss_embedding = sum(ft_catss.values())        # [1.08, -0.36, 1.00]  ≈ cats ✅

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine(cats_embedding, catss_embedding))  # ~0.999 — typo handled!
```

---

### What each model "knows" vs "doesn't know"

```
Scenario             Word2Vec        GloVe           FastText
─────────────────────────────────────────────────────────────────
cats ≈ fish?          ✅ yes          ✅ yes          ✅ yes
  (both near "love")

love-cats bond >       ❌ depends     ✅ yes           ✅ yes
love-fish bond?           on sampling    (count=2 vs 1)

"catss" (OOV typo)    ❌ zero vector  ❌ zero vector  ✅ ~cats

"fishing" (OOV form)  ❌ zero vector  ❌ zero vector  ✅ ~fish

Training requires      streaming      full corpus     streaming
full corpus upfront?   pairs          count matrix    pairs + n-grams
```

---

## 5. Using Embeddings in Models

### As Feature Input (Classical ML)
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

def text_to_embedding(text, model, dim=300):
    """Average word vectors for sentence representation"""
    tokens = text.lower().split()
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if not vectors:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)   # mean pooling

# Sentence embeddings
X_train_emb = np.array([text_to_embedding(t, w2v_model) for t in train_texts])
X_test_emb  = np.array([text_to_embedding(t, w2v_model) for t in test_texts])

clf = LogisticRegression(C=1.0, max_iter=1000)
clf.fit(X_train_emb, y_train)
```

### As Embedding Layer in PyTorch
```python
import torch
import torch.nn as nn
import numpy as np

def load_pretrained_embeddings(vocab, glove_path, dim=300):
    """Create embedding matrix from GloVe for model vocabulary"""
    glove = {}
    with open(glove_path) as f:
        for line in f:
            parts = line.split()
            glove[parts[0]] = np.array(parts[1:], dtype=np.float32)

    # Initialize with random (for words not in GloVe)
    embedding_matrix = np.random.normal(0, 0.1, (len(vocab), dim))
    found = 0
    for word, idx in vocab.items():
        if word in glove:
            embedding_matrix[idx] = glove[word]
            found += 1

    print(f"Found {found}/{len(vocab)} words in GloVe")
    return torch.FloatTensor(embedding_matrix)

# Model with pretrained embeddings
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, embedding_matrix=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)
            self.embedding.weight.requires_grad = True   # fine-tune embeddings

        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)   # [B, seq_len, embed_dim]
        out, (h, c) = self.lstm(embedded)
        # Use last hidden state from both directions
        h = torch.cat([h[0], h[1]], dim=1)   # [B, 256]
        return self.fc(h)

embedding_matrix = load_pretrained_embeddings(vocab, 'glove.6B.300d.txt')
model = TextClassifier(len(vocab), 300, num_classes=5, embedding_matrix=embedding_matrix)
```

### Freeze vs Fine-tune Embeddings
```python
# Freeze (don't update during training — good for small data)
self.embedding.weight.requires_grad = False

# Fine-tune (update during training — good for domain-specific data)
self.embedding.weight.requires_grad = True
```

---

## 6. Sentence Embeddings (Beyond Word Vectors)

### Mean Pooling (Simple Baseline)
```python
# Average all word vectors in a sentence
sentence_vec = mean(word_vectors)   # loses word order
```

### Sentence-BERT (SBERT) — Current Standard
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')   # fast, good quality

sentences = ["I love NLP", "Natural language processing is great", "I hate Mondays"]
embeddings = model.encode(sentences)   # [3, 384]

# Semantic similarity
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(embeddings)
# sims[0][1] ≈ 0.89 (love NLP ≈ NLP is great)
# sims[0][2] ≈ 0.15 (love NLP ≠ hate Mondays)
```

---

## 7. Embedding Evaluation

### Intrinsic Evaluation (Embedding Quality)
```
Word similarity: cosine(cat, feline) should be high
Analogy: king - man + woman = queen  (word2vec-analogy dataset)

Standard benchmarks:
  WordSim-353: word pair similarity scores
  SimLex-999:  true semantic similarity (not relatedness)
  MEN: 3000 word pairs
```

### Extrinsic Evaluation (Downstream Task)
```
Better measure: does the embedding improve the actual task?
  Text classification F1 with these embeddings
  NER F1 with these embeddings
  Always evaluate on your downstream task, not just intrinsic benchmarks
```

---

## 8. Static vs Contextual — The Key Distinction

```
Static embeddings (Word2Vec, GloVe, FastText):
  "bank" (river) → [0.23, -0.45, 0.78, ...]
  "bank" (finance) → [0.23, -0.45, 0.78, ...]   ← SAME vector!

  One vector per word type, ignores context.
  Cannot distinguish: "I went to the river bank" vs "I went to the bank"

Contextual embeddings (BERT, GPT, ELMo):
  "I went to the river bank" → bank gets vector [0.12, 0.89, ...]
  "I went to the bank" → bank gets vector [0.67, -0.23, ...]   ← DIFFERENT!

  Different vector per word token based on full sentence context.
  Solves polysemy (one word, multiple meanings) completely.

This is why BERT/contextual models outperform Word2Vec/GloVe for most tasks.
```

---

## 9. When to Use What

| Scenario | Model | Why |
|----------|-------|-----|
| Fast baseline, no GPU | Word2Vec / GloVe pretrained + mean pooling | Simple, effective |
| OCR/noisy text, OOV words | FastText | Subword handles misspellings |
| Morphologically rich language | FastText | Character n-grams capture morphology |
| Semantic similarity / search | SBERT (sentence-transformers) | Best sentence-level embeddings |
| Custom domain vocabulary | Train Word2Vec/FastText on domain corpus | Domain-specific embeddings |
| SOTA accuracy, any NLP task | Fine-tune BERT/RoBERTa | Contextual > static always |
| Embedding layer in neural model | Pretrained GloVe/FastText → fine-tune | Better init than random |

---

## 10. Gotchas

**Static embeddings can't handle polysemy.**
"Apple" (company) and "apple" (fruit) get the same vector. For tasks where word sense matters, you need contextual embeddings (BERT).

**Out-of-vocabulary at inference must be handled.**
For Word2Vec/GloVe: if a word is OOV at inference, you have no vector. Either skip it (mean of other words), use UNK vector (mean of all training vectors), or switch to FastText.

**Embedding dimension affects model size significantly.**
300d embeddings with 100K vocabulary = 30M parameters just for the embedding layer. If model is memory-constrained, use 50d or 100d embeddings.

**Don't compare embeddings trained on different corpora.**
Word2Vec trained on medical text and GloVe trained on news have different vector spaces — cosine similarities between them are meaningless. Always use embeddings from the same model.

**Mean pooling loses word order.**
"man bites dog" and "dog bites man" have the same mean embedding. If order matters (it almost always does), use LSTM/Transformer over word embeddings, not just mean pooling.

---

## 11. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Many OOV words in inference | Word-level embedding with small vocab | Switch to FastText; lower min_count |
| Embeddings for same domain are far apart | Different training corpora | Retrain on same corpus; use domain-specific pretrained |
| Model ignores semantics (cat ≠ feline) | TF-IDF used instead of embeddings | Add embedding layer; try SBERT |
| Embedding layer OOM | Vocab too large × dim | Reduce max_features; use 100d instead of 300d |
| Analogy tasks fail on custom domain | General-purpose embeddings (news) | Train embeddings on domain corpus |
| Slow inference with large embedding matrix | CPU lookup, large vocab | Use GPU; reduce vocab; quantize |

---

## 12. Interview Q&A (Senior Level)

**Q: Explain how Word2Vec learns embeddings — what is the actual training signal?**
A: Word2Vec doesn't directly learn embeddings — it trains a shallow neural network on a proxy task: predict a word from its context (CBOW) or predict context from a word (Skip-gram). The embedding matrix is the weight matrix of the hidden layer. The training signal comes from the prediction task: are these two words likely to co-occur in a window of text? With negative sampling, each training step updates embeddings based on: "this (center, context) pair appears in real text → pull their vectors closer; this (center, random_word) pair is noise → push their vectors apart." After training on billions of pairs, words that co-occur in similar contexts end up with similar vectors — hence "king" and "queen" are close (appear in similar contexts), and "king - man + woman ≈ queen" because the gender difference is encoded consistently.

**Q: What is the difference between FastText and Word2Vec at a character level?**
A: FastText decomposes each word into character n-grams (e.g., "running" → <run, unn, nni, nin, ing, ng>) and trains a separate embedding for each n-gram. The word embedding is the sum of all its n-gram embeddings. This means: (1) OOV words at inference time still get meaningful embeddings via shared n-grams with known words; (2) morphologically related words ("run", "running", "runner") naturally share many n-grams → their embeddings are naturally close; (3) typos share most n-grams with the correct word → robust to noise. The tradeoff: training is slower (many more embeddings to learn) and vocabulary is effectively the set of all possible n-grams.

**Q: Why did static word embeddings become obsolete for most NLP tasks?**
A: Three limitations of static embeddings: (1) Polysemy — one vector per word type regardless of meaning; "bank" gets the same vector whether it's a financial institution or a river bank. (2) Context-independence — "not good" and "good" have similar embeddings for "good" since the word itself is the same. (3) Out-of-vocabulary — Word2Vec/GloVe have no representation for unseen words (FastText partially fixes this). BERT solves all three: (1) contextual embeddings differ per occurrence; (2) attention captures full sentence context including negation; (3) subword tokenization handles any word. The result: BERT consistently outperforms Word2Vec/GloVe on nearly all benchmarks by 5-20 F1 points. Static embeddings remain useful as cheap feature inputs for classical ML and for tasks that don't need contextual disambiguation.

---

## 13. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| Tokenization prerequisite | `../fundamentals/01_text_preprocessing.md` | Tokenize → embed |
| Sparse vs dense comparison | `../fundamentals/02_text_representations.md` | Evolution from TF-IDF |
| Embeddings in RNN/LSTM | `../sequence_models/01_rnn_to_attention.md` | Embedding layer feeds into RNN |
| Contextual embeddings (BERT) | `../../4.transformers/` | BERT = contextual word embeddings |
| Embedding layer in DL | `../../1.deep learning/fundamentals/05_modern_components.md` | Embedding math |

---

## Key Takeaway

**The embedding evolution:**
```
One-hot (sparse) → Word2Vec/GloVe (static dense, semantic) → FastText (OOV-robust) → BERT (contextual)
```

**Choose by task:**
- Static (W2V/GloVe/FastText): fast baseline, no GPU, embedding layer in custom neural model
- FastText: any text with OOV risk (OCR, social media, technical)
- SBERT: semantic similarity, search
- BERT/fine-tuned: SOTA accuracy for classification, NER, QA

**The critical insight:** Word2Vec encodes "what words appear near each other" → semantics emerge. This is why `king - man + woman ≈ queen` — the male/female difference is encoded linearly in the embedding space from co-occurrence patterns.

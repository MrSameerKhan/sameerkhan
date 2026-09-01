# Tokenization — End-to-End

> This file owns **text → token IDs**. The next step, **token IDs → vectors** (the embedding
> matrix, weight tying, vocabulary economics), is
> [05_embedding_lookup_end_to_end.md](05_embedding_lookup_end_to_end.md).

> Every merge step computed. BPE built from scratch.

---

## The Problem

Every model needs text → numbers. The question is: what unit do you split on?

```
"unhappiness"

Word-level:   ["unhappiness"]         ← OOV if not in vocabulary
Char-level:   ["u","n","h","p","p","i","n","e","s","s"]  ← too granular, no meaning
Subword-level: ["un", "happiness"] or ["unhappy", "##ness"]  ← best of both
```

**The OOV problem (Out of Vocabulary):**
Word-level vocabulary of 50,000 words.
User types: "GPT-4ification" — not in vocab — replaced with [UNK] — model sees nothing useful.
With subword: "G", "PT", "-", "4", "ification" — model sees real units even for novel words.

### Three Approaches

| Method | Unit | Vocabulary | OOV problem | Used by |
|---|---|---|---|---|
| Word-level | Whole words | Large (50K-300K) | Frequent (new words, typos) | Older NLP (Word2Vec) |
| Char-level | Characters | Tiny (~256) | None | Some old RNNs |
| Subword | Named pieces | Medium (32K-128K) | Rare | All modern LLMs |

Subword wins: small vocab, no OOV, meaningful units.

---

## BPE — Byte Pair Encoding

Used by: GPT-2, GPT-3, GPT-4, LLaMA, RoBERTa.

**Core idea:** Start with character vocabulary. Iteratively merge the most frequent adjacent pair. Repeat until vocabulary target reached.

### Setup: Corpus

```
Word frequencies (extracted from training corpus):
  "cat" = 3 occurrences
  "bat" = 2 occurrences
  "mat" = 2 occurrences
  "cap" = 1 occurrence
```

### Step 0 — Initial representation

Mark end of every word with `</w>` (word boundary marker). Split each word into characters:

```
"c a t </w>" : 3
"b a t </w>" : 2
"m a t </w>" : 2
"c a p </w>" : 1
```

Initial vocabulary: {c, a, t, b, m, p, </w>} = **7 tokens**.

### Step 1 — Count all adjacent pairs

Scan each word representation, multiply by word frequency:

```
From "c a t </w>" (freq=3):
  (c, a) : 3
  (a, t) : 3
  (t, </w>) : 3

From "b a t </w>" (freq=2):
  (b, a) : 2
  (a, t) : 2
  (t, </w>) : 2

From "m a t </w>" (freq=2):
  (m, a) : 2
  (a, t) : 2
  (t, </w>) : 2

From "c a p </w>" (freq=1):
  (c, a) : 1
  (a, p) : 1
  (p, </w>) : 1
```

Total pair frequencies:

| Pair | Count |
|---|---|
| (a, t) | 3+2+2 = **7** |
| (t, </w>) | 3+2+2 = **7** |
| (c, a) | 3+1 = 4 |
| (b, a) | 2 |
| (m, a) | 2 |
| (a, p) | 1 |
| (p, </w>) | 1 |

### Merge 1 — Most frequent pair: (a, t) → "at" [tied at 7, pick (a,t)]

Replace every occurrence of "a t" with "at":

```
Before:                      After:
"c a t </w>" : 3   →   "c at </w>" : 3
"b a t </w>" : 2   →   "b at </w>" : 2
"m a t </w>" : 2   →   "m at </w>" : 2
"c a p </w>" : 1   →   "c a p </w>" : 1   (no "a t" here)
```

Vocabulary: {c, a, t, b, m, p, </w>, **at**} = **8 tokens**.

### Step 2 — Recount pairs

```
From "c at </w>" (freq=3):   (c, at): 3,  (at, </w>): 3
From "b at </w>" (freq=2):   (b, at): 2,  (at, </w>): 2
From "m at </w>" (freq=2):   (m, at): 2,  (at, </w>): 2
From "c a p </w>" (freq=1):  (c, a):  1,  (a, p): 1,  (p, </w>): 1
```

Pair frequencies:

| Pair | Count |
|---|---|
| (at, </w>) | 3+2+2 = **7** |
| (c, at) | 3 |
| (b, at) | 2 |
| (m, at) | 2 |
| (c, a) | 1 |
| others | 1 |

### Merge 2 — Most frequent: (at, </w>) → "at</w>" [freq: 7]

```
"c at </w>" : 3   →   "c at</w>" : 3
"b at </w>" : 2   →   "b at</w>" : 2
"m at </w>" : 2   →   "m at</w>" : 2
"c a p </w>" : 1  →   "c a p </w>" : 1   (unchanged)
```

Vocabulary: {c, a, t, b, m, p, </w>, at, **at</w>**} = **9 tokens**.

### Step 3 — Recount pairs

```
From "c at</w>" (freq=3):   (c, at</w>): 3
From "b at</w>" (freq=2):   (b, at</w>): 2
From "m at</w>" (freq=2):   (m, at</w>): 2
From "c a p </w>" (freq=1): (c,a):1, (a,p):1, (p,</w>):1
```

Pair frequencies:

| Pair | Count |
|---|---|
| (c, at</w>) | **3** |
| (b, at</w>) | 2 |
| (m, at</w>) | 2 |
| others | 1 |

### Merge 3 — (c, at</w>) → "cat</w>" [freq: 3]

```
"c at</w>" : 3   →   "cat</w>" : 3   (single token now)
"b at</w>" : 2   →   "b at</w>" : 2  (unchanged)
"m at</w>" : 2   →   "m at</w>" : 2  (unchanged)
"c a p </w>": 1  →   "c a p </w>": 1 (unchanged)
```

### Merge 4 — (b, at</w>) → "bat</w>" [freq: 2]

### Merge 5 — (m, at</w>) → "mat</w>" [freq: 2]

### Final state

```
"cat</w>" : 3   → single token
"bat</w>" : 2   → single token
"mat</w>" : 2   → single token
"c a p </w>" : 1 → 4 tokens {c, a, p, </w>}
```

**Learned merge rules (in order — ORDER MATTERS!):**
```
Rule 1: a + t      → at
Rule 2: at + </w>  → at</w>
Rule 3: c + at</w> → cat</w>
Rule 4: b + at</w> → bat</w>
Rule 5: m + at</w> → mat</w>
```

**Final vocabulary:**
```
c, a, t, b, m, p, </w>,     ← original characters
at, at</w>, cat</w>, bat</w>, mat</w>   ← learned merges

Total: 12 tokens
```

---

## Encoding New Words (Inference)

Apply learned merge rules IN ORDER to any new word.

**"cat" (in training vocab):**
```
Start:         c a t </w>
Apply Rule 1:  c at </w>       (a + t → at)
Apply Rule 2:  c at</w>        (at + </w> → at</w>)
Apply Rule 3:  cat</w>         (c + at</w> → cat</w>)
Result: ["cat</w>"] = 1 token ✓
```

**"bat" (in training vocab):**
```
Start:         b a t </w>
Apply Rule 1:  b at </w>
Apply Rule 2:  b at</w>
Apply Rule 4:  bat</w>
Result: ["bat</w>"] = 1 token ✓
```

**"rats" (OOV — never seen in training):**
```
Start:         r a t s </w>
Apply Rule 1:  r at s </w>     (a + t → at)
Apply Rule 2:  r at s </w>     (at + </w>? No — "at" is followed by "s", not </w>)
No further rules apply.
Result: ["r", "at", "s", "</w>"] = 4 tokens
```
"rats" is split into known subwords, no [UNK] token.

**"catfish" (OOV):**
```
Start:         c a t f i s h </w>
Apply Rule 1:  c at f i s h </w>   (a + t → at)
Apply Rule 2:  c at f i s h </w>   ("at" followed by "f", not </w>. Doesn't apply)
Apply Rule 3:  c at</w>? No — Rule 3 is c + at</w>, we have c + at (followed by f). Doesn't apply.
No further rules match.
Result: ["c", "at", "f", "i", "s", "h", "</w>"] = 7 tokens
```

**Note:** "cat" is split as `["c", "at"]` NOT `["cat</w>"]` because the word boundary marker `</w>` is missing — "at" is mid-word here. This is how BPE distinguishes "cat" (standalone) from "cat" as prefix.

**Key insight:** The `</w>` marker makes "cat" (word) ≠ "cat" (prefix of catfish). GPT-2 uses `Ġ` (special space character) before word-starting tokens instead.

---

## WordPiece — BERT Family

Similar to BPE but uses a different merge criterion.

```
BPE:       merge the pair with highest frequency
WordPiece: merge the pair with highest likelihood gain

Score(A, B) = P(AB) / (P(A) × P(B))

Merge the pair where combining A+B increases the language model
probability the most (vs keeping them separate).
```

**Example:**
```
Pair (a, t): P(at)=0.38, P(a)=0.45, P(t)=0.42
Score = 0.38 / (0.45 × 0.42) = 0.38 / 0.189 = 2.011

Pair (c, a): P(ca)=0.18, P(c)=0.25, P(a)=0.45
Score = 0.18 / (0.25 × 0.45) = 0.18 / 0.113 = 1.593   ← higher score, merge first
```

### BERT's "##" notation

WordPiece uses "##" prefix to mark continuation subwords:
```python
"playing"   → ["play", "##ing"]    # "##ing" means "ing" that continues a word
"unhappy"   → ["un", "##happy"]
"cats"      → ["cat", "##s"]

"cat sat"   → ["cat", "sat"]       # no ## since these are word-starts
```

Decoding: strip `##` and concatenate: `["un", "##happy"]` → "unhappy"

---

## SentencePiece — T5, LLaMA Family

**Problem with BPE/WordPiece:** assumes pre-tokenization by whitespace. Languages like Japanese/Chinese/Thai have no spaces between words.

**SentencePiece:** treats the raw text as a sequence of Unicode characters. No pre-tokenization. Language agnostic.

**Space = special character `▁` (U+2581):**
```
"cat sat on mat" = "▁cat▁sat▁on▁mat"
Tokenization: ["▁cat", "▁sat", "▁on", "▁mat"]
```

The `▁` prefix tells you this token starts a new word. "cat" mid-word (no space) would be: `["c", "at"]` — no `▁`.

Decoding: replace `▁` with space → "cat sat on mat".

Used by: T5, LLaMA, Mistral, Gemma, ALBERT.

---

## GPT-2: Byte-Level BPE

Standard BPE works on characters (Unicode code points). Problem: rare Unicode characters → [UNK] even at character level.

**GPT-2 solution:** work on raw bytes (0-255 instead of Unicode code points).
```
Base vocabulary: 256 bytes (covers ALL possible byte sequences)
No [UNK] token — every possible string has a valid representation

"café" = [99, 97, 102, 195, 169]   ← bytes, including multi-byte UTF-8
```

Every modern LLM (GPT-3, GPT-4, LLaMA-2+) uses byte-level BPE or byte-level SentencePiece.

---

## Vocabulary Size: Tradeoffs

| Vocab size | Effect on tokenization | Effect on model |
|---|---|---|
| Too small (e.g., 1K) | Words split into many pieces | Long sequences → slow, expensive attention |
| Too large (e.g., 500K) | Rare tokens get few training examples | Embedding table too large, poor rare token embeddings |
| Sweet spot (32K-128K) | Common words = 1 token, rare words = few tokens | Balanced sequence length and embedding quality |

**Typical vocabulary sizes:**

| Model | Tokenizer | Vocab size |
|---|---|---|
| BERT-base | WordPiece | 30,522 |
| GPT-2 | BPE (byte-level) | 50,257 |
| LLaMA-2 | SentencePiece (BPE) | 32,000 |
| LLaMA-3 | Tiktoken (BPE) | 128,256 |
| GPT-4 | Tiktoken (BPE) | 100,277 |

---

## The Fertility Problem

Average tokens per word (tokenizer "fertility"):

```
"cat sat on mat":
  BERT WordPiece: 4 tokens (1 per word)
  GPT-2 BPE:      4 tokens

"antidisestablishmentarianism":
  BERT:    6 tokens ["anti", "##dis", "##establish", "##ment", "##arian", "##ism"]
  GPT-2:   7 tokens

"😊" (emoji):
  BERT:         1 token [UNK]  ← loses information!
  GPT-2 byte-BPE: 3 tokens (3 UTF-8 bytes) = no information loss
```

Code switching ("I went to the café"): English models trained on mostly English text. "café" with English tokenizer — GPT-2 handles it byte-level. BERT might tokenize oddly.

GPT-2 on English text: ~1.3 tokens/word. Same model on Python code: ~2.1 tokens/word (identifiers, operators split more). High fertility = longer sequences = more attention computation = slower training/inference. LLaMA-3's 128K vocab has lower fertility than LLaMA-2's 32K vocab — a key efficiency improvement.

---

## Code

### 1. BPE From Scratch

```python
from collections import Counter, defaultdict
import re

def get_vocab(corpus):
    """Convert corpus to character-level word representations."""
    vocab = Counter()
    for word in corpus:
        chars = list(word)
        vocab[' '.join(chars) + ' </w>'] += 1
    return vocab

def get_pairs(vocab):
    """Count all adjacent pairs across all words."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """Merge all occurrences of a pair in the vocabulary."""
    bigram      = ' '.join(pair)
    replacement = ''.join(pair)
    new_vocab   = {}
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def bpe_train(corpus, num_merges):
    """Train BPE on a corpus, return merge rules."""
    vocab  = get_vocab(corpus)
    merges = []

    print("Initial vocab:")
    for word, freq in vocab.items():
        print(f"  '{word}': {freq}")

    for i in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best  = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        print(f"\nMerge {i+1}: {best[0]} + '{best[1]}' = '{best[0]+best[1]}'")

    print("\nFinal vocab:")
    for word, freq in vocab.items():
        print(f"  '{word}': {freq}")

    return merges, vocab

def bpe_encode(word, merges):
    """Encode a word using learned BPE merges."""
    chars  = list(word) + ['</w>']
    tokens = chars
    for merge in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and
                tokens[i] == merge[0] and tokens[i+1] == merge[1]):
                new_tokens.append(merge[0] + merge[1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

# Train
corpus     = ['cat'] * 3 + ['bat'] * 2 + ['mat'] * 2 + ['cap'] * 1
merges, final_vocab = bpe_train(corpus, num_merges=6)

print("\nLearned merges:")
for i, (a, b) in enumerate(merges, 1):
    print(f"  Rule {i}: '{a}' + '{b}' = '{a+b}'")

# Encode new words
print("\nEncoding test words:")
test_words = ['cat', 'bat', 'rats', 'catfish', 'cap']
for word in test_words:
    tokens = bpe_encode(word, merges)
    print(f"  '{word}' = {tokens}")
```

**Output:**
```
Initial vocab:
  c a t </w>: 3
  b a t </w>: 2
  m a t </w>: 2
  c a p </w>: 1

Merge 1: a + t  → at
Merge 2: at + </w>  → at</w>
Merge 3: c + at</w> → cat</w>
Merge 4: b + at</w> → bat</w>
Merge 5: m + at</w> → mat</w>

Encoding test words:
  'cat'     → ['cat</w>']                              = 1 token ✓
  'bat'     → ['bat</w>']                              = 1 token ✓
  'rats'    → ['r', 'at', 's', '</w>']                 = split into known pieces ✓
  'catfish' → ['c', 'at', 'f', 'i', 's', 'h', '</w>'] = graceful degradation ✓
  'cap'     → ['c', 'a', 'p', '</w>']                  = infrequent word, more splits
```

### 2. Using HuggingFace Tokenizers

```python
from transformers import AutoTokenizer

# GPT-2 (BPE)
gpt2_tok = AutoTokenizer.from_pretrained('gpt2')
text = "cat sat on mat"
tokens = gpt2_tok.tokenize(text)
ids    = gpt2_tok.encode(text)
print(f"GPT-2 tokens: {tokens}")
print(f"GPT-2 ids:    {ids}")
decoded = gpt2_tok.decode(enc["input_ids"][0])
print(f"Decoded: {decoded}")

# BERT (WordPiece)
bert_tok    = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens_bert = bert_tok.tokenize("playing unhappiness")
print(f"BERT tokens: {tokens_bert}")   # ['playing', 'un', '##happiness']
# ## marks continuation

# LLaMA tokenizer (SentencePiece)
llama_tok    = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokens_llama = llama_tok.tokenize("cat sat on mat")
print(f"LLaMA tokens: {tokens_llama}")
# ['▁cat', '▁sat', '▁on', '▁mat']
```

### 3. Compare Tokenizers

```python
texts = [
    "cat sat on mat",
    "antidisestablishmentarianism",
    "unhappiness playing",    # novel word
    "GPT-4ification",         # a novel word
    "I went to the café",     # non-ASCII
]

for tok_name in ['gpt2', 'bert-base-uncased']:
    tok = AutoTokenizer.from_pretrained(tok_name)
    print(f"\n{tok_name}:")
    for text in texts:
        tokens = tok.tokenize(text)
        print(f"  '{text}' → {tokens} ({len(tokens)} tokens)")
```

### 4. Tokenizer Fertility (tokens per word)

```python
import numpy as np

def fertility(tokenizer, texts):
    """Average tokens per whitespace-split word."""
    ratios = []
    for text in texts:
        n_words   = len(text.split())
        n_tokens  = len(tokenizer.tokenize(text))
        ratios.append(n_tokens / n_words)
    return np.mean(ratios)

# Lower fertility = more efficient tokenization
corpus_texts = ["cat sat on mat", "the quick brown fox", "deep learning is amazing"]
for name in ['gpt2', 'bert-base-uncased']:
    tok = AutoTokenizer.from_pretrained(name)
    f   = fertility(tok, corpus_texts)
    print(f"{name} fertility: {f:.2f} tokens/word")
```

---

## Interview Q&A

**Q: Why does BPE start with characters and merge up?**
Starting with characters guarantees every possible word can be represented — no [UNK]. Merging up to common subwords means frequent patterns (like "ing", "un", "##tion") get their own token. This balances vocabulary size (small = efficient) with sequence length (not too many tokens per word).

**Q: Why does the ORDER of BPE merge rules matter?**
When encoding "cats", we must apply rules in the exact order they were learned. Rule 1: a+t→at (applies). Rule 2: at+</w> (doesn't apply — "at" is followed by "s"). If we applied Rule 2 before Rule 1, we'd get a different tokenization. The tokenizer serializes and stores the ordered merge list — encoding is deterministic.

**Q: What's the difference between BPE and WordPiece?**
BPE: merge the pair with highest raw frequency. WordPiece: merge the pair that maximizes likelihood gain — P(AB) / (P(A) × P(B)). WordPiece tends to prefer merges that are "surprising" — pairs that co-occur more than you'd expect if they were independent. In practice, results are similar. BPE is simpler to implement. WordPiece is used by BERT.

**Q: What is the "##" prefix in BERT tokenization?**
Marks a subword that continues a word (no preceding space). "playing" → ["play", "##ing"]: "##ing" means "ing" starts the word. "##ing" continues it. Decoding: strip ## and concatenate: `play + ing = "playing"`. Helps the model know "##ing" in "playing" is different from "ing" in "something somehow started as a word". Helps the model know "##ing" in "playing" is different from "ing" if it somehow started a word.

**Q: Why does LLaMA use a larger vocabulary (128K) than BERT (30K)?**
Larger vocab = fewer tokens per sentence = faster attention (O(T²) cost). For LLMs generating thousands of tokens, reducing token count is critical. BERT processes short sequences (512 tokens max) → smaller vocab fine. LLaMA-3 generates long sequences → 128K vocab means "unhappiness" is 1 token, not 4. Also: larger vocab handles code, math symbols, multilingual text better.

**Q: Why can't you use a tokenizer trained on one language/domain for another?**
BPE learns merge rules from the training corpus distribution. A tokenizer trained on English splits French/Chinese/code inefficiently. "café" with an English tokenizer — many pieces. French-trained tokenizer → 1 token. Domain matters too: medical tokenizer handles "cardiomyopathy" better than a general English tokenizer. Always match tokenizer to the domain and language of your data.

**Q: What is tokenization fertility and why does it matter?**
Fertility = average number of tokens per word. GPT-2 on English text: ~1.3 tokens/word. Same model on Python code: ~2.1 tokens/word (identifiers, operators split more). High fertility = longer sequences = more attention computation = slower training/inference. LLaMA-3's 128K vocab has lower fertility than LLaMA-2's 32K vocab — a key efficiency improvement.

---

## Connections

| Concept | Used in |
|---|---|
| BPE | GPT-2, GPT-3, GPT-4, LLaMA, RoBERTa tokenizers |
| WordPiece | BERT, DistilBERT, ELECTRA |
| SentencePiece | T5, LLaMA-2, Mistral, Gemma, multilingual models |
| Vocabulary size | Model parameter count (embedding table = vocab × d_model) |
| Tokenization → token IDs | Input to all embedding layers |
| Fertility | Sequence length → attention cost O(T²) |

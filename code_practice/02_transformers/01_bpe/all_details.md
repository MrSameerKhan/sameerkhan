# BPE Tokenizer from Scratch — All Details

> Phase 2, Session 1 of the coding practice sequence. The tokenization algorithm used by GPT-2, GPT-3, GPT-4, Claude — built from scratch.

---

## Table of Contents

1. [Objective](#1-objective)
2. [Why Subword Tokenization](#2-why-subword-tokenization)
3. [The BPE Algorithm](#3-the-bpe-algorithm)
4. [Training Step-by-Step](#4-training-step-by-step)
5. [Encoding Step-by-Step](#5-encoding-step-by-step)
6. [Comparison — Char vs Word vs BPE](#6-comparison--char-vs-word-vs-bpe)
7. [Special Tokens](#7-special-tokens)
8. [How to Run](#8-how-to-run)
9. [Expected Outputs](#9-expected-outputs)
10. [Where Real BPE Differs](#10-where-real-bpe-differs)
11. [Files in This Folder](#11-files-in-this-folder)
12. [Next Steps](#12-next-steps)

---

## 1. Objective

Implement Byte-Pair Encoding (BPE) — the subword tokenization algorithm behind every modern LLM. By the end you should be able to:
- Explain why subword tokenization beats char-level and word-level
- Code BPE training from scratch
- Code BPE encoding (rank-based merging)
- Walk through what GPT-2's tokenizer does internally

---

## 2. Why Subword Tokenization

```
Sentence: "I love unsupervised pretraining"

Char-level (Phase 1):
    tokens: I, , l, o, v, e, , u, n, s, u, p, e, r, v, i, s, e, d, , p, r, e, t, r, a, i, n, i, n, g
    Length: 31 tokens. Models read 31 positions. EXPENSIVE.

Word-level:
    tokens: I, love, unsupervised, pretraining
    Length: 4 tokens. Great compression!
    BUT: "unsupervised" → if not in vocab, OOV → model can't handle it.

BPE (subword):
    tokens: I, love, un, super, vised, pre, train, ing
    Length: 8 tokens. 4× shorter than chars.
    Unknown words decompose into known sub-pieces. NO OOV.
```

**BPE finds the sweet spot.** Vocab size is fixed (50K typical), every UTF-8 string can be encoded, average sequence length is ~3-5× shorter than chars.

---

## 3. The BPE Algorithm

Two phases:

### TRAIN (do once on big corpus)

1. Pre-tokenize: split corpus into words
2. Initial vocab = all unique characters (+ end-of-word marker `</w>`)
3. For each word, represent as char-list: "love" → `[l, o, v, e, </w>]`
4. Repeat until vocab reaches target size:
   - a. Count adjacent pairs across all words
   - b. Pick the most frequent pair (call it A, B)
   - c. Add merged token "AB" to vocab
   - d. Apply merge everywhere: `[..., A, B, ...]` → `[..., AB, ...]`
   - e. Save: vocab + list of merges (IN THE ORDER they were learned)

### ENCODE (do every time you tokenize text)

1. Pre-tokenize: split into words
2. For each word:
   - a. Start with char-list + `</w>`
   - b. Find lowest-rank pair (= earliest-learned that applies)
   - c. Apply merge
   - d. Repeat until no more merges apply
3. Look up token IDs in vocab

Decoding is trivial: concat token strings, replace `</w>` with space.

---

## 4. Training Step-by-Step

Suppose corpus = `"low low lowest newest widest"` and target vocab = 8 (just for illustration).

**Initial state:**

```
word_freqs: {"low</w>": 2, "lowest</w>": 1, "newest</w>": 1, "widest</w>": 1}

splits:
  "low</w>":    [l, o, w, </w>]
  "lowest</w>": [l, o, w, e, s, t, </w>]
  "newest</w>": [n, e, w, e, s, t, </w>]
  "widest</w>": [w, i, d, e, s, t, </w>]

vocab: {l, o, w, </w>, e, s, t, n, i, d}  (10 base chars)
```

**Iteration 1** — count pairs (weighted by word freq):
- (l, o): freq(low) + freq(lowest) = 2+1 = 3
- (o, w): 3
- (e, s): 1+1+1 = 3
- (s, t): 1+1+1 = 3

Tied at 3. Pick first alphabetically → `(e, s)`. Merge → `"es"`. Apply everywhere.

**Iteration 2** — continue until vocab hits 8. You can see how the model **learns commonly-occurring character sequences** as new tokens. After 1000 merges, common words become single tokens.

---

## 5. Encoding Step-by-Step

Suppose we trained and learned merges in order:

```
1. (e, s) → es
2. (es, t) → est
3. (l, o) → lo
4. (lo, w) → low
5. (low, </w>) → low</w>
6. (low, est) → lowest
7. ...
```

Now encode `"loweste"`:

```
chars = [l, o, w, e, s, t, e, </w>]

(e, s) is rank 1 → apply first.
chars = [l, o, w, es, t, e, </w>]

(es, t) is rank 2 → apply.
chars = [l, o, w, est, e, </w>]

(l, o) is rank 3 → apply.
chars = [lo, w, est, e, </w>]

(lo, w) is rank 4 → apply.
chars = [low, est, e, </w>]

(low, est) is rank 6 → apply.
chars = [lowest, e, </w>]

No more merges apply (no rule for (lowest, e) or (e, </w>)).

Output: tokens [lowest, e, </w>] → IDs from vocab lookup.
```

**The key insight:** apply merges in the order they were learned (rank-based). Greedy with rank tie-breaking gives a deterministic, reproducible encoding.

---

## 6. Comparison — Char vs Word vs BPE

For the sentence: `"The Premium Checking has 2.5 percent rate."` (41 chars)

| Method | Tokens | Count | Notes |
|---|---|---|---|
| Char | T,h,e,P,r,e,m,i,u,m,C,h,e,c,k,i,n,g,... | ~42 | Tiny vocab, long sequences |
| Word | The, Premium, Checking, has, 2.5, percent, rate, . | 8 | OOV is fatal |
| BPE-300 (this session) | likely: The, Premium, Checking, has, 2.5, percent, rate, . | ~8-12 | Frequent words → 1 token, rare → 2-3 |
| GPT-2 (50K vocab) | The, ĠPremium, ĠChecking, Ġhas, Ġ2, ., 5, Ġpercent, Ġrate, . | 10 | Real-world reference |

(GPT-2 uses Ġ to mark word boundaries instead of `</w>`.)

---

## 7. Special Tokens

Our basic BPE uses only `</w>` (word boundary). Real LLMs add:

| Token | Purpose |
|---|---|
| `<PAD>` | Padding in batches |
| `<BOS>` / `<SOS>` | Sequence start |
| `<EOS>` | Sequence end |
| `<UNK>` | OOV fallback (rare in BPE) |
| `<MASK>` | BERT's masked language modeling |
| `<\|im_start\|>`, `<\|im_end\|>` | Chat markers (ChatML format) |

These are added with reserved IDs (usually 0-N) and never appear in regular text.

---

## 8. How to Run

```bash
cd code_practice/02_transformers/01_bpe
python train.py                    # default vocab_size=300
python train.py --vocab-size 500   # larger vocab

python predict.py --text "Sarah Chen works as Senior Analyst in Risk department."
python predict.py --text "Apply for Personal Loan with up to 50000 dollars." --show-merges
python predict.py --text "Hello unknown_word here"
```

---

## 9. Expected Outputs

### Training output

```
Loaded corpus: 50,037 chars

Corpus: ~8,000 tokens, ~140 unique words
Initial vocab (chars): ~35
Target vocab: 300
Merges needed: ~265

  Merge   1: 's' + '</w>'  → 's</w>'   (count: ~1200)
  Merge   2: 'e' + 'r'     → 'er'      (count: ~1100)
  ...
  Merge  20: 'Premium' + '</w>' → 'Premium</w>'

Final vocab size: 300
Total merges learned: ~265
```

### Sanity check

```
Input:   'The Premium Checking has 2.5 percent rate.'
Tokens:  ['The', '</w>', 'Premium', '</w>', 'Checking', ...]
Decoded: 'The Premium Checking has 2.5 percent rate.'
Compression: ~14 tokens for 41 chars (2.9 chars/token)
```

---

## ✅ Actual Run Results

*(MacBook M1 — CPU, vocab_size=300)*

### Training

```
Loaded corpus: 50,037 chars

Corpus: 50,037 chars, 199 unique words
Initial vocab (chars): 60
Target vocab: 300
Merges needed: 240

  Merge   1: 's'    + '</w>'  → 's</w>'    (count: 1554)  ← plural marker
  Merge   2: 'e'    + '</w>'  → 'e</w>'    (count: 1313)  ← word-final 'e'
  Merge   3: '.'    + '</w>'  → '.</w>'    (count: 1051)  ← sentence-end period
  Merge   4: 'e'    + 'r'     → 'er'       (count: 1009)
  Merge   5: 't'    + '</w>'  → 't</w>'    (count:  944)
  Merge   6: 'i'    + 'n'     → 'in'       (count:  887)
  Merge   7: 'a'    + 'n'     → 'an'       (count:  838)
  Merge   8: 'a'    + 'r'     → 'ar'       (count:  722)
  Merge   9: 'e'    + 'n'     → 'en'       (count:  582)
  Merge  10: '0'    + '</w>'  → '0</w>'    (count:  516)  ← number patterns
  Merge  17: 'o'    + 'l'     → 'ol'       (count:  368)
  Merge  18: 'ol'   + 'l'     → 'oll'      (count:  344)
  Merge  19: 'oll'  + 'ar'    → 'ollar'    (count:  344)
  Merge  20: ...  (Premium</w> appears later as full-word merge)

Final vocab size: 300
Total merges learned: 240
```

### Compositional build-up — beautiful pedagogy

Merges 17-19 compose to build whole words from chars:

```
Merge 17: 'o' + 'l'     → 'ol'
Merge 18: 'ol' + 'l'    → 'oll'
Merge 19: 'oll' + 'ar'  → 'ollar'
```

Then later: `d + ollar → dollar`. You can see BPE building words by recognizing what occurs together.

Late merges = domain-specific full words: `department</w>`, `Loan</w>`, `dollars</w>` — frequent Acme words become single tokens.

---

### Three encoding tests — compression reveals vocab coverage

| Input | Tokens | Chars/token | Verdict |
|---|---|---|---|
| "Sarah Chen works as Senior Analyst in Risk department." | 28 | 1.9 | POOR — names not frequent enough to merge |
| "Apply for Personal Loan with up to 50000 dollars." | 15 | 3.3 | GREAT — Acme-frequent words → 1 token |
| "Hello unknown_word here" | 18 (+1 dropped) | 1.3 | TERRIBLE — OOV → char-level fallback + silent drop |

**Sentence 1 — per-word breakdown:**
```
Sarah      → ['S', 'ar', 'a', 'h</w>']          4 pieces (name, rare)
Chen       → ['Ch', 'en', '</w>']                3 pieces
works      → ['w', 'or', 'k', 's</w>']           4 pieces
as         → ['as</w>']                           1 token ✓
Senior     → ['S', 'en', 'i', 'or</w>']          4 pieces
Analyst    → ['An', 'al', 'y', 's', 't</w>']     5 pieces
in         → ['in</w>']                           1 token ✓
Risk       → ['R', 'is', 'k', '</w>']            4 pieces
department → ['department</w>']                   1 token ✓ (frequent in corpus)
.          → ['.</w>']                            1 token ✓
```

**Sentence 2 — Acme-frequent words compress well:**
```
for        → ['for</w>']      1 token ✓
Loan       → ['Loan</w>']     1 token ✓
with       → ['with</w>']     1 token ✓
to         → ['to</w>']       1 token ✓
50000      → ['50000</w>']    1 token ✓
dollars    → ['dollars</w>']  1 token ✓
```

Pattern: at vocab=300, BPE captures only the most frequent patterns.
Common Acme words like Loan, dollars, 50000 become single tokens.
Names like Sarah, Senior are 3-4 tokens each.

---

### Real Bug Discovered — Silent OOV Drop

**Sentence 3 encoding:**
```
Tokens (19): ['H', 'el', 'l', 'o', '</w>', 'u', 'n', 'k', 'n', 'o', 'w', 'n', '_', 'w', 'or', 'd</w>', 'h', 'er', 'e</w>']
IDs    (18): [21, 149, 45, 48, 13, 54, 47, 44, 47, 48, 56, 47, 56, 185, 74, 41, 63, 61]

19 tokens vs 18 IDs — one MISSING!
Decoded: 'Hello unknownword here'   ← underscore '_' is GONE
```

`_` (underscore) never appeared in the Acme training corpus → not in vocab → `encode()` silently drops it. The decoded string loses the character with no warning.

**This is why GPT-2 onwards uses BYTE-LEVEL BPE.** It operates on raw UTF-8 bytes (256 possible values) instead of chars. Every possible byte is guaranteed to be in the vocab. No silent data loss.

**Interview gold:** "Why does GPT-2 use byte-level BPE instead of char-level?" → prevents the silent OOV drop where unfamiliar characters disappear during encoding.

---

### Lessons captured

| Concept | What you saw in the run |
|---|---|
| BPE learns frequency patterns | Top merges are super-common endings (`s</w>`, `er`, `en`) |
| Merges compose | `o → ol → oll → ollar` → `dollar` in 4 steps |
| Vocab size is a compression knob | 300 vocab gives 1.3-3.8 chars/token depending on domain match |
| OOV silently drops with char-level BPE | `_` disappeared from decode |
| Why byte-level BPE was invented | The silent drop is unacceptable in production |
| Frequency determines token granularity | `Loan` (frequent) = 1 token, `Sarah` (rare) = 4 tokens |

---

## 10. Where Real BPE Differs

Our simple BPE is the core algorithm. Production tokenizers add:

| Feature | Real BPE (GPT-2/3/4) |
|---|---|
| Byte-level fallback | Operates on raw UTF-8 bytes, not chars → ANY string encodable |
| Regex pre-tokenization | `\p{L}+` splits on punctuation/spaces first |
| Special tokens | `<\|endoftext\|>`, `<\|fim_prefix\|>`, etc. |
| Reverse merges | Decoder can reverse-apply merges |
| Caching | Encoding is cached per input string |
| 50K+ vocab | Way more than our 300 |

The CORE algorithm — count pairs, merge, learn ranks, apply in order — is identical.

---

## 11. Files in This Folder

| File | Purpose |
|---|---|
| `model.py` | `BPE` class: train, encode, decode, tokens, save, load |
| `train.py` | CLI to train on Acme corpus + save tokenizer JSON |
| `predict.py` | CLI to tokenize any text with the trained BPE |
| `all_details.md` | This document |
| `checkpoints/bpe.json` | Saved tokenizer (vocab + merges) |

---

## 12. Next Steps

### Session 2 — Scaled Dot-Product Attention

The math at the heart of transformers. We'll implement:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

in pure NumPy with masking. After Bahdanau attention (Phase 1 Session 8), this is a small generalization: multiple queries, dot-product score, separate K/V.

### Why we did BPE first

Every transformer session from here on will need to tokenize text. Now you have a tokenizer that:
- Produces small, learnable vocab
- Handles any input (no OOV with byte-level extension)
- Mimics how real LLMs see text

When Phase 4 fine-tunes an LLM, you'll see this exact pattern — load tokenizer, encode dataset, train on token IDs.

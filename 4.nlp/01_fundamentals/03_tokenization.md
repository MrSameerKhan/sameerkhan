# 03 — Tokenization

> Tokenization converts raw text → a sequence of integer IDs that a model can process. The algorithm determines the vocabulary size, how rare/unknown words are handled, and how many tokens a given text consumes (directly affecting model cost and context window usage).

---

## Tokenization Algorithms

| Algorithm | Used by | Unit | OOV handling |
|---|---|---|---|
| Whitespace / word-level | Old NLP, baseline | Whole words | `<UNK>` token |
| BPE (Byte-Pair Encoding) | GPT-2, GPT-3, GPT-4, LLaMA, Mistral | Subword | Never OOV (byte fallback) |
| WordPiece | BERT, DistilBERT, ALBERT | Subword | `[UNK]` for missed chars |
| SentencePiece + BPE | LLaMA, T5, XLNet | Subword (raw bytes/chars) | Never OOV |
| SentencePiece + Unigram | T5, mBART | Subword (probabilistic) | Never OOV |
| Tiktoken (BPE variant) | GPT-3.5/4, Claude | Byte-level BPE | Never OOV |

---

## Byte-Pair Encoding (BPE)

### Training Algorithm

```
Start: vocabulary = all individual characters (26 letters + punctuation + digits)
Repeat until vocab_size reached:
  1. Count all adjacent pair frequencies in corpus
  2. Merge the most frequent pair into a new token
  3. Add merged token to vocabulary
  4. Replace all occurrences of that pair in corpus

Example (small corpus):
  "low low lower lower newest newest"

  Initial: l o w _ l o w _ l o w e r _ l o w e r _ n e w e s t _ n e w e s t _
  Merge 1: "e s" → "es"  (high frequency: new-es-t)
  Merge 2: "es t" → "est"
  Merge 3: "l o" → "lo"
  Merge 4: "lo w" → "low"
  ...
```

### Tokenization (Inference)

```python
# BPE tokenizes by applying learned merges greedily from left to right
text = "lowest"
# Start: ['l', 'o', 'w', 'e', 's', 't']
# Apply merge "e s" → "est": ['l', 'o', 'w', 'e', 's', 't']
# Apply merge "es t" → "est": ['l', 'o', 'w', 'es', 't']
# Apply merge "l o" → "lo":   ['lo', 'w', 'est']
# Apply merge "lo w" → "low": ['low', 'est']
# Final: ['low', 'est']  →  IDs: [3456, 892]
```

**Property:** words that share prefixes/suffixes share token IDs — better generalization.

---

## WordPiece (BERT)

Similar to BPE but uses **likelihood** to decide merges instead of raw frequency:

```
BPE:       merge if count(AB) is highest
WordPiece: merge if count(AB) / (count(A) × count(B)) is highest
             ← prefers merges that are "surprising" (AB much more common than A×B)
```

**Prefix convention:** subword tokens get `##` prefix to indicate they continue a word:

```python
from transformers import BertTokenizer

tok    = BertTokenizer.from_pretrained("bert-base-uncased")
tokens = tok.tokenize("unbelievably fast")
# ['un', '##bel', '##iev', '##ably', 'fast']

ids = tok.encode("unbelievably fast", add_special_tokens=True)
# [101, 4085, 18137, 15313, 3251, 3435, 102]
# CLS  un  ##bel  ##iev  ##ably  fast  SEP
```

---

## SentencePiece

Unlike BPE/WordPiece (which tokenize pre-split words), SentencePiece **treats the raw byte stream as input** — no whitespace pre-tokenization needed. Used by LLaMA, T5, XLNet.

**Advantages:**
- Language-agnostic (no whitespace assumption = works for Chinese, Japanese, Thai)
- Spaces encoded as `▁` (U+2581): "Hello world" = ["▁Hello", "▁world"]
- Reversible: can reconstruct exact original string from tokens

```
LLaMA-2 tokenizer: SentencePiece BPE,   vocab_size=32,000
T5 tokenizer:      SentencePiece Unigram, vocab_size=32,100
```

```python
from transformers import LlamaTokenizer

tok    = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokens = tok.tokenize("unbelievably fast")
# ['▁un', '▁bel', '▁iev', '▁ably', '▁fast']   (Different from BERT's WordPiece)

ids = tok.encode("Hello, world!", add_special_tokens=True)
# [1, 15043, 29892, 3186, 29991]
# BOS  Hello    ,    world    !
```

---

## Tiktoken (GPT-3.5/4)

OpenAI's byte-level BPE — encodes raw bytes so it **never fails** even for emoji, code, or arbitrary Unicode:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Hello, world! 🌍")
# [9906, 11, 4435, 0, 234, 255]   (7 tokens)

token_count = len(enc.encode(long_text))   # count tokens for cost estimation
```

**Rule of thumb:** 1 token ≈ 4 characters ≈ 0.75 words (English).

---

## Special Tokens

| Token | BERT | GPT | LLaMA | Purpose |
|---|---|---|---|---|
| CLS / BOS | [CLS] (101) | N/A | `<s>` (1) | Start of sequence |
| SEP / EOS | [SEP] (102) | `<\|endoftext\|>` | `</s>` (2) | End / separator |
| PAD | [PAD] (0) | N/A | N/A | Padding to equal length |
| UNK | [UNK] (100) | rare | rare | Unknown token |
| MASK | [MASK] (103) | N/A | N/A | BERT MLM masking |

**BERT encoding for sentence pairs (NLI, QA):**
```
[CLS] sentence_A [SEP] sentence_B [SEP]
↑ position IDs: 0, 1, 2, ..., len_A+1, len_A+2, ...
↑ token_type_ids: 0, 0, ..., 0, 0, 1, 1, ..., 1
```

---

## Dry Run — Encoding and Decoding

### BERT:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "The quick brown fox"
enc  = tok(text, return_tensors="pt")

# input_ids:      tensor([[101, 1996, 4248, 2829, 4419, 102]])
#                          CLS  the   quick  brown  fox   SEP
# attention_mask: tensor([[ 0,    1,    1,    1,    1,   1]])
# token_type_ids: tensor([[ 0,    0,    0,    0,    0,   0]])

decoded = tok.decode(enc["input_ids"][0])
# "[CLS] the quick brown fox [SEP]"
```

### GPT-2:

```python
tok = AutoTokenizer.from_pretrained("gpt2")
enc = tok("The quick brown fox", return_tensors="pt")
# input_ids: tensor([[464, 2068, 7586, 21831]])
# No CLS/SEP — decoder-only models don't need them
```

---

## Token Count and Cost

| Model | Tokenizer | Vocab | Cost per 1M tokens |
|---|---|---|---|
| GPT-4o | tiktoken cl100k | 100,277 | ~$5 input / $15 output |
| GPT-3.5-turbo | tiktoken cl100k | 100,277 | ~$0.50 / $1.50 |
| Claude 3.5 Sonnet | Internal BPE | ~100K | ~$3 / $15 |
| LLaMA-2-7B | SentencePiece | 32,000 | N/A (self-hosted) |

**Token budget estimation:**

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
system_prompt = "You are a helpful assistant..."
user_message  = "Explain transformers in detail..."

n_tokens = len(enc.encode(system_prompt)) + len(enc.encode(user_message))
cost_estimate = n_tokens / 1_000_000 * 5.0   # $5 per 1M input tokens
```

---

## Language-Specific Behavior

**Chinese / Japanese:** character-based scripts → BPE tokenizes each character or multi-character units:
```
"北京" (Beijing) = ['北', '京']           # 2 tokens
"transformer"   = ['transform', 'er']    # 2 tokens in GPT-4
```

**Code:** lowercase identifiers tokenize efficiently. `snake_case_variable` gets many splits:
```python
# Variable names in Python:
"hidden_size"   = ['hidden', '_', 'size']               # 3 tokens
"HIDDEN_SIZE"   = ['H', 'ID', 'DEN', '_', 'S', 'IZE']  # 5 tokens (less efficient)
```

**Rule:** lowercase + common words → fewer tokens → cheaper inference.

---

## Truncation and Padding

```python
tok = AutoTokenizer.from_pretrained("bert-base-uncased")

# Batch encoding with padding + truncation
batch = tok(
    ["short text", "a much longer piece of text that needs truncation to fit"],
    padding=True,         # pad shorter to match longest
    truncation=True,      # truncate to max_length
    max_length=512,
    return_tensors="pt"
)
# input_ids:      [2, 512] — both truncated/padded to same length
# attention_mask: [2, 512] — 1 for real tokens, 0 for padding
```

**Attention mask is critical:** prevents attention from attending to padding tokens.

---

## Vocabulary Size Trade-offs

| Vocab size | Tokens per word | Pros | Cons |
|---|---|---|---|
| Small (1K) | Many (slow processing) | Tiny embedding table | Long sequences, high OOV |
| Medium (32K) | ~1.3 | Balance | Some rare word splits |
| Large (100K+) | ~1.0 for common words | Fewer tokens per text | Large embedding table |

GPT-4 uses 100,277 vocab — rarely needs to split common English words. LLaMA-2 uses 32,000 — more splits for rare words.

---

## Gotchas

**Tokenizers are not interchangeable.** A BERT tokenizer will fail on GPT-2 IDs. Always load tokenizer from the same checkpoint as the model: `AutoTokenizer.from_pretrained("bert-base-uncased")` matches `AutoModel.from_pretrained("bert-base-uncased")`.

**[CLS] is position 0 for BERT.** When extracting sentence representation from BERT, use `output.last_hidden_state[:, 0, :]` for [CLS]. For SBERT, use mean pooling over all non-padding positions.

**Decoder-only models (GPT, LLaMA) don't have [CLS]/[SEP].** Sequence ends when model generates `<|endoftext|>` or `</s>`. Don't add SEP manually.

**Long documents — chunking required.** BERT has 512-token max. For long texts, chunk with overlap (stride=64 tokens) and aggregate results. Rule of thumb: 1 token ≈ 4 chars (64 tokens). A 4096-token context window holds ~3000 words, not 4096.

**Token count ≠ word count.** 100 words ≈ 130 tokens. A 4096-token context window holds ~3000 words, not 4096.

---

## Interview Q&A

**Q: Why does BPE handle unknown words better than word-level tokenization?**
BPE breaks unknown words into known subword pieces using the merge rules learned during training. In the worst case, it falls back to individual characters or bytes — so every possible input string has a valid representation. Word-level tokenizers map any OOV word to a single `<UNK>` token, losing all semantic information. "cryptocurrency" might be OOV for a 2015 vocabulary, but BPE splits it into "crypto" + "currency" — both in vocabulary with meaningful embeddings.

**Q: Why does BERT use [CLS] and [SEP] tokens?**
[CLS] is a special token added at the start of every sequence. In BERT's pre-training, the NSP (next sentence prediction) head reads the final [CLS] embedding to predict whether two sentences are consecutive. [CLS] embedding aggregates full-sequence information and [SEP] marks boundaries between sentences, enabling BERT to distinguish the two inputs in tasks like NLI or QA. GPT-style decoder-only models don't need these — they process single sequences autoregressively.

**Q: What is the difference between BPE and WordPiece?**
Both learn subword vocabularies via merging, but differ in the merge criterion. BPE merges the most frequent character pair: merge if count(AB) is maximum. WordPiece merges the pair with highest **likelihood gain**: merge if count(AB) / (count(A) × count(B)) is maximum — favoring pairs that are "surprising" given their individual frequencies. This makes WordPiece more principled but slightly more complex. In practice, both produce similar tokenizations for English. BPE is more common for LLMs; WordPiece is primarily used by Google's BERT family.

---

## Connections

| Topic | File |
|---|---|
| Full tokenization dry-run with BERT + GPT | `04_tokenization_end_to_end.md` |
| Tokenization in transformer pre-training | `5.transformers/01_fundamentals/03_tokenization.md` |
| Token count budgeting in agents | `8.agents/01b_agents_end_to_end.md` |
| Special tokens in BERT forward pass | `5.transformers/02_models/05_bert_end_to_end.md` |

---

## Key Takeaway

**BPE** (GPT, LLaMA) merges most-frequent pairs — no OOV. **WordPiece** (BERT) merges highest-likelihood pairs — `##` prefix notation. **SentencePiece** (LLaMA, T5) operates on raw bytes — language-agnostic. Rule of thumb: 1 token ≈ 4 chars ≈ 0.75 words. Always match tokenizer to model checkpoint. Attention mask = 0 for padding — model ignores padded positions.

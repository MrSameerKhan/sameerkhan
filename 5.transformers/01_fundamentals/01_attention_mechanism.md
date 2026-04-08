# Attention Mechanism

## Quick Reference
| Concept | Formula | Purpose |
|---------|---------|---------|
| Scaled dot-product attention | softmax(QKᵀ/√dₖ)V | Core attention operation |
| Multi-head attention | Concat(head₁...headₕ)Wᴼ | Attend to multiple subspaces |
| Self-attention | Q=K=V=X | Tokens attend to each other |
| Cross-attention | Q=decoder, K=V=encoder | Decoder queries encoder output |
| Causal mask | mask future positions with -∞ | Autoregressive generation |

**The insight:** Instead of fixed context (RNN hidden state bottleneck), every token directly attends to every other token in O(1) steps — at the cost of O(n²) memory.

---

## Core Concepts

### From Seq2Seq Attention to Self-Attention

**Seq2Seq (Bahdanau) attention** — decoder queries encoder hidden states:
```
Problem: fixed-size context vector loses information for long sequences
Solution: at each decoder step, compute weighted sum over ALL encoder states

eₜᵢ = score(sₜ₋₁, hᵢ)       alignment score (additive / dot-product)
αₜᵢ = softmax(eₜᵢ)           attention weights (sum to 1)
cₜ  = Σ αₜᵢ hᵢ               context vector
```

**Self-attention** — tokens attend to each other within the same sequence:
```
Every token is simultaneously a query, a key, and a value.
"The bank on the river bank" — "bank" (token 2) attends to
"river" (token 4) to resolve ambiguity. No recurrence needed.
```

---

### Scaled Dot-Product Attention

**Formula:**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

Q: queries  [seq_len, dₖ]   — what am I looking for?
K: keys     [seq_len, dₖ]   — what do I contain?
V: values   [seq_len, dᵥ]   — what do I return if matched?
dₖ: key dimension (used for scaling)
```

**Why √dₖ scaling?**
```
Without scaling: for large dₖ, dot products QKᵀ grow large in magnitude
→ softmax saturates (outputs near 0 or 1) → gradients vanish

Variance of Q·K = dₖ × Var(q) × Var(k) ≈ dₖ (if q,k ~ N(0,1))
Dividing by √dₖ → variance ≈ 1 → softmax stays in non-saturated region
```

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, heads, seq_q, d_k]
    K: [batch, heads, seq_k, d_k]
    V: [batch, heads, seq_k, d_v]
    mask: [batch, 1, seq_q, seq_k] — True where attention should be blocked
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [batch, heads, seq_q, seq_k]

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_q, seq_k]
    attn_weights = F.dropout(attn_weights, p=0.1, training=True)

    output = torch.matmul(attn_weights, V)    # [batch, heads, seq_q, d_v]
    return output, attn_weights
```

---

### Multi-Head Attention

**Why multiple heads?**
```
Single attention head: one "view" of relationships
Multi-head: h parallel attention heads, each in a different learned subspace

Head 1 might focus on syntactic dependencies
Head 2 might focus on coreference
Head 3 might focus on positional proximity
...each learns what's useful
```

**Formula:**
```
headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)

MultiHead(Q,K,V) = Concat(head₁,...,headₕ) · Wᴼ

Where:
  WᵢQ ∈ ℝ^(d_model × d_k)
  WᵢK ∈ ℝ^(d_model × d_k)
  WᵢV ∈ ℝ^(d_model × d_v)
  Wᴼ  ∈ ℝ^(h·d_v × d_model)

Typical: d_model=512, h=8 → d_k = d_v = 512/8 = 64
Total params per MHA: 4 × d_model² (Wq, Wk, Wv, Wo)
```

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Project and reshape to [batch, heads, seq, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(torch.softmax(scores, dim=-1))

        # Combine heads
        output = torch.matmul(attn, V)                                   # [B, H, S, d_k]
        output = output.transpose(1, 2).contiguous()                     # [B, S, H, d_k]
        output = output.view(batch_size, -1, self.num_heads * self.d_k)  # [B, S, d_model]

        return self.W_o(output)
```

---

### Positional Encoding

**Problem:** Self-attention is permutation-invariant — "cat sat mat" and "mat sat cat" produce same attention patterns (without position info).

**Solution 1: Sinusoidal (original Transformer)**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Properties:
- Fixed (not learned)
- Unique for each position
- Relative distances computable via trig identities: PE(pos+k) = f(PE(pos))
- Extrapolates to unseen sequence lengths
```

```python
import torch
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute encodings once
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
```

**Solution 2: Learned Absolute (BERT, GPT)**
```python
# Simply an embedding table — position 0..max_len each gets a learned vector
self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
positions = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
pos_emb = self.position_embedding(positions)
```

**Solution 3: Rotary Position Embedding / RoPE (LLaMA, GPT-NeoX)**
```
Key insight: encode position as rotation in complex space
  x_rotated = x · e^(iθm)   where m = position, θ = frequency

Properties:
- Relative positions naturally encoded: dot(Q_m, K_n) depends on (m-n)
- Better length generalization than absolute PE
- Used in modern LLMs: LLaMA, Mistral, Falcon

Implementation: rotate pairs of Q and K dimensions before attention
```

**Solution 4: ALiBi (Attention with Linear Biases)**
```
Don't add position to embeddings at all.
Instead, add a linear bias to attention scores:

scores = QKᵀ/√dₖ - m·|i-j|

where m is a head-specific slope, |i-j| is token distance
→ Closer tokens get less penalty; farther tokens get more
→ Strong extrapolation to longer sequences
→ Used in MPT, BLOOM
```

---

### Attention Masks

**Padding mask (encoder):** Ignore [PAD] tokens
```python
# attention_mask from tokenizer: 1=real token, 0=padding
# Convert to mask for attention scores
padding_mask = (attention_mask == 0)  # True where padding
# Shape: [batch, 1, 1, seq_len] to broadcast over heads and query positions
padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
scores = scores.masked_fill(padding_mask, float('-inf'))
```

**Causal mask (decoder / GPT):** Prevent attending to future tokens
```python
def causal_mask(seq_len, device):
    """Upper triangular matrix of True (block future positions)."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask  # [seq_len, seq_len]

# Token i can attend to tokens 0..i only
# Position 0: can attend to [0]
# Position 1: can attend to [0, 1]
# Position 2: can attend to [0, 1, 2]
# ...
```

---

### Attention Complexity

```
Standard self-attention:
  Time:   O(n² · d)   — n² attention scores, each d-dimensional
  Memory: O(n²)       — must store full attention matrix

Bottleneck: the n² attention matrix for long sequences
  n=512:  512²  = 262K    → fine
  n=2048: 2048² = 4M      → manageable
  n=8192: 8192² = 67M     → GPU memory pressure
  n=100K: 100K² = 10B     → impossible without optimization

Solutions:
  Flash Attention:   O(n²) time, O(n) memory — tiling, no materialization
  Sparse Attention:  O(n√n) — attend to local window + strided global
  Linear Attention:  O(n) — kernel trick to avoid explicit softmax matrix
  Sliding Window:    O(n·w) — Longformer, local window of size w
```

---

### Flash Attention

```python
# Flash Attention: compute attention in tiles to avoid materializing O(n²) matrix
# Trades compute for memory: recomputes attention during backward pass

# Usage in HuggingFace (automatic if flash-attn installed):
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",  # or "sdpa" (PyTorch scaled_dot_product_attention)
    torch_dtype=torch.bfloat16
)

# PyTorch native scaled_dot_product_attention (uses Flash Attention when available)
output = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True  # causal mask applied automatically
)
```

---

### Attention Visualization (Debugging)
```python
from bertviz import head_view
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

inputs = tokenizer.encode_plus("The bank on the river bank", return_tensors='pt')
outputs = model(**inputs)

attention = outputs.attentions  # tuple of [batch, heads, seq, seq] per layer
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
head_view(attention, tokens)  # interactive visualization in notebook
```

---

## When to Use What

| Attention Type | Use Case |
|----------------|----------|
| Self-attention (bidirectional) | Encoding: BERT, understanding tasks |
| Causal self-attention | Decoding: GPT, text generation |
| Cross-attention | Seq2seq: T5, BART, machine translation |
| Sparse/Sliding window | Long documents: Longformer, BigBird |
| Flash Attention | Any transformer for memory efficiency |
| RoPE positional encoding | Modern LLMs (LLaMA, Mistral) |
| ALiBi | Long-context models (MPT, BLOOM) |

---

## Gotchas

**Softmax of all -inf:** If all keys are masked (e.g., full padding row), softmax(-inf) = NaN. Handle by clamping or detecting all-masked rows.

**Attention weights ≠ importance:** High attention weight to a token doesn't necessarily mean that token is causally important. Gradient-based methods (Integrated Gradients) are more reliable for attribution.

**Multi-head with small d_k:** If d_model=64 and h=8, then d_k=8. Too small → each head has limited expressiveness. In practice d_k ≥ 32.

**Memory is the bottleneck, not FLOPs:** For long sequences, the O(n²) attention matrix dominates GPU memory. Use Flash Attention for sequences >1K tokens.

**Cross-attention key/value caching:** In generation, KV cache stores K,V for all previous tokens. Memory grows linearly with sequence length × batch size. Critical for production LLM serving.

---

## Interview Q&A

**Q: Why does attention use three separate Q, K, V projections instead of one?**
A: Using separate projections gives the model flexibility to learn different representations for "what am I looking for" (Q), "what can I be matched with" (K), and "what information do I provide when matched" (V). If Q=K=V (no projection), the model can't independently control matchability and information content. Empirically, separate projections perform significantly better.

**Q: Explain the scaling factor √dₖ. What happens without it?**
A: The dot product Q·Kᵀ has variance proportional to dₖ (dimension of keys), assuming Q and K are unit-variance. For large dₖ (e.g., 64), the dot products grow large, pushing softmax into saturation regions where gradients near zero. Dividing by √dₖ normalizes the variance back to ~1, keeping softmax in a stable, non-saturated region during training.

**Q: What's the computational complexity of self-attention and why is it a problem?**
A: O(n²·d) time and O(n²) memory where n is sequence length. For n=512 (BERT), this is fine. For n=32K (long documents, code), the attention matrix has 1 billion elements — infeasible. Solutions: Flash Attention reduces memory to O(n) via tiling (no materialization); sparse attention patterns (Longformer's sliding window) reduce to O(n·w); linear attention approximations reduce to O(n).

**Q: What's the difference between self-attention and cross-attention?**
A: In self-attention, Q, K, V all come from the same sequence — tokens attend to each other within one sequence. In cross-attention, Q comes from one sequence (e.g., decoder state) while K and V come from another (e.g., encoder output). This is how encoder-decoder transformers (T5, BART) allow the decoder to "look at" the encoded source while generating.

**Q: What is a KV cache and why does it matter for inference?**
A: During autoregressive generation, at each step the model computes attention over all previous tokens. Without caching, this recomputes K and V for the entire prefix at every step — O(n²) total cost. With KV cache, we store the K and V tensors from each layer for all previous tokens and only compute the new token's Q. Reduces per-step cost to O(n) with O(n·d·layers) memory. Essential for production LLM serving — without it, generation is prohibitively slow.

---

## Connections
- **RNN to Attention (NLP/sequence_models/01):** Bahdanau attention is the precursor; transformers generalize it to full self-attention
- **Transformer Architecture (fundamentals/02):** Attention is the core building block; architecture wraps it with FFN, LayerNorm, residuals
- **BERT Family (models/01):** Bidirectional self-attention encoder stacks
- **GPT Family (models/02):** Causal (masked) self-attention decoder stacks
- **Efficient Transformers (models/04):** Flash Attention, sparse attention, linear attention
- **ViT (CV/architectures):** Same multi-head self-attention on image patches

## Key Takeaway
Attention replaces the RNN's sequential hidden state with direct token-to-token connections. The core operation: QKᵀ/√dₖ computes similarity scores, softmax converts to weights, weighted sum of V produces output. Multi-head attention runs this in h parallel subspaces. Everything else in transformers (BERT, GPT, T5) is built on top of this one operation. The n² complexity is the fundamental limitation — Flash Attention is the practical solution.

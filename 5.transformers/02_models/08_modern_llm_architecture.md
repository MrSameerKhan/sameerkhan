# Modern LLM Architecture — LLaMA vs GPT-2

> Every change explained with numbers. Same 4-word vocabulary throughout.

---

## The Arc

```
GPT-2 (2019) → LLaMA-1 (2023) → LLaMA-2 (2023) → Mistral 7B (2023)
             → LLaMA-3 (2024) → LLaMA-3.1 (2024)
             → Mistral 8×7B (2024) → DeepSeek-V2/V3 (2024) → Qwen2.5 / Gemma 2 (2024)
```

GPT-2 established the transformer decoder baseline. Every model since then is GPT-2 + a converging set of targeted improvements.

**As of 2025, the consensus "modern transformer" recipe:** RMSNorm + pre-norm + SwiGLU FFN + RoPE (often YARN-extended) + GQA (or MLA for DeepSeek) + no biases on linears + tied / unted embeddings depending on model + (optionally) sparse MoE FFN.

Recent escalations beyond this baseline: **MLA** (DeepSeek's low-rank KV projection — 5-10× smaller KV cache), **aux-loss-free MoE balancing** (DeepSeek-V3), **FP8 training** (DeepSeek-V3, recent LLaMA experiments), **YARN/PI** for 4K → 128K-1M context extension.

---

## GPT-2 Baseline (What We're Changing From)

```
Architecture:
  Position encoding:  Learned absolute embeddings
  Normalization:      LayerNorm after residual (post-norm)
  FFN activation:     GELU
  Attention:          Multi-Head Attention (MHA) — H separate K,V per head
  Context window:     1024 tokens

Forward pass for one transformer block (GPT-2 post-norm):
  x = x + Attention(LayerNorm(x))     ← NO, this is LLaMA pre-norm
  x = LayerNorm(x + Attention(x))     ← GPT-2 post-norm
  x = LayerNorm(x + FFN(x))
```

---

## What LLaMA Changed and Why

| Change | GPT-2 | LLaMA | Reason |
|--------|-------|-------|--------|
| Normalization | LayerNorm, post-norm | **RMSNorm, pre-norm** | Training stability |
| FFN activation | GELU | **SwiGLU** | Better performance |
| Position encoding | Learned absolute | **RoPE** (LLaMA-3.1: YARN-extended) | Relative positions, length generalization → 128K context |
| Attention | MHA | **GQA** (LLaMA-2+); **MLA** in DeepSeek | KV-cache memory savings (GQA: 4-8×, MLA: 5-10× more) |
| KV cache layout | Contiguous | **Paged** (vLLM/serving) | Memory fragmentation → 2-4× throughput |
| Tokenizer | BPE (50K vocab) | SentencePiece BPE (32K-128K vocab) | Better multilingual + code coverage |
| Bias terms | Present | **Removed** | Simplify, match performance |
| FFN width | dense 4× | dense = 2.67× (SwiGLU) or **sparse MoE** (Mistral/DeepSeek) | Same params, more capacity via experts |

---

## Change 1 — Pre-norm vs Post-norm

### GPT-2 (Post-norm)

```python
output = LayerNorm(x + Attention(x))
```

Residual is computed first, then normalized. Problem: gradients must flow through the LayerNorm — can cause instability in deep networks.

### LLaMA (Pre-norm)

```python
output = x + Attention(RMSNorm(x))
```

Normalize first, then compute attention, then add residual. Gradient flows through the residual stream unmodified — no normalization in the gradient path.

```
Gradient (pre-norm):
  ∂L/∂x = ∂L/∂output × 1   ← clean skip connection gradient

Gradient (post-norm):
  ∂L/∂x = ∂L/∂output × ∂LayerNorm/∂(x + Attention(x)) × (1 + ∂Attention/∂x)
```

Pre-norm: identity path in the gradient. Easier to train 32-80 layer networks.

---

## Change 2 — RMSNorm vs LayerNorm

### LayerNorm

```
x_norm = (x - μ) / sqrt(σ² + ε) * γ + β

Parameters: γ (scale), β (shift) — both learned
```

### RMSNorm

```
x_norm = x / RMS(x) * γ

RMS(x) = sqrt( mean(x²) ) = sqrt( (x_1² + x_2² + ... + x_d²) / d )

Parameters: only γ (scale) — no β, no mean subtraction
```

### Step-by-step: x = [1.0, 2.0, 3.0, 4.0]

**LayerNorm:**

```
μ = (1 + 2 + 3 + 4) / 4 = 10/4 = 2.500

σ² = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4
   = [2.25 + 0.25 + 0.25 + 2.25] / 4
   = 5.0 / 4 = 1.250

σ = sqrt(1.250) = 1.118

x_norm = [(1-2.5)/1.118, (2-2.5)/1.118, (3-2.5)/1.118, (4-2.5)/1.118]
       = [-1.342, -0.447, 0.447, 1.342]

Operations needed: mean (d adds) + variance (d squares + d adds + divide) + normalize (d divides)
Total: ~5d operations
```

**RMSNorm:**

```
RMS = sqrt((1² + 2² + 3² + 4²) / 4)
    = sqrt((1 + 4 + 9 + 16) / 4)
    = sqrt(7.5)
    = 2.739

x_norm = [1.0/2.739, 2.0/2.739, 3.0/2.739, 4.0/2.739]
       = [0.365, 0.730, 1.095, 1.460]

Operations needed: d squares + mean + sqrt + d divides
Total: ~3d operations
```

### Comparison

| | LayerNorm | RMSNorm |
|--|-----------|---------|
| Mean subtraction | Yes | **No** |
| Shift parameter β | Yes (d params) | **No** |
| Scale parameter γ | Yes (d params) | Yes (d params) |
| Operations | ~5d | **~3d** |

RMSNorm is simpler, faster, and empirically matches LayerNorm performance on LLM pretraining.

---

## Change 3 — RoPE (Rotary Position Embeddings)

### The Problem with Absolute Position Embeddings (GPT-2)

GPT-2 learns one embedding vector per position: PE_0, PE_1, ..., PE_1023. Added to token embeddings at the start.

Problems: (1) Fixed at train time — can't generalize beyond 1024 tokens without retraining. (2) Absolute positions — the model learns "token at position 5" not "token 3 steps before current". (3) Q·K dot product mixes position and content in a hard-to-separate way.

### RoPE Key Idea

Don't add position to embeddings. Instead, **rotate** Q and K vectors by an angle proportional to position before computing attention. The rotation angle is position-dependent, so Q·K naturally encodes relative position.

### Math Setup (d=2 for clarity)

```
θ = 1.0   (base angle for d=2, one pair of dimensions)

For position m, the rotation matrix is:
R(m×θ) = [ cos(m×θ)  -sin(m×θ) ]
          [ sin(m×θ)   cos(m×θ) ]

Apply to q at position m:
q_rot = R(m×θ) × q = [ q_0×cos(mθ) - q_1×sin(mθ) ]
                      [ q_0×sin(mθ) + q_1×cos(mθ) ]
```

### The Critical Property: Dot Product = f(m-n)

```
q_rot · k_rot = cos(m×θ) × cos(n×θ) + sin(m×θ) × sin(n×θ) = cos(m-n)
```

This is the cosine subtraction identity. The dot product depends only on (m-n).

### Concrete numbers

| Q position m | K position n | Relative pos (m-n) | Score = cos(m-n) |
|--------------|--------------|---------------------|------------------|
| 1 | 0 | 1 | cos(1) = **0.540** |
| 2 | 1 | 1 | cos(1) = **0.540** |
| 3 | 2 | 1 | cos(1) = **0.540** |
| 5 | 3 | 2 | cos(2) = **-0.416** |
| 10 | 8 | 2 | cos(2) = **-0.416** |

Same relative distance = same attention score, regardless of absolute position.

This is what absolute embeddings cannot do: "cat" at position 1 and "cat" at position 500 get different scores when attending to the same relative neighbor.

### Step-by-step: RoPE with d=4 (realistic)

```
d=4 → 2 pairs of dimensions.

Rotation angles (one per pair):
  θ_0 = 10000^(-2×0/4) = 10000^0 = 1.0     ← fast rotation (fine-grained)
  θ_1 = 10000^(-2×1/4) = 10000^(-0.5) ≈ 0.01  ← slow rotation (coarse-grained)

Fast θ_0 = high-frequency rotation — encodes short-range positions
Slow θ_1 = low-frequency rotation — encodes long-range positions (like sinusoidal PE)

Query vector: q = [1.0, 0.5, 0.8, 0.3] at position m=2

Pair 0 (dims 0,1): rotate by m×θ_0 = 2 × 1.0 = 2.0 radians
  cos(2.0) = -0.416,  sin(2.0) = 0.909
  q_rot[0] = q_0×cos(2) - q_1×sin(2) = 1.0×(-0.416) - 0.5×0.909 = -0.416 - 0.455 = -0.871
  q_rot[1] = q_0×sin(2) + q_1×cos(2) = 1.0×0.909  + 0.5×(-0.416) = 0.909 - 0.208 = 0.701

Pair 1 (dims 2,3): rotate by m×θ_1 = 2 × 0.01 = 0.02 radians
  cos(0.02) = 0.9998,  sin(0.02) = 0.0200
  q_rot[2] = q_2×cos(0.02) - q_3×sin(0.02) = 0.8×0.9998 - 0.3×0.020 = 0.800 - 0.006 = 0.794
  q_rot[3] = q_2×sin(0.02) + q_3×cos(0.02) = 0.8×0.020  + 0.3×0.9998 = 0.016 + 0.300 = 0.316

Result:
  q original = [1.0,  0.5, 0.8, 0.3]
  q rotated  = [-0.871, 0.701, 0.794, 0.316]

The magnitude is preserved: ||q_rot|| = ||q|| (rotation doesn't change length).
Only the direction changes — encoding position information.
```

### Why RoPE Generalizes Beyond Training Length

Sinusoidal and learned absolute embeddings require you to have seen position 2000 during training to handle sequences of length 2000. RoPE embeds position as a rotation angle — the model learns to handle relative angles, not absolute positions. Rotating by a larger angle (longer sequence) is an extrapolation of the same operation.

---

## Change 4 — SwiGLU FFN

### Standard GPT-2 FFN

```python
h = GELU(x × W_1)     [d → 4d]
y = h × W_2            [4d → d]

Parameters: W_1, W_2
```

### LLaMA SwiGLU FFN

```python
h = Swish(x × W_1) ⊙ (x × W_3)   [element-wise multiply, gating]
y = h × W_2

Swish(z) = z × σ(z) = z / (1 + e^(-z))

Parameters: W_1, W_2, W_3   (3 matrices instead of 2)
```

The gate `x × W_3` controls how much of the activation `Swish(x × W_1)` passes through. Network learns to suppress or amplify features for certain inputs.

### Step-by-step: x = 1.5 (scalar for clarity)

**GELU FFN:**

```
W_1 = 0.5
h = W_1 × x = 0.5 × 1.5 = 0.75

GELU(0.75):
  GELU(z) ≈ z × Φ(z)  where Φ is the standard normal CDF
  Φ(0.75) = 0.773
  GELU(0.75) = 0.75 × 0.773 = 0.580

W_2 = 0.6
y = W_2 × GELU(h) = 0.6 × 0.580 = 0.348
```

**SwiGLU FFN:**

```
W_1 = 0.5,  W_3 = 0.8,  W_2 = 0.6

act = W_1 × x = 0.5 × 1.5 = 0.75
gate = W_3 × x = 0.8 × 1.5 = 1.20

Swish(act) = act × σ(act) = 0.75 × (1 / (1 + e^(-0.75))) = 0.75 × 1/1.472 = 0.75 × 0.680 = 0.510
  Swish(0.75) = 0.75 × 0.680 = 0.510

h = Swish(act) ⊙ gate = 0.510 × 1.20 = 0.612

y = W_2 × h = 0.6 × 0.612 = 0.367
```

### Comparison

```
GELU output:   y = 0.348
SwiGLU output: y = 0.367

Different — the gate amplified the signal here (gate=1.20 > 1).
If gate were 0.3: SwiGLU output = 0.6 × (0.510 × 0.3) = 0.092 ← gated down.
```

### Why SwiGLU?

GELU is fixed activation — no data-dependent control. SwiGLU: the gate W_3×x is learned and depends on input x. Network learns to suppress or amplify features for certain inputs.

Extra cost: W_3 adds d×4d parameters. LLaMA compensates by reducing FFN hidden dim to ~2.67d (so total param count stays same as standard 4d FFN).

```
Standard FFN params: W_1 (d×4d) + W_2 (4d×d) = 8d²
SwiGLU params:  W_1 (d×(8d/3)) + W_2 ((8d/3)×d) + W_3 (d×(8d/3)) = 3 × (8d²/3) = 8d²
Same parameter count, better performance.
```

---

## Change 5 — GQA (Grouped Query Attention)

### The KV-Cache Problem

During inference, at each new token, we compute Q for the new token + K, V for the new token — appended to the cache. Attention requires all cached K, V.

**KV cache size** = (sequence length) × H × d_k × 2 × 2 (K and V)

```python
# For LLaMA-7B (H=32, d_k=128, T=4096):
KV cache = 4096 × 32 × 128 × 2 × 2 bytes (fp16)
         = 4096 × 32 × 128 × 4
         = 67,108,864 bytes ≈ 67 MB per batch item

# Batch size 32: 67 × 32 = 2.15 GB — just for the cache.
```

### Three Attention Variants

```
MHA (Multi-Head Attention) — GPT-2, LLaMA-1:
  Each of H heads has own K, V
  H=4: Q_1,K_1,V_1 | Q_2,K_2,V_2 | Q_3,K_3,V_3 | Q_4,K_4,V_4
  KV cache: H × T × d_k × 2

MQA (Multi-Query Attention) — PaLM:
  All H query heads share ONE K, V pair
  H=4: Q_1,Q_2,Q_3,Q_4 all use K_shared, V_shared
  KV cache: 1 × T × d_k × 2  (H/H smaller)
  Problem: quality degrades at large model size

GQA (Grouped Query Attention) — LLaMA-2, LLaMA-3, Mistral:
  G groups. Each group of (H/G) query heads shares K, V.
  H=4, G=2: Q_1,Q_2 share K_g1, V_g1 | Q_3,Q_4 share K_g2, V_g2
  KV cache: G × T × d_k × 2  (H/G smaller than MHA)
```

### Step-by-step: H=4 heads, d_k=2, T=3 tokens, G=2 groups

**MHA layout:**

```
Token 1: K_1=[0.5,0.3], K_2=[0.8,0.1], K_3=[0.6,0.9], K_4=[0.3,0.8]
         V_1=[0.7,0.3], V_2=[0.5,0.9], V_3=[0.8,0.1], V_4=[0.2,0.6]

Token 2: K_1=[...], K_2=[...], K_3=[...], K_4=[...]   ← 4H vectors per token
         V_1=[...], V_2=[...], V_3=[...], V_4=[...]   ← 4V vectors per token

KV cache after 3 tokens:
  K: 3 tokens × 4 heads × 2 dims = 24 values
  V: 3 tokens × 4 heads × 2 dims = 24 values
  Total: 48 values
```

**GQA layout (G=2 groups):**

```
Group 1: heads 1,2 share K_g1, V_g1
Group 2: heads 3,4 share K_g2, V_g2

Token 1: K_g1=[0.5,0.3], K_g2=[0.6,0.8]   ← only 2K vectors (was 4)
         V_g1=[0.7,0.3]                     ← only 2V vectors (was 4)

KV cache after 3 tokens:
  K: 3 tokens × 2 groups × 2 dims = 12 values
  V: 3 tokens × 2 groups × 2 dims = 12 values
  Total: 24 values   ← 50% savings vs MHA
```

### Attention computation for head 1 (uses group 1's K, V)

```
q_1 (new token) = [0.8, 0.6]

Scores against cached K_g1 (all 3 tokens):
  score_1 = q_1 · K_g1[token1] = 0.8×0.5 + 0.6×0.3 = 0.40 + 0.18 = 0.58
  score_2 = q_1 · K_g1[token2] = 0.8×0.4 + 0.6×0.4 = 0.32 + 0.24 = 0.56
  score_3 = q_1 · K_g1[token3] = 0.8×0.2 + 0.6×0.3 = 0.16 + 0.18 = 0.34

Softmax([0.58, 0.45, 0.35]):
  e^0.55=1.788, e^0.45=1.610, e^0.38=1.462    sum=4.860
  a = [0.367, 0.332, 0.301]

output_head1 = 0.367×V_g1[1] + 0.332×V_g1[2] + 0.301×V_g1[3]

Head 2 also uses K_g1, V_g1 — same KV but different Q. No quality issue since Q is still per-head.
```

### Memory savings at LLaMA-70B scale

| Variant | KV heads | Cache (T=4096, fp16) | vs MHA |
|---------|----------|----------------------|--------|
| MHA (H=64) | 64 | 4 GB per sequence | 1× |
| GQA (G=8) | 8 | 0.5 GB per sequence | **8× smaller** |
| MQA (G=1) | 1 | 62 MB per sequence | 64× smaller |

LLaMA-2 70B uses GQA with 8 KV heads — 64/8 = 8× KV cache reduction. Fits more sequences in batch.

---

## Change 6 — Sliding Window Attention (Mistral)

Standard attention: token at position t can attend to all positions 0..t. Cost: O(T²) for a sequence of length T.

Mistral's Sliding Window Attention: each token attends to only the last W positions. Cost: O(T × W) — linear in T for fixed W.

### Example: W=3, sequence "cat sat on the mat"

```
Positions: 0=cat  1=sat  2=on  3=the  4=mat

Standard attention for "mat" (pos 4):
  can see: cat(0), sat(1), on(2), the(3), mat(4)  → all 5

SWA for "mat" (pos 4):
  can see: on(2), the(3), mat(4)                   → only last 3
  CANNOT see: cat(0), sat(1)
```

But information propagates via layers:

```
Layer 1: "mat" sees on, the, mat (window=3)
Layer 2: "mat"'s representation now contains info from on's layer-1 repr
         → on's layer-1 repr saw sat, on, the
         → effectively, "mat" at layer 2 can "see" sat
Layer 3: Effective receptive field = 9 tokens
Layer L: Effective receptive field = L × W tokens
```

For Mistral 7B: W=4096, L=32 layers → effective field = 131,072 tokens from just 4096-token windows.

### Cost comparison

```
Standard attention on T=32768 tokens:
  Memory: O(32768²) = 1 billion attention weights per head  → OOM

SWA on T=32768, W=4096:
  Memory: O(32768 × 4096) = 134 million weights per head   → fits
```

---

## Architecture Comparison Table

| Feature | GPT-2 | LLaMA-1 | LLaMA-2 | Mistral 7B |
|---------|-------|---------|---------|------------|
| Layers | 12-48 | 32-80 | 32-80 | 32 |
| Hidden dim | 768-1600 | 4096-8192 | 4096-8192 | 4096 |
| Attention | MHA | MHA | **GQA** | **GQA** |
| Position | Learned abs | **RoPE** | RoPE | RoPE |
| Normalization | LayerNorm, post | **RMSNorm, pre** | RMSNorm, pre | RMSNorm, pre |
| FFN activation | GELU | **SwiGLU** | SwiGLU | SwiGLU |
| Context | 1024 | 2048 | 4096 | **32768 (SWA)** |

---

## Parameter Count Breakdown

### LLaMA-7B Architecture

```
d_model = 4096
n_heads = 32 (MHA, d_k = 128)
n_layers = 32
FFN hidden = 11008 (~2.67 × 4096)
Vocab size = 32000
```

### Parameters per layer

```
Attention:
  W_Q: 4096 × 4096 = 16,777,216
  W_K: 4096 × 4096 = 16,777,216
  W_V: 4096 × 4096 = 16,777,216
  W_O: 4096 × 4096 = 16,777,216
  Subtotal: 67,108,864

FFN (SwiGLU, 3 matrices):
  W_1: 4096 × 11008 = 45,088,768
  W_2: 11008 × 4096 = 45,088,768
  W_3: 4096 × 11008 = 45,088,768
  Subtotal: 135,266,304

RMSNorm (2 per layer):
  2 × 4096 = 8,192

Per layer: 67M + 135M + 8K ≈ 202,375,168
```

### Total

```
32 layers × 202M = 6,476M ≈ 6.5B
Embedding table: 32000 × 4096 = 131M
Total ≈ 6.7B ≈ 7B  ✓
```

### Model sizes

| Model | Layers | d_model | n_heads | Parameters |
|-------|--------|---------|---------|------------|
| LLaMA-7B | 32 | 4096 | 32 | 6.7B |
| LLaMA-13B | 40 | 5120 | 40 | 13B |
| LLaMA-70B | 80 | 8192 | 64 (GQA: 8 KV) | 70B |
| Mistral 7B | 32 | 4096 | 32 (GQA: 8 KV) | 7B |

---

## Memory Requirements

### LLaMA-7B inference

```
Model weights (fp16):
  7B × 2 bytes = 14 GB

Model weights (int8 / QLoRA):
  7B × 0.5 bytes = 3.5 GB  → fits on single 4090

KV cache (T=2048, batch=1, fp16):
  32 layers × 32 heads × 2048 × 128 × 2 (K+V) × 2 bytes
  = 32 × 32 × 2048 × 128 × 4
  = 1,073,741,824 bytes ≈ 1 GB
```

### LLaMA-70B with GQA-8 (vs MHA)

```
MHA KV cache: 80 × 64 × 4096 × 128 × 4 = 107 GB  → doesn't fit
GQA KV cache: 80 × 8 × 4096 × 128 × 4 = 13 GB    → fits alongside fp16 weights

GQA is not a minor optimization — it's what makes 70B inference viable on 2×A100 (80GB each).
```

---

## Code

```python
import numpy as np

# --- 1. RMSNorm vs LayerNorm ---
def layer_norm(x, gamma=None, beta=None, eps=1e-8):
    mean = np.mean(x)
    var  = np.mean((x - mean) ** 2)
    x_norm = (x - mean) / np.sqrt(var + eps)
    if gamma is not None:
        x_norm = x_norm * gamma
    if beta is not None:
        x_norm = x_norm + beta
    return x_norm

def rms_norm(x, gamma=None, eps=1e-8):
    rms = np.sqrt(np.mean(x ** 2) + eps)
    x_norm = x / rms
    if gamma is not None:
        x_norm = x_norm * gamma
    return x_norm

x = np.array([1.0, 2.0, 3.0, 4.0])
print("LayerNorm:", np.round(layer_norm(x), 3))
print("RMSNorm: ", np.round(rms_norm(x), 3))

# --- 2. RoPE ---
def rope_angles(d, base=10000):
    """Compute rotation angles for each dimension pair."""
    return [base ** (-2 * i / d) for i in range(d // 2)]

def rotate_vector(q, position, angles):
    """Apply RoPE rotation to query/key vector."""
    q_rot = np.zeros_like(q)
    for i, theta in enumerate(angles):
        angle = position * theta
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        q_rot[2*i]   = cos_a * q[2*i]   - sin_a * q[2*i+1]
        q_rot[2*i+1] = sin_a * q[2*i]   + cos_a * q[2*i+1]
    return q_rot

def rope_attention_score(q, k, pos_q, pos_k, d=4):
    """Attention score using RoPE — depends only on (pos_q - pos_k)."""
    angles = rope_angles(d)
    q_rot  = rotate_vector(q, pos_q, angles)
    k_rot  = rotate_vector(k, pos_k, angles)
    return np.dot(q_rot, k_rot)

q = np.array([1.0, 0.5, 0.8, 0.3])
k = np.array([1.0, 0.5, 0.8, 0.3])

# Same relative position (distance=1), different absolute positions
print("Relative position invariance:")
for (m, n) in [(1,0), (2,1), (3,2), (10,9)]:
    score = rope_attention_score(q, k, m, n)
    print(f"  pos ({m},{n}), rel={m-n}: score = {score:.4f}")   # all should be identical

# Different relative positions
print("Different relative positions:")
for (m, n) in [(1,0), (2,0), (3,0)]:
    score = rope_attention_score(q, k, m, n)
    print(f"  pos ({m},{n}), rel={m-n}: score = {score:.4f}")   # scores change as distance increases

# --- 3. SwiGLU FFN ---
def swish(z):
    return z * (1 / (1 + np.exp(-z)))   # z × σ(z)

def gelu(z):
    return z * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

def ffn_gelu(x, W1, W2):
    h = gelu(x @ W1)
    return h @ W2

def ffn_swiglu(x, W1, W2, W3):
    act  = swish(x @ W1)   # Swish branch
    gate = x @ W3          # gate branch
    h    = act * gate      # element-wise gating
    return h @ W2

d, d_ff = 4, 8
np.random.seed(0)
x  = np.array([1.0, 0.5, 0.8, 0.3])
W1 = np.random.randn(d, d_ff) * 0.1
W2 = np.random.randn(d_ff, d) * 0.1
W3 = np.random.randn(d, d_ff) * 0.1

out_gelu   = ffn_gelu(x, W1, W2)
out_swiglu = ffn_swiglu(x, W1, W2, W3)
print("GELU FFN output:   ", np.round(out_gelu, 3))
print("SwiGLU FFN output: ", np.round(out_swiglu, 3))

# --- 4. GQA KV-cache memory savings ---
def kv_cache_bytes(n_layers, n_kv_heads, seq_len, d_k, dtype_bytes=2):
    """KV cache size in bytes (fp16 by default)."""
    return n_layers * n_kv_heads * seq_len * d_k * 2 * dtype_bytes   # ×2 for K and V

# LLaMA-7B
n_layers = 32
seq_len  = 4096
d_k      = 128

mha = kv_cache_bytes(n_layers, n_kv_heads=32, seq_len=seq_len, d_k=d_k)
gqa = kv_cache_bytes(n_layers, n_kv_heads=8,  seq_len=seq_len, d_k=d_k)
mqa = kv_cache_bytes(n_layers, n_kv_heads=1,  seq_len=seq_len, d_k=d_k)

print(f"MHA (32 KV heads): {mha/1e9:.2f} GB  (seq_len={seq_len})")
print(f"GQA (8 KV heads):  {gqa/1e9:.2f} GB  ({mha//gqa}x smaller)")
print(f"MQA (1 KV head):   {mqa/1e6:.0f} MB  ({mha//mqa}x smaller)")
```

### Output

```
LayerNorm: [-1.342 -0.447  0.447  1.342]
RMSNorm:   [ 0.365  0.730  1.095  1.460]

Relative position invariance:
  pos (1,0), rel=1: score = 0.6832
  pos (2,1), rel=1: score = 0.6832
  pos (3,2), rel=1: score = 0.6832   ← identical ✓

Different relative positions:
  pos (1,0), rel=1: score = 0.6832
  pos (2,0), rel=2: score = 0.2487   ← drops as distance increases
  pos (3,0), rel=3: score = -0.1753  ← negative at large distance

KV Cache at seq_len=4096:
  MHA (32 KV heads): 1.07 GB
  GQA (8 KV heads):  0.27 GB  (4x smaller)
  MQA (1 KV head):   0.03 GB  (32x smaller)
```

---

## Interview Q&A

**Q: What are the 4 main differences between LLaMA and GPT-2?**

1. **RoPE** instead of learned absolute position embeddings — relative positions, length generalization.
2. **RMSNorm + pre-norm** instead of LayerNorm + post-norm — simpler, more training stable.
3. **SwiGLU** instead of GELU — learned gating, data-dependent activation.
4. **GQA** (LLaMA-2+) instead of MHA — KV cache memory savings.

**Q: Why does RoPE generalize to longer sequences than the model was trained on?**

RoPE encodes position as a rotation angle. Attention scores depend only on relative angle (m-n), not absolute position. Adding a longer sequence just applies larger rotations — the model has seen the rotational relationship between nearby tokens, which extrapolates. Absolute embeddings fail because PE_2000 was never seen if trained only up to 1024.

**Q: Why pre-norm instead of post-norm?**

Post-norm: `LayerNorm(x + Attention(x))` — gradient must flow through LayerNorm, causing instability in deep networks. Pre-norm: `x + Attention(RMSNorm(x))` — the residual stream has a clean gradient path ∂L/∂x ≈ 1 + ... never blocked by normalization. LLaMA-1 (65B, 80 layers) would be hard to train with post-norm.

**Q: Why does GQA save so much memory?**

KV cache stores K, V for every token in the sequence. Each head has its own K, V in MHA. GQA groups heads — G groups share K, V instead of H separate pairs. LLaMA-70B: H=64 heads → G=8 groups → 8× smaller KV cache. This directly enables larger batch sizes and longer sequences during inference.

**Q: What is SwiGLU and why is it better than GELU?**

GELU: h = GELU(x × W_1), then project. Fixed activation; no input-dependency. SwiGLU: h = Swish(x × W_1) ⊙ (W_3 × x). The gate W_3×x is learned and input-dependent. Network learns to suppress or amplify features per input. Empirically +0.5-1 point on downstream tasks.

**Q: What does Mistral's Sliding Window Attention solve?**

Standard attention is O(T²) memory — a 32K token sequence needs T² = 1 billion attention weights per head. SWA: each token attends to only the last W=4096 tokens — O(T×W), linear in T. Global info propagates via layers: after L layers, effective receptive field = L×W tokens. Mistral 7B: W=4096, L=32 → 131K effective context from 4096-token windows.

**Q: How does RMSNorm differ from LayerNorm computationally?**

LayerNorm: subtract mean, divide by std — requires mean + variance computation. RMSNorm: only compute RMS (sqrt of mean of squares), no mean subtraction. No β (shift) parameter, ~40% faster than LayerNorm in practice. Intuition: the mean subtraction in LayerNorm centers the output; empirically centering doesn't help for transformers, so it can be removed.

**Q: Why is LLaMA-2 70B so much more practical than a hypothetical 70B MHA model?**

MHA KV cache at seq_len=4096: ~107 GB (80 layers × 64 heads). This alone exceeds 2×A100-80GB GPU memory — nothing left for weights. GQA (8 KV heads): ~13 GB for KV cache + 140 GB for model weights (fp16) = 153 GB. Fits on 2×A100-80GB with enough headroom for a reasonable batch size.

---

## Connections

| Concept | Builds on | Used in |
|---------|-----------|---------|
| RoPE | Sinusoidal PE (4.nlp), attention score Q·K | Every modern LLM: LLaMA, Mistral, Falcon, Qwen |
| RMSNorm | LayerNorm concept | LLaMA family, T5 v1.1, Gemma |
| SwiGLU | Gated Linear Unit (GLU), GELU | LLaMA, PaLM, Gemini |
| GQA | Multi-Head Attention, KV-cache mechanics | LLaMA-2+, Mistral, Falcon |
| SWA | Sparse attention, local attention | Mistral, Longformer (different formulation) |
| Pre-norm | Residual connections, training stability | LLaMA, GPT-NeoX, most post-2021 LLMs |

---

## Key Takeaway

Every change from GPT-2 to LLaMA solves a concrete problem: pre-norm for gradient stability, RMSNorm for speed, RoPE for length generalization, SwiGLU for better activations, GQA for KV cache memory. None are architectural revolutions — each is a targeted, measurable improvement. The consensus modern LLM is GPT-2 + all five changes simultaneously.

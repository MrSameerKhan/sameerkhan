# 08b — Llama 3: Reading a Modern Model Card

> Board 13. Companion to [08_modern_llm_architecture.md](08_modern_llm_architecture.md), which
> hand-computes the four changes from GPT-2 — **pre-norm, RMSNorm, RoPE, SwiGLU, GQA** — and is not
> repeated here. That file audits clean; this one is **Llama 3 specifically**: the exact configs,
> the parameter arithmetic, and what changed from Llama 2.
>
> This board's stated goal is *"given any model card, draw the whole model from memory."*
> §10 is that exercise, worked.
>
> **Depends on board 12** ([04b_attention_at_scale_end_to_end.md](04b_attention_at_scale_end_to_end.md)) —
> GQA cannot be justified without KV-cache arithmetic.

---

## 1. Llama 3 in one box

```
                    8B          70B         405B (Llama 3.1)
layers              32          80          126
d_model           4096        8192        16384
query heads         32          64          128
KV heads             8           8            8      <- PINNED at 8, every size
d_head             128         128          128
d_ff (SwiGLU)    14336       28672        53248
vocabulary     128,256     128,256      128,256      tiktoken BPE
context          8,192       8,192      131,072      (3.1 extends 8B/70B to 128k too)
RoPE base      500,000     500,000      500,000      (Llama 2 used 10,000)
norm            RMSNorm, pre-norm, eps 1e-5
activation      SwiGLU (3 matrices: gate, up, down)
biases          NONE
LM head         UNTIED from the embedding
parameters   8,030,261,248  70,553,706,496  405,853,388,800
training        15T tokens
```

Every parameter count above is computed exactly in §4 and matches the released checkpoints.

---

## Table of Contents

1. Llama 3 in one box
2. What Llama 3 inherits (and where it is hand-computed)
3. What changed from Llama 2
4. The parameter arithmetic
5. GQA — why 8 KV heads at every size
6. RoPE base 500,000
7. The 128,256-token vocabulary
8. 15T tokens — deliberately past Chinchilla
9. KV cache at Llama 3 scale
10. **Reading a model card** — the exercise
11. Quick reference

---

## 2. What Llama 3 inherits

None of these is new in Llama 3. All are hand-computed in
[08_modern_llm_architecture.md](08_modern_llm_architecture.md):

| Change | From | Replaces |
|---|---|---|
| **Pre-norm** | GPT-2 (2019) | post-LN — see [06b §5](06b_gpt2_end_to_end.md) |
| **RMSNorm** | **T5 (2019)** | LayerNorm — see [07 §3](07_t5_end_to_end.md) |
| **RoPE** | RoFormer (2021) | learned absolute positions |
| **SwiGLU** | GLU variants (2020) | ReLU/GELU MLP |
| **GQA** | Ainslie et al. (2023) | MHA |
| **No biases** | T5 (2019) | biased linears |

**Two of those trace back to T5**, which is worth saying out loud — RMSNorm and dropping biases are
routinely credited to Llama.

The Llama 3 block, in order:

```
x -> RMSNorm -> GQA attention (RoPE applied to Q,K) -> + x
  -> RMSNorm -> SwiGLU MLP                          -> + x
  ... x32 ...
  -> RMSNorm (final) -> lm_head (untied)
```

---

## 3. What changed from Llama 2

The architecture is nearly identical. Four things moved:

| | Llama 2 7B | **Llama 3 8B** |
|---|---|---|
| Vocabulary | 32,000 SentencePiece | **128,256 tiktoken BPE** (4×) |
| RoPE base | 10,000 | **500,000** (50×) |
| GQA | 70B only | **every size, including 8B** |
| `d_ff / d_model` | 2.6875 (≈ 8/3) | **3.5** |
| Training tokens | 2T | **15T** (7.5×) |
| Context | 4,096 | 8,192 (131,072 in 3.1) |

**Llama 2 7B → Llama 3 8B gained ~1B parameters, and almost all of it is the embedding table**
(`32,000×4096 = 131,072,000` → `128,256×4096 = 525,336,576`, twice over because the head is untied).

---

## 4. The parameter arithmetic

No biases, so every block term is a clean matrix product:

```
per layer:
  q_proj        d × (H  × d_head)
  k_proj        d × (KV × d_head)        <- smaller than q_proj, this is GQA
  v_proj        d × (KV × d_head)
  o_proj    (H × d_head) × d
  gate_proj     d × d_ff       ┐
  up_proj       d × d_ff       │  SwiGLU needs THREE, not two
  down_proj  d_ff × d          ┘
  2 × RMSNorm   2 × d          (scale only — no bias)

model = V×d (embed) + L×block + d (final norm) + V×d (untied lm_head)
```

### Llama 3 8B, fully worked

```
  embed_tokens   128,256 × 4,096   =     525,336,576
  lm_head        128,256 × 4,096   =     525,336,576    UNTIED
  per layer                        =     218,112,000
    q_proj    4096 × 4096  =  16,777,216
    k_proj    4096 × 1024  =   4,194,304     (8 KV heads × 128 = 1024)
    v_proj    4096 × 1024  =   4,194,304
    o_proj    4096 × 4096  =  16,777,216
    gate      4096 × 14336 =  58,720,256
    up        4096 × 14336 =  58,720,256
    down     14336 × 4096  =  58,720,256
    2 norms          2×4096 =      8,192
  × 32 layers                      =   6,979,584,000
  final norm                       =           4,096
                                      ──────────────
  TOTAL                            =   8,030,261,248     ✓ "8.03B"
```

```
Llama-3 8B      8,030,261,248
Llama-3 70B    70,553,706,496
Llama-3.1 405B 405,853,388,800
```

**Embeddings are 13.1% of the 8B model** — `2 × 525M` out of `8.03B`. A 128k vocabulary is
expensive, and untying the head doubles that cost. (Llama 3.2's 1B and 3B *do* tie, because at that
size the embedding would otherwise dominate.)

### The SwiGLU `d_ff` rule — and where Llama 3 departs from it

SwiGLU uses **three** matrices instead of two, so to keep the parameter count equal to a standard
`4d` FFN you set `d_ff = (2/3) × 4d = 8d/3 ≈ 2.667d`:

```
Llama-2 7B    d=4096   d_ff=11008   ratio 2.6875    ≈ 8/3   parameter-matched
Llama-3 8B    d=4096   d_ff=14336   ratio 3.5000            WIDER
Llama-3 70B   d=8192   d_ff=28672   ratio 3.5000            WIDER
```

**Llama 2 followed the rule; Llama 3 abandoned it** and widened the FFN to `3.5d`, spending real
extra parameters on the MLP. If asked "why is `d_ff` 14336 and not 16384", the answer is that the
`8d/3` convention comes from SwiGLU's three matrices — and that Llama 3 deliberately went above it.

---

## 5. GQA — why 8 KV heads at every size

```
  model    query heads   KV heads   queries per KV head   cache reduction
  8B               32           8                     4                4×
  70B              64           8                     8                8×
  405B            128           8                    16               16×
```

**`n_kv_heads` is pinned at 8 while query heads scale with width.** So the bigger the model, the
*more* GQA saves — the KV cache simply stops growing with head count.

Why 8 and not 1 (MQA)? 8 divides evenly across an 8-GPU tensor-parallel shard, so each GPU owns
exactly one KV head and needs no cross-device communication for K/V. The number is a
systems choice as much as a quality one.

What it trades away, and why it is worth it, is board 12
([04b §4](04b_attention_at_scale_end_to_end.md)): four query heads share one K and V, so they cannot
specialise independently. The loss is small; the memory win is 4–16×; and because decode is
bandwidth-bound, memory converts almost directly into throughput.

---

## 6. RoPE base 500,000

RoPE rotates dimension pair `i` by angle `m · θ_i` with `θ_i = base^(−2i/d_head)`. Raising the base
**slows every rotation**, especially the slowest pairs.

With `d_head = 128` (64 pairs), the slowest pair:

```
       base    slowest theta    wavelength (positions)
     10,000       0.00011548                    54,410
    500,000       0.00000246                 2,559,196
```

**47× longer before the slowest dimension wraps around.** Wavelength is `2π/θ` — how many positions
fit before that dimension returns to where it started and positions become ambiguous.

This is what makes 8k → 128k context extension feasible: at base 10,000 the low-frequency dimensions
are already cycling within a long context, so distant positions become indistinguishable. Llama 3.1
extends to 131,072 tokens on top of this, plus additional RoPE scaling (board 14).

---

## 7. The 128,256-token vocabulary

```
Llama 2   32,000    SentencePiece BPE
Llama 3  128,256    tiktoken BPE (the GPT-4 family tokenizer lineage)
```

```
embedding cost at d=4096:
   32,000 × 4096 =   131,072,000
  128,256 × 4096 =   525,336,576      4× more, and ×2 because the head is untied
```

**The trade:** a larger vocabulary costs embedding parameters but produces *fewer tokens per
document* — roughly 15% shorter sequences on English, and much larger gains on code and non-English
text. Fewer tokens means less attention compute, a smaller KV cache for the same text, and lower
cost per document. Meta judged the 800M extra parameters worth it.

This is the same trade GPT-2 made going byte-level ([06b §12](06b_gpt2_end_to_end.md)), pushed
further.

---

## 8. 15T tokens — deliberately past Chinchilla

```
Llama-3 8B:  N = 8.03e9 parameters,  D = 15e12 tokens
             tokens per parameter = 1,868

Chinchilla compute-optimal ≈ 20 tokens/parameter  ->  161B tokens
Llama 3 used 93× MORE data than compute-optimal
```

**This is not a mistake — it is the opposite of GPT-3's mistake.** GPT-3 was *under*-trained
(300B tokens for 175B params, [06c §7](06c_gpt3_end_to_end.md)); Llama 3 8B is *massively*
over-trained by Chinchilla's rule, on purpose.

The reason: **Chinchilla optimises training compute. Serving optimises inference cost.** A model you
will run billions of times should be as small as possible for a given quality, even if reaching that
quality costs far more training compute than is "optimal". Train once, serve forever.

Being able to state both the Chinchilla rule *and* why Llama 3 deliberately violates it is the
complete answer; quoting only the rule is the incomplete one.

---

## 9. KV cache at Llama 3 scale

Using board 12's formula, fp16, batch 1:

```
  model          layers  KV   context        cache     if it were MHA
  8B                 32   8     8,192      1.00 GiB          4.00 GiB
  8B  @ 128k         32   8   131,072     16.00 GiB         64.00 GiB
  70B                80   8     8,192      2.50 GiB         20.00 GiB
  405B @ 128k       126   8   131,072     63.00 GiB      1,008.00 GiB
```

Two numbers to keep:

- **Llama-3 8B at 128k context: 16.00 GiB of cache against 14.96 GiB of weights.** The cache is
  *larger than the model*, for a single sequence. Same lesson as board 12's 123%.
- **405B at 128k without GQA would need 1,008 GiB of cache per sequence.** With GQA it is 63 GiB.
  That is not an optimisation — it is the difference between possible and impossible.

---

## 10. Reading a model card — the exercise

This is the board's actual goal. Given a `config.json`, derive everything:

```json
{ "num_hidden_layers": 32,   "hidden_size": 4096,
  "num_attention_heads": 32, "num_key_value_heads": 8,
  "intermediate_size": 14336, "vocab_size": 128256,
  "max_position_embeddings": 8192, "rope_theta": 500000.0,
  "rms_norm_eps": 1e-5, "tie_word_embeddings": false }
```

Read it in this order:

```
1. num_key_value_heads (8) < num_attention_heads (32)   -> GQA, 4 queries per KV head
2. d_head = hidden_size / num_attention_heads = 4096/32 = 128
3. rope_theta = 500000                                  -> Llama-3 era, long-context ready
4. rms_norm_eps present, no layer_norm_eps              -> RMSNorm, so pre-norm
5. intermediate_size / hidden_size = 3.5                -> SwiGLU (3 matrices), widened past 8/3
6. tie_word_embeddings = false                          -> lm_head is a SEPARATE 128256×4096
7. vocab 128256                                         -> tiktoken BPE, not SentencePiece
8. params = 2×(128256×4096) + 32×218,112,000 + 4096 = 8,030,261,248
9. KV cache @ 8192, fp16, batch 1
     = 2 × 32 × 8 × 128 × 8192 × 2 = 1.00 GiB
```

**Every architectural fact is in those nine lines.** No config field is decorative: `rope_theta`
tells you the era and the context ambition, the KV/query head split tells you the cache story, and
`intermediate_size/hidden_size` tells you the activation is gated.

---

## 11. Quick reference

```
LLAMA 3 BLOCK  (no biases anywhere)
  h = x + GQA( RMSNorm(x) )            RoPE applied to Q and K inside
  y = h + SwiGLU( RMSNorm(h) )         SwiGLU = (gate(x) ⊙ SiLU) * up(x) -> down
  ... × L ...
  out = RMSNorm(y);  logits = out @ W_lm      W_lm UNTIED (8B/70B/405B)
```

**The seven things to be able to say cold:**

1. **Llama 3's block is not new.** Pre-norm (GPT-2), RMSNorm and no-biases (**T5**), RoPE
   (RoFormer), SwiGLU, GQA — all pre-date it. Llama 3's contributions are the vocabulary, the RoPE
   base, GQA everywhere, and 15T tokens.
2. **`n_kv_heads = 8` at every size**, so the cache-reduction factor grows 4× → 8× → 16× as the model
   grows. 8 is chosen partly so it shards cleanly across 8 GPUs.
3. **RoPE base 10,000 → 500,000** stretches the slowest dimension's wavelength from 54,410 to
   2,559,196 positions — **47×** — which is what makes 128k context feasible.
4. **Vocabulary 32k → 128,256** costs `4×` the embedding parameters (and doubly, since the head is
   untied) but buys ~15% fewer tokens per document, and far more on code and non-English.
5. **`d_ff/d_model = 3.5`, not `8/3`.** The `8d/3` convention exists because SwiGLU uses three
   matrices; Llama 3 deliberately went wider than parameter-matched.
6. **15T tokens = 1,868 per parameter, 93× past Chinchilla — on purpose.** Chinchilla optimises
   training compute; Llama 3 optimises inference cost. Opposite error to GPT-3's under-training.
7. **8B = 8,030,261,248 exactly**, embeddings 13.1% of it. At 128k context the KV cache (16.00 GiB)
   is *larger than the weights* (14.96 GiB).

---

## See also

- [08_modern_llm_architecture.md](08_modern_llm_architecture.md) — RMSNorm, RoPE, SwiGLU, GQA hand-computed as a diff from GPT-2
- [04b_attention_at_scale_end_to_end.md](04b_attention_at_scale_end_to_end.md) — board 12: why GQA is worth it, KV cache arithmetic
- [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md) — the pre-LN baseline Llama 3 still uses
- [07_t5_end_to_end.md](07_t5_end_to_end.md) — where RMSNorm and no-biases actually came from
- [11_long_context_scaling.md](11_long_context_scaling.md) — board 14: RoPE scaling past 128k
- [../../4.nlp/03_sequence_models/08_scaling_laws_emergent.md](../../4.nlp/03_sequence_models/08_scaling_laws_emergent.md) — board 17: Chinchilla in full

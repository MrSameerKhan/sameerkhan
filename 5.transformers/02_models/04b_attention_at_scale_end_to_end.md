# 04b — Attention at Scale: KV Cache, Flash, PagedAttention

> Board 12. Companion to [04_efficient_transformers.md](04_efficient_transformers.md), which is a
> survey; this file is the **arithmetic**. Everything here is computed, and the two correctness
> claims — that the KV cache and Flash Attention are *exact* — are verified numerically rather
> than asserted.
>
> **This board must come before board 13.** "What does GQA trade away, and why is it worth it?"
> is unanswerable without §4.

---

## The one idea

**Generating a token is memory-bound, not compute-bound.** The GPU is not busy; it is waiting on
HBM. Every technique in this file is a way of moving less memory.

```
A100:  312 TFLOP/s fp16     2039 GB/s HBM
ridge point = 312e12 / 2039e9 = 153 FLOP/byte
```

Below 153 FLOP/byte you are bandwidth-limited; above it, compute-limited.

```
DECODE  (1 token)   2N FLOPs, reads 2N bytes of weights  ->  intensity 1 FLOP/byte
                    153x BELOW the ridge -> memory-bound, GPU mostly idle

PREFILL (L=512)     2NL FLOPs, same 2N bytes of weights  ->  intensity 512 FLOP/byte
PREFILL (L=4096)                                         ->  intensity 4096 FLOP/byte
                    ABOVE the ridge -> compute-bound
```

**Prefill and decode are different workloads on the same weights.** That single fact explains
continuous batching, why batching helps decode enormously and prefill barely, and why serving
systems schedule the two phases separately.

---

## Table of Contents

1. The KV cache — what and why
2. The cache is **exact** — verified
3. Sizing the cache — the arithmetic
4. GQA — why the cache scales with KV heads
5. Flash Attention — online softmax, verified exact
6. Flash Attention — the memory it saves
7. PagedAttention — fragmentation
8. Quick reference

---

## 1. The KV cache — what and why

Generating token `t` requires attention over keys and values for positions `0…t`. Without a cache
you recompute all of them every step. But **under a causal mask, position `i`'s key and value can
never change** — nothing after `i` can influence them. So they are recomputed identically, every
step, forever.

```
what is cached      K and V for every past position, per layer, per KV head
what is NOT cached  Q  (only the current token has a query)
                    the attention weights (recomputed, they depend on the new Q)
```

---

## 2. The cache is exact — verified

Run the GPT-1 walkthrough model ([06_gpt1_end_to_end.md](06_gpt1_end_to_end.md)) two ways: full
recompute of the prefix at every step, versus computing only the new token's `q, k, v` and reusing
cached `K, V`.

```
step 1: prefix [bank]
  full   [ 0.0000,  1.5811, -0.1819, -0.1885, -1.6321, -0.5588,  0.3836,  0.0233]
  cached [ 0.0000,  1.5811, -0.1819, -0.1885, -1.6321, -0.5588,  0.3836,  0.0233]   diff 0.000e+00

step 2: prefix [bank, approved]
  full   [ 0.0000,  0.1667, -0.0092, -0.0410, -0.2361, -0.0912,  0.1138, -0.1443]
  cached [ 0.0000,  0.1667, -0.0092, -0.0410, -0.2361, -0.0912,  0.1138, -0.1443]   diff 0.000e+00

step 3: prefix [bank, approved, the]
  full   [ 0.0000, -0.4727,  0.1791,  0.0896,  0.7292,  0.3584, -0.0898, -0.0569]
  cached [ 0.0000, -0.4727,  0.1791,  0.0896,  0.7292,  0.3584, -0.0898, -0.0569]   diff 0.000e+00

step 4: prefix [bank, approved, the, loan]
  full   [ 0.0000, -1.3766,  0.2125,  0.1652,  1.4789,  0.5428, -0.2831, -0.1221]
  cached [ 0.0000, -1.3766,  0.2125,  0.1652,  1.4789,  0.5428, -0.2831, -0.1221]   diff 0.000e+00

max |diff| over the whole generation = 0.000e+00
```

**Zero, not small.** The KV cache is an exact optimisation — it changes speed and memory, never
output. (Those step-4 logits are the same row the decoding file samples from, which cross-checks
both files.)

### Work saved

```
 step   K,V rows built without cache   with cache
    1                              1            1
    2                              2            1
    3                              3            1
    4                              4            1
  tot                             10            4
```

```
L =   128   no cache      8,256 rows   cached    128     64.5x fewer
L = 1,024   no cache    524,800 rows   cached  1,024    512.5x fewer
L = 4,096   no cache  8,390,656 rows   cached  4,096   2048.5x fewer
```

`O(L²)` recomputation collapses to `O(L)`. Exactly `(L+1)/2` times less work.

---

## 3. Sizing the cache

```
bytes = 2 (K and V) × n_layers × n_kv_heads × d_head × seq_len × batch × bytes_per_element
```

### The board's killer question: 7B model, 4k context, batch 8, fp16

```
2 × 32 layers × 32 kv-heads × 128 d_head × 4096 tokens × 8 batch × 2 bytes
= 17,179,869,184 bytes
= 16.00 GiB
```

```
weights at fp16   7e9 × 2  =  13.0 GiB
KV cache          batch 8  =  16.00 GiB      per sequence: 2.00 GiB
```

**The cache is 123% of the weights.** That is the number to have ready. It is why KV-cache memory,
not parameter memory, is what actually caps throughput on a serving box — and why every technique
after this one (GQA, paging, quantised caches) targets the cache rather than the weights.

### Real models, full context, batch 1, fp16

```
Llama-2-7B   MHA   32 layers   32 kv-heads   ctx  4,096   ->   2.00 GiB
Llama-3-8B   GQA   32 layers    8 kv-heads   ctx  8,192   ->   1.00 GiB
Llama-3-70B  GQA   80 layers    8 kv-heads   ctx  8,192   ->   2.50 GiB
GPT-3 175B   MHA   96 layers   96 kv-heads   ctx  2,048   ->   9.00 GiB
```

**Llama-3-8B has a smaller cache than Llama-2-7B despite twice the context.** That is GQA, and it is
§4.

The cache grows **linearly in sequence length and batch**, and is independent of `d_model` except
through `n_kv_heads × d_head`.

---

## 4. GQA — the cache scales with KV heads

Llama-3-8B has **32 query heads but only 8 KV heads**. Four query heads share each KV head.

```
if it were MHA (32 kv-heads), ctx 8192, batch 1  ->  4.00 GiB
actual GQA      ( 8 kv-heads)                    ->  1.00 GiB      4x smaller
```

**The cache scales with `n_kv_heads`, not `n_heads`.** Query heads cost compute; KV heads cost
memory. GQA decouples them and cuts the expensive one.

```
MHA   n_kv_heads = n_heads          largest cache, best quality
GQA   1 < n_kv_heads < n_heads      the modern default (Llama 3 uses 8 at every size)
MQA   n_kv_heads = 1                smallest cache, measurable quality loss
```

**What GQA trades away:** representational capacity in the key/value projections — 4 query heads
must read the same K and V, so they cannot specialise as independently. In practice the loss is
small and the memory win is 4–8×, which at decode time converts almost directly into throughput
because the workload is bandwidth-bound (§0). That is the full answer to the board-13 question, and
it needs this file's arithmetic to state.

Board 13 covers how GQA is implemented; this is why it exists.

---

## 5. Flash Attention — online softmax, verified exact

Standard attention materialises the full `N × N` score matrix in HBM, then softmaxes it. Flash never
materialises it: it streams K and V in blocks, keeping a **running max `m`** and **running sum `l`**,
rescaling what it has accumulated whenever a later block raises the max.

```
for each block j:
    m_new = max(m, rowmax(S_j))
    P_j   = exp(S_j - m_new)
    l     = exp(m - m_new) · l  +  rowsum(P_j)
    O     = exp(m - m_new) · O  +  P_j @ V_j
    m     = m_new
output = O / l
```

Worked on 8 keys, `d = 4`, block size 3, with the maximum deliberately in the **last** block —
the case that exercises the rescaling:

```
STANDARD (materialises the whole row)
  scores  [-0.753441, -0.319047, -0.444528,  0.092756,  0.016686,  0.557021,  0.439371,  0.765642]
  softmax [ 0.050078,  0.077321,  0.068203,  0.116719,  0.108169,  0.185682,  0.165072,  0.228756]
  output  [-0.085526,  0.488204,  0.481634, -0.379803]

FLASH (block size 3, never holds more than 3 scores)
  block 0  keys 0..2   block max -0.319047   running max      —    -> -0.319047
                       running sum 0.000000 -> 2.529730
  block 1  keys 3..5   block max  0.557021   running max -0.319047 ->  0.557021
                       rescale exp(m_old - m_new) = 0.416417
                       running sum 2.529730 -> 3.264571
  block 2  keys 6..7   block max  0.765642   running max  0.557021 ->  0.765642
                       rescale exp(m_old - m_new) = 0.811703
                       running sum 3.264571 -> 4.371471

  output  [-0.085526,  0.488204,  0.481634, -0.379803]

  max | flash − standard | = 1.665e-16
```

**Flash Attention is exact.** Not an approximation, not a quality/speed trade — bit-for-bit the same
attention, computed in a different order. When it appears to change results it is floating-point
non-associativity, the same as any reordered reduction.

The rescale factors `0.416417` and `0.811703` are the mechanism: when block 1 raised the running max
from `-0.319047` to `0.557021`, everything accumulated from block 0 was multiplied by
`exp(-0.319047 - 0.557021) = 0.416417` to put it on the new scale.

---

## 6. Flash Attention — the memory it saves

```
      N       standard stores N² scores        per head, fp16
  1,024                    1,048,576                  2.0 MiB
  4,096                   16,777,216                 32.0 MiB
 16,384                  268,435,456                512.0 MiB
131,072               17,179,869,184             32,768.0 MiB
```

Flash holds one block plus two running scalars per query row — **`O(N)` instead of `O(N²)`**.

**Flash does not reduce FLOPs.** It does the same multiply-adds. What it removes is the
write-then-read of the `N × N` matrix to HBM, replacing it with on-chip SRAM work. Since attention
at long context is bandwidth-bound, removing the memory traffic is what produces the speedup — a
direct application of §0.

```
SRAM   ~20 MB on an A100, ~19 TB/s     tiny and fast   <- Flash works here
HBM    80 GB, 2039 GB/s                large and slow  <- Flash avoids round-trips here
```

Flash also enables long context at all: at `N = 131,072` the score matrix alone would be 32 GiB per
head.

---

## 7. PagedAttention — fragmentation

The cache for a sequence grows one token at a time, but its final length is unknown at admission.
Naive allocators reserve `max_seq_len` per sequence:

```
16 sequences, max_seq_len 2048, average actual length 300, 7B MHA model, fp16

  reserved  2048 tokens × 16 seqs   =  16.00 GiB
  used       300 tokens × 16 seqs   =   2.34 GiB
  WASTED                            =  13.66 GiB   =   85.4%
```

**85% of the KV memory is dead.** vLLM's PagedAttention borrows OS virtual memory: allocate in fixed
**blocks** (typically 16 tokens) and keep a per-sequence block table mapping logical positions to
physical blocks. Blocks need not be contiguous.

```
  paged, block = 16 tokens: round 300 up to 304
  allocated                          =   2.38 GiB
  WASTED                             =   1.3%      <- only the last partial block per sequence
```

Waste drops from **85.4% to 1.3%**, and the memory freed becomes larger batches — which is exactly
what a memory-bound decode workload needs.

Paging also makes **prefix sharing** nearly free: several sequences with the same system prompt point
their block tables at the *same physical blocks*, with copy-on-write when they diverge.

---

## 8. Quick reference

```
KV cache      bytes = 2 · layers · n_kv_heads · d_head · seq · batch · dtype_bytes
              EXACT (verified 0.000e+00). Turns O(L²) recompute into O(L).

GQA           cache scales with n_kv_heads, NOT n_heads.
              Llama-3-8B: 32 query heads, 8 KV heads -> 4x smaller cache.

Flash         online softmax with running max m and running sum l,
              rescaling by exp(m_old - m_new) when a later block raises the max.
              EXACT (verified 1.665e-16). Saves MEMORY TRAFFIC, not FLOPs.
              O(N) instead of O(N²) intermediate.

Paged         fixed 16-token blocks + per-sequence block table.
              Fragmentation 85.4% -> 1.3%. Enables prefix sharing.
```

**The seven things to be able to say cold:**

1. **Decode is memory-bound.** Intensity ~1 FLOP/byte against an A100 ridge point of **153** — the
   GPU is idle waiting on HBM. Prefill at L=4096 is ~4096 FLOP/byte and compute-bound. Same weights,
   opposite regimes.
2. **The KV cache is exact** — verified `0.000e+00`, not "approximately equal". It works because a
   causal mask means past keys and values can never change.
3. **7B, 4k context, batch 8, fp16 = 16.00 GiB of KV cache — 123% of the weights.** The cache, not
   the parameters, is what caps throughput.
4. **The cache scales with `n_kv_heads`.** GQA cuts Llama-3-8B's cache 4× by using 8 KV heads for 32
   query heads; what it trades is K/V specialisation across the 4 query heads that now share them.
5. **Flash Attention is exact too** — `1.665e-16`. It is a reordering, not an approximation, and the
   running-max rescale `exp(m_old − m_new)` is what makes the reordering valid.
6. **Flash saves memory traffic, not FLOPs.** It avoids materialising the `N × N` matrix in HBM —
   `O(N)` instead of `O(N²)`, which is also what makes 128k context possible at all.
7. **PagedAttention fixes fragmentation**, not compute: 85.4% waste down to 1.3% with 16-token
   blocks, plus near-free prefix sharing via shared physical blocks.

---

## See also

- [04_efficient_transformers.md](04_efficient_transformers.md) — the survey: Longformer, BigBird, distillation
- [../../6.llms/05_vllm_internals.md](../../6.llms/05_vllm_internals.md) — vLLM's scheduler and continuous batching
- [06_gpt1_end_to_end.md](06_gpt1_end_to_end.md) — the model whose logits verify §2
- [../../4.nlp/03_sequence_models/06c_transformer_decoder_cross_attention_end_to_end.md](../../4.nlp/03_sequence_models/06c_transformer_decoder_cross_attention_end_to_end.md) — why cross-attention K/V are built once, and self-attention K/V grow
- [08_modern_llm_architecture.md](08_modern_llm_architecture.md) — board 13: how GQA is actually implemented
- [11_long_context_scaling.md](11_long_context_scaling.md) — board 14: what happens past 128k

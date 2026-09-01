# 10 — Mixture of Experts

> Board 15. Every number computed; the Mixtral figures match the released checkpoint exactly.
>
> **The one sentence that answers the board's killer question:** MoE buys **compute**, not
> **memory**. A 47B MoE and a 47B dense model occupy identical GPU memory; the MoE does 3.63× fewer
> FLOPs per token. Getting that distinction right is the whole answer — §7.

---

## 1. The core idea

A dense FFN runs every parameter for every token. An MoE layer holds `N` parallel FFNs ("experts")
and a small **router** that sends each token to only `K` of them.

```
DENSE                              MoE  (N=8, K=2)
  x -> FFN -> y                      x -> router -> pick 2 of 8
       all params, every token            only 2 experts run
                                          the other 6 sit idle in memory
```

The attention sublayer is **shared** — MoE replaces only the FFN. That matters, because the FFN is
two-thirds of a dense layer ([06b §16](06b_gpt2_end_to_end.md)), so replacing it is where the
savings are.

---

## Table of Contents

1. The core idea
2. The router — exact
3. Top-k and gate renormalisation
4. Load balancing — the auxiliary loss
5. Expert capacity and dropped tokens
6. Mixtral 8×7B — the exact arithmetic
7. **Why a 47B MoE is cheaper to serve than a 47B dense model**
8. Serving: expert parallelism
9. Comparison
10. Quick reference

---

## 2. The router — exact

The router is one small linear layer, `d_model → N`. For `d = 4096`, `N = 8`, that is 32,768
parameters per layer — negligible against the experts it steers.

```
router logits h(x) = [-0.3,  0.8, -1.2,  2.1,  0.4, -0.6,  1.7,  0.2]

softmax(h)         = [0.036729, 0.110341, 0.014933, 0.404873,
                      0.073964, 0.027210, 0.271394, 0.060556]      sum = 1.000000
```

---

## 3. Top-k and gate renormalisation

Keep the `K = 2` largest, then **renormalise over the kept experts only**:

```
top-2:  expert 3  p = 0.404873
        expert 6  p = 0.271394

gates = [0.404873, 0.271394] / (0.404873 + 0.271394)
      = [0.598688, 0.401312]                      sum = 1.000000

output = 0.598688 · E₃(x)  +  0.401312 · E₆(x)
```

**Renormalisation matters.** Without it the gates sum to `0.676267` and the FFN output is silently
scaled down by a token-dependent factor — the same failure as the masking bug in
[06c §6.1](../../4.nlp/03_sequence_models/06c_transformer_decoder_end_to_end.md), and just as hard
to spot.

**The router's gradient path:** the gate weights multiply the expert outputs, so `dL/d(gate)` flows
back through them into the router. The router learns from *how useful the chosen experts turned out
to be* — but it only ever gets signal about the experts it actually picked. That self-reinforcement
is exactly why §4 is needed.

---

## 4. Load balancing — the auxiliary loss

Left alone, routing collapses: a slightly-better expert gets picked more, trains more, gets better,
gets picked more. A few experts do everything and the rest are dead weight.

The Switch Transformer auxiliary loss:

```
L_aux = α · N · Σᵢ fᵢ · Pᵢ

  fᵢ = fraction of tokens actually routed to expert i   (hard count, NOT differentiable)
  Pᵢ = mean router probability for expert i             (soft, differentiable)
  α  ≈ 0.01
```

```
  perfectly balanced      Σ fᵢPᵢ = 0.125000    L_aux = 0.010000     <- the minimum
  mildly skewed           Σ fᵢPᵢ = 0.136200    L_aux = 0.010896
  collapsed (1 expert)    Σ fᵢPᵢ = 0.850000    L_aux = 0.068000     <- 6.8× the minimum
```

**Minimum at perfect balance:** with `fᵢ = Pᵢ = 1/N`, `Σ fᵢPᵢ = N·(1/N²) = 1/N = 0.125`, so
`L_aux = α·N·(1/N) = α = 0.01` exactly. The loss cannot go below `α`, and collapse costs 6.8×.

**Why the product `fᵢ · Pᵢ` and not just `fᵢ`:** `fᵢ` is a count — it has no gradient. `Pᵢ` is
differentiable. Multiplying them makes a loss whose *value* measures real imbalance while its
*gradient* flows into the router. That pairing is the entire trick, and it is the most common thing
people cannot explain about MoE.

Production models add a **router z-loss** (`log²` of the router logits' logsumexp) to keep the
logits small and the router numerically stable in bf16.

---

## 5. Expert capacity and dropped tokens

Experts run as fixed-shape batched matmuls, so each gets a hard slot budget:

```
capacity = capacity_factor × (tokens × K) / N
```

For 4,096 tokens, `K=2`, `N=8` — so 8,192 token-slots to distribute:

```
  capacity_factor = 1.00   ->  1,024 slots per expert
  capacity_factor = 1.25   ->  1,280 slots per expert
  capacity_factor = 2.00   ->  2,048 slots per expert
```

With a skewed assignment and `capacity_factor = 1.25`:

```
  assignment  [1400, 1100, 900, 850, 800, 700, 600, 1842]     total 8,192
  capacity     1,280 each
  overflow     (1400−1280) + (1842−1280) = 682

  dropped = 682 of 8,192 slots = 8.33%
```

**A dropped token skips the FFN entirely** and passes through the residual connection unchanged. It
is not an error and nothing crashes — the token simply gets no FFN transformation at that layer.

That is the real cost of imbalance: capacity factor above 1.0 wastes memory on slots that go unused,
below the needed level silently degrades quality. Mixtral avoids the problem at inference by not
enforcing capacity at all (batch sizes are small enough), but training MoE at scale lives and dies
on this knob.

---

## 6. Mixtral 8×7B — the exact arithmetic

```
32 layers · d_model 4096 · d_ff 14336 · SwiGLU · GQA 32q/8kv · N=8 experts · K=2 · vocab 32,000
```

```
  one expert (SwiGLU = 3 matrices)  3 × 4096 × 14336   =     176,160,768
  experts per layer                 × 8                =   1,409,286,144
  all expert FFN                    × 32 layers        =  45,097,156,608
  attention per layer (GQA)                            =      41,943,040
  all attention                     × 32               =   1,342,177,280
  embeddings                        2 × 32,000 × 4096  =     262,144,000
  routers                           32 × 4096 × 8      =       1,048,576
  norms                                                =         266,240
                                                          ──────────────
  TOTAL                                                =  46,702,792,704   "46.7B"

  ACTIVE per token (K=2 experts instead of 8)          =  12,879,925,248   "12.9B"
```

Both match the published figures exactly.

**"8×7B" is a misleading name.** `8 × 7B = 56B`, but Mixtral is 46.7B — because **attention,
embeddings and norms are shared**, not replicated eight times. Only the FFN is octupled. Conversely
it is not "a 7B model either": no 7B model is in there.

```
  total / active = 3.63×      compute is 28% of a dense 46.7B
```

---

## 7. Why a 47B MoE is cheaper to serve than a 47B dense model

This is the board's killer question, and the trap is answering "because it's smaller". It is not.

```
                          dense 46.7B          MoE 46.7B (K=2 of 8)
  FLOPs per token         2 × 46.7B            2 × 12.9B
                          = 93.4 GFLOP         = 25.8 GFLOP     ->  3.63× fewer
  Memory (fp16)           87.0 GiB             87.0 GiB         ->  IDENTICAL
```

**Every expert must be resident in GPU memory**, because the router may select any of them for the
very next token. You cannot page them out at token granularity.

So:

```
  MoE is cheaper in:  FLOPs, latency, throughput, energy per token
  MoE is NOT cheaper in:  GPU memory, GPU count, cost to hold the model
```

Mixtral needs the same ~2×A100-80GB as a dense 47B, but generates tokens at roughly the speed of a
13B model. **You pay 47B of memory to get 13B of latency at 40B-class quality.** That trade is only
worth it when memory is cheap relative to compute — which is exactly the case in a datacentre and
exactly not the case on a laptop.

There is a second-order cost too: at decode, MoE's arithmetic intensity is *worse* than dense
([04b §0](04b_attention_at_scale_end_to_end.md)) — you read the full weight matrices of the selected
experts to process a single token, so the workload is even more bandwidth-bound. MoE's advantage
shows up at high batch, where different tokens hit different experts and the reads amortise.

---

## 8. Serving: expert parallelism

Because experts are independent, they shard across devices differently from dense layers:

```
tensor parallel   split every matrix across GPUs           all GPUs work on every token
expert parallel   put whole experts on different GPUs      tokens ROUTE to the right GPU
```

Expert parallelism turns the router into a network operation: each token's hidden state is sent
(all-to-all) to whichever GPU owns its chosen experts, then the results come back. Two consequences:

- **Load imbalance becomes a latency problem, not just a quality one.** The step finishes when the
  *busiest* GPU finishes, so a skewed router stalls the whole batch.
- **`N = 8` is not an accident.** It shards cleanly onto 8 GPUs, one expert each — the same
  reasoning as Llama 3 pinning 8 KV heads ([08b §5](08b_llama3_end_to_end.md)).

---

## 9. Comparison

| | Dense 47B | MoE 47B (8×, K=2) | Dense 13B |
|---|---|---|---|
| Total parameters | 46.7B | 46.7B | 13B |
| Active per token | 46.7B | **12.9B** | 13B |
| FLOPs / token | 93.4 GFLOP | **25.8 GFLOP** | 26 GFLOP |
| Memory (fp16) | 87 GiB | **87 GiB** | 24 GiB |
| Quality | ~47B | **~40B class** | ~13B |
| Training stability | straightforward | **needs `L_aux`, z-loss, capacity tuning** | straightforward |

**MoE sits where you want big-model quality at small-model latency and can afford big-model memory.**

---

## 10. Quick reference

```
router     h = x @ W_r          (d_model x N),  softmax, take top-K
gates      renormalise the K kept probabilities to sum to 1
output     sum_k gate_k * E_k(x)          only K of N experts run

L_aux      alpha * N * sum_i f_i * P_i    f = hard counts, P = soft probs
           minimum = alpha at perfect balance
capacity   capacity_factor * tokens * K / N ; overflow tokens SKIP the FFN

Mixtral    46,702,792,704 total | 12,879,925,248 active | 3.63x | memory IDENTICAL to dense
```

**The seven things to be able to say cold:**

1. **MoE buys compute, not memory.** 46.7B MoE and 46.7B dense both occupy 87 GiB; the MoE does
   **3.63× fewer FLOPs** per token. Cheaper in latency, *not* in GPU count.
2. **"8×7B" ≠ 56B.** Attention, embeddings and norms are **shared**; only the FFN is replicated.
   Hence 46.7B.
3. **Gates must be renormalised** over the kept experts — `[0.404873, 0.271394]` → `[0.598688,
   0.401312]`. Skip it and the FFN output is silently scaled by a token-dependent factor.
4. **`L_aux = α·N·Σ fᵢPᵢ`**, minimised at `α` when balanced (`0.010000`), rising to `0.068000` under
   collapse. `fᵢ` is a non-differentiable count and `Pᵢ` carries the gradient — that pairing is why
   the loss is a *product*.
5. **Routing collapses without it**, because the router only gets signal about experts it already
   picked.
6. **Capacity is a hard budget**; overflow tokens **skip the FFN** and pass through the residual.
   `capacity_factor × tokens × K / N`.
7. **`N = 8` shards onto 8 GPUs**, one expert each. Under expert parallelism, load imbalance becomes
   a *latency* problem — the step waits for the busiest GPU.

---

## See also

- [06b_gpt2_end_to_end.md](06b_gpt2_end_to_end.md) — why the FFN is two-thirds of a layer, i.e. why MoE targets it
- [04b_attention_at_scale_end_to_end.md](04b_attention_at_scale_end_to_end.md) — memory- vs compute-bound, which decides when MoE pays
- [08b_llama3_end_to_end.md](08b_llama3_end_to_end.md) — SwiGLU's three matrices, used in the expert arithmetic above
- [../../2.deep learning/02_architectures/09_mixture_of_experts.md](../../2.deep%20learning/02_architectures/09_mixture_of_experts.md) — MoE as a general ML technique
- [../../6.llms/05_vllm_internals.md](../../6.llms/05_vllm_internals.md) — serving MoE in practice

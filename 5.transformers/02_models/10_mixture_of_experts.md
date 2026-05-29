# Mixture of Experts (MoE)

> Used in: GPT-4, Mistral 8×7B, Gemini 1.5, Switch Transformer, GLaM

---

## The Core Idea

```
Dense model: every token passes through ALL parameters
  LLaMA-7B: 7B params, every forward pass uses all 7B

MoE model: each token is routed to only K of N expert FFN layers
  Mistral 8×7B: 8 experts per layer, each token uses 2 experts
  Total params:  46.7B (8 experts × ~5.7B each)
  Active params per token: ~12.9B  (2 experts active)

Result: near-13B quality at ~7B compute cost
(you pay for 2 experts, but benefit from 8 experts' specialized knowledge)
```

---

## Architecture

```
Standard Transformer FFN Layer:
  x → Linear(d, 4d) + GELU + Linear(4d, d) → output
  (one FFN, all tokens use it)

MoE Transformer FFN Layer:
  x → Router → select top-K experts → weighted sum of expert outputs

  Experts:  E_1, E_2, ..., E_N   (each is an independent FFN: d→4d→d)
  Router:   Linear(d, N) + softmax + top-K indices + weights

Standard attention layers are SHARED (not multiplied)
Only the FFN layers are replaced by MoE
```

### Router in Detail

```python
Input token: x ∈ ℝ^d   (d = 4096 for Mistral)

Router scores:
  h(x) = Linear(d, N)(x)      # N = 8 for Mistral
  h(x) ∈ ℝ^N                  # 8 raw logits

Softmax over all experts:
  p(x) = softmax(h(x))         # 8 probabilities, sum to 1

Top-K selection (K=2):
  top2_indices  = argsort(p(x))[-2:]         → [3, 6]  (expert 3 and 6)
  top2_weights  = softmax(p(x)[top2_indices]) → [0.62, 0.38]
                  ↑ re-normalize top-2 to sum to 1

Expert computation:
  output = 0.62 × E_3(x) + 0.38 × E_6(x)
```

---

## Forward Pass Dry Run

```
Token: "invoice"
d = 4096, N = 8 experts, K = 2

Step 1 — Router Logits:
  h(x) = Linear(4096, 8)(x)
  h(x) = [-0.3, 0.8, -1.2, 2.1, 0.4, -0.6, 1.7, 0.2]

Step 2 — Softmax:
  p(x) = softmax(h) = [0.04, 0.11, 0.02, 0.40, 0.07, 0.03, 0.28, 0.06]
                                              ↑ expert 3        ↑ expert 6

Step 3 — Top-2:
  Expert 3: p=0.40  ← highest
  Expert 6: p=0.28  ← second highest
  Re-normalize [0.40/(0.40+0.28), 0.28/(0.40+0.28)] = [0.588, 0.412]

Step 4 — Compute:
  E_3("invoice") = vector v_3 ∈ ℝ^4096
  E_6("invoice") = vector v_6 ∈ ℝ^4096

Step 5 — Weighted sum:
  output = 0.588 × v_3 + 0.412 × v_6

Step 6 — Interpretation:
  Expert 3 may have specialized in financial/document tokens (learned during training)
  Expert 6 may have specialized in entity/noun tokens
  Both contribute to "invoice" representation
```

---

## Mistral 8×7B — Key Numbers

```
Architecture:
  N = 8 experts per FFN layer
  K = 2 active experts per token
  Layers: 32 transformer layers
  d = 4096 (hidden dim)
  Each expert FFN: 4096 × 14336 × 4096

Parameter breakdown:
  Attention layers (shared):   ~2.3B params
  FFN experts (8 per layer):  ~44.4B params (32 layers × 8 experts × ~174M each)
  Total:                        46.7B params
  Active per token:             ~12.9B  (2 experts + shared attention)

Serving cost: similar to a 12-13B dense model
Quality:      similar to a 30-40B dense model
→ MoE gives ~3× efficiency gain
```

---

## 2024-2025 MoE Landscape

| Model | Year | Experts | Active per token | Total params | Active params | Notes |
|-------|------|---------|-----------------|-------------|---------------|-------|
| Switch Transformer (Google) | 2021 | 128-2048 | 1 | up to 1.6T | ~7B | Single-expert routing |
| Mistral 8×7B | 2023 | 8 | 2 | 47B | 13B | Standard reference MoE |
| Mistral 8×22B | 2024 | 8 | 2 | 141B | 39B | Bigger experts |
| DBRX (Databricks) | 2024 | 16 | 4 | 132B | 36B | Fine-grained experts |
| Snowflake Arctic | 2024 | 128 | 2 | 480B | 17B | Hybrid dense + MoE FFN |
| Qwen1.5-MoE-A2.7B | 2024 | 64 | 4 + shared | 14B | 2.7B | Shared expert pattern |
| DeepSeek-V2 | 2024 | 160 routed + 2 shared | 6 routed + 2 shared | 236B | 21B | Fine-grained + MLA |
| DeepSeek-V3 | 2024 | 256 routed + 1 shared | 8 routed + 1 shared | 671B | 37B | Aux-loss-free balancing + MLA + FP8 training |
| Llama 4 (rumored) | 2025+ | TBD | TBD | TBD | TBD | Expected to be MoE |

**Key 2024 trends:**
- **Fine-grained experts** (256+ instead of 8) — finer specialization, better routing diversity
- **Shared experts** — always-on expert(s) capture general knowledge; routed experts specialize
- **Auxiliary-loss-free balancing** (DeepSeek-V3) — instead of penalizing imbalance with an aux loss, bias the router with per-expert affinity scores that update based on observed load. Avoids accuracy degradation from heavy aux loss
- **MLA (Multi-head Latent Attention)** — compresses KV cache via low-rank projection; paired with MoE in DeepSeek for both compute and memory efficiency

---

## Load Balancing Problem

**The collapse problem:** without constraints, the router learns to always send tokens to the same 1-2 experts (they get more gradient signal → get better → get more tokens → collapse).

```
Without load balancing:
  Expert 1: 90% of tokens  → over-trained
  Experts 2-8: 1% each     → under-trained, wasted capacity

With auxiliary loss:
  Force roughly equal token distribution across all experts
```

### Auxiliary Load Balancing Loss

```python
def load_balancing_loss(router_probs, expert_indices, n_experts):
    """
    router_probs:   (batch×seq, n_experts) — softmax probabilities
    expert_indices: (batch×seq, K)          — selected expert indices
    """
    # Fraction of tokens routed to each expert
    # (ideal: 1/n_experts = 0.125 for 8 experts)
    expert_mask      = F.one_hot(expert_indices, n_experts).float()  # (B×S, K, E)
    tokens_per_expert = expert_mask.sum(dim=(0, 1))                   # (E,)

    # Average router probability per expert
    router_prob_per_expert = router_probs.mean(dim=0)                 # (E,)

    # Loss = n_experts × dot(tokens_per_expert, avg_router_prob)
    # Minimized when all are uniform (1/n_experts each)
    aux_loss = n_experts * (tokens_per_expert * router_prob_per_expert).sum()
    return aux_loss

# Total loss
main_loss  = cross_entropy(logits, labels)
total_loss = main_loss + 0.01 * load_balancing_loss(...)
#                         ↑ coefficient (0.01-0.001 typical)
```

**Dry run — balanced vs collapsed:**

```
8 experts, 1000 tokens routed

Balanced (ideal):
  tokens_per_expert = [0.125, 0.125, 0.125, ..., 0.125]   (all equal)
  router_prob_per_expert = [0.125, 0.125, ..., 0.125]
  aux_loss = 8 × (8 × 0.125 × 0.125) = 8 × 0.125 = 1.0   ← minimum

Collapsed:
  tokens_per_expert    = [0.94, 0.01, ..., 0.01]
  router_prob_per_expert = [0.85, 0.02, ..., 0.02]
  aux_loss = 8 × (0.94×0.85 + 7×0.01×0.02)
           = 8 × (0.799 + 0.0014)
           = 6.40   ← higher, penalized
```

---

## MoE vs Dense — Comparison

| Property | Dense (LLaMA-7B) | MoE (Mistral 8×7B) |
|----------|-----------------|-------------------|
| Total params | 7B | 46.7B |
| Active params/token | 7B | 12.9B |
| Training FLOPs | 1× | ~1.7× |
| Inference memory | 14GB | 93GB (all experts) |
| Inference compute | 1× | ~1.7× |
| Quality | ~7B dense | ~30-40B dense |

**The catch:** MoE needs ALL experts in memory during inference, even though only K=2 are used per token. Mistral 8×7B needs ~93GB VRAM at FP16 — requires 2× A100 80GB for inference. Quantized (4-bit): ~24GB — fits on 2× RTX 3090.

---

## Switch Transformer (Google, 2021)

The first scaled MoE transformer — simplified routing (top-1 instead of top-2):

```
K=1 (route each token to exactly ONE expert):
  Simpler: no re-normalization needed
  Expert capacity: each expert has a "capacity" C = (tokens/experts) × capacity_factor
  If expert is over capacity: token dropped (no expert computation, uses residual)
  capacity_factor=1.25 → 25% slack

Switch Transformer result:
  7× training speedup vs T5-11B at same quality
  Key: simplicity of K=1 routing, capacity factor handles imbalance
```

---

## Expert Specialization

After training, experts specialize in different input types — not explicitly programmed; emerges naturally:

```
Observed patterns in MoE language models:

  Expert A: mathematical expressions, equations, numbers
  Expert B: code tokens (keywords, brackets, operators)
  Expert C: proper nouns, named entities
  Expert D: common function words (the, a, of, in)
  Expert E: document/legal/financial terminology
  Expert F: question words, interrogatives
  Expert G: punctuation, formatting tokens
  Expert H: multilingual tokens, non-English text

Token "invoice":
  High router weight → Expert E (financial) and Expert C (entity)

Token "def":
  High router weight → Expert B (code) and Expert D (common words)
```

---

## Serving MoE Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Mistral — needs 2× A100 80GB or offloading
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-8×7B-Instruct-v0.1",
    torch_dtype=torch.float16,
    device_map="auto",    # splits across available GPUs
)

# With 4-bit quantization (fits on ~24GB)
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-8×7B-Instruct-v0.1",
    quantization_config=bnb_config,
    device_map="auto",
)
# Memory: ~24GB (fits 2× RTX 2090 or 1× A100 80GB with room to spare)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-8×7B-Instruct-v0.1")
inputs  = tokenizer("Extract invoice number from: INV-2024-0432", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

---

## GPT-4 as MoE (Rumored Architecture)

GPT-4 architecture was never officially disclosed, but the widely cited leaked information suggests:

```
~1.8T total parameters
16 experts per MoE layer
2 active experts per token
~220B active params per forward pass
8 model replicas trained independently, then ensembled
```

This matches the MoE efficiency pattern: massive total capacity, fraction used per token.

---

## Gotchas

**Memory vs compute mismatch.** MoE is compute-efficient (only K experts run) but memory-inefficient (all N experts must be loaded). For batch size 1 inference, MoE is wasteful — experts sit idle. MoE shines for large-batch throughput where expert utilization can be spread.

**Expert dropping.** When token volume exceeds expert capacity, some tokens are "dropped" (no expert runs, residual connection used). This creates a hard performance ceiling at high sequence lengths. capacity_factor=1.5 adds buffer but wastes expert slots.

**Training instability.** MoE models are harder to train — router can collapse or oscillate. Auxiliary load balancing loss coefficient needs tuning per model. Too high → hurts task performance. Too low → expert collapse.

**Communication overhead in distributed training.** Tokens get routed to experts potentially on different GPUs — all-to-all communication required. This is the main scaling challenge for MoE in distributed settings.

---

## Interview Q&A

**Q: Explain MoE and why it's efficient.**

A MoE (Mixture of Experts) replaces dense FFN layers with N expert FFN networks and a learned router. For each token, the router selects the top-K experts (typically K=2) and computes a weighted sum of their outputs. Mistral 8×7B has 46.7B total parameters but only ~12.9B activate per token — similar compute to a 13B dense model, but with the knowledge capacity of a much larger model. The efficiency gain: you pay for K experts' compute but benefit from N experts' specialized representations.

**Q: What is the load balancing problem in MoE?**

Without regularization, the router collapses — it learns to always send tokens to the same 1-2 experts because those experts get more gradient signal, become better, and thus receive more tokens. This is a feedback loop leading to most experts being unused (wasted capacity). Solution: add an auxiliary load balancing loss that penalizes unequal token distribution across experts. The loss is minimized when each expert receives roughly 1/N of all tokens.

**Q: MoE vs dense — when is each better?**

MoE better when: large batch inference (experts are well-utilized, training at scale (FLOPs-efficient for same quality), and you have the memory budget to hold all experts. Dense better when: single-request inference (most experts idle), memory constrained (MoE needs N× memory), simpler deployment, or latency-sensitive (routing overhead + potential expert imbalance).

---

## Connections

- **Transformer FFN details:** `5.transformers/01_fundamentals/02_transformer_architecture.md`
- **Efficient transformers:** `5.transformers/02_models/04_efficient_transformers.md`
- **LLM architectures:** `5.transformers/02_models/08_modern_llm_architecture.md`
- **vLLM serving MoE:** `10.mlops/10_serving_optimization_end_to_end.md`

---

## Key Takeaway

MoE = N expert FFN layers + router that sends each token to top-K experts. Mistral 8×7B: 46.7B total, 12.9B active — 13B compute cost, ~35B quality. Key challenge: load balancing (auxiliary loss forces equal token distribution). GPT-4 reportedly uses MoE (~1.8T total, 16 experts/layer, 2 active). Specialization emerges naturally — code tokens go to code experts, financial tokens to finance experts.

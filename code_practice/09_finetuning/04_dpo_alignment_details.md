# Session 4 — DPO Alignment
Status: `🔧 Code-built`

Theory: [../../../6.llms/03_alignment.md](../../6.llms/03_alignment.md) · [../../../6.llms/06_alignment_follow_ups.md](../../6.llms/06_alignment_follow_ups.md)

---

## Use Case

The LoRA fine-tuned model answers factually but sounds dry and generic. DPO teaches it to prefer responses that include specific figures, professional tone, and complete coverage — matching the style you want without writing a reward model.

---

## DPO vs RLHF

| | RLHF | DPO |
|-|------|-----|
| Needs reward model | Yes (separate training run) | No |
| Training stability | Tricky (PPO) | Simple (supervised-like) |
| Compute | 3 models in memory | 2 models (policy + ref) |
| Quality | Gold standard | ~Equivalent on most tasks |
| Code complexity | High | Low (10 lines with trl) |

DPO derives the reward signal implicitly from the preference data — no separate reward model needed.

---

## The DPO Objective

```
loss = -log σ(β · (log π(chosen)/πref(chosen) - log π(rejected)/πref(rejected)))

Where:
  π     = policy model (the one being trained)
  πref  = reference model (frozen copy of π before DPO)
  β     = KL penalty strength (0.1 = standard)

Intuition:
  Push π(chosen) UP relative to πref (LLM assigns higher prob to good response)
  Push π(rejected) DOWN relative to πref (LLM assigns lower prob to bad response)
  β controls how far π can deviate from πref (prevents reward hacking)
```

---

## β Parameter Guide

| β | Effect | Risk |
|---|--------|------|
| 0.01 | Very aggressive alignment | Forgetting general capability |
| 0.1 | Standard (DPO paper default) | Balanced |
| 0.5 | Conservative | Minimal alignment benefit |

---

## Dataset Format

```json
[
  {
    "prompt": "What is the maximum LTV for a first-time buyer?",
    "chosen": "First-time buyers can borrow up to 95% LTV under Help to Buy. On a £300,000 property that requires a minimum £15,000 (5%) deposit.",
    "rejected": "You can get a mortgage as a first-time buyer. There are various loan options available to you."
  }
]
```

**chosen vs rejected difference:** both answer the question but chosen has exact numbers, specific scheme name, and a worked example. Rejected is vague.

---

## Expected Output

```
Device: mps
β (KL penalty): 0.1

trainable params: 302,080 || all params: 125,843,456 || trainable%: 0.2402

Training DPO...
{'loss': 0.6932, 'rewards/chosen': -0.042, 'rewards/rejected': 0.031}
{'loss': 0.5821, 'rewards/chosen': 0.183, 'rewards/rejected': -0.124}
{'loss': 0.4903, 'rewards/chosen': 0.421, 'rewards/rejected': -0.289}

DPO adapter saved to models/09_finetuning/dpo_opt125m

── DPO training objective ──
  Decreasing loss → model assigns higher probability to chosen over rejected
  β=0.1: moderate KL penalty
  β too low  (<0.05): risks reward hacking / losing general capability
  β too high (>0.5):  conservative — minimal alignment benefit
```

---

## How to Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE python 04_dpo_alignment.py
```

Run session 03 first to generate domain-specific DPO pairs. Script falls back to synthetic pairs if not found.
MPS training time: ~15–20 min for 2 epochs on 200 pairs.

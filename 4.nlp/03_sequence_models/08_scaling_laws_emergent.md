# Scaling Laws + Emergent Abilities + In-Context Learning

> The reason LLMs went from "neat NLP toy" to "global infrastructure" between 2018 and 2024. Chinchilla, the GPT-3 inflection, instruction tuning.

---

## Table of Contents

1. Objective
2. Kaplan scaling laws → Chinchilla
3. The GPT-3 inflection — in-context learning
4. Emergent abilities
5. Instruction tuning and alignment
6. Failure modes
7. Interview questions (5)
8. Further reading

---

## 1. Objective

The leap from BERT (2018) to ChatGPT (2022) wasn't a single algorithm — it was a sequence of scaling discoveries:

1. Performance scales predictably with parameters, data, and compute (Kaplan 2020)
2. Above ~7B params, NEW capabilities emerge — in-context learning, multi-step reasoning
3. Compute-optimal scaling — Chinchilla (2022): more DATA matters more than more PARAMS
4. Instruction tuning + RLHF turns base models into useful assistants

Senior interview Q: "What is the Chinchilla scaling law and why did it change LLM development?"

---

## 2. Kaplan Scaling Laws → Chinchilla

### Kaplan et al. 2020 — "Scaling Laws for Neural Language Models"

The first formal scaling law:

```
Loss ∝  N^(-α) + D^(-β) + C
where  N = parameters, D = training tokens
       α, β fitted from experiments
```

Implication: loss scales predictably with size. Doubling N gives a predictable loss reduction. Same for D.

**Kaplan's key conclusion: train BIG models on less data.** With a fixed compute budget, prefer bigger model.

OpenAI used this to scale GPT-2 (1.5B) → GPT-3 (175B). Worked spectacularly.

### Chinchilla (Hoffmann et al. 2022) — the correction

DeepMind re-ran the experiments more carefully. They found Kaplan UNDER-estimated the value of data. The correct compute-optimal recipe:

```
For a fixed compute budget C = 6 × N × D:
  Optimal N ∝ C^0.5    (scale params with square root of compute)
  Optimal D ∝ C^0.5    (scale data with square root of compute)
  N and D should scale together — roughly equally.

The optimal ratio is roughly D/N ≈ 20.
```

GPT-3: 175B params, 300B tokens. Ratio of D/N = 1.7. **Chinchilla recipe says optimal: D/N = 20.** GPT-3 was 10× under-trained on data.

DeepMind trained Chinchilla: 70B params, 1.4T tokens (D/N=20). It BEAT GPT-3 on most benchmarks despite being 2.5× smaller.

### The post-Chinchilla landscape

- **LLaMA-2 7B**: 7B params, 2T tokens — D/N = 285 (Meta SCALED PAST Chinchilla optimal)
- **LLaMA-3 8B**: 8B params, 15T tokens — D/N = 1875

For inference cost reasons, modern open models train BEYOND Chinchilla optimal. The marginal loss reduction per token decreases, but inference is dirt cheap per token, so the over-trained smaller models are economically dominant.

**Senior interview answer:** "Chinchilla says scale data and params equally; Llama-3 ignored this and trained smaller models on MUCH more data because inference cost dominates total LLM economics."

---

## 3. The GPT-3 Inflection — In-Context Learning

GPT-2 (2019) was a language model. You could fine-tune it for downstream tasks.

GPT-3 (Brown et al. 2020) showed something new: **without fine-tuning, with just FEW-SHOT EXAMPLES in the prompt, the model performs reasonably on new tasks.**

```
Prompt: "Translate to French.
  sea otter => loutre de mer
  peppermint => menthe poivrée
  cheese => "

GPT-3 completion: "fromage"
```

No fine-tuning. The model learned to do translation just from 2 examples. This was qualitatively new.

### Why this matters

**Before GPT-3:** every new task required a labeled dataset and fine-tuning. **After GPT-3:** you can write a prompt and get usable results. The model adapts to the task at inference time.

This is **in-context learning (ICL)** — the model "learns" from examples in its context window, without weight updates.

### Where ICL emerges

GPT-3 paper showed ICL improves dramatically from 1B → 13B → 175B. Small models can't do ICL effectively; big models can.

The "learning" happens in the forward pass — the model conditions on examples in its context to infer the task pattern. Emerges around 7B+ parameters; absent in smaller models.

It's an emergent ability, which leads to...

---

## 4. Emergent Abilities

Wei et al. 2022 ("Emergent Abilities of Large Language Models"): some capabilities appear suddenly at a critical model size and are basically absent below.

**Examples:**
- **Arithmetic (3-digit addition)** — near-zero at < 10B params, then sharp jump
- **Chain-of-thought reasoning** — requires ~60B+ to work reliably
- **Multi-step instruction following** — emerges around 100B+
- **Code generation** — improves smoothly with scale

### The controversy

Schaeffer et al. 2023 ("Are Emergent Abilities a Mirage?") argued: emergent abilities are ARTIFACTS of using discrete metrics (e.g., exact-match accuracy). When you use continuous metrics, performance scales smoothly.

This is the more nuanced 2023+ view. Some capabilities still appear to have phase transitions; others were measurement artifacts.

### Practical takeaway

Don't expect a 1B model to do chain-of-thought. Some capabilities need scale. Pick the model size based on whether your target capability has emerged at that size.

---

## 5. Instruction Tuning and Alignment

GPT-3 was a base model — completed text but didn't follow instructions well.

```
Base model prompt: "How do I bake a cake?"
Base model output: "How do I bake a cake? How do I make bread? How do I ..."  (continuation)
```

It just predicted what tokens follow. Not a useful assistant.

### Instruction tuning (Wei et al. 2021, "FLAN")

Fine-tune the model on (instruction, response) pairs across many tasks. Result: the model learns to FOLLOW instructions rather than continue them.

```
Base + instruction tuning: "How do I bake a cake?"
Response: "1. Preheat oven to 350°F. 2. Mix flour, sugar, eggs..."
```

This is the SFT phase in modern LLM training. Datasets used: FLAN, Super-NaturalInstructions, Self-Instruct.

### RLHF (Ouyang et al. 2022, "InstructGPT")

After SFT, the model is helpful but not always preferred by humans. RLHF aligns outputs to human preference:

1. Train a reward model on (prompt, response_A, response_B, human_preference) triples
2. Use PPO to fine-tune the LLM to maximize reward model scores: 3 + KL constraint to stay close to the SFT model (avoid reward hacking)

InstructGPT's surprising finding: **a 1.3B InstructGPT-tuned model was preferred over the 175B GPT-3 base.** Alignment > scale for usefulness.

This was the recipe behind ChatGPT.

### DPO replaces RLHF

2023+: DPO (Rafailov et al. 2023) achieves the same alignment without PPO's complexity. Modern open-source pipelines almost all use DPO instead. See `6.llms/06_alignment_follow_ups.md` for ORPO/KTO/IPO.

---

## 6. Failure Modes

1. **Loss is going down — must be improving capability** — not always. Loss reduction can come from format / fluency improvements while ICL capability stays flat. Always measure downstream tasks, not just loss.

2. **Following Chinchilla too literally on inference-optimized models** — LLaMA-3 etc. deliberately over-train for inference economics. Chinchilla-optimal is for compute-budget-bound training, not for deployment.

3. **Assuming all "emergent" abilities are real** — many were artifacts of discontinuous metrics. Re-measure with continuous proxies (likelihood of correct tokens, partial credit) before claiming emergence.

4. **Instruction tuning on narrow data** — destroys general capability. Always mix domain data with general instruction data (FLAN, Alpaca-mix).

5. **Reward hacking in RLHF** — model produces overly-long, hedged, polite responses that humans rate highly but are less useful. KL constraint helps; DPO is more stable.

---

## 7. Interview Questions (5)

**Q1: What's the Chinchilla scaling law?**

For a fixed compute budget C = 6ND (N = params, D = training tokens), the compute-optimal allocation is N ∝ √C and D ∝ √C — scale them equally. The optimal ratio is roughly D/N ≈ 20. GPT-3 had D/N = 1.7, so it was 10× under-trained. DeepMind's Chinchilla (70B, 1.4T tokens) beat GPT-3 (175B, 300B tokens) despite being 2.5× smaller.

**Q2: Why do modern open models like LLaMA-3 ignore Chinchilla?**

Inference cost matters more than training cost over a model's lifetime. A smaller model trained on more data has same/similar quality but is much cheaper to serve. LLaMA-3 8B trained on 15T tokens (D/N=1875, far above Chinchilla's 20) — quality close to 70B Chinchilla, inference cost ~10× lower.

**Q3: What is in-context learning?**

GPT-3 showed that with just FEW-SHOT EXAMPLES in the prompt, a large model can perform new tasks without weight updates — the model conditions on examples in its context to infer the task pattern. Emerges around 7B+ parameters; absent in smaller models.

**Q4: What are "emergent abilities" and what's the controversy?**

Some LLM capabilities (arithmetic, multi-step reasoning) appear absent below a certain size, then sharply emerge above. Wei et al. 2022 documented this. Schaeffer et al. 2023 challenged the framing: many "emergent" abilities are artifacts of using discrete accuracy metrics — using continuous metrics (likelihood-based), many scaling curves are smooth. Some emergence may be real; much is measurement artifact.

**Q5: Walk me through the recipe that made ChatGPT from scratch.**

(1) Pretrain a base LLM on internet text (next-token prediction, billions of tokens; scaling-law optimized). (2) Supervised Fine-Tune (SFT) on instruction-response pairs across many tasks (FLAN, Self-Instruct). (3) Train a reward model on human preference pairs (prompt, response_A, response_B, human label). (4) RLHF — use PPO to fine-tune the LLM against the reward model, with KL constraint to the SFT model. The 1.3B SFT model was preferred over 175B base GPT-3 — alignment beats scale for usefulness.

---

## 8. Further Reading

- Kaplan et al. 2020 — "Scaling Laws for Neural Language Models" — arXiv:2001.08361
- Brown et al. 2020 / GPT-3 — "Language Models are Few-Shot Learners" — arXiv:2005.14165
- Hoffmann et al. 2022 / Chinchilla — "Training Compute-Optimal Large Language Models" — arXiv:2203.15556
- Wei et al. 2022 — "Emergent Abilities of Large Language Models" — arXiv:2206.07682
- Schaeffer et al. 2023 — "Are Emergent Abilities a Mirage?" — arXiv:2304.15004
- Ouyang et al. 2022 / InstructGPT — arXiv:2203.02155 — RLHF recipe
- Wei et al. 2021 / FLAN — arXiv:2109.01652 — instruction tuning origin

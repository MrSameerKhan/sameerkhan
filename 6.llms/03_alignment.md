# LLM Alignment (RLHF, DPO)

> Alignment = making LLMs do what humans want. RLHF does this via human preference data → reward model → PPO. DPO simplifies RLHF by directly optimizing preferences without a separate reward model — same quality, much simpler. In practice: SFT first, then DPO. The KL penalty is critical to prevent reward hacking. Constitutional AI / RLAIF scales this with AI-generated feedback instead of human labels.

---

## Quick Reference

| Method | Key Idea | Models Needed | Stability |
|--------|----------|---------------|-----------|
| RLHF (PPO) | Human prefs → reward model → RL | SFT + RM + PPO policy + ref | Hard to tune |
| DPO | Skip RM; optimize prefs directly | SFT + ref (frozen) | Stable |
| IPO | DPO variant with regularization | SFT + ref | More stable |
| ORPO | SFT + DPO in one step | SFT only | Simplest |
| KTO | Binary feedback (good/bad) | SFT + ref | Stable |
| GRPO | Group relative policy opt (reasoning) | SFT + verifiable reward | Stable |
| RLOO | REINFORCE Leave-One-Out | SFT + reward signal | Moderate |
| Constitutional AI | AI-generated critiques as training signal | SFT + strong LLM judge | Stable |
| RLAIF | AI feedback replaces human labelers | SFT + RM + PPO | Hard |

**2025 practical hierarchy:**
1. SFT on demonstration data
2. DPO or ORPO for general alignment (preference data)
3. KTO when binary good/bad feedback is available
4. GRPO / RLOO when there's a verifiable reward (math, code)
5. PPO still appears in some pipelines (Llama-3 alignment)

**Goal of alignment:** Make LLMs helpful, harmless, and honest (HHH).

**Reasoning model alignment (o1, o3, DeepSeek-R1):** RLVR on verifiable rewards + GRPO — covered in `../5.transformers/models/14_reasoning_models.md`.

---

## Core Concepts

### The Alignment Problem

```
Pretrained LLM objective: predict next token on internet text
→ Model learns to complete any text, including harmful text
  - "How do I make a bomb?" → model completes with instructions (it's seen this online)
  - "2+2=" might produce "5" if that's the training continuation

What we want: helpful assistant that:
1. Follows instructions accurately
2. Refuses genuinely harmful requests
3. Admits uncertainty instead of hallucinating
4. Maintains consistent persona

Gap: Pretraining maximizes token prediction. Alignment bridges pretraining + assistant behavior.
```

---

## RLHF (Reinforcement Learning from Human Feedback)

### Three Stages

```
Stage 1: Supervised Fine-Tuning (SFT)
  - Collect high-quality [prompt, ideal_response] pairs from human writers
  - Fine-tune pretrained LLM on these demonstrations
  - Result: base instruction-following capability

Stage 2: Reward Model Training
  - For each prompt, collect k model responses
  - Human labelers rank responses (best → worst)
  - Train reward model R(prompt, response) → scalar score
  - Training signal: preferred response should score higher than rejected

Stage 3: RL Fine-Tuning (PPO)
  - Use reward model to score LLM outputs
  - Use PPO to update LLM to maximize reward
  - KL penalty: don't deviate too far from SFT model
  - Objective: E[R(response)] - β·KL(π_θ || π_SFT)
```

### Reward Model Training

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Reward model: takes prompt + response → scalar reward
class RewardModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1  # single scalar output
        )

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask).logits.squeeze(-1)

# Training with preference pairs
def bradley_terry_loss(reward_chosen, reward_rejected):
    """
    Bradley-Terry model: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    Loss: -log P(chosen > rejected) = -log σ(r_chosen - r_rejected)
    """
    return -torch.nn.functional.logsigmoid(reward_chosen - reward_rejected).mean()

# Training loop
for batch in dataloader:
    r_chosen = reward_model(batch['chosen_input_ids'], batch['chosen_attention_mask'])
    r_rejected = reward_model(batch['rejected_input_ids'], batch['rejected_attention_mask'])
    loss = bradley_terry_loss(r_chosen, r_rejected)
    loss.backward()
    optimizer.step()
```

### PPO for LLMs (simplified)

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Value head: estimates expected future reward from current state
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft-model")  # frozen
reward_model = load_reward_model()

ppo_config = PPOConfig(
    learning_rate=1.4e-5,
    batch_size=128,
    mini_batch_size=128,
    gradient_accumulation_steps=1,
    ppo_epochs=4,
    kl_penalty="kl",          # KL divergence penalty from reference model
    init_kl_coef=0.2,         # β: weight of KL penalty
    target_kl=6.0,            # adaptive: adjust β to keep KL near target
    cliprange=0.2,            # PPO clip ratio ε
)

trainer = PPOTrainer(config=ppo_config, model=model, ref_model=ref_model, tokenizer=tokenizer)

for batch in dataloader:
    query_tensors = batch["input_ids"]

    # Generate responses from current policy
    response_tensors = trainer.generate(query_tensors, max_new_tokens=200)

    # Compute rewards using reward model
    rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]

    # PPO update
    stats = trainer.step(query_tensors, response_tensors, rewards)
```

### RLHF Problems

```
1. Reward hacking: model finds ways to get high reward ≠ what humans want
   e.g., very long verbose responses score higher (length bias in reward model)

2. Reward model errors: RM generalizes poorly → model exploits RM mistakes
   "Goodhart's Law: when a measure becomes a target, it ceases to be a good measure"

3. Expensive: requires human labelers for preference data + complex PPO training

4. Instability: PPO is notoriously hard to tune; can collapse or reward hack
```

---

## DPO (Direct Preference Optimization)

**Key insight:** Skip the reward model entirely. Optimize preferences directly.

```
RLHF insight: the optimal policy π* has a closed-form relationship with reward r:
r(x, y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)

This means: the reward is implicitly defined by the policy!
We don't need to train a separate reward model.
```

DPO rearranges the RLHF objective to directly use preference pairs:

```
L_DPO(π_θ) = -E[(x, y_w, y_l)] {
    log σ(
        β · log(π_θ(y_w|x) / π_ref(y_w|x))
        - β · log(π_θ(y_l|x) / π_ref(y_l|x))
    )
}

Where:
y_w = winning (preferred) response
y_l = losing (rejected) response
π_ref = reference SFT model (frozen)
β = controls deviation from reference

Interpretation: increase probability of preferred response relative to reference,
decrease probability of rejected response relative to reference.
```

### DPO Training Code

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# DPO requires:
# 1. SFT model (π_θ, trainable)
# 2. Reference model (π_ref, frozen — copy of SFT model)
# 3. Preference dataset: {"prompt": ..., "chosen": ..., "rejected": ...}

model = AutoModelForCausalLM.from_pretrained("sft-model")
ref_model = AutoModelForCausalLM.from_pretrained("sft-model")  # frozen copy
tokenizer = AutoTokenizer.from_pretrained("sft-model")

# Dataset format
preference_dataset = [
    {
        "prompt": "What is the capital of France?",
        "chosen": "Paris is the capital of France.",
        "rejected": "I'm not sure, but I think it might be Lyon."
    },
    # ... thousands more pairs
]

dpo_config = DPOConfig(
    beta=0.1,                            # temperature parameter
    learning_rate=5e-7,                  # very low LR — don't deviate too much
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,                  # typically 1-3 epochs
    max_prompt_length=512,
    max_length=1024,
    bf16=True,
    output_dir="./dpo-model",
)

trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### DPO vs RLHF

```
RLHF (PPO):
✓ More flexible (can use any reward signal, not just preferences)
✓ Online: generate new responses during training → better exploration
✗ Requires separate reward model
✗ PPO is complex, unstable, compute-intensive

DPO:
✓ Simpler: no reward model, no RL loop
✓ Stable: standard supervised training
✓ Same or better performance on most benchmarks
✗ Offline: uses fixed preference dataset
✗ Can't explore new responses → distribution shift
✗ Sensitive to reference model quality

In practice: DPO is now the default for alignment fine-tuning.
PPO when you have a clear scalar reward (e.g., code execution correctness, game score)
```

---

## ORPO (Odds Ratio Preference Optimization)

```
Key innovation: combine SFT + DPO into single training step
No separate reference model needed — uses within-batch contrast

L_ORPO = L_SFT + λ · L_OR

L_OR = -log(σ(log(odds_ratio(y_w) / odds_ratio(y_l))))

odds_ratio(y) = P(y|x) / (1 - P(y|x))

Benefits:
- Single training stage (SFT + alignment simultaneously)
- No reference model (saves memory)
- Simpler pipeline
- Competitive with DPO on benchmarks
```

---

## Constitutional AI (CAI)

Anthropic's approach to scalable oversight:

```
Step 1: Generate harmful/problematic responses (red-teaming)
  Prompt: "How can I hurt my sister?"
  Raw response: [potentially harmful]

Step 2: Critique using a "constitution" of principles
  Constitution: "Choose the response that is least likely to contain harmful content..."
  Critique: "This response is harmful because..."

Step 3: Revise response based on critique
  Revised: [harmless helpful response]

Step 4: Use these (harmful, revised) pairs as RLHF training data
  + RLAIF: Reinforcement Learning from AI Feedback
  + Scales feedback generation without expensive human labeling

Principles in constitution (examples):
  - "Prefer responses that are not harmful"
  - "Prefer responses that are honest about uncertainty"
  - "Prefer responses that don't assist with illegal activities"
```

```python
from anthropic import Anthropic

client = Anthropic()

CONSTITUTION = [
    "Prefer responses that don't assist with illegal activities.",
    "Prefer responses that are honest about uncertainty.",
    "Prefer responses that are helpful without causing harm.",
]

def constitutional_revision(harmful_prompt: str, harmful_response: str) -> dict:
    """Given a harmful (prompt, response) pair, generate a revised version."""

    # Step 1: Critique
    critique = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": f"""Consider this principle:
{CONSTITUTION[0]}

Critique this response and explain what's wrong with it.
Response: {harmful_response}"""}]
    ).content[0].text

    # Step 2: Revise
    revised = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": f"""Critique: {critique}

Revise the response to fix the issues identified while still being helpful.
Original question: {harmful_prompt}
Revised response:"""}]
    ).content[0].text

    return {
        "prompt": harmful_prompt,
        "chosen": revised,          # aligned response
        "rejected": harmful_response,  # original harmful response
    }
```

---

## KL Divergence Penalty — Why It's Critical

```
Without KL penalty: model maximizes reward by producing gibberish that
tricks the reward model (reward hacking)

KL penalty: constrain policy to stay close to reference (SFT) model
Objective: E[reward] - β · KL(π_θ || π_ref)

As β → 0: pure RL, ignores reference → reward hacking
As β → ∞: pure SFT, ignores reward signal → no alignment

β selection:
  - Monitor KL divergence during training
  - Target KL = 6-10 nats for typical RLHF
  - Adaptive: adjust β to keep KL near target_kl
```

---

## Gotchas

**Preference data quality is critical:** DPO/RLHF is only as good as the preference labels. If annotators have inconsistent standards or the chosen-rejected gap is ambiguous, the model learns noisy signal. Use clear annotation guidelines with inter-annotator agreement checks.

**Length bias in reward models:** Reward models often prefer longer responses (length correlates with perceived helpfulness). Mitigate: normalize reward by response length; include both short and long examples in RM training; add length penalty.

**Reference model drift:** If you update the reference model during DPO training, you're no longer minimizing the right objective. The reference model must stay frozen at the SFT checkpoint.

**DPO is sensitive to β:** Too low → model deviates wildly from reference; too high → model ignores preference signal. Typical range: 0.01-0.5. Tune on a held-out preference eval set.

**Don't skip SFT before DPO:** DPO requires a reasonable starting point (SFT model). Applying DPO directly to a pretrained base model doesn't work well — the model doesn't understand the instruction-following format yet.

---

## Interview Q&A

**Q: Explain RLHF. Why is it needed if we already have SFT?**

SFT teaches the model to mimic good responses but doesn't optimize for human preference in a nuanced way. SFT treats all tokens in the response equally, even though some responses are much better than others overall. RLHF adds: (1) a reward model trained on pairwise human preferences (which of two responses is better?); (2) RL (PPO) to optimize the LLM to generate responses that maximize this reward while staying close to the SFT baseline via KL penalty. SFT alone produces a model that follows instructions; RLHF produces one that follows them in the way humans actually prefer.

**Q: What is DPO and why is it preferred over PPO for alignment?**

DPO shows that the optimal RLHF policy π* has an equivalent closed-form description directly in terms of the policy probabilities — no separate reward model needed. The loss directly increases the likelihood of preferred responses and decreases the likelihood of rejected responses, relative to a frozen reference model. DPO is simpler (no RM training, no PPO loop), more stable (standard supervised training), and achieves comparable or better results than PPO on most benchmarks. PPO is still used when you have a clear scalar reward signal (code execution, game score) rather than preference pairs.

**Q: What is reward hacking and how is it prevented?**

Reward hacking occurs when the model finds responses that score high on the reward model but don't actually represent what humans want. Example: verbose responses that trick the RM's length bias, sycophantic agreements with the user, or generating special tokens that trick the RM. Prevention: (1) KL penalty keeps the policy close to the SFT reference — prevents drifting too far; (2) diverse RM training data covering adversarial cases; (3) regularization of the reward model; (4) monitoring KL divergence during training (if it spikes, the model is likely hacking); (5) red-teaming the trained model with adversarial prompts.

---

## Connections

- **LLM Fine-Tuning (6.llms/02):** SFT is stage 1 of RLHF; DPO builds on SFT checkpoint
- **LLM Prompting (6.llms/01):** Aligned models respond better to prompts — RLHF is why Claude/GPT follow instructions
- **LLM Evaluation (6.llms/06):** Alignment quality measured by safety/helpfulness benchmarks, LLM-as-judge
- **Reasoning models (5.transformers/models/14):** RLVR / GRPO for verifiable-reward alignment

---

## Key Takeaway

Alignment = making LLMs do what humans want. RLHF does this via human preference data → reward model → PPO. DPO simplifies RLHF by directly optimizing preferences without a separate reward model — same quality, much simpler. In practice: SFT first, then DPO. The KL penalty is critical to prevent reward hacking. Constitutional AI / RLAIF scales this with AI-generated feedback instead of human labels.

---

## Code Practice — Wired by Phase 6

- `code_practice/03_prompting/` — prompting baseline before alignment
- `6.llms/10_alignment_end_to_end.md` — RLHF + DPO full dry-run with numbers

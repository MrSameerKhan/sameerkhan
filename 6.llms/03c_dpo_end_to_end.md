# 03c — RLHF → DPO: The Derivation

> Board 20. Companions: [03_alignment.md](03_alignment.md) (the landscape),
> [03b_alignment_end_to_end.md](03b_alignment_end_to_end.md) (worked RLHF),
> [06_alignment_follow_ups.md](06_alignment_follow_ups.md).
>
> **This file exists for one step those files skip:** they state
> `r(x,y) = β·log(π*/π_ref) + β·log Z(x)` and then jump to the DPO loss, without explaining that
> **`β·log Z(x)` cancels**. That cancellation *is* DPO. §5 derives and verifies it.
>
> Board 18 (SFT) is [02c_sft_end_to_end.md](02c_sft_end_to_end.md) — alignment starts where it ends.

---

## 1. The three stages

```
1. SFT       teach the format          (board 18)
2. RM        learn what humans prefer
3. RLHF/DPO  optimise the policy toward those preferences
```

**SFT teaches the model to imitate; alignment teaches it to be preferred.** The difference matters:
SFT can only reach the quality of its demonstrations, because it is maximum likelihood on them.
Preference learning can exceed them — it only needs humans to *rank* outputs, which is far easier
than writing ideal ones.

---

## Table of Contents

1. The three stages
2. The reward model — Bradley-Terry
3. The RLHF objective
4. **What β is actually stopping**
5. **The DPO derivation — the cancellation**
6. Why DPO won in practice
7. What DPO gives up
8. Quick reference

---

## 2. The reward model — Bradley-Terry

Humans compare two responses rather than scoring one. Bradley-Terry converts comparisons into a
scalar reward:

```
P(y_w ≻ y_l | x)  =  σ( r(x,y_w) − r(x,y_l) )

L_RM  =  −log σ( r(x,y_w) − r(x,y_l) )
```

**The reward is only ever defined up to a difference.** Nothing pins its absolute scale — add a
constant to every reward for a given prompt and every preference probability is unchanged. Hold onto
that; §5 turns it into the whole trick.

The reward model is usually the SFT model with the LM head replaced by a scalar head.

---

## 3. The RLHF objective

```
max_π   E_{y~π}[ r(x,y) ]  −  β · KL( π(·|x) ‖ π_ref(·|x) )
        └─ maximise reward ─┘    └─ but stay near the SFT model ─┘
```

`π_ref` is the frozen SFT model. This objective has a **closed-form optimum**:

```
π*(y|x)  =  (1/Z(x)) · π_ref(y|x) · exp( r(x,y) / β )

Z(x) = Σ_y  π_ref(y|x) · exp( r(x,y)/β )        the partition function
```

Worked, with four candidate responses, `β = 0.5`:

```
π_ref   = [0.4000, 0.3000, 0.2000, 0.1000]
r       = [1.2000, 0.3000, -0.5000, 0.8000]

unnormalised π_ref·exp(r/β) = [4.409271, 0.546636, 0.073576, 0.495303]
Z(x)    = 5.524785
π*      = [0.798089, 0.098942, 0.013317, 0.089651]
```

Inverting recovers the reward exactly:

```
r(x,y) = β·log( π*(y|x) / π_ref(y|x) ) + β·log Z(x)

recovered = [1.2000, 0.3000, -0.5000, 0.8000]
true      = [1.2000, 0.3000, -0.5000, 0.8000]
max|diff| = 5.551e-17
```

**But `Z(x)` is intractable.** Here it sums over 4 strings; for a real language model it sums over
every possible response — `|V|^L`. You cannot compute it, which is why this identity looks useless.

---

## 4. What β is actually stopping

The board's second killer question. Sweep `β` and watch `π*`:

```
    β         A         B         C         D    KL(π* ‖ π_ref)
  ref    0.4000    0.3000    0.2000    0.1000
 10.00    0.4260    0.2920    0.1797    0.1023          0.0020
  2.00    0.5273    0.2521    0.1127    0.1079          0.0454
  1.00    0.6394    0.1950    0.0584    0.1072          0.1515
  0.50    0.7981    0.0989    0.0133    0.0897          0.3957
  0.20    0.9594    0.0080    0.0001    0.0325          0.7732
  0.05    0.9999    0.0000    0.0000    0.0001          0.9155
   →0    1.0000    0.0000    0.0000    0.0000             ∞
```

```
β large  ->  π* ≈ π_ref. The reward barely moves anything (KL 0.0020 at β=10).
β small  ->  π* collapses onto the argmax-reward response (KL 0.9155 at β=0.05).
β → 0    ->  a delta function on one response.
```

**What the KL term stops is reward hacking.** The reward model is a *learned, imperfect proxy* for
human preference. A policy free to maximise it will find its errors rather than genuinely improve —
degenerate outputs that score high and read badly. The classic symptoms are excessive length,
sycophancy, and hedging boilerplate, all of which reward models systematically over-score.

The KL term is a leash to the SFT model: *"get more preferred, but do not stop talking like the
model humans actually demonstrated."* `β` sets the leash length.

---

## 5. The DPO derivation — the cancellation

Here is the step the companion files omit.

**Bradley-Terry depends only on the *difference* of rewards.** Substitute the inverted closed form
for both responses:

```
r(x,y_w) − r(x,y_l)

  =  [ β·log(π*_w/π_ref_w) + β·log Z(x) ]  −  [ β·log(π*_l/π_ref_l) + β·log Z(x) ]
                              ^^^^^^^^^^^                              ^^^^^^^^^^^
                                            these are IDENTICAL

  =  β·log(π*_w/π_ref_w)  −  β·log(π*_l/π_ref_l)
```

Verified:

```
r_w − r_l                            = 1.700000
with both β·log Z terms kept         = 1.700000
with the Z terms dropped             = 1.700000

max|difference| = 0.000e+00
```

**`Z(x)` depends only on `x`, so it is the same for `y_w` and `y_l`, and it cancels exactly.** The
intractable term disappears — not approximated, *cancelled*.

That gives the DPO loss directly:

```
L_DPO = −log σ(  β·log( π_θ(y_w|x) / π_ref(y_w|x) )
               − β·log( π_θ(y_l|x) / π_ref(y_l|x) )  )
```

And it is *the same loss* as Bradley-Terry on the true rewards:

```
L_BT  = −log σ( r_w − r_l )                                = 0.167786
L_DPO = −log σ( β·log(π_w/ref_w) − β·log(π_l/ref_l) )      = 0.167786
identical
```

**So the policy is its own reward model.** `β·log(π_θ/π_ref)` is an implicit reward. There is nothing
left to train separately, and no RL.

> **The one-sentence version:** *the KL-constrained optimum lets you write the reward in terms of the
> policy, up to a term that depends only on the prompt — and since preference learning only ever
> uses reward **differences**, that term cancels.*

---

## 6. Why DPO won in practice

**Model count**, at Llama-3 8B (`8.03e9` params; bf16 weights, fp32 grads, Adam states for trainable):

```
PPO needs                    trainable?      GiB
  policy                          train    104.7
  reference (frozen)             frozen     15.0
  reward model (frozen)          frozen     15.0
  value model                     train    104.7
  TOTAL                                     239.3

DPO needs                    trainable?      GiB
  policy                          train    104.7
  reference (frozen)             frozen     15.0
  TOTAL                                     119.7
```

**2.00× less memory — and that understates it**, because DPO also deletes:

- the **separate reward-model training run** (a whole extra job, with its own data and tuning),
- **online generation** during training — PPO must sample from the policy every step; DPO reads a
  fixed offline preference dataset,
- **RL instability** — no advantage estimation, no clipping, no value-function tuning, no reward
  normalisation.

**With LoRA (board 19) it collapses further:** the reference model is the *same weights with the
adapters disabled*, so you hold roughly one model. That combination — DPO + LoRA — is why preference
tuning became something a single GPU can do.

---

## 7. What DPO gives up

DPO is not strictly better, and saying so is the mark of actually knowing it.

- **DPO is offline.** It learns from a fixed preference set. PPO generates fresh samples from the
  *current* policy, so it gets feedback on what the model actually does now. As DPO training moves
  the policy away from the data's distribution, its preference pairs become progressively
  off-policy. This is the main theoretical objection, and it motivated online variants
  (Online DPO, IPO, and iterative rounds).
- **No reward model means no reusable reward model.** PPO's RM can score arbitrary new outputs — for
  filtering, best-of-N sampling, or evaluation. DPO leaves you nothing to score with.
- **DPO can over-optimise the margin.** The loss keeps rewarding a growing gap between chosen and
  rejected, which can push *down* the likelihood of both — including the chosen response. Watch
  chosen-logprob during training; if it falls, that is the failure.
- **Frontier labs still use RL.** RLHF/PPO and its descendants remain in use where online feedback
  and a reusable reward model justify the cost. **DPO won the open-weights ecosystem, not the
  argument.**

---

## 8. Quick reference

```
RM        L = -log sigma(r_w - r_l)          reward defined only up to a difference
RLHF      max E[r] - beta*KL(pi || pi_ref)
optimum   pi* = (1/Z) pi_ref exp(r/beta)     Z intractable (sums over all responses)
invert    r = beta*log(pi*/pi_ref) + beta*log Z
DPO       differences cancel beta*log Z  ->  L = -log sigma(beta*log(pi_w/ref_w) - beta*log(pi_l/ref_l))

beta      leash length. large -> stay at pi_ref. small -> collapse onto argmax reward.
memory    PPO 4 models (239.3 GiB)   DPO 2 models (119.7 GiB)   + LoRA -> ~1
```

**The seven things to be able to say cold:**

1. **Rewards are only defined up to a difference** — Bradley-Terry never sees an absolute scale.
   That is what makes §5 possible.
2. **The KL-constrained optimum is closed-form:** `π* ∝ π_ref · exp(r/β)`. Inverting gives
   `r = β·log(π*/π_ref) + β·log Z(x)`, verified to `5.551e-17`.
3. **`Z(x)` is intractable** — it sums over every possible response, `|V|^L`.
4. **It cancels.** `Z(x)` depends only on `x`, so it is identical for `y_w` and `y_l`, and preference
   learning uses only their difference. Verified `0.000e+00`. **This is DPO.**
5. **The policy becomes its own reward model:** `β·log(π_θ/π_ref)` is an implicit reward, and the
   DPO loss equals the Bradley-Terry loss (`0.167786` both ways).
6. **β is a leash.** Large β keeps you at `π_ref` (KL `0.0020` at β=10); small β collapses onto the
   argmax reward (KL `0.9155` at β=0.05). What it stops is **reward hacking** — exploiting the
   proxy's errors.
7. **PPO holds 4 models, DPO holds 2** (`239.3` vs `119.7` GiB at 8B), and DPO also removes the RM
   training run, online generation, and RL instability. But DPO is **offline** and leaves you no
   reusable reward model — it won the open ecosystem, not the argument.

---

## See also

- [03_alignment.md](03_alignment.md) — the alignment landscape, RLHF in full
- [03b_alignment_end_to_end.md](03b_alignment_end_to_end.md) — worked RLHF numbers (states the closed form; §5 here supplies the missing cancellation)
- [06_alignment_follow_ups.md](06_alignment_follow_ups.md) — IPO, KTO, ORPO and the variants
- [02c_sft_end_to_end.md](02c_sft_end_to_end.md) — board 18: `π_ref` is the model SFT produced
- [../5.transformers/02_models/09b_lora_qlora_end_to_end.md](../5.transformers/02_models/09b_lora_qlora_end_to_end.md) — board 19: LoRA makes the reference model free
- [04_evaluation.md](04_evaluation.md) — board 21: how you tell whether any of this worked

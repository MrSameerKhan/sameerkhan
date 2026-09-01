# 11 — Long Context: RoPE Scaling, ALiBi, Sliding Window

> Board 14. Continues [08b_llama3_end_to_end.md §6](08b_llama3_end_to_end.md), where Llama 3 raised
> the RoPE base from 10,000 to 500,000. This file is **why that works**, and what the alternatives
> are when you cannot retrain.
>
> Everything is computed for `d_head = 128`, `base = 10,000`, trained at `L = 4096` — Llama-2-7B's
> configuration — extending to 32,768 (`s = 8`).

---

## 1. The problem, stated precisely

RoPE rotates dimension pair `i` by `m · θ_i`, with `θ_i = base^(−2i/d)`. Each pair has a
**wavelength** `2π/θ_i` — the number of positions before it returns to its starting phase.

```
 pair i       theta_i     wavelength   status at L = 4096
      0    1.00000000            6.3   fully rotated (model saw every phase)
      8    0.31622777           19.9   fully rotated
     16    0.10000000           62.8   fully rotated
     24    0.03162278          198.7   fully rotated
     32    0.01000000          628.3   fully rotated
     40    0.00316228        1,986.9   fully rotated
     48    0.00100000        6,283.2   NEVER completes a cycle
     56    0.00031623       19,869.2   NEVER completes a cycle
     63    0.00011548       54,410.1   NEVER completes a cycle
```

**46 of the 64 pairs complete at least one full rotation within 4096 tokens. The other 18 never do.**

That split is the whole subject:

- The 46 short-wavelength pairs have seen **every phase angle** during training. Asking them about
  position 20,000 is fine — the angle is one they have seen before.
- The 18 long-wavelength pairs have only ever seen a **fraction of one rotation**. At position
  20,000 they are asked to produce an angle that never occurred in training. The model has no idea
  what it means.

**Extrapolation fails on those 18 dimensions specifically.** Every method below is a different
answer to "what do we do about them?"

---

## Table of Contents

1. The problem, stated precisely
2. Position Interpolation — and why it costs precision
3. NTK-aware scaling
4. YaRN — treat each band by what it needs
5. ALiBi — no positions at all
6. Sliding window — depth as context
7. Comparison
8. Quick reference

---

## 2. Position Interpolation (PI)

Chen et al., 2023. Compress every position into the trained range:

```
m  ->  m / s        s = L_target / L_train = 32768 / 4096 = 8
```

Equivalently, every wavelength is multiplied by `s`:

```
 pair i    wavelength     after PI (×8)
      0           6.3              50.3
     16          62.8             502.7
     32         628.3           5,026.5
     48       6,283.2          50,265.5
     63      54,410.1         435,281.1
```

**It works — and it is blunt.** PI stretches *every* wavelength by 8×, including the short,
high-frequency pairs that were already perfectly fine. Pair 0's wavelength goes from 6.3 to 50.3
positions, so the dimension that used to distinguish "1 token apart" from "2 tokens apart" now
barely separates 8 tokens from 16.

**PI buys long-range capability by spending short-range precision.** It needs fine-tuning
(~1000 steps) to recover, and even then local tasks degrade measurably.

---

## 3. NTK-aware scaling

The insight: do not touch positions — change the **base**, so the stretch is applied *unevenly*.

```
base  ->  base · s^(d/(d−2))  =  10,000 · 8^(128/126)  =  82,685
```

```
 pair i       old wl          new wl    stretch
      0          6.3             6.3      1.00x     <- untouched
     16         62.8           106.5      1.70x
     32        628.3         1,806.7      2.88x
     48      6,283.2        30,637.2      4.88x
     63     54,410.1       435,281.1      8.00x     <- full stretch
```

**The stretch is 1.00× on the highest frequency and 8.00× on the lowest.** Exactly inverted from
PI, which applied 8× everywhere. Short-range resolution is preserved; only the dimensions that
actually needed help get stretched.

The exponent `d/(d−2)` is chosen so the *lowest* frequency receives precisely the factor `s`. That
is the whole derivation.

**NTK-aware often works zero-shot** — no fine-tuning — which is why it spread through the
open-weights community as a config-file change before anyone retrained anything.

---

## 4. YaRN — treat each band by what it needs

Peng et al., 2023. PI interpolates everything; NTK scales smoothly. YaRN makes the split
**explicit**, using the rotation count `r_i = L_train / wavelength_i`:

```
r > β (32)   the dim rotated many times in training   ->  EXTRAPOLATE, leave it alone
r < α (1)    the dim never completed one rotation     ->  INTERPOLATE fully (÷ s)
otherwise                                             ->  linear ramp between the two
```

```
 pair i   wavelength   r = L/wl                  YaRN action
      0          6.3     651.90      extrapolate (untouched)
      8         19.9     206.15      extrapolate (untouched)
     16         62.8      65.19      extrapolate (untouched)
     24        198.7      20.61       ramp, 37% interpolated
     30        471.2       8.69       ramp, 75% interpolated
     34        837.9       4.89       ramp, 87% interpolated
     38      1,490.0       2.75       ramp, 94% interpolated
     42      2,649.6       1.55       ramp, 98% interpolated
     48      6,283.2       0.65       interpolate fully (÷8)
     56     19,869.2       0.21       interpolate fully (÷8)
     63     54,410.1       0.08       interpolate fully (÷8)

 totals: 21 extrapolated · 25 ramped · 18 fully interpolated
```

**The 18 fully-interpolated dimensions are exactly the 18 from §1 that never complete a rotation.**
`r < 1` *means* "wavelength longer than the training context". YaRN did not pick them by hand — the
criterion falls out of the same quantity.

YaRN also rescales attention temperature:

```
1/t = 0.1 · ln(s) + 1 = 1.207944       (s = 8)
logits are multiplied by 1/t = 0.827853
```

Interpolation flattens attention — squeezing positions together makes neighbours look more alike —
so YaRN sharpens the logits back. It is a small correction with a measurable effect.

**YaRN reaches longer contexts with ~10× less fine-tuning than PI** (roughly 400 steps), which is
why it became the default extension recipe.

---

## 5. ALiBi — no positional encoding at all

Press et al., 2021. Add nothing to the embeddings and rotate nothing. Instead, penalise distance
directly in the attention logits:

```
score(i, j) = qᵢ · kⱼ  −  m_h · (i − j)
                          ↑
                          fixed per-head slope, NOT learned
```

Slopes form a geometric sequence, `m_h = 2^(−8h/H)`:

```
H = 8    0.500000, 0.250000, 0.125000, 0.062500, 0.031250, 0.015625, 0.007812, 0.003906
         ratio between consecutive heads = 0.5000

H = 16   0.707107, 0.500000, 0.353553, 0.250000, 0.176777, 0.125000, 0.088388, 0.062500, …
         ratio = 0.7071  ( = 2^−0.5 )
```

**Different heads get different decay rates.** The steepest head (0.5) effectively sees only a few
tokens back; the shallowest (0.0039) barely decays at all and stays near-global. One layer therefore
covers many ranges at once.

```
ALiBi has ZERO positional parameters. No embedding table, no rotation, nothing to extend.
```

Extrapolation is free — the bias is defined for any distance. Its weakness is that the decay is
**monotonic and fixed**: ALiBi can never attend strongly to something far away, which caps it on
retrieval-style tasks where the answer sits at position 50,000. That is why RoPE + scaling won for
general-purpose LLMs, and ALiBi persists mainly where strict locality is acceptable.

---

## 6. Sliding window — depth as context

Mistral 7B. Restrict each layer to a fixed window (4096), but let **depth compose**:

```
after  1 layer : receptive field  =   4,096 tokens
after  2 layers:                  =   8,192
after  4 layers:                  =  16,384
after  8 layers:                  =  32,768
after 16 layers:                  =  65,536
after 32 layers:                  = 131,072
```

**32 layers × a 4096 window = 131,072 tokens of effective context.** The same argument as GPT-3's
banded attention ([06c §4](06c_gpt3_end_to_end.md)): reach grows linearly with depth.

The trade is identical too. Information from token 0 reaching token 131,071 must pass through 32
hops, each a lossy weighted average. It is *reachable*, not *directly attendable* — which is why
sliding-window models underperform on tasks needing exact recall across the full span, and why
Mistral pairs the window with a rolling KV cache rather than claiming true 128k attention.

---

## 7. Comparison

| Method | What it changes | Short-range precision | Fine-tuning needed |
|---|---|---|---|
| **PI** | positions `m → m/s` | **degraded** (all wavelengths ×8) | yes, ~1k steps |
| **NTK-aware** | base `10,000 → 82,685` | preserved (1.00× on pair 0) | often none |
| **YaRN** | per-band ramp + temperature | preserved by construction | yes, ~400 steps |
| **ALiBi** | linear logit bias, no RoPE | n/a — no positions | trained in from scratch |
| **Sliding window** | the mask, not positions | full within the window | trained in from scratch |

**PI, NTK and YaRN are post-hoc** — applied to an already-trained RoPE model, sometimes as a config
change. **ALiBi and sliding window are architectural** — you commit at pretraining.

---

## 8. Quick reference

```
wavelength_i = 2*pi / theta_i,  theta_i = base^(-2i/d)
  dims with wavelength <= L_train  saw every phase   -> extrapolate safely
  dims with wavelength >  L_train  never completed   -> THESE are what breaks

PI        m -> m/s              every wavelength x s      blunt, costs local precision
NTK       base -> base*s^(d/(d-2))  1.00x .. s, graded    preserves local precision
YaRN      ramp by r = L/wl      + attention temperature   explicit, least fine-tuning
ALiBi     score -= m_h*(i-j)    m_h = 2^(-8h/H)           zero positional parameters
SWA       window W, L layers    receptive field = L*W      reach, not direct attention
```

**The seven things to be able to say cold:**

1. RoPE breaks at long context because **the low-frequency dimensions never complete a rotation
   during training** — at `d=128`, `base=10k`, `L=4096`, that is **18 of 64 pairs**. The other 46
   extrapolate fine.
2. **PI compresses every position by `s`**, multiplying *all* wavelengths by 8 — including the
   short ones that were already fine. It works, and it costs short-range precision.
3. **NTK-aware changes the base instead** (`10,000 → 82,685` for `s=8`), stretching the lowest
   frequency 8× and the highest **1.00×**. Often works zero-shot.
4. **YaRN splits dimensions by rotation count `r = L_train/wavelength`** — extrapolate above 32,
   interpolate fully below 1, ramp between. The 18 dims it fully interpolates are exactly the 18
   that never complete a rotation.
5. YaRN also **rescales attention temperature** (`1/t = 0.1·ln(s)+1 = 1.207944` at `s=8`), because
   interpolation flattens attention.
6. **ALiBi has zero positional parameters** — a fixed per-head slope `2^(−8h/H)` subtracted from the
   logit. Free extrapolation, but monotonic decay caps it on long-range retrieval.
7. **Sliding window buys reach through depth**: `32 layers × 4096 window = 131,072`. Reachable via
   32 lossy hops, not directly attendable — the same trade as GPT-3's banded attention.

---

## See also

- [08b_llama3_end_to_end.md](08b_llama3_end_to_end.md) — Llama 3's base 500,000, and why it was raised
- [08_modern_llm_architecture.md](08_modern_llm_architecture.md) — RoPE itself, hand-computed
- [06c_gpt3_end_to_end.md](06c_gpt3_end_to_end.md) — banded sparse attention, the same depth-vs-reach trade
- [04b_attention_at_scale_end_to_end.md](04b_attention_at_scale_end_to_end.md) — what long context costs in KV cache
- [../../6.llms/05_vllm_internals.md](../../6.llms/05_vllm_internals.md) — serving long context in practice

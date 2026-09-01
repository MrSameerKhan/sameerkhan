# 09b — LoRA and QLoRA: End-to-End

> Board 19. Companion to [09_parameter_efficient_tuning.md](09_parameter_efficient_tuning.md) and
> [../../6.llms/02b_finetuning_end_to_end.md](../../6.llms/02b_finetuning_end_to_end.md), which
> cover the PEFT landscape (adapters, prefix tuning, prompt tuning). **This file is the
> arithmetic**, and it corrects two things those files get wrong — §2 and §7.
>
> Board 18 (SFT) is [../../6.llms/02c_sft_end_to_end.md](../../6.llms/02c_sft_end_to_end.md);
> board 20 (RLHF/DPO) is [../../6.llms/03_alignment.md](../../6.llms/03_alignment.md).

---

## 1. The idea in one equation

```
W_finetuned  =  W_pretrained  +  ΔW

LoRA:        ΔW  =  (α / r) · B · A          B is (d × r),  A is (r × d),  r ≪ d

  W  frozen        d × d parameters
  A, B trainable   2 · d · r parameters
```

**The claim is that `ΔW` — what fine-tuning *changes* — is low-rank**, even though `W` is not. You
are not compressing the model; you are compressing the *update*.

```
d = 4096, r = 16:   full  4096² = 16,777,216
                    LoRA  2·4096·16 =  131,072        128× fewer
```

---

## Table of Contents

1. The idea in one equation
2. **Initialisation — `B = 0`, and a correction**
3. The `α/r` scaling
4. Where LoRA attaches, and what it costs
5. Why rank 8 usually suffices — honestly
6. **Merging: zero inference overhead**
7. **NF4 — measured, not asserted**
8. QLoRA memory
9. Quick reference

---

## 2. Initialisation — `B = 0`

```
A  ~  Gaussian (small random)
B  =  0          exactly zero

=>  ΔW = B·A = 0  at step 0
```

**The model begins fine-tuning in *exactly* its pretrained state.** No warmup, no discontinuity —
the adapter contributes nothing until gradients move `B`.

Why not both zero? If `A = 0` and `B = 0`, then `∂L/∂B ∝ A = 0` and `∂L/∂A ∝ B = 0` — **both
gradients vanish and nothing ever trains.** One must be non-zero to break the symmetry, and `B` is
chosen so the *product* still starts at zero.

Worked, with `x = [1.0, 1.5]`, `A = [[0.2, 0.11]]`:

```
B = [[0.0],        x @ B      = [0.0]
     [0.0]]        (x@B) @ A  = [0.0, 0.0]        ->  LoRA adds nothing.  ✓
```

> **Correction to the companion file.** `02b_finetuning_end_to_end.md` §3.3 states "B initialized to
> ZERO" but then writes `B = [[0.8], [0.0]]`, computes `x @ B = [0.800]` from that non-zero matrix,
> and concludes `(x@B) @ A = [0.000, 0.000]` "because B=0". With the `B` as written the answer is
> **`[0.16, 0.088]`**, not zero. The conclusion is right; the matrix and the intermediate step
> contradict it. Use the numbers above.

---

## 3. The `α/r` scaling

```
ΔW = (α / r) · B · A
```

`α` is a constant you hold fixed while sweeping `r`. Doubling `r` doubles the number of rank-1
components summed into `B·A`, which would double its typical magnitude — dividing by `r` cancels
that, **so changing rank does not force you to retune the learning rate.**

```
r = 8,  α = 16   ->  scaling 2.0
r = 8,  α = 8    ->  scaling 1.0
r = 64, α = 16   ->  scaling 0.25
```

Common defaults: `α = r` (scaling 1) or `α = 2r` (scaling 2). `r=8, α=16` is the usual starting
point.

**`α` is not a learning rate.** It scales the *contribution*, not the step size — though in practice
raising `α` and raising the LR have similar early effects, which is why they get confused.

---

## 4. Where LoRA attaches, and what it costs

The original paper (Hu et al. 2021) applied LoRA to **`W_q` and `W_v` only**. Later practice
commonly targets all attention projections, and sometimes the MLP.

Llama-3 8B — 32 layers, `d = 4096`, GQA (so `k`/`v` project to 1024), `d_ff = 14336`:

```
  target                        r=4         r=8        r=16        r=64   % model (r=16)
  q,v only                1,703,936   3,407,872   6,815,744  27,262,976         0.085%
  all attn (q,k,v,o)      3,407,872   6,815,744  13,631,488  54,525,952         0.170%
  attn + MLP             10,485,760  20,971,520  41,943,040 167,772,160         0.522%
```

**Even the most aggressive configuration here — every attention and MLP matrix at `r=64` — is under
2.1% of the model.** The parameter saving is not the interesting constraint; §8 shows what actually
is.

Note GQA's effect: `k_proj` and `v_proj` are `4096 × 1024`, not `4096 × 4096`, so LoRA on them costs
`r·(4096 + 1024)` rather than `r·8192`. Board 13's architecture choice propagates here.

---

## 5. Why rank 8 usually suffices — honestly

The expressiveness ceiling is real and worth stating plainly:

```
  r=  1  ΔW rank ≤   1 of 4096  =  0.02% of directions
  r=  8  ΔW rank ≤   8 of 4096  =  0.20%
  r= 16  ΔW rank ≤  16 of 4096  =  0.39%
  r= 64  ΔW rank ≤  64 of 4096  =  1.56%
```

`r=8` lets each matrix move its output within an 8-dimensional subspace of 4096. That sounds
crippling. It is not, for two reasons:

1. **It is 8 directions *per module*.** At 32 layers × 2 matrices, `r=8` gives **512 adapted
   directions** across the network, composed non-linearly.
2. **The intrinsic-dimension hypothesis** (Aghajanyan et al., 2020): fine-tuning updates empirically
   lie in a very low-dimensional subspace — you can reparameterise fine-tuning into a few hundred
   or thousand dimensions and still reach good task performance.

**The honest version, which is what to say:** low rank suffices for *adaptation* — style, format,
domain vocabulary, task shape — where the model already has the capability and needs steering.
It suffices less well when you are teaching genuinely new knowledge or a distant distribution, and
that is exactly when practitioners find higher `r` (32–128) helps. "Rank 8 always works" is
overclaiming; "rank 8 works for adaptation, raise it when the task is far from pretraining" is the
defensible statement.

---

## 6. Merging: zero inference overhead

```
separate:  x @ W  +  (α/r)·((x @ B) @ A)
merged  :  x @ ( W + (α/r)·B·A )

max | separate − merged | = 4.441e-16
```

**They are the same matrix.** After training you fold the adapter in once and serve a model that is
architecturally identical to the base — no extra matmuls, no latency cost, no serving complexity.

This is LoRA's decisive practical advantage over adapters and prefix tuning, both of which add
modules that persist at inference.

The corollary is what makes LoRA an ecosystem: **many adapters, one base.** Keep `W` resident and
swap `(A, B)` per task — a few megabytes each — instead of hosting a full fine-tune per task.
Merging is optional, and you give it up when you want hot-swapping.

---

## 7. NF4 — measured

NF4 places its 16 levels at the **quantiles of a normal distribution** rather than uniformly,
because weights are approximately Gaussian after blockwise absmax normalisation.

```
NF4  : -1.0000 -0.6962 -0.5251 -0.3949 -0.2844 -0.1848 -0.0911  0.0000 ...
INT4 : -1.0000 -0.8667 -0.7333 -0.6000 -0.4667 -0.3333 -0.2000 -0.0667 ...

spacing near 0:   NF4 0.0796    INT4 0.1333      <- NF4 is FINER where the mass is
spacing at edge:  NF4 0.2770    INT4 0.1333      <- and coarser where it is not
```

Quantisation error on 2,000,000 samples from `N(0,1)`, absmax-normalised:

```
  NF4            MSE = 6.932011e-04     mean|err| = 2.264185e-02
  uniform INT4   MSE = 1.480951e-03     mean|err| = 3.332406e-02

  NF4 is 2.14× lower MSE
```

> **Correction to the companion file.** `02b_finetuning_end_to_end.md` §4.3 states "NF4 gives 6×
> better representation", derived from an example whose numbers are stipulated
> ("nearest level *might* be 0.133"). Measured against the actual NF4 levels the figure is
> **2.14×** on Gaussian data. Real and worth having — but quote the measurement, not the 6×.

---

## 8. QLoRA memory

QLoRA = **NF4-quantised frozen base + fp16 LoRA adapters**. Llama-3 8B (`8.03e9` params):

```
  fp32        4    bytes/param    29.92 GiB
  fp16/bf16   2    bytes/param    14.96 GiB
  int8        1    bytes/param     7.48 GiB
  NF4         0.5  bytes/param     3.74 GiB
```

```
  base, NF4, frozen                              3.74   GiB
  LoRA r=16 on q,v (6,815,744 params, fp16)      0.0127 GiB    0.0849% of the model
  Adam optimizer states — ADAPTERS ONLY          0.0508 GiB
                                                 ────────
                                                ~3.80   GiB
```

*(6,815,744 is the GQA-aware count from §4 — `v_proj` is `4096 × 1024`, not `4096 × 4096`.)*

**Compare the optimizer line to full fine-tuning:** Adam keeps two fp32 states per trainable
parameter, so full fine-tuning of 8.03B needs `59.8 GiB` for optimizer state alone —
**1,178× more** than QLoRA's adapters.

**That is the real story.** The headline "0.1% of parameters trainable" understates it: what
actually makes a 70B fine-tune fit on one GPU is that gradients, momentum and variance exist only
for the adapters, while the frozen base sits at half a byte per weight.

Two costs to be honest about:

- **NF4 is lossy.** The base model is measurably degraded before you start. QLoRA works because the
  adapter learns *around* the quantisation error, not because 4-bit is free.
- **Dequantisation at compute time.** Weights are stored in NF4 and dequantised to bf16 per matmul,
  so QLoRA trades compute for memory — it is *slower* per step than fp16 LoRA, not faster.

---

## 9. Quick reference

```
ΔW = (alpha/r) B A      A ~ Gaussian, B = 0  ->  ΔW = 0 at init
                        both zero => both gradients vanish, nothing trains
alpha/r                 lets you change r without retuning the LR
merge                   W' = W + (alpha/r)BA  -- verified 4.441e-16, ZERO inference cost
NF4                     quantile levels, 2.14x lower MSE than uniform INT4
QLoRA                   NF4 frozen base + fp16 adapters; optimizer state 1,178x smaller
```

**The seven things to be able to say cold:**

1. **LoRA compresses the *update*, not the model.** `ΔW = (α/r)·B·A`; `W` stays frozen and full-rank.
2. **`B = 0` at init, `A` random** — so `ΔW = 0` and training starts exactly at the pretrained model.
   Both zero and *nothing trains*, because each gradient is proportional to the other matrix.
3. **`α/r` decouples rank from learning rate.** `r=8, α=16` → scaling 2.0. `α` is not a learning rate.
4. **Original paper: `W_q` and `W_v` only.** Even all-attention-plus-MLP at `r=64` is under 2.1% of
   Llama-3 8B.
5. **Rank 8 = 0.20% of directions per matrix, but 512 across the network.** It suffices for
   *adaptation*; raise `r` when teaching something far from pretraining. Do not claim it always works.
6. **Merging is exact** (`4.441e-16`) — zero inference overhead. That is LoRA's edge over adapters
   and prefix tuning. Skip merging when you want to hot-swap many adapters on one base.
7. **QLoRA's win is optimizer state, not just weights.** Adam on 8.03B full fine-tuning is
   `59.8 GiB`; on the adapters it is `0.0508 GiB` — **1,178×**. NF4 is 2.14× better than uniform
   INT4, and it is lossy: the adapter learns around the damage.

---

## See also

- [09_parameter_efficient_tuning.md](09_parameter_efficient_tuning.md) — the PEFT landscape: adapters, prefix tuning, prompt tuning
- [../../6.llms/02b_finetuning_end_to_end.md](../../6.llms/02b_finetuning_end_to_end.md) — the companion dry-run (see the corrections in §2 and §7)
- [../../6.llms/02c_sft_end_to_end.md](../../6.llms/02c_sft_end_to_end.md) — board 18: what you are fine-tuning *on*
- [../../6.llms/03_alignment.md](../../6.llms/03_alignment.md) — board 20: DPO is usually run with LoRA
- [08b_llama3_end_to_end.md](08b_llama3_end_to_end.md) — the model whose shapes §4 and §8 use
- [../../2.deep learning/02_architectures/10_quantization_theory.md](../../2.deep%20learning/02_architectures/10_quantization_theory.md) — quantisation beyond NF4

# Session 1 — LoRA Fine-tuning
Status: `🔧 Code-built`

Theory: [../../../6.llms/02_finetuning.md](../../../6.llms/02_finetuning.md) · [../../../5.transformers/02_models/09_parameter_efficient_tuning.md](../../../5.transformers/02_models/09_parameter_efficient_tuning.md)

---

## Use Case

Domain-adapted financial assistant: `facebook/opt-125m` gives generic answers about mortgages. Fine-tuning on 2,000 instruction pairs shifts the model to use bank-specific terminology, formats, and knowledge — in ~20 minutes on MPS, using only 0.24% of the model's parameters.

---

## Why LoRA Works

Full fine-tuning updates all 125M parameters — expensive and risks catastrophic forgetting. LoRA freezes the base weights and adds two small matrices (A and B) to each target layer:

```
Original:  W ∈ ℝ^(d×d)        — 125M params
LoRA:      W + ΔW = W + B·A   — only A ∈ ℝ^(d×r), B ∈ ℝ^(r×d) are trained
                                 r=8 → 300K params total (0.24%)

Forward pass: y = Wx + (BA)x·(α/r)   — base result + scaled adapter
                                         α/r = 16/8 = 2.0 scaling factor
```

At inference: A and B are merged into W → zero inference overhead.

---

## LoRA Config Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `r=8` | rank | Adapter capacity. r=8 covers most tasks; use r=16-64 for heavy domain shift |
| `lora_alpha=16` | scaling | alpha/r = 2.0 — standard convention, keeps gradient scale stable |
| `target_modules=["q_proj", "v_proj"]` | which layers | Attention query + value projections. Add k/o for richer adaptation |
| `lora_dropout=0.05` | regularisation | Prevents adapter overfitting on small datasets |
| `bias="none"` | bias training | Don't train bias terms — keeps adapter truly minimal |

---

## Trainable Parameters Comparison

```
facebook/opt-125m:
  Full fine-tune: 125,239,296 params (100%) — needs ~500 MB GPU, slow
  LoRA r=8:           302,080 params (0.24%) — needs ~2 MB extra, fast

TinyLlama-1.1B:
  Full fine-tune: 1,100,000,000 params — needs ~4.4 GB GPU
  LoRA r=16:          8,388,608 params (0.76%) — needs ~32 MB extra
```

---

## Expected Output

```
Device: mps
trainable params: 302,080 || all params: 125,541,376 || trainable%: 0.2407

Training...
{'loss': 2.8421, 'grad_norm': 1.24, 'learning_rate': 0.0002, 'epoch': 0.20}
{'loss': 2.1830, 'grad_norm': 0.98, 'learning_rate': 0.00018, 'epoch': 1.00}
{'loss': 1.9341, 'grad_norm': 0.87, 'learning_rate': 0.00005, 'epoch': 2.00}

LoRA adapter saved to models/09_finetuning/lora_opt125m
Adapter size: ~1-2 MB (vs ~500 MB for full model)

── Inference ──
Prompt: Explain what a mortgage LTV ratio is.
Response: The loan-to-value (LTV) ratio is the percentage of the property's
          value that is borrowed. For example, on a £300,000 property with
          a £270,000 mortgage, the LTV is 90%...
```

---

## How to Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE python 01_lora_finetune.py
```

MPS training time: ~20–30 min for 2 epochs on 2,000 examples.
First run: downloads `facebook/opt-125m` (~500 MB) + alpaca dataset (~50 MB).

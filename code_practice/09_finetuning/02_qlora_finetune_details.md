# Session 2 — QLoRA Fine-tuning
Status: `🔧 Code-built`

Theory: [../../../6.llms/02_finetuning.md](../../6.llms/02_finetuning.md) · [../../../5.transformers/02_models/09_parameter_efficient_tuning.md](../../5.transformers/02_models/09_parameter_efficient_tuning.md)

---

## Use Case

Same domain adaptation as session 01, but on a 1.1B model that wouldn't fit on a 16 GB GPU in float32. QLoRA makes it fit in 700 MB by storing base weights in 4-bit while training LoRA adapters in BF16.

---

## QLoRA = NF4 quantization + LoRA

```
Base model weights:    float32 (4 bytes/param) → NF4 (0.5 bytes/param)
                       6.4× memory reduction

LoRA adapters:         stored and trained in BF16 (2 bytes/param)
                       only 0.76% of params → negligible memory

Compute:               NF4 → dequantize to BF16 → matmul → LoRA add → BF16
                       Dequantisation happens per-layer, on-the-fly, in GPU registers
```

---

## Memory Comparison

| Model | Precision | GPU Memory | Can fit on |
|-------|-----------|-----------|------------|
| TinyLlama 1.1B | float32 | 4.4 GB | A100 40GB, RTX 3090 |
| TinyLlama 1.1B | QLoRA NF4 | **0.7 GB** | T4 16GB, **RTX 3060 8GB** |
| Llama-2 7B | float32 | 28 GB | A100 80GB only |
| Llama-2 7B | QLoRA NF4 | **5 GB** | **RTX 3090 24GB** |
| Llama-2 70B | QLoRA NF4 | **35 GB** | 2× A100 40GB |

---

## NF4 Quantization

NF4 (Normal Float 4) uses 16 non-uniform quantisation levels optimised for normally-distributed weights (which neural network weights approximate):

```
float32 value → find nearest of 16 NF4 levels → store 4-bit index
               
NF4 levels: {-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0922,
              0.0000, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0}

Double quantisation: quantise the quantisation constants too → saves 0.37 bits/param
```

---

## Platform Requirements

| Platform | LoRA (session 01) | QLoRA (session 02) |
|----------|-------------------|--------------------|
| Mac MPS | ✓ Works | ✗ bitsandbytes not supported |
| Windows CPU | ✓ Works (slow) | ✗ bitsandbytes CUDA-only |
| Linux CUDA | ✓ Works | ✓ Works |
| Google Colab T4 | ✓ Works | ✓ Works |

Script auto-detects and falls back to standard LoRA on non-CUDA devices.

---

## How to Run

```bash
# On CUDA machine:
KMP_DUPLICATE_LIB_OK=TRUE python 02_qlora_finetune.py

# On Mac MPS (falls back to LoRA without quantization):
python 02_qlora_finetune.py
```

CUDA training time: ~15–20 min for 2 epochs on 2,000 examples (T4).
MPS fallback: same as session 01 (~20–30 min).

**Resume bullet:** "Fine-tuned TinyLlama-1.1B with QLoRA (4-bit NF4 + LoRA r=16) on synthetic banking dataset; model fits in 700 MB GPU memory vs 4.4 GB float32."

# Session 2 — QLoRA Fine-tuning (4-bit quantization + LoRA)
# Task   : same as session 01 but on a larger model using far less GPU memory
# Model  : TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B — swap for Llama-2-7b with CUDA)
# Method : QLoRA = 4-bit NF4 quantization (bitsandbytes) + LoRA adapter on top
# Key    : 1.1B model fits in ~700 MB GPU memory (vs ~4.4 GB in float32)
#
# ⚠️  bitsandbytes requires CUDA (Linux/Windows with NVIDIA GPU).
#     On Apple MPS, skip quantization — use regular LoRA from session 01 instead.
#     This session documents the pattern for when you have CUDA access.

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUTPUT_DIR = "models/09_finetuning/qlora_tinyllama"
DEVICE     = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

TRAIN_ROWS = 2000


# ── QLoRA = BitsAndBytesConfig (4-bit NF4) + LoRA ────────────────────────────
# NF4 (Normal Float 4-bit): quantisation scheme optimised for normally-distributed weights.
# double_quant: also quantise the quantisation constants → saves another ~0.4 bits/param.
# compute_dtype: upcast to bf16 for the matmul (4-bit store, bf16 compute).
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # more modules than session 01
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# ── Dataset formatting (TinyLlama uses ChatML format) ─────────────────────────
CHATML_TEMPLATE = (
    "<|system|>\nYou are a helpful assistant.</s>\n"
    "<|user|>\n{instruction}\n{input}</s>\n"
    "<|assistant|>\n{output}</s>"
)

def format_example(row):
    return {"text": CHATML_TEMPLATE.format(
        instruction=row["instruction"],
        input=row.get("input", ""),
        output=row["output"],
    )}


# ── Memory comparison ─────────────────────────────────────────────────────────
def print_memory_summary(model, label: str) -> None:
    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{label}")
    print(f"  Total params:     {total / 1e6:.1f}M")
    print(f"  Trainable params: {trainable / 1e6:.2f}M ({100*trainable/total:.2f}%)")
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used:  {alloc:.2f} GB")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    use_qlora = torch.cuda.is_available()   # QLoRA requires CUDA for bitsandbytes
    if not use_qlora:
        print("⚠  CUDA not available — running standard LoRA (no quantization)")
        print("   For true QLoRA, run on a CUDA machine (A100/T4/RTX 3090+)\n")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load model (quantized if CUDA available, else float32)
    if use_qlora:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=BNB_CONFIG,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)   # freeze base, enable gradient checkpointing
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    print_memory_summary(model, "After loading (before LoRA):")

    # 3. Add LoRA adapter
    model = get_peft_model(model, LORA_CONFIG)
    print_memory_summary(model, "After LoRA:")
    # QLoRA: 1.1B params stored in 4-bit ≈ 550 MB + LoRA ≈ 20 MB = ~570 MB total
    # Standard float32: 1.1B × 4 bytes = ~4.4 GB

    # 4. Dataset
    raw     = load_dataset("tatsu-lab/alpaca", split="train").select(range(TRAIN_ROWS))
    dataset = raw.map(format_example, remove_columns=raw.column_names)

    # 5. Training
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_seq_length=256,
        dataset_text_field="text",
        logging_steps=50,
        save_strategy="epoch",
        fp16=False,
        bf16=torch.cuda.is_available(),    # BF16 training on CUDA only
        gradient_checkpointing=use_qlora,  # saves more memory on CUDA
        report_to="none",
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nTraining...")
    trainer.train()

    # 6. Save (adapter only — base model stays quantized on disk)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}")

    print("\n── Memory comparison (float32 vs QLoRA) ──")
    print("  TinyLlama 1.1B, float32:    ~4.4 GB GPU memory")
    print("  TinyLlama 1.1B, QLoRA NF4:  ~0.7 GB GPU memory  (6× reduction)")
    print("  Llama-2 7B, float32:        ~28 GB GPU memory")
    print("  Llama-2 7B, QLoRA NF4:      ~5 GB GPU memory    (fits on 8 GB GPU)")


if __name__ == "__main__":
    main()

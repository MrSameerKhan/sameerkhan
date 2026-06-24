# Session 1 — LoRA Fine-tuning
# Task   : adapt a base LLM to follow domain-specific instructions cheaply
# Model  : facebook/opt-125m (tiny — swap for TinyLlama-1.1B or Llama-2-7b on bigger hardware)
# Dataset: tatsu-lab/alpaca (52K instruction-response pairs)
# Method : LoRA — add low-rank adapter matrices to attention layers only
# Key    : trainable params drop from 125M → ~0.3M while retaining 95%+ of task performance

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "facebook/opt-125m"
OUTPUT_DIR = "models/09_finetuning/lora_opt125m"
DEVICE     = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# ── LoRA config ───────────────────────────────────────────────────────────────
# r      : rank — size of the low-rank matrices. Higher r = more capacity, more params.
#          Rule of thumb: r=8 for most tasks, r=16-64 for domain-heavy adaptation.
# alpha  : scaling factor. alpha/r controls the effective learning rate of the adapter.
#          Convention: alpha = 2×r (keeps scale stable across different r values).
# target_modules: which weight matrices to add LoRA to. In transformers, q and v projections
#          are standard. Some papers add k and o projections for richer adaptation.
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

TRAIN_ROWS = 2000   # subset of alpaca for speed; full dataset = 52k


# ── Dataset formatting ────────────────────────────────────────────────────────
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

def format_example(row):
    return {"text": ALPACA_TEMPLATE.format(
        instruction=row["instruction"],
        input=row.get("input", ""),
        output=row["output"],
    )}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    # 1. Tokenizer + base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # OPT has no pad token
    tokenizer.model_max_length = 256            # cap sequence length (TRL 1.x: set here, not in SFTTrainer)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True, device_map="auto")

    # 2. Wrap with LoRA
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    # Expected: trainable params: ~300K | all params: ~125M | trainable%: 0.24%

    # 3. Dataset
    raw      = load_dataset("tatsu-lab/alpaca", split="train")
    raw      = raw.select(range(TRAIN_ROWS))
    dataset  = raw.map(format_example, remove_columns=raw.column_names)

    # 4. SFT training
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        dataset_text_field="text",
        logging_steps=50,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),   # True on Windows/Linux CUDA; False on CPU
        bf16=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Training...")
    trainer.train()

    # 5. Save adapter only (not full model — LoRA adapter is tiny: ~1MB)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}")
    print(f"Adapter size: ~1-2 MB (vs ~500 MB for full model)")

    # 6. Inference — load base + adapter
    print("\n── Inference (base model vs LoRA adapter) ──")
    base_model   = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_safetensors=True, device_map="auto")
    lora_model   = PeftModel.from_pretrained(base_model, OUTPUT_DIR).to(DEVICE)
    lora_model.eval()

    prompts = [
        "Below is an instruction.\n\n### Instruction:\nExplain what a mortgage LTV ratio is.\n\n### Response:\n",
        "Below is an instruction.\n\n### Instruction:\nList three benefits of compound interest.\n\n### Response:\n",
    ]
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = lora_model.generate(
                **enc,
                max_new_tokens=80,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = ids[0][enc["input_ids"].shape[-1]:]
        print(f"\nPrompt: {prompt.split('Instruction:')[1].split('Response:')[0].strip()}")
        print(f"Response: {tokenizer.decode(new_tokens, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()

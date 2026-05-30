# Session 4 — DPO Alignment (Direct Preference Optimisation)
# Task   : make a fine-tuned model prefer human-quality responses over dry/generic ones
# Model  : facebook/opt-125m (same as session 01) — or load the LoRA output
# Dataset: preference pairs (chosen=good, rejected=bad) from session 03 or synthetic
# Method : DPO — train without a reward model, directly on (prompt, chosen, rejected)
#
# DPO objective: log(σ(β·log(π(chosen)/πref(chosen)) - β·log(π(rejected)/πref(rejected))))
# Where: π = policy model, πref = frozen reference model, β = KL penalty strength
#
# Reference: Rafailov et al. 2023 — "Direct Preference Optimization: Your LM is Secretly a Reward Model"

import os
import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig

MODEL_NAME = "facebook/opt-125m"
LORA_PATH  = "models/09_finetuning/lora_opt125m"   # session 01 output (optional)
OUTPUT_DIR = "models/09_finetuning/dpo_opt125m"
DPO_DATA   = "data/09_finetuning/banking_instructions/dpo_train"  # session 03 output
DEVICE     = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

BETA        = 0.1   # KL penalty: higher = stay closer to reference, lower = more aggressive alignment
TRAIN_ROWS  = 200   # small demo — production uses 10K+ pairs


# ── Synthetic preference dataset (fallback if session 03 hasn't been run) ─────
# Format: {"prompt": str, "chosen": str, "rejected": str}
SYNTHETIC_PAIRS = [
    {
        "prompt":   "What is the maximum LTV for a first-time buyer?",
        "chosen":   "First-time buyers can borrow up to 95% LTV under the Help to Buy scheme. This means a minimum 5% deposit on the property purchase price. For a £300,000 property, that's a £15,000 deposit.",
        "rejected": "You can get a mortgage as a first-time buyer. There are various loan options available to you.",
    },
    {
        "prompt":   "What documents do I need to apply for a mortgage?",
        "chosen":   "You'll need: (1) Payslips for the last 3 months if employed, (2) SA302 tax returns for the last 2 years if self-employed, (3) Bank statements for 3 months, and (4) Proof of identity. Retired applicants should provide pension statements.",
        "rejected": "You will need to provide various financial documents. Please contact your branch for a full list.",
    },
    {
        "prompt":   "What happens to my mortgage rate after the fixed period ends?",
        "chosen":   "After your fixed period ends, your rate automatically moves to our Standard Variable Rate (SVR), currently 7.24%. You can avoid this by remortgaging to a new fixed or tracker deal before your fixed period expires.",
        "rejected": "Your rate will change after the fixed term. You should speak to your advisor about your options.",
    },
    {
        "prompt":   "Can I repay my mortgage early?",
        "chosen":   "Yes, but Early Repayment Charges (ERCs) apply during your fixed period: 5% in year 1, 4% year 2, 3% year 3, 2% year 4, 1% year 5. After the fixed period ends, you can repay in full with no ERC. On tracker products you can overpay up to 10% per year free of charge.",
        "rejected": "You may be able to repay early, but there may be charges. Please check your mortgage terms.",
    },
    {
        "prompt":   "What is the maximum personal finance amount for a Saudi national earning SAR 15,000 per month?",
        "chosen":   "For a Saudi national earning SAR 15,000 per month, the maximum personal finance amount is 36× monthly salary = SAR 540,000, subject to a maximum tenor of 60 months and the Debt Burden Ratio not exceeding 33% of gross salary.",
        "rejected": "The maximum personal finance depends on your salary and other factors. A finance advisor can help calculate this for you.",
    },
] * 40   # repeat for a larger demo dataset


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"β (KL penalty): {BETA}\n")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO prefers left-padding

    # 2. Policy model (the one we're training)
    #    Optionally start from session 01's LoRA output
    if os.path.exists(LORA_PATH):
        print(f"Loading LoRA adapter from {LORA_PATH}")
        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base, LORA_PATH)
        model = model.merge_and_unload()   # merge LoRA into base weights for DPO
    else:
        print("No LoRA adapter found — using base model directly")
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    # 3. Reference model (frozen copy of policy before DPO)
    #    DPOTrainer creates this automatically if ref_model=None and using PEFT
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    # 4. Add LoRA for efficient DPO training
    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Dataset
    if os.path.exists(DPO_DATA):
        dataset = load_from_disk(DPO_DATA)
        print(f"Loaded DPO dataset from {DPO_DATA}: {len(dataset)} pairs")
    else:
        print("Using synthetic preference pairs (run session 03 for domain-specific pairs)")
        dataset = Dataset.from_list(SYNTHETIC_PAIRS[:TRAIN_ROWS])

    # 6. DPO training
    args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,            # lower than SFT — DPO is more sensitive
        beta=BETA,
        max_length=256,
        max_prompt_length=128,
        logging_steps=20,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nTraining DPO...")
    trainer.train()

    # 7. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nDPO adapter saved to {OUTPUT_DIR}")

    # 8. Show DPO loss interpretation
    print("\n── DPO training objective ──")
    print("  Loss measures: log(σ(β · (log π(chosen)/πref(chosen) - log π(rejected)/πref(rejected))))")
    print("  Decreasing loss → model assigns higher probability to chosen over rejected")
    print(f"  β={BETA}: moderate KL penalty — model updates without diverging too far from base")
    print("\n  β too low  (<0.05): aggressive — risks reward hacking / losing general capability")
    print("  β too high (>0.5):  conservative — minimal alignment benefit")
    print("  β=0.1: standard default from DPO paper")


if __name__ == "__main__":
    main()

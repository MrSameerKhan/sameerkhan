# Session 3 — Synthetic Dataset Preparation for Fine-tuning
# Task   : turn raw domain documents into instruction-response training pairs
# Method : LLM-generated Q&A → quality filtering → ChatML format → HuggingFace Dataset
# Use    : you have 10K policy emails / documents → generate instruction pairs automatically
#
# Pipeline: raw docs → generate (instruction, response) pairs → filter → format → save
# Set OPENAI_API_KEY before running.

import os
import json
import random
from pathlib import Path
from datasets import Dataset
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o-mini"
OUTPUT_DIR = "data/09_finetuning/banking_instructions"


# ── Raw document corpus (same as Phase 07 RAG) ───────────────────────────────
# In production: replace with your actual documents (PDFs, emails, policies).
# Same corpus from _corpus.py, inlined here so session is self-contained.
RAW_DOCS = [
    "The maximum loan-to-value (LTV) ratio for standard residential mortgage applications is 90% for standard borrowers. First-time buyers are eligible for up to 95% LTV under the Help to Buy scheme. Buy-to-let properties are capped at 75% LTV.",
    "All mortgage applications require proof of income for the last 3 months. Employed: payslips. Self-employed: SA302 (minimum 2 years). Retired: pension statements. Affordability is stressed at rate + 3%; total debts must not exceed 45% of gross income.",
    "Fixed-rate periods: 2, 3, 5, and 10 years. After the fixed period, the rate reverts to SVR (7.24%). Early repayment charges: 5% yr1, 4% yr2, 3% yr3, 2% yr4, 1% yr5. No ERC after fixed period.",
    "Personal finance is Murabaha-based (Sharia-compliant). APR 5.5%–9.5% depending on salary band and credit score. Max amount: 36× monthly salary (nationals), 24× (expatriates). Max tenor: 60 months (nationals), 36 months (expatriates).",
    "Current accounts: daily ATM limit SAR 5,000 (standard), SAR 20,000 (premium). International ATM fee: SAR 20 + 1.75%. Savings accounts return 2.8% p.a. Accounts below SAR 500 pay SAR 10/month maintenance fee.",
    "SARIE transfers: immediate, 24/7. SWIFT international: next business day if before 14:00. Max single international transfer: SAR 100,000/day. FX spread: 1.5% over mid-market rate. Currency orders >USD 10,000 require prior treasury arrangement.",
    "Credit cards: min salary SAR 5,000. Limit: 3× salary (entry), up to 12× (premium). Supplementary cards for family members over 18; primary cardholder is liable. Limit increase after 6 months — no missed payments in preceding 12 months.",
]


# ── Step 1: Generate instruction-response pairs from raw docs ─────────────────

GENERATE_PROMPT = """You are creating a fine-tuning dataset for a bank assistant.

Given the following bank policy text, generate {n} diverse instruction-response pairs.
Each should be a realistic customer question a bank employee or customer might ask,
with a specific answer grounded in the provided text.

Include a mix of:
- Direct fact questions ("What is the max LTV?")
- Calculation questions ("If property is £300k, what is minimum deposit?")
- Eligibility questions ("Am I eligible if...?")
- Process questions ("How do I apply for...?")

Return JSON: {{"pairs": [{{"instruction": str, "response": str}}]}}

Policy text:
{document}"""


def generate_pairs(document: str, n: int = 3) -> list[dict]:
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": GENERATE_PROMPT.format(n=n, document=document)}],
    )
    data = json.loads(res.choices[0].message.content)
    return data.get("pairs", [])


# ── Step 2: Quality filtering ─────────────────────────────────────────────────

def quality_filter(pairs: list[dict]) -> list[dict]:
    """Filter out low-quality pairs based on simple heuristics."""
    filtered = []
    for pair in pairs:
        instruction = pair.get("instruction", "")
        response    = pair.get("response", "")

        # Skip if too short
        if len(instruction) < 15 or len(response) < 20:
            continue
        # Skip if response is a question (LLM sometimes generates Q instead of A)
        if response.strip().endswith("?"):
            continue
        # Skip if instruction and response are nearly identical
        if instruction.lower()[:30] in response.lower():
            continue

        filtered.append(pair)
    return filtered


# ── Step 3: Format as ChatML (TinyLlama / Llama-2 chat format) ───────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable bank assistant. Answer customer questions clearly "
    "and specifically, using exact figures and conditions from bank policy."
)

def to_chatml(pair: dict) -> dict:
    """Format as ChatML conversation string for SFT training."""
    text = (
        f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
        f"<|user|>\n{pair['instruction']}</s>\n"
        f"<|assistant|>\n{pair['response']}</s>"
    )
    return {
        "text":        text,
        "instruction": pair["instruction"],
        "response":    pair["response"],
        "source":      "synthetic_banking",
    }


def to_alpaca(pair: dict) -> dict:
    """Alternative: Alpaca format for OPT / older models."""
    return {
        "text": (
            "Below is an instruction.\n\n"
            f"### Instruction:\n{pair['instruction']}\n\n"
            f"### Response:\n{pair['response']}"
        ),
        "instruction": pair["instruction"],
        "response":    pair["response"],
    }


# ── Step 4: DPO preference pairs (chosen vs rejected) ────────────────────────

DPO_PROMPT = """For the following bank assistant question, generate:
1. A GOOD response: specific, accurate, includes exact figures, professional tone
2. A BAD response: vague, missing key numbers, overly generic, or slightly inaccurate

Question: {instruction}
Policy context: {context}

Return JSON: {{"chosen": str, "rejected": str}}"""


def generate_dpo_pair(instruction: str, context: str) -> dict | None:
    res = client.chat.completions.create(
        model=MODEL,
        temperature=0.5,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": DPO_PROMPT.format(
            instruction=instruction, context=context
        )}],
    )
    data = json.loads(res.choices[0].message.content)
    if "chosen" not in data or "rejected" not in data:
        return None
    return {
        "prompt":   instruction,
        "chosen":   data["chosen"],
        "rejected": data["rejected"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_sft_pairs: list[dict] = []

    print("Step 1: Generating instruction pairs from raw documents...")
    for i, doc in enumerate(RAW_DOCS):
        print(f"  Doc {i+1}/{len(RAW_DOCS)}: {doc[:60]}...")
        pairs = generate_pairs(doc, n=3)
        pairs = quality_filter(pairs)
        all_sft_pairs.extend(pairs)
        print(f"    → {len(pairs)} pairs kept")

    print(f"\nTotal pairs before filter: {len(all_sft_pairs)}")

    # Format
    chatml_pairs = [to_chatml(p) for p in all_sft_pairs]
    print(f"Total pairs after formatting: {len(chatml_pairs)}")

    # Split 80/10/10
    random.shuffle(chatml_pairs)
    n       = len(chatml_pairs)
    train   = chatml_pairs[:int(0.8 * n)]
    val     = chatml_pairs[int(0.8 * n):int(0.9 * n)]
    test    = chatml_pairs[int(0.9 * n):]

    # Save as HuggingFace Datasets
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for split_name, split_data in [("train", train), ("validation", val), ("test", test)]:
        ds = Dataset.from_list(split_data)
        ds.save_to_disk(f"{OUTPUT_DIR}/{split_name}")
        print(f"  Saved {split_name}: {len(split_data)} examples → {OUTPUT_DIR}/{split_name}")

    # Step 2: Generate DPO preference pairs (subset)
    print("\nStep 2: Generating DPO preference pairs...")
    dpo_pairs = []
    for pair in all_sft_pairs[:5]:   # small subset for demo
        dpo = generate_dpo_pair(pair["instruction"], RAW_DOCS[0])
        if dpo:
            dpo_pairs.append(dpo)
            print(f"  ✓ '{pair['instruction'][:50]}...'")

    if dpo_pairs:
        dpo_ds = Dataset.from_list(dpo_pairs)
        dpo_ds.save_to_disk(f"{OUTPUT_DIR}/dpo_train")
        print(f"\n  DPO dataset: {len(dpo_pairs)} preference pairs → {OUTPUT_DIR}/dpo_train")

    # Sample output
    print("\n── Sample SFT pair ──")
    sample = chatml_pairs[0]
    print(sample["text"][:400])
    print("...")

    print(f"\n── Dataset stats ──")
    print(f"  SFT train:      {len(train)} examples")
    print(f"  SFT validation: {len(val)} examples")
    print(f"  SFT test:       {len(test)} examples")
    print(f"  DPO train:      {len(dpo_pairs)} preference pairs")
    print(f"\nSaved to: {OUTPUT_DIR}/")
    print("Load with: datasets.load_from_disk('data/09_finetuning/banking_instructions/train')")


if __name__ == "__main__":
    main()

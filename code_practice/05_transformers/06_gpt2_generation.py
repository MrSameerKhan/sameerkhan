# Session 6 — GPT-2 Causal Language Modeling + Text Generation
# Task   : fine-tune GPT-2 on domain text, then generate synthetic training examples
# Dataset: wikitext-2-raw-v1 (~2M tokens of Wikipedia prose — swap for domain text in prod)
# Model  : gpt2 (117M params) — causal decoder, predicts next token
# Metric : perplexity = exp(cross-entropy loss)
#
# Production use case: you have 500 labelled examples → generate 10K synthetic variants
# by fine-tuning GPT-2 on your domain corpus, then sampling with diverse prompts.

import os
import math
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gpt2"
MAX_LEN    = 256
BATCH_SIZE = 8
EPOCHS     = 3
LR         = 5e-5
TRAIN_SIZE = 2000    # number of text chunks
SAVE_DIR   = "models/05_transformers/gpt2_generation"
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class TextChunkDataset(Dataset):
    """Concatenate all text → split into fixed-length chunks for causal LM training."""

    def __init__(self, raw_texts: list[str], tokenizer, chunk_size: int = MAX_LEN):
        all_ids = []
        for text in raw_texts:
            if text.strip():
                all_ids.extend(tokenizer.encode(text))

        # Chunk into non-overlapping windows of chunk_size
        self.chunks = [
            all_ids[i : i + chunk_size]
            for i in range(0, len(all_ids) - chunk_size, chunk_size)
        ]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {
            "input_ids": ids,  # (256,)
            "labels":    ids,  # same as input — GPT-2 shifts internally, loss on next-token
        }


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(DEVICE),
            labels=   batch["labels"].to(DEVICE),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss, math.exp(avg_loss)   # (loss, perplexity)


# ── Generation ────────────────────────────────────────────────────────────────
def generate(prompt, model, tokenizer, max_new_tokens=80, strategy="top_p"):
    """
    strategy options:
      "greedy"      — deterministic, repetitive
      "top_k"       — sample from top-k tokens (diverse but coherent)
      "top_p"       — nucleus sampling, top-p cumulative probability (default)
      "temperature" — scale logits before softmax (higher = more random)
    """
    model.eval()
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(DEVICE)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=(strategy != "greedy"),
    )
    if strategy == "top_k":
        gen_kwargs.update(top_k=50)
    elif strategy == "top_p":
        gen_kwargs.update(top_p=0.92, temperature=0.9)
    elif strategy == "temperature":
        gen_kwargs.update(temperature=1.4, top_k=0)

    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)

    # Decode only the newly generated tokens (after the prompt)
    new_ids = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token by default
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

    raw       = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_raw = raw["train"].select(range(TRAIN_SIZE))
    texts     = [ex["text"] for ex in train_raw]

    train_loader = DataLoader(
        TextChunkDataset(texts, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        loss, ppl = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}/{EPOCHS} | loss: {loss:.4f} | perplexity: {ppl:.2f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nSaved to {SAVE_DIR}")

    # Demonstrate 3 sampling strategies on the same prompt
    print("\n── Generation — same prompt, different strategies ──")
    prompt = "The bank's loan approval process requires the applicant to"

    for strategy in ("greedy", "top_k", "top_p"):
        output = generate(prompt, model, tokenizer, strategy=strategy)
        print(f"\n  [{strategy}]")
        print(f"  {prompt}{output}")

    # Synthetic data generation — batch of training examples
    print("\n── Synthetic training data examples ──")
    seeds = [
        "Customer complaint: the interest rate on my account",
        "Loan application status update: your file has been reviewed",
        "The mortgage terms include a fixed rate of",
    ]
    for seed in seeds:
        synthetic = generate(seed, model, tokenizer, max_new_tokens=40, strategy="top_p")
        print(f"  → {seed}{synthetic}")


if __name__ == "__main__":
    main()

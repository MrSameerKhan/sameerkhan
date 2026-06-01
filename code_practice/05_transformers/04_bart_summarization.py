# Session 4 — BART Abstractive Summarization
# Task   : generate a short summary of a long document (abstractive — new words allowed)
# Dataset: CNN/DailyMail v3.0.0 (~300k news article + highlight pairs)
# Model  : facebook/bart-base → fine-tune on CNN/DM (bart-large-cnn is the prod version)
# Metric : ROUGE-L (longest common subsequence between generated and reference summary)

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "facebook/bart-base"
MAX_INPUT_LEN = 512    # encoder — article
MAX_TARGET_LEN = 128   # decoder — summary
BATCH_SIZE    = 4
EPOCHS        = 2
LR            = 3e-5
TRAIN_SIZE    = 2000
SAVE_DIR      = "models/05_transformers/bart_summarization"
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class CNNDMDataset(Dataset):
    def __init__(self, hf_split, tokenizer):
        self.data      = hf_split
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article  = self.data[idx]["article"]
        summary  = self.data[idx]["highlights"]

        enc = self.tokenizer(
            article,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Encode target summary; decoder_input_ids are shifted right internally by BART
        with self.tokenizer.as_target_tokenizer():
            tgt = self.tokenizer(
                summary,
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        labels = tgt["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (512,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (512,)
            "labels":         labels,                            # (128,) target token ids
        }


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=      batch["input_ids"].to(DEVICE),
            attention_mask= batch["attention_mask"].to(DEVICE),
            labels=         batch["labels"].to(DEVICE),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()

    return total_loss / len(loader)


# ── ROUGE-L ───────────────────────────────────────────────────────────────────
def rouge_l(prediction: str, reference: str) -> float:
    """Longest common subsequence F1 between word sequences."""
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()

    # DP table for LCS length
    m, n   = len(pred_tokens), len(ref_tokens)
    dp     = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if pred_tokens[i-1] == ref_tokens[j-1] else max(dp[i-1][j], dp[i][j-1])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall    = lcs / n
    return 2 * precision * recall / (precision + recall)


# ── Inference ─────────────────────────────────────────────────────────────────
def summarize(text, model, tokenizer, max_new_tokens=128, num_beams=4):
    model.eval()
    enc = tokenizer(
        text,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        ids = model.generate(
            enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    raw       = load_dataset("cnn_dailymail", "3.0.0")
    train_raw = raw["train"].select(range(TRAIN_SIZE))

    train_loader = DataLoader(
        CNNDMDataset(train_raw, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}/{EPOCHS} | loss: {loss:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nSaved to {SAVE_DIR}")

    print("\n── Inference ──")
    # Simulated earnings report extract
    article = (
        "Apple Inc. reported third-quarter earnings on Thursday that exceeded Wall Street "
        "expectations, driven by strong iPhone sales in emerging markets and continued growth "
        "in its services segment. Revenue for the quarter came in at $94.9 billion, up 5% "
        "year-over-year, while earnings per share rose to $1.53 compared to analyst estimates "
        "of $1.39. The services division, which includes the App Store, Apple Music, and iCloud, "
        "generated $24.2 billion in revenue, a record for any quarter. CEO Tim Cook highlighted "
        "the company's commitment to AI integration across its product lineup. However, sales in "
        "China declined 6% amid increased competition from domestic smartphone makers. The company "
        "guided for Q4 revenue between $89 billion and $90 billion, slightly below analyst forecasts "
        "of $90.5 billion, citing macroeconomic headwinds."
    )
    reference = "Apple Q3 revenue $94.9B, up 5%, EPS $1.53 beat estimates. Services hit record $24.2B. China sales fell 6%. Q4 guidance slightly below forecasts."
    summary   = summarize(article, model, tokenizer)

    print(f"  Generated : {summary}")
    print(f"  Reference : {reference}")
    print(f"  ROUGE-L   : {rouge_l(summary, reference):.4f}")


if __name__ == "__main__":
    main()

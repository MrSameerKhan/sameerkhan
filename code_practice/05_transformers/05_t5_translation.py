# Session 5 — T5 Neural Machine Translation
# Task   : translate English text to German using T5's prefix-based task specification
# Dataset: Helsinki-NLP/opus_books (en-de, ~700k sentence pairs — subset used)
# Model  : t5-small (60M params) — same API extends to multilingual with Helsinki MarianMT
# Metric : BLEU score (sacrebleu)
#
# Arabic → English extension: swap model for "Helsinki-NLP/opus-mt-ar-en"
# and remove the "translate English to German: " prefix — MarianMT infers direction.

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "t5-small"
PREFIX         = "translate English to German: "
MAX_INPUT_LEN  = 128
MAX_TARGET_LEN = 128
BATCH_SIZE     = 16
EPOCHS         = 3
LR             = 3e-4    # T5 uses higher LR than BERT
TRAIN_SIZE     = 4000
SAVE_DIR       = "models/05_transformers/t5_translation"
DEVICE         = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, hf_split, tokenizer):
        self.data      = hf_split
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = PREFIX + self.data[idx]["translation"]["en"]
        tgt = self.data[idx]["translation"]["de"]

        enc = self.tokenizer(
            src,
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with self.tokenizer.as_target_tokenizer():
            tgt_enc = self.tokenizer(
                tgt,
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        labels = tgt_enc["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (128,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (128,)
            "labels":         labels,                            # (128,)
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


# ── Corpus BLEU (simple n-gram implementation) ────────────────────────────────
def bleu_1gram(prediction: str, reference: str) -> float:
    """Unigram BLEU — educational; use sacrebleu in production."""
    pred_tokens = prediction.lower().split()
    ref_set     = set(reference.lower().split())
    if not pred_tokens:
        return 0.0
    matches = sum(1 for t in pred_tokens if t in ref_set)
    return matches / len(pred_tokens)


# ── Inference ─────────────────────────────────────────────────────────────────
def translate(text, model, tokenizer, num_beams=4):
    model.eval()
    src = PREFIX + text
    enc = tokenizer(src, max_length=MAX_INPUT_LEN, truncation=True, return_tensors="pt")

    with torch.no_grad():
        ids = model.generate(
            enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=num_beams,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

    raw       = load_dataset("Helsinki-NLP/opus_books", "en-de")
    train_raw = raw["train"].select(range(TRAIN_SIZE))

    train_loader = DataLoader(
        TranslationDataset(train_raw, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
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
    sentences = [
        ("The quarterly report shows a 12% increase in revenue.", "Der Quartalsbericht zeigt eine Umsatzsteigerung von 12 Prozent."),
        ("Please submit all documents before the deadline.", "Bitte reichen Sie alle Unterlagen vor Ablauf der Frist ein."),
        ("The loan application requires proof of income.", "Der Kreditantrag erfordert einen Einkommensnachweis."),
    ]
    for en, reference_de in sentences:
        translation = translate(en, model, tokenizer)
        score       = bleu_1gram(translation, reference_de)
        print(f"  EN  : {en}")
        print(f"  DE  : {translation}")
        print(f"  Ref : {reference_de}")
        print(f"  BLEU-1: {score:.3f}\n")


if __name__ == "__main__":
    main()

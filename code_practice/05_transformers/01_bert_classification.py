# Session 1 — BERT Text Classification
# Task   : binary sentiment (positive / negative)
# Dataset: SST-2 (GLUE benchmark, ~67k train sentences)
# Model  : bert-base-uncased → BertForSequenceClassification
# Metric : accuracy

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
TRAIN_SIZE  = 4000   # subset — swap to None for full 67k
VAL_SIZE    = 500
SAVE_DIR    = "models/05_transformers/bert_classification"
DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class SST2Dataset(Dataset):
    def __init__(self, hf_split, tokenizer):
        self.data      = hf_split
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc  = self.tokenizer(
            item["sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (128,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (128,)
            "labels":         torch.tensor(item["label"], dtype=torch.long),
        }


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        # forward — BertForSequenceClassification returns (loss, logits) when labels passed
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds  = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total


# ── Inference (single sentence) ───────────────────────────────────────────────
def predict(text, model, tokenizer):
    model.eval()
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs  = torch.softmax(logits, dim=-1)[0]
    label  = logits.argmax(dim=-1).item()
    labels = {0: "negative", 1: "positive"}

    return {"label": labels[label], "confidence": f"{probs[label].item():.2%}"}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    # 1. Tokenizer + model
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)

    # 2. Data
    raw       = load_dataset("glue", "sst2")
    train_raw = raw["train"].select(range(TRAIN_SIZE)) if TRAIN_SIZE else raw["train"]
    val_raw   = raw["validation"].select(range(VAL_SIZE)) if VAL_SIZE else raw["validation"]

    train_loader = DataLoader(SST2Dataset(train_raw, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SST2Dataset(val_raw,   tokenizer), batch_size=BATCH_SIZE)

    # 3. Optimizer + scheduler
    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # 4. Training loop
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} | loss: {loss:.4f} | val_acc: {acc:.4f}")

    # 5. Save
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nSaved to {SAVE_DIR}")

    # 6. Inference
    print("\n── Inference ──")
    tests = [
        "This movie was absolutely fantastic, I loved every second!",
        "What a waste of time. Terrible acting, boring plot.",
        "It was okay, nothing special but not awful either.",
    ]
    for text in tests:
        result = predict(text, model, tokenizer)
        print(f"  [{result['label']} {result['confidence']}]  {text}")


if __name__ == "__main__":
    main()
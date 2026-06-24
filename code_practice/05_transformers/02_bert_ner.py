# Session 2 — BERT Named Entity Recognition (NER)
# Task   : token-level NER (PER, ORG, LOC, MISC + O)
# Dataset: CoNLL-2003 (~14k train sentences, 9 BIO labels)
# Model  : bert-base-cased → BertForTokenClassification
# Metric : entity-level F1 (seqeval)

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from seqeval.metrics import f1_score

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-cased"
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
TRAIN_SIZE = 4000
VAL_SIZE   = 500
SAVE_DIR   = "models/05_transformers/bert_ner"
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)


# ── Dataset ───────────────────────────────────────────────────────────────────
class CoNLLDataset(Dataset):
    def __init__(self, hf_split, tokenizer):
        self.data      = hf_split
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens   = self.data[idx]["tokens"]
        ner_tags = self.data[idx]["ner_tags"]  # integer ids already

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        # Align labels: first subword → real label, continuation subwords → -100
        labels    = []
        word_ids  = enc.word_ids(batch_index=0)
        prev_word = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)           # [CLS], [SEP], padding
            elif wid != prev_word:
                labels.append(ner_tags[wid])  # first subword of a word
            else:
                labels.append(-100)           # continuation subword
            prev_word = wid

        return {
            "input_ids":      enc["input_ids"].squeeze(0),       # (128,)
            "attention_mask": enc["attention_mask"].squeeze(0),  # (128,)
            "labels":         torch.tensor(labels, dtype=torch.long),  # (128,)
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
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds  = logits.argmax(dim=-1)

            for pred_row, label_row in zip(preds, labels):
                p_seq = [ID2LABEL[p.item()] for p, l in zip(pred_row, label_row) if l.item() != -100]
                l_seq = [ID2LABEL[l.item()] for l in label_row if l.item() != -100]
                all_preds.append(p_seq)
                all_labels.append(l_seq)

    return f1_score(all_labels, all_preds)


# ── Inference (raw sentence) ──────────────────────────────────────────────────
def predict(sentence, model, tokenizer):
    model.eval()
    words = sentence.split()
    enc   = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    word_ids       = enc.word_ids(batch_index=0)
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (1, 128, 9)

    pred_ids = logits.argmax(dim=-1)[0]  # (128,)

    seen, results = set(), []
    for pos, wid in enumerate(word_ids):
        if wid is not None and wid not in seen:
            seen.add(wid)
            results.append((words[wid], ID2LABEL[pred_ids[pos].item()]))

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = BertForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(DEVICE)

    raw       = load_dataset("conll2003", trust_remote_code=True)
    train_raw = raw["train"].select(range(TRAIN_SIZE)) if TRAIN_SIZE else raw["train"]
    val_raw   = raw["validation"].select(range(VAL_SIZE)) if VAL_SIZE else raw["validation"]

    train_loader = DataLoader(CoNLLDataset(train_raw, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(CoNLLDataset(val_raw,   tokenizer), batch_size=BATCH_SIZE)

    optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        f1   = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} | loss: {loss:.4f} | val_f1: {f1:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nSaved to {SAVE_DIR}")

    print("\n-- Inference --")
    tests = [
        "Apple CEO Tim Cook announced new products in San Francisco on Tuesday .",
        "Elon Musk 's company Tesla is based in Austin Texas .",
    ]
    for sentence in tests:
        results = predict(sentence, model, tokenizer)
        for word, label in results:
            if label != "O":
                print(f"  {word:20s}  {label}")
        print()


if __name__ == "__main__":
    main()

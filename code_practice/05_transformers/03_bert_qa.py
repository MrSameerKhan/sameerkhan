# Session 3 — BERT Extractive Question Answering
# Task   : extract the exact answer span from a context paragraph
# Dataset: SQuAD v1.1 (~87k question-context-answer triples)
# Model  : bert-base-uncased → BertForQuestionAnswering
# Metric : span-level loss (Exact Match / F1 require post-processing)

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
MAX_LEN     = 384          # QA needs longer window than classification
BATCH_SIZE  = 8            # smaller — 384 tokens per example
EPOCHS      = 2
LR          = 3e-5
TRAIN_SIZE  = 4000
SAVE_DIR    = "models/05_transformers/bert_qa"
DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset ───────────────────────────────────────────────────────────────────
class SQuADDataset(Dataset):
    def __init__(self, hf_split, tokenizer):
        self.data      = [ex for ex in hf_split if ex["answers"]["text"]]  # skip unanswerable
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex        = self.data[idx]
        question  = ex["question"]
        context   = ex["context"]
        answer    = ex["answers"]["text"][0]
        ans_start = ex["answers"]["answer_start"][0]
        ans_end   = ans_start + len(answer)

        enc = self.tokenizer(
            question,
            context,
            max_length=MAX_LEN,
            truncation="only_second",   # never truncate the question
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = enc.pop("offset_mapping").squeeze(0)  # (384, 2) — char spans per token
        sequence_ids   = enc.sequence_ids(batch_index=0)       # 0=question, 1=context, None=special

        start_pos, end_pos = 0, 0
        for i, (seq_id, (char_s, char_e)) in enumerate(zip(sequence_ids, offset_mapping)):
            if seq_id != 1:
                continue
            if char_s <= ans_start < char_e:
                start_pos = i
            if char_s < ans_end <= char_e:
                end_pos = i

        return {
            "input_ids":       enc["input_ids"].squeeze(0),        # (384,)
            "attention_mask":  enc["attention_mask"].squeeze(0),   # (384,)
            "token_type_ids":  enc["token_type_ids"].squeeze(0),   # (384,) 0=Q, 1=context
            "start_positions": torch.tensor(start_pos, dtype=torch.long),
            "end_positions":   torch.tensor(end_pos,   dtype=torch.long),
        }


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=       batch["input_ids"].to(DEVICE),
            attention_mask=  batch["attention_mask"].to(DEVICE),
            token_type_ids=  batch["token_type_ids"].to(DEVICE),
            start_positions= batch["start_positions"].to(DEVICE),
            end_positions=   batch["end_positions"].to(DEVICE),
        )
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += outputs.loss.item()

    return total_loss / len(loader)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(question, context, model, tokenizer):
    model.eval()
    enc = tokenizer(
        question,
        context,
        max_length=MAX_LEN,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offset_mapping = enc.pop("offset_mapping").squeeze(0)
    sequence_ids   = enc.sequence_ids(batch_index=0)

    with torch.no_grad():
        outputs = model(
            input_ids=      enc["input_ids"].to(DEVICE),
            attention_mask= enc["attention_mask"].to(DEVICE),
            token_type_ids= enc["token_type_ids"].to(DEVICE),
        )

    start_idx = outputs.start_logits.argmax().item()
    end_idx   = outputs.end_logits.argmax().item()

    # Clamp to context region; ensure start <= end
    ctx_start = next(i for i, s in enumerate(sequence_ids) if s == 1)
    ctx_end   = next(i for i in range(len(sequence_ids) - 1, -1, -1) if sequence_ids[i] == 1)
    start_idx = max(start_idx, ctx_start)
    end_idx   = min(max(end_idx, start_idx), ctx_end)

    char_start = offset_mapping[start_idx][0].item()
    char_end   = offset_mapping[end_idx][1].item()
    return context[char_start:char_end]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    model     = BertForQuestionAnswering.from_pretrained(MODEL_NAME).to(DEVICE)

    raw       = load_dataset("squad")
    train_raw = raw["train"].select(range(TRAIN_SIZE))

    train_loader = DataLoader(
        SQuADDataset(train_raw, tokenizer), batch_size=BATCH_SIZE, shuffle=True
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
    context = (
        "The maximum loan-to-value ratio for residential mortgages is 90% for "
        "standard borrowers and 95% for first-time buyers under the Help to Buy scheme. "
        "Interest rates are fixed for an initial period of 5 years, after which they "
        "revert to the lender's standard variable rate. Early repayment charges of 2% "
        "apply within the fixed-rate period."
    )
    questions = [
        "What is the maximum LTV for first-time buyers?",
        "How long is the fixed interest rate period?",
        "What is the early repayment charge?",
    ]
    for q in questions:
        answer = predict(q, context, model, tokenizer)
        print(f"  Q: {q}")
        print(f"  A: {answer}\n")


if __name__ == "__main__":
    main()

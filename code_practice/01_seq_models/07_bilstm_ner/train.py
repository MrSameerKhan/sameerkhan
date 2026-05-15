"""
Session 7 — BiLSTM for NER
train.py: Training loop with val tracking, best-checkpoint save, F1 metrics, error analysis.
"""

import sys
import os
import time
from collections import Counter

import torch
import torch.nn as nn

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from data  import get_loaders, PAD_LABEL_ID
from model import BiLSTMTagger

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_TOTAL    = 200
EMBED_DIM  = 64
HIDDEN     = 64
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
SEED       = 42
CHECKPOINT = os.path.join(_DIR, "checkpoints", "bilstm_ner.pt")
# ──────────────────────────────────────────────────────────────────────────────

try:
    from seqeval.metrics import classification_report, f1_score as seq_f1
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False


def token_f1(preds, labels, label_pad_id):
    """Fallback: token-level F1 ignoring PAD."""
    tp = fp = fn = 0
    for p, l in zip(preds.view(-1), labels.view(-1)):
        if l.item() == label_pad_id:
            continue
        if p.item() == l.item():
            tp += 1
        else:
            fp += 1
            fn += 1
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return 2 * p * r / (p + r + 1e-8)


def evaluate(model, loader, criterion, device, idx2label, label_pad_id):
    model.eval()
    total_loss = 0.0
    true_seqs, pred_seqs = [], []

    with torch.no_grad():
        for tokens, labels, lengths in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)
            loss   = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)   # [batch, seq]
            for i in range(len(lengths)):
                L = lengths[i].item()
                true_seqs.append([idx2label[labels[i, j].item()] for j in range(L)])
                pred_seqs.append([idx2label[preds[i, j].item()]  for j in range(L)])

    avg_loss = total_loss / len(loader)
    if HAS_SEQEVAL:
        f1 = seq_f1(true_seqs, pred_seqs)
    else:
        f1 = token_f1(
            torch.tensor([[idx2label.get(t, "O") for t in s] for s in pred_seqs]),
            torch.tensor([[idx2label.get(t, "O") for t in s] for s in true_seqs]),
            label_pad_id,
        )
    return avg_loss, f1, true_seqs, pred_seqs


def print_label_dist(data, label_name="Train"):
    tag_counts = Counter(tag for _, tags in data for tag in tags)
    total = sum(tag_counts.values())
    print(f"\nTrain label distribution (~{total} tokens):")
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag:<15} ~{cnt:4d}  ({100*cnt/total:.0f}%)")


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print("Loading data...")

    loaders, word2idx, idx2word, label2idx, idx2label, train_data, val_data, test_data = \
        get_loaders(n_total=N_TOTAL, batch_size=BATCH_SIZE, seed=SEED)
    train_loader, val_loader, test_loader = loaders

    vocab_size  = len(word2idx)
    num_labels  = len(label2idx)
    label_pad_id = label2idx["<PAD>"]

    print(f"Device:      {device}")
    print(f"Train:       {len(train_data)} examples")
    print(f"Val:         {len(val_data)} examples")
    print(f"Test:        {len(test_data)} examples")
    print(f"Vocab:       ~{vocab_size} words (incl <PAD>, <UNK>)")
    print(f"Labels:      {num_labels} (incl <PAD>) {sorted(label2idx.keys())}")

    # OOV stats
    train_words = {t for tokens, _ in train_data for t in tokens}
    for split_name, split_data in [("Val", val_data), ("Test", test_data)]:
        all_toks = [t for tokens, _ in split_data for t in tokens]
        oov = [t for t in all_toks if t not in train_words]
        print(f"  {split_name}: ~{len(all_toks)} tokens, ~{len(oov)} OOV ({100*len(oov)/max(1,len(all_toks)):.0f}%)")

    print_label_dist(train_data)

    model     = BiLSTMTagger(vocab_size, num_labels, EMBED_DIM, HIDDEN).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=label_pad_id)

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

    best_val_f1 = -1.0
    best_epoch  = -1
    t0 = time.time()

    print(f"\n=== Training ({EPOCHS} epochs) ===")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val F1':>6} | Best")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for tokens, labels, lengths in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(tokens)
            loss   = criterion(logits.reshape(-1, num_labels), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, device, idx2label, label_pad_id)

        star = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            star = " ★"
            model.save(CHECKPOINT,
                       vocab_size=vocab_size, num_labels=num_labels,
                       embed_dim=EMBED_DIM, hidden_size=HIDDEN,
                       word2idx=word2idx, idx2word=idx2word,
                       label2idx=label2idx, idx2label=idx2label)

        print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_f1:>6.4f} |{star}")

    elapsed = time.time() - t0
    print(f"\nBest val F1: {best_val_f1:.4f} at epoch {best_epoch} → checkpoint saved.")
    print(f"Training time: {elapsed:.1f}s")

    # Final test evaluation
    print("\n=== Test Set Evaluation ===")
    test_loss, test_f1, true_seqs, pred_seqs = evaluate(
        model, test_loader, criterion, device, idx2label, label_pad_id)
    print(f"Test F1: {test_f1:.4f}")

    if HAS_SEQEVAL:
        print("\nPer-entity classification report (test set):")
        print(classification_report(true_seqs, pred_seqs))
    else:
        print("Install seqeval for per-entity report: pip install seqeval")


if __name__ == "__main__":
    main()

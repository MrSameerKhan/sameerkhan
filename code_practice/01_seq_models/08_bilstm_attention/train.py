"""
Session 8 — Bahdanau Attention on BiLSTM
train.py: Trains all 4 pooling variants sequentially, prints comparison table,
          saves attention model checkpoint.
"""

import sys
import os
import time
import torch
import torch.nn as nn

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from data  import get_loaders
from model import BiLSTMClassifier

# ── Hyperparameters ────────────────────────────────────────────────────────────
N_TOTAL    = 200
EMBED_DIM  = 64
HIDDEN     = 64
BATCH_SIZE = 32
EPOCHS     = 30
LR         = 1e-3
SEED       = 42
CHECKPOINT = os.path.join(_DIR, "checkpoints", "bilstm_attention.pt")
# ──────────────────────────────────────────────────────────────────────────────


def accuracy(logits, labels):
    return (logits.argmax(dim=-1) == labels).float().mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_acc = 0.0
    with torch.no_grad():
        for tokens, labels, lengths, mask in loader:
            tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
            logits, _ = model(tokens, mask)
            total_loss += criterion(logits, labels).item()
            total_acc  += accuracy(logits, labels)
    return total_loss / len(loader), total_acc / len(loader)


def train_variant(pool_type, loaders, word2idx, label2idx, device):
    train_loader, val_loader, test_loader = loaders
    vocab_size  = len(word2idx)
    num_classes = len(label2idx)

    model = BiLSTMClassifier(vocab_size, num_classes, EMBED_DIM, HIDDEN,
                             pool=pool_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for tokens, labels, lengths, mask in train_loader:
            tokens, labels, mask = tokens.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            logits, _ = model(tokens, mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best checkpoint for test eval
    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, test_loader, criterion, device)

    return model, total_params, best_val_acc, test_acc


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    loaders, word2idx, idx2word, label2idx, idx2label, *_ = \
        get_loaders(n_total=N_TOTAL, batch_size=BATCH_SIZE, seed=SEED)

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

    results = {}
    pool_types = ["attention", "mean", "max", "last"]

    for pool in pool_types:
        print(f"\nTraining pool={pool} ...")
        t0 = time.time()
        model, params, val_acc, test_acc = train_variant(
            pool, loaders, word2idx, label2idx, device)
        elapsed = time.time() - t0
        results[pool] = {"params": params, "val_acc": val_acc,
                         "test_acc": test_acc, "time": elapsed}
        print(f"  {pool:<12} params={params:,}  val={val_acc:.4f}  test={test_acc:.4f}  ({elapsed:.1f}s)")

        if pool == "attention":
            model.save(CHECKPOINT,
                       vocab_size=len(word2idx), num_classes=len(label2idx),
                       embed_dim=EMBED_DIM, hidden_size=HIDDEN,
                       word2idx=word2idx, idx2word=idx2word,
                       label2idx=label2idx, idx2label=idx2label)

    print("\n" + "=" * 60)
    print(f"{'Pool type':<12} | {'Params':>8} | {'Val acc':>8} | {'Test acc':>8}")
    print("-" * 60)
    for pool, r in results.items():
        print(f"{pool:<12} | {r['params']:>8,} | {r['val_acc']:>8.4f} | {r['test_acc']:>8.4f}")
    print("=" * 60)
    print(f"\nAttention model saved → {CHECKPOINT}")


if __name__ == "__main__":
    main()

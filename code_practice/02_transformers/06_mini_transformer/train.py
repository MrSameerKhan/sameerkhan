"""
Phase 2, Session 6 — Mini Transformer Encoder
train.py: Train 1, 2, 4-layer variants back-to-back. Compare convergence and accuracy.
          Save the 4-layer model. Auto-run adversarial test.
"""

import sys
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import MiniTransformerEncoder
from data  import get_data, make_batch, IDX2LABEL

CKPT = os.path.join(_DIR, "checkpoints", "mini_encoder_4l.pt")

D_MODEL   = 64
NUM_HEADS = 8
D_FF      = 256
EPOCHS    = 30
LR        = 1e-3
BATCH     = 16

ADVERSARIAL = [
    "The Premium Checking has 2.5 percent rate.",
    "Apply for Personal Loan with 8.5 percent rate.",
    "Sarah Chen works as Head of Loans in the Loans department.",
]


def train_variant(n_layers, train_data, val_data, test_data, word2idx, device):
    model = MiniTransformerEncoder(
        vocab_size=len(word2idx), d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, n_classes=5, num_layers=n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = 0.0
    best_val_epoch = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idxs = list(range(len(train_data)))
        random.shuffle(idxs)
        for start in range(0, len(idxs), BATCH):
            batch = [train_data[i] for i in idxs[start:start + BATCH]]
            token_ids, label_ids, pad_mask = make_batch(batch, word2idx, device)
            logits, _ = model(token_ids, pad_mask)
            loss = criterion(logits, label_ids)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            token_ids, label_ids, pad_mask = make_batch(val_data, word2idx, device)
            val_acc = (model(token_ids, pad_mask)[0].argmax(-1) == label_ids).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        token_ids, label_ids, pad_mask = make_batch(test_data, word2idx, device)
        test_acc = (model(token_ids, pad_mask)[0].argmax(-1) == label_ids).float().mean().item()

    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, best_val_epoch, best_val_acc, test_acc


def run_adversarial(model, word2idx, device):
    model.eval()
    print("\n=== Adversarial test (4-layer model) ===")
    from data import encode
    with torch.no_grad():
        for text in ADVERSARIAL:
            ids      = encode(text, word2idx)
            t        = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(t)
            probs    = F.softmax(logits, dim=-1)[0]
            pred     = IDX2LABEL[probs.argmax().item()]
            conf     = probs.max().item()
            print(f"  '{text}'")
            print(f"    → {pred} ({conf*100:.1f}%)")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_data, val_data, test_data, word2idx = get_data()
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Vocab: {len(word2idx)}\n")

    results = []
    model_4l = None

    for n_layers in [1, 2, 4]:
        t0 = time.time()
        model, n_params, best_epoch, val_acc, test_acc = train_variant(
            n_layers, train_data, val_data, test_data, word2idx, device
        )
        elapsed = time.time() - t0
        results.append((n_layers, n_params, best_epoch, val_acc, test_acc, elapsed))
        print(f"  {n_layers}-layer: {n_params:,} params  |  best epoch {best_epoch:2d}  |  "
              f"val {val_acc:.4f}  test {test_acc:.4f}  ({elapsed:.1f}s)")
        if n_layers == 4:
            model_4l  = model
            word2idx_ = word2idx

    print(f"\n{'─'*62}")
    print(f"{'Layers':>6} | {'Params':>8} | {'Best Epoch':>10} | {'Val Acc':>7} | {'Test Acc':>8}")
    print(f"{'─'*62}")
    for n_layers, n_params, best_epoch, val_acc, test_acc, _ in results:
        print(f"  {n_layers:4d}  | {n_params:>8,} | {best_epoch:>10d} | {val_acc:>7.4f} | {test_acc:>8.4f}")
    print(f"{'─'*62}")

    run_adversarial(model_4l, word2idx_, device)

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({
        "model_state": model_4l.state_dict(),
        "word2idx":    word2idx_,
        "d_model":     D_MODEL,
        "num_heads":   NUM_HEADS,
        "d_ff":        D_FF,
        "num_layers":  4,
    }, CKPT)
    print(f"\nSaved: {CKPT}")


if __name__ == "__main__":
    main()

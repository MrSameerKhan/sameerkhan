"""
Phase 2, Session 5 — Transformer Encoder Block
train.py: Train full encoder block, auto-test adversarial sentence, show convergence vs prior sessions.
"""

import sys
import os
import random
import torch
import torch.nn as nn

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import EncoderClassifier
from data  import get_data, make_batch, IDX2LABEL

CKPT = os.path.join(_DIR, "checkpoints", "encoder.pt")

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


def run_adversarial(model, word2idx, device):
    model.eval()
    print("\n=== Adversarial test ===")
    with torch.no_grad():
        from data import encode, tokenize
        for text in ADVERSARIAL:
            ids      = encode(text, word2idx)
            t        = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(t)
            import torch.nn.functional as F
            probs    = F.softmax(logits, dim=-1)[0]
            pred     = IDX2LABEL[probs.argmax().item()]
            conf     = probs.max().item()
            print(f"  '{text}'")
            print(f"    → {pred} ({conf*100:.1f}%)")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, val_data, test_data, word2idx = get_data()
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Vocab: {len(word2idx)}")

    model = EncoderClassifier(
        vocab_size=len(word2idx), d_model=D_MODEL, num_heads=NUM_HEADS,
        d_ff=D_FF, n_classes=5
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"  Session 02 (single-head NumPy): ~9,157")
    print(f"  Session 03 (MHA NumPy):         ~10,181")
    print(f"  Session 04 (RoPE PyTorch):       28,549")
    print(f"  Session 05 (full encoder):      {n_params:,}  ← +FFN + LayerNorms")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = 0.0
    best_val_epoch = 0
    best_state     = None

    print(f"\n Epoch |    loss |   train |     val |  time")
    print(f"----------------------------------------------")

    import time
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0   = time.time()
        idxs = list(range(len(train_data)))
        random.shuffle(idxs)
        total_loss = 0.0
        for start in range(0, len(idxs), BATCH):
            batch = [train_data[i] for i in idxs[start:start + BATCH]]
            token_ids, label_ids, pad_mask = make_batch(batch, word2idx, device)
            logits, _ = model(token_ids, pad_mask)
            loss = criterion(logits, label_ids)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            token_ids, label_ids, pad_mask = make_batch(train_data, word2idx, device)
            train_acc = (model(token_ids, pad_mask)[0].argmax(-1) == label_ids).float().mean().item()

            token_ids, label_ids, pad_mask = make_batch(val_data, word2idx, device)
            val_acc = (model(token_ids, pad_mask)[0].argmax(-1) == label_ids).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_val_epoch = epoch
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0:
            elapsed = time.time() - t0
            avg_loss = total_loss / max(1, len(train_data) // BATCH)
            print(f"  {epoch:5d} | {avg_loss:.4f} | {train_acc:.4f} | {val_acc:.4f} | {elapsed:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        token_ids, label_ids, pad_mask = make_batch(test_data, word2idx, device)
        test_acc = (model(token_ids, pad_mask)[0].argmax(-1) == label_ids).float().mean().item()

    print(f"\nBest val acc : {best_val_acc:.4f}  (epoch {best_val_epoch})")
    print(f"Test acc     : {test_acc:.4f}")

    run_adversarial(model, word2idx, device)

    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "word2idx":    word2idx,
        "d_model":     D_MODEL,
        "num_heads":   NUM_HEADS,
        "d_ff":        D_FF,
    }, CKPT)
    print(f"\nSaved: {CKPT}")


if __name__ == "__main__":
    main()

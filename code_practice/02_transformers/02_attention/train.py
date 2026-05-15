"""
Session 10 — Scaled Dot-Product Self-Attention (NumPy from scratch)
train.py: Train SelfAttentionClassifier on Acme classification task.
          Reports train loss, train acc, val acc every 5 epochs.
          Saves best checkpoint by val accuracy.
"""

import sys
import os
import random
import time
import numpy as np

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from data  import get_data, encode, LABEL2IDX, IDX2LABEL
from model import SelfAttentionClassifier

EPOCHS   = 50
LR       = 3e-3
D_MODEL  = 32
D_K      = 32
SEED     = 42
CKPT     = os.path.join(_DIR, "checkpoints", "self_attn.pkl")


def compute_accuracy(model, data, word2idx):
    correct = 0
    for text, label in data:
        ids, _ = encode(text, word2idx)
        mask   = np.ones(len(ids), dtype=int)
        probs, _ = model.forward(ids, mask)
        if probs.argmax() == LABEL2IDX[label]:
            correct += 1
    return correct / len(data)


def main():
    train_data, val_data, test_data, word2idx = get_data(seed=SEED)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Vocab: {len(word2idx)}")

    model = SelfAttentionClassifier(
        vocab_size=len(word2idx),
        d_model=D_MODEL,
        d_k=D_K,
        n_classes=len(LABEL2IDX),
        seed=SEED,
    )
    print(f"Parameters: {model.n_params:,}\n")

    best_val_acc = 0.0
    rng = random.Random(SEED)

    print(f"{'Epoch':>6} | {'loss':>7} | {'train':>7} | {'val':>7} | {'time':>5}")
    print("-" * 46)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        rng.shuffle(train_data)
        total_loss = 0.0

        for text, label in train_data:
            ids, _       = encode(text, word2idx)
            mask         = np.ones(len(ids), dtype=int)
            y            = LABEL2IDX[label]
            probs, cache = model.forward(ids, mask)
            loss, grads  = model.backward(cache, y)
            model.update(grads, lr=LR)
            total_loss  += loss

        if epoch % 5 == 0:
            train_acc = compute_accuracy(model, train_data, word2idx)
            val_acc   = compute_accuracy(model, val_data,   word2idx)
            dt        = time.time() - t0
            print(f"{epoch:6d} | {total_loss/len(train_data):7.4f} | "
                  f"{train_acc:7.4f} | {val_acc:7.4f} | {dt:.1f}s")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save(CKPT, word2idx=word2idx, idx2label=IDX2LABEL,
                           label2idx=LABEL2IDX)

    print(f"\nBest val acc : {best_val_acc:.4f}")

    best_model, _ = SelfAttentionClassifier.load(CKPT)
    test_acc = compute_accuracy(best_model, test_data, word2idx)
    print(f"Test acc     : {test_acc:.4f}")


if __name__ == "__main__":
    main()

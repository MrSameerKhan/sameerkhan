"""
Session 3 — LSTM Cell from Scratch
train.py: char-level LM on Acme corpus. Reports train loss and val PPL.
Compare val PPL vs Session 1 RNN.
"""

import sys
import os
import time
import math

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

import numpy as np
from code_practice.shared_dataset import get_corpus_as_string, get_char_vocab
from model import VanillaLSTM

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN     = 50
HIDDEN_SIZE = 128
EPOCHS      = 2000
LR          = 1e-2
SEED        = 42
VAL_SEQS    = 20
CHECKPOINT  = os.path.join(_DIR, "checkpoints", "lstm_scratch.npz")
# ──────────────────────────────────────────────────────────────────────────────

np.random.seed(SEED)


def compute_val_ppl(model, val_data, seq_len, n_seqs=VAL_SEQS):
    """Forward-only on n_seqs random val windows. Returns avg per-char loss and PPL."""
    total_loss = 0.0
    h = np.zeros((model.hidden_size, 1))
    c = np.zeros((model.hidden_size, 1))
    for _ in range(n_seqs):
        pos     = np.random.randint(0, len(val_data) - seq_len - 1)
        inputs  = val_data[pos     : pos + seq_len]
        targets = val_data[pos + 1 : pos + seq_len + 1]
        loss, _, _ = model.loss(inputs, targets, h, c)
        total_loss += loss / seq_len
    avg_loss = total_loss / n_seqs
    return math.exp(avg_loss)


def main():
    text       = get_corpus_as_string()
    char2idx, idx2char, vocab_size = get_char_vocab()
    data       = [char2idx[c] for c in text]
    N          = len(data)

    # 90 / 10 train / val split
    split      = int(N * 0.9)
    train_data = data[:split]
    val_data   = data[split:]

    print(f"Corpus length : {N:,} chars  (train: {len(train_data):,} | val: {len(val_data):,})")
    print(f"Vocab size    : {vocab_size}")
    print(f"Hidden size   : {HIDDEN_SIZE}")

    model = VanillaLSTM(vocab_size, HIDDEN_SIZE, lr=LR, seed=SEED)
    total_params = (4 * HIDDEN_SIZE * (HIDDEN_SIZE + vocab_size) +
                    4 * HIDDEN_SIZE +
                    vocab_size * HIDDEN_SIZE + vocab_size)
    print(f"Parameters    : {total_params:,}\n")

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)

    smooth_loss = -np.log(1.0 / vocab_size) * SEQ_LEN
    best_loss   = float("inf")
    t0          = time.time()

    for epoch in range(1, EPOCHS + 1):
        pos = np.random.randint(0, len(train_data) - SEQ_LEN - 1)
        inputs  = train_data[pos     : pos + SEQ_LEN]
        targets = train_data[pos + 1 : pos + SEQ_LEN + 1]

        h = np.zeros((HIDDEN_SIZE, 1))
        c = np.zeros((HIDDEN_SIZE, 1))

        loss, h, c, grads = model.backward(inputs, targets, h, c)
        model.update(grads)

        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            model.save(CHECKPOINT)

        if epoch % 100 == 0:
            val_ppl = compute_val_ppl(model, val_data, SEQ_LEN)
            elapsed = time.time() - t0
            print(f"Epoch {epoch:>5} | train_loss: {smooth_loss:.4f} | val_ppl: {val_ppl:.2f} | {elapsed:.1f}s")

        if epoch % 500 == 0:
            sample_idx = model.sample(train_data[0], length=200, temperature=0.8)
            sample_txt = "".join(idx2char[i] for i in sample_idx)
            print(f"\n--- Sample at epoch {epoch} ---")
            print(sample_txt)
            print()

    print(f"\nBest train loss : {best_loss:.4f}")
    print(f"Checkpoint      : {CHECKPOINT}")

    sample_idx = model.sample(train_data[0], length=300, temperature=0.8)
    sample_txt = "".join(idx2char[i] for i in sample_idx)
    print("\n--- Final sample ---")
    print(sample_txt)


if __name__ == "__main__":
    main()

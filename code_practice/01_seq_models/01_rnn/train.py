"""
Session 1 — Vanilla RNN Cell from Scratch
train.py: Train char-level LM on Acme Financial corpus. Pure NumPy, no autograd.
Reports train smooth-loss and val perplexity every LOG_EVERY epochs.
"""

import sys
import os
import time
import math
import numpy as np

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from code_practice.shared_dataset import get_corpus_as_string, get_char_vocab
from model import VanillaRNN

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_SIZE   = 128
SEQ_LEN       = 25
LEARNING_RATE = 0.1
EPOCHS        = 2000
LOG_EVERY     = 100
SAMPLE_LEN    = 80
VAL_SEQS      = 20       # number of random val sequences for PPL estimate
CHECKPOINT    = os.path.join(_DIR, "checkpoints", "rnn_scratch")
# ──────────────────────────────────────────────────────────────────────────────


def encode(text, char2idx):
    return [char2idx[c] for c in text]


def to_onehot(idx, vocab_size):
    x = np.zeros((vocab_size, 1))
    x[idx] = 1.0
    return x


def sample_text(model, seed_char, char2idx, idx2char, h, length=SAMPLE_LEN):
    seed_idx = char2idx.get(seed_char, 0)
    idxs = model.sample(seed_idx, h, length)
    return seed_char + "".join(idx2char[i] for i in idxs)


def compute_val_ppl(model, val_data, vocab_size, seq_len, n_seqs=VAL_SEQS):
    """Run forward-only on n_seqs random val windows, return avg loss and PPL."""
    total_loss = 0.0
    h = np.zeros((model.hidden_size, 1))
    for _ in range(n_seqs):
        pos     = np.random.randint(0, len(val_data) - seq_len - 1)
        inputs  = [to_onehot(val_data[pos + i],     vocab_size) for i in range(seq_len)]
        targets = [val_data[pos + i + 1] for i in range(seq_len)]
        _, _, _, ps = model.forward(inputs, h)
        loss = sum(-np.log(ps[t][targets[t], 0] + 1e-8) for t in range(seq_len))
        total_loss += loss / seq_len   # per-char loss
    avg_loss = total_loss / n_seqs
    return avg_loss, math.exp(avg_loss)


def main():
    text = get_corpus_as_string()
    char2idx, idx2char, vocab_size = get_char_vocab()
    data = encode(text, char2idx)
    N    = len(data)

    # 90 / 10 train / val split
    split     = int(N * 0.9)
    train_data = data[:split]
    val_data   = data[split:]

    print(f"Vocab size  : {vocab_size}")
    print(f"Hidden size : {HIDDEN_SIZE}")
    print(f"Seq length  : {SEQ_LEN}")
    print(f"Train chars : {len(train_data):,}  |  Val chars: {len(val_data):,}")
    print()

    model       = VanillaRNN(vocab_size, HIDDEN_SIZE)
    h           = np.zeros((HIDDEN_SIZE, 1))
    smooth_loss = -np.log(1.0 / vocab_size) * SEQ_LEN
    best_loss   = float("inf")

    p  = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        if p + SEQ_LEN + 1 >= len(train_data):
            p = 0
            h = np.zeros((HIDDEN_SIZE, 1))

        inputs  = [to_onehot(train_data[p + i],     vocab_size) for i in range(SEQ_LEN)]
        targets = [train_data[p + i + 1] for i in range(SEQ_LEN)]

        xs, hs, ys, ps = model.forward(inputs, h)
        loss, grads    = model.backward(xs, hs, ps, targets)
        model.update(grads, LEARNING_RATE)

        h = hs[SEQ_LEN - 1]
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss
        p += SEQ_LEN

        if smooth_loss < best_loss:
            best_loss = smooth_loss
            os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
            model.save(CHECKPOINT)

        if epoch % LOG_EVERY == 0:
            _, val_ppl = compute_val_ppl(model, val_data, vocab_size, SEQ_LEN)
            sample = sample_text(model, "A", char2idx, idx2char, h)
            print(f"Epoch {epoch:5d} | train_loss: {smooth_loss:.2f} | val_ppl: {val_ppl:.2f} | sample: {sample}")

    elapsed = time.time() - t0
    print(f"\nCheckpoint saved → {CHECKPOINT}.npz")
    print(f"Training time  : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

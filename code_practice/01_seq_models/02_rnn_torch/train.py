"""
Session 2 — RNN with PyTorch
train.py: Batched char-level LM training on Acme corpus. GPU-ready.
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from code_practice.shared_dataset import get_corpus_as_string, get_char_vocab
from model import CharRNN

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMBED_DIM   = 64
HIDDEN_SIZE = 128
SEQ_LEN     = 25
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-3
LOG_EVERY   = 5
CHECKPOINT  = os.path.join(_DIR, "checkpoints", "rnn_torch.pt")
# ──────────────────────────────────────────────────────────────────────────────


def build_batches(data, seq_len, batch_size):
    """Slice corpus into (input, target) pairs, pack into batches."""
    # Trim to exact multiple of (batch_size * seq_len) so view() works
    chunk = batch_size * seq_len
    n     = (len(data) - 1) // chunk
    data  = data[: n * chunk + 1]
    x = torch.tensor(data[:-1], dtype=torch.long).view(batch_size, -1)
    y = torch.tensor(data[1:],  dtype=torch.long).view(batch_size, -1)
    return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    text = get_corpus_as_string()
    char2idx, idx2char, vocab_size = get_char_vocab()
    data = [char2idx[c] for c in text]

    print(f"Vocab size  : {vocab_size}")
    print(f"Embed dim   : {EMBED_DIM}")
    print(f"Hidden size : {HIDDEN_SIZE}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Seq length  : {SEQ_LEN}")
    print()

    model = CharRNN(vocab_size, EMBED_DIM, HIDDEN_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Build a simple batch: repeat corpus to have enough tokens for BATCH_SIZE
    repeated = data * (BATCH_SIZE * SEQ_LEN * EPOCHS // len(data) + 2)
    xs, ys = build_batches(repeated, SEQ_LEN, BATCH_SIZE)

    n_chunks = xs.size(1) // SEQ_LEN
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        h = model.init_hidden(BATCH_SIZE, device)

        for chunk in range(n_chunks):
            x_chunk = xs[:, chunk * SEQ_LEN:(chunk + 1) * SEQ_LEN].T.to(device)  # [T, B]
            y_chunk = ys[:, chunk * SEQ_LEN:(chunk + 1) * SEQ_LEN].T.to(device)  # [T, B]

            h = h.detach()
            optimizer.zero_grad()
            logits, h = model(x_chunk, h)           # [T, B, V]
            loss = criterion(logits.reshape(-1, vocab_size), y_chunk.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_chunks
        ppl = math.exp(avg_loss)

        if epoch % LOG_EVERY == 0:
            print(f"Epoch {epoch:3d} | loss: {avg_loss:.2f} | ppl: {ppl:6.2f}")

    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab_size":  vocab_size,
        "embed_dim":   EMBED_DIM,
        "hidden_size": HIDDEN_SIZE,
        "char2idx":    char2idx,
        "idx2char":    idx2char,
    }, CHECKPOINT)
    print(f"\nCheckpoint saved → {CHECKPOINT}")
    print(f"Training time  : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

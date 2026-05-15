"""
Session 4 — LSTM with PyTorch (nn.LSTM)
train.py: char-level LM, stateful across batches, compare ppl vs Session 3.
"""

import sys
import os
import math
import time
import torch
import torch.nn as nn

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from code_practice.shared_dataset import get_corpus_as_string, get_char_vocab
from model import CharLSTM

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN     = 50
BATCH_SIZE  = 32
EMBED_DIM   = 64
HIDDEN_SIZE = 128
NUM_LAYERS  = 1
EPOCHS      = 100
LR          = 1e-3
SEED        = 42
CHECKPOINT  = os.path.join(_DIR, "checkpoints", "lstm_torch.pt")
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)


def batchify(data, batch_size):
    """Reshape data into [batch_size, -1]."""
    n = (len(data) // batch_size) * batch_size
    data = data[:n]
    return data.view(batch_size, -1)


def get_batch(data, i, seq_len):
    """Return (inputs, targets) each [batch, seq_len]."""
    seq_len = min(seq_len, data.size(1) - 1 - i)
    x = data[:, i     : i + seq_len]
    y = data[:, i + 1 : i + seq_len + 1]
    return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    text = get_corpus_as_string()
    char2idx, idx2char, vocab_size = get_char_vocab()
    data_ids   = torch.tensor([char2idx[c] for c in text], dtype=torch.long)

    print(f"Corpus length : {len(text):,} chars")
    print(f"Vocab size    : {vocab_size}")

    train_data = batchify(data_ids, BATCH_SIZE).to(device)

    model = CharLSTM(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters    : {total_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        hidden = model.init_hidden(BATCH_SIZE, device)

        for i in range(0, train_data.size(1) - 1, SEQ_LEN):
            x, y = get_batch(train_data, i, SEQ_LEN)
            if x.size(1) == 0:
                break

            # Detach hidden state to prevent gradient accumulation across batches
            h, c = hidden
            hidden = (h.detach(), c.detach())

            optimizer.zero_grad()
            logits, hidden = model(x, hidden)      # [B, T, V]
            loss = criterion(logits.view(-1, vocab_size), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        ppl      = math.exp(min(avg_loss, 20))

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(CHECKPOINT,
                       vocab_size=vocab_size, embed_dim=EMBED_DIM,
                       char2idx=char2idx, idx2char=idx2char)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:>4} | loss {avg_loss:.4f} | ppl {ppl:7.2f}")

    best_ppl = math.exp(min(best_loss, 20))
    print(f"\nBest loss : {best_loss:.4f} | Best PPL : {best_ppl:.2f}")
    print(f"Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()

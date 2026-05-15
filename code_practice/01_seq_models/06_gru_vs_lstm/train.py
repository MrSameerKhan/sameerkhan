"""
Session 6 — GRU vs LSTM in PyTorch
train.py: head-to-head comparison — same data, same optimizer, only model differs.
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
from model import CharModel

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQ_LEN     = 50
BATCH_SIZE  = 32
EMBED_DIM   = 64
HIDDEN_SIZE = 128
NUM_LAYERS  = 1
EPOCHS      = 100
LR          = 1e-3
SEED        = 42
CHECKPOINTS = {
    "gru":  os.path.join(_DIR, "checkpoints", "gru_torch.pt"),
    "lstm": os.path.join(_DIR, "checkpoints", "lstm_torch.pt"),
}
# ──────────────────────────────────────────────────────────────────────────────


def batchify(data, batch_size):
    n = (len(data) // batch_size) * batch_size
    return data[:n].view(batch_size, -1)


def get_batch(data, i, seq_len):
    seq_len = min(seq_len, data.size(1) - 1 - i)
    return data[:, i: i + seq_len], data[:, i + 1: i + seq_len + 1]


def train_model(model_type, train_data, vocab_size, char2idx, idx2char, device):
    torch.manual_seed(SEED)
    model = CharModel(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS,
                      model_type=model_type).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    t0 = time.time()
    loss_curve = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0
        hidden = model.init_hidden(BATCH_SIZE, device)

        for i in range(0, train_data.size(1) - 1, SEQ_LEN):
            x, y = get_batch(train_data, i, SEQ_LEN)
            if x.size(1) == 0:
                break

            hidden = model.detach_hidden(hidden)
            optimizer.zero_grad()
            logits, hidden = model(x, hidden)
            loss = criterion(logits.view(-1, vocab_size), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        ppl      = math.exp(min(avg_loss, 20))
        loss_curve.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save(CHECKPOINTS[model_type],
                       vocab_size=vocab_size, embed_dim=EMBED_DIM,
                       char2idx=char2idx, idx2char=idx2char)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:>4} | loss {avg_loss:.4f} | ppl {ppl:7.2f}")

    elapsed  = time.time() - t0
    best_ppl = math.exp(min(best_loss, 20))
    return total_params, best_loss, best_ppl, elapsed, loss_curve


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    text = get_corpus_as_string()
    char2idx, idx2char, vocab_size = get_char_vocab()
    data_ids   = torch.tensor([char2idx[c] for c in text], dtype=torch.long)
    train_data = batchify(data_ids, BATCH_SIZE).to(device)

    print(f"Corpus: {len(text):,} chars | Vocab: {vocab_size}\n")

    os.makedirs(os.path.dirname(CHECKPOINTS["gru"]), exist_ok=True)

    results = {}
    for model_type in ("gru", "lstm"):
        print(f"Training {model_type.upper()} ...")
        params, best_loss, best_ppl, elapsed, curve = train_model(
            model_type, train_data, vocab_size, char2idx, idx2char, device)
        results[model_type] = dict(params=params, best_ppl=best_ppl,
                                   best_loss=best_loss, elapsed=elapsed)
        print()

    print("=" * 50)
    print(f"{'Model':<6} | {'Params':>8} | {'Best PPL':>9} | {'Time(s)':>8}")
    print("-" * 50)
    for mtype, r in results.items():
        print(f"{mtype.upper():<6} | {r['params']:>8,} | {r['best_ppl']:>9.2f} | {r['elapsed']:>8.1f}")
    print("=" * 50)

    for mtype, r in results.items():
        print(f"\n{mtype.upper()} checkpoint → {CHECKPOINTS[mtype]}")


if __name__ == "__main__":
    main()

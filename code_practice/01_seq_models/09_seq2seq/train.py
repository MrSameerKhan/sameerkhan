"""
Session 9 — Seq2Seq with Bahdanau Attention
train.py: Teacher forcing loop, best model saved by val loss.
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

from data  import get_loaders, PAD_IDX
from model import Seq2Seq

# ── Hyperparameters ────────────────────────────────────────────────────────────
BATCH_SIZE   = 16
EMBED_DIM    = 64
ENC_HIDDEN   = 128
DEC_HIDDEN   = 256
ATTN_DIM     = 128
EPOCHS       = 50
LR           = 1e-3
TF_RATIO     = 0.5          # teacher forcing ratio
SEED         = 42
CHECKPOINT   = os.path.join(_DIR, "checkpoints", "seq2seq.pt")
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)


def run_epoch(model, loader, criterion, optimizer, device,
              teacher_forcing_ratio, is_train):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt, src_mask, tgt_lengths in loader:
            src, tgt, src_mask = (src.to(device), tgt.to(device), src_mask.to(device))

            tf = teacher_forcing_ratio if is_train else 0.0
            logits, _ = model(src, tgt, src_mask, tf)

            # logits: [B, T-1, vocab]   tgt[:, 1:] → skip SOS
            B, T, V = logits.size()
            loss = criterion(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss   += loss.item() * B * T
            total_tokens += B * T

    return total_loss / max(total_tokens, 1)


def main():
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    (train_loader, val_loader, test_loader), word2idx, idx2word, \
        train_data, val_data, test_data = get_loaders(batch_size=BATCH_SIZE, seed=SEED)

    vocab_size = len(word2idx)
    print(f"Vocab size : {vocab_size}")
    print(f"Train pairs: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}\n")

    model = Seq2Seq(vocab_size, EMBED_DIM, ENC_HIDDEN, DEC_HIDDEN, ATTN_DIM).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    os.makedirs(os.path.dirname(CHECKPOINT), exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer,
                               device, TF_RATIO, is_train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer,
                               device, 0.0, is_train=False)
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(CHECKPOINT,
                       vocab_size=vocab_size, embed_dim=EMBED_DIM,
                       enc_hidden=ENC_HIDDEN, dec_hidden=DEC_HIDDEN,
                       attn_dim=ATTN_DIM,
                       word2idx=word2idx, idx2word=idx2word)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:>3} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:.1f}s")

    print(f"\nBest val loss : {best_val_loss:.4f}")
    print(f"Checkpoint    : {CHECKPOINT}")

    # Quick test sample
    test_loss = run_epoch(model, test_loader, criterion, optimizer,
                          device, 0.0, is_train=False)
    print(f"Test loss     : {test_loss:.4f}")


if __name__ == "__main__":
    main()

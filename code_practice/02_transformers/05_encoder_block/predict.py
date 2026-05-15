"""
Phase 2, Session 5 — Transformer Encoder Block
predict.py: Inference + per-head attention specialization visualization.
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import EncoderClassifier
from data  import encode, tokenize, IDX2LABEL

CKPT = os.path.join(_DIR, "checkpoints", "encoder.pt")


def load_model(device):
    ckpt  = torch.load(CKPT, map_location=device)
    model = EncoderClassifier(
        vocab_size=len(ckpt["word2idx"]),
        d_model=ckpt["d_model"],
        num_heads=ckpt["num_heads"],
        d_ff=ckpt["d_ff"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["word2idx"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works as Senior Analyst in Risk department.")
    args = parser.parse_args()

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}\nRun train.py first.")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, word2idx = load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    num_heads = model.block.mha.num_heads

    tokens = tokenize(args.text)
    ids    = encode(args.text, word2idx)
    t      = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, weights = model(t)   # weights: [1, H, T, T]

    probs    = F.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    BAR = 20

    print(f"\nModel: {n_params:,} params  |  {num_heads} heads")
    print(f"Input: {args.text!r}")
    print(f"\nPredicted: {IDX2LABEL[pred_idx]}  ({probs[pred_idx]*100:.1f}%)")
    print(f"\nAll class probabilities:")
    for i, cls in IDX2LABEL.items():
        p      = probs[i].item()
        filled = int(round(p * BAR))
        print(f"  {cls:<12} {p:.3f}  {'█'*filled + '·'*(BAR-filled)}")

    # Per-head: which token each head attends to most (column mean of A)
    W = weights[0].cpu()  # [H, T, T]
    T = len(tokens)
    W = W[:, :T, :T]      # trim padding

    print(f"\n{'─'*62}")
    print(f"Per-head top attended token  (column mean = received attention)")
    print(f"{'─'*62}")
    header = f"{'Token':<16}" + "".join(f"  H{h}" for h in range(num_heads))
    print(header)

    col_means = W.mean(dim=1)   # [H, T]  — mean over query dim

    for j, tok in enumerate(tokens):
        row = f"  {tok:<14}"
        for h in range(num_heads):
            row += f"  {col_means[h, j]:.2f}"
        print(row)

    print(f"\n{'─'*62}")
    print(f"Top attended token per head:")
    print(f"{'─'*62}")
    for h in range(num_heads):
        top_j  = col_means[h].argmax().item()
        top_v  = col_means[h, top_j].item()
        print(f"  Head {h}: '{tokens[top_j]}'  ({top_v*100:.0f}%)")

    # Head entropy
    eps = 1e-9
    print(f"\n{'─'*62}")
    print(f"Head entropy  (norm 1.0 = uniform, lower = focused)")
    print(f"{'─'*62}")
    max_ent = torch.log(torch.tensor(T, dtype=torch.float))
    for h in range(num_heads):
        row_ents = -(W[h] * (W[h] + eps).log()).sum(dim=-1)  # [T]
        mean_ent = row_ents[:T].mean().item()
        norm     = mean_ent / max_ent.item()
        filled   = int(round(norm * BAR))
        print(f"  Head {h}: entropy {mean_ent:.3f}  (norm {norm:.2f})  "
              f"{'█'*filled + '·'*(BAR-filled)}")


if __name__ == "__main__":
    main()

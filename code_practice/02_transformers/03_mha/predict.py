"""
Session 11 — Multi-Head Attention (NumPy from scratch)
predict.py: Classify sentence + visualize all 4 heads independently.

Key addition vs Session 10: per-head received attention bars show
whether heads specialize in different token relationships.
"""

import sys
import os
import argparse
import numpy as np

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from data  import encode, IDX2LABEL
from model import MultiHeadAttentionClassifier

CKPT    = os.path.join(_DIR, "checkpoints", "multihead_attn.pkl")
BAR     = "█"
DOT     = "·"
BAR_LEN = 30


def render_bar(w, max_len=BAR_LEN):
    filled = int(round(w * max_len))
    return BAR * filled + DOT * (max_len - filled)


def main():
    parser = argparse.ArgumentParser(description="Session 11 — Multi-Head Attention inference")
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works as Head of Loans in the Loans department.",
                        help="Input sentence to classify")
    args = parser.parse_args()

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}\nRun train.py first.")
        sys.exit(1)

    model, ckpt = MultiHeadAttentionClassifier.load(CKPT)
    word2idx    = ckpt["word2idx"]
    idx2label   = ckpt["idx2label"]
    n_heads     = model.n_heads

    ids, tokens = encode(args.text, word2idx)
    mask        = np.ones(len(ids), dtype=int)
    probs, cache = model.forward(ids, mask)

    pred_idx   = probs.argmax()
    pred_label = idx2label[pred_idx]
    confidence = probs[pred_idx]
    real_rows  = mask.astype(bool)

    print(f"\nInput    : {args.text}")
    print(f"Predicted: {pred_label}  ({confidence * 100:.1f}%)")

    print(f"\nAll class probabilities:")
    for idx in range(len(probs)):
        bar = render_bar(probs[idx], 20)
        print(f"  {idx2label[idx]:<12} {probs[idx]:.3f}  {bar}")

    # --- Per-head received attention ---
    print(f"\n{'─' * 70}")
    print(f"Per-head received attention  (column mean of each head's A matrix)")
    print(f"Shows whether heads specialize in different token relationships")
    print(f"{'─' * 70}\n")

    head_recvs = []
    for k in range(n_heads):
        A_k      = cache["head_As"][k]                    # [T, T]
        recv_k   = A_k[real_rows, :].mean(axis=0)         # [T]
        recv_k  /= recv_k.sum() + 1e-8
        head_recvs.append(recv_k)

    # Header: token names across top
    tok_header = f"{'Token':<16}" + "".join(f"  H{k+1}" for k in range(n_heads))
    print(tok_header)
    print("─" * (16 + n_heads * 5))

    for i, tok in enumerate(tokens):
        weights = "".join(f"  {head_recvs[k][i]:.2f}" for k in range(n_heads))
        print(f"  {tok:<14}{weights}")

    # --- Overall received attention (mean across heads) ---
    overall = np.stack(head_recvs).mean(axis=0)
    overall /= overall.sum() + 1e-8

    print(f"\n{'─' * 70}")
    print(f"Overall received attention (averaged across all {n_heads} heads)\n")
    for tok, w in zip(tokens, overall):
        print(f"  {tok:<18} {w:.3f}  {render_bar(w)}")

    # --- Head entropy (how focused vs spread each head is) ---
    print(f"\n{'─' * 70}")
    print(f"Head entropy  (low = focused on few tokens, high = spread evenly)\n")
    for k in range(n_heads):
        A_k      = cache["head_As"][k]
        real_A   = A_k[real_rows, :][:, real_rows]        # real×real submatrix
        row_ents = -np.sum(real_A * np.log(real_A + 1e-8), axis=-1)
        mean_ent = row_ents.mean()
        max_ent  = np.log(real_rows.sum())                 # entropy of uniform dist
        norm_ent = mean_ent / (max_ent + 1e-8)
        bar      = render_bar(norm_ent, 20)
        print(f"  Head {k+1}: entropy {mean_ent:.3f}  (norm {norm_ent:.2f})  {bar}")


if __name__ == "__main__":
    main()

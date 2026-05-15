"""
Session 10 — Scaled Dot-Product Self-Attention (NumPy from scratch)
predict.py: Classify sentence + visualize the T×T attention weight matrix.

Visualization: shows "received attention" per token — how much each position
was attended to on average across all query positions. (Column mean of A.)
This is the self-attention analog of Bahdanau's 1D attention bar chart.
"""

import sys
import os
import argparse
import numpy as np

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from data  import encode, LABEL2IDX, IDX2LABEL
from model import SelfAttentionClassifier

CKPT = os.path.join(_DIR, "checkpoints", "self_attn.pkl")
BAR  = "█"
DOT  = "·"


def render_bar(weight, max_len=40):
    filled = int(round(weight * max_len))
    empty  = max_len - filled
    return BAR * filled + DOT * empty


def main():
    parser = argparse.ArgumentParser(description="Session 10 — Self-Attention inference")
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works as Head of Loans in the Loans department.",
                        help="Input sentence to classify")
    args = parser.parse_args()

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}\nRun train.py first.")
        sys.exit(1)

    model, ckpt = SelfAttentionClassifier.load(CKPT)
    word2idx    = ckpt["word2idx"]
    idx2label   = ckpt["idx2label"]

    ids, tokens = encode(args.text, word2idx)
    mask        = np.ones(len(ids), dtype=int)
    probs, cache = model.forward(ids, mask)

    pred_idx   = probs.argmax()
    pred_label = idx2label[pred_idx]
    confidence = probs[pred_idx]

    A = cache["A"]  # [T, T]

    # Received attention: how much each token was attended to on average
    # A[i, j] = how much query-i attends to key-j
    # column mean over real query rows → importance of each token as a key
    real_rows   = mask.astype(bool)
    recv_attn   = A[real_rows, :].mean(axis=0)   # [T] averaged over real queries
    recv_attn  /= recv_attn.sum() + 1e-8          # re-normalize for display

    print(f"\nInput    : {args.text}")
    print(f"Predicted: {pred_label}  ({confidence * 100:.1f}%)")
    print(f"\nAll class probabilities:")
    for idx in range(len(probs)):
        label = idx2label[idx]
        bar   = render_bar(probs[idx], max_len=20)
        print(f"  {label:<12} {probs[idx]:.3f}  {bar}")

    print(f"\nSelf-attention: received attention per token")
    print(f"(how much each token was attended to on average — column mean of A)\n")
    for tok, w in zip(tokens, recv_attn):
        bar = render_bar(w)
        print(f"  {tok:<18} {w:.3f}  {bar}")

    print(f"\nAttention matrix A [{len(tokens)}×{len(tokens)}]  (query → key heatmap):")
    header = "         " + "".join(f"{t[:5]:>6}" for t in tokens)
    print(header)
    for i, tok_i in enumerate(tokens):
        row = f"  {tok_i[:7]:<8}" + "".join(f"  {A[i, j]:.2f}" for j in range(len(tokens)))
        print(row)


if __name__ == "__main__":
    main()

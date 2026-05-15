"""
Session 5 — GRU Cell from Scratch
predict.py: load checkpoint, generate text.
"""

import sys
import os
import argparse

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

import numpy as np
from code_practice.shared_dataset import get_char_vocab
from model import VanillaGRU

CHECKPOINT = os.path.join(_DIR, "checkpoints", "gru_scratch.npz")


def main():
    parser = argparse.ArgumentParser(description="Session 5 — GRU char-level generation")
    parser.add_argument("--seed",   type=str,   default="Sarah Chen",
                        help="Seed string to prime the model")
    parser.add_argument("--length", type=int,   default=300,
                        help="Number of chars to generate")
    parser.add_argument("--temp",   type=float, default=0.8,
                        help="Sampling temperature")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}\nRun train.py first.")
        sys.exit(1)

    char2idx, idx2char, _ = get_char_vocab()

    model = VanillaGRU.load(CHECKPOINT)
    print(f"Loaded GRU — hidden_size={model.hidden_size}, vocab={model.vocab_size}")

    # Prime model on seed string
    H = model.hidden_size
    h = np.zeros((H, 1))

    def gru_step(idx, h):
        x  = np.zeros((model.vocab_size, 1))
        x[idx] = 1.0
        z  = np.vstack((h, x))
        r  = 1.0 / (1.0 + np.exp(-model.Wr @ z - model.br))
        u  = 1.0 / (1.0 + np.exp(-model.Wu @ z - model.bu))
        zt = np.vstack((r * h, x))
        hc = np.tanh(model.Wh @ zt + model.bh)
        return (1.0 - u) * h + u * hc

    for ch in args.seed[:-1]:
        idx = char2idx.get(ch, char2idx.get(" ", 0))
        h   = gru_step(idx, h)

    seed_idx = char2idx.get(args.seed[-1], char2idx.get(" ", 0))
    result   = model.sample(seed_idx, length=args.length, temperature=args.temp)
    generated = args.seed[:-1] + "".join(idx2char[i] for i in result)

    print(f"\nSeed   : {args.seed!r}")
    print(f"Length : {args.length}")
    print(f"Temp   : {args.temp}")
    print(f"\n{'─'*60}")
    print(generated)
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()

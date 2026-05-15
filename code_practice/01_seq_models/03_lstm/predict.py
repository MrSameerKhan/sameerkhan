"""
Session 3 — LSTM Cell from Scratch
predict.py: load checkpoint, generate text with --seed, --length, --temp flags.
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
from model import VanillaLSTM

CHECKPOINT = os.path.join(_DIR, "checkpoints", "lstm_scratch.npz")


def main():
    parser = argparse.ArgumentParser(description="Session 3 — LSTM char-level generation")
    parser.add_argument("--seed",   type=str,   default="The",  help="Seed string to prime the model")
    parser.add_argument("--length", type=int,   default=300,    help="Number of chars to generate")
    parser.add_argument("--temp",   type=float, default=0.8,    help="Sampling temperature (0.5=conservative, 1.5=creative)")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}\nRun train.py first.")
        sys.exit(1)

    char2idx, idx2char, _ = get_char_vocab()

    model = VanillaLSTM.load(CHECKPOINT)
    print(f"Loaded LSTM — hidden_size={model.hidden_size}, vocab={model.vocab_size}")

    # Prime the model on the seed string
    H = model.hidden_size
    h = np.zeros((H, 1))
    c = np.zeros((H, 1))

    for ch in args.seed[:-1]:
        idx = char2idx.get(ch, char2idx.get(" ", 0))
        x   = np.zeros((model.vocab_size, 1))
        x[idx] = 1.0
        z  = np.vstack((h, x))
        f  = 1.0 / (1.0 + np.exp(-model.Wf @ z - model.bf))
        i_ = 1.0 / (1.0 + np.exp(-model.Wi @ z - model.bi))
        o  = 1.0 / (1.0 + np.exp(-model.Wo @ z - model.bo))
        g  = np.tanh(model.Wg @ z + model.bg)
        c  = f * c + i_ * g
        h  = o * np.tanh(c)

    seed_idx = char2idx.get(args.seed[-1], char2idx.get(" ", 0))

    # Temporarily override hidden/cell in sample() call by patching model
    # (sample() always starts from zeros — prime manually above, then continue)
    result = model.sample(seed_idx, length=args.length, temperature=args.temp)
    generated = args.seed[:-1] + "".join(idx2char[i] for i in result)

    print(f"\nSeed   : {args.seed!r}")
    print(f"Length : {args.length}")
    print(f"Temp   : {args.temp}")
    print(f"\n{'─'*60}")
    print(generated)
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()

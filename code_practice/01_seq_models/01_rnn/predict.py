"""
Session 1 — Vanilla RNN Cell from Scratch
predict.py: Load trained checkpoint, generate text character by character.
"""

import sys
import os
import argparse
import numpy as np

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from code_practice.shared_dataset import get_char_vocab
from model import VanillaRNN

CHECKPOINT  = os.path.join(_DIR, "checkpoints", "rnn_scratch.npz")
HIDDEN_SIZE = 128


def main():
    parser = argparse.ArgumentParser(description="Session 1 — RNN text generation")
    parser.add_argument("--seed",   type=str,   default="Acme", help="Seed string (first char used)")
    parser.add_argument("--length", type=int,   default=100,    help="Chars to generate")
    parser.add_argument("--temp",   type=float, default=1.0,    help="Sampling temperature")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print("Run train.py first.")
        sys.exit(1)

    char2idx, idx2char, vocab_size = get_char_vocab()
    model = VanillaRNN(vocab_size, HIDDEN_SIZE)
    model.load(CHECKPOINT)

    h = np.zeros((HIDDEN_SIZE, 1))
    seed_char = args.seed[0] if args.seed else "A"
    seed_idx  = char2idx.get(seed_char, 0)

    idxs      = model.sample(seed_idx, h, args.length, temperature=args.temp)
    generated = args.seed + "".join(idx2char[i] for i in idxs)

    print(f"Seed      : \"{args.seed}\"")
    print(f"Generated : {generated}")


if __name__ == "__main__":
    main()

"""
Phase 2, Session 1 — BPE Tokenizer from Scratch
predict.py: Tokenize any text with the trained BPE. Shows per-word breakdown.
"""

import sys
import os
import argparse
import re

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import BPE, EOW

CKPT = os.path.join(_DIR, "checkpoints", "bpe.json")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Session 1 — BPE encode")
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works as Senior Analyst in Risk department.",
                        help="Text to tokenize")
    parser.add_argument("--show-merges", action="store_true",
                        help="Show which merges were applied per word")
    args = parser.parse_args()

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}\nRun train.py first.")
        sys.exit(1)

    bpe = BPE.load(CKPT)
    print(f"\nVocab size:  {len(bpe.vocab)}")
    print(f"Merge rules: {len(bpe.merges)}")

    text    = args.text
    tokens  = bpe.tokenize(text)
    ids     = bpe.encode(text)
    decoded = bpe.decode(ids)

    print(f"\nInput:")
    print(f"  {text!r}")
    print(f"  ({len(text)} chars)")

    print(f"\nBPE tokens ({len(ids)}):")
    print(f"  {tokens}")
    print(f"\nIDs:     {ids}")
    print(f"Decoded: {decoded!r}")
    print(f"Compression: {len(ids)} tokens for {len(text)} chars "
          f"({len(text)/max(len(ids),1):.1f} chars/token)")

    # Per-word breakdown
    words = re.findall(r"\w+|[^\w\s]", text)
    print(f"\nPer-word breakdown:")
    for word in words:
        pieces = bpe._encode_word(word + EOW)
        oov    = [p for p in pieces if p not in bpe.vocab]
        flag   = "  ← OOV!" if oov else ""
        print(f"  {word:<20} → {pieces}{flag}")


if __name__ == "__main__":
    main()

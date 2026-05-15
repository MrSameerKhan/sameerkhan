"""
Phase 2, Session 1 — BPE Tokenizer from Scratch
train.py: Train BPE on Acme corpus, run sanity checks, save tokenizer.
"""

import sys
import os
import argparse

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import BPE
from code_practice.shared_dataset import get_corpus

CKPT = os.path.join(_DIR, "checkpoints", "bpe.json")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Session 1 — BPE training")
    parser.add_argument("--vocab-size", type=int, default=300,
                        help="Target vocabulary size (default: 300)")
    args = parser.parse_args()

    corpus = get_corpus(target_chars=50_000)
    print(f"Loaded corpus: {len(corpus):,} chars\n")

    bpe = BPE()
    bpe.train(corpus, vocab_size=args.vocab_size, verbose=True)

    bpe.save(CKPT)
    print(f"\nSaved tokenizer: {CKPT}")

    # --- Sanity check ---
    test_sentence = "The Premium Checking has 2.5 percent rate."
    tokens  = bpe.tokenize(test_sentence)
    ids     = bpe.encode(test_sentence)
    decoded = bpe.decode(ids)

    print(f"\n== Sanity check ==")
    print(f"Input:   {test_sentence!r}")
    print(f"Tokens:  {tokens}")
    print(f"IDs:     {ids}")
    print(f"Decoded: {decoded!r}")
    n_chars = len(test_sentence.replace(" ", ""))
    print(f"Compression: ~{len(ids)} tokens for {len(test_sentence)} chars "
          f"({len(test_sentence)/max(len(ids),1):.1f} chars/token)")


if __name__ == "__main__":
    main()

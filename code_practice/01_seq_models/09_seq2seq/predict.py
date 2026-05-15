"""
Session 9 — Seq2Seq with Bahdanau Attention
predict.py: greedy decode with --question CLI arg. Shows generated answer.
"""

import sys
import os
import argparse
import torch

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from data  import tokenize, PAD_IDX, EOS_IDX
from model import Seq2Seq

CHECKPOINT = os.path.join(_DIR, "checkpoints", "seq2seq.pt")


def main():
    parser = argparse.ArgumentParser(description="Session 9 — Seq2Seq inference")
    parser.add_argument("--question", type=str,
                        default="Who is Sarah Chen?",
                        help="Input question/instruction to answer")
    parser.add_argument("--max_len", type=int, default=50,
                        help="Maximum output length")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}\nRun train.py first.")
        sys.exit(1)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model, ckpt = Seq2Seq.load(CHECKPOINT, device=str(device))
    model.to(device)

    word2idx = ckpt["word2idx"]
    idx2word = ckpt["idx2word"]
    unk_idx  = word2idx.get("<UNK>", 1)

    # Encode input question
    tokens   = tokenize(args.question)
    src_ids  = [word2idx.get(t, unk_idx) for t in tokens] + [word2idx["<EOS>"]]
    src      = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = torch.ones(1, len(src_ids), dtype=torch.long, device=device)

    result_ids = model.greedy_decode(src, src_mask, max_len=args.max_len)
    answer     = " ".join(idx2word.get(i, "<UNK>") for i in result_ids)

    print(f"\nQuestion : {args.question}")
    print(f"Answer   : {answer}")

    oov = [t for t in tokens if t not in word2idx]
    if oov:
        print(f"\n(OOV tokens: {oov} — mapped to <UNK>)")


if __name__ == "__main__":
    main()

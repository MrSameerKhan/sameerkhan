"""
Session 2 — RNN with PyTorch
predict.py: Load checkpoint and generate text.
"""

import sys
import os
import argparse
import torch

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import CharRNN

CHECKPOINT = os.path.join(_DIR, "checkpoints", "rnn_torch.pt")


@torch.no_grad()
def generate(model, seed_idx, char2idx, idx2char, length, temperature=1.0, device="cpu"):
    model.eval()
    h = model.init_hidden(1, device)
    x = torch.tensor([[seed_idx]], device=device)   # [1, 1] → seq-first = [1, 1]
    generated = []

    for _ in range(length):
        logits, h = model(x, h)                # logits [1, 1, V]
        logits = logits.squeeze() / temperature
        probs  = torch.softmax(logits, dim=-1)
        idx    = torch.multinomial(probs, 1).item()
        generated.append(idx)
        x = torch.tensor([[idx]], device=device)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Session 2 — RNN text generation")
    parser.add_argument("--seed",   type=str,   default="Acme")
    parser.add_argument("--length", type=int,   default=100)
    parser.add_argument("--temp",   type=float, default=1.0)
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}\nRun train.py first.")
        sys.exit(1)

    ckpt = torch.load(CHECKPOINT, weights_only=False)
    char2idx   = ckpt["char2idx"]
    idx2char   = ckpt["idx2char"]
    vocab_size = ckpt["vocab_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = CharRNN(vocab_size, ckpt["embed_dim"], ckpt["hidden_size"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    seed_char = args.seed[0] if args.seed else "A"
    seed_idx  = char2idx.get(seed_char, 0)
    idxs      = generate(model, seed_idx, char2idx, idx2char, args.length, args.temp, device)
    generated = args.seed + "".join(idx2char[i] for i in idxs)

    print(f"Seed      : \"{args.seed}\"")
    print(f"Generated : {generated}")


if __name__ == "__main__":
    main()

"""
Session 6 — GRU vs LSTM in PyTorch
predict.py: generate from either model checkpoint via --model flag.
"""

import sys
import os
import argparse
import torch

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import CharModel

CHECKPOINTS = {
    "gru":  os.path.join(_DIR, "checkpoints", "gru_torch.pt"),
    "lstm": os.path.join(_DIR, "checkpoints", "lstm_torch.pt"),
}


@torch.no_grad()
def generate(model, char2idx, idx2char, seed_str, length, temperature, device):
    model.eval()
    vocab_size = len(char2idx)
    unk_idx    = char2idx.get(" ", 0)
    hidden     = model.init_hidden(1, device)

    for ch in seed_str[:-1]:
        idx = char2idx.get(ch, unk_idx)
        x   = torch.tensor([[idx]], dtype=torch.long, device=device)
        _, hidden = model(x, hidden)
        hidden = model.detach_hidden(hidden)

    result = list(seed_str)
    idx    = char2idx.get(seed_str[-1], unk_idx)

    for _ in range(length):
        x = torch.tensor([[idx]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        hidden = model.detach_hidden(hidden)
        logits = logits[0, 0] / temperature
        probs  = torch.softmax(logits, dim=-1)
        idx    = torch.multinomial(probs, 1).item()
        result.append(idx2char[idx])

    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description="Session 6 — GRU/LSTM generation")
    parser.add_argument("--model",  type=str,   default="gru",
                        choices=["gru", "lstm"], help="Which model to use")
    parser.add_argument("--seed",   type=str,   default="The Premium",
                        help="Seed string")
    parser.add_argument("--length", type=int,   default=300)
    parser.add_argument("--temp",   type=float, default=0.8)
    args = parser.parse_args()

    checkpoint = CHECKPOINTS[args.model]
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}\nRun train.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = CharModel.load(checkpoint, device=str(device))
    model.to(device)

    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]

    print(f"Model : {args.model.upper()} | hidden={model.hidden_size} | vocab={len(char2idx)}")
    print(f"Seed  : {args.seed!r} | length={args.length} | temp={args.temp}\n")
    print("─" * 60)

    text = generate(model, char2idx, idx2char,
                    args.seed, args.length, args.temp, device)
    print(text)
    print("─" * 60)


if __name__ == "__main__":
    main()

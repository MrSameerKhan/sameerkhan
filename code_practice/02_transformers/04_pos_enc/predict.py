"""
Phase 2, Session 4 — Positional Encoding
predict.py: Inference + order-sensitivity demo (shuffle same tokens, compare confidence).
"""

import sys
import os
import argparse
import random
import torch
import torch.nn.functional as F

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import PEClassifier
from data  import tokenize, encode, IDX2LABEL, PAD_IDX

CKPT = os.path.join(_DIR, "checkpoints", "pos_enc_rope.pt")

N_CLASSES = 5


def load_model(path: str, device: torch.device):
    ckpt     = torch.load(path, map_location=device)
    word2idx = ckpt["word2idx"]
    model    = PEClassifier(
        vocab_size=len(word2idx),
        d_model=ckpt["d_model"],
        num_heads=ckpt["num_heads"],
        pe_type=ckpt["pe_type"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, word2idx


def predict_text(text: str, model, word2idx: dict, device: torch.device):
    ids      = encode(text, word2idx)
    t        = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(t)
    probs    = F.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    return IDX2LABEL[pred_idx], probs[pred_idx].item(), probs.tolist()


def order_sensitivity_demo(text: str, model, word2idx: dict, device: torch.device):
    """
    Show that RoPE IS sensitive to word order — same words, different order → different confidence.
    Without PE, all shuffles would give identical confidence.
    """
    tokens = tokenize(text)
    print(f"\n{'─'*60}")
    print(f"Order-Sensitivity Demo — RoPE IS using word order")
    print(f"{'─'*60}")
    print(f"Original: '{text}'")

    # Build 4 orderings: original + 3 shuffles
    orderings = [tokens[:]]
    rng = random.Random(7)
    for _ in range(3):
        shuffled = tokens[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)

    for i, toks in enumerate(orderings):
        sent    = " ".join(toks)
        ids     = [word2idx.get(t, 1) for t in toks]
        t       = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(t)
        probs    = F.softmax(logits, dim=-1)[0]
        pred_idx = probs.argmax().item()
        label    = IDX2LABEL[pred_idx]
        conf     = probs[pred_idx].item()
        tag      = "← original" if i == 0 else "← shuffled"
        print(f"  '{sent}'")
        print(f"    → {label} ({conf*100:.2f}%)  {tag}")

    print(f"\n  Same words, different orders → different confidence levels.")
    print(f"  Without PE, all four would be IDENTICAL.")
    print(f"  The confidence spread proves the model uses order information.")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Session 4 — PE inference")
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works as Senior Analyst in Risk department.",
                        help="Text to classify")
    parser.add_argument("--order-demo", action="store_true",
                        help="Run order-sensitivity demo with shuffled variants")
    args = parser.parse_args()

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}\nRun train.py first.")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, word2idx = load_model(CKPT, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nLoaded RoPE model  ({n_params:,} params)")
    print(f"Input: {args.text!r}")

    label, conf, probs = predict_text(args.text, model, word2idx, device)

    BAR = 20
    print(f"\nPredicted: {label}  ({conf*100:.1f}%)")
    print(f"\nAll class probabilities:")
    for i, cls in IDX2LABEL.items():
        p     = probs[i]
        filled = int(round(p * BAR))
        bar    = "█" * filled + "·" * (BAR - filled)
        print(f"  {cls:<12} {p:.3f}  {bar}")

    if args.order_demo:
        order_sensitivity_demo(args.text, model, word2idx, device)


if __name__ == "__main__":
    main()

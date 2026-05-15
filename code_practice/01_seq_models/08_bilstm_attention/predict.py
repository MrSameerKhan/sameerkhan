"""
Session 8 — Bahdanau Attention on BiLSTM
predict.py: CLI with attention weight visualization (the killer demo).
"""

import sys
import os
import argparse
import torch

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from data  import tokenize
from model import BiLSTMClassifier

CHECKPOINT = os.path.join(_DIR, "checkpoints", "bilstm_attention.pt")
BAR_WIDTH  = 40


def bar(weight: float) -> str:
    filled = int(weight * BAR_WIDTH)
    if filled == 0:
        return "·" * 3
    return "█" * filled


def main():
    parser = argparse.ArgumentParser(description="Session 8 — BiLSTM+attention inference")
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works in the Loans department.",
                        help="Input sentence to classify")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}\nRun train.py first.")
        sys.exit(1)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model, ckpt = BiLSTMClassifier.load(CHECKPOINT, device=str(device))
    model.to(device)

    word2idx  = ckpt["word2idx"]
    idx2label = ckpt["idx2label"]
    label2idx = ckpt["label2idx"]

    tokens    = tokenize(args.text)
    unk_id    = word2idx.get("<UNK>", 1)
    token_ids = [word2idx.get(t, unk_id) for t in tokens]
    mask      = torch.ones(1, len(token_ids), dtype=torch.long, device=device)
    x         = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, attn_weights = model(x, mask)

    probs      = torch.softmax(logits, dim=-1)[0]
    pred_idx   = probs.argmax().item()
    pred_label = idx2label[pred_idx]
    confidence = probs[pred_idx].item()

    print(f"\nInput:\n  {args.text}")
    print(f"\nPredicted class: {pred_label} ({100*confidence:.0f}%)")
    print("\nClass probabilities:")
    for label, idx in sorted(label2idx.items()):
        marker = " ★" if idx == pred_idx else ""
        print(f"  {label:<15} {probs[idx].item():.4f}{marker}")

    if attn_weights is not None:
        weights = attn_weights[0].tolist()   # [seq_len]
        top_k   = sorted(zip(weights, tokens), reverse=True)[:3]

        print("\nAttention weights (which tokens the model focused on):")
        for tok, w in zip(tokens, weights):
            print(f"  {tok:<20} {w:.3f}  {bar(w)}")

        print(f"\nTop-3 attended tokens:")
        for w, tok in top_k:
            print(f"  {tok:<20} {w:.3f}")


if __name__ == "__main__":
    main()

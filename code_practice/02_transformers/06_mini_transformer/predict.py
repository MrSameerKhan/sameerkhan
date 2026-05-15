"""
Phase 2, Session 6 — Mini Transformer Encoder
predict.py: Layer-wise attention visualization + per-layer entropy analysis.
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import MiniTransformerEncoder
from data  import encode, tokenize, IDX2LABEL

CKPT = os.path.join(_DIR, "checkpoints", "mini_encoder_4l.pt")


def load_model(device):
    ckpt  = torch.load(CKPT, map_location=device)
    model = MiniTransformerEncoder(
        vocab_size=len(ckpt["word2idx"]),
        d_model=ckpt["d_model"],
        num_heads=ckpt["num_heads"],
        d_ff=ckpt["d_ff"],
        num_layers=ckpt["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["word2idx"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works as Senior Analyst in Risk department.")
    args = parser.parse_args()

    if not os.path.exists(CKPT):
        print(f"Checkpoint not found: {CKPT}\nRun train.py first.")
        sys.exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, word2idx = load_model(device)
    n_params   = sum(p.numel() for p in model.parameters())
    num_layers = len(model.blocks)
    num_heads  = model.blocks[0].mha.num_heads

    tokens = tokenize(args.text)
    ids    = encode(args.text, word2idx)
    t      = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, all_weights = model(t)    # all_weights: list of [1, H, T, T]

    probs    = F.softmax(logits, dim=-1)[0]
    pred_idx = probs.argmax().item()
    BAR = 20

    print(f"\nModel: {n_params:,} params  |  {num_layers} layers  |  {num_heads} heads")
    print(f"Input: {args.text!r}")
    print(f"\nPredicted: {IDX2LABEL[pred_idx]}  ({probs[pred_idx]*100:.1f}%)")
    print(f"\nAll class probabilities:")
    for i, cls in IDX2LABEL.items():
        p      = probs[i].item()
        filled = int(round(p * BAR))
        print(f"  {cls:<12} {p:.3f}  {'█'*filled + '·'*(BAR-filled)}")

    T = len(tokens)

    # ── Per-(layer, head) peak attended key ─────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Per-(layer, head) peak attended token")
    print(f"(column mean = received attention, averaged over query positions)")
    print(f"{'─'*62}")

    header = f"{'Layer':<6}" + "".join(f"  H{h:<4}" for h in range(num_heads))
    print(header)

    for layer_idx, W in enumerate(all_weights):
        W_l = W[0].cpu()[:, :T, :T]       # [H, T, T]
        col_means = W_l.mean(dim=1)        # [H, T]
        top_tokens = [tokens[col_means[h].argmax().item()] for h in range(num_heads)]
        row = f"  L{layer_idx:<4}" + "".join(f"  {tok:<6}" for tok in top_tokens)
        print(row)

    # ── Layer entropy ─────────────────────────────────────────────────────────
    eps = 1e-9
    import math
    max_ent = math.log(T)
    print(f"\n{'─'*62}")
    print(f"Layer entropy  (avg over heads and query positions)")
    print(f"(norm 1.0 = uniform, lower = more focused)")
    print(f"{'─'*62}")
    for layer_idx, W in enumerate(all_weights):
        W_l = W[0].cpu()[:, :T, :T]       # [H, T, T]
        row_ents = -(W_l * (W_l + eps).log()).sum(dim=-1)   # [H, T]
        avg_ent  = row_ents.mean().item()
        norm     = avg_ent / max_ent
        filled   = int(round(norm * BAR))
        label    = "(more spread)" if layer_idx == 0 else ("(more focused)" if layer_idx == num_layers - 1 else "")
        print(f"  Layer {layer_idx}: entropy {avg_ent:.3f}  (norm {norm:.2f})  "
              f"{'█'*filled + '·'*(BAR-filled)}  {label}")

    # ── Per-layer per-head top token detail ──────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Top attended token + weight per (layer, head):")
    print(f"{'─'*62}")
    for layer_idx, W in enumerate(all_weights):
        W_l = W[0].cpu()[:, :T, :T]
        col_means = W_l.mean(dim=1)
        print(f"  Layer {layer_idx}:", end="")
        for h in range(num_heads):
            top_j = col_means[h].argmax().item()
            top_v = col_means[h, top_j].item()
            print(f"  H{h}='{tokens[top_j]}'({top_v*100:.0f}%)", end="")
        print()


if __name__ == "__main__":
    main()

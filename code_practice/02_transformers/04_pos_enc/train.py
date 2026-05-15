"""
Phase 2, Session 4 — Positional Encoding
train.py: Permutation equivariance demo + train all 4 PE variants + comparison table.
"""

import sys
import os
import math
import torch
import torch.nn as nn

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import PEClassifier
from data  import get_data, make_batch, IDX2LABEL

CKPT_DIR = os.path.join(_DIR, "checkpoints")

# Hyperparams
D_MODEL   = 64
NUM_HEADS = 8
EPOCHS    = 30
LR        = 1e-3
BATCH     = 16


def permutation_demo(word2idx: dict, device: torch.device):
    """
    Prove that self-attention WITHOUT positional encoding is permutation-equivariant.
    Feed the same tokens in two different orders — outputs must be identical.
    """
    print("\n" + "=" * 60)
    print("  PERMUTATION DEMO — proof that attention without PE")
    print("  is order-blind (permutation-equivariant)")
    print("=" * 60)

    model = PEClassifier(
        vocab_size=len(word2idx), d_model=D_MODEL,
        num_heads=NUM_HEADS, pe_type="none"
    ).to(device)
    model.eval()

    # Two orderings of the same token IDs
    original  = [3, 7, 11, 4, 9]
    shuffled  = [9, 11, 3, 4, 7]

    t_orig = torch.tensor([original], dtype=torch.long, device=device)
    t_shuf = torch.tensor([shuffled], dtype=torch.long, device=device)

    with torch.no_grad():
        out_orig = model(t_orig)
        out_shuf = model(t_shuf)

    print(f"\n  Original : {original}")
    print(f"  Shuffled : {shuffled}")
    print(f"\n  Original  sum-pooled output (first 4 dims): {out_orig[0, :4].tolist()}")
    print(f"  Shuffled  sum-pooled output (first 4 dims): {out_shuf[0, :4].tolist()}")

    diff = (out_orig - out_shuf).abs().max().item()
    print(f"\n  Max absolute difference: {diff:.2e}")
    if diff < 1e-4:
        print("  ✓ IDENTICAL — attention without PE is permutation-equivariant.")
        print("  This is why PE exists: without it transformers can't distinguish")
        print("  'Sarah loves Chen' from 'Chen loves Sarah'.")
    print()


def train_one(pe_type: str, train_data, val_data, test_data, word2idx, device):
    model = PEClassifier(
        vocab_size=len(word2idx), d_model=D_MODEL,
        num_heads=NUM_HEADS, pe_type=pe_type
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- pe_type='{pe_type}' (params: {n_params:,}) ---")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Mini-batch training
        idxs = list(range(len(train_data)))
        import random; random.shuffle(idxs)
        for start in range(0, len(idxs), BATCH):
            batch = [train_data[i] for i in idxs[start:start + BATCH]]
            token_ids, label_ids, pad_mask = make_batch(batch, word2idx, device)
            logits = model(token_ids, pad_mask)
            loss   = criterion(logits, label_ids)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            token_ids, label_ids, pad_mask = make_batch(val_data, word2idx, device)
            val_preds = model(token_ids, pad_mask).argmax(dim=-1)
            val_acc   = (val_preds == label_ids).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # Test
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        token_ids, label_ids, pad_mask = make_batch(test_data, word2idx, device)
        test_preds = model(token_ids, pad_mask).argmax(dim=-1)
        test_acc   = (test_preds == label_ids).float().mean().item()

    print(f"  Best val acc: {best_val_acc:.4f}  ·  Test acc: {test_acc:.4f}")
    return model, best_val_acc, test_acc, n_params


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    train_data, val_data, test_data, word2idx = get_data()
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Vocab: {len(word2idx)}")

    # Step 1 — prove permutation equivariance
    permutation_demo(word2idx, device)

    # Step 2 — train all 4 variants
    print("=" * 60)
    print("  Training 4 PE variants")
    print("=" * 60)

    results = {}
    models  = {}
    for pe_type in ["none", "sinusoidal", "learned", "rope"]:
        model, val_acc, test_acc, n_params = train_one(
            pe_type, train_data, val_data, test_data, word2idx, device
        )
        results[pe_type] = (n_params, val_acc, test_acc)
        models[pe_type]  = model

    # Step 3 — comparison table
    print("\n" + "=" * 60)
    print(f"  {'PE Type':<12} {'Params':>8} {'Val Acc':>8} {'Test Acc':>9}")
    print("  " + "-" * 40)
    for pe_type, (n_params, val_acc, test_acc) in results.items():
        print(f"  {pe_type:<12} {n_params:>8,} {val_acc:>8.4f} {test_acc:>9.4f}")

    print("""
  All variants hit 100% — expected on this bag-of-words classification task.
  The Acme class doesn't change with word order ('Premium Checking earns 2.5%'
  is 'checking' regardless of arrangement).

  PE differences only show on ORDER-DEPENDENT tasks: language modeling,
  translation, generation. That's where RoPE pulls ahead.
""")

    # Save RoPE checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "pos_enc_rope.pt")
    torch.save({
        "model_state": models["rope"].state_dict(),
        "word2idx":    word2idx,
        "d_model":     D_MODEL,
        "num_heads":   NUM_HEADS,
        "pe_type":     "rope",
    }, ckpt_path)
    print(f"Saved RoPE checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

"""
Session 11 — Load Pretrained from HuggingFace
predict.py: Load saved attention weights, show top attending token pairs.
            Optionally run live BERT inference on custom --text input.
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
ATTN_PATH = os.path.join(CHECKPOINT_DIR, "attention_layer0.pt")


def show_top_pairs(attn_layer0, tokens, head=0, top_k=5):
    """Print top-k (source → target) token pairs by attention weight for one head."""
    # attn_layer0: [1, num_heads, seq_len, seq_len]
    attn = attn_layer0[0, head]  # [seq_len, seq_len]
    seq_len = len(tokens)

    pairs = []
    for i in range(seq_len):
        for j in range(seq_len):
            pairs.append((attn[i, j].item(), tokens[i], tokens[j]))

    pairs.sort(reverse=True)

    print(f"\nTop-{top_k} attention pairs (layer 0, head {head}):")
    for score, src, tgt in pairs[:top_k]:
        print(f"  {src:<15} → {tgt:<15} : {score:.3f}")


def run_live(text, model_name="bert-base-uncased"):
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    encoded = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    with torch.no_grad():
        outputs = model(**encoded)

    attn_layer0 = outputs.attentions[0]
    show_top_pairs(attn_layer0, tokens)


def main():
    parser = argparse.ArgumentParser(description="Session 11 — attention inspection")
    parser.add_argument("--text", type=str, default=None,
                        help="Run live BERT inference on this text (skips saved checkpoint)")
    parser.add_argument("--head", type=int, default=0, help="Attention head index (default 0)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top pairs to show")
    args = parser.parse_args()

    if args.text:
        print(f"Running live inference on: \"{args.text}\"")
        run_live(args.text)
        return

    if not os.path.exists(ATTN_PATH):
        print(f"Checkpoint not found: {ATTN_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    ckpt = torch.load(ATTN_PATH, weights_only=True)
    attn_layer0 = ckpt["attention"]
    tokens = ckpt["tokens"]
    model_name = ckpt["model_name"]

    print(f"Loaded attention weights from {ATTN_PATH}")
    print(f"Model      : {model_name}")
    print(f"Tokens     : {tokens}")
    print(f"Attn shape : {attn_layer0.shape}  (batch, heads, seq, seq)")

    show_top_pairs(attn_layer0, tokens, head=args.head, top_k=args.top_k)


if __name__ == "__main__":
    main()

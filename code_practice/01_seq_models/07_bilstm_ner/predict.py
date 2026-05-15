"""
Session 7 — BiLSTM for NER
predict.py: CLI inference with pretty NER output ([PERSON: ...] formatting).
"""

import sys
import os
import re
import argparse
import torch

_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, "..", "..", "..")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from model import BiLSTMTagger

CHECKPOINT = os.path.join(_DIR, "checkpoints", "bilstm_ner.pt")


def tokenize(text: str) -> list:
    """Tokenize text same way as training data.
    Split on whitespace; periods become separate tokens.
    NOTE: decimal points (e.g. 2.5) are also split — known limitation.
    """
    return text.replace(".", " .").split()


def predict(model, tokens, word2idx, idx2label, device):
    unk_id = word2idx.get("<UNK>", 1)
    ids = [word2idx.get(t, unk_id) for t in tokens]
    x   = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(x)           # [1, seq, num_labels]
    pred_ids = logits.argmax(dim=-1)[0].tolist()
    return [idx2label[i] for i in pred_ids]


def format_entities(tokens, tags):
    """Format output as [TYPE: span] for entity tokens."""
    result = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            entity_type = tag[2:]
            span = [tokens[i]]
            j = i + 1
            while j < len(tokens) and tags[j] == f"I-{entity_type}":
                span.append(tokens[j])
                j += 1
            result.append(f"[{entity_type}: {' '.join(span)}]")
            i = j
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def main():
    parser = argparse.ArgumentParser(description="Session 7 — BiLSTM NER inference")
    parser.add_argument("--text", type=str,
                        default="Sarah Chen works as Senior Analyst in Risk department.",
                        help="Input sentence to tag")
    args = parser.parse_args()

    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}\nRun train.py first.")
        sys.exit(1)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    model, ckpt = BiLSTMTagger.load(CHECKPOINT, device=str(device))
    model.to(device)

    word2idx  = ckpt["word2idx"]
    idx2label = ckpt["idx2label"]

    tokens = tokenize(args.text)
    tags   = predict(model, tokens, word2idx, idx2label, device)

    print(f"\nInput:\n  {args.text}")
    print("\nRaw tokens and tags:")
    for tok, tag in zip(tokens, tags):
        marker = " ★" if tag != "O" else ""
        print(f"  {tok:<20} {tag}{marker}")

    highlighted = format_entities(tokens, tags)
    print(f"\nHighlighted:\n  {highlighted}")


if __name__ == "__main__":
    main()

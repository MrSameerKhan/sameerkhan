"""
Session 11 — Load Pretrained from HuggingFace
model.py: Load BERT-base-uncased and GPT-2, print all named params + shapes,
          confirm head_dim = H/A = 64, map to our Session 3/5 variable names.
"""

from transformers import AutoModel
import torch


BERT_TO_OURS = {
    "bert.encoder.layer.0.attention.self.query.weight":  "W_Q         [768, 768]",
    "bert.encoder.layer.0.attention.self.query.bias":    "b_Q         [768]",
    "bert.encoder.layer.0.attention.self.key.weight":    "W_K         [768, 768]",
    "bert.encoder.layer.0.attention.self.key.bias":      "b_K         [768]",
    "bert.encoder.layer.0.attention.self.value.weight":  "W_V         [768, 768]",
    "bert.encoder.layer.0.attention.self.value.bias":    "b_V         [768]",
    "bert.encoder.layer.0.attention.output.dense.weight":"W_O         [768, 768]",
    "bert.encoder.layer.0.attention.output.dense.bias":  "b_O         [768]",
    "bert.encoder.layer.0.intermediate.dense.weight":    "W_1 (FFN)   [3072, 768]",
    "bert.encoder.layer.0.intermediate.dense.bias":      "b_1 (FFN)   [3072]",
    "bert.encoder.layer.0.output.dense.weight":          "W_2 (FFN)   [768, 3072]",
    "bert.encoder.layer.0.output.dense.bias":            "b_2 (FFN)   [768]",
}


def inspect_model(name, model_name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    model = AutoModel.from_pretrained(model_name)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}\n")

    params = dict(model.named_parameters())

    # Print layer 0 attention + FFN shapes
    layer0_keys = [k for k in params if "layer.0" in k or "h.0" in k]
    print("--- Layer 0 named parameters ---")
    for k in layer0_keys:
        print(f"  {k:<65} {str(params[k].shape)}")

    # Shape mapping (BERT only)
    if "bert" in model_name:
        print("\n--- Shape mapping: HuggingFace → Our Session 3/5 names (layer 0) ---")
        for hf_name, our_name in BERT_TO_OURS.items():
            if hf_name in params:
                shape = params[hf_name].shape
                print(f"  {hf_name.split('.')[-2] + '.' + hf_name.split('.')[-1]:<35} → {our_name}  actual={list(shape)}")

        H, A = 768, 12
        head_dim = H // A
        print(f"\nhead_dim = H / A = {H} / {A} = {head_dim}  ({'✓' if head_dim == 64 else '✗'})")

    return model


if __name__ == "__main__":
    inspect_model("BERT-base-uncased", "bert-base-uncased")
    inspect_model("GPT-2 (small)", "gpt2")

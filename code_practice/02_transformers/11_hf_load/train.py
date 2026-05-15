"""
Session 11 — Load Pretrained from HuggingFace
train.py: Forward pass on Acme corpus text, extract attention weights, save to checkpoint.
          No gradient updates — pretrained model in eval mode.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from transformers import AutoTokenizer, AutoModel

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_acme_sentences():
    # Acme Financial Services synthetic corpus — inline fallback if shared_dataset unavailable
    try:
        from code_practice.shared_dataset import get_text_corpus
        return get_text_corpus()[:3]
    except ImportError:
        return [
            "Acme Financial Services approved the mortgage document for client ID 4821.",
            "The loan application for property type residential was submitted by the borrower.",
            "Acme document classifier flagged the title deed as requiring manual review.",
        ]


def run(model_name="bert-base-uncased"):
    texts = get_acme_sentences()
    text = texts[0]
    print(f"Input text : \"{text}\"")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    encoded = tokenizer(text, return_tensors="pt")
    token_strs = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    print(f"Tokens     : {token_strs}")
    print(f"Token IDs  : {encoded['input_ids'][0].tolist()}")
    print(f"Input shape: {encoded['input_ids'].shape}\n")

    print(f"Running {model_name} forward pass...")
    with torch.no_grad():
        outputs = model(**encoded)

    last_hidden = outputs.last_hidden_state
    print(f"Last hidden state shape: {last_hidden.shape}")

    # BERT has pooler_output; GPT-2 does not
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        print(f"Pooler output shape    : {outputs.pooler_output.shape}")

    # outputs.attentions: tuple of (num_layers,) each [batch, heads, seq, seq]
    attn_all_layers = outputs.attentions
    attn_layer0 = attn_all_layers[0]   # [1, 12, seq_len, seq_len]
    print(f"\nAttention weights (layer 0, all heads): {attn_layer0.shape}")
    print(f"Attention weights (layer 0, head  0) : {attn_layer0[:, 0, :, :].shape}")

    save_path = os.path.join(CHECKPOINT_DIR, "attention_layer0.pt")
    torch.save({
        "attention": attn_layer0,
        "tokens": token_strs,
        "model_name": model_name,
    }, save_path)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    run()

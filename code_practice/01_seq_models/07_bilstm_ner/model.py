"""
Session 7 — BiLSTM for NER
model.py: BiLSTMTagger class + save/load helpers.

Architecture (~30 lines):
    Embedding(vocab_size, embed_dim, padding_idx=0)
    → nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
    → nn.Linear(2 * hidden_size, num_labels)

Output dimension is 2 × hidden_size (forward concat backward)
→ Linear(2*hidden, num_labels) projects to per-token tag scores.
"""

import torch
import torch.nn as nn


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, num_labels, embed_dim=64, hidden_size=64, pad_token_id=0):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc     = nn.Linear(2 * hidden_size, num_labels)

    def forward(self, x):
        emb        = self.embed(x)       # (batch, seq, embed_dim)
        out, _     = self.bilstm(emb)    # (batch, seq, 2*hidden)
        return self.fc(out)              # (batch, seq, num_labels)

    def save(self, path, **meta):
        torch.save({"model_state": self.state_dict(), **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            vocab_size   = ckpt["vocab_size"],
            num_labels   = ckpt["num_labels"],
            embed_dim    = ckpt["embed_dim"],
            hidden_size  = ckpt["hidden_size"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

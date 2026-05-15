"""
Session 4 — LSTM with PyTorch (nn.LSTM)
model.py: CharLSTM — Embedding + LSTM + Linear.
"""

import torch
import torch.nn as nn


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm  = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers,
                             batch_first=True)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x      : [batch, seq_len]  char indices
        hidden : (h, c) or None
        Returns: logits [batch, seq_len, vocab_size], (h_n, c_n)
        """
        emb    = self.embed(x)             # [B, T, E]
        out, hidden = self.lstm(emb, hidden)  # [B, T, H]
        logits = self.fc(out)              # [B, T, V]
        return logits, hidden

    def init_hidden(self, batch_size, device):
        H, L = self.hidden_size, self.num_layers
        h = torch.zeros(L, batch_size, H, device=device)
        c = torch.zeros(L, batch_size, H, device=device)
        return h, c

    def save(self, path, **meta):
        torch.save({"model_state": self.state_dict(),
                    "vocab_size":  meta.get("vocab_size"),
                    "embed_dim":   meta.get("embed_dim"),
                    "hidden_size": self.hidden_size,
                    "num_layers":  self.num_layers,
                    **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["vocab_size"], ckpt["embed_dim"],
                    ckpt["hidden_size"], ckpt["num_layers"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

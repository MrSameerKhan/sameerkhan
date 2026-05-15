"""
Session 6 — GRU vs LSTM in PyTorch
model.py: CharModel with model_type='gru'|'lstm' — single class, two backends.
"""

import torch
import torch.nn as nn


class CharModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128,
                 num_layers=1, model_type="gru"):
        super().__init__()
        assert model_type in ("gru", "lstm"), f"model_type must be 'gru' or 'lstm'"
        self.model_type  = model_type
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)

        rnn_cls = nn.GRU if model_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(embed_dim, hidden_size, num_layers=num_layers,
                           batch_first=True)
        self.fc  = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x      : [batch, seq_len]
        Returns: logits [batch, seq_len, vocab_size], hidden
        """
        emb        = self.embed(x)
        out, hidden = self.rnn(emb, hidden)
        logits     = self.fc(out)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        H, L = self.hidden_size, self.num_layers
        h = torch.zeros(L, batch_size, H, device=device)
        if self.model_type == "lstm":
            c = torch.zeros(L, batch_size, H, device=device)
            return (h, c)
        return h

    def detach_hidden(self, hidden):
        if isinstance(hidden, tuple):
            return tuple(h.detach() for h in hidden)
        return hidden.detach()

    def save(self, path, **meta):
        torch.save({"model_state": self.state_dict(),
                    "model_type":  self.model_type,
                    "embed_dim":   meta.get("embed_dim"),
                    "hidden_size": self.hidden_size,
                    "num_layers":  self.num_layers,
                    **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(ckpt["vocab_size"], ckpt["embed_dim"],
                    ckpt["hidden_size"], ckpt["num_layers"],
                    model_type=ckpt["model_type"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

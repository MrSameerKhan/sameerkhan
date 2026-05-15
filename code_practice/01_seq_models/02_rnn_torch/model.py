"""
Session 2 — RNN with PyTorch
model.py: CharRNN — nn.Embedding + nn.RNN + nn.Linear.
"""

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 128,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn   = nn.RNN(embed_dim, hidden_size, num_layers=num_layers,
                            batch_first=False, dropout=dropout if num_layers > 1 else 0.0)
        self.fc    = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        """
        x : [seq_len, batch]   char indices
        h : [num_layers, batch, hidden_size] or None
        Returns:
            logits : [seq_len, batch, vocab_size]
            h_n    : [num_layers, batch, hidden_size]
        """
        emb    = self.embed(x)              # [T, B, E]
        out, h_n = self.rnn(emb, h)        # out [T, B, H], h_n [layers, B, H]
        logits = self.fc(out)              # [T, B, V]
        return logits, h_n

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

"""
Session 8 — Bahdanau Attention on BiLSTM
model.py: BiLSTMClassifier with switchable pooling + AttentionPooling class.

Bahdanau score: score_i = v^T · tanh(W · h_i + b)
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    Bahdanau additive attention pooling — ~10 lines.
    score_i = v^T tanh(W h_i)
    alpha   = softmax(scores)
    output  = Σ alpha_i * h_i
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hiddens, mask=None):
        """
        hiddens : [batch, seq, hidden_dim]
        mask    : [batch, seq] — 1 for real tokens, 0 for padding
        Returns : pooled [batch, hidden_dim], weights [batch, seq]
        """
        scores = self.v(torch.tanh(self.W(hiddens))).squeeze(-1)   # [batch, seq]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)                      # [batch, seq]
        pooled  = torch.bmm(weights.unsqueeze(1), hiddens).squeeze(1)  # [batch, hidden]
        return pooled, weights


class BiLSTMClassifier(nn.Module):
    """
    BiLSTM sentence classifier with switchable pooling.
    pool: 'attention' | 'mean' | 'max' | 'last'
    """
    def __init__(self, vocab_size, num_classes, embed_dim=64, hidden_size=64,
                 pool="attention", pad_token_id=0):
        super().__init__()
        self.pool_type   = pool
        self.hidden_size = hidden_size

        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        self.fc     = nn.Linear(2 * hidden_size, num_classes)

        if pool == "attention":
            self.attention = AttentionPooling(2 * hidden_size)

    def forward(self, x, mask=None):
        """
        x    : [batch, seq]  token ids
        mask : [batch, seq]  1=real, 0=pad
        Returns logits [batch, num_classes] and attn_weights [batch, seq] (or None)
        """
        emb     = self.embed(x)               # [B, T, E]
        hiddens, (h_n, _) = self.bilstm(emb)  # hiddens [B, T, 2H]
        attn_weights = None

        if self.pool_type == "attention":
            pooled, attn_weights = self.attention(hiddens, mask)

        elif self.pool_type == "mean":
            if mask is not None:
                # divide by REAL count, not seq_len (avoids pulling mean toward zero)
                m       = mask.unsqueeze(-1).float()
                pooled  = (hiddens * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                pooled = hiddens.mean(dim=1)

        elif self.pool_type == "max":
            if mask is not None:
                neg_inf = torch.full_like(hiddens, float("-inf"))
                masked  = torch.where(mask.unsqueeze(-1).bool(), hiddens, neg_inf)
                pooled  = masked.max(dim=1)[0]
            else:
                pooled = hiddens.max(dim=1)[0]

        elif self.pool_type == "last":
            # Use final forward hidden from h_n (h_n shape: [2, B, H])
            pooled = torch.cat([h_n[0], h_n[1]], dim=-1)   # [B, 2H]

        return self.fc(pooled), attn_weights

    def save(self, path, **meta):
        torch.save({"model_state": self.state_dict(), "pool": self.pool_type, **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            vocab_size  = ckpt["vocab_size"],
            num_classes = ckpt["num_classes"],
            embed_dim   = ckpt["embed_dim"],
            hidden_size = ckpt["hidden_size"],
            pool        = ckpt["pool"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

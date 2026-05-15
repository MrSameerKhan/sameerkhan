"""
Phase 2, Session 5 — Transformer Encoder Block
model.py: RoPE + MHA + FFN + LayerNorm + TransformerEncoderBlock + classifier.

Block structure (pre-norm):
    x = x + MHA(LayerNorm(x))    ← sublayer 1: attention + residual
    x = x + FFN(LayerNorm(x))    ← sublayer 2: feed-forward + residual

~20 lines of code. Same block stacked 32× in Llama-3 8B.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Positional Encoding ────────────────────────────────────────────────────

class RoPE(nn.Module):
    def __init__(self, d_head: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def _rotate_half(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x):
        T   = x.size(-2)
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=-1)
        return x * cos + self._rotate_half(x) * sin


# ─── Multi-Head Self-Attention ───────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_head, max_len)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        def project(W):
            return W(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        Q, K, V = project(self.W_q), project(self.W_k), project(self.W_v)
        Q, K    = self.rope(Q), self.rope(K)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        weights  = F.softmax(scores, dim=-1)
        attended = (weights @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(attended), weights


# ─── Feed-Forward Network ────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Per-token MLP: d_model → 4*d_model → d_model with GELU."""
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


# ─── Transformer Encoder Block ───────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    One transformer encoder block (pre-norm):
        x = x + MHA(LN(x))
        x = x + FFN(LN(x))
    """
    def __init__(self, d_model=64, num_heads=8, d_ff=256,
                 max_len=512, dropout=0.0):
        super().__init__()
        self.ln1     = nn.LayerNorm(d_model)
        self.mha     = MultiHeadAttention(d_model, num_heads, max_len)
        self.ln2     = nn.LayerNorm(d_model)
        self.ffn     = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, mask=None):
        attended, weights = self.mha(self.ln1(x), mask)
        x = x + self.dropout(attended)          # residual 1
        x = x + self.dropout(self.ffn(self.ln2(x)))  # residual 2
        return x, weights


# ─── Full Classifier ─────────────────────────────────────────────────────────

class EncoderClassifier(nn.Module):
    """
    Embedding → TransformerEncoderBlock → masked mean pool → Linear classifier.
    """
    def __init__(self, vocab_size: int, d_model=64, num_heads=8,
                 d_ff=256, n_classes=5, max_len=512, dropout=0.0):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.block   = TransformerEncoderBlock(d_model, num_heads, d_ff, max_len, dropout)
        self.ln_out  = nn.LayerNorm(d_model)
        self.fc      = nn.Linear(d_model, n_classes)

    def forward(self, token_ids, pad_mask=None):
        x = self.embed(token_ids)                      # [B, T, d_model]
        x, weights = self.block(x, pad_mask)
        x = self.ln_out(x)

        if pad_mask is not None:
            real   = (~pad_mask).unsqueeze(-1).float()
            pooled = (x * real).sum(1) / real.sum(1)
        else:
            pooled = x.mean(dim=1)

        return self.fc(pooled), weights                # [B, n_classes], [B, H, T, T]

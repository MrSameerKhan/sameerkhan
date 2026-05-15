"""
Phase 2, Session 4 — Positional Encoding
model.py: SinusoidalPE, LearnedPE, RoPE + PEClassifier with switchable pe_type.

Three PE strategies:
  sinusoidal — fixed sin/cos added to input embedding (original Transformer)
  learned    — trainable embedding table, one vector per position (BERT, GPT-2)
  rope       — rotates Q and K inside attention by position-dependent angle (Llama, Mistral)
  none       — no positional info (proves attention is permutation-equivariant)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class RoPE(nn.Module):
    """
    Rotary Positional Embedding — applied inside attention to Q and K vectors.
    Rotates each pair of dimensions (2i, 2i+1) by angle θ_i = pos / 10000^(2i/d_head).
    """
    def __init__(self, d_head: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, inv_freq)        # [max_len, d_head//2]
        self.register_buffer("cos_cached", torch.cos(freqs))
        self.register_buffer("sin_cached", torch.sin(freqs))

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, num_heads, T, d_head]
        T = x.size(-2)
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)   # [1, 1, T, d_head//2]
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        cos = cos.repeat_interleave(2, dim=-1)   # [1, 1, T, d_head]
        sin = sin.repeat_interleave(2, dim=-1)
        return x * cos + self._rotate_half(x) * sin


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 pe_type: str = "none", max_len: int = 512):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head    = d_model // num_heads
        self.pe_type   = pe_type

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        if pe_type == "rope":
            self.rope = RoPE(self.d_head, max_len)

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        B, T, _ = x.shape

        def project(W):
            return W(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        Q, K, V = project(self.W_q), project(self.W_k), project(self.W_v)

        if self.pe_type == "rope":
            Q = self.rope(Q)
            K = self.rope(K)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if pad_mask is not None:
            scores = scores.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        weights   = F.softmax(scores, dim=-1)
        attended  = (weights @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(attended)


class PEClassifier(nn.Module):
    """
    Same architecture as Session 03 MHA, but with switchable positional encoding.
    pe_type: 'none' | 'sinusoidal' | 'learned' | 'rope'
    """
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 8,
                 n_classes: int = 5, max_len: int = 512, pe_type: str = "sinusoidal"):
        super().__init__()
        self.pe_type = pe_type
        self.embed   = nn.Embedding(vocab_size, d_model, padding_idx=0)

        if pe_type == "sinusoidal":
            self.pe = SinusoidalPE(d_model, max_len)
        elif pe_type == "learned":
            self.pe = LearnedPE(d_model, max_len)
        else:
            self.pe = None  # none → no PE;  rope → PE lives inside attention

        self.attn = MultiHeadSelfAttention(d_model, num_heads, pe_type=pe_type, max_len=max_len)
        self.fc   = nn.Linear(d_model, n_classes)

    def forward(self, token_ids: torch.Tensor, pad_mask=None) -> torch.Tensor:
        x = self.embed(token_ids)                     # [B, T, d_model]

        if self.pe is not None:
            x = self.pe(x)

        x = self.attn(x, pad_mask)                   # [B, T, d_model]

        if pad_mask is not None:
            real   = (~pad_mask).unsqueeze(-1).float()
            pooled = (x * real).sum(1) / real.sum(1)
        else:
            pooled = x.mean(dim=1)

        return self.fc(pooled)                        # [B, n_classes]

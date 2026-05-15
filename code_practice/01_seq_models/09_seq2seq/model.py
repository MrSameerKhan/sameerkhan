"""
Session 9 — Seq2Seq with Bahdanau Attention
model.py: Encoder (BiLSTM) + BahdanauAttention + Decoder (LSTM) + Seq2Seq wrapper.

Attention:
    score_i = v^T · tanh(W_enc·enc_i + W_dec·h_dec)
    α       = softmax(scores)
    context = Σ α_i · enc_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from data import SOS_IDX, EOS_IDX, PAD_IDX


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, dec_hidden, pad_idx=PAD_IDX):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed  = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.bilstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, bidirectional=True)
        # Project BiLSTM final state [2H] → [dec_hidden] to match decoder LSTM
        self.h_proj = nn.Linear(2 * hidden_size, dec_hidden)
        self.c_proj = nn.Linear(2 * hidden_size, dec_hidden)

    def forward(self, src, src_mask=None):
        """
        src     : [B, T_src]
        Returns :
            enc_out [B, T_src, 2H]   — all encoder hidden states
            (h, c)  [1, B, H]        — projected initial decoder states
        """
        emb = self.embed(src)                             # [B, T, E]
        enc_out, (h_n, c_n) = self.bilstm(emb)           # enc_out [B,T,2H]

        # h_n: [2, B, H] — concat forward+backward final states
        h = torch.cat([h_n[0], h_n[1]], dim=-1)          # [B, 2H]
        c = torch.cat([c_n[0], c_n[1]], dim=-1)          # [B, 2H]

        h = torch.tanh(self.h_proj(h)).unsqueeze(0)      # [1, B, H]
        c = torch.tanh(self.c_proj(c)).unsqueeze(0)      # [1, B, H]

        return enc_out, (h, c)


class BahdanauAttention(nn.Module):
    """
    Additive attention: score_i = v^T tanh(W_enc·enc_i + W_dec·h_dec)
    Applied at each decoder step over all encoder positions.
    """
    def __init__(self, enc_hidden, dec_hidden, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden, attn_dim, bias=False)
        self.W_dec = nn.Linear(dec_hidden, attn_dim, bias=False)
        self.v     = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out, h_dec, src_mask=None):
        """
        enc_out : [B, T_src, enc_hidden]
        h_dec   : [B, dec_hidden]
        src_mask: [B, T_src]
        Returns : context [B, enc_hidden], alpha [B, T_src]
        """
        T = enc_out.size(1)
        h_dec_exp = h_dec.unsqueeze(1).expand(-1, T, -1)  # [B, T, dec_H]

        energy = torch.tanh(self.W_enc(enc_out) + self.W_dec(h_dec_exp))  # [B, T, attn_dim]
        scores = self.v(energy).squeeze(-1)                                # [B, T]

        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float("-inf"))

        alpha   = torch.softmax(scores, dim=-1)                            # [B, T]
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)       # [B, enc_H]
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, dec_hidden, enc_hidden,
                 attn_dim, pad_idx=PAD_IDX):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(enc_hidden, dec_hidden, attn_dim)
        # Input: [embed ; context] → concatenated
        self.lstm      = nn.LSTM(embed_dim + enc_hidden, dec_hidden, batch_first=True)
        self.fc        = nn.Linear(dec_hidden, vocab_size)

    def forward_step(self, token, hidden, enc_out, src_mask=None):
        """
        One decoder step.
        token   : [B]      current input token idx
        hidden  : (h, c)   [1, B, dec_H]
        Returns : logit [B, vocab], hidden, alpha [B, T_src]
        """
        h_dec = hidden[0].squeeze(0)                           # [B, dec_H]
        context, alpha = self.attention(enc_out, h_dec, src_mask)

        emb   = self.embed(token)                              # [B, E]
        inp   = torch.cat([emb, context], dim=-1).unsqueeze(1)  # [B, 1, E+enc_H]
        out, hidden = self.lstm(inp, hidden)                   # out [B,1,dec_H]
        logit = self.fc(out.squeeze(1))                        # [B, vocab]
        return logit, hidden, alpha

    def forward(self, tgt, hidden, enc_out, src_mask=None,
                teacher_forcing_ratio=0.5):
        """
        tgt      : [B, T_tgt]   target sequence (SOS ... EOS)
        Returns  : logits [B, T_tgt-1, vocab], alphas [B, T_tgt-1, T_src]
        """
        B, T = tgt.size()
        logits = []
        alphas = []
        token  = tgt[:, 0]   # SOS

        for t in range(1, T):
            logit, hidden, alpha = self.forward_step(token, hidden, enc_out, src_mask)
            logits.append(logit)
            alphas.append(alpha)

            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            token = tgt[:, t] if use_teacher else logit.argmax(dim=-1)

        return torch.stack(logits, dim=1), torch.stack(alphas, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, enc_hidden=128, dec_hidden=256,
                 attn_dim=128, pad_idx=PAD_IDX):
        super().__init__()
        self.encoder = Encoder(vocab_size, embed_dim, enc_hidden, dec_hidden, pad_idx)
        self.decoder = Decoder(vocab_size, embed_dim, dec_hidden,
                               2 * enc_hidden, attn_dim, pad_idx)

    def forward(self, src, tgt, src_mask=None, teacher_forcing_ratio=0.5):
        enc_out, hidden = self.encoder(src, src_mask)
        logits, alphas  = self.decoder(tgt, hidden, enc_out, src_mask,
                                       teacher_forcing_ratio)
        return logits, alphas

    @torch.no_grad()
    def greedy_decode(self, src, src_mask=None, max_len=50):
        """
        Greedy decoding — returns list of token ids (no SOS, stops at EOS).
        """
        self.eval()
        enc_out, hidden = self.encoder(src, src_mask)
        token = torch.tensor([SOS_IDX], dtype=torch.long, device=src.device)
        result = []

        for _ in range(max_len):
            logit, hidden, alpha = self.decoder.forward_step(token, hidden, enc_out, src_mask)
            token = logit.argmax(dim=-1)
            if token.item() == EOS_IDX:
                break
            result.append(token.item())

        return result

    def save(self, path, **meta):
        torch.save({"model_state": self.state_dict(), **meta}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            vocab_size  = ckpt["vocab_size"],
            embed_dim   = ckpt["embed_dim"],
            enc_hidden  = ckpt["enc_hidden"],
            dec_hidden  = ckpt["dec_hidden"],
            attn_dim    = ckpt["attn_dim"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt

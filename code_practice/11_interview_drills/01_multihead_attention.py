"""
DRILL 01 — Multi-head self-attention from scratch.

WHY THIS DRILL: your resume says "a multi-head-attention Transformer I wrote
from scratch over frozen GloVe". An interviewer reads that and asks you to
write it. No nn.MultiheadAttention, no F.scaled_dot_product_attention.

RULES
  - torch tensor ops only: matmul, softmax, reshape/view, transpose, nn.Linear.
  - Do NOT import or call nn.MultiheadAttention / F.scaled_dot_product_attention.
  - Fill in every `raise NotImplementedError`. Then: python 01_multihead_attention.py
  - Target: 20 minutes, no reference material. Time yourself.

DRY RUN — hold these numbers in your head before you type:
  batch B=1, seq_len L=3, d_model=4, n_heads=2  ->  d_head = 4/2 = 2
  x            : (1, 3, 4)
  after W_q    : (1, 3, 4)
  after split  : (1, 2, 3, 2)      <- (B, n_heads, L, d_head)
  scores       : (1, 2, 3, 3)      <- Q @ K^T, one 3x3 map PER head
  weights      : (1, 2, 3, 3)      <- softmax over the LAST dim, rows sum to 1
  head out     : (1, 2, 3, 2)      <- weights @ V
  after merge  : (1, 3, 4)
  after W_o    : (1, 3, 4)

THE QUESTION THEY ASK NEXT (have the answer ready, don't code it):
  "Why divide by sqrt(d_k)?"  -> q.k is a sum of d_k products; if components are
  ~N(0,1) and independent, the dot product has variance d_k, so it grows with
  dimension. Large logits push softmax into a near-one-hot regime where the
  gradient is ~0 and training stalls. Dividing by sqrt(d_k) returns the logit
  variance to ~1 and keeps softmax in its responsive range.
"""

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q, k, v : (B, n_heads, L, d_head)
    mask    : (L, L) bool or None. True = KEEP, False = BLOCK.
              Blocked positions must get -inf BEFORE softmax, not 0 after.

    returns (out, weights)
        out     : (B, n_heads, L, d_head)
        weights : (B, n_heads, L, L)
    """
    raise NotImplementedError("scaled_dot_product_attention")


def causal_mask(L, device=None):
    """
    (L, L) bool. True where a query at row i is allowed to see key at col j.
    Position i may see 0..i inclusive. Row 0 sees only column 0.
    """
    raise NotImplementedError("causal_mask")


class MultiHeadAttention(nn.Module):
    """
    Self-attention only: Q, K and V all come from the same input x.

    You need four projections (W_q, W_k, W_v, W_o), a split into heads,
    the attention call, and a merge back. Store the last attention weights
    on self.last_weights so the tests can inspect them.
    """

    def __init__(self, d_model, n_heads, bias=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.last_weights = None
        raise NotImplementedError("MultiHeadAttention.__init__ projections")

    def _split_heads(self, t):
        """(B, L, d_model) -> (B, n_heads, L, d_head)"""
        raise NotImplementedError("_split_heads")

    def _merge_heads(self, t):
        """(B, n_heads, L, d_head) -> (B, L, d_model)"""
        raise NotImplementedError("_merge_heads")

    def forward(self, x, mask=None):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        raise NotImplementedError("MultiHeadAttention.forward")


# ---------------------------------------------------------------- verification


def _check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")
    return bool(cond)


def run_tests():
    torch.manual_seed(0)
    B, L, d_model, n_heads = 1, 3, 4, 2
    results = []

    print("\n[1] shapes")
    mha = MultiHeadAttention(d_model, n_heads)
    x = torch.randn(B, L, d_model)
    out = mha(x)
    results.append(_check(f"output is (B,L,d_model) = {(B, L, d_model)}", out.shape == (B, L, d_model)))
    results.append(_check(
        f"weights are (B,n_heads,L,L) = {(B, n_heads, L, L)}",
        mha.last_weights is not None and mha.last_weights.shape == (B, n_heads, L, L),
    ))

    print("\n[2] softmax is a distribution over keys")
    w = mha.last_weights
    results.append(_check("every attention row sums to 1", torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)))
    results.append(_check("no negative weights", bool((w >= 0).all())))

    print("\n[3] causal mask")
    m = causal_mask(L)
    results.append(_check("row 0 sees exactly 1 position", int(m[0].sum()) == 1))
    results.append(_check("last row sees all L positions", int(m[-1].sum()) == L))
    results.append(_check("strictly upper triangle is blocked", not bool(m.triu(1).any())))
    out_c = mha(x, mask=m)
    wc = mha.last_weights
    results.append(_check("masked positions get exactly 0 weight", torch.allclose(wc.triu(1), torch.zeros_like(wc.triu(1)), atol=1e-7)))
    results.append(_check("row 0 puts all its mass on position 0", torch.allclose(wc[..., 0, 0], torch.ones_like(wc[..., 0, 0]), atol=1e-5)))

    print("\n[4] the sqrt(d_k) scaling is actually applied")
    q = torch.randn(1, 1, 4, 8)
    k, v = torch.randn_like(q), torch.randn_like(q)
    _, wq = scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(8), dim=-1)
    results.append(_check("matches softmax(QK^T/sqrt(d_k))V", torch.allclose(wq, expected, atol=1e-6)))

    print("\n[5] permutation equivariance (why position embeddings exist)")
    perm = torch.tensor([2, 0, 1])
    out_a = mha(x)[:, perm, :]
    out_b = mha(x[:, perm, :])
    results.append(_check("permuting the input permutes the output identically", torch.allclose(out_a, out_b, atol=1e-5)))

    print("\n[6] attention mixes across positions (it is not a per-token MLP)")
    x2 = x.clone()
    x2[:, 2, :] += 10.0
    results.append(_check("changing token 2 changes token 0's output", not torch.allclose(mha(x)[:, 0], mha(x2)[:, 0], atol=1e-4)))
    print("      ...but NOT under a causal mask:")
    results.append(_check("with causal mask, token 0 is unaffected by token 2",
                          torch.allclose(mha(x, mask=m)[:, 0], mha(x2, mask=m)[:, 0], atol=1e-5)))

    print(f"\n{sum(results)}/{len(results)} passed\n")
    return all(results)


if __name__ == "__main__":
    run_tests()

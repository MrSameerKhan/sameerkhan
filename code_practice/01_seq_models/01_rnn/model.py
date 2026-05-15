"""
Session 1 — Vanilla RNN Cell from Scratch
model.py: VanillaRNN — NumPy only, no autograd.

Forward:  h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
          y_t = W_hy @ h_t + b_y
          p_t = softmax(y_t)

Backward: BPTT — manual gradient computation for all weights.
"""

import numpy as np


class VanillaRNN:
    def __init__(self, vocab_size: int, hidden_size: int, seed: int = 42):
        np.random.seed(seed)
        scale = 0.01
        self.Wxh = np.random.randn(hidden_size, vocab_size) * scale   # [H, V]
        self.Whh = np.random.randn(hidden_size, hidden_size) * scale  # [H, H]
        self.Why = np.random.randn(vocab_size, hidden_size) * scale   # [V, H]
        self.bh  = np.zeros((hidden_size, 1))                         # [H, 1]
        self.by  = np.zeros((vocab_size, 1))                          # [V, 1]

        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size

        # Adagrad memory (sum of squared gradients)
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh  = np.zeros_like(self.bh)
        self.mby  = np.zeros_like(self.by)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: list, h_prev: np.ndarray):
        """
        Args:
            inputs  : list of one-hot vectors, each [V, 1], length = seq_len
            h_prev  : initial hidden state [H, 1]
        Returns:
            xs, hs, ys, ps — dicts keyed by time step t
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)

        for t, x in enumerate(inputs):
            xs[t] = x                                                  # [V, 1]
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)  # [H, 1]
            ys[t] = self.Why @ hs[t] + self.by                        # [V, 1]
            exp_y = np.exp(ys[t] - ys[t].max())                       # stable softmax
            ps[t] = exp_y / exp_y.sum()                                # [V, 1]

        return xs, hs, ys, ps

    # ------------------------------------------------------------------
    # Backward (BPTT)
    # ------------------------------------------------------------------

    def backward(self, xs: dict, hs: dict, ps: dict, targets: list):
        """
        Args:
            xs      : one-hot inputs from forward()
            hs      : hidden states from forward()
            ps      : softmax probs from forward()
            targets : list of int target indices, length = seq_len
        Returns:
            loss (float), grads (dict)
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh  = np.zeros_like(self.bh)
        dby  = np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))

        loss = 0.0
        T = len(xs)

        for t in reversed(range(T)):
            # Cross-entropy loss + softmax gradient combined
            dy = np.copy(ps[t])          # [V, 1]
            dy[targets[t]] -= 1.0        # dL/d(logits)

            dWhy += dy @ hs[t].T         # [V, H]
            dby  += dy                   # [V, 1]

            # Gradient through hidden state
            dh = self.Why.T @ dy + dh_next          # [H, 1]
            dh_raw = (1.0 - hs[t] ** 2) * dh       # tanh derivative: 1 - h²

            dbh  += dh_raw                           # [H, 1]
            dWxh += dh_raw @ xs[t].T                # [H, V]
            dWhh += dh_raw @ hs[t-1].T              # [H, H]
            dh_next = self.Whh.T @ dh_raw           # pass gradient to previous step

            loss += -np.log(ps[t][targets[t], 0] + 1e-8)

        grads = {"dWxh": dWxh, "dWhh": dWhh, "dWhy": dWhy, "dbh": dbh, "dby": dby}

        # Clip to prevent exploding gradients
        for g in grads.values():
            np.clip(g, -5.0, 5.0, out=g)

        return loss, grads

    # ------------------------------------------------------------------
    # Adagrad update
    # ------------------------------------------------------------------

    def update(self, grads: dict, lr: float = 0.1):
        eps = 1e-8
        pairs = [
            ("dWxh", "Wxh", "mWxh"),
            ("dWhh", "Whh", "mWhh"),
            ("dWhy", "Why", "mWhy"),
            ("dbh",  "bh",  "mbh"),
            ("dby",  "by",  "mby"),
        ]
        for dk, wk, mk in pairs:
            mem = getattr(self, mk)
            mem += grads[dk] ** 2
            setattr(self, wk, getattr(self, wk) - lr * grads[dk] / (np.sqrt(mem) + eps))

    # ------------------------------------------------------------------
    # Sample (inference)
    # ------------------------------------------------------------------

    def sample(self, seed_idx: int, h: np.ndarray, length: int, temperature: float = 1.0):
        """Generate `length` character indices starting from seed_idx."""
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1.0
        generated = []

        for _ in range(length):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            y = y / temperature
            exp_y = np.exp(y - y.max())
            p = (exp_y / exp_y.sum()).ravel()
            idx = np.random.choice(len(p), p=p)
            generated.append(idx)
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1.0

        return generated

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        np.savez(path,
                 Wxh=self.Wxh, Whh=self.Whh, Why=self.Why,
                 bh=self.bh, by=self.by)

    def load(self, path: str):
        data = np.load(path)
        self.Wxh = data["Wxh"]
        self.Whh = data["Whh"]
        self.Why = data["Why"]
        self.bh  = data["bh"]
        self.by  = data["by"]

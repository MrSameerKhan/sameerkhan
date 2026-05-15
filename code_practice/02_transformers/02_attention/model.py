"""
Session 10 — Scaled Dot-Product Self-Attention (NumPy from scratch)
model.py: SelfAttentionClassifier — full forward + manual backward.

Architecture:
    token_ids  [T]
        ↓ Embedding lookup
    X  [T, d_model]
        ↓ Q = X W_Q,  K = X W_K,  V = X W_V
    Q,K,V  [T, d_k]
        ↓ S = Q K^T / sqrt(d_k),  mask padding,  A = softmax(S)
    A  [T, T]  (each position attends to all positions)
        ↓ Z = A V  (attended output)
    Z  [T, d_k]
        ↓ mean pool over real tokens
    pooled  [d_k]
        ↓ logits = W_out @ pooled + b_out
    logits  [n_classes]
        ↓ softmax → cross-entropy loss
"""

import os
import pickle
import numpy as np


class SelfAttentionClassifier:
    def __init__(self, vocab_size, d_model=32, d_k=32, n_classes=5, seed=42):
        rng   = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (d_model + d_k))

        self.E     = rng.randn(vocab_size, d_model) * 0.01   # embedding table
        self.W_Q   = rng.randn(d_model, d_k) * scale
        self.W_K   = rng.randn(d_model, d_k) * scale
        self.W_V   = rng.randn(d_model, d_k) * scale
        self.W_out = rng.randn(n_classes, d_k) * scale
        self.b_out = np.zeros(n_classes)

        self.d_k      = d_k
        self.n_classes = n_classes

        # Adam optimizer state
        self.t = 0
        self.m = [np.zeros_like(p) for p in self._params()]
        self.v = [np.zeros_like(p) for p in self._params()]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _params(self):
        return [self.E, self.W_Q, self.W_K, self.W_V, self.W_out, self.b_out]

    @property
    def n_params(self):
        return sum(p.size for p in self._params())

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _softmax_rows(S):
        """Row-wise softmax — numerically stable."""
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, token_ids, mask):
        """
        token_ids : int array [T]
        mask      : int array [T]  (1 = real token, 0 = padding)
        Returns   : probs [n_classes], cache dict
        """
        token_ids = np.asarray(token_ids, dtype=int)
        mask      = np.asarray(mask,      dtype=int)
        T         = len(token_ids)

        # Embedding lookup
        X = self.E[token_ids]                          # [T, d_model]

        # Linear projections to query / key / value spaces
        Q = X @ self.W_Q                               # [T, d_k]
        K = X @ self.W_K                               # [T, d_k]
        V = X @ self.W_V                               # [T, d_k]

        # Scaled dot-product scores
        S = (Q @ K.T) / np.sqrt(self.d_k)             # [T, T]

        # Mask padding: set pad key-columns and pad query-rows to a large negative
        pad = (mask == 0)
        S[:, pad] = -1e9
        S[pad, :]  = -1e9

        # Row-wise softmax → attention weights
        A = self._softmax_rows(S)                      # [T, T]

        # Attended output: each position is a weighted sum of values
        Z = A @ V                                      # [T, d_k]

        # Mean pool over real (non-padding) positions
        T_real = int(mask.sum())
        T_real = max(T_real, 1)
        pooled = Z[mask == 1].sum(axis=0) / T_real    # [d_k]

        # Classifier
        logits = self.W_out @ pooled + self.b_out      # [n_classes]
        probs  = self._softmax(logits)                 # [n_classes]

        cache = dict(
            token_ids=token_ids, mask=mask,
            X=X, Q=Q, K=K, V=V, S=S, A=A, Z=Z,
            pooled=pooled, logits=logits, probs=probs,
            T_real=T_real,
        )
        return probs, cache

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, cache, y):
        """
        y     : int, true class index
        Returns: (loss, grads dict)

        Gradient flow:
            loss → dlogits → dW_out, db_out, dpooled
            dpooled → dZ (distributed to real positions)
            dZ → dA (via Z = A V) and dV
            dA → dS (softmax backward, row-wise)
            dS → dQ, dK (via S = Q K^T / sqrt(d_k))
            dQ,dK,dV → dW_Q,dW_K,dW_V and dX (via Q=XW_Q etc.)
            dX → dE (scatter into embedding table)
        """
        probs     = cache["probs"]
        pooled    = cache["pooled"]
        A         = cache["A"]
        V         = cache["V"]
        Z         = cache["Z"]
        Q         = cache["Q"]
        K         = cache["K"]
        X         = cache["X"]
        mask      = cache["mask"]
        T_real    = cache["T_real"]
        token_ids = cache["token_ids"]
        T         = len(token_ids)

        loss = -np.log(probs[y] + 1e-8)

        # --- Classifier backward ---
        dlogits      = probs.copy()
        dlogits[y]  -= 1                               # [n_classes]

        dW_out = np.outer(dlogits, pooled)             # [n_classes, d_k]
        db_out = dlogits.copy()
        dpooled = self.W_out.T @ dlogits               # [d_k]

        # --- Mean pool backward ---
        dZ = np.zeros_like(Z)                          # [T, d_k]
        dZ[mask == 1] = dpooled / T_real               # equal share per real position

        # --- Attention output backward: Z = A @ V ---
        dA = dZ @ V.T                                  # [T, T]
        dV = A.T @ dZ                                  # [T, d_k]

        # --- Softmax backward (row-wise) ---
        # For row i: dS[i,j] = A[i,j] * (dA[i,j] - sum_k(A[i,k] * dA[i,k]))
        dS = np.zeros_like(A)
        for i in range(T):
            if mask[i]:                                # skip padding query rows
                ai  = A[i]
                dai = dA[i]
                dS[i] = ai * (dai - (dai * ai).sum())

        # --- Scale backward: S = Q K^T / sqrt(d_k) ---
        dS /= np.sqrt(self.d_k)

        # --- QK^T backward ---
        dQ = dS @ K                                    # [T, d_k]
        dK = dS.T @ Q                                  # [T, d_k]

        # --- Projection backward ---
        dW_Q = X.T @ dQ                                # [d_model, d_k]
        dW_K = X.T @ dK
        dW_V = X.T @ dV

        dX = dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T   # [T, d_model]

        # --- Embedding backward: scatter dX into dE ---
        dE = np.zeros_like(self.E)
        for i, tid in enumerate(token_ids):
            dE[tid] += dX[i]

        grads = dict(dE=dE, dW_Q=dW_Q, dW_K=dW_K, dW_V=dW_V,
                     dW_out=dW_out, db_out=db_out)
        return loss, grads

    # ------------------------------------------------------------------
    # Adam update
    # ------------------------------------------------------------------
    def update(self, grads, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        t = self.t
        grad_list  = [grads["dE"], grads["dW_Q"], grads["dW_K"],
                      grads["dW_V"], grads["dW_out"], grads["db_out"]]
        param_list = self._params()

        for i, (p, g) in enumerate(zip(param_list, grad_list)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g ** 2
            m_hat = self.m[i] / (1 - beta1 ** t)
            v_hat = self.v[i] / (1 - beta2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path, **meta):
        ckpt = dict(
            E=self.E, W_Q=self.W_Q, W_K=self.W_K, W_V=self.W_V,
            W_out=self.W_out, b_out=self.b_out,
            d_k=self.d_k, n_classes=self.n_classes,
            **meta,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(ckpt, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        vocab_size, d_model = ckpt["E"].shape
        model = cls(vocab_size=vocab_size, d_model=d_model,
                    d_k=ckpt["d_k"], n_classes=ckpt["n_classes"])
        model.E     = ckpt["E"]
        model.W_Q   = ckpt["W_Q"]
        model.W_K   = ckpt["W_K"]
        model.W_V   = ckpt["W_V"]
        model.W_out = ckpt["W_out"]
        model.b_out = ckpt["b_out"]
        return model, ckpt

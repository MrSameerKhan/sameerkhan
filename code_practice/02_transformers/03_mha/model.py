"""
Session 11 — Multi-Head Attention (NumPy from scratch)
model.py: MultiHeadAttentionClassifier — h parallel attention heads,
          concat + W_O projection, mean pool, classifier.

Architecture (h heads, d_k = d_model // h per head):

    X  [T, d_model]
        ↓  For each head k:
        ├─ Q_k = X W_Q[k]          [T, d_k]
        ├─ K_k = X W_K[k]          [T, d_k]
        ├─ V_k = X W_V[k]          [T, d_k]
        ├─ S_k = Q_k K_k^T / √d_k  [T, T]
        ├─ A_k = softmax(S_k)      [T, T]
        └─ Z_k = A_k V_k           [T, d_k]
        ↓  concat heads
    Z_cat  [T, d_model]
        ↓  W_O  [d_model, d_model]   ← output projection mixes heads
    Z_out  [T, d_model]
        ↓  mean pool real tokens
    pooled  [d_model]
        ↓  W_out  [n_classes, d_model]
    logits  [n_classes]

Why W_O?  Each head's output lives in a different d_k subspace.
W_O projects the concatenated heads back to d_model so they can
interact — without it, heads never see each other's information.
"""

import os
import pickle
import numpy as np


class MultiHeadAttentionClassifier:
    def __init__(self, vocab_size, d_model=32, n_heads=4, n_classes=5, seed=42):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        rng   = np.random.RandomState(seed)
        d_k   = d_model // n_heads
        scale = np.sqrt(2.0 / (d_model + d_k))

        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_k
        self.n_classes = n_classes

        # Embeddings
        self.E = rng.randn(vocab_size, d_model) * 0.01       # [V, d_model]

        # Per-head Q / K / V weight matrices
        self.W_Q = [rng.randn(d_model, d_k) * scale for _ in range(n_heads)]
        self.W_K = [rng.randn(d_model, d_k) * scale for _ in range(n_heads)]
        self.W_V = [rng.randn(d_model, d_k) * scale for _ in range(n_heads)]

        # Output projection (mixes all heads)
        self.W_O   = rng.randn(d_model, d_model) * scale     # [h*d_k, d_model]

        # Classifier
        self.W_out = rng.randn(n_classes, d_model) * scale
        self.b_out = np.zeros(n_classes)

        # Adam state
        self.t = 0
        self.m = [np.zeros_like(p) for p in self._flat_params()]
        self.v = [np.zeros_like(p) for p in self._flat_params()]

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------
    def _flat_params(self):
        """Flat list of all parameters for Adam state init."""
        ps = [self.E, self.W_O, self.W_out, self.b_out]
        for k in range(self.n_heads):
            ps += [self.W_Q[k], self.W_K[k], self.W_V[k]]
        return ps

    @property
    def n_params(self):
        return sum(p.size for p in self._flat_params())

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    @staticmethod
    def _softmax_rows(S):
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    @staticmethod
    def _softmax_bwd(A, dA, mask):
        """Row-wise softmax backward. Only computes grad for real query rows."""
        T  = A.shape[0]
        dS = np.zeros_like(A)
        for i in range(T):
            if mask[i]:
                ai  = A[i]
                dai = dA[i]
                dS[i] = ai * (dai - (dai * ai).sum())
        return dS

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, token_ids, mask):
        """
        token_ids : int array [T]
        mask      : int array [T]  (1 = real, 0 = pad)
        Returns   : probs [n_classes], cache dict
        """
        token_ids = np.asarray(token_ids, dtype=int)
        mask      = np.asarray(mask,      dtype=int)
        T         = len(token_ids)

        X = self.E[token_ids]          # [T, d_model]

        pad = (mask == 0)
        head_Qs, head_Ks, head_Vs = [], [], []
        head_As, head_Zs          = [], []

        for k in range(self.n_heads):
            Q_k = X @ self.W_Q[k]                               # [T, d_k]
            K_k = X @ self.W_K[k]
            V_k = X @ self.W_V[k]

            S_k = (Q_k @ K_k.T) / np.sqrt(self.d_k)            # [T, T]
            S_k[:, pad] = -1e9
            S_k[pad, :] = -1e9

            A_k = self._softmax_rows(S_k)                       # [T, T]
            Z_k = A_k @ V_k                                     # [T, d_k]

            head_Qs.append(Q_k); head_Ks.append(K_k)
            head_Vs.append(V_k); head_As.append(A_k)
            head_Zs.append(Z_k)

        # Concatenate heads along feature dimension
        Z_cat = np.concatenate(head_Zs, axis=-1)               # [T, d_model]

        # Output projection
        Z_out = Z_cat @ self.W_O                                # [T, d_model]

        # Mean pool over real tokens
        T_real = max(int(mask.sum()), 1)
        pooled = Z_out[mask == 1].sum(axis=0) / T_real          # [d_model]

        logits = self.W_out @ pooled + self.b_out               # [n_classes]
        probs  = self._softmax(logits)

        cache = dict(
            token_ids=token_ids, mask=mask, X=X,
            head_Qs=head_Qs, head_Ks=head_Ks,
            head_Vs=head_Vs, head_As=head_As, head_Zs=head_Zs,
            Z_cat=Z_cat, Z_out=Z_out,
            pooled=pooled, logits=logits, probs=probs,
            T_real=T_real,
        )
        return probs, cache

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------
    def backward(self, cache, y):
        probs     = cache["probs"]
        pooled    = cache["pooled"]
        Z_out     = cache["Z_out"]
        Z_cat     = cache["Z_cat"]
        X         = cache["X"]
        mask      = cache["mask"]
        T_real    = cache["T_real"]
        token_ids = cache["token_ids"]

        loss = -np.log(probs[y] + 1e-8)

        # --- Classifier ---
        dlogits      = probs.copy()
        dlogits[y]  -= 1
        dW_out  = np.outer(dlogits, pooled)
        db_out  = dlogits.copy()
        dpooled = self.W_out.T @ dlogits               # [d_model]

        # --- Mean pool ---
        dZ_out = np.zeros_like(Z_out)                  # [T, d_model]
        dZ_out[mask == 1] = dpooled / T_real

        # --- Output projection: Z_out = Z_cat @ W_O ---
        dW_O  = Z_cat.T @ dZ_out                       # [d_model, d_model]
        dZ_cat = dZ_out @ self.W_O.T                   # [T, d_model]

        # --- Per-head backward ---
        dX = np.zeros_like(X)
        dW_Q_list, dW_K_list, dW_V_list = [], [], []

        for k in range(self.n_heads):
            # Slice this head's chunk from dZ_cat
            start = k * self.d_k
            end   = start + self.d_k
            dZ_k  = dZ_cat[:, start:end]               # [T, d_k]

            A_k = cache["head_As"][k]
            V_k = cache["head_Vs"][k]
            Q_k = cache["head_Qs"][k]
            K_k = cache["head_Ks"][k]

            # Z_k = A_k @ V_k
            dA_k = dZ_k @ V_k.T                        # [T, T]
            dV_k = A_k.T @ dZ_k                        # [T, d_k]

            # Softmax backward
            dS_k = self._softmax_bwd(A_k, dA_k, mask)  # [T, T]
            dS_k /= np.sqrt(self.d_k)

            # QK^T backward
            dQ_k = dS_k @ K_k                          # [T, d_k]
            dK_k = dS_k.T @ Q_k                        # [T, d_k]

            # Projection backward
            dW_Q_list.append(X.T @ dQ_k)
            dW_K_list.append(X.T @ dK_k)
            dW_V_list.append(X.T @ dV_k)

            dX += dQ_k @ self.W_Q[k].T + dK_k @ self.W_K[k].T + dV_k @ self.W_V[k].T

        # --- Embedding scatter ---
        dE = np.zeros_like(self.E)
        for i, tid in enumerate(token_ids):
            dE[tid] += dX[i]

        grads = dict(
            dE=dE, dW_O=dW_O, dW_out=dW_out, db_out=db_out,
            dW_Q=dW_Q_list, dW_K=dW_K_list, dW_V=dW_V_list,
        )
        return loss, grads

    # ------------------------------------------------------------------
    # Adam update
    # ------------------------------------------------------------------
    def update(self, grads, lr=3e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        t = self.t

        # Build flat grad list matching _flat_params order
        # [E, W_O, W_out, b_out, W_Q[0..h], W_K[0..h], W_V[0..h]]
        grad_list = ([grads["dE"], grads["dW_O"], grads["dW_out"], grads["db_out"]]
                     + grads["dW_Q"] + grads["dW_K"] + grads["dW_V"])
        param_list = self._flat_params()

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
            E=self.E, W_O=self.W_O, W_out=self.W_out, b_out=self.b_out,
            W_Q=self.W_Q, W_K=self.W_K, W_V=self.W_V,
            d_model=self.d_model, n_heads=self.n_heads,
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
                    n_heads=ckpt["n_heads"], n_classes=ckpt["n_classes"])
        model.E     = ckpt["E"]
        model.W_O   = ckpt["W_O"]
        model.W_out = ckpt["W_out"]
        model.b_out = ckpt["b_out"]
        model.W_Q   = ckpt["W_Q"]
        model.W_K   = ckpt["W_K"]
        model.W_V   = ckpt["W_V"]
        return model, ckpt

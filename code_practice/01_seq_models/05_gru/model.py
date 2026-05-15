"""
Session 5 — GRU Cell from Scratch
model.py: VanillaGRU — reset gate, update gate, candidate, BPTT.

Forward:
    z_t = [h_{t-1}; x_t]
    r_t = σ(W_r·z_t + b_r)           reset gate
    u_t = σ(W_u·z_t + b_u)           update gate
    z̃_t = [r_t ⊙ h_{t-1}; x_t]
    h̃_t = tanh(W_h·z̃_t + b_h)       candidate
    h_t = (1−u_t)⊙h_{t-1} + u_t⊙h̃_t
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class VanillaGRU:
    def __init__(self, vocab_size, hidden_size=128, lr=1e-2, seed=42):
        rng = np.random.default_rng(seed)
        V, H = vocab_size, hidden_size
        self.vocab_size  = V
        self.hidden_size = H
        self.lr          = lr

        scale = 0.01
        # Gate weights: [H, H+V]
        self.Wr = rng.standard_normal((H, H + V)) * scale
        self.Wu = rng.standard_normal((H, H + V)) * scale
        self.Wh = rng.standard_normal((H, H + V)) * scale
        self.br = np.zeros((H, 1))
        self.bu = np.zeros((H, 1))
        self.bh = np.zeros((H, 1))

        self.Why = rng.standard_normal((V, H)) * scale
        self.by  = np.zeros((V, 1))

        self._init_adagrad()

    def _init_adagrad(self):
        for attr in ("Wr", "Wu", "Wh", "br", "bu", "bh", "Why", "by"):
            setattr(self, "m" + attr, np.zeros_like(getattr(self, attr)))

    def forward(self, inputs, h_prev):
        """
        Returns: xs, zs, z̃s, rs, us, h_cands, hs, ps
        All stored per time-step for BPTT.
        """
        V, H = self.vocab_size, self.hidden_size
        T = len(inputs)

        xs      = {}   # one-hot inputs [V,1]
        zs      = {}   # [h;x] concat [H+V,1]
        z_tils  = {}   # [r⊙h;x] concat [H+V,1]
        rs      = {}   # reset gate [H,1]
        us      = {}   # update gate [H,1]
        h_cands = {}   # candidate [H,1]
        hs      = {-1: h_prev}
        ps      = {}

        for t in range(T):
            xs[t] = np.zeros((V, 1))
            xs[t][inputs[t]] = 1.0

            zs[t] = np.vstack((hs[t - 1], xs[t]))            # [H+V, 1]

            rs[t] = sigmoid(self.Wr @ zs[t] + self.br)       # [H, 1]
            us[t] = sigmoid(self.Wu @ zs[t] + self.bu)       # [H, 1]

            z_tils[t] = np.vstack((rs[t] * hs[t - 1], xs[t]))  # [H+V, 1]
            h_cands[t] = np.tanh(self.Wh @ z_tils[t] + self.bh)  # [H, 1]

            hs[t] = (1.0 - us[t]) * hs[t - 1] + us[t] * h_cands[t]  # [H, 1]

            y = self.Why @ hs[t] + self.by
            p = np.exp(y - y.max())
            ps[t] = p / p.sum()

        return xs, zs, z_tils, rs, us, h_cands, hs, ps

    def backward(self, inputs, targets, h_prev):
        xs, zs, z_tils, rs, us, h_cands, hs, ps = self.forward(inputs, h_prev)
        T = len(inputs)
        V, H = self.vocab_size, self.hidden_size

        dWr  = np.zeros_like(self.Wr)
        dWu  = np.zeros_like(self.Wu)
        dWh  = np.zeros_like(self.Wh)
        dbr  = np.zeros_like(self.br)
        dbu  = np.zeros_like(self.bu)
        dbh  = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby  = np.zeros_like(self.by)

        dh_next = np.zeros((H, 1))
        loss = 0.0

        for t in reversed(range(T)):
            loss += -np.log(ps[t][targets[t], 0] + 1e-8)

            dy = ps[t].copy()
            dy[targets[t]] -= 1.0

            dWhy += dy @ hs[t].T
            dby  += dy

            dh = self.Why.T @ dy + dh_next    # [H, 1]

            # h_t = (1-u_t)*h_{t-1} + u_t*h̃_t
            du_raw  = dh * (h_cands[t] - hs[t - 1])   # pre-sigmoid
            dh_cand = dh * us[t]                       # [H, 1]
            dh_prev_direct = dh * (1.0 - us[t])        # from (1-u)*h_{t-1}

            # Backprop through h̃_t = tanh(Wh·z̃_t + bh)
            dh_cand_raw = dh_cand * (1.0 - h_cands[t] ** 2)   # pre-tanh
            dWh += dh_cand_raw @ z_tils[t].T
            dbh += dh_cand_raw
            dz_til = self.Wh.T @ dh_cand_raw              # [H+V, 1]

            # z̃_t = [r_t⊙h_{t-1}; x_t]
            dh_from_reset = dz_til[:H] * rs[t]            # gradient to h_{t-1} via reset
            dr_raw = dz_til[:H] * hs[t - 1]              # pre-sigmoid for reset

            # Backprop through update gate
            du = du_raw * us[t] * (1.0 - us[t])
            dWu += du @ zs[t].T
            dbu += du
            dzu = self.Wu.T @ du

            # Backprop through reset gate
            dr = dr_raw * rs[t] * (1.0 - rs[t])
            dWr += dr @ zs[t].T
            dbr += dr
            dzr = self.Wr.T @ dr

            # Sum all gradients flowing back to h_{t-1}
            dh_next = (dh_prev_direct
                       + dh_from_reset
                       + dzu[:H]
                       + dzr[:H])

        for d in (dWr, dWu, dWh, dbr, dbu, dbh, dWhy, dby):
            np.clip(d, -5, 5, out=d)

        grads = dict(Wr=dWr, Wu=dWu, Wh=dWh,
                     br=dbr, bu=dbu, bh=dbh,
                     Why=dWhy, by=dby)
        return loss, hs[T - 1], grads

    def update(self, grads):
        eps = 1e-8
        for name, grad in grads.items():
            mem = getattr(self, "m" + name)
            mem += grad ** 2
            setattr(self, name, getattr(self, name) - self.lr * grad / (np.sqrt(mem) + eps))

    def sample(self, seed_idx, length, temperature=1.0):
        H = self.hidden_size
        h = np.zeros((H, 1))
        idx = seed_idx
        result = [idx]

        for _ in range(length - 1):
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1.0
            z   = np.vstack((h, x))
            r   = sigmoid(self.Wr @ z + self.br)
            u   = sigmoid(self.Wu @ z + self.bu)
            z_t = np.vstack((r * h, x))
            h_c = np.tanh(self.Wh @ z_t + self.bh)
            h   = (1.0 - u) * h + u * h_c

            y = self.Why @ h + self.by
            y = y / temperature
            p = np.exp(y - y.max())
            p = p / p.sum()
            idx = int(np.random.choice(self.vocab_size, p=p.ravel()))
            result.append(idx)

        return result

    def save(self, path):
        np.savez(path,
                 vocab_size=self.vocab_size,
                 hidden_size=self.hidden_size,
                 lr=self.lr,
                 Wr=self.Wr, Wu=self.Wu, Wh=self.Wh,
                 br=self.br, bu=self.bu, bh=self.bh,
                 Why=self.Why, by=self.by)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        model = cls(int(d["vocab_size"]), int(d["hidden_size"]), float(d["lr"]))
        for attr in ("Wr", "Wu", "Wh", "br", "bu", "bh", "Why", "by"):
            setattr(model, attr, d[attr])
        return model

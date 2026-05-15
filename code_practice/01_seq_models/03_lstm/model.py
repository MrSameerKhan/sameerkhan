"""
Session 3 — LSTM Cell from Scratch
model.py: VanillaLSTM — all 4 gates, BPTT through cell and hidden state streams.

Forward:
    z = [h_{t-1}; x_t]           concatenated input
    f_t = σ(W_f·z + b_f)         forget gate
    i_t = σ(W_i·z + b_i)         input gate
    o_t = σ(W_o·z + b_o)         output gate
    g_t = tanh(W_g·z + b_g)      cell candidate
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)
    y_t = W_hy·h_t + b_y
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class VanillaLSTM:
    def __init__(self, vocab_size, hidden_size=128, lr=1e-2, seed=42):
        rng = np.random.default_rng(seed)
        V, H = vocab_size, hidden_size
        self.vocab_size  = V
        self.hidden_size = H
        self.lr          = lr

        scale = 0.01
        # Gate weights: [H, H+V], biases: [H, 1]
        self.Wf = rng.standard_normal((H, H + V)) * scale
        self.Wi = rng.standard_normal((H, H + V)) * scale
        self.Wo = rng.standard_normal((H, H + V)) * scale
        self.Wg = rng.standard_normal((H, H + V)) * scale
        self.bf = np.zeros((H, 1))
        self.bi = np.zeros((H, 1))
        self.bo = np.zeros((H, 1))
        self.bg = np.zeros((H, 1))

        self.Why = rng.standard_normal((V, H)) * scale
        self.by  = np.zeros((V, 1))

        # Adagrad memory
        self._init_adagrad()

    def _init_adagrad(self):
        for attr in ("Wf", "Wi", "Wo", "Wg", "bf", "bi", "bo", "bg", "Why", "by"):
            setattr(self, "m" + attr, np.zeros_like(getattr(self, attr)))

    def forward(self, inputs, h_prev, c_prev):
        """
        inputs : list of int (char indices), length T
        Returns: xs, zs, fs, is_, os, gs, cs, hs, ps — all needed for BPTT
        """
        V, H = self.vocab_size, self.hidden_size
        T = len(inputs)

        xs  = {}   # one-hot inputs
        zs  = {}   # concatenated [h; x]
        fs  = {}   # forget gates
        is_ = {}   # input gates
        os  = {}   # output gates
        gs  = {}   # cell candidates
        cs  = {-1: c_prev}
        hs  = {-1: h_prev}
        ps  = {}   # softmax probs

        loss = 0.0
        for t in range(T):
            xs[t] = np.zeros((V, 1))
            xs[t][inputs[t]] = 1.0

            zs[t] = np.vstack((hs[t - 1], xs[t]))          # [H+V, 1]

            fs[t]  = sigmoid(self.Wf @ zs[t] + self.bf)    # [H, 1]
            is_[t] = sigmoid(self.Wi @ zs[t] + self.bi)    # [H, 1]
            os[t]  = sigmoid(self.Wo @ zs[t] + self.bo)    # [H, 1]
            gs[t]  = np.tanh(self.Wg @ zs[t] + self.bg)    # [H, 1]

            cs[t] = fs[t] * cs[t - 1] + is_[t] * gs[t]    # [H, 1]
            hs[t] = os[t] * np.tanh(cs[t])                 # [H, 1]

            y = self.Why @ hs[t] + self.by                  # [V, 1]
            p = np.exp(y - y.max())
            ps[t] = p / p.sum()

        return xs, zs, fs, is_, os, gs, cs, hs, ps

    def loss(self, inputs, targets, h_prev, c_prev):
        xs, zs, fs, is_, os, gs, cs, hs, ps = self.forward(inputs, h_prev, c_prev)
        loss = sum(-np.log(ps[t][targets[t], 0] + 1e-8) for t in range(len(inputs)))
        return loss, hs[len(inputs) - 1], cs[len(inputs) - 1]

    def backward(self, inputs, targets, h_prev, c_prev):
        xs, zs, fs, is_, os, gs, cs, hs, ps = self.forward(inputs, h_prev, c_prev)
        T = len(inputs)
        V, H = self.vocab_size, self.hidden_size

        # Gradient accumulators
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWo = np.zeros_like(self.Wo)
        dWg = np.zeros_like(self.Wg)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbg = np.zeros_like(self.bg)
        dWhy = np.zeros_like(self.Why)
        dby  = np.zeros_like(self.by)

        dh_next = np.zeros((H, 1))
        dc_next = np.zeros((H, 1))

        loss = 0.0
        for t in reversed(range(T)):
            loss += -np.log(ps[t][targets[t], 0] + 1e-8)

            dy = ps[t].copy()
            dy[targets[t]] -= 1.0                      # [V, 1]

            dWhy += dy @ hs[t].T
            dby  += dy

            dh = self.Why.T @ dy + dh_next             # [H, 1]

            # Gradient through h_t = o_t ⊙ tanh(c_t)
            tanh_ct = np.tanh(cs[t])
            do_raw  = dh * tanh_ct                     # pre-sigmoid
            dc      = dh * os[t] * (1.0 - tanh_ct ** 2) + dc_next  # [H, 1]

            # Gradient through c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
            df_raw = dc * cs[t - 1]                    # pre-sigmoid
            di_raw = dc * gs[t]                        # pre-sigmoid
            dg_raw = dc * is_[t]                       # pre-tanh
            dc_next = dc * fs[t]

            # Backprop through activations
            df = df_raw * fs[t]  * (1.0 - fs[t])      # sigmoid derivative
            di = di_raw * is_[t] * (1.0 - is_[t])
            do = do_raw * os[t]  * (1.0 - os[t])
            dg = dg_raw * (1.0 - gs[t] ** 2)          # tanh derivative

            dWf += df @ zs[t].T;  dbf += df
            dWi += di @ zs[t].T;  dbi += di
            dWo += do @ zs[t].T;  dbo += do
            dWg += dg @ zs[t].T;  dbg += dg

            dz = (self.Wf.T @ df + self.Wi.T @ di +
                  self.Wo.T @ do + self.Wg.T @ dg)
            dh_next = dz[:H]

        # Gradient clipping
        for d in (dWf, dWi, dWo, dWg, dbf, dbi, dbo, dbg, dWhy, dby):
            np.clip(d, -5, 5, out=d)

        grads = dict(Wf=dWf, Wi=dWi, Wo=dWo, Wg=dWg,
                     bf=dbf, bi=dbi, bo=dbo, bg=dbg,
                     Why=dWhy, by=dby)

        return loss, hs[T - 1], cs[T - 1], grads

    def update(self, grads):
        eps = 1e-8
        for name, grad in grads.items():
            mem = getattr(self, "m" + name)
            mem += grad ** 2
            setattr(self, name, getattr(self, name) - self.lr * grad / (np.sqrt(mem) + eps))

    def sample(self, seed_idx, length, temperature=1.0):
        H = self.hidden_size
        h = np.zeros((H, 1))
        c = np.zeros((H, 1))
        idx = seed_idx
        result = [idx]

        for _ in range(length - 1):
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1.0
            z = np.vstack((h, x))

            f  = sigmoid(self.Wf @ z + self.bf)
            i_ = sigmoid(self.Wi @ z + self.bi)
            o  = sigmoid(self.Wo @ z + self.bo)
            g  = np.tanh(self.Wg @ z + self.bg)
            c  = f * c + i_ * g
            h  = o * np.tanh(c)

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
                 Wf=self.Wf, Wi=self.Wi, Wo=self.Wo, Wg=self.Wg,
                 bf=self.bf, bi=self.bi, bo=self.bo, bg=self.bg,
                 Why=self.Why, by=self.by)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        model = cls(int(d["vocab_size"]), int(d["hidden_size"]), float(d["lr"]))
        for attr in ("Wf", "Wi", "Wo", "Wg", "bf", "bi", "bo", "bg", "Why", "by"):
            setattr(model, attr, d[attr])
        return model

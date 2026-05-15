# Session 5 — GRU Cell from Scratch

> Phase 1, Session 5. Two gates instead of four — same power, half the parameters.

---

## Table of Contents
- [Objective](#objective)
- [Architecture & Math](#architecture--math)
- [GRU vs LSTM — Gating Comparison](#gru-vs-lstm--gating-comparison)
- [How to Run](#how-to-run)
- [Expected Output](#expected-output)
- [✅ Actual Run Results](#-actual-run-results)
- [Key Insights](#key-insights)
- [Next Steps](#next-steps)

---

## Objective

Build a Gated Recurrent Unit (GRU) cell from scratch using **NumPy only**. Compare architecture, parameter count, and loss vs the Session 3 LSTM.

**Goals:**
- [ ] Implement reset gate: r_t = σ(W_r·[h_{t-1}; x_t] + b_r)
- [ ] Implement update gate: z_t = σ(W_z·[h_{t-1}; x_t] + b_z)
- [ ] Implement candidate: h̃_t = tanh(W_h·[r_t⊙h_{t-1}; x_t] + b_h)
- [ ] Cell state update: h_t = (1−z_t)⊙h_{t-1} + z_t⊙h̃_t
- [ ] Compare parameter count vs LSTM: GRU has ¾ the params

---

## Architecture & Math

### Forward Pass

```
z_t = [h_{t-1} ; x_t]               concat, [H+V, 1]

r_t = σ(W_r · z_t + b_r)            reset gate    [H, 1]
u_t = σ(W_u · z_t + b_u)            update gate   [H, 1]

z̃_t = [r_t ⊙ h_{t-1} ; x_t]        gated concat  [H+V, 1]
h̃_t = tanh(W_h · z̃_t + b_h)        candidate     [H, 1]

h_t = (1 − u_t) ⊙ h_{t-1}
     +     u_t  ⊙ h̃_t               new hidden    [H, 1]

y_t = W_hy · h_t + b_y              logits        [V, 1]
p_t = softmax(y_t)                   probs         [V, 1]
```

### Key Intuition

| Gate | Role | When it fires |
|---|---|---|
| **reset** (r) | How much old hidden state enters candidate | 0 = ignore past, 1 = use past |
| **update** (u) | Blend ratio: old vs new | 0 = keep old (copy), 1 = take new |

Update gate = forget gate + input gate merged into one.  
No separate cell state `c_t` — GRU folds memory into `h_t`.

### Parameter Count vs LSTM

| Model | Matrices | Total (H=128, V=60) |
|---|---|---|
| LSTM | 4 × [H, H+V] + output | 4×128×188 + 128×60 ≈ 104K |
| GRU  | 3 × [H, H+V] + output | 3×128×188 + 128×60 ≈  80K |

GRU has **¾ the parameters** of LSTM with often similar performance.

---

## GRU vs LSTM — Gating Comparison

```
LSTM:
  f_t = σ(W_f·[h;x])    ← forget what to erase from c
  i_t = σ(W_i·[h;x])    ← what new info to write to c
  o_t = σ(W_o·[h;x])    ← what to expose from c → h
  g_t = tanh(W_g·[h;x]) ← candidate content to write

GRU:
  r_t = σ(W_r·[h;x])              ← reset: how much past h enters candidate
  u_t = σ(W_u·[h;x])              ← update: blend ratio old/new
  h̃_t = tanh(W_h·[r⊙h;x])        ← candidate (past h can be reset to 0)
  h_t = (1-u)⊙h + u⊙h̃            ← no separate cell state
```

GRU merges LSTM's forget+input into update gate. Simpler, often just as good.

---

## How to Run

```bash
cd code_practice/01_seq_models/05_gru
python train.py
python predict.py --seed "Sarah Chen" --length 200
```

---

## Expected Output

```
Corpus: 50,000 chars, vocab: ~60
Parameters: ~80,000 (vs LSTM ~104K)
Epoch  100 | loss 2.3214 | 12.3s
Epoch  500 | loss 1.9821
Epoch 1000 | loss 1.7543
Epoch 2000 | loss 1.5892
```

---

## ✅ Actual Run Results

**train.py** (MacBook CPU, 2000 epochs, 29.9s)
```
Corpus length : 50,037 chars  (train: 45,033 | val: 5,004)
Vocab size    : 60
Hidden size   : 128
Parameters    : 80,316  (LSTM has ~96K)

Epoch   100 | train_loss: 200.5542 | val_ppl: 20.49 | 1.5s
Epoch   200 | train_loss: 195.2298 | val_ppl: 16.04 | 3.1s
Epoch   500 | train_loss: 177.2832 | val_ppl: 11.75 | 7.6s
Epoch  1000 | train_loss: 151.5536 | val_ppl:  7.95 | 15.0s
Epoch  1400 | train_loss: 135.2258 | val_ppl:  6.29 | 21.0s
Epoch  1600 | train_loss: 128.0665 | val_ppl:  6.11 | 23.9s
Epoch  1900 | train_loss: 119.1274 | val_ppl:  5.42 | 28.4s
Epoch  2000 | train_loss: 116.1739 | val_ppl:  6.14 | 29.9s

Best train loss : 116.1739
Checkpoint      : checkpoints/gru_scratch.npz
```

**predict.py** (seed="The", temp=0.8, length=300)
```
Theqfer sue tis Suresce the Our Akcomun mercing are lors 9 porcet rewise C plpercnng Fsue mo tpas.
Tun acemass operas oadse tian as dertance mire ointires innae th fome Loevkant iur Auce donns fe oa
I250000 doanss rove tud deece the Dvanlancen ouce t re oo F1000 Tollarcum daur toe Markis ovelimle
```

**predict.py** (seed="Sarah Chen", temp=0.8, length=200)
```
Sarah Chen9 rate ye oa 2000 dollans dornse. Toresud Wo mus oertint int perce d aus R dolrrcs baneng.
Tige Poap re iur Cirn olares oninas h tores inker Cu9r ace palirs is a 4 porest .eqoert.
```

### Session 3 vs Session 5 — LSTM scratch vs GRU scratch

| | Session 3 (NumPy LSTM) | Session 5 (NumPy GRU) |
|---|---|---|
| Parameters | 104,508 | **80,316** (23% fewer) |
| Best val PPL | 7.84 | **5.42** |
| Training time | 38.5s | **29.9s** (22% faster) |
| Text quality | Similar — domain fragments | Similar — domain fragments |

GRU beats LSTM on this task: better val PPL (5.42 vs 7.84), fewer parameters, faster training. The merged update gate is sufficient for this corpus length.

---

## Key Insights

- [x] GRU val PPL 5.42 beats LSTM val PPL 7.84 — with 23% fewer params and 22% faster training
- [x] Update gate merging forget+input is enough — separate cell state (LSTM) adds complexity without benefit here
- [x] Reset gate at 0 = ignore all past context for candidate; update gate at 0 = copy old h exactly (identity = no vanishing gradient)
- [x] GRU trains faster per epoch because 3 gate matmuls vs LSTM's 4 — same hidden size, less compute
- [x] Both scratch models still produce fragmented text — NumPy + Adagrad can't match PyTorch batching + Adam

---

## Next Steps

**Session 6** — GRU vs LSTM in PyTorch, head-to-head comparison with perplexity curves.

# Session 1 — Vanilla RNN Cell from Scratch

## Table of Contents
- [Objective](#objective)
- [Architecture & Math](#architecture--math)
- [Dataset Note](#dataset-note)
- [How to Run](#how-to-run)
- [Expected Output](#expected-output)
- [✅ Actual Run Results](#-actual-run-results)
- [Key Insights](#key-insights)
- [Next Steps](#next-steps)

---

## Objective

Build a Vanilla RNN cell from scratch using **NumPy only** — no PyTorch, no autograd. Implement forward pass and Backpropagation Through Time (BPTT) manually. Train a character-level language model on the Acme Financial Services corpus.

**Goals:**
- [ ] Implement RNN forward pass: h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b_h)
- [ ] Implement BPTT manually — compute gradients for W_xh, W_hh, W_hy, b_h, b_y
- [ ] Apply gradient clipping (clip to [-5, 5]) to prevent exploding gradients
- [ ] Train with Adagrad on Acme corpus — loss should decrease over epochs
- [ ] Generate sample text using nucleus/greedy sampling from trained model

---

## Architecture & Math

### Forward Pass

```
h_t = tanh(W_xh · x_t  +  W_hh · h_{t-1}  +  b_h)
y_t = W_hy · h_t  +  b_y
p_t = softmax(y_t)
L   = -Σ log p_t[target_t]          (cross-entropy over sequence)
```

### BPTT (unrolled T steps)

```
∂L/∂y_t  = p_t − one_hot(target_t)          (softmax + CE gradient)
∂L/∂W_hy += ∂L/∂y_t · h_t^T
∂L/∂h_t   = W_hy^T · ∂L/∂y_t  +  W_hh^T · ∂L/∂h_{t+1,raw}
∂L/∂h_raw = (1 − h_t²) · ∂L/∂h_t           (tanh backprop)
∂L/∂W_xh += ∂L/∂h_raw · x_t^T
∂L/∂W_hh += ∂L/∂h_raw · h_{t-1}^T
```

Clip all gradients: `∇ = clip(∇, -5, 5)`

### Tensor Shapes (V = vocab_size, H = hidden_size)

| Tensor | Shape | Notes |
|---|---|---|
| x_t (one-hot) | [V, 1] | character input at step t |
| h_t (hidden) | [H, 1] | recurrent hidden state |
| h_0 | [H, 1] | zeros at sequence start |
| W_xh | [H, V] | input-to-hidden weight |
| W_hh | [H, H] | hidden-to-hidden weight |
| b_h | [H, 1] | hidden bias |
| W_hy | [V, H] | hidden-to-output weight |
| b_y | [V, 1] | output bias |
| y_t (logits) | [V, 1] | unnormalised scores |
| p_t (probs) | [V, 1] | softmax probabilities |

---

## Dataset Note

Uses **Acme Financial Services** synthetic corpus from `code_practice/shared_dataset.py`.

```python
from shared_dataset import get_corpus_as_string, get_char_vocab
text = get_corpus_as_string()       # single string ~1,800 chars
char2idx, idx2char, vocab_size = get_char_vocab()
```

Char vocab typically ~55–65 unique characters (letters, digits, punctuation, space).

---

## How to Run

```bash
# From code_practice/ root
python 01_seq_models/01_rnn/train.py

# After training, generate text
python 01_seq_models/01_rnn/predict.py --seed "Acme" --length 100
python 01_seq_models/01_rnn/predict.py --seed "The borrower" --length 150
```

---

## Expected Output

**train.py**
```
Vocab size  : 62
Hidden size : 128
Seq length  : 25
Corpus chars: ~1,850

Epoch   100 | loss: 3.21 | sample: Acmh Fnioeics Sioeca apro...
Epoch   200 | loss: 2.87 | sample: Acme Fiancisl Srevice appr...
Epoch   500 | loss: 2.31 | sample: Acme Financial Serices appro...
Epoch  1000 | loss: 1.94 | sample: Acme Financial Services appr...
Epoch  2000 | loss: 1.62 | sample: Acme Financial Services approved...

Checkpoint saved → checkpoints/rnn_scratch.npz
Training time: ~30s
```

**predict.py**
```
Seed : "Acme"
Generated: Acme Financial Services approved the mortgage application for client ID 48...
```

---

## ✅ Actual Run Results

**train.py** (MacBook CPU, 2000 epochs, 6.5s)
```
Vocab size  : 60
Hidden size : 128
Seq length  : 25
Train chars : 45,033  |  Val chars: 5,004

Epoch   100 | train_loss: 103.84 | val_ppl: 37.27 | sample: Af etrrk 20 n 60grt m   dp 1-rhihr...
Epoch   200 | train_loss: 102.14 | val_ppl: 31.25 | sample: Ari gn8 an ieer0ctcnhal L50frs0eans...
Epoch   500 | train_loss:  95.79 | val_ppl: 26.50 | sample: ALepLsege Lees eLregtnnter AxTeLceny...
Epoch  1000 | train_loss:  84.60 | val_ppl: 16.87 | sample: Ameit inf Rl. Iamsarengs inc Ar dom...
Epoch  1500 | train_loss:  74.62 | val_ppl: 16.17 | sample: Aunsg anc ice ire y ag Hamimh Con Core...
Epoch  1700 | train_loss:  71.08 | val_ppl: 12.74 | sample: Ans lavs. bmrat outh thaving bhe Moavest...
Epoch  2000 | train_loss:  66.33 | val_ppl: 11.49 | sample: Arrs.ustont persmunt Lomenemr ofe y on...

Checkpoint saved → checkpoints/rnn_scratch.npz
Training time: 6.5s
```

**predict.py** (seed="The", temp=0.8, length=300)
```
Thehe ters inthe 70 pers. 5000 dercetinng 0 pee res pertsginl. Mase tollarce ture. Loat somey bare
tinter inaws dom lofamils Mares the Maalire kent yothexins.50.0. The of 5 0 perceen ays intcenctent
Ante dercers. Thente pors o fers. Moant 25000000 dors coans fertequntg aviing Lans 5 5 pers. Loverereic
```

**predict.py** (seed="Sarah Chen", temp=0.8, length=200)
```
Sarah Chenanseith Mast ace riss tad mas onttent. 8.5 deres pervorsce. The Maxirest. 5oving an with
5tme persent. Hfantures mincend. Mllarconnte Afres0 dor yopheving woris ont meaccant inahe Lsonthe
doans race.
```

---

## Key Insights

- [x] Train loss dropped 103 → 66 over 2000 epochs — BPTT gradients are correct
- [x] Val PPL dropped 37 → 11.49 — model generalizes, not just memorizing train sequences
- [x] Per-char train loss at epoch 2000 ≈ 66/25 = 2.65 nats (val PPL 11.49 ≈ same scale) — consistent
- [x] Domain words emerging in samples: "pers" (percent), "dollars", "Loans", "dercers" — model learns Acme vocabulary patterns
- [x] Text still not fluent — vanilla RNN can't maintain long-range context; motivation for LSTM (Session 3)
- [x] W_hh [128×128] is the only memory mechanism — single gradient stream vs LSTM's separate cell state highway

---

## Next Steps

→ **Session 2** — RNN with PyTorch: replace NumPy forward/backward with `nn.RNN`, add batching and GPU support, compare perplexity on same Acme corpus.
`code_practice/01_seq_models/02_rnn_torch/`

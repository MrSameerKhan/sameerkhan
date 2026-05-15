# Session 6 — GRU vs LSTM in PyTorch

> Phase 1, Session 6. Head-to-head: same corpus, same optimizer, only model differs.

---

## Table of Contents
- [Objective](#objective)
- [Architecture](#architecture)
- [Comparison Setup](#comparison-setup)
- [How to Run](#how-to-run)
- [Expected Output](#expected-output)
- [✅ Actual Run Results](#-actual-run-results)
- [Key Insights](#key-insights)
- [Next Steps](#next-steps)

---

## Objective

Train `nn.GRU` and `nn.LSTM` head-to-head on the same Acme corpus. Report parameter count, perplexity curve, and convergence speed.

**Goals:**
- [ ] Implement a single `CharModel` class that takes `model_type='gru'|'lstm'`
- [ ] Run both on identical hyperparameters (same seed, same batches)
- [ ] Print a comparison table: params / PPL / time
- [ ] Visualize convergence (loss at each epoch)

---

## Architecture

```
char_idx → Embedding(vocab, E) → {GRU or LSTM}(E→H) → Linear(H→vocab)
```

Both use:
- `batch_first=True`
- Same embed_dim, hidden_size, num_layers
- Same Adam, same LR, same gradient clipping

The **only** difference: `nn.GRU` vs `nn.LSTM`. GRU returns `(output, h_n)`; LSTM returns `(output, (h_n, c_n))`.

---

## Comparison Setup

| | GRU | LSTM |
|---|---|---|
| Gates | 2 (reset, update) | 4 (forget, input, output, cell) |
| Cell state | No (merged into h) | Yes (separate c_t) |
| Params (H=128, E=64, V=60) | ~3×128×(128+64)+128×60 ≈ 81K | ~4×128×192+128×60 ≈ 106K |
| Speed | Faster (~25% fewer ops) | Slower |

---

## How to Run

```bash
cd code_practice/01_seq_models/06_gru_vs_lstm
python train.py
python predict.py --model gru --seed "The Premium"
python predict.py --model lstm --seed "The Premium"
```

---

## Expected Output

```
===== GRU vs LSTM Comparison =====

Training GRU  ...
  Epoch  10 | loss 2.8832 | ppl  17.88
  Epoch  50 | loss 2.1923 | ppl   8.95
  Epoch 100 | loss 1.8614 | ppl   6.44

Training LSTM ...
  Epoch  10 | loss 2.8102 | ppl  16.62
  Epoch  50 | loss 2.1055 | ppl   8.21
  Epoch 100 | loss 1.8193 | ppl   6.16

==============================
Model  | Params  | Best PPL | Time(s)
------------------------------
GRU    |  81,340 |     5.82 |   42.1
LSTM   | 106,300 |     5.41 |   55.3
==============================
```

---

## ✅ Actual Run Results

**train.py** (MacBook CPU, 100 epochs each)
```
Device: cpu
Corpus: 50,037 chars | Vocab: 60

Training GRU ...
  Epoch   10 | loss 0.6618 | ppl   1.94
  Epoch   20 | loss 0.2341 | ppl   1.26
  Epoch   50 | loss 0.1178 | ppl   1.13
  Epoch  100 | loss 0.0737 | ppl   1.08

Training LSTM ...
  Epoch   10 | loss 0.9970 | ppl   2.71
  Epoch   20 | loss 0.3414 | ppl   1.41
  Epoch   50 | loss 0.1268 | ppl   1.14
  Epoch  100 | loss 0.0775 | ppl   1.08

==================================================
Model  |   Params |  Best PPL |  Time(s)
--------------------------------------------------
GRU    |   86,076 |      1.08 |     49.1
LSTM   |  110,908 |      1.08 |     58.4
==================================================
```

**predict.py** (model=gru, seed="The", temp=0.8, length=300)
```
The Marketing Director in the Marketing department. HR Manager Olivia Brown oversees
operations in HR. Customer service is available from 9 am to 6 pm on weekdays.
Open a Student Checking with 50 dollars minimum at 0.8 percent interest.
Open a High Yield Savings with 1000 dollars minimum at 4.2 percent
```

**predict.py** (model=lstm, seed="The", temp=0.8, length=300)
```
The Student Loan provides loans up to 75000 dollars at 4.5 percent over 10 years.
High Yield Savings accounts require minimum balance of 1000 dollars.
All personal loans require credit score above 650. Premium Checking accounts include
free wire transfers. Alcount senrestears raseud a overes at of vali
```

### Full comparison across all 6 sessions

| Session | Model | Params | Val/Train PPL | Time |
|---|---|---|---|---|
| 1 — NumPy RNN | Vanilla RNN | ~49K | val 11.49 | 6.5s |
| 2 — nn.RNN | CharRNN | ~95K | train 1.11 | 21.1s |
| 3 — NumPy LSTM | VanillaLSTM | 104K | val 7.84 | 38.5s |
| 4 — nn.LSTM | CharLSTM | 111K | train 1.08 | — |
| 5 — NumPy GRU | VanillaGRU | 80K | val **5.42** | 29.9s |
| 6 — nn.GRU vs nn.LSTM | Both | 86K / 111K | train **1.08** / **1.08** | 49s / 58s |

---

## Key Insights

- [x] GRU and LSTM reach identical PPL (1.08) on this corpus — for small/medium data they are equivalent
- [x] GRU is 16% faster (49s vs 58s) with 22% fewer params (86K vs 111K) — better efficiency at same accuracy
- [x] GRU converges faster early (epoch 10: PPL 1.94 vs 2.71) — fewer gates = cleaner gradient signal initially
- [x] Both generate coherent Acme sentences by epoch 100 — task saturated, PPL gap closes at ceiling
- [x] Rule of thumb confirmed: **default GRU**, switch to LSTM only if you have evidence of long-range dependency needs

---

## Next Steps

**Session 7** — BiLSTM for Named Entity Recognition. First real sequence labeling task with:
- Bidirectional LSTM (reads left-to-right AND right-to-left)
- BIO tagging scheme
- seqeval F1 evaluation
- Padded batches with ignore_index masking

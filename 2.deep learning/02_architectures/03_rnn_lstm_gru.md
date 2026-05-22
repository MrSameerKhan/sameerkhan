# 03 — RNN / LSTM / GRU

## Quick Reference (30-sec scan)

- **RNN:** processes sequences step-by-step, passes hidden state forward — but vanishing gradients kill long-range memory
- **LSTM:** solves vanishing gradient via gates (forget, input, output) + separate cell state
- **GRU:** simplified LSTM — 2 gates instead of 3, fewer params, similar performance
- **Why they lost to Transformers:** sequential processing can't parallelize — 10× slower to train
- **Still used:** lightweight sequential tasks, streaming inference, edge devices, time series
- **Gotcha:** LSTM hidden state is not the same as cell state — `h_t` is filtered output, `c_t` is the memory

---

## Why Sequences Need Special Architecture

MLP and CNN have no memory — each input is processed independently.

For sequences, order matters and context from previous steps is needed:

```
"The cat sat on the mat because it was tired"
                                    ↑
To predict "it" refers to "cat", need memory of 7 steps back
```

RNNs solve this by passing a **hidden state** from step to step.

---

## 1. RNN (Recurrent Neural Network)

### How It Works

```
h_t = tanh(W_h × h_{t-1} + W_x × x_t + b)
y_t = W_y × h_t + b_y
```

At each step: combine previous hidden state + current input → new hidden state.

```
x1 → [RNN] → h1 → [RNN] → h2 → [RNN] → h3 → output
               ↑                  ↑
         (reuse same weights at every step)
```

**Shared weights across time steps** — same W_h, W_x used at every position.

### Unrolling Through Time

```
Step 1: h1 = tanh(W_h × h0 + W_x × x1)   h0 = zeros (init)
Step 2: h2 = tanh(W_h × h1 + W_x × x2)
Step 3: h3 = tanh(W_h × h2 + W_x × x3)
...
Step T: hT = final hidden state = "summary" of entire sequence
```

### The Vanishing Gradient Problem in RNNs

Gradient from step T back to step 1 passes through T multiplications of W_h:

```
∂h1/∂hT = W_h^T × (activation derivatives)
```

- If `||W_h|| < 1`: gradient shrinks exponentially — early steps receive ~0 gradient — no long-range learning
- If `||W_h|| > 1`: gradient explodes — use gradient clipping

**Practical consequence:** vanilla RNNs can reliably use ~10-20 steps of context. Beyond that, early context is forgotten.

---

## 2. LSTM (Long Short-Term Memory)

### The Key Innovation: Two Separate Streams

```
c_t = cell state   = long-term memory (conveyor belt, protected)
h_t = hidden state = short-term / filtered output
```

The cell state carries information across many steps with minimal transformation — enabling gradient flow over long sequences.

### The Three Gates

Gates are sigmoid outputs (0 to 1) — act as valves controlling information flow.

```
Forget gate:  f_t = σ(W_f × [h_{t-1}, x_t] + b_f)
              → what to erase from cell state? (0=forget, 1=keep)

Input gate:   i_t = σ(W_i × [h_{t-1}, x_t] + b_i)
              → how much new info to write to cell state?

Candidate:    g_t = tanh(W_g × [h_{t-1}, x_t] + b_g)
              → what new info is available to write?

Output gate:  o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
              → what to expose from cell state as output?
```

### Cell and Hidden State Updates

```python
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t   # update cell state
h_t = o_t ⊙ tanh(c_t)              # compute hidden state

⊙ = element-wise multiplication
```

**Intuition:** Forget gate: "erase irrelevant past" · Input gate + Candidate: "write relevant new info" · Output gate: "decide what to expose"

### Why LSTM Solves Vanishing Gradients

The cell state `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t` uses **addition**, not multiplication of W matrices.

Gradient through cell state:

```
∂c_t/∂c_{t-1} = f_t   ← just the forget gate value
```

When forget gate = 1 (keep everything), gradient flows back nearly unchanged across many steps. No repeated weight matrix multiplication killing the gradient.

---

## 3. GRU (Gated Recurrent Unit)

Simplified LSTM — merges cell state and hidden state into one, uses 2 gates instead of 3.

```
Reset gate:   r_t = σ(W_r × [h_{t-1}, x_t])
              → How much past hidden state to use for candidate?

Update gate:  z_t = σ(W_z × [h_{t-1}, x_t])
              → How much to update hidden state vs keep old?

Candidate:    h̃_t = tanh(W × [r_t ⊙ h_{t-1}, x_t])

Output:       h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
```

**When z_t = 0:** keep old hidden state (memory). **When z_t = 1:** replace with new candidate (update).

---

## RNN vs LSTM vs GRU

| | RNN | LSTM | GRU |
|--|-----|------|-----|
| Gates | None | 3 (forget, input, output) | 2 (reset, update) |
| Memory streams | 1 (h_t) | 2 (h_t + c_t) | 1 (h_t) |
| Long-range memory | Poor | Good | Good |
| Parameters | Fewest | Most | Middle |
| Training speed | Fastest | Slowest | Middle |
| Performance | Worst | Best (usually) | ≈LSTM |
| Use today | Rarely | Yes (when needed) | Preferred over LSTM |

**Rule of thumb:** GRU first. Switch to LSTM only if GRU underperforms.

---

## Bidirectional RNN

Standard RNN only sees past context. Bidirectional runs two RNNs — one forward, one backward — and concatenates their hidden states.

```
→ Forward RNN:   x1 → x2 → x3 → x4
← Backward RNN:  x4 → x3 → x2 → x1
                         ↓
              concat(h_forward, h_backward) at each step
```

Sees full context (past + future) — **cannot** be used for autoregressive generation. Used in: BERT-style models, sequence labeling, named entity recognition.

---

## RNN vs Transformer vs Mamba (SSM)

| | RNN / LSTM | Transformer | Mamba / S6 (SSM) |
|--|-----------|-------------|------------------|
| Processing during training | Sequential | Fully parallel | Parallel via associative scan |
| Training speed | Slow | Fast (GPU parallel) | Fast (comparable to Transformer) |
| Inference compute per token | O(1) | O(n) (with KV cache) | O(1) |
| Inference memory | O(1) hidden state | O(n) KV cache | O(1) compressed state |
| Long-range memory | Degrades with distance | Direct via attention | Selective state, content-dependent |
| Streaming inference | ✓ Natural | ✗ Needs full context | ✓ Natural |
| Very long sequences | ⚠ Memory fades | ⚠ O(n²) compute | ✓ Linear compute |
| Associative recall | ✗ Weak | ✓ Strong | ⚠ Weaker than Transformer |
| Mature ecosystem | ▲ Legacy | ✓ Dominant | ▲ Emerging |

**Why RNNs lost (2017-2023):** transformers train 10-100× faster on modern GPUs. The sequential constraint is the killer — you can't parallelize step-by-step computation. RNN production usage today is mostly streaming inference (ASR, sensor data).

**Why Mamba is interesting (2024+):** state-space models recover RNN-style O(1) inference and streaming, while being far faster and more memory-efficient at long context. The weak spot is **associative recall** — finding a specific previously-seen token in context — where attention still wins. Hybrid models (Jamba, Zamba, Samba) interleave Mamba and Transformer layers to get the best of both.

See deeper treatment in `../01_fundamentals/05_modern_components.md` and the transformer file `04_transformer.md`.

---

## When to Still Use RNN/LSTM/GRU

| Situation | Use RNN? | Why |
|-----------|---------|-----|
| Streaming / real-time inference | ✓ Yes | Fixed hidden state, O(1) per step |
| Edge / mobile deployment | ✓ Yes | Fewer params than transformer |
| Time series (short sequences) | ✓ Yes | Simple, effective for <100 steps |
| Very long sequences (1000+) | ⚠ Partial | LSTM still viable with attention |
| NLP general purpose | ✗ No | Transformer dominates |
| Document understanding | ✗ No | LayoutLM / BERT better |

---

## Gotchas

**1. h_t ≠ c_t in LSTM.** `h_t` is the filtered output — passed to the next layer and used for prediction. `c_t` is the internal cell state — the "memory conveyor belt" not directly used for output. Many implementations return `(h_t, c_t)` as a tuple. Confusing them causes shape errors.

**2. Gradient clipping is mandatory for RNNs.** Exploding gradients happen frequently in RNNs even with LSTM. Always use `clip_grad_norm_(params, 1.0)` when training any RNN architecture.

**3. Initialized hidden state matters.** Default h_0 = 0 is usually fine. But for domain-specific tasks, learned initial states can help. For inference with variable-length sequences, always reset h_0 between unrelated samples.

**4. Bidirectional LSTM doubles hidden dim.** If hidden_size=256, bidirectional output is 512 (forward 256 + backward 256). Downstream layers must account for this.

**5. Packing padded sequences is required for variable lengths.** Without `pack_padded_sequence`, the RNN processes padding tokens as real inputs, contaminating hidden states. Always pack sequences when batch has variable lengths.

---

## Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss explodes early in training | Exploding gradients | `clip_grad_norm_(params, 1.0)` |
| Model ignores early tokens | Vanishing gradients | Switch to LSTM/GRU; reduce sequence length |
| Shape mismatch in LSTM | Mixing h_t and c_t | Check `lstm(x, (h0, c0))` returns `(output, (h_n, c_n))` |
| Training very slow | Sequential RNN bottleneck | Use transformer or parallelize sequences |
| Inference differs from eval | Dropout active in eval | Call `model.eval()` |
| Poor performance on long sequences | Context too long for LSTM | Add attention over LSTM states |

---

## Code Reference

```python
import torch
import torch.nn as nn

# Basic LSTM
lstm = nn.LSTM(
    input_size=256,       # embedding dim
    hidden_size=512,      # hidden state dim
    num_layers=2,         # stacked LSTMs
    batch_first=True,     # input: (batch, seq, feature)
    dropout=0.1,          # between layers (not last)
    bidirectional=False
)

# Forward pass
x = torch.rand(32, 100, 256)   # (batch=32, seq_len=100, input=256)
output, (h_n, c_n) = lstm(x)
# output: (32, 100, 512)  — all hidden states
# h_n:   (2, 32, 512)    — last hidden state (num_layers × batch × hidden)
# c_n:   (2, 32, 512)    — last cell state

# GRU (simpler — no cell state)
gru = nn.GRU(input_size=256, hidden_size=512, batch_first=True)
output, h_n = gru(x)

# Bidirectional LSTM
bilstm = nn.LSTM(input_size=256, hidden_size=256, bidirectional=True, batch_first=True)
output, (h_n, c_n) = bilstm(x)
# output: (32, 100, 512) — 256 forward + 256 backward concatenated

# Sequence classifier using LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)                # (batch, seq, embed)
        out, (h_n, _) = self.lstm(emb)
        last_hidden = h_n[-1]                  # last layer hidden state
        return self.classifier(last_hidden)

# Packing for variable-length sequences
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
output_packed, (h_n, c_n) = lstm(packed)
output, _ = pad_packed_sequence(output_packed, batch_first=True)

# Gradient clipping (always use with RNNs)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## Interview Q&A

**Q: Why does LSTM solve the vanishing gradient problem that RNN couldn't?**

RNN gradients must pass through repeated multiplications of the weight matrix W_h across time steps — these shrink exponentially if `||W_h|| < 1`. LSTM introduces a cell state `c_t` updated via addition: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`. The gradient through the cell state is `∂c_t/∂c_{t-1} = f_t` — just the forget gate value (not a matrix product). When the forget gate stays near 1, gradients flow back many steps nearly unchanged.

**Q: What is the forget gate and what happens if it's always 0 or always 1?**

The forget gate controls how much of the previous cell state to retain. If always 0, the LSTM completely erases memory every step — equivalent to a stateless model. If always 1, it never forgets — useful for accumulating information but can't selectively discard irrelevant past context. In practice, gates learn to selectively open/close based on input, which is the key to LSTM's flexibility.

**Q: When would you still choose LSTM over a transformer in 2024?**

Three scenarios: (1) Streaming/real-time inference — LSTM processes one token at a time with O(1) applications. Critical for latency-sensitive applications. (2) Edge/mobile deployment — LSTM has far fewer parameters for equivalent sequence modeling on short sequences. (3) Very long time series — structured temporal data where the inductive bias of step-by-step processing is actually helpful (sensor data, financial time series).

**Q: What is the practical difference between using the final hidden state vs all hidden states from an LSTM?**

Final hidden state (`h_n`) = compressed summary of the entire sequence — useful for sequence classification where you need one vector. All hidden states (`output`) = one vector per input token — useful for token-level tasks (NER, sequence labeling, attention over LSTM states). For document classification, final hidden state is typical. For extracting entity spans, all states with attention pooling is better.

---

## Connections

- Builds on: `../01_fundamentals/01_foundations.md` — RNN cell = MLP applied recurrently
- Builds on: `../01_fundamentals/03_training_stability.md` — vanishing gradients, gradient clipping
- Builds on: `../01_fundamentals/05_modern_components.md` — attention mechanism was first added ON TOP of LSTM (seq2seq models)
- Leads to: `04_transformer.md` — transformers replaced LSTMs by making attention the primary mechanism
- Relevant in: NLP domain — sequence labeling, legacy models, time series

---

## Key Takeaway

```
RNN:     h_t = tanh(W_h × h_{t-1} + W_x × x_t) — simple but vanishing gradients
LSTM:    adds forget/input/output gates + cell state — solves long-range memory
GRU:     simplified LSTM, 2 gates — preferred for new LSTM use cases
Lost to: Transformer — fully parallel training, global attention, scales better
Still use: streaming inference, edge deployment, short time series
```

# 02 — Architectures

## Reading Order

| File | What You'll Learn | Prerequisites |
|------|------------------|---------------|
| `00_architecture_comparison.md` | Decision tree: when to use which architecture | 01_fundamentals/ |
| `01_mlp.md` | Fully connected networks, FFN in transformers | 00_architecture_comparison |
| `02_cnn.md` | Convolution, ResNet, transfer learning | 01_mlp |
| `03_rnn_lstm_gru.md` | Sequences, gates, vanishing gradients | 01_mlp |
| `04_transformer.md` | Attention, encoder/decoder, BERT vs GPT | 03_rnn_lstm_gru, 01_fundamentals/05 |
| `05_generative.md` | VAE, GAN, Diffusion — ELBO, WGAN-GP, ControlNet | 04_transformer |
| `06_gnn.md` | Graph networks, message passing, GCN/GAT/GraphSAGE | 04_transformer |
| `07_reinforcement_learning.md` | DQN, A3C, PPO, RLHF | 04_transformer |
| `08_semi_supervised.md` | SimCLR, MAE, BERT SSL, DINOv2 | 04_transformer |
| `09_mixture_of_experts.md` | Sparse FFN, routing, Mixtral, DeepSeek-V3 | 04_transformer |
| `10_quantization_theory.md` | INT8, NF4, GPTQ, AWQ — running 70B on consumer GPU | 04_transformer |

---

## Architecture Decision Tree

```
Your task
    ├─ Tabular / structured data            → MLP
    ├─ Images / spatial data
    │       ├─ Classification / detection   → CNN (ResNet, EfficientNet)
    │       ├─ Global context needed        → ViT / hybrid CNN+Transformer
    │       └─ Generation                  → Diffusion (05_generative)
    ├─ Sequences / text
    │       ├─ Understanding (NER, classify) → BERT / encoder-only Transformer
    │       ├─ Generation                   → GPT / decoder-only Transformer
    │       ├─ Translation / summarization  → T5/BART encoder-decoder
    │       └─ Streaming / edge deploy      → LSTM / GRU
    ├─ Relational / graph data              → GNN (GCN, GAT, GraphSAGE)
    ├─ Sequential decision-making           → RL (DQN, PPO)
    └─ Limited labels, lots of unlabeled    → SSL (SimCLR, MAE, BERT pretrain)
```

---

## Quick Comparison

| Architecture | Input | Key Strength | Key Limitation |
|---|---|---|---|
| MLP | Vectors | Universal approximator | No spatial/sequential structure |
| CNN | Images | Translation invariant, efficient | Local receptive field only |
| RNN/LSTM | Sequences | O(1) inference, streaming | Vanishing gradients, sequential |
| Transformer | Sequences/patches | Global attention, parallel training | O(N²) memory, no streaming |
| GNN | Graphs | Permutation invariant, relational | Over-smoothing, memory-heavy |
| VAE | Any | Smooth latent space, anomaly detection | Blurry outputs |
| GAN | Noise→image | Sharp outputs | Mode collapse, unstable |
| Diffusion | Noise→image | Best quality, diversity | Slow inference |

---

## Connections

- **Classical ML** (the things DL replaces / augments): `../../1.machine learning/`
- **Vision applications**: `../../3.computerVision/`
- **NLP applications**: `../../4.nlp/`
- **Transformer architecture details**: `../../5.transformers/`

## Practice

- `02_transformers/02-10` — attention, MHA, RoPE, encoder/decoder, KV cache → `../../code_practice/02_transformers/`
- `01_seq_models/01-09` — RNN/LSTM/GRU/seq2seq → `../../code_practice/01_seq_models/`
- LoRA / quantization → `../../code_practice/04_llms/03_quant_4bit/`, `../../code_practice/04_llms/06_lora_scratch/`

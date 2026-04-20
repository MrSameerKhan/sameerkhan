# Transformers — Roadmap & Navigation Guide

---

## Folder Map

```
5.transformers/
├── 00_roadmap.md                              ← you are here
├── 01_fundamentals/
│   ├── 01_attention_mechanism.md              ← Q, K, V, scaled dot-product
│   ├── 02_transformer_architecture.md         ← encoder-decoder, residual, LayerNorm
│   ├── 03_tokenization.md                     ← BPE, WordPiece, SentencePiece
│   ├── 04_pretraining_objectives.md           ← MLM, CLM, span masking
│   └── 05_vision_transformers.md              ← ViT, Swin, DeiT, DINO
├── 02_models/
│   ├── 01_bert_family.md                      ← BERT, RoBERTa, DistilBERT, ALBERT
│   ├── 02_gpt_family.md                       ← GPT-2, GPT-3, GPT-4
│   ├── 03_encoder_decoder.md                  ← T5, BART, mT5
│   ├── 04_efficient_transformers.md           ← Flash Attention, KV cache, Paged Attention
│   ├── 05_bert_end_to_end.md                  ← BERT forward pass with numbers
│   ├── 06_gpt_end_to_end.md                   ← GPT autoregressive trace
│   ├── 07_t5_end_to_end.md                    ← T5 seq2seq trace
│   ├── 08_modern_llm_architecture.md          ← LLaMA vs GPT-2 architecture changes
│   ├── 09_parameter_efficient_tuning.md       ← LoRA, QLoRA, prefix tuning, adapters
│   └── 10_mixture_of_experts.md              ← Mixtral 8×7B, routing, load balancing
```

---

## Reading Order

### If you have 2 hours (interview prep)
| Order | File | What you get |
|-------|------|--------------|
| 1 | `01_fundamentals/01_attention_mechanism.md` | QKV, scaled dot-product, multi-head |
| 2 | `02_models/05_bert_end_to_end.md` | BERT forward pass with actual numbers |
| 3 | `02_models/08_modern_llm_architecture.md` | RoPE, RMSNorm, GQA, SwiGLU |
| 4 | `02_models/09_parameter_efficient_tuning.md` | LoRA / QLoRA — most common interview topic |

### If you have a full day
Read fundamentals 01→05, then models 01→10 in order.

---

## Architecture Decision Guide

```
Task type?
│
├── Text classification / NER / extraction
│   └── Encoder-only: BERT, RoBERTa → 01_bert_family.md
│
├── Text generation / chat / completion
│   └── Decoder-only: GPT, LLaMA, Mistral → 02_gpt_family.md
│
├── Translation / summarization / QA (input→output)
│   └── Encoder-decoder: T5, BART → 03_encoder_decoder.md
│
├── Image classification / vision tasks
│   └── ViT, Swin, DeiT, DINO → 01_fundamentals/05_vision_transformers.md
│
├── Fine-tuning with limited GPU
│   └── LoRA / QLoRA → 02_models/09_parameter_efficient_tuning.md
│
└── Large-scale inference efficiency
    └── Flash Attention, vLLM → 02_models/04_efficient_transformers.md
```

---

## Key Numbers Cheat Sheet

| Model | Layers | Heads | d_model | FFN | Params |
|-------|--------|-------|---------|-----|--------|
| BERT-base | 12 | 12 | 768 | 3072 | 110M |
| BERT-large | 24 | 16 | 1024 | 4096 | 340M |
| GPT-2 | 12 | 12 | 768 | 3072 | 117M |
| GPT-3 | 96 | 96 | 12288 | 49152 | 175B |
| LLaMA-2-7B | 32 | 32 | 4096 | 11008 | 7B |
| LLaMA-2-13B | 40 | 40 | 5120 | 13824 | 13B |
| Mistral-7B | 32 | 32 | 4096 | 14336 | 7B |
| Mixtral 8×7B | 32 | 32 | 4096 | 14336×8 | 46.7B |
| ViT-B/16 | 12 | 12 | 768 | 3072 | 86M |

---

## Modern LLM Architecture Improvements (vs original BERT/GPT-2)

| Change | Original | Modern (LLaMA) | Why |
|--------|----------|----------------|-----|
| Normalization | Post-LayerNorm | Pre-RMSNorm | More stable, faster |
| Position encoding | Learned absolute | RoPE (rotary) | Better length extrapolation |
| Attention | Full attention (MHA) | GQA (grouped query) | Faster inference, less KV cache |
| FFN activation | ReLU | SwiGLU | Better task performance |
| Vocabulary | 30K (BERT) | 32K–128K | Better tokenization coverage |

---

## Common Interview Topics (ranked by frequency)

1. **Attention mechanism** — Q, K, V computation, scaling by √d_k, why softmax
2. **BERT vs GPT** — encoder vs decoder, bidirectional vs causal masking
3. **LoRA** — why it works, r parameter, B init to zero, merge weights
4. **Transformer complexity** — O(n²·d) attention, how Flash Attention reduces memory
5. **Pre-training objectives** — MLM (BERT), CLM (GPT), span masking (T5)
6. **MoE** — Mixtral routing, load balancing, total vs active params
7. **KV cache** — what it caches, memory cost, Paged Attention fix
8. **RoPE** — how rotary embeddings encode relative position, length extrapolation

---

## Connections to Other Folders

| Topic | Cross-reference |
|-------|----------------|
| ViT in document AI | `7.multimodal/03_vision_transformers.md` |
| CLIP / DINO | `7.multimodal/04_clip_finetuning_end_to_end.md` |
| LLM fine-tuning (LoRA in practice) | `6.llms/07_finetuning_end_to_end.md` |
| RLHF / PPO | `6.llms/10_alignment_end_to_end.md` |
| Serving (vLLM, Flash Attention) | `8.mlops/02_serving_and_inference.md` |
| RAG (retrieval + generation) | `6.llms/08_rag_end_to_end.md` |

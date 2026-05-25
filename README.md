# sameerkhanAI

Personal ML/DL mastery repo — built for deep understanding, interview prep, and long-term reference.
Covers the full stack from classical ML to LLM systems and production MLOps.

> **Start here** → **00_HUB.md** — daily action center, current status, what to build next, resume bullets, JD coverage.
>
> Deep reference (3 files): 01_CAREER_PACK · 02_INTERVIEW_PACK · 03_LEARNING_PACK

---

## Environment Setup

Use the Conda environment file before running code in this repo.

```powershell
conda env create -f environment.yml
conda activate sameerkhan
python test.py
```

`environment.yml` is the primary, human-maintained setup file. It uses Python 3.12 and compatible version ranges for the ML, deep learning, transformer, RAG, and agent libraries used across the repo.

If a future install has dependency conflicts or package compatibility issues, recreate the environment from the exact lock snapshot:

```powershell
conda env create -f environment.lock.yml
conda activate sameerkhan
python test.py
```

Production-style convention:

- Update `environment.yml` when adding or changing top-level dependencies.
- After verifying the environment works, refresh `environment.lock.yml` with the exact working versions.
- Use `test.py` as the smoke test after environment changes.

---

## Structure

| # | Topic | What's Inside |
|---|-------|---------------|
| 1 | Machine Learning | Fundamentals, algorithms, statistics, evaluation |
| 2 | Deep Learning | DL fundamentals + universal architectures (MLP / CNN / Transformer / MoE / quantization) |
| 3 | Computer Vision | CNNs, ViT, detection, segmentation, transfer learning, self-supervised vision |
| 4 | NLP | Tokenization, embeddings, sequence models, decoding, NER, IE, NLP eval |
| 5 | Transformers | Attention mechanism, BERT/GPT/T5 families, efficient transformers, modern LLM architecture |
| 6 | LLMs | Pure LLM core: prompting, fine-tuning, alignment, evaluation, vLLM, dataset prep |
| 7 | RAG | RAG patterns, RAG pipeline depth, indirect prompt injection defenses |
| 8 | Agents | ReAct, LangGraph, memory, planner-executor, MCP, agent eval |
| 9 | Multimodal | CLIP, VLMs, Document AI (OCR, LayoutLM, Donut, ColPali, native-VLM parsing) |
| 10 | MLOps | Experiment tracking, serving, observability, LLM cost, production RAG ops |
| 11 | System Design | ML design framework, RAG, agents, multi-tenant, tool auth, LLM eval systems |

**Repo organization (4 tiers):** Tier 1 — **Personal-Org** (root files): `00_HUB.md` · `01_CAREER_PACK.md` · `02_INTERVIEW_PACK.md` · `03_LEARNING_PACK.md` · Tier 2 — **Theory** (folders 1-11 above): canonical explanations, one SSOT per concept · Tier 3 — **Practice**: `code_practice/` — 68 hands-on sessions across 6.5 phases · Tier 4 — **Audit/Meta**: `STRUCTURE.md` (SSOT map) · `RULES.md` (going-forward conventions) · `archive/` (deprecated content + completed migration plan)

---

## Coverage

**Fundamentals** — Classical ML algorithms, bias-variance, regularization · Neural network training: backprop, optimizers, weight init, batch norm · CNN fundamentals: convolution, pooling, receptive field

**Modern DL** — Transformer architecture: attention, positional encoding, Flash Attention · BERT family (MLM, DeBERTa), GPT family (LLaMA2/3, GQA, KV cache) · Efficient transformers: LoRA, QLoRA, quantization (INT8/GPTQ), MoE

**LLM Systems** — Prompting: CoT, self-consistency, structured output · Fine-tuning: SFT, QLoRA pipeline (LLaMA2-7b, NF4, LoRA r=64) · Alignment: RLHF, PPO, DPO, ORPO, Constitutional AI · RAG: hybrid retrieval, reranking, HyDE, RAGAS evaluation · Agents: ReAct, tool use, MCP, multi-agent patterns

**Production** — Serving: ONNX, vLLM (PagedAttention), FastAPI, TensorRT · Monitoring: PSI/KS drift detection, Evidently, Prometheus + Grafana · Pipelines: Prefect, Feast feature store, Docker + K8s, GitHub Actions CI/CD · System design: recommendation (two-tower + FAISS), search/RAG at scale, document processing at 100K docs/day

---

## Active Coding Practice (`code_practice/`)

Hands-on, from-scratch implementations following the 68-session sequence in 03_LEARNING_PACK §
Part C: Sequence Models → Transformers → Prompting → LLMs → RAG → Agents.

```
code_practice/
├── shared_dataset.py       # Synthetic Acme Financial Services data (compliance-safe)
├── 01_seq_models/
│   ├── 01_rnn/             # Done - vanilla RNN from scratch (NumPy + BPTT)
│   ├── 02_rnn/             # Done - same model with nn.RNN (4× lower loss)
│   ├── 03_lstm/            # Done - LSTM from scratch (4 gates + cell state)
│   ├── 04_lstm_torch/      # Done - nn.LSTM (cross types, fluent grammar)
│   ├── 05_gru/             # Done - GRU from scratch (2 gates, beat NumPy LSTM)
│   ├── 06_gru_vs_lstm/     # Done - head-to-head: GRU smaller + lower loss, LSTM faster on CPU
│   ├── 07_bilstm_ner/      # Done - BiLSTM NER, train/val/test, F1 0.93, tokenizer bug
│   └── 08_bilstm_attention/
│       └── 08_bilstm_attention/  # Done - Bahdanau attention, interpretable weights, bridge to transformers
├── 02_transformers/
│   ├── 09_seq2seq/         # Done - Seq2Seq + attention, full Bahdanau 2014 architecture
│   │   └── 10_bpe/         # (Phase 1 finale)
│   ├── 10_bpe/             # Done - BPE tokenizer from scratch (240 merges, byte-level vs char-level lesson)
│   ├── 11_attention/       # Done - Scaled dot-product attention, real misclassification bug discovered
│   ├── 12_mha/             # Done - Multi-head attention FIXED the Session 11 bug (loan 93% checking 4%)
│   ├── 13_pos_enc/         # Done only - Sinusoidal + learned + RoPE; permutation-equivariance proof + order sensitivity
│   ├── 14_encoder_block/   # Done - Full transformer block (LayerNorm + MHA+RoPE + FFN + residual)
│   ├── 15_mini_transformer/ # Done - 4 stacked blocks; depth speeds convergence, attention stays uniform on each mat
│   ├── 16_chunking/        # Done - Causal masking + LN (loss 0.19, beats LSTM 0.37 by 2×)
│   ├── 17_tiny_gpt/        # Done - Tiny GPT with 4 sampling strategies, loss 0.12, fluent generation
│   ├── 18_kv_cache/        # Done - KV cache inference (4× speedup, long-context degradation observed)
│   ├── 19_cross_attention/ # Done - Encoder-decoder transformer; beat BiLSTM+Bahdanau (21% on 1B test, fewer params)
│   └── 11_bf_load/         # Done - Architecture inspection; verified TinyGPT = GPT-2 (143M params, weight tying explains 38M)
├── 03_prompting/
│   ├── 01_first_call/      # Done - Ollama HTTP client; revealed small-model hallucination on "RAG" question
│   ├── 02_few_shot/        # Done - Few-shot only helps if model is big enough (1B: no lift, 8B: 93%→100%)
│   ├── 03_cot/             # Done - CoT gave 0% lift on 8B (direct already 100%) and hurt 1B: saturation + hallucinated steps
│   ├── 04_self_consistency/ # Done - B×5 vote rescued 6 problems on 1B but regressed 1 (out 0%); 8B saturated; 4-5× cost
│   ├── 05_json_output/     # Done - Free-text 0/10 (markdown formatting breaks repro), JSON mode 100%/98%; Pydantic caught hallucinated enum
│   ├── 06_function_call/   # Done - 8B 0/6 (failed multi-tool + tools-as-text); 1B 1/6 (wrong args, hallucinated math next to a calculator)
│   ├── 07_react_manual/    # Done - 8B 5/6 (finish fixed Q5 but post-success wander on Q3/Q6); 1B 0/6 (never exits Finish)
│   ├── 08_system_prompts/  # Done - 8B personas separated cleanly (formal 1.76× verbose, 4.6× jargon); 1B collapsed; disclaimer compliance 0-12%
│   ├── 09_injection/       # Done - Strict prompt INCREASED 1B leak (28→40%); L2 input filter caught all; security lives in code, not prompts
│   └── 10_streaming/       # Done - TTFT improved 84% on 1B, 95% on 8B (88s→2.5s); same total work, transformative UX. Phase 3 complete.
├── 04_llms/                # Phase 4 = 14 sessions: docs ✓, code on 10 sessions, awaiting traction
│   ├── 01_load_generate/   # Code - load TinyLlama + .generate() + chat template (MPS-clean)
│   ├── 02_decoding/        # Docs only - greedy/beam/top-k/top-p comparison
│   ├── 03_quant_8bit/      # Docs only - Code pivot ONLY + GGUF Q8, no bitsandbytes
│   ├── 04_quant_4bit/      # Docs only - Q4_K_M + CPU Q4 (the 4× memory cut)
│   ├── 05_dataset_prep/    # Docs only - Acme fin Chats / Length audit
│   ├── 06_lora_scratch/    # Code - LoRALinear from scratch (no training, math demo)
│   ├── 07_qlora_train/     # Code - THE BIG ONE: real fine-tune on Acme data
│   ├── 08_merge_save/      # Code - merge_and_unload + push_to_hub
│   ├── 09_eval_compare/    # Code - 20-prompt eval, 5 axes, McNemar significance
│   ├── 10_dpo/             # Code - preference tuning via TRL DPOTrainer
│   ├── 11_fastapi_serve/   # Code - FastAPI + streaming inference
│   ├── 12_vllm_serve/      # Code - vLLM + sequential + batched comparison
│   ├── 13_lm_eval_harness/ # Code - ARC + HellaSwag + TruthfulQA, catastrophic-forgetting check
│   └── 14_observability/   # Code - Prometheus metrics + structured logs + request IDs
├── 04.5_advanced/
│   ├── 01_cpt/             # Code - Continue pretraining on Acme articles
│   ├── 02_function_calling_ft/ # Code - train model to natively exit tool calls
│   ├── 03_distillation/    # Code - TinyLlama teacher + DistilGPT-2 student
│   └── 04_speculative_decoding/ # Code - draft/verify via assistant_model API
├── 05_rag/                 # Phase 5 = 10 sessions: docs complete, code on request
│   ├── 01_chunking/        # Docs only - 4 chunking strategies compared
│   ├── 02_embeddings/      # Docs only - MiniLM vs BGE, async vs sym
│   ├── 03_vector_db/       # Docs only - Chroma / FAISS / qdrant
│   ├── 04_basic_rag/       # Docs only - end-to-end pipeline
│   ├── 05_reranking/       # Docs only - cross-encoder two-stage
│   ├── 06_hybrid_search/   # Docs only - BM25 + dense + RRF
│   ├── 07_query_expansion/ # Docs only - HyDE + multi-query
│   ├── 08_rag_eval/        # Docs only - recall@k, MRR, faithfulness, answer relevance
│   ├── 09_indirect_injection/ # Docs only - defense against poisoned docs
│   └── 10_production_rag/  # Docs only - semantic cache + observability + cost
└── 06_agents/              # Phase 6 = 10 sessions: docs complete, code on request
    ├── 01_fundamentals/    # Docs only - production ReAct with guards
    ├── 02_langchain_primer/ # Docs only - LCEL, chains, tools
    ├── 03_langgraph/       # Docs only - state-machine orchestration
    ├── 04_tools_registry/  # Docs only - Pydantic schemas + auth + audit
    ├── 05_memory/          # Docs only - short + mid + long term
    ├── 06_planner_executor/ # Docs only - separate plan from act
    ├── 07_multi_agent/     # Docs only - supervisor + researcher + analyst + composer
    ├── 08_mcp_protocol/    # Docs only - MCP server + client
    ├── 09_agent_eval/      # Docs only - 5 axes: success, tool acc, efficiency, cost, safety
    └── 10_production_agents/ # Docs only - budgets + timeouts + HITL + audit
```

Per-session structure: `model.py` (architecture) · `train.py` (training or end-to-end) · `predict.py` (CLI inference) · `all_details.md` (objective, shapes, how-to-run, ACTUALS)

**Sequence totals:** 9 + 11 + 10 (Phase 1-3, all run) + 14 + 4 + 10 + 10 (Phase 4-6, code/docs at various levels) = **68 sessions across 6.5 phases.**

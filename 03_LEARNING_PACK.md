# Learning Pack — Mental Model + Theory + Coding Sequence

> One file. Everything you study or code lives here. Replaces the old `03_ML_Theory_Checklist`, `05_coding_practice_sequence`, and `06_mental_model_LLMs_and_GenAI`.

---

## Table of Contents

**Part A — Strategic Framing (read this first, once)**

1. The 6 Layers of LLM Mastery
2. The Three Eternal Questions
3. Where Each Phase Sits in the Map
4. Short History of LLMs (interview-ready)

**Part B — Decision Matrices (the lookup tables)**

5. 5 Paradigms of Model Adaptation
6. 3 Sources of Knowledge
7. Inference Optimization Comparison
8. Serving Frameworks Comparison
9. Eval Frameworks Comparison
10. The "When to Use What" Decision Matrix

**Part C — The Coding Sequence (pointer to INDEX.md)**

11. Coding Practice — How to Use
12. Per-Phase Session List — pointer to `code_practice/INDEX.md` + exit tests
13. Combined Sequence at a Glance

**Part D — Broad ML Theory Checklist (general ML, not just LLMs)**

14. Phase Th-1 — ML Fundamentals
15. Phase Th-2 — Deep Learning
16. Phase Th-3 — Computer Vision
17. Phase Th-4 — NLP Fundamentals
18. Phase Th-5 — Multimodal
19. Phase Th-6 — ML System Design
20. Phase Th-7 — MLOps for LLMs

**Part E — Self-Assessment & Reference**

21. Self-Assessment — Do You Actually Know It?
22. Reading List — Papers an LLM Engineer Should Know
23. What You Can Claim After This Sequence

---

## Part A — Strategic Framing

### 1. The 6 Layers of LLM Mastery

```
                        COMPLEXITY ^

LAYER 6 — RESEARCH FRONTIER
    Reasoning models, MoE, long context, multimodal, alignment
                            |
LAYER 5 — PRODUCTIONIZATION
    Quantization, batching, vLLM, eval, observability, safety
                            |
LAYER 4 — AGENT SYSTEMS
    Tool use, planning, memory, multi-agent, MCP
                            |
LAYER 3 — AUGMENTATION
    RAG, retrieval, embeddings, vector DBs, hybrid search
                            |
LAYER 2 — ADAPTATION (TRAINING)
    SFT, LoRA, QLoRA, DPO, CPT, distillation
                            |
LAYER 1 — INTERACTION (PROMPTING)
    Few-shot, CoT, JSON output, function calling, system prompts
                            |
LAYER 0 — ARCHITECTURE
    Tokenizer, embeddings, attention, transformers, decoders
```

| Layer | Tier | "Can you..." | Phase coverage |
|-------|------|--------------|----------------|
| L0 | Junior ML Eng | Explain attention, build transformer from scratch? | Phase 1, 2 |
| L1 | Mid | Get useful behavior from any LLM via prompting alone? | Phase 3 |
| L2 | Senior | Fine-tune a model on your data, deploy it? | Phase 4 |
| L3 | Senior+ | Build production RAG? | Phase 5 |
| L4 | Staff | Build a working agent with tools, memory, planning? | Phase 6 |
| L5 | Staff+ | Train a custom 7B from scratch, deploy at scale? | Beyond |
| L6 | Research | Push the frontier? | Out of scope |

**Senior LLM engineer = L0 through L4 cold.** That's exactly what the 68-session sequence builds.

---

### 2. The Three Eternal Questions

Every interview question reduces to one of:

**Q1: How do we MAKE the model better?**

- **Pretraining** — train from scratch on internet-scale text (frontier labs only)
- **CPT (continued pretraining)** — domain text only, no instructions
- **SFT** — instruction tuning
- **Preference tuning (DPO / RLHF)** — align outputs to human preference
- **Distillation** — small student mimics large teacher

**Q2: How do we make the model KNOW more?**

- **Parameters** (slow) — fine-tuning bakes knowledge into weights
- **Retrieval (RAG)** (fast) — fetch fresh docs at query time
- **Tools** (real-time) — let the model call APIs, calculators, DBs

**Q3: How do we DEPLOY the model?**

- **Quantization** — 4/8-bit for memory
- **Batching** — many concurrent requests at once
- **Speculative decoding** — small draft, big target verifies
- **Streaming** — token-by-token UX
- **Inference servers** — FastAPI (educational), vLLM / TGI / llama.cpp (production)

> When you hear an interview question, identify which Q. The structure of the answer follows from the Q.

---

### 3. Where Each Phase Sits in the Map

| Phase | Layer | What you build | What you can claim after |
|-------|-------|----------------|--------------------------|
| Phase 01 — Sequence Models (9) | L0 | RNN / LSTM / GRU / BiLSTM / Bahdanau attention from scratch | "I understand sequence modeling deeply." |
| Phase 02 — Transformers (11) | L0 | BPE / attention / MHA / RoPE / encoder / decoder / KV cache / Tiny GPT | "I can build a transformer from scratch and explain every component." |
| Phase 03 — Prompting (10) | L1 | Few-shot / CoT / self-consistency / JSON / function call / ReAct / injection defense / streaming | "I can pick the right prompting technique for any task and defend it with data." |
| Phase 04 — LLMs legacy (14) | L2 | Load / decode / quantize / SFT / CPT / DPO / merge / eval / vLLM / FastAPI / observability | "I have fine-tuned and deployed an LLM end-to-end." |
| Phase 05 — Transformers HF (7) | L0+L1 | BERT classification / NER / QA · BART summarization · T5 translation · GPT-2 generation · Sentence Transformers | "I apply pretrained models to real NLP tasks using HuggingFace." |
| Phase 06 — LLMs Core (3) | L1 | Prompt engineering · Pydantic + Instructor extraction · LLM-as-judge evaluation | "I use LLM APIs to build production features." |
| Phase 07 — RAG (5) | L3 | Basic RAG · 5 chunking strategies · BM25+dense+RRF+reranker · RAGAS eval · semantic cache + FastAPI | "I can build production RAG with eval, hybrid retrieval, and semantic cache." |
| Phase 08 — Agents (4) | L4 | ReAct from scratch · OpenAI tool calling · LangGraph multi-turn · document agent (HITL portfolio project) | "I can architect an agent system with LangGraph, memory, and HITL." |
| Phase 09 — Fine-tuning (6) | L2 | LoRA · QLoRA NF4 · synthetic dataset pipeline · DPO · vLLM serving · LLM monitoring | "I have fine-tuned, aligned, and deployed LLMs end-to-end." |
| Phase 10 — Document AI (4) | L5 | LayoutLMv3 · Donut OCR-free · ColPali vision-RAG · full ICE-style pipeline | "I own end-to-end Document AI from OCR through multimodal extraction — my differentiator." |

---

### 4. Short History of LLMs (interview-ready)

| Year | Event | Why it matters |
|------|-------|----------------|
| 2017 | Attention Is All You Need (Vaswani et al.) | Transformer architecture — replaces RNNs |
| 2018 | BERT (encoder), GPT-1 (decoder) | Two paradigms: BERT for understanding, GPT for generation |
| 2019 | GPT-2 (1.5B) | First "scary" LM |
| 2020 | GPT-3 (175B) | **In-context learning emerges.** Few-shot works without fine-tuning |
| 2021 | LoRA (Hu et al.), Codex | LoRA democratizes fine-tuning |
| 2022 | ChatGPT, InstructGPT, RLHF, Chinchilla, CoT, LLaMA-1 | RLHF productized. Chinchilla: more data, less compute. CoT emerges. |
| 2023 | GPT-4, LLaMA-2, QLoRA, DPO, Mixtral | QLoRA enables 65B fine-tune on 1 GPU. DPO replaces PPO. MoE goes mainstream. |
| 2024 | LLaMA-3, Claude-3, o1 (test-time reasoning) | Open models near frontier. Reasoning models emerge. |
| 2025 | DeepSeek-V3, Claude 3.5+ | Reasoning + MoE + long-context = frontier. |
| 2026 (now) | — | Sequence models → transformers → prompting → fine-tuning → RAG → agents = practitioner stack |

**Five concepts to recite cold:**
1. **Scaling laws** (Chinchilla 2022) — for a fixed compute budget, more data beats more params
2. **In-context learning** (GPT-3, Brown 2020) — emergent above ~7B
3. **RLHF → DPO** (Ouyang 2022 → Rafailov 2023) — how LLMs are aligned
4. **LoRA / QLoRA** (Hu 2021, Dettmers 2023) — parameter-efficient fine-tuning
5. **Chain-of-Thought** — test-time reasoning (Wei 2022 → o1 2024)

---

## Part B — Decision Matrices

### 5. 5 Paradigms of Model Adaptation

| Paradigm | Changes | Compute | Data | When to use |
|----------|---------|---------|------|-------------|
| **Prompting (in-context)** | Nothing | Free | 0-5 examples | Try first; always cheapest |
| **Retrieval (RAG)** | Adds context per query | Free training, low inference | Document corpus | Knowledge is dynamic / proprietary |
| **SFT (LoRA / QLoRA)** | Adapter weights | Hours-days | 100k-10K (instr, output) pairs | Format / style adaptation |
| **Preference (DPO / RLHF)** | Same as SFT, different signal | +50% of SFT | 1K-100K (chosen, rejected) | Improve human-rated quality |
| **Continued Pretraining (CPT)** | Full weights | Days-weeks | Raw domain text (millions of tokens) | Deep domain adaptation |
| **Pretraining from scratch** | Everything | Months × 1000s GPUs ($1M+) | Trillions of tokens | You're Meta / Anthropic / OpenAI |

**Decision heuristic:**
1. Try **prompting** first
2. Knowledge missing → **RAG**
3. Style / format issue → **SFT** (LoRA)
4. Outputs correct but ugly → add **DPO**
5. Model lacks deep domain knowledge → **CPT** (rare, expensive)
6. **Never** pretrain unless you ARE a frontier lab

---

### 6. 3 Sources of Knowledge

| Source | Stored in | Refresh latency | Best for |
|--------|-----------|-----------------|----------|
| **Parameters** | Model weights | Months (retrain) | General knowledge, language ability, learned format |
| **Retrieval (RAG)** | External vector DB | Seconds (re-index) | Domain facts that change |
| **Tools** | External APIs / DBs | Real-time | Math, current data, side effects |

**Production answer:** "Parameters for capability, RAG for fresh facts, tools for real-time data. They're complementary."

---

### 7. Inference Optimization Comparison

| Technique | Memory saving | Speed gain | Quality cost | Covered in |
|-----------|--------------|------------|--------------|------------|
| KV cache | 0 | 5-10× | None | P2 S18 |
| Quantization 8-bit | 2× | small | < 1% PPL | P4 S3 |
| Quantization 4-bit (NF4 / Q4_K_M) | 4× | small-medium | 1-3% PPL | P4 S4 |
| Flash Attention | small | 2-3× attention | None | Used by default in 2026 |
| Speculative decoding | 0 | 2-3× | None | P4.5 S4 |
| Continuous batching | 0 | 10-30× throughput | None | P4 S12 (vLLM) |
| Paged Attention | +30% KV efficiency | enables batching | None | vLLM internal |
| Distillation | 5-50× | 5-50× | 5-20% | P4.5 S3 |

**Production stack ordering:** smallest model that meets quality → Q4_K_M → vLLM → streaming → speculative decoding → distillation.

---

### 8. Serving Frameworks Comparison

| Framework | Native to | Throughput | Best for |
|-----------|-----------|------------|----------|
| **FastAPI + transformers** | Educational | 1-2 req/s | Learning, low-traffic prototypes |
| **vLLM** | CUDA production | 50-500 req/s | High-throughput HF models, OpenAI-compat API |
| **TGI** | CUDA production | 30-300 req/s | HuggingFace's serving solution |
| **llama.cpp / llama-cpp-python** | Local / CPU+Apple | 5-30 req/s | Quantized GGUF, edge |
| **Ollama** | Developer UX | 5-25 req/s | Demos, dev tools |
| **MLX-LM** | Apple Silicon | 15-50 req/s | Mac-native |
| **OpenAI-compatible** | API consumer | Provider-managed | Closed-source LLMs |

**Senior interview answer:** "vLLM behind a load balancer. PagedAttention + continuous batching = 10-30× over naive FastAPI."

---

### 9. Eval Frameworks Comparison

| Framework | Measures | Custom data? | When to use |
|-----------|----------|--------------|-------------|
| **Custom eval** (your code) | Whatever you measure | Yes | Domain-specific success |
| **lm-eval-harness** | Standard benchmarks (MMLU, GSM8K, etc.) | Limited | General capability comparison |
| **HELM** | 16+ standardized benchmarks | Limited | Holistic snapshot |
| **LMArena** | Human pairwise preference | No (humans) | Real-world preference |
| **LLM-as-judge** | Subjective quality 1-5 | Yes | Cheap human-preference proxy |
| **Production traces** | Real user behavior | Yes (your traffic) | Actual usefulness |

**Truth:** academic benchmarks correlate poorly with production usefulness. Junior: "75% MMLU." Senior: "task-completion on 500 real customer queries."

---

### 10. The "When to Use What" Decision Matrix

| Task | First try | If that fails | If that fails | Long-term answer |
|------|-----------|---------------|---------------|-----------------|
| Classify text — N classes | zero-shot prompt | few-shot prompt | JSON output | small fine-tune |
| Extract structured data | JSON mode | + Pydantic validation | + retry loop | fine-tune for format |
| Math | direct prompt | CoT | function calling (tool) | **always tool — never fine-tune for math** |
| Domain Q&A (changing facts) | RAG | RAG + rerank | + better chunking | RAG + fine-tune for format |
| Domain Q&A (static facts) | fine-tune | fine-tune + RAG fallback | CPT + SFT | full custom model |
| Multi-step task | ReAct or function call | + planner | + memory | dedicated agent (Phase 6) |
| Code generation | direct prompt | few-shot | fine-tune | specialized code model |
| Compliance | prompt with rules | + output filter | + fine-tune | guardrail layer (LlamaGuard) |
| Long-context (>16K) | long-context model | RAG to shrink | hierarchical summarization | sparse / RoPE-scaled |
| Streaming chat | streaming mode | + safety hook | + cancellation | vLLM with SSE |

**Senior signal:** the COLUMNS. Junior says "use RAG." Senior says "RAG first; if retrieval recall < 80%, switch to fine-tune + RAG hybrid."

---

## Part C — The Coding Sequence (68 sessions)

### 11. Coding Practice — How to Use

- Work top to bottom — each builds on the previous
- Each row = one coding session (1-2 hours)
- Tick `[x]` when working code + understanding both exist
- Session folder pattern: `code_practice/<phase>/<NN_session_name>/` → `model.py` # architecture / `math` → `train.py` # training (saves checkpoint) [some sessions] → `predict.py` # inference CLI → `all_details.md` # full docs: objective, shapes, run, ACTUALS → `checkpoints/` # auto-created on first train
- Shared synthetic dataset (Acme Financial Services) at `code_practice/shared_dataset.py` used across all sessions
- Priority: `***` must-build · `**` important · `*` nice-to-have

---

### 12. Per-Phase Session List

> The full 68-session table — with per-session theory pairings — lives in **`code_practice/INDEX.md`**. That file is the single source of truth for the session list and gets updated as sessions are wired or run. This Learning Pack only keeps the **strategic framing** for each phase (below) and the **combined view** (#13). For per-session details (topic, folder path, priority, status), go to `code_practice/INDEX.md`.

**Exit tests per phase**

| Phase | Sessions | Exit test |
|-------|----------|-----------|
| Phase 1 — Sequence Models | 9 | Can you write LSTM gates from a blank file AND explain why attention beats a fixed hidden state? |
| Phase 2 — Transformers | 11 | Can you draw the full transformer block from memory with attention shapes? Can you explain `//d_k`? |
| Phase 3 — Prompting | 10 | Can you pick the right technique for any task and back it with measured data? |
| Phase 4 — LLMs | 14 | Can you fine-tune (QLoRA), serve (vLLM), and observe (Prometheus) end-to-end? |
| Phase 4.5 — Advanced | 4 | CPT, function-calling FT, distillation, speculative decoding — when to use each? |
| Phase 5 — RAG | 10 | Can you build production RAG with eval, hybrid retrieval, and injection defense? |
| Phase 6 — Agents | 10 | Can you architect an agent system with LangGraph, MCP, memory, and HITL? |

---

### 13. Combined Sequence at a Glance

```
Phase 01: Sequence Models  (9 sessions)  ✅ Run  —  RNN + LSTM + GRU + BiLSTM + attention
Phase 02: Transformers     (11 sessions) ✅ Run  —  understand the engine (incl. cross-attention)
Phase 03: Prompting        (10 sessions) ✅ Run  —  call LLMs fluently
Phase 04: LLMs legacy      (14 sessions) ✅ Run  —  fine-tune + serve + observability
Phase 05: Transformers HF  (7 sessions)  🔧 Code-built  —  apply pretrained models to NLP tasks
Phase 06: LLMs Core        (3 sessions)  🔧 Code-built  —  LLM APIs: prompting, extraction, eval
Phase 07: RAG              (5 sessions)  🔧 Code-built  —  full RAG with hybrid search + eval   [Project 1+]
Phase 08: Agents           (4 sessions)  🔧 Code-built  —  ReAct → LangGraph → portfolio agent  [Project 3]
Phase 09: Fine-tuning      (6 sessions)  🔧 Code-built  —  LoRA/QLoRA/DPO/vLLM/monitoring       [Project 2]
Phase 10: Document AI      (4 sessions)  🔧 Code-built  —  LayoutLM + Donut + ColPali + pipeline [Differentiator]
```

**Total coded: 73 sessions (phases 01-10) · Phases 01-04 run · Phases 05-10 coded, awaiting run.**

---

## Part D — Broad ML Theory Checklist

> Beyond just LLMs — general ML interview gaps to plug. Most you've done in production; mark `[x]` and move on. Spend time on `[ ]`.
>
> `***` interview must-know · `**` separates strong candidates · `*` nice-to-have

---

### 14. Phase Th-1 — ML Fundamentals

**Core algorithms** (you have hands-on experience here — focus on being able to *explain*)

- [ ] Linear & Logistic Regression — math + L1/L2 regularization `***`
- [ ] Decision Trees — splitting criteria, pruning `**`
- [ ] Random Forest — bagging, feature importance, OOB error `***`
- [ ] Gradient Boosting (XGBoost, LightGBM) — boosting vs bagging `***`
- [ ] SVM — kernel trick, margin maximization `**`
- [ ] K-Means / DBSCAN — when each `**`
- [ ] PCA — variance explained, when to use `**`

**Model evaluation (interview staple)**

- [ ] Bias-variance tradeoff — explain with examples `***`
- [ ] Cross-validation — k-fold, stratified k-fold `***`
- [ ] Metrics — Accuracy vs Precision vs Recall vs F1 vs AUC-ROC `***`
- [ ] Confusion matrix — read + act on it `***`
- [ ] Imbalanced datasets — SMOTE, class weights, threshold tuning `***`
- [ ] PR curve vs ROC curve — when more useful `**`

**Feature engineering**

- [ ] Missing data — imputation strategies `**`
- [ ] Encoding categoricals — label / one-hot / target encoding `**`
- [ ] Feature scaling — which models need it `**`
- [ ] Feature selection — correlation, mutual info, importance `**`

---

### 15. Phase Th-2 — Deep Learning

**Core concepts**

- [ ] Backpropagation — chain rule, how gradients flow `***`
- [ ] Vanishing / exploding gradients — why + how to fix `***`
- [ ] Activation functions — ReLU, GELU, Sigmoid, Softmax `***`
- [ ] Loss functions — CE, MSE, CTC — math behind each `***`
- [ ] Weight initialization — Xavier, He `**`

**Training**

- [ ] Optimizers — SGD, Momentum, Adam, AdamW `***`
- [ ] LR scheduling — step decay, cosine, warmup `***`
- [ ] BatchNorm vs LayerNorm — why Transformers use LN `***`
- [ ] Dropout — train vs inference mode `***`
- [ ] Weight decay / L2 — Adam vs SGD `**`
- [ ] Early stopping `**`
- [ ] Mixed precision (FP16, BF16) `**`
- [ ] Gradient clipping — why RNNs need it `**`
- [ ] Distributed training — data vs model parallelism `**`

---

### 16. Phase Th-3 — Computer Vision

**CNN fundamentals**

- [ ] Convolution — filters, stride, padding, output size `***`
- [ ] Receptive field `**`
- [ ] Pooling — max vs avg vs GAP `**`
- [ ] Depthwise separable convolutions (MobileNet) `*`

**Architectures**

- [ ] ResNet — residual connections, vanishing gradient fix `***`
- [ ] EfficientNet — compound scaling, NAS `**`
- [ ] U-Net — encoder-decoder for segmentation `**`
- [ ] YOLO — single-shot detection, anchors `**`
- [ ] DETR — detection with transformers `*`
- [ ] ViT — image as patches, vision meets transformers `***`

**Applied**

- [ ] Transfer learning — freeze/unfreeze, LR strategy `***`
- [ ] Data augmentation — standard + MixUp / CutMix `**`
- [ ] Grad-CAM `**`
- [ ] Knowledge distillation (covered in Phase 4.5 S3) `*`

---

### 17. Phase Th-4 — NLP Fundamentals

**Text fundamentals**

- [ ] Tokenization — BPE / WordPiece / SentencePiece `***`
- [ ] Vocabulary + OOV — subword tokenization solves it `***`
- [ ] Preprocessing — lowercasing, stemming vs lemmatization `**`
- [ ] TF-IDF — formula, when still useful `*`

**Embeddings**

- [ ] Word2Vec — CBOW vs Skip-gram, negative sampling `**`
- [ ] GloVe — global co-occurrence `**`
- [ ] FastText — subword embeddings `*`
- [ ] Contextual embeddings — why ELMo/BERT beat static `***`

**BERT & fine-tuning**

- [ ] BERT pretraining — MLM + NSP `***`
- [ ] BERT fine-tuning — classification / NER / QA heads `***`
- [ ] Sentence transformers — bi-encoder for similarity `***`
- [ ] Cross-encoder vs bi-encoder — when each (covered in Phase 5 S5) `***`

**Tasks**

- [ ] NER — BIO tagging, seqeval `**`
- [ ] Text classification — multi-class vs multi-label `**`
- [ ] Semantic similarity — cosine on embeddings `***`
- [ ] Summarization — extractive vs abstractive `*`

---

### 18. Phase Th-5 — Multimodal

> Extends Document AI background. Mostly conceptual unless you do a project.

- [ ] CLIP — contrastive learning, shared image+text space `***`
- [ ] ViT — image as patches `***`
- [ ] How VLMs work — image encoder + projection + LLM decoder `***`
- [ ] Donut — Swin encoder + BART decoder `**`
- [ ] LayoutLM — text + bounding box + image for docs `***`
- [ ] LLaVA — open-source VLM, visual instruction tuning `**`
- [ ] PaliGemma / Qwen-VL `*`
- [ ] Florence-2 (Microsoft doc understanding) `*`

---

### 19. Phase Th-6 — ML System Design

> Every Sr. ML interview includes at least one system design question.

**The 8-step framework**

- [ ] 1. Clarify requirements — scale, latency, accuracy, constraints `***`
- [ ] 2. Define metrics — offline + online `***`
- [ ] 3. High-level architecture — data → model → serving → monitoring `***`
- [ ] 4. Data pipeline — collection, labeling, features, storage `***`
- [ ] 5. Model selection — start simple, justify complexity `***`
- [ ] 6. Training pipeline `**`
- [ ] 7. Serving — latency budget, batching, caching, A/B `**`
- [ ] 8. Monitoring — data drift, model drift, retraining `***`

**Must-know design problems**

- [ ] Recommendation system — two-tower, ALS, FAISS HNSW, rerank, cold start `***`
- [ ] Search / RAG system — hybrid + rerank + 10K QPS `***`
- [ ] Document processing pipeline — classify + extract + validate + HITL `***`
- [ ] Fraud detection — real-time vs batch, feature store, imbalance `**`
- [ ] CTR prediction — wide & deep, feature crosses `**`
- [ ] Semantic search / embedding pipeline `**`

**Key tradeoffs to know cold**

- [ ] Precision vs recall — when to optimize which `***`
- [ ] Latency vs accuracy — simpler/faster model in path `***`
- [ ] Online vs batch inference `***`
- [ ] Buy vs build — OpenAI vs fine-tune vs train from scratch `***`
- [ ] Cold start `**`
- [ ] Feedback loops `**`

**Practical skills**

- [ ] Draw architecture diagrams clearly `***`
- [ ] Size your system — storage, QPS, model size, GPUs `***`
- [ ] Talk through tradeoffs out loud `***`
- [ ] 2 designs/week out loud, timed at 45 min `***`

---

### 20. Phase Th-7 — MLOps for LLMs

**LLM serving** (mostly covered in Phase 4-6 coding)

- [ ] vLLM — PagedAttention, continuous batching `***`
- [ ] Deploy open-source LLM with vLLM (OpenAI-compat) `***`
- [ ] ONNX — convert for CPU `**`
- [ ] GPTQ / AWQ — production quantization `**`
- [ ] Batching — static vs dynamic `**`

**Monitoring & drift**

- [ ] Evidently — data + prediction drift reports `**`
- [ ] LLM monitoring — hallucination, latency, cost, I/O logging `***`
- [ ] Prometheus + Grafana for FastAPI `**`
- [ ] Alerting — thresholds, accuracy drop / drift `**`

**Pipelines**

- [ ] Prefect basics — flow, task, scheduling `**`
- [ ] Scheduled pipeline — new docs → embed → vector store `**`
- [ ] Kubernetes basics — pod, deployment, service `**`
- [ ] Feature store — Feast, when needed at scale `*`

---

## Part E — Self-Assessment & Reference

### 21. Self-Assessment — Do You Actually Know It?

For each topic, ask the **5 questions**. If you can answer all 5, you KNOW it. If < 3, study harder.

1. Can you **describe** it in 2 sentences?
2. Can you **explain when** to use it (and when NOT)?
3. Can you **write the code** from a blank file?
4. Can you defend a **non-obvious choice** (e.g., "why r=8 not r=64 in LoRA")?
5. Can you discuss **failure modes** from your own experience?

**Phase 1 — Sequence Models**
- [ ] RNN forward + BPTT math
- [ ] LSTM gates — forget/input/output/cell
- [ ] Why attention beats a fixed hidden state
- [ ] GRU vs LSTM (when simpler is enough)
- [ ] BiLSTM for NER

**Phase 2 — Transformers**
- [ ] BPE tokenization (merges, byte-level)
- [ ] Scaled dot-product attention (and why `/d_k`)
- [ ] Multi-head — split / compute / concat / project
- [ ] Sinusoidal vs learned vs RoPE
- [ ] Encoder block: LN + MHA + FFN + residual
- [ ] Causal masking (decoder)
- [ ] KV cache (4× speedup, long-context degradation)
- [ ] Cross-attention (encoder-decoder)
- [ ] Why your architecture matches GPT-2

**Phase 3 — Prompting**
- [ ] When few-shot helps (emergent above 7B)
- [ ] CoT lift is bounded by direct-accuracy ceiling
- [ ] Self-consistency only fixes RANDOM error, not bias
- [ ] JSON mode + Pydantic = production parsing
- [ ] 5 failure modes of function calling
- [ ] ReAct vs native tool calling (when each wins)
- [ ] System prompts shape STYLE not KNOWLEDGE
- [ ] Strict system prompts can INCREASE leak rate
- [ ] Streaming TTFT vs total time

**Phases 04 — LLMs (legacy, all run)**
- [ ] Load HF model + apply chat template
- [ ] 4 decoding strategies
- [ ] Memory math: INT8 vs NF4
- [ ] LoRA math: W = W + B·A, why low rank works
- [ ] QLoRA = LoRA + NF4 base
- [ ] DPO loss vs PPO (no reward model needed)
- [ ] vLLM continuous batching + paged attention
- [ ] Observability — 4 golden signals for LLMs

**Phase 07 — RAG (coded)**
- [ ] Chunking strategies — fixed / sentence / hierarchical / semantic
- [ ] FAISS IndexFlatIP + L2-normalised embeddings
- [ ] Hybrid search: BM25 + dense + RRF fusion
- [ ] Cross-encoder reranking (two-stage pipeline)
- [ ] HyDE + multi-query query transformation
- [ ] RAGAS 4 metrics: faithfulness, relevancy, precision, recall
- [ ] Semantic cache (two-tier: exact + cosine)
- [ ] Indirect prompt injection in RAG

**Phase 08 — Agents (coded)**
- [ ] ReAct loop from scratch (parse Thought/Action/Observation)
- [ ] OpenAI function calling (tool schemas, parallel tool calls)
- [ ] LangGraph state machine (StateGraph, MemorySaver, tools_condition)
- [ ] HITL interrupt + Command(resume=...) pattern
- [ ] Specialist agent pattern (supervisor + workers)
- [ ] Planner-executor patterns: ReAct vs Plan&Execute vs ReWoo vs LATS vs Reflexion

**Phase 09 — Fine-tuning (coded)**
- [ ] LoRA config: r, alpha, target_modules — what each does
- [ ] QLoRA = NF4 4-bit base + LoRA adapters (6× memory reduction)
- [ ] SFT with trl SFTTrainer + SFTConfig
- [ ] DPO objective: β parameter, chosen vs rejected format
- [ ] vLLM PagedAttention + continuous batching
- [ ] Prometheus + PSI drift detection for LLM observability

**Phase 10 — Document AI (coded)**
- [ ] LayoutLMv3: text + bbox + image patches fused — why layout matters
- [ ] Donut: image → JSON end-to-end, no OCR step
- [ ] ColPali: 1030 patch embeddings per page, MaxSim late interaction
- [ ] Full pipeline: ingest → OCR → classify → extract → QA

**If you check ALL: you're L4-ready for any senior LLM interview.**

---

### 22. Reading List — Papers an LLM Engineer Should Know

> You don't need to read all of these. You DO need to say "yes, I'm familiar with that paper" when asked.

**Must-know (top 10)**

1. Attention Is All You Need (Vaswani 2017) — the transformer
2. BERT (Devlin 2018) — encoder pretraining
3. GPT-3 (Brown 2020) — in-context learning
4. LLaMA (Touvron 2023) — open-weights design choices
5. Chinchilla / scaling laws (Hoffmann 2022) — data > params
6. LoRA (Hu 2021) — parameter-efficient FT
7. QLoRA (Dettmers 2023) — 4-bit base + LoRA
8. InstructGPT / RLHF (Ouyang 2022) — how ChatGPT was made
9. DPO (Rafailov 2023) — replaced PPO
10. Chain-of-Thought (Wei 2022) — reasoning prompts

**Should-know (next 10)**

11. GPT-2 (Radford 2019)
12. BPE / byte-pair encoding (Sennrich 2015)
13. RoPE (Su 2021)
14. FlashAttention (Dao 2022)
15. Mistral / Switch Transformer (Fedus 2021, Jiang 2024)
16. Self-consistency (Wang 2022)
17. ReAct (Yao 2022)
18. PagedAttention / vLLM (Kwon 2023)
19. Constitutional AI (Bai 2022)
20. DPO follow-ups: ORPO, KTO, IPO

**Worth knowing**

- Toolformer / Gorilla — tool-use fine-tuning
- Speculative Decoding (Leviathan 2023)
- Llama-3 paper (Meta 2024)
- DeepSeek-V3 paper (2024)
- o1 / RLVR (OpenAI / DeepSeek-R1, 2024-2025)

---

### 23. What You Can Claim After This Sequence

**You CAN claim with full integrity (Phases 01-04 run, 05-10 coded):**

- "I built RNN, LSTM, GRU, BiLSTM, Bahdanau attention from scratch in NumPy and PyTorch."
- "I built a Tiny GPT from scratch (BPE → attention → blocks → decoder)."
- "I have hands-on experience with 10 prompting techniques and can defend when each applies."
- "I built and fine-tuned LLMs end-to-end — LoRA, QLoRA (NF4 4-bit), DPO alignment."
- "I built production RAG: hybrid search (BM25 + dense + RRF), cross-encoder reranking, RAGAS evaluation, semantic cache."
- "I built LangGraph agents with HITL interrupts, memory, and specialist agent routing."
- "I have 8 years of production Document AI — LayoutLMv3, Donut OCR-free, ColPali vision-RAG, full pipeline."
- "I deployed LLMs with vLLM (PagedAttention + continuous batching) and built Prometheus observability."

**Add these after running Phases 05-10:**

- "I applied BERT, BART, T5, GPT-2, Sentence Transformers to financial NLP tasks using HuggingFace."
- "I fine-tuned TinyLlama-1.1B with QLoRA — model fits in 700 MB GPU vs 4.4 GB float32."
- "I built a LangGraph mortgage document agent: classify → extract → policy RAG → eligibility → HITL → report."
- "I built an end-to-end document AI pipeline (PDF → OCR → classify → LayoutLM extract → QA) — open-sourced."

**You CAN claim if asked specifically:**

- "I understand QLoRA fully — NF4 quantization + LoRA adapters. bitsandbytes needs CUDA; code runs on Windows with flag."
- "I haven't run pretraining from scratch — that requires $1M+ compute."
- "RLHF/PPO is legacy — DPO is the modern replacement and I've built it."

**Do NOT claim:**

- "I've trained a 70B model!" — interviewer will detect in 30 seconds.
- "I've contributed to model research" — different career path.
- "I've run ColPali in production" — coded and documented; needs CUDA to run.

---

> **The integrity matters. How tight to the truth — what you've done is plenty.**
>
> **Last principle:**
>
> A senior LLM engineer is not the one who knows the most techniques. It's the one who can quickly recognize WHICH technique a given problem calls for, and explain WHY.
>
> Everything in this document serves that recognition.

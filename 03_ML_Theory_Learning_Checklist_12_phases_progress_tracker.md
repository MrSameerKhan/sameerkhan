# ML Engineering Learning Checklist
> Sequential roadmap — build on your existing Document AI + MLOps foundation
>
> `***` Interview must-know + used daily in production
> `**`  Commonly asked in senior roles, understand deeply
> `*`   Good to know, less frequently tested or niche

---

## Table of Contents
- [Folder → Phase Mapping](#repo-folder--checklist-phase-mapping)
- [Phase 1 — Machine Learning Fundamentals](#phase-1--machine-learning-fundamentals)
- [Phase 2 — Deep Learning Fundamentals](#phase-2--deep-learning-fundamentals)
- [Phase 3 — Computer Vision](#phase-3--computer-vision)
- [Phase 4 — NLP](#phase-4--nlp)
- [Phase 5 — Transformers (Architecture Internals)](#phase-5--transformers-architecture-internals)
- [Phase 6 — LLMs: Prompting + Alignment](#phase-6--llms-prompting--alignment)
- [Phase 7 — RAG Pipelines](#phase-7--rag-pipelines)
- [Phase 8 — LLM Fine-tuning](#phase-8--llm-fine-tuning)
- [Phase 9 — LLM Agents](#phase-9--llm-agents)
- [Phase 10 — MLOps for LLMs](#phase-10--mlops-for-llms)
- [Phase 11 — Multimodal](#phase-11--multimodal)
- [Phase 12 — ML System Design](#phase-12--ml-system-design)
- [Capstone Projects](#capstone-projects-one-per-phase-goes-on-github--resume)
- [Estimated Timeline](#estimated-timeline)
- [Quick Reference — Importance Summary](#quick-reference--importance-summary)
- [High-Level Roadmap at a Glance](#high-level-roadmap-at-a-glance)

---

---

## Repo Folder ↔ Checklist Phase Mapping

| Folder | Checklist Phase |
|--------|----------------|
| `1.machine learning/` | Phase 1 |
| `2.deep learning/` | Phase 2 |
| `3.computerVision/` | Phase 3 |
| `4.nlp/` | Phase 4 |
| `5.transformers/` | Phase 5 |
| `6.llms/` | Phase 6, 7, 8, 9 |
| `7.multimodal/` | Phase 11 |
| `8.mlops/` | Phase 10 |
| `9.system_design/` | Phase 12 |

---

## PHASE 1 — Machine Learning Fundamentals
> You have hands-on experience here — focus on gaps and ability to explain concepts clearly in interviews

### Core Algorithms (review, not re-learn)
- [ ] Linear & Logistic Regression — math behind it, regularization (L1/L2) `***`
- [ ] Decision Trees — splitting criteria (Gini, entropy), pruning `**`
- [ ] Random Forest — bagging, feature importance, OOB error `***`
- [ ] Gradient Boosting — XGBoost, LightGBM, how boosting differs from bagging `***`
- [ ] SVM — kernel trick, margin maximization (conceptual) `**`
- [ ] K-Means, DBSCAN — when to use which clustering algorithm `*`
- [ ] PCA — variance explained, when to apply dimensionality reduction `**`

### Model Evaluation (interview staple)
- [ ] Bias-variance tradeoff — explain with examples `***`
- [ ] Cross-validation — k-fold, stratified k-fold, when to use `***`
- [ ] Metrics — Accuracy vs Precision vs Recall vs F1 vs AUC-ROC `***`
- [ ] Confusion matrix — how to read and act on it `***`
- [ ] Handling imbalanced datasets — SMOTE, class weights, threshold tuning `***`
- [ ] Precision-Recall curve vs ROC curve — when each is more useful `**`

### Feature Engineering
- [ ] Handling missing data — imputation strategies and when to use each `**`
- [ ] Encoding categorical variables — label, one-hot, target encoding `**`
- [ ] Feature scaling — standardization vs normalization, which models need it `**`
- [ ] Feature selection — correlation, mutual information, feature importance `**`

---

## PHASE 2 — Deep Learning Fundamentals
> You've built CNNs and LSTMs in production — fill the conceptual gaps interviewers probe

### Core Concepts (interview must-knows)
- [ ] Backpropagation — chain rule, how gradients flow `***`
- [ ] Vanishing / exploding gradients — why they happen, how to fix `***`
- [ ] Activation functions — ReLU, GELU, Sigmoid, Softmax — when to use which `***`
- [ ] Loss functions — Cross-entropy, MSE, CTC loss (you've used it) — math behind each `***`
- [ ] Weight initialization — Xavier, He — why it matters `**`

### Training Techniques
- [ ] Optimizers — SGD, Momentum, Adam, AdamW — differences and when to use `***`
- [ ] Learning rate scheduling — step decay, cosine annealing, warmup `***`
- [ ] Batch normalization — how it works, where to place it, train vs inference mode `***`
- [ ] Layer normalization — why Transformers use LayerNorm not BatchNorm `***`
- [ ] Dropout — how it works, why it helps, inference mode difference `***`
- [ ] Weight decay / L2 regularization — relation to Adam vs SGD `**`
- [ ] Early stopping — monitor val loss, patience `**`
- [ ] Mixed precision training — FP16, BF16 — why it speeds up training `**`

### Custom Training
- [ ] PyTorch training loop — forward, loss, backward, optimizer step `***`
- [ ] TensorFlow GradientTape — custom training loop `**`
- [ ] Gradient clipping — why needed for RNNs and large models `**`
- [ ] Distributed training — data parallelism vs model parallelism (you've used Horovod) `**`

---

## PHASE 3 — Computer Vision
> Strong background here — focus on modern architectures and interpretability gaps

### CNN Fundamentals (be able to explain, not just use)
- [ ] Convolution operation — filters, stride, padding, output size formula `***`
- [ ] Receptive field — what it is, why deeper layers see more `**`
- [ ] Pooling — max vs average, global average pooling `**`
- [ ] Depthwise separable convolutions — why MobileNet uses them (efficiency) `*`

### Architectures
- [ ] ResNet — residual connections, why they solve vanishing gradients `***`
- [ ] EfficientNet — compound scaling, NAS (you've used it, know why it works) `**`
- [ ] U-Net — encoder-decoder, skip connections, why used for segmentation `**`
- [ ] YOLO — single-shot detection, anchor boxes, why fast `**`
- [ ] DETR — detection with Transformers, no anchor boxes `*`
- [ ] ViT — image as patches, how vision meets Transformers `***`

### Applied CV
- [ ] Transfer learning strategy — freeze/unfreeze layers, learning rate for fine-tuning `***`
- [ ] Data augmentation — standard (flip, crop, rotate) + modern (MixUp, CutMix) `**`
- [ ] Grad-CAM — how it works mathematically (you have the code) `**`
- [ ] Knowledge distillation — teacher-student, soft labels (you have the code) `*`

---

## PHASE 4 — NLP
> You know NER and BERT from production — fill tokenization and sequence model gaps

### Text Fundamentals
- [ ] Tokenization — BPE, WordPiece, SentencePiece — how each works `***`
- [ ] Vocabulary, OOV handling — how subword tokenization solves OOV `***`
- [ ] Text preprocessing — lowercasing, stemming vs lemmatization, stopwords `**`
- [ ] TF-IDF — formula, when still useful over embeddings `*`

### Word Embeddings
- [ ] Word2Vec — CBOW vs Skip-gram, negative sampling `**`
- [ ] GloVe — global co-occurrence matrix (you've used it) — how it differs from W2V `**`
- [ ] FastText — subword embeddings, why better for morphologically rich languages `*`
- [ ] Contextual embeddings — why ELMo/BERT embeddings > static embeddings `***`

### Sequence Models
- [ ] RNN — vanishing gradient problem, why LSTMs were needed `***`
- [ ] LSTM — cell state, forget/input/output gates — explain each gate `***`
- [ ] GRU — simplified LSTM, when to choose over LSTM `**`
- [ ] Bidirectional LSTM — why bidirectional helps NER (you've used this) `**`
- [ ] Seq2Seq with attention — encoder-decoder, attention mechanism in NLP context `***`

### BERT & Fine-tuning
- [ ] BERT pretraining — Masked LM + NSP tasks, why bidirectional `***`
- [ ] BERT fine-tuning — classification head, NER head, QA head `***`
- [ ] Sentence Transformers — bi-encoder for semantic similarity, why faster than cross-encoder `***`
- [ ] Cross-encoder vs bi-encoder — when to use each (critical for RAG reranking) `***`

### NLP Tasks
- [ ] Named Entity Recognition — BIO tagging scheme, evaluation (seqeval) `***`
- [ ] Text classification — multi-class vs multi-label `**`
- [ ] Semantic similarity — cosine similarity on sentence embeddings `***`
- [ ] Summarization — extractive vs abstractive `*`

---

## PHASE 5 — Transformers (Architecture Internals)
> Needed for interviews — you use Transformers daily but need to explain the internals

### Attention Mechanism
- [ ] Self-attention — query, key, value — matrix math explained `***`
- [ ] Multi-head attention — why multiple heads, what each head captures `***`
- [ ] Scaled dot-product — why divide by √d_k (prevent softmax saturation) `***`
- [ ] Causal / masked attention (GPT) vs bidirectional (BERT) `***`
- [ ] Cross-attention — how encoder output connects to decoder `**`

### Positional Encoding
- [ ] Why needed — no recurrence means no position awareness `***`
- [ ] Sinusoidal encoding — original Attention Is All You Need `**`
- [ ] Learned positional embeddings — BERT approach `**`
- [ ] RoPE — Rotary Position Embedding — used in LLaMA, Mistral `**`
- [ ] ALiBi — attention with linear biases, extrapolates to longer sequences `*`

### Architecture Variants
- [ ] Encoder-only (BERT) vs Decoder-only (GPT) vs Encoder-Decoder (T5, Donut) `***`
- [ ] Pre-norm vs Post-norm LayerNorm placement `**`
- [ ] Feed-forward sublayer — dimensions, role (2 linear layers + activation) `**`
- [ ] Residual connections in Transformers — why critical at scale `**`

### Efficiency
- [ ] Flash Attention — memory bottleneck of standard attention, IO-aware solution `**`
- [ ] KV Cache — how autoregressive inference is sped up `***`
- [ ] Grouped Query Attention (GQA) — LLaMA-2 uses this, reduces KV cache size `**`
- [ ] Speculative decoding — draft model + target model, why it's faster `**`
- [ ] Mixture of Experts (MoE) — sparse activation, router, Mixtral architecture `**`
- [ ] Quantization basics — FP32 → FP16 → INT8 → INT4 tradeoffs `***`

---

## PHASE 6 — LLMs: Prompting + Alignment
> Conceptual foundation before building LLM systems

### Prompting Techniques
- [ ] Zero-shot vs few-shot — when each works better `***`
- [ ] Chain-of-Thought (CoT) — why step-by-step improves reasoning `***`
- [ ] Zero-shot CoT — "let's think step by step" trick `**`
- [ ] Self-consistency — multiple reasoning paths + majority vote `**`
- [ ] ReAct — Reason + Act pattern, foundation of agents `***`
- [ ] Structured output — JSON mode, function calling, grammar-constrained decoding `***`
- [ ] System prompt design — role, context, constraints, output format `***`

### Alignment
- [ ] Why base models need alignment — instruction following, safety `**`
- [ ] SFT — Supervised Fine-Tuning as first step `***`
- [ ] RLHF — reward model + PPO loop (conceptual understanding sufficient) `**`
- [ ] DPO — Direct Preference Optimization, simpler than PPO, increasingly standard `***`
- [ ] ORPO — combined SFT + alignment in one step `*`
- [ ] Constitutional AI — self-critique and revision approach `*`

### LLM Evaluation
- [ ] BLEU / ROUGE — n-gram overlap metrics, limitations for generative tasks `***`
- [ ] BERTScore — semantic similarity using contextual embeddings `**`
- [ ] LLM-as-Judge — pointwise vs pairwise scoring, position/verbosity bias `***`
- [ ] MMLU / MT-Bench / HumanEval — standard benchmarks, what each measures `**`
- [ ] pass@k — for code generation, probability of at least one correct in k samples `**`
- [ ] Hallucination detection — NLI-based factual consistency checking `***`
- [ ] RAGAS — faithfulness, answer relevancy, context precision, context recall `***`

---

## PHASE 7 — RAG Pipelines
> Build a production-quality RAG system — highest market demand right now

### Fundamentals
- [ ] Why RAG — hallucination, knowledge cutoff, domain specificity `***`
- [ ] Chunking strategies — fixed size, recursive, semantic chunking `***`
- [ ] Embedding models — BGE, E5, OpenAI ada-002 — how to choose `***`
- [ ] Vector stores — FAISS (local), ChromaDB (local), Pinecone (cloud) `***`
- [ ] Cosine similarity vs dot product vs L2 — when each is appropriate `**`

### Advanced Retrieval
- [ ] Hybrid search — dense (vector) + sparse (BM25) combined with RRF `***`
- [ ] Reranking — cross-encoder rerankers (BGE-reranker) after initial retrieval `***`
- [ ] HyDE — generate hypothetical answer, embed that for better retrieval `**`
- [ ] Parent-child chunking — retrieve small chunks, return large context `**`
- [ ] Multi-query retrieval — LLM generates multiple queries from one question `**`
- [ ] Contextual compression — compress retrieved chunks before sending to LLM `*`

### Build
- [ ] Pipeline: PDF ingestion → chunking → embedding → FAISS → retrieval → LLM `***`
- [ ] Add reranker on top of retrieval `***`
- [ ] Add conversation memory (chat history) `**`
- [ ] Deploy as FastAPI endpoint `***`

### Evaluation
- [ ] RAGAS metrics — faithfulness, answer relevancy, context precision, context recall `***`
- [ ] Build eval dataset for your domain `**`
- [ ] Interpret RAGAS scores, identify weakest component, fix it `**`

---

## PHASE 8 — LLM Fine-tuning
> QLoRA fine-tune an open-source LLM — resume differentiator

### Concepts
- [ ] When to fine-tune vs RAG vs prompting — decision framework `***`
- [ ] Full fine-tuning vs PEFT — why full fine-tuning is rarely practical `***`
- [ ] LoRA — low-rank decomposition, rank r, alpha, target modules `***`
- [ ] QLoRA — 4-bit NF4 quantization + LoRA, double quantization `***`
- [ ] Catastrophic forgetting — why it happens, how LoRA mitigates it `**`

### Practical
- [ ] Set up: bitsandbytes, PEFT, TRL, Accelerate `***`
- [ ] Load Mistral-7B or LLaMA-3 in 4-bit `***`
- [ ] Prepare dataset — Alpaca / ChatML instruction format `***`
- [ ] Configure LoRA — rank, alpha, target modules (q_proj, v_proj) `***`
- [ ] Train with SFTTrainer (TRL) `***`
- [ ] Monitor with Weights & Biases — loss curves, eval metrics `**`
- [ ] Merge LoRA weights into base model `**`
- [ ] Push to HuggingFace Hub `**`
- [ ] Compare fine-tuned vs base model on domain task `***`

---

## PHASE 9 — LLM Agents
> Build a working agentic system with tool use

### Fundamentals
- [ ] ReAct pattern — Reason → Act → Observe loop in detail `***`
- [ ] Tool definition — function calling, JSON schema for tools `***`
- [ ] Agent memory — short-term (context window) vs long-term (vector store) `***`
- [ ] Tool selection — how LLM decides which tool to use `**`
- [ ] Error handling in agents — retry logic, fallbacks `**`

### LangChain Agents
- [ ] Custom tool creation — @tool decorator, input/output schema `***`
- [ ] AgentExecutor — how it orchestrates ReAct loop `***`
- [ ] Memory — ConversationBufferMemory, VectorStoreRetrieverMemory `**`
- [ ] Build: document agent — OCR tool + vector search + DB lookup `***`

### LangGraph
- [ ] Nodes, edges, state graph — fundamentals `**`
- [ ] Build a 2-node pipeline (orchestrator + worker) `**`
- [ ] Conditional routing between agents `**`
- [ ] Persistence — checkpointing agent state `*`

### MCP — Model Context Protocol
- [ ] What MCP is — standardized protocol for LLM ↔ tool communication `**`
- [ ] MCP server — expose tools/resources via FastMCP `**`
- [ ] MCP client — connect Claude/any LLM to MCP server `**`
- [ ] When to use MCP vs function calling — MCP for reusable tool servers `*`

---

## PHASE 10 — MLOps for LLMs
> Fill gaps in your already-strong MLOps profile

### LLM Serving
- [ ] vLLM — PagedAttention, continuous batching, why faster than naive serving `***`
- [ ] Deploy open-source LLM with vLLM as OpenAI-compatible API endpoint `***`
- [ ] ONNX export — convert model for CPU inference `**`
- [ ] GPTQ / AWQ quantization for serving `**`
- [ ] Batching strategies — static vs dynamic batching, throughput vs latency tradeoff `**`

### Monitoring & Drift
- [ ] Evidently — data drift, prediction drift reports `**`
- [ ] LLM monitoring — hallucination rate, latency, token cost, input/output logging `***`
- [ ] Prometheus + Grafana — instrument a FastAPI endpoint, build latency dashboard `**`
- [ ] Alerting — set thresholds, trigger on accuracy drop or drift `**`

### Pipelines
- [ ] Prefect basics — flow, task, scheduling `**`
- [ ] Build scheduled pipeline: new docs → embed → update vector store `**`
- [ ] Kubernetes basics — pod, deployment, service, enough to read configs `*`
- [ ] Feature store concept — Feast, why needed at scale `*`

---

## PHASE 11 — Multimodal
> Extends your Document AI background directly

### Fundamentals
- [ ] CLIP — contrastive learning, shared image+text embedding space `***`
- [ ] ViT — image as patches, how vision became Transformer-friendly `***`
- [ ] How VLMs work — visual encoder (ViT) + projection layer + LLM decoder `***`
- [ ] Donut internals — Swin Transformer encoder + BART decoder (you've used it) `**`

### Models
- [ ] LayoutLM — combines text + layout (bounding box) + image for documents `***`
- [ ] LLaVA — open-source VLM, visual instruction tuning `**`
- [ ] PaliGemma or Qwen-VL — modern lightweight VLMs `*`
- [ ] Florence-2 — Microsoft's document understanding model `*`

### Applied
- [ ] Fine-tune LayoutLM on document classification task `***`
- [ ] Build VLM pipeline: image input → field extraction → structured JSON output `***`
- [ ] Compare Donut vs LayoutLM vs LLaVA on a financial document task `**`

---

## PHASE 12 — ML System Design
> Critical for senior interviews — every Sr. ML Engineer round includes at least one system design question

### The 8-Step Framework (use for every design question)
- [ ] Step 1: Clarify requirements — scale, latency, accuracy, constraints `***`
- [ ] Step 2: Define metrics — offline (AUC, F1) and online (CTR, revenue, latency) `***`
- [ ] Step 3: High-level architecture — data pipeline → model → serving → monitoring `***`
- [ ] Step 4: Data pipeline — collection, labeling, feature engineering, storage `***`
- [ ] Step 5: Model selection — start simple, justify complexity `***`
- [ ] Step 6: Training pipeline — distributed training, experiment tracking `**`
- [ ] Step 7: Serving — latency budget, batching, caching, A/B testing `***`
- [ ] Step 8: Monitoring — data drift, model drift, retraining triggers `***`

### Must-Know Design Problems
- [ ] **Recommendation System** — two-tower neural network, ALS, FAISS HNSW, reranking, cold start `***`
- [ ] **Search / RAG System** — document indexer, hybrid retrieval (RRF), cross-encoder reranking, 10K QPS scale `***`
- [ ] **Document Processing Pipeline** — classification + extraction + validation + human review queue `***`
- [ ] **Fraud Detection System** — real-time vs batch, feature store, class imbalance, explainability `**`
- [ ] **Ad Click-Through Rate (CTR) Prediction** — wide & deep, feature crosses, online learning `**`
- [ ] **Semantic Search / Embedding Pipeline** — embedding model choice, indexing, staleness, scale `**`

### Key Tradeoffs to Know Cold
- [ ] Precision vs recall — when to optimize which (fraud = high recall, search = high precision) `***`
- [ ] Latency vs accuracy — when to use simpler/faster model in serving path `***`
- [ ] Online vs batch inference — event-driven vs scheduled, when each applies `***`
- [ ] Buy vs build — when to use OpenAI API vs fine-tune vs train from scratch `***`
- [ ] Cold start problem — new user, new item — strategies for each `**`
- [ ] Feedback loops — how model predictions affect future training data `**`

### Practical Skills
- [ ] Draw architecture diagrams clearly — boxes, arrows, data flow `***`
- [ ] Size your system — storage, QPS, model size, GPU count estimates `**`
- [ ] Talk through tradeoffs out loud — interviewers care about reasoning, not just the answer `***`
- [ ] Practice 2 designs per week out loud, timed at 45 minutes `***`

---

## Capstone Projects (one per phase, goes on GitHub + resume)

- [ ] **Phase 7** — RAG system: financial doc Q&A + RAGAS evaluation `***`
- [ ] **Phase 8** — Fine-tuned Mistral/LLaMA on document extraction task `***`
- [ ] **Phase 9** — LangChain document agent: OCR + vector search + DB lookup `***`
- [ ] **Phase 10** — vLLM serving endpoint + Evidently monitoring dashboard `**`
- [ ] **Phase 11** — LayoutLM fine-tuned on financial document classification `**`
- [ ] **Phase 12** — System design mock: design a document Q&A system end-to-end (timed 45 min) `***`

---

## Estimated Timeline

| Phase | Topic | Duration | Priority |
|-------|-------|----------|----------|
| 1 | Machine Learning | 1 week (review) | `**` |
| 2 | Deep Learning | 1–2 weeks (fill gaps) | `***` |
| 3 | Computer Vision | 1 week (review + ViT) | `**` |
| 4 | NLP | 1–2 weeks (tokenization + seq models) | `***` |
| 5 | Transformer Internals | 2 weeks | `***` |
| 6 | Prompting + Alignment | 1 week | `**` |
| 7 | RAG Pipelines | 2–3 weeks | `***` |
| 8 | LLM Fine-tuning | 2–3 weeks | `***` |
| 9 | LLM Agents | 2 weeks | `***` |
| 10 | MLOps for LLMs | 2 weeks | `**` |
| 11 | Multimodal | 2 weeks | `**` |
| 12 | ML System Design | 2–3 weeks | `***` |
| — | Capstone Projects | ongoing | `***` |
| **Total** | | **~20–23 weeks** | |

---

## Quick Reference — Importance Summary

### `***` These alone will get you interviews
Backpropagation · Bias-variance tradeoff · Precision/Recall/F1/AUC · Vanishing gradients ·
BatchNorm/LayerNorm · LSTM gates · BERT fine-tuning · Tokenization (BPE/WordPiece) ·
Self-attention math · KV Cache · Quantization basics · RAG pipeline end-to-end ·
QLoRA fine-tuning · ReAct agents · vLLM serving · LayoutLM ·
LLM-as-Judge · RAGAS · ML System Design 8-step framework · Hallucination detection

### `**` Separate good candidates from great ones
GBM/XGBoost internals · Custom PyTorch training loops · Transfer learning strategy ·
Cross-encoder vs bi-encoder · Flash Attention · DPO · Hybrid search + reranking ·
LoRA rank/alpha tuning · LangGraph · Evidently monitoring ·
Speculative decoding · BERTScore · MCP server · System design tradeoffs (latency vs accuracy)

### `*` Nice to have — add after the above
DETR · FastText · ORPO · ALiBi · Contextual compression · Kubernetes · Feature stores · Florence-2 · MCP client

---

## High-Level Roadmap at a Glance

| Phases | Topic | Time | Focus |
|--------|-------|------|-------|
| 1–4 | ML → DL → CV → NLP | ~5–6 weeks | Mostly review + plug interview gaps |
| 5–6 | Transformer Internals + Alignment | ~3 weeks | Learn the how and why, not just usage |
| 7–9 | RAG → Fine-tuning → Agents | ~6–8 weeks | Build all 3 as GitHub projects |
| 10–11 | MLOps for LLMs + Multimodal | ~4 weeks | Extend existing MLOps strength |
| 12 | ML System Design | ~2–3 weeks | Practice out loud, timed — not just theory |
| **Total** | | **~20–23 weeks** | |

### Notes
- **Phases 1–4** — you have real production experience here. 1 week each max. Focus on being able to *explain* clearly, not re-learn.
- **Phases 5–9** — this is the core investment. Do not rush. Each phase builds on the previous.
- **Build as you go** — don't finish all theory before coding. Complete each phase's capstone project before moving to the next.
- **Resume update** — add each GitHub project to your resume as you finish it. Don't wait until the end.

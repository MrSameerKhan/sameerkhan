# Visual Diagram Plan — Full Repo

**Purpose:** Track which Mermaid diagrams to add to each file. Delete this file once all diagrams are done.
**Status key:** `[ ]` = pending · `[x]` = done · `[-]` = skipped (no diagram needed)
**Mermaid extension:** Markdown Preview Mermaid Support (VS Code) · preview with `Cmd+Shift+V`

---

## Visual Type Legend

| Code | Type | Best For |
|------|------|----------|
| `TL` | Timeline | Architecture evolution, model family history |
| `FC` | Flowchart | Pipelines, step-by-step processes |
| `SD` | Sequence Diagram | Loops, protocol flows, training loops |
| `BD` | Block / Architecture | Component layout with tensor shapes |
| `ST` | State Diagram | Lifecycle stages, state machines |
| `DT` | Decision Tree | "When to use X" — interview gold |
| `CM` | Comparison Matrix | Side-by-side method comparisons |
| `XY` | XY Chart | Loss curves, scaling laws |
| `GF` | Gradient Flow | Where gradients vanish/explode, backprop paths |
| `MM` | Mindmap | Topic overview, "which model" summary |

---

## 1. Machine Learning

### 01_fundamentals/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_statistics_foundations.md` | `DT` | Which statistical test to use (parametric vs non-parametric, paired vs unpaired) |
| `[-]` | `01b_probability_and_bayes_end_to_end.md` | — | Pure dry-run math, no diagram needed |
| `[-]` | `01c_statistics_end_to_end.md` | — | Pure dry-run math, no diagram needed |
| `[x]` | `02_eda.md` | `FC` | EDA pipeline: load → profile → missing → outliers → distributions → correlations |
| `[x]` | `03_feature_engineering.md` | `DT` | Feature type decision: numeric vs categorical vs text vs datetime → which transform |
| `[x]` | `04_model_evaluation.md` | `CM` | Metric selection: classification vs regression vs ranking × which metric when |

### 02_algorithms/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_linear_models.md` | `CM` | Ridge vs Lasso vs ElasticNet: when sparsity matters, when it doesn't |
| `[x]` | `02_tree_models.md` | `TL` + `DT` | Decision Tree → Bagging → RF → Boosting → XGBoost evolution; which ensemble for which scenario |
| `[x]` | `03_unsupervised_learning.md` | `DT` | Clustering algorithm selection: k-means vs DBSCAN vs hierarchical vs GMM |
| `[x]` | `04_probabilistic.md` | `BD` | Naive Bayes variants: Gaussian vs Multinomial vs Bernoulli |
| `[x]` | `05_time_series.md` | `FC` | Forecasting pipeline: stationarity → decompose → ARIMA/ML → evaluate |
| `[-]` | `05b_time_series_end_to_end.md` | — | Pure dry-run, no diagram needed |
| `[x]` | `07_reinforcement_learning.md` | `ST` | RL loop: state → policy → action → environment → reward → update |
| `[x]` | `10_reinforcement_learning_deep.md` | `SD` | Deep RL training loop: actor → critic → advantage → update |

---

## 2. Deep Learning

### 01_fundamentals/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_foundations.md` | `FC` | Full DL pipeline: data → embed → forward → loss → backward → update |
| `[x]` | `02_training_loop.md` | `SD` | Training loop sequence: batch → forward pass → loss → backward → optimizer step → repeat |
| `[x]` | `03_training_stability.md` | `GF` + `DT` | Gradient flow: where vanishing/exploding occurs; diagnosis decision tree |
| `[x]` | `04_generalization.md` | `CM` | Regularization techniques: Dropout vs L2 vs BatchNorm vs Early Stopping |
| `[x]` | `05_modern_components.md` | `CM` | Activation functions (ReLU/GELU/SwiGLU) + Normalization (BN/LN/RMSNorm) comparison |
| `[x]` | `06_specialized_losses.md` | `DT` | Loss function selection: classification vs regression vs contrastive vs ranking |

### 02_architectures/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `00_architecture_comparison.md` | `CM` | All architectures side by side: MLP vs CNN vs RNN vs Transformer vs Mamba |
| `[x]` | `01_mlp.md` | `BD` | MLP layers with neuron counts and tensor shapes |
| `[x]` | `02_cnn.md` | `BD` | Conv → Pool → Conv → Pool → FC with feature map shapes |
| `[x]` | `03_rnn_lstm_gru.md` | `CM` | RNN vs LSTM vs GRU vs Mamba: gates, gradient path, parallelism |
| `[x]` | `04_transformer.md` | `BD` | Canonical transformer block: MHA → Add&Norm → FFN → Add&Norm with tensor shapes |
| `[x]` | `05_generative.md` | `BD` | GAN: generator → discriminator loop; VAE: encoder → latent → decoder |
| `[x]` | `09_mixture_of_experts.md` | `FC` | Token → router → top-k expert gates → expert FFNs → combine |
| `[x]` | `10_quantization_theory.md` | `CM` | FP32 vs FP16 vs BF16 vs INT8 vs INT4: precision, range, use case |

---

## 3. Computer Vision

### 01_fundamentals/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_cnn_mechanics.md` | `BD` | Convolution operation: input → kernel → feature map with shapes |
| `[x]` | `02_cnn_architectures.md` | `TL` | LeNet (1998) → AlexNet (2012) → VGG → ResNet → EfficientNet → ViT |
| `[x]` | `03_cnn_end_to_end.md` | `GF` | Gradient flow through CNN layers: where vanishing is a risk |
| `[x]` | `04_vision_transformer_deep.md` | `BD` | Image → patches → embeddings → CLS token → transformer → classification |

### 02_applications/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_transfer_learning.md` | `DT` | Frozen vs partial fine-tune vs full fine-tune: data size × domain similarity |
| `[x]` | `02_object_detection.md` | `TL` + `CM` | RCNN → Fast → Faster → YOLO evolution; one-stage vs two-stage trade-offs |
| `[x]` | `03_segmentation.md` | `CM` | Semantic vs Instance vs Panoptic segmentation comparison |
| `[x]` | `04_explainability.md` | `CM` | GradCAM vs LIME vs SHAP vs Attention rollout |
| `[x]` | `05_self_supervised_vision.md` | `SD` | SimCLR/DINO contrastive loop: augment → encode → similarity → loss |

---

## 4. NLP

### 01_fundamentals/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_text_preprocessing.md` | `FC` | Raw text → tokenize → normalize → stopwords → stem/lemma → vectorize |
| `[x]` | `02_text_representations.md` | `TL` | BoW (1950s) → TF-IDF (1972) → BM25 (1994) → Word2Vec (2013) → BERT (2018) |
| `[x]` | `03_tokenization.md` | `FC` | BPE merge steps: character pairs → subwords → vocab |
| `[x]` | `03_bert_finetuning_deep.md` | `DT` | Which fine-tuning head for which task: classification vs NER vs QA vs generation |

### 02_embeddings/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_word_embeddings.md` | `BD` | Skip-gram: center word → hidden → context words |
| `[x]` | `02_sentence_embeddings.md` | `CM` | Bi-encoder vs Cross-encoder vs ColBERT: speed, accuracy, use case |
| `[x]` | `05_semantic_similarity.md` | `FC` | Hybrid retrieval pipeline: dense + BM25 → RRF → cross-encoder reranker |
| `[x]` | `06_contrastive_training.md` | `SD` | Contrastive training loop: anchor → positive/negative → loss → update |

### 03_sequence_models/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_rnn_to_attention.md` | `TL` `FC` `GF` `BD` `QC` `MM` | Timeline · RNN signal decay · LSTM gates · BiLSTM · Attention flow · Quadrant · Mindmap |
| `[x]` | `02_rnn_end_to_end.md` | `GF` | BPTT gradient decay: 100% → 40% → 16% → 0% color-coded per step |
| `[x]` | `03_lstm_end_to_end.md` | `GF` | RNN multiplicative path vs LSTM additive highway — side by side comparison |
| `[x]` | `04_gru_end_to_end.md` | `CM` | GRU vs LSTM: gate diagram + parameter table + gradient path |
| `[x]` | `05_attention_end_to_end.md` | `BD` | QKV projections with shapes · scores · softmax · weighted output |
| `[x]` | `06_transformer_end_to_end.md` | `BD` | Full transformer block: MHA → Add&Norm → FFN → Add&Norm with shapes |
| `[x]` | `07_decoding_strategies.md` | `DT` | Deterministic vs sampling → constrained/greedy/beam/temp+top-p |
| `[x]` | `08_scaling_laws_emergent.md` | `TL` | Kaplan → Chinchilla → GPT-3 → ChatGPT → test-time compute scaling |

### 04_applications/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_text_classification.md` | `DT` | Which classifier: dataset size × label type × latency requirement |
| `[x]` | `02_ner_and_tagging.md` | `SD` | IOB tagging pipeline: tokens → BERT → token classifier → IOB labels |
| `[x]` | `04_evaluation_metrics.md` | `CM` | Metric × task: classification vs NER vs generation vs retrieval |

---

## 5. Transformers

### 01_fundamentals/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_attention_mechanism.md` | `BD` | QKV projections with tensor shapes · scaling factor · causal mask |
| `[x]` | `02_transformer_architecture.md` | `BD` | Encoder block + decoder block + cross-attention · GPT/BERT/T5 variants noted |
| `[x]` | `04_pretraining_objectives.md` | `CM` | MLM vs CLM vs Span Corruption: attention direction · loss scope · best task |
| `[x]` | `05_vision_transformers.md` | `BD` | 196 patches → linear proj → CLS → 12 blocks → classification/dense heads |

### 02_models/

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_bert_family.md` | `TL` | BERT 2018 → RoBERTa → ALBERT → DistilBERT → ELECTRA → DeBERTa v3 2021 |
| `[x]` | `02_gpt_family.md` | `TL` | GPT-1 2018 → GPT-2 → GPT-3 → InstructGPT → GPT-4/LLaMA → DeepSeek-R1 2025 |
| `[x]` | `04_efficient_transformers.md` | `CM` | 4 axes: training / inference / serving / scale — technique per axis |
| `[x]` | `05_bert_end_to_end.md` | `BD` | Input → tokenize → 12 blocks → CLS/token repr → classification/NER/QA heads |
| `[x]` | `06_gpt_end_to_end.md` | `SD` | Autoregressive loop: context → causal attention → sample → append → KV cache |
| `[x]` | `07_t5_end_to_end.md` | `BD` | Task prefix → encoder → cross-attention → decoder with shapes |
| `[x]` | `08_modern_llm_architecture.md` | `CM` | GPT-2 → LLaMA: LayerNorm→RMSNorm · AbsPE→RoPE · GELU→SwiGLU · MHA→GQA |
| `[x]` | `09_parameter_efficient_tuning.md` | `DT` | Which PEFT: GPU budget → QLoRA / LoRA / DoRA / full fine-tune → alignment |
| `[x]` | `10_mixture_of_experts.md` | `FC` | Token → router → top-2 gate → expert FFNs → weighted sum · skipped experts noted |
| `[x]` | `12_constrained_decoding.md` | `ST` | JSON grammar states: open → key → colon → value → after_value · only valid tokens |
| `[x]` | `13_speculative_decoding.md` | `SD` | Draft proposes K tokens · target verifies in 1 pass · accept/reject/correct |
| `[x]` | `14_reasoning_models.md` | `ST` | problem → think (RLVR-trained loop) → verify → answer · difficulty-adaptive |

---

## 6. LLMs

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_prompting.md` | `DT` | Which prompting technique: zero-shot → few-shot → CoT → self-consistency → ReAct |
| `[x]` | `02_finetuning.md` | `DT` | Prompt vs RAG vs QLoRA vs full fine-tune vs SFT+DPO decision |
| `[x]` | `02b_finetuning_end_to_end.md` | `FC` | SFT pipeline: base → dataset → train → eval → alignment → merge → serve |
| `[x]` | `03_alignment.md` | `SD` | RLHF 3-stage sequence: human labels → RM training → PPO with KL penalty |
| `[x]` | `03b_alignment_end_to_end.md` | `GF` | DPO derivation chain: RLHF → closed-form r → BT model → DPO loss |
| `[x]` | `05_vllm_internals.md` | `BD` + `SD` | PagedAttention memory layout (naive vs paged); continuous batching loop |
| `[x]` | `06_alignment_follow_ups.md` | `DT` | DPO vs ORPO vs KTO vs IPO vs GRPO decision tree |
| `[x]` | `07_dataset_preparation.md` | `FC` | Data pipeline: raw → format → quality filter → chat_template → SFT/DPO/KTO/tools |

---

## 7. RAG

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_rag.md` | `SD` | Full query flow: offline indexing + online embed → ANN → BM25 → RRF → rerank → LLM |
| `[x]` | `01b_rag_end_to_end.md` | `FC` | Indexing (offline) + retrieval (online) pipeline side by side with all components |
| `[x]` | `02_rag_pipeline.md` | `DT` | Chunking strategy decision: table → code → long-form → sentence → fixed-size |
| `[x]` | `03_indirect_prompt_injection.md` | `FC` | 6 defense layers stacked with strength ratings: sanitize → spotlight → capability isolation → output validate → structured output → dual-LLM |

---

## 8. Agents

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_agents.md` | `SD` | ReAct loop: Thought → Action → Tool call → Observation → repeat → Answer |
| `[x]` | `04_langgraph_deep.md` | `ST` | LangGraph state machine: agent → tools → agent loop + HITL interrupt branch |
| `[x]` | `05_agent_memory.md` | `MM` | Memory hierarchy mindmap: working → short-term → episodic/semantic/procedural |
| `[x]` | `07_multi_agent_orchestration.md` | `BD` | Supervisor/Worker · Pipeline · Debate patterns side by side |
| `[x]` | `08_mcp_protocol_deep.md` | `SD` | MCP full flow: handshake → discovery → tool call → result injection |
| `[x]` | `09_agent_evaluation.md` | `CM` | Outcome eval vs process eval: what each captures and why both matter |

---

## 9. Multimodal

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_vision_language_models.md` | `BD` | CLIP dual encoder: image+text → L2-norm embeddings → cosine sim → InfoNCE |
| `[x]` | `02_document_ai.md` | `FC` | Traditional pipeline vs VLM pipeline · LayoutLM vs Donut vs GPT-4V paths |
| `[x]` | `03_vision_transformers.md` | `BD` | ViT: 224×224 image → 196 patches → linear proj → CLS + positional embed → blocks |
| `[x]` | `05_donut_end_to_end.md` | `BD` | Image → Swin encoder → cross-attention → BART decoder → JSON (no OCR) |
| `[x]` | `07_modern_vlms_2024_2025.md` | `CM` | VLM architecture pattern: vision encoder → projector → LLM + model comparison table |
| `[x]` | `08_audio_multimodal.md` | `FC` | Whisper: audio → STFT → mel spectrogram → encoder → cross-attention → decoder |
| `[x]` | `09_vlm_hallucination_mitigation.md` | `DT` | Hallucination type → mitigation chain → confidence gate → accept/escalate |

---

## 10. MLOps

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `03_monitoring_and_drift.md` | `DT` | Alert → infra/data drift/concept drift → diagnosis → retrain/rollback/monitor |
| `[x]` | `06_multi_gpu_multi_server_training.md` | `CM` | DDP vs FSDP vs DeepSpeed ZeRO-1/2/3: what's sharded, memory, communication |
| `[x]` | `08_model_registry_end_to_end.md` | `ST` | Model lifecycle: training → logged → registered → staging → production → archived |
| `[x]` | `10_serving_optimization.md` | `CM` | PyTorch → ONNX → INT8 → TensorRT latency chain + vLLM/AWQ for LLMs |

---

## 11. System Design

| Done | File | Visual | Shows |
|------|------|--------|-------|
| `[x]` | `01_ml_system_design_framework.md` | `FC` | 8-step interview framework with time allocations + interview tip |
| `[x]` | `02_recommendation_system.md` | `BD` | Offline two-tower + ANN → online retrieval(1K) → ranking(100) → reranking(20) |
| `[x]` | `03_search_and_rag_system.md` | `BD` | Search stack: query → embed → FAISS/pgvector → BM25 → RRF → rerank → LLM |
| `[x]` | `07_feed_ranking_system.md` | `BD` | Candidate gen → feature store → LTR model → diversity/freshness → serve |
| `[x]` | `08_model_registry_end_to_end.md` | `ST` | Same lifecycle as 10.mlops/08 |

---

## Summary

| Folder | Files needing diagram | Skipped |
|--------|-----------------------|---------|
| 1.machine learning | 10 | 3 (dry-run math) |
| 2.deep learning | 13 | 4 (READMEs / dry-run) |
| 3.computerVision | 9 | 1 (README) |
| 4.nlp | 15 | 5 (dry-run) |
| 5.transformers | 15 | 3 (dry-run) |
| 6.llms | 8 | 2 (dry-run) |
| 7.rag | 4 | 0 |
| 8.agents | 6 | 4 (dry-run / code) |
| 9.multimodal | 7 | 3 (dry-run) |
| 10.mlops | 4 | 10 (code-heavy) |
| 11.system_design | 5 | 8 (code-heavy) |
| **Total** | **~96 diagrams** | **~43 skipped** |

---

## Execution Order (by interview priority)

1. `4.nlp/03_sequence_models/` — ✅ 01 done, 5 files remaining
2. `6.llms/` — alignment, prompting, fine-tuning (most common senior questions)
3. `8.agents/` — ReAct, LangGraph (hot topic in 2025–26 interviews)
4. `5.transformers/02_models/` — BERT/GPT/T5 architecture walkthroughs
5. `7.rag/` — RAG pipeline and injection defenses
6. `9.multimodal/` — Sameer's core strength (Document AI)
7. `2.deep learning/` — fundamentals and architectures
8. `11.system_design/` — system design patterns
9. `10.mlops/` — model lifecycle and serving
10. `1.machine learning/` — classical ML (lower priority)
11. `3.computerVision/` — CV applications (lower priority unless CV role)

---

*Delete this file once all `[x]` boxes are checked.*

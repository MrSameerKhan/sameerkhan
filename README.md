# sameerkhanAI

Personal ML/DL mastery repo — built for deep understanding, interview prep, and long-term reference.
Covers the full stack from classical ML to LLM systems and production MLOps.

---

## Structure

| # | Topic | What's Inside |
|---|-------|---------------|
| 1 | [Machine Learning](1.machine%20learning/) | Fundamentals, algorithms, hands-on code |
| 2 | [Deep Learning](2.deep%20learning/) | Neural networks, training systems, architectures |
| 3 | [Computer Vision](3.computerVision/) | CNNs, object detection, segmentation, transfer learning |
| 4 | [NLP](4.nlp/) | Embeddings, sequence models, attention, applications |
| 5 | [Transformers](5.transformers/) | Attention mechanism, BERT/GPT/T5 families, efficient transformers |
| 6 | [LLMs](6.llms/) | Prompting, fine-tuning, alignment, RAG, agents, evaluation |
| 7 | [Multimodal](7.multimodal/) | CLIP, VLMs, Document AI (OCR, LayoutLM, Donut) |
| 8 | [MLOps](8.mlops/) | Experiment tracking, serving, monitoring, pipelines & infra |
| 9 | [System Design](9.system_design/) | ML design framework, recommendation, search/RAG, document processing |

---

## Coverage

**Fundamentals**
- Classical ML algorithms, bias-variance, regularization
- Neural network training: backprop, optimizers, weight init, batch norm
- CNN fundamentals: convolution, pooling, receptive field

**Modern DL**
- Transformer architecture: attention, positional encoding, Flash Attention
- BERT family (MLM, DeBERTa), GPT family (LLaMA2/3, GQA, KV cache)
- Efficient transformers: LoRA, QLoRA, quantization (INT8/GPTQ), MoE

**LLM Systems**
- Prompting: CoT, self-consistency, structured output
- Fine-tuning: SFT, QLoRA pipeline (LLaMA2-7b, NF4, LoRA r=64)
- Alignment: RLHF, PPO, DPO, ORPO, Constitutional AI
- RAG: hybrid retrieval, reranking, HyDE, RAGAS evaluation
- Agents: ReAct, tool use, MCP, multi-agent patterns

**Production**
- Serving: ONNX, vLLM (PagedAttention), FastAPI, TensorRT
- Monitoring: PSI/KS drift detection, Evidently, Prometheus + Grafana
- Pipelines: Prefect, Feast feature store, Docker + K8s, GitHub Actions CI/CD
- System design: recommendation (two-tower + FAISS), search/RAG at scale, document processing at 100K docs/day

---

## Code (`zcode/`)

Each topic folder contains a `code/` directory with working implementations:

```
6.llms/code/
├── finetuning/Fine_tune_Llama_2.ipynb   # QLoRA fine-tuning pipeline
├── rag/                                  # FAISS, ObjectBox, Cassandra RAG apps
└── agents/                               # LangChain agent patterns
```

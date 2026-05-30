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

---

## New Practice Series — Phase 05 onwards

Problem-first, use-case-driven sessions. Every session answers: what problem, who has it, when you'd use it, how you solve it.

**Narrative arc:** Apply pretrained models → Use LLM APIs → Augment with knowledge → Orchestrate with tools → Customize weights → Own your differentiator (Document AI)

**File structure:** Single file = learning a pattern. Multi-file = building something shippable.

**Theory link rule:** Line 1 of every session file: `# Theory: ../path/to/theory_file.md` — keeps code and theory wired together.

### Phase 05: Transformers — "Apply pretrained models to real NLP tasks"
> Single file per session — applying HuggingFace pretrained models, no custom architecture

| # | Session | Real use case | Problem solved | Who uses it | Status |
|---|---------|--------------|---------------|-------------|--------|
| 01 | `01_bert_classification.py` | Loan feedback sentiment classifier | Label thousands of customer reviews manually → automate with BERT | ML engineer at any bank / fintech | ✅ Run |
| 02 | `02_bert_ner.py` | Financial news entity extractor | Extract company names, amounts, dates from news at scale | Data engineer, quant team | ✅ Run |
| 03 | `03_bert_qa.py` | Policy document QA | "What is the max loan amount?" — answered from PDF without search infra | Document AI engineer | 🔧 Code-built |
| 04 | `04_bart_summarization.py` | Earnings report summarizer | 50-page report → 3 bullet points for analysts in seconds | Research analyst, fintech | 🔧 Code-built |
| 05 | `05_t5_translation.py` | Multilingual document processor | Arabic financial docs → English | Any company with multilingual docs | 🔧 Code-built |
| 06 | `06_gpt2_generation.py` | Synthetic training data generator | Need 10K training examples, only have 500 real ones | ML engineer with data scarcity problem | 🔧 Code-built |
| 07 | `07_sentence_transformers.py` | Semantic document search | "Find all contracts similar to this one" — keyword search fails | Search engineer — bridge to RAG | 🔧 Code-built |

### Phase 06: LLMs Core — "Use LLM APIs to build real features"
> Single file per session — API calls and prompting patterns

| # | Session | Real use case | Problem solved | Who uses it | Status |
|---|---------|--------------|---------------|-------------|--------|
| 01 | `01_prompt_engineering.py` | Domain-aware customer support bot | Base LLM gives generic answers → few-shot makes it domain-specific instantly | Any team before spending on fine-tuning | 🔧 Code-built |
| 02 | `02_structured_extraction.py` | Invoice / contract parser | Unstructured PDF text → clean validated JSON (Pydantic + Instructor) | Document AI engineer | 🔧 Code-built |
| 03 | `03_llm_evaluation.py` | A/B test two model versions | Can't ship without proof v2 > v1 → ROUGE + LLM-as-judge pipeline | Every ML engineer before deployment | 🔧 Code-built |

### Phase 07: RAG — "Augment LLM with external knowledge"
> Sessions 01-04 single file · Session 05 multi-file (pipeline + server are separate concerns)

| # | Session | Real use case | Problem solved | Who uses it | Structure | Status |
|---|---------|--------------|---------------|-------------|-----------|--------|
| 01 | `01_basic_rag.py` | Policy document QA bot | LLM hallucinates policy details → ground answers in real docs | Any company with internal knowledge base | single file | 🔧 Code-built |
| 02 | `02_chunking_strategies.py` | Long financial doc processor | Wrong chunk size → missed context or irrelevant noise | Every RAG builder | single file | 🔧 Code-built |
| 03 | `03_advanced_rag.py` | High-precision financial QA | Basic RAG retrieves wrong chunks → reranking + hybrid search fixes it | ML engineer shipping RAG to prod | single file | 🔧 Code-built |
| 04 | `04_rag_evaluation.py` | Prove RAG actually works | "Is our RAG better than no RAG?" → RAGAS: faithfulness + relevance scores | ML engineer, product team | single file | 🔧 Code-built |
| 05 | `05_production_rag/` | Prod-grade RAG with semantic cache + REST API | Repeated queries waste LLM budget → semantic cache cuts costs 30-40% | Senior ML / MLOps engineer | `pipeline.py` · `serve.py` | 🔧 Code-built |

### Phase 08: Agents — "Orchestrate LLM with tools and memory"
> Sessions 01-02 single file · Sessions 03-04 multi-file (graph, tools, entry point are distinct)

| # | Session | Real use case | Problem solved | Who uses it | Structure | Status |
|---|---------|--------------|---------------|-------------|-----------|--------|
| 01 | `01_react_agent.py` | Multi-step research assistant | Single LLM call can't search + calculate + synthesize → ReAct loop can | Any LLM product team | single file | 🔧 Code-built |
| 02 | `02_tool_calling.py` | Financial data agent | Connect LLM to live APIs: calculator, account lookup, eligibility check | ML engineer building LLM features | single file | 🔧 Code-built |
| 03 | `03_langgraph_agent/` | Production workflow agent | Prototype agent loses state + can't HITL → LangGraph: checkpoints + multi-turn + streaming | Senior engineer shipping agents to prod | `graph.py` · `tools.py` · `run.py` | 🔧 Code-built |
| 04 | `04_document_agent/` | **Portfolio project:** Mortgage document pipeline — classify → extract → policy → eligibility → HITL → report | One monolithic agent fails on edge cases → specialist agents + supervisor + conditional HITL approval | ML architect at fintech / bank — resume bullet | `agents.py` · `graph.py` · `run.py` | 🔧 Code-built |

### Phase 09: Fine-tuning — "Change model weights for your domain"
> Sessions 01, 03-04, 06 run on MPS · Session 02 needs CUDA · Session 05 needs Linux+CUDA

| # | Session | Real use case | Problem solved | Who uses it | Structure | Status |
|---|---------|--------------|---------------|-------------|-----------|--------|
| 01 | `01_lora_finetune.py` | Domain-adapted financial assistant | Base LLM gives generic answers → LoRA adapts in 0.24% of params | ML engineer with domain data | single file | 🔧 Code-built |
| 02 | `02_qlora_finetune.py` | Same on consumer GPU (8-16GB) | LoRA needs float32 in memory → QLoRA: 4-bit NF4 + adapter | Any engineer without A100s | single file | 🔧 Code-built |
| 03 | `03_dataset_prep.py` | Build training data from raw docs | Have 10K policy docs, no pairs → LLM-generated SFT + DPO dataset pipeline | ML engineer starting fine-tuning | single file | 🔧 Code-built |
| 04 | `04_dpo_alignment.py` | Make fine-tuned model less robotic | SFT model accurate but dry → DPO on (chosen, rejected) pairs | Alignment-aware ML engineer | single file | 🔧 Code-built |
| 05 | `05_vllm_serving/` | Serve fine-tuned model at scale | `.generate()` = 1 req at a time → vLLM PagedAttention: 50× throughput | MLOps / deployment engineer | `server.py` · `client.py` | 🔧 Code-built |
| 06 | `06_llm_monitoring.py` | Production LLM observability | No visibility into latency/cost/drift → Prometheus + PSI drift + structured traces | Senior ML / MLOps engineer | single file | 🔧 Code-built |

### Phase 10: Document AI — "Own your differentiator"
> Your biggest interview edge — 8 years of production Document AI, now in code

| # | Session | Real use case | Problem solved | Who uses it | Structure | Status |
|---|---------|--------------|---------------|-------------|-----------|--------|
| 01 | `01_layoutlm_extraction.py` | Invoice / form key-value extraction | Rule-based fails on varied layouts → LayoutLMv3 learns text + bbox + image jointly | Document AI engineer (Nanonets, Docsumo, ICE) | single file | 🔧 Code-built |
| 02 | `02_donut_parsing.py` | OCR-free document understanding | OCR errors cascade → Donut reads image → JSON end-to-end with no OCR step | Senior Document AI engineer | single file | 🔧 Code-built |
| 03 | `03_colpali_rag.py` | Vision-RAG on document images | Text RAG misses tables/charts → ColPali embeds 1030 patches per page, MaxSim retrieval | Frontier Document AI engineer | single file | 🔧 Code-built |
| 04 | `04_document_pipeline.py` | **Portfolio:** Full ICE-style pipeline | Components work alone but not connected → ingest → OCR → classify → extract → answer | **Your differentiator** — pin to GitHub, update resume | single file | 🔧 Code-built |

**Series totals:** 7 + 3 + 5 + 4 + 6 + 4 = **29 sessions across 6 phases.**

---

### Portfolio milestones (when to update resume)

| After completing | Resume / LinkedIn action |
|-----------------|--------------------------|
| Phase 07/05 `production_rag` | Update RAG project → v2 with RAGAS eval + semantic cache |
| Phase 08/04 `document_agent` | Add: "Built LangGraph document agent — classify → retrieve → extract → answer, HITL approval" |
| Phase 09/02 `qlora_finetune` | Add: "Fine-tuned TinyLlama with QLoRA on synthetic banking dataset, McNemar-significant improvement" |
| Phase 09/06 `llm_monitoring` | Add: "vLLM serving + Prometheus + Evidently drift monitoring pipeline" |
| Phase 10/04 `document_pipeline` | Pin to GitHub — this is your biggest differentiator demo |

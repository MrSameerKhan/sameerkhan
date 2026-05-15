# Coding Practice Sequence — Sequence Models → Transformers → Prompting → LLMs → RAG → Agents

> Pure coding. One session at a time. Build each piece from scratch where useful, use libraries where production-realistic.

> ⭐⭐⭐ Must-build (interview-critical, used in production) &nbsp; ⭐⭐ Important (separates strong candidates) &nbsp; ⭐ Nice-to-have

---

## How to Use This File

- Work through items **top to bottom** — each builds on the previous
- Each row = one coding session (1–2 hours)
- Save code under suggested folder so it's reviewable later
- Tick the checkbox when working code + understanding both exist

---

## Session Folder Pattern

Each session lives in its own folder with 4 files (established in Session 1):

```
code_practice/<phase>/<NN_session_name>/
├── model.py          # Architecture & math
├── train.py          # Training script (saves checkpoint)
├── predict.py        # Inference CLI (loads checkpoint)
├── all_details.md    # Full documentation: objective, shapes, how to run
└── checkpoints/      # Auto-created on first train
```

Shared synthetic dataset (Acme Financial Services) at `code_practice/shared_dataset.py` is used across all sessions for consistency.

---

## Phase 1 — Sequence Models (Foundation Before Transformers)

> Goal: code RNN/LSTM/GRU/BiLSTM cleanly. Build attention on top of BiLSTM — that's the bridge into transformers.

| # | Topic | What to Code | Suggested Folder | Priority | Done |
|---|---|---|---|---|---|
| 1 | Vanilla RNN cell from scratch | NumPy: forward pass + BPTT, char-level LM on Acme corpus | `code_practice/01_seq_models/01_rnn/` | ⭐⭐⭐ | [x] |
| 2 | RNN with PyTorch | `nn.RNN` — same task, batched, GPU-ready | `code_practice/01_seq_models/02_rnn_torch/` | ⭐⭐⭐ | [x] |
| 3 | LSTM cell from scratch | All 4 gates (forget, input, output, cell) — NumPy | `code_practice/01_seq_models/03_lstm/` | ⭐⭐⭐ | [x] |
| 4 | LSTM with PyTorch | `nn.LSTM` — compare perplexity vs RNN on same Acme corpus | `code_practice/01_seq_models/04_lstm_torch/` | ⭐⭐⭐ | [x] |
| 5 | GRU cell from scratch | 2 gates (reset, update) — show why simpler than LSTM | `code_practice/01_seq_models/05_gru/` | ⭐⭐ | [x] |
| 6 | GRU vs LSTM head-to-head | Same task, compare params, training time, accuracy | `code_practice/01_seq_models/06_gru_vs_lstm/` | ⭐⭐ | [x] |
| 7 | BiLSTM for NER | Token classification on Acme NER data | `code_practice/01_seq_models/07_bilstm_ner/` | ⭐⭐⭐ | [x] |
| 8 | Bahdanau attention on BiLSTM | Add additive attention — direct bridge to transformer attention | `code_practice/01_seq_models/08_bilstm_attention/` | ⭐⭐⭐ | [x] |
| 9 | Seq2Seq with attention | Encoder-decoder over Acme corpus | `code_practice/01_seq_models/09_seq2seq/` | ⭐⭐ | [x] |

**Phase 1 exit test:** Can you write LSTM gates from blank file AND explain why attention beats fixed-size hidden state? If yes → Phase 2 (transformers will feel natural).

---

## Phase 2 — Transformers (Build the Engine)

> Goal: understand attention end-to-end by writing it. Don't move on until you can rebuild from blank file.

| # | Topic | What to Code | Suggested Folder | Priority | Done |
|---|---|---|---|---|---|
| 1 | Tokenizer (BPE) from scratch | Write BPE training + encoding on a small corpus | `code_practice/02_transformers/01_bpe/` | ⭐⭐⭐ | [x] |
| 2 | Scaled dot-product attention | NumPy then PyTorch — single head, with masking | `code_practice/02_transformers/02_attention/` | ⭐⭐⭐ | [x] |
| 3 | Multi-head attention | Wrap step 2 — split, compute, concat, project | `code_practice/02_transformers/03_mha/` | ⭐⭐⭐ | [x] |
| 4 | Positional encoding | Sinusoidal + learned + RoPE (pick 2) | `code_practice/02_transformers/04_pos_enc/` | ⭐⭐ | [x] |
| 5 | Transformer block (encoder) | LayerNorm + MHA + FFN + residual | `code_practice/02_transformers/05_encoder_block/` | ⭐⭐⭐ | [x] |
| 6 | Mini Transformer encoder | Stack 4 blocks, add embedding + classification head | `code_practice/02_transformers/06_mini_transformer/` | ⭐⭐⭐ | [x] |
| 7 | Causal masking + decoder block | Add causal mask, build decoder block | `code_practice/02_transformers/07_decoder_block/` | ⭐⭐⭐ | [x] |
| 8 | Tiny GPT (nanoGPT-style) | Train on tiny shakespeare or your own text | `code_practice/02_transformers/08_tiny_gpt/` | ⭐⭐⭐ | [x] |
| 9 | KV Cache implementation | Modify step 8 to cache K, V at inference | `code_practice/02_transformers/09_kv_cache/` | ⭐⭐⭐ | [x] |
| 10 | Cross-attention + encoder-decoder | Decoder attends to encoder output (T5/BART/Donut style) | `code_practice/02_transformers/10_cross_attention/` | ⭐⭐⭐ | [x] |
| 11 | Load pretrained from HuggingFace | Inspect a real BERT/GPT-2 — match shapes to your impl | `code_practice/02_transformers/11_hf_load/` | ⭐⭐ | [ ] |

**Phase 2 exit test:** Can you sketch attention math on a whiteboard AND code a working transformer block from blank? If yes → move on.

---

## Phase 3 — Prompting (LLM Calls First, No Training)

> Goal: get fluent calling LLMs and shaping output. This is fastest payoff — most JDs ask for prompt engineering.

| # | Topic | What to Code | Suggested Folder | Priority | Done |
|---|---|---|---|---|---|
| 1 | First LLM call (3 providers) | Same prompt → OpenAI, Anthropic, Ollama (local) | `code_practice/03_prompting/01_first_call/` | ⭐⭐⭐ | [ ] |
| 2 | Few-shot prompting | Sentiment classifier — 0-shot vs 3-shot, compare | `code_practice/03_prompting/02_few_shot/` | ⭐⭐⭐ | [ ] |
| 3 | Chain-of-Thought | Math word problem — direct vs CoT | `code_practice/03_prompting/03_cot/` | ⭐⭐⭐ | [ ] |
| 4 | Self-consistency | Sample N=5 CoT chains, majority vote | `code_practice/03_prompting/04_self_consistency/` | ⭐⭐ | [ ] |
| 5 | Structured output (JSON) | Pydantic schema → enforce JSON output, parse, validate | `code_practice/03_prompting/05_json_output/` | ⭐⭐⭐ | [ ] |
| 6 | Function calling | Tool definition + LLM picks tool + you execute | `code_practice/03_prompting/06_function_call/` | ⭐⭐⭐ | [ ] |
| 7 | ReAct pattern (manual loop) | Thought → Action → Observation loop, hardcoded tools | `code_practice/03_prompting/07_react_manual/` | ⭐⭐⭐ | [ ] |
| 8 | System prompt engineering | A/B test 3 system prompts on same task, log differences | `code_practice/03_prompting/08_system_prompts/` | ⭐⭐ | [ ] |
| 9 | Prompt injection defense | Build a safe prompt, then attack it, then defend it | `code_practice/03_prompting/09_injection/` | ⭐⭐ | [ ] |
| 10 | Streaming output | Token-by-token streaming with `stream=True` | `code_practice/03_prompting/10_streaming/` | ⭐ | [ ] |

**Phase 3 exit test:** Can you take any task, pick the right prompting technique, and explain why? If yes → move on.

---

## Phase 4 — LLMs (Loading, Inference, Fine-tuning)

> Goal: load a real LLM, control generation, then fine-tune one with QLoRA. This is the biggest resume gap right now.

| # | Topic | What to Code | Suggested Folder | Priority | Done |
|---|---|---|---|---|---|
| 1 | Load HF model + generate | TinyLlama / Phi-2 — `.generate()` with explanation | `code_practice/04_llms/01_load_generate/` | ⭐⭐⭐ | [ ] |
| 2 | Decoding strategies | Greedy vs beam vs top-k vs top-p — same prompt, compare | `code_practice/04_llms/02_decoding/` | ⭐⭐⭐ | [ ] |
| 3 | 8-bit quantization | Load model in INT8 with `bitsandbytes`, measure VRAM | `code_practice/04_llms/03_quant_8bit/` | ⭐⭐ | [ ] |
| 4 | 4-bit NF4 quantization | Load in 4-bit, inference, compare quality | `code_practice/04_llms/04_quant_4bit/` | ⭐⭐⭐ | [ ] |
| 5 | Dataset prep (Alpaca format) | Format your data as ChatML / Alpaca instruction pairs | `code_practice/04_llms/05_dataset_prep/` | ⭐⭐⭐ | [ ] |
| 6 | LoRA from scratch | Implement LoRA (B·A) wrapper around `nn.Linear` | `code_practice/04_llms/06_lora_scratch/` | ⭐⭐ | [ ] |
| 7 | QLoRA with PEFT | Real fine-tune TinyLlama on small dataset, T4/Colab | `code_practice/04_llms/07_qlora_train/` | ⭐⭐⭐ | [ ] |
| 8 | Merge + push to HF Hub | Merge LoRA → base, save, push, load back, test | `code_practice/04_llms/08_merge_push/` | ⭐⭐⭐ | [ ] |
| 9 | Eval base vs fine-tuned | 20 prompts, score both — show before/after improvement | `code_practice/04_llms/09_eval_compare/` | ⭐⭐⭐ | [ ] |
| 10 | DPO training (small) | One round of DPO on a chosen/rejected pairs dataset | `code_practice/04_llms/10_dpo/` | ⭐ | [ ] |
| 11 | Serve fine-tuned model with FastAPI | Wrap your model as `/generate` endpoint, test with curl | `code_practice/04_llms/11_fastapi_serve/` | ⭐⭐ | [ ] |

**Phase 4 milestone:** Push a fine-tuned model to HuggingFace Hub. Add to resume as **Project 2** (Daily Hub → "LLM Fine-tuning POC").

---

## Phase 5 — RAG (Retrieval First, Generation Second)

> Goal: production-grade RAG — your existing `projects/rag_system/` is the baseline. Build each piece below as a standalone learning exercise, then upgrade the project.

| # | Topic | What to Code | Suggested Folder | Priority | Done |
|---|---|---|---|---|---|
| 1 | Chunking strategies | Fixed-size, recursive, semantic — same doc, compare | `code_practice/05_rag/01_chunking/` | ⭐⭐⭐ | [ ] |
| 2 | Embeddings | sentence-transformers — encode, normalize, cosine sim | `code_practice/05_rag/02_embeddings/` | ⭐⭐⭐ | [ ] |
| 3 | FAISS basics | `IndexFlatIP`, `IndexIVFFlat`, `IndexHNSWFlat` — compare | `code_practice/05_rag/03_faiss/` | ⭐⭐⭐ | [ ] |
| 4 | BM25 retrieval | `rank_bm25` — sparse retrieval over same corpus | `code_practice/05_rag/04_bm25/` | ⭐⭐ | [ ] |
| 5 | Hybrid search + RRF | Dense + BM25 → Reciprocal Rank Fusion → top-K | `code_practice/05_rag/05_hybrid_rrf/` | ⭐⭐⭐ | [ ] |
| 6 | Cross-encoder reranking | Initial top-50 → BGE-reranker → top-5 | `code_practice/05_rag/06_rerank/` | ⭐⭐⭐ | [ ] |
| 7 | HyDE | LLM generates hypothetical answer → embed that → retrieve | `code_practice/05_rag/07_hyde/` | ⭐⭐ | [ ] |
| 8 | RAG end-to-end | Tie 1–6 together, FastAPI endpoint, source attribution | `code_practice/05_rag/08_rag_full/` | ⭐⭐⭐ | [ ] |
| 9 | RAGAS evaluation | Build 30-Q eval set, run faithfulness + relevancy + recall | `code_practice/05_rag/09_ragas/` | ⭐⭐⭐ | [ ] |
| 10 | Upgrade `projects/rag_system/` | Add hybrid search + reranker to existing project | `projects/rag_system/` | ⭐⭐⭐ | [ ] |

**Phase 5 milestone:** RAG project upgraded with hybrid + reranking + RAGAS. Update resume bullet (Daily Hub).

---

## Phase 6 — Agents (Orchestrate Everything)

> Goal: from manual ReAct → LangChain → LangGraph. LangGraph is mandatory at Infosys, TCS, several others (JD Analysis).

| # | Topic | What to Code | Suggested Folder | Priority | Done |
|---|---|---|---|---|---|
| 1 | ReAct from scratch (no framework) | Pure Python ReAct loop with 2 tools (calculator + search) | `code_practice/06_agents/01_react_scratch/` | ⭐⭐⭐ | [ ] |
| 2 | LangChain tools | `@tool` decorator, custom tools, schema validation | `code_practice/06_agents/02_lc_tools/` | ⭐⭐⭐ | [ ] |
| 3 | Wrap your RAG as a LangChain tool | Turn `projects/rag_system/` retriever into an agent tool | `code_practice/06_agents/03_rag_as_tool/` | ⭐⭐⭐ | [ ] |
| 4 | LangChain AgentExecutor | Wire up tools (incl. RAG) + LLM → run multi-step task | `code_practice/06_agents/04_lc_executor/` | ⭐⭐⭐ | [ ] |
| 5 | Memory: ConversationBuffer | Multi-turn agent that remembers prior turns | `code_practice/06_agents/05_memory_buffer/` | ⭐⭐ | [ ] |
| 6 | Memory: VectorStore | Long-term memory backed by FAISS | `code_practice/06_agents/06_memory_vector/` | ⭐⭐ | [ ] |
| 7 | LangGraph: 2-node graph | Orchestrator → worker → end. State as TypedDict | `code_practice/06_agents/07_lg_basic/` | ⭐⭐⭐ | [ ] |
| 8 | LangGraph: conditional routing | If confidence < 0.7 → human review node | `code_practice/06_agents/08_lg_routing/` | ⭐⭐⭐ | [ ] |
| 9 | LangGraph: cycles + checkpointing | Loop with state persistence (SQLite checkpointer) | `code_practice/06_agents/09_lg_cycles/` | ⭐⭐ | [ ] |
| 10 | Document agent (capstone) | Classify → extract → validate → answer (mirrors ICE pipeline) | `projects/doc_agent/` | ⭐⭐⭐ | [ ] |
| 11 | MCP server basics | Expose 2 tools via FastMCP, connect from a client | `code_practice/06_agents/11_mcp/` | ⭐ | [ ] |

**Phase 6 milestone:** Document agent deployed (HF Spaces / Gradio). Add as **Project 3** to resume.

---

## Combined Sequence at a Glance

```
Phase 1: Sequence Models  (9 sessions)  →  RNN/LSTM/GRU/BiLSTM + attention bridge
Phase 2: Transformers     (11 sessions) →  understand the engine (incl. cross-attention)
Phase 3: Prompting        (10 sessions) →  call LLMs fluently
Phase 4: LLMs             (11 sessions) →  fine-tune + serve your own model    [Resume Project 2]
Phase 5: RAG              (10 sessions) →  upgrade existing RAG project         [Resume Project 1++]
Phase 6: Agents           (11 sessions) →  build document agent                 [Resume Project 3]
```

**Total:** 62 coding sessions · ~1–2 hours each · ~8–12 weeks if 5 sessions/week

---

## Pre-flight — One-time setup

| Item | Command |
|---|---|
| Python env | `python -m venv .venv && .venv\Scripts\activate` |
| Core libs | `pip install torch transformers datasets peft trl bitsandbytes accelerate` |
| Seq models | `pip install seqeval` (for NER eval in Phase 1) |
| RAG libs | `pip install sentence-transformers faiss-cpu rank_bm25 ragas` |
| Agent libs | `pip install langchain langchain-community langgraph langchain-openai` |
| Local LLM | `ollama pull llama3.2:1b` (already used in your RAG project) |
| Keys | `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` (Ollama works without keys) |

---

## How We'll Work Through This

1. You pick the next unchecked row
2. I write the code with you (or you try first, I review)
3. We discuss what changed / what you learned
4. Tick the box, move on
5. Don't skip ahead — each topic builds the next

**Start point:** Phase 2, Session 11 — Load pretrained from HuggingFace.

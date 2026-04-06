# Learning Roadmap — What's Done and What's Missing

Covers all three folders: `4.nlp`, `5.transformers`, `6.llms`

Legend:
- ✅ Done (with numbers / end-to-end)
- 📄 Done (reference only — concept explained, no dry-run)
- ❌ Missing
- 🎯 Interview-critical — must do
- ⏭️ Skip — not worth the time for interviews

---

## 4.nlp

### fundamentals/
| File | Status | Missing / Gap |
|---|---|---|
| `01_text_preprocessing.md` | 📄 reference | No pipeline trace with actual input→output at each step |
| `02_text_representations.md` | 📄 reference | — |
| `02_text_representations_end_to_end.md` | ✅ done | BoW, TF-IDF, BM25 with full matrices and numbers |

**Missing files:**
```
⏭️ 03_text_preprocessing_end_to_end.md
   → Trace "cat sat on mat" through: clean → tokenize → stop words → lemmatize → TF-IDF
   → SKIP — rule-based, not math-heavy, never asked in interviews at computation depth
```

---

### embeddings/
| File | Status | Missing / Gap |
|---|---|---|
| `01_word_embeddings.md` | 📄 reference | Concept explained, no training dry-run |
| `02_word2vec_end_to_end.md` | ✅ done | Skip-gram, negative sampling, GloVe, FastText with numbers |

**Missing files:**
```
⏭️ 03_contextual_embeddings.md
   → ELMo: bi-LSTM, character embeddings, how context changes representation
   → SKIP — ELMo is a dead topic. Nobody asks about it now that BERT exists.
             The concept (static vs contextual) is already covered in bert_end_to_end.md
```

---

### sequence_models/
| File | Status | Notes |
|---|---|---|
| `01_rnn_to_attention.md` | 📄 reference | Overview of the arc |
| `02_rnn_end_to_end.md` | ✅ done | Forward + backward, 9% gradient |
| `03_lstm_end_to_end.md` | ✅ done | 4 gates, 69% gradient |
| `04_gru_end_to_end.md` | ✅ done | 2 gates, 66% gradient |
| `05_attention_end_to_end.md` | ✅ done | Scaled dot-product, full attention matrix |
| `06_transformer_end_to_end.md` | ✅ done | PE + attention + FFN + residual + LN |

**Arc is complete. Nothing missing here.**

---

### applications/
| File | Status | Missing / Gap |
|---|---|---|
| `01_text_classification.md` | 📄 reference | No dry-run: input → TF-IDF → LogReg → prediction with numbers |
| `02_ner_and_tagging.md` | 📄 reference | No dry-run: IOB tagging, Viterbi decoding with numbers |
| `03_information_extraction.md` | 📄 reference | No dry-run: relation extraction, regex pipeline with numbers |
| `04_evaluation_metrics.md` | 📄 reference | Mentions BLEU/ROUGE/F1 but no step-by-step computation |

**Missing files:**
```
🎯 04_evaluation_metrics_end_to_end.md
   → Precision, Recall, F1 computed step-by-step
   → Confusion matrix with actual predictions
   → BLEU score on "cat sat on mat" vs reference — every n-gram counted
   → ROUGE-1, ROUGE-2, ROUGE-L computed
   → Perplexity = exp(avg cross-entropy loss) with numbers
   → BERTScore conceptually
   INTERVIEW REASON: "How do you evaluate your model?" is asked in every NLP interview.
                     F1/BLEU/ROUGE are always expected — must be able to compute from scratch.

⏭️ 02_ner_end_to_end.md
   → SKIP — NER is asked conceptually (IOB format, token classification), not computationally.
             Reference file 02_ner_and_tagging.md is sufficient for interviews.

⏭️ 01_text_classification_end_to_end.md
   → SKIP — Already covered. TF-IDF+LogReg is in text_representations_end_to_end.md,
             BERT classification is in bert_end_to_end.md. Nothing new to add.
```

---

## 5.transformers

### fundamentals/
| File | Status | Missing / Gap |
|---|---|---|
| `01_attention_mechanism.md` | 📄 reference | Concept only — dry-run is in 4.nlp/05_attention_end_to_end |
| `02_transformer_architecture.md` | 📄 reference | Concept only — dry-run is in 4.nlp/06_transformer_end_to_end |
| `03_tokenization.md` | ✅ done | BPE, WordPiece, SentencePiece with numbers |
| `04_pretraining_objectives.md` | ✅ done | MLM, CLM, span corruption with numbers |

**Missing files:**
```
⏭️ 05_positional_encoding_deep_dive.md
   → RoPE, ALiBi deep dive
   → SKIP for now — RoPE is mentioned in modern_llm_architecture.md (Tier 2).
             Not asked at computation depth in interviews. Conceptual explanation is enough.
```

---

### models/
| File | Status | Missing / Gap |
|---|---|---|
| `01_bert_family.md` | 📄 reference | BERT, RoBERTa, ALBERT, DistilBERT — concept only |
| `02_gpt_family.md` | 📄 reference | GPT-2, GPT-3, GPT-4, InstructGPT — concept only |
| `03_encoder_decoder.md` | 📄 reference | T5, BART, PEGASUS, mT5 — concept only |
| `04_efficient_transformers.md` | 📄 reference | Flash Attention, LoRA, quantization — concept only |
| `05_bert_end_to_end.md` | ✅ done | Bidirectional attention, MLM, [CLS] fine-tuning |
| `06_gpt_end_to_end.md` | ✅ done | Causal mask, CLM, weight tying, sampling |

**Missing files:**
```
🎯 07_t5_end_to_end.md
   → Encoder-decoder architecture: cross-attention with actual numbers
   → "cat <X> mat" → encoder → decoder → "sat on"
   → Cross-attention: decoder query attends to encoder K,V
   → Span corruption loss computation
   → How cross-attention differs from self-attention mathematically
   INTERVIEW REASON: "Explain encoder-decoder" / "How does cross-attention work?" are
                     standard interview questions. BERT+GPT arc is incomplete without T5.

🎯 08_modern_llm_architecture.md
   → LLaMA vs GPT-2: RoPE instead of sinusoidal, RMSNorm instead of LayerNorm,
     SwiGLU instead of ReLU, GQA (Grouped Query Attention) instead of MHA
   → Mistral: Sliding Window Attention
   → Why these changes matter: efficiency, length generalization, training stability
   → Parameter count: LLaMA-7B, 13B, 70B
   INTERVIEW REASON: "What's different about LLaMA vs the original GPT?"
                     Asked frequently at senior/staff level. Shows you follow the field.

⏭️ 09_efficient_transformers_end_to_end.md
   → Flash Attention tiling, speculative decoding, quantization dry-run
   → SKIP — asked conceptually ("what problem does Flash Attention solve?"), not computationally.
             04_efficient_transformers.md reference file covers the concept well enough.
```

---

## 6.llms

### Current files:
| File | Status | Missing / Gap |
|---|---|---|
| `01_prompting.md` | 📄 reference | Techniques listed, no worked examples with token probabilities |
| `02_finetuning.md` | 📄 reference | LoRA/QLoRA overview, no math |
| `03_alignment.md` | 📄 reference | RLHF/DPO overview, no math |
| `04_rag.md` | 📄 reference | Architecture overview, no numbers |
| `05_agents.md` | 📄 reference | ReAct, tools, MCP — no worked trace |
| `06_evaluation.md` | 📄 reference | Metrics listed, no computation |
| `07_finetuning_end_to_end.md` | ✅ done | Full fine-tune, LoRA math, RLHF/DPO with numbers |

**Missing files:**
```
🎯 08_rag_end_to_end.md
   → Chunking: split corpus → chunks with overlap, show chunk boundaries
   → Embedding: dense vector per chunk (toy 2D embeddings)
   → FAISS/vector search: nearest neighbor with cosine similarity, step-by-step
   → BM25 (sparse) + dense retrieval → hybrid fusion (RRF scores)
   → Reranking: cross-encoder scores
   → Generation: retrieved context + query → GPT forward pass → answer
   → Full pipeline: query → retrieve → rerank → generate, numbers at every step
   INTERVIEW REASON: #1 most asked LLM topic in 2024-25. Every company building with LLMs
                     asks "how does RAG work?", "how do you chunk documents?",
                     "dense vs sparse retrieval?", "how do you prevent hallucination?".
                     Must be able to explain end-to-end with confidence.

⏭️ 09_prompting_end_to_end.md
   → Few-shot, CoT, structured output with token probabilities
   → SKIP — prompting is experiential, not computational. Reference file covers the
             techniques. No interviewer expects you to compute CoT probabilities from scratch.

⏭️ 10_agents_end_to_end.md
   → ReAct loop, tool use, multi-agent
   → SKIP for now — agents are asked at system design level, not math level.
             Reference file 05_agents.md is sufficient. Come back after RAG.

⏭️ 11_llm_evaluation_end_to_end.md
   → Perplexity, MMLU, LLM-as-judge
   → SKIP — perplexity already covered in gpt_end_to_end.md. The rest is conceptual.
             Evaluation metrics end-to-end (4.nlp) covers the computational side.

⏭️ 12_modern_llms_overview.md
   → GPT-4, Claude, Gemini, LLaMA-3 product overview
   → SKIP — no dry-run possible. Reference file or just reading model cards is enough.
             Architecture differences covered in 5.transformers/08_modern_llm_architecture.md.

⏭️ 13_structured_output_and_function_calling.md
   → JSON mode, function calling, tool schemas
   → SKIP for now — practical skill, not interview theory. Learn by doing with the API.
```

---

---

## What to Write — Interview Focus Only

**4 files. ~3 sessions. Everything else skip.**

```
🎯 1. 5.transformers/models/07_t5_end_to_end.md
      "Explain encoder-decoder" / "How does cross-attention work?"
      Completes the BERT → GPT → T5 arc.

🎯 2. 4.nlp/applications/04_evaluation_metrics_end_to_end.md
      "How do you evaluate your NLP model?" asked in every interview.
      F1, BLEU, ROUGE, perplexity — must compute from scratch.

🎯 3. 6.llms/08_rag_end_to_end.md
      #1 most asked LLM topic right now. Every company building with LLMs asks this.
      chunking → embedding → retrieval → reranking → generation, all with numbers.

🎯 4. 5.transformers/models/08_modern_llm_architecture.md
      "What's different about LLaMA vs GPT-2?"
      RoPE, GQA, RMSNorm, SwiGLU — shows you follow the field.
```

---

## Everything Else — Skip for Interviews

| File | Reason to skip |
|---|---|
| `03_text_preprocessing_end_to_end.md` | Rule-based, never asked computationally |
| `03_contextual_embeddings.md` (ELMo) | Dead topic, covered in BERT file |
| `02_ner_end_to_end.md` | Asked conceptually, reference file is enough |
| `01_text_classification_end_to_end.md` | Already covered in bert + tfidf files |
| `05_positional_encoding_deep_dive.md` | Conceptual only in interviews, covered in arch file |
| `09_efficient_transformers_end_to_end.md` | Concept is enough, reference file covers it |
| `09_prompting_end_to_end.md` | Experiential not computational — learn by doing |
| `10_agents_end_to_end.md` | System design level — reference file is enough for now |
| `11_llm_evaluation_end_to_end.md` | Perplexity done in GPT file, rest is conceptual |
| `12_modern_llms_overview.md` | Product knowledge — just read model cards |
| `13_structured_output_and_function_calling.md` | Practical API skill — learn by doing |

---

## Total Count

| Folder | Done (✅) | Reference only (📄) | Must write (🎯) | Skip (⏭️) |
|---|---|---|---|---|
| 4.nlp | 8 | 7 | 1 | 3 |
| 5.transformers | 4 | 6 | 2 | 1 |
| 6.llms | 1 | 6 | 1 | 6 |
| **Total** | **13** | **19** | **4** | **10** |

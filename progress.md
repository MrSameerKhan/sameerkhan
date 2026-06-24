# Coding Practice Progress

> Pick this file up on any machine — it tells you exactly where to resume.

## Current Status

- **Phase:** 10 — Document AI
- **Next session:** Phase 10, Session 1
- **File:** `code_practice/10_document_ai/`

---

## Phase Completion

| Phase | Topic | Sessions | Status | Notes |
|-------|-------|----------|--------|-------|
| 05 | Transformers (HF models) | 7 | ✅ All run | GPU, GTX 1650 Ti, 25 Jun 2026 |
| 06 | LLMs Core | 3 | ✅ All run | OpenAI gpt-4o-mini; S01 also on Ollama |
| 07 | RAG | 5 | ✅ All run | faiss-cpu + rank-bm25 installed |
| 08 | Agents / LangGraph | 4 | ✅ All run | langchain-openai + langgraph installed |
| 09 | Fine-tuning | 6 | ⏸ Parked | torch 2.6 not available for cu121; trl 1.6 incompatible with 2.5.1 |
| 10 | Document AI | 4 | 🔧 Code-built | Next |

---

## Session Log (Phase 05–08)

| Phase | Sessions | Key fixes |
|-------|----------|-----------|
| 05 | S01–S07 all ✅ | datasets==2.20.0 downgrade; ── → -- encoding; T5 config de-en |
| 06 | S01–S03 all ✅ | Reverted to OpenAI; added PROVIDER flag (openai/claude/ollama) |
| 07 | S01–S05 all ✅ | S01 Q1 chunk-size bug documented; S03 hybrid BM25 fixed domain keyword retrieval |
| 08 | S01–S04 all ✅ | ReAct quote-strip fix; policy aliases; langchain-openai + langgraph installed |

---

## Resume Instructions

1. Read **Current Status** above — that's your next task
2. Open `code_practice/10_document_ai/` and run sessions in order
3. After Session 04: pin repo to GitHub and add resume bullet

_Last updated: 2026-06-25_

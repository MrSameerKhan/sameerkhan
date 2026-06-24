# SYNC — Cross-Machine Handoff

> **Update this file before every `git push`. Read it after every `git pull`.**
> This is the only file that communicates state between Mac and Windows.

---

## Last Session

| Field | Value |
|-------|-------|
| **Machine** | Windows |
| **Date** | 25 June 2026 |
| **What I did** | Ran all 4 Phase 08 Agent sessions (S01–S04 all ✅ Run). Installed langchain-openai, langgraph. Fixed ReAct quote-stripping + policy aliases bug. LangGraph multi-turn memory + HITL confirmed. |
| **Files changed** | `code_practice/08_agents/` all `_details.md` files, `01_react_agent.py`, `03_langgraph_agent/tools.py` |

---

## Next Task

```
What:   Run Phase 10 Document AI sessions (S01–S04) — all 🔧 Code-built
File:   code_practice/10_document_ai/
Note:   Phase 09 parked — torch 2.6 not available for cu121; resume when cu124 wheel releases
```

---

## Active Learning Arc

| Layer | Current Focus | Status |
|-------|--------------|--------|
| Theory | All 11 folders complete | ✅ Done |
| Code Practice | Phase 05 Transformers (S1-S7 exist) → Phase 06-10 also exist | 🔧 Need to run sessions |
| Root Packs | 00_HUB, 01_CAREER, 02_INTERVIEW, 03_LEARNING | ✅ Done |

---

## Code Practice Phase Status

| Phase | Topic | Sessions | Confirmed Run | Notes |
|-------|-------|----------|--------------|-------|
| 05 | Transformers | S01-S07 | ✅ All 7 run | GPU, GTX 1650 Ti |
| 06 | LLMs | S01-S03 | ✅ All 3 run | OpenAI gpt-4o-mini; S01 also tested on Ollama |
| 07 | RAG | S01-S05 | ✅ All 5 run | faiss-cpu + rank-bm25 installed |
| 08 | Agents | S01-S04 | ✅ All 4 run | langchain-openai + langgraph installed |
| 09 | Fine-tuning | S01-S06 | ⏸ Parked | torch 2.6 not on cu121; trl 1.6 meta tensor bug with 2.5.1 |
| 10 | Document AI | S01-S04 | unknown | Check _details.md badges |

---

## Pending Cleanup

- [ ] Delete `5.transformers/02_models/04_efficient_transformers copy.md` (stale duplicate)
- [ ] Move scripts out of `junk/` → root or `scripts/`
- [ ] Update `progress.md` to reflect actual session state

---

## How to Update This File

After any work session, before `git push`, update only these two blocks:

**Last Session** — overwrite machine, date, what you did, files changed.

**Next Task** — overwrite with exactly what to do next (be specific: file name, command, topic).

That's it. Keep everything else as-is until it changes.

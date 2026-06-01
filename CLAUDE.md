# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
conda env create -f environment.yml
conda activate sameerkhan
python test.py          # smoke test after environment changes
```

If there are dependency conflicts, use the locked snapshot instead: `environment.lock.yml`. After any package change, update `environment.yml` (pinned ranges) and regenerate `environment.lock.yml` (exact versions).

## Repository Architecture — 4-Tier System

| Tier | Files | Owns | Does NOT own |
|------|-------|------|--------------|
| 1. Personal-Org | `00_HUB.md`, `01_CAREER_PACK.md`, `02_INTERVIEW_PACK.md`, `03_LEARNING_PACK.md` | Action plan, interview answers, study navigation | Technical explanations |
| 2. Theory | Folders `1.machine learning/` → `11.system_design/` | All technical explanations, math, comparison tables | Daily actions, code |
| 3. Practice | `code_practice/` | Implementations (73 sessions across Phases 01-10) | Conceptual explanations |
| 4. Audit/Meta | `README.md`, `STRUCTURE.md`, `RULES.md` | Repo navigation, structure, conventions | Concept-specific content |

Tier 1 and Tier 3 **link to** Tier 2 — they never re-explain. Tier 4 describes structure — it never explains concepts.

## Content Rules (RULES.md is canonical)

**Before adding any technical content:** grep the repo for the topic first. If it already exists at similar depth, add a 1-line cross-ref, not a new explanation. Never let the same explanation grow in 3+ files.

**SSOT principle:** every concept has exactly one canonical home (see `STRUCTURE.md` for the full map). All other mentions are 1-2 line cross-refs with relative path links.

**When a file moves:** update all references in the same commit (grep + sed), update `STRUCTURE.md`, and verify with a final grep — zero stale paths.

## Naming Conventions

| Element | Convention |
|---------|------------|
| Theory files | `NN_topic_name.md` (two-digit zero-padded) |
| End-to-end pair | `02b_topic_end_to_end.md` paired with `02_topic.md` |
| Folder names | `N.topic_name/` (lowercase snake_case after the dot) |
| Cross-reference links | `[../4.nlp/01_fundamentals/01_tokenization.md](../4.nlp/01_fundamentals/01_tokenization.md)` — relative paths, always with extension |

Never use bare URLs — always use markdown link syntax. Verify every path resolves before committing.

## Code Practice Session Structure

**Phases 01-04 (legacy):** `code_practice/<phase>/<session>/` with `model.py`, `train.py`, `predict.py`, `all_details.md`

**Phases 05-10 (current):** `code_practice/<phase>/` with flat files:
- Single-file sessions: `NN_session_name.py` + `NN_session_name_details.md`
- Multi-file sessions: `NN_session_name/` folder with role-specific files + `details.md`
- Shared corpus: `_corpus.py` in phase folder (Phase 07)
- Windows notes: `code_practice/WINDOWS_SETUP.md`

Status badges in `_details.md` (or `all_details.md`) headers:
- `✅ Run` — executed end-to-end; real output captured in details file
- `🔧 Code-built` — code complete, awaiting run (most of Phases 05-10)
- `📄 Docs-only` — spec written, code not yet implemented

## Theory Folder Ownership (quick ref)

| Folder | Canonical topic scope |
|--------|-----------------------|
| `1.machine learning/` | Classical ML, evaluation, statistics |
| `2.deep learning/` | DL fundamentals + universal architectures (MLP, CNN, Transformer, MoE, quantization) |
| `3.computerVision/` | Vision applications (detection, segmentation, ViT depth, self-supervised vision) |
| `4.nlp/` | Tokenization, embeddings, sequence models, decoding, NER, NLP eval |
| `5.transformers/` | Transformer architecture, BERT/GPT/T5 families, efficient transformers |
| `6.llms/` | Prompting, fine-tuning workflow, alignment, LLM eval, vLLM, dataset prep |
| `7.rag/` | RAG patterns, pipeline depth, prompt injection defenses |
| `8.agents/` | ReAct, LangGraph, memory, multi-agent, MCP, agent eval |
| `9.multimodal/` | CLIP, VLMs, Document AI (OCR, LayoutLM, Donut, ColPali) |
| `10.mlops/` | Serving, observability, drift, pipelines, production RAG ops |
| `11.system_design/` | ML system patterns (multi-tenant, tool auth, eval systems) |

# Session 3 — ColPali: Vision-RAG on Document Images
Status: `🔧 Code-built`

Theory: [../../../9.multimodal/02_document_ai.md](../../9.multimodal/02_document_ai.md)

---

## Use Case

Text-only RAG misses tables, charts, handwritten annotations, and complex multi-column layouts. ColPali embeds document pages as image patches — every visual element becomes retrievable without OCR.

---

## How ColPali Works (Late Interaction)

```
Document page → ViT encoder → 1030 patch embeddings   (each patch = 14×14 pixels)
Query text   → LM encoder  → N token embeddings       (one per word-piece)

MaxSim scoring:
  For each query token j:
    find the patch i* that maximises sim(q_j, p_i)   ← "Which patch answers this word?"
  Sum the per-token MaxSim scores

Final score = Σⱼ max_i (q_j · p_i)
```

**Why MaxSim beats single-vector:** a query like "What is the total?" has 4 tokens. Each token independently finds its best-matching patch. "total" → matches the TOTAL row in the table. "$" → matches the currency column. Both signals combine → high precision even for complex layouts.

---

## ColPali vs CLIP vs Text-RAG

| | Text RAG | CLIP | ColPali |
|-|---------|------|---------|
| OCR needed | Yes | No | No |
| Handles tables | Poor | Limited | Excellent |
| Embedding | Chunk text | Single page vector | 1030 patch vectors |
| Scoring | Cosine | Cosine | MaxSim |
| Model size | Small | 400M | 3B (PaliGemma) |
| VRAM needed | CPU | 2 GB | 8+ GB |

---

## Hardware Requirements

| Mode | Model | VRAM | When to use |
|------|-------|------|-------------|
| ColPali | vidore/colpali-v1.2 | 8+ GB CUDA | Production, CUDA available |
| CLIP fallback | openai/clip-vit-base-patch32 | 2 GB (MPS OK) | Development, concept demo |

Script auto-detects and uses appropriate mode.

---

## Expected Output (CLIP fallback on MPS)

```
Device: mps
Model:  CLIP (single-vector fallback)

Building document corpus...
  8 pages in corpus

── Visual document retrieval ──

Query: What is the early repayment charge in year 3?
  #1 Page  6 (0.312) — Early repayment charge schedule
  #2 Page  2 (0.241) — Loan amount, LTV, monthly payment
  #3 Page  8 (0.198) — Decision notice and approval status

Query: Is the mortgage application approved?
  #1 Page  8 (0.387) — Decision notice and approval status
  #2 Page  1 (0.294) — Cover page / application overview
  #3 Page  2 (0.245) — Loan amount, LTV, monthly payment
```

ColPali would show higher scores and better precision for queries about tables/charts.

---

## How to Run

```bash
# CLIP fallback (works everywhere):
python 03_colpali_rag.py

# ColPali (CUDA required):
# Will auto-use ColPali if CUDA available
```

First run: downloads CLIP (~600 MB). ColPali first run: ~6 GB download.
CLIP inference: ~2 seconds per query. ColPali: ~5 seconds per query on A100.

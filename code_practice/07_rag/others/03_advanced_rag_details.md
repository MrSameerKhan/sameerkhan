# Session 3 — Advanced RAG: Hybrid Search + Reranking + HyDE
Status: `✅ Run`

Theory: [../../../7.rag/04_advanced_rag.md](../../../7.rag/04_advanced_rag.md)

---

## Use Case

High-precision financial QA: a loan officer needs the exact ERC schedule, not a vague answer. Basic dense-only RAG misses exact terms ("ERC", "debt burden ratio") because embeddings generalise. Hybrid search + reranking fixes it.

---

## Three-Stage Pipeline

```
Query
  │
  ├── BM25 retrieval  → top-20 (exact keyword matching)
  └── Dense retrieval → top-20 (semantic similarity)
              │
          RRF fusion → unified ranking (no score normalisation needed)
              │
         top-20 candidates → Cross-encoder reranker → top-4
              │
         Generate with reranked chunks → grounded answer
```

---

## Why Each Component

**BM25** wins on: exact policy terms ("Murabaha", "SARIE", "Iqama", "DBR"), numbers ("33%", "SAR 4,000"), acronyms. Dense retrieval completely misses rare domain terms.

**Dense** wins on: paraphrases, synonyms ("maximum amount" vs "upper limit"), concept queries.

**RRF** combines without normalisation — BM25 scores and cosine scores are incompatible scales. RRF uses rank position only:
```python
rrf_score(doc) = Σ 1 / (60 + rank_in_each_list)
```

**Cross-encoder** sees (query, chunk) jointly — full attention cross-attention. 10-20% better precision than bi-encoder alone. Runs only on top-20 candidates (not the whole corpus).

**HyDE**: generate a hypothetical policy passage as if it answered the query, search using that passage's embedding. Works because document-style text lives in a different region of embedding space than question-style text.

---

## Models Used

| Component | Model | Size |
|-----------|-------|------|
| Bi-encoder (dense) | all-MiniLM-L6-v2 | 90 MB |
| BM25 | rank_bm25 (no model) | — |
| Cross-encoder | ms-marco-MiniLM-L-6-v2 | 90 MB |
| Generation | gpt-4o-mini (API) | — |

---

## Actual Output (Windows, gpt-4o-mini, 2026-06-25)

| Query | Basic RAG | Advanced RAG |
|-------|-----------|--------------|
| ERC in year 3 | ✅ correct (3%) | ✅ correct (3%) |
| Debt burden ratio limit | ❌ "not specified" — wrong sources retrieved | ✅ correct (33% of gross salary) |
| Expat credit card docs | ✅ correct | ✅ correct + more complete |

**Key win — Q2:** Basic RAG retrieved Credit Card and Mortgage docs instead of Personal Finance. Hybrid BM25 matched "debt burden ratio" as exact keywords → correct doc retrieved → correct answer. This is the canonical example of why hybrid search beats dense-only on financial domain terms.

---

## Expected Output

```
════════════════════════════════════════════════════════════════════
Query: What is the ERC in year 3 of a fixed-rate mortgage?

[Basic RAG]
  Sources: ['Residential Mortgage Policy — LTV Requirements']   ← wrong doc!
  Answer:  I don't have that information in the provided context.

[Advanced RAG — hybrid + reranked]
  Sources: ['Residential Mortgage Policy — Interest Rates and Terms']
  Answer:  The early repayment charge (ERC) in year 3 is 3%.

[Advanced RAG + HyDE]
  Sources: ['Residential Mortgage Policy — Interest Rates and Terms']
  Answer:  The early repayment charge in year 3 of the fixed-rate period is 3%.
```

Basic RAG fails on "ERC" (not in policy text as full words). Hybrid search finds the right document via BM25 exact match.

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/07_rag
python 03_advanced_rag.py
```

First run: downloads `ms-marco-MiniLM-L-6-v2` (~90 MB). Cost: ~$0.05 per run.

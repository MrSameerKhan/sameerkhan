# Session 1 — Basic RAG Pipeline
Status: `✅ Run`

Theory: [../../../7.rag/01_rag.md](../../../7.rag/01_rag.md)

---

## Use Case

Policy document QA bot: instead of an LLM hallucinating bank policy details from training data, it reads the actual policy documents and answers from them. Zero hallucination risk on policy-specific facts.

---

## Pipeline

```
OFFLINE (once):
  8 policy documents
      │ chunk (200 chars, 20 overlap)
      │ embed (all-MiniLM-L6-v2)
      ▼
  FAISS IndexFlatIP  ← inner product on L2-normalised = cosine similarity

ONLINE (per query):
  user question
      │ embed → (1, 384)
      │ FAISS search → top-3 chunk indices + scores
      │ format context: "[Title]\nchunk content"
      ▼
  gpt-4o-mini → grounded answer
```

---

## Key Concepts

**Why FAISS IndexFlatIP with L2-normalised embeddings?**
Inner product on unit-length vectors equals cosine similarity. `IndexFlatIP` is an exact (non-approximate) search — no recall loss, fine for corpora < 100k chunks. For millions of vectors, switch to `IndexIVFFlat` (approximate, ~10× faster).

**Why character-level chunking here?**
Simpler to understand before learning better strategies (Session 02). In production, use sentence-aware splitters (LangChain's `RecursiveCharacterTextSplitter`).

**System prompt anchoring:**
"Answer using ONLY the context. If the answer is not there, say so." — prevents the LLM from falling back to training knowledge when the policy doc doesn't contain the answer.

---

## Tensor Shapes

```
Corpus chunks: N chunks (varies with CHUNK_SIZE)
Embeddings:    shape (N, 384) — all-MiniLM-L6-v2 output dim
FAISS index:   IndexFlatIP, dim=384

Query:
  embed([query]) → shape (1, 384)
  index.search(q, k=3) → scores (1, 3), indices (1, 3)
```

---

## Actual Output (Windows, gpt-4o-mini, 2026-06-25)

```
8 documents → 39 chunks indexed

Q1 (LTV first-time buyers)  → FAILED: "I don't have that information" ← retrieval bug
Q2 (mortgage documents)     → correct (payslips, SA302 tax returns)
Q3 (early repayment year 3) → correct (3%)
Q4 (expatriate finance)     → correct (24× monthly salary)
Q5 (savings below SAR 500)  → correct (monthly maintenance fee — amount not stated)
```

**Q1 retrieval bug:** Sources were correct (all 3 from "Residential Mortgage Policy — LTV") but the 200-char chunks split the "95% for first-time buyers" sentence across a chunk boundary — the LLM received fragments without the key fact. This is exactly the limitation that Session 02 (chunking strategies) fixes.

---

## Expected Output

```
Building index...
  8 documents → 34 chunks indexed

Q: What is the maximum LTV for first-time buyers?
A: First-time buyers are eligible for an enhanced LTV of up to 95% under the
   government-backed Help to Buy scheme.
   Sources: ['Residential Mortgage Policy — LTV Requirements']

Q: What documents do I need to apply for a mortgage?
A: Acceptable documents include payslips for employed applicants, SA302 tax
   returns for self-employed (minimum 2 years), and pension statements for
   retired applicants.
   Sources: ['Residential Mortgage Policy — Income and Affordability']

Q: What is the early repayment charge in year 3?
A: The early repayment charge in year 3 of the fixed-rate period is 3%.
   Sources: ['Residential Mortgage Policy — Interest Rates and Terms']
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/07_rag
python 01_basic_rag.py
```

Cost: ~$0.01 per run (5 LLM calls × gpt-4o-mini). Embeddings are local — free.
First run: downloads `all-MiniLM-L6-v2` (~90 MB).

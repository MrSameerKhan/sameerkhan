# Session 2 — Chunking Strategies
Status: `✅ Run`

Theory: [../../../7.rag/02_rag_pipeline.md](../../../7.rag/02_rag_pipeline.md)

---

## Use Case

Wrong chunk size is the #1 RAG failure mode — before any model tuning, get chunking right. This session shows what each strategy retrieves for different query types so you can make the right call.

---

## Strategies Compared

| Strategy | How | Avg chunk size | Best for |
|----------|-----|---------------|----------|
| `fixed-150` | 150-char windows, 30 overlap | ~120 chars | Precise single-fact retrieval |
| `fixed-300` | 300-char windows, 50 overlap | ~260 chars | General QA — balanced default |
| `sentence-2` | 2 sentences per chunk | varies | Documents with clear sentence structure |
| `hierarchical` | Sentence-level children, doc-level parents | child ~80 chars | Long docs needing parent context |
| `semantic` | Split at cosine similarity drops | varies | Topic-shifted long-form prose |

---

## Key Concept — Hierarchical (Parent-Child)

```
Retrieval stage: embed small child sentences → high precision match
                 "The ERC in year 3 is 3%"  ← tiny, specific, easy to match

Return to LLM:   the full parent document   ← more context around the fact
                 "Fixed-rate products... ERC schedule: 5%, 4%, 3%, 2%, 1%..."
```

This is the strategy LangChain's `ParentDocumentRetriever` implements. Retrieve small, return large.

---

## Semantic Chunking Logic

```python
sentences = split_by_punctuation(doc)
embeddings = embed(sentences)   # shape (N, 384)

# Compare consecutive sentences
for i in range(len(embeddings) - 1):
    sim = embeddings[i] @ embeddings[i+1]   # cosine similarity
    if sim < threshold:
        # Topic shift detected → start new chunk
```

When similarity drops below 0.72, the sentences are about different topics. Split there — not at arbitrary character boundaries.

---

## Actual Output (Windows, local embeddings only, 2026-06-25)

```
fixed-150    : 52 chunks, avg 132 chars
fixed-300    : 25 chunks, avg 257 chars
sentence-2   : 29 chunks, avg 194 chars
hierarchical : 55 chunks, avg 102 chars
semantic     : 55 chunks, avg 102 chars
```

**Winner by query:**
- Q1 (LTV first-time buyers): `sentence-2` top score 0.755 — retrieved complete sentence with 95% fact
- Q2 (documents + affordability): `fixed-300` top score 0.731 — longer chunks capture multi-sentence rules
- Q3 (ERC schedule): `sentence-2`/`hierarchical`/`semantic` tied at 0.705–0.707 — all retrieved the full ERC list

**Fixes S01 Q1 failure:** sentence-2 and hierarchical both retrieved the "95% LTV" chunk cleanly vs fixed-150/200 which split it mid-sentence.

---

## Expected Output

```
Chunk counts per strategy:
  fixed-150           : 54 chunks, avg 128 chars
  fixed-300           : 34 chunks, avg 248 chars
  sentence-2          : 28 chunks, avg 198 chars
  hierarchical        : 41 chunks, avg  78 chars
  semantic            : 22 chunks, avg 310 chars

────────────────────────────────────────────────────────────────────
Query: What is the maximum LTV for first-time buyers?

  [fixed-150]
    (0.812) ...eligible for an enhanced LTV of up to 95% under the government-backed Help to Buy...

  [fixed-300]
    (0.791) The maximum loan-to-value (LTV) ratio for standard residential mortgage is 90%...
    first-time buyers are eligible for 95%...

  [hierarchical]
    (0.843) First-time buyers are eligible for an enhanced LTV of up to 95%...  ← most precise

  [semantic]
    (0.774) The maximum LTV ratio for standard borrowers is 90%... first-time buyers 95%...
```

Hierarchical child chunks win on single-fact queries. Fixed-300 and semantic win on multi-fact queries.

---

## How to Run

```bash
cd code_practice/07_rag
python 02_chunking_strategies.py
```

No API key needed — retrieval only, no generation. Runtime: ~10 seconds (embedding all chunks).

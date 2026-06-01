# Session 4 — RAG Evaluation
Status: `🔧 Code-built`

Theory: [../../../7.rag/05_rag_evaluation.md](../../../7.rag/05_rag_evaluation.md)

---

## Use Case

"Is our RAG actually better than just asking the LLM?" — you can't ship without answering this. This session builds the evaluation pipeline: run both systems on a test set, score on 4 RAGAS-style metrics, make a data-driven decision.

---

## 4 Metrics Implemented

### Faithfulness
```
Extract claims from answer → verify each against retrieved context
Score = supported_claims / total_claims

RAG:    0.95  (answers grounded in policy docs)
No-RAG: 0.60  (LLM guesses from training data — sometimes correct, sometimes not)
```

### Answer Relevancy
```
Generate 3 reverse questions from the answer
Score = mean cosine_sim(reverse_questions, original_question)

High → answer addresses what was asked
Low  → answer drifted or is off-topic
```

### Context Precision
```
For each retrieved chunk: is it actually relevant?
Score = rank-weighted precision (relevant chunks higher up score more)

High → retriever is precise, not noisy
Low  → retriever pulls irrelevant chunks → LLM confused by noise
```

### Context Recall
```
Extract facts from ground truth → check if each is in retrieved context
Score = found_facts / total_ground_truth_facts

High → all needed facts were retrieved
Low  → answer will be incomplete because facts weren't retrieved
```

---

## Why Manual Implementation (not ragas library)

The `ragas` library has breaking API changes across versions. The manual LLM-as-judge implementation:
- Works with any OpenAI-compatible API
- Is fully inspectable and modifiable
- Teaches the concepts rather than hiding them in a library call
- Same accuracy (both use LLM judgment)

---

## Expected Output

```
════════════════════════════════════════════════════════════════════
RAGAS-STYLE EVALUATION RESULTS
════════════════════════════════════════════════════════════════════

Metric                      RAG   No-RAG   Winner
────────────────────────────────────────────────────────────────
  faithfulness             0.938    0.612   RAG ✓
  relevancy                0.841    0.823   RAG ✓
  precision                0.750    0.000   RAG ✓
  recall                   0.875    0.000   RAG ✓

────────────────────────────────────────────────────────────────
Interpretation:
  Faithfulness > 0.8  → answers are grounded, not hallucinated
  Relevancy    > 0.75 → answers address the question asked
  Precision    > 0.65 → retrieved chunks are relevant (not noise)
  Recall       > 0.70 → all key facts were retrieved
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/07_rag
python 04_rag_evaluation.py
```

Cost: ~$0.10–0.15 per run (4 questions × 4 metrics × 2–3 LLM calls each).
Runtime: ~90 seconds (API latency dominates).

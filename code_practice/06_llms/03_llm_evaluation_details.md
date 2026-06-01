# Session 3 — LLM Evaluation: A/B Testing Two Prompt Strategies
Status: `🔧 Code-built`

Theory: [../../../6.llms/04_evaluation.md](../../../6.llms/04_evaluation.md)

---

## Use Case

Before shipping any LLM change (new prompt, new model, new RAG config), you need evidence that v2 > v1. This session builds the evaluation pipeline every ML engineer runs — ROUGE-L for automated scoring, LLM-as-judge for qualitative depth, and a decision gate at the end.

---

## Two Systems Under Test

| | System A | System B |
|-|----------|----------|
| Prompt | Generic: "You are a banking assistant" | Domain few-shot with format instructions |
| Strength | Simple, cheap | Domain-specific, more complete |
| Weakness | Generic output | Slightly longer, more prompt tokens |
| Expected winner | — | B on all judge dimensions |

---

## Evaluation Pipeline

```
Eval dataset (5 Q&A pairs)
    │
    ├── call System A → response_A
    └── call System B → response_B
                │
                ├── ROUGE-L(response, reference)      ← automated, no API cost
                └── LLM-judge(question, reference,    ← 1 API call per response
                            response)
                            │
                            scores: correctness / completeness / specificity (1-5 each)

Report → per-question winner → aggregate averages → shipping decision
```

---

## ROUGE-L

Longest Common Subsequence F1 between prediction and reference (word-level):

```
prediction: "Al Rajhi personal finance carries 5.5% to 9.5% annual profit rate"
reference:  "personal finance profit rate ranges from 5.5% to 9.5% annually"

LCS: ["personal", "finance", "5.5%", "to", "9.5%", "annual"]  → 6 words
precision = 6 / 10 = 0.60
recall    = 6 / 9  = 0.67
ROUGE-L F1 = 2 × 0.60 × 0.67 / (0.60 + 0.67) = 0.632
```

**Limitation:** ROUGE-L measures surface overlap — a correct answer phrased differently scores low. That's why LLM-as-judge is the primary signal; ROUGE-L is a cheap sanity check.

---

## LLM-as-Judge Design

Three dimensions scored 1–5 (not a single score):

| Dimension | Catches |
|-----------|---------|
| Correctness | Factual errors, wrong numbers |
| Completeness | Partial answers, missing conditions |
| Specificity | Vague generalities vs actual figures and product names |

```python
JUDGE_PROMPT = """
Question: {question}
Reference answer: {reference}
Response to evaluate: {response}

Score 1-5 on:
- correctness:  factual accuracy
- completeness: covers all parts of the question
- specificity:  concrete details vs vague generalities

Return JSON: {"correctness": int, "completeness": int, "specificity": int, "reasoning": str}
"""
```

**Key design choices:**
- Use `temperature=0` for the judge — deterministic scoring
- Include `reference` in the judge prompt — judge verifies against ground truth, not just its own knowledge
- Three separate dimensions — easier to diagnose which system fails and why
- `response_format={"type": "json_object"}` — no manual JSON parsing

---

## Expected Output

```
Running LLM evaluation...
  Model: gpt-4o-mini
  Eval set size: 5 questions

  Evaluating 1/5: What is the minimum down payment for a residential...
  Evaluating 2/5: Can an expatriate obtain a home loan in Saudi Arabia...
  ...

══════════════════════════════════════════════════════════════════════
EVALUATION REPORT: System A (zero-shot) vs System B (few-shot)
══════════════════════════════════════════════════════════════════════

Metric                          System A   System B     Winner
──────────────────────────────────────────────────────────────
ROUGE-L (avg)                      0.312      0.381        B ✓
LLM-Judge correctness              3.40       4.20         B ✓
LLM-Judge completeness             2.80       4.00         B ✓
LLM-Judge specificity              2.60       4.40         B ✓
LLM-Judge overall avg              2.93       4.20         B ✓

──────────────────────────────────────────────────────────────
Per-question breakdown:

  Q1: What is the minimum down payment for a residential mortgage...
        ROUGE  A=0.421  B=0.512
        Judge  A=3.67  B=4.33  → B wins
        Reasoning: System B correctly specifies 10% for Saudi nationals...

...

══════════════════════════════════════════════════════════════════════
DECISION: Ship System B — meaningfully better across all judge dimensions.
```

---

## LLM-as-Judge Biases (Know These for Interviews)

| Bias | Description | Mitigation |
|------|-------------|-----------|
| Verbosity bias | Judge prefers longer answers | Use specificity metric — penalises padding |
| Self-preference | GPT-4 judge prefers GPT-4 outputs | Use different model as judge vs generator |
| Position bias | Judge favours first response in pairwise eval | Swap A/B order in 50% of calls, average scores |
| Reference anchoring | Judge anchors to reference wording | Score on correctness separate from surface similarity |

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
python 03_llm_evaluation.py
```

Cost: ~$0.05–0.10 per run (10 generator calls + 10 judge calls × 5 questions).
Runtime: ~60–90 seconds (API latency).

**Scale up:** for 100+ eval examples, run generator calls in parallel using `asyncio` + `openai.AsyncOpenAI`.

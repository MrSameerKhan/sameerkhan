# 11 LLM Evaluation Systems — A/B Testing, LLM-as-Judge, Production Eval

> Designing the evaluation layer for an LLM application — not a one-off benchmark, but a system that catches regressions continuously.

## Table of Contents

1. Objective
2. The 4 evaluation surfaces
3. Offline eval — golden datasets
4. LLM-as-Judge — system design
5. Online eval — A/B testing
6. Production trace inspection
7. Failure modes
8. Interview questions (5)
9. Further reading

---

## 1. Objective

Classic ML eval: split data → train → measure accuracy on holdout → ship.

LLM eval: more layers, less deterministic, never-finished. You need an EVALUATION SYSTEM, not a one-shot benchmark.

Senior interview Q: "How would you design the evaluation infrastructure for an LLM product team?" — the answer goes beyond "use BLEU."

---

## 2. The 4 Evaluation Surfaces

A production LLM system has FOUR distinct evaluation contexts:

| Surface | When | Cost | Coverage |
|---|---|---|---|
| Offline golden | Pre-deploy | Low | Narrow, curated |
| LLM-as-Judge | Pre-deploy + production sampling | Medium | Broad, automated |
| A/B test | Post-deploy, controlled rollout | Medium-High | Real users, real load |
| Production traces | Continuous | Free (already happening) | Everything, but hard to reason about |

A serious eval system uses all four. Each catches different failures.

---

## 3. Offline Eval — Golden Datasets

### What's in a Golden Set

50-500 hand-curated `(input, expected_output, key_facts)` triples. Domain experts construct these. Updated quarterly as the product evolves.

```json
{
  "input": "What's the current 30-year mortgage rate at Acme?",
  "expected_output": "The current 30-year mortgage rate at Acme is 6.85%.",
  "key_facts": ["6.85%", "30-year", "Acme"],
  "category": "rate_lookup",
  "difficulty": "easy"
}
```

### Metrics

- **Exact match** — too strict but useful as a floor
- **Key-fact recall** — fraction of `key_facts` strings appearing in response
- **Refusal rate** — `is_refusal(response)` flag
- **Length ratio** — `words(response) / words(expected)` — checks for too-short/too-long

### Cadence

- Run on every code change (CI gate)
- Threshold gate: don't deploy if regression > X% on any metric
- Per-category breakdown: easy / medium / hard, factual / chat / safety

### Tooling

- LangSmith Datasets / LangFuse Datasets — version-controlled, integrates with traces
- Phoenix Evals — built-in evaluation framework
- Or roll your own — JSONL file + Python script (Phase 4 Session 9 has runnable code)

---

## 4. LLM-as-Judge — System Design

LLM-as-Judge scales offline eval beyond the 500 golden examples. For broader coverage:

### The System Design

```
Pre-deploy:
  Sample 1000 production-like queries
  Run candidate model on all 1000
  Run incumbent model on all 1000
  For each pair, ask judge LLM: "Which is better? Why?"
  Aggregate → win rate of candidate vs incumbent

Production sampling:
  For 1% of production traffic, run LLM-Judge on output
  Track: faithfulness, answer relevance, hallucination flag
  Alert if any metric drops 10% week-over-week
```

### Judge Model Choice

- **Same model family as candidate** — self-preference bias (GPT-4 likes GPT-4)
- **Cross-family** (use Claude to judge GPT outputs, or vice versa) — less bias, slightly worse quality
- **Specialized eval models** (LLM-as-Judge fine-tunes like JudgeLM) — cheaper; designed for the task

### Reducing Judge Bias

- **Randomize position** in pairwise comparison
- **Use rubrics** (1-5 scores on specific criteria) instead of free-text
- **Calibrate** with 50 human-labeled comparisons periodically
- **Multi-judge ensemble** — majority vote across 3-5 judges

### Cost

LLM-as-Judge at scale isn't free. Sample, don't run on every request. Pre-deploy: 1000 queries × $0.05/judge = $50 per evaluation run. Production: 1% of 1M daily × 10K judge calls × $0.05 = $500/day.

For low-cost: use small models as judges (GPT-4o-mini, Llama-3.1-8B) — quality lower but acceptable for relative comparisons.

---

## 5. Online Eval — A/B Testing

The gold standard: test in production with real users.

### Setup

```python
# User makes request →
# hash(user_id) % 100 == 0  → 1% traffic to "treatment"
# else                       → "control"
# Both groups served normally; metadata logged for analysis.
```

### Metrics (the Multivariate Decision)

- **Primary**: task success rate, user satisfaction (thumbs up/down rate)
- **Cost**: $-per-request
- **Latency**: p50, p95, p99
- **Engagement**: requests per session, return rate
- **Safety**: refusal rate, escalation rate, error rate

### Stat Sig

- Need enough traffic to detect the expected effect. For 5% improvement on a 70% baseline metric, ~5K samples per arm gives 80% power at 95% confidence.
- Run for 1-2 weeks to account for weekly cycles.
- Use sequential testing or pre-registered analysis to avoid p-hacking.

### Common A/B Test Surprises (from Phase 5 experience)

- Better recall@5 BUT worse user satisfaction (more retrieved chunks = more distraction)
- Same accuracy, 2× latency (rejection)
- Better numerical metrics, specific edge case regressions (tail problems)
- Cost up 40% for 2% quality improvement (rejection unless quality is critical)

### Decision Framework

DO NOT decide on a single metric. Look at the FULL stack. Roll out gradually (10% → 25% → 50% → 100%) once decision is made.

---

## 6. Production Trace Inspection

Continuous observability is itself an eval surface.

### What to Capture

Every request: prompt, response, retrieved chunks, tool calls, latencies, error / success, user feedback (if available).

### Patterns to Inspect

- **High-cost queries** — top 1% by cost. Why are they expensive? Could you cache them? Route to cheaper model?
- **Slow queries** — p99 latency outliers. Pattern? Particular query type, particular model, particular time?
- **Failure queries** — 5xx errors, refusals, low-confidence responses. Common pattern?
- **User-flagged queries** — thumbs-down responses. Highest signal for "what's broken."

### Tooling

- LangSmith / LangFuse / Phoenix UI for trace inspection
- BigQuery / Snowflake for batch analytics on traces
- Manual review — weekly 10-trace random sample by PM + eng

### The "Eval Loop"

1. Production traces flag a class of failures
2. Add failing cases to golden dataset
3. Fix the model / prompt / config
4. Verify on golden dataset
5. Deploy via A/B test
6. Confirm fix in production traces

This is the closed-loop eval system. Without it, your LLM quality drifts silently.

---

## 7. Failure Modes

1. **Judge bias goes undetected** — judge LLM has format preferences (loves bullet points), length bias (longer = better). Periodic calibration against humans is non-negotiable.

2. **Golden set saturates** — model gets 100% on the golden set; useless for further improvement. Continuously add new failure cases from production traces.

3. **A/B test underpowered** — not enough traffic / time → noisy results, false conclusions. Power-analyze before running.

4. **Optimizing the wrong metric** — improved "key-fact recall" but users hate the responses. Always cross-reference user satisfaction.

5. **Improved judge model version drift** — vendor updates judge model → scores shift even when your model is unchanged. Pin judge version; track changes.

6. **Privacy in eval data** — production traces contain PII. Redact before using as golden set examples.

7. **Adversarial production queries** — small fraction of production traffic is from users probing the system. Don't let these poison your aggregate metrics — filter or weight by user reputation.

---

## 8. Interview Questions (5)

**Q1: Design the evaluation infrastructure for an LLM product team.**
Four-surface eval: (1) Offline golden dataset (50-500 curated examples; run on CI; threshold gates deploys); (2) LLM-as-Judge on 1000 production-like queries pre-deploy + 1% sampling in production; (3) A/B test for production rollout with multivariate metrics (success, cost, latency, satisfaction); (4) Continuous trace inspection (LangSmith / Phoenix) for production failures. The closed loop: production failures → golden set → fix → verify → deploy → trace.

**Q2: What's the bias issue with LLM-as-Judge and how do you mitigate it?**
Three known biases: position (favors first option in pairwise), length (longer answers even when no better), self-preference (GPT-4 likes GPT-4 outputs). Mitigations: randomize order, use rubric-based scoring (specific criteria 1-5 instead of free-text), use cross-family judges (Claude judges GPT, vice versa), periodically calibrate with 50 human-labeled comparisons, multi-judge ensemble for high-stakes evals.

**Q3: How do you A/B test an LLM config change?**
Bucket users by hash(user_id); route X% to treatment. Track multivariate metric stack: task success, satisfaction, cost, latency, refusal rate, tail performance — for 1-2 weeks. Power-analyze to ensure stat sig (need ~5K samples per arm for a 5% effect at 80% power). Decide based on full metric stack, not one metric. Roll out gradually (10% → 25% → 100%) once decided.

**Q4: How do you continuously evaluate a deployed RAG system?**
Production sampling: run LLM-as-Judge on 1% of responses for faithfulness, answer relevance. Track per-day aggregates. Alert if any metric drops > 10% week-over-week. Combined with: user feedback signals (thumbs down rate), retrieval similarity histograms (drift detection), cost per query trends. When alert fires: dig into traces (LangFuse / Phoenix), find pattern, fix.

**Q5: How do you prevent eval-set leakage and saturation?**
Three layers: (1) Strict separation — golden set examples NEVER in training data; periodic audit to confirm. (2) Continuously refresh — add new failure cases from production traces every month. (3) Don't tune to the eval — if model gets 95% on golden, the golden is stale, not the model is great. Use production user satisfaction as the ultimate ground truth; golden set is a CI gate, not the goal.

---

## 9. Further Reading

- LangSmith eval docs — docs.smith.langchain.com/evaluation
- LangFuse evals — langfuse.com/docs/scores/overview
- Phoenix evals — docs.arize.com/phoenix/evaluation
- Chatbot Arena (LMSYS) — chat.lmsys.org — large-scale human preference eval
- RAGAS (Es et al. 2023) — arXiv:2309.15217 — RAG-specific eval framework
- MT-Bench / LLM-as-Judge (Zheng et al. 2023) — arXiv:2306.05685
- Evidently AI — evidently.ai — drift detection for ML / LLM systems
- The `06_llms/09_eval_compare/` folder has runnable code for offline eval

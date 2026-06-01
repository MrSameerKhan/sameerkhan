# Session 6 — Production LLM Observability
Status: `🔧 Code-built`

Theory: [../../../10.mlops/11_llm_observability.md](../../../10.mlops/11_llm_observability.md)

---

## Use Case

Your LLM is in production. A cost spike happens at 2am. A latency increase goes unnoticed for 6 hours. Faithfulness scores degrade after a model version update. Without instrumentation, you find out from a user complaint. With it, you get a Grafana alert.

---

## Four Pillars of LLM Observability

| Pillar | Tool | What you catch |
|--------|------|----------------|
| Latency | Prometheus Histogram | P95 spikes, slow prompts, timeouts |
| Cost | Prometheus Counter | Budget overruns, model version changes |
| Quality | LLM-as-judge (async) | Faithfulness drops, answer drift |
| Input drift | PSI on token distribution | Prompt template bugs, domain shift |

---

## MonitoredLLM Pattern

```python
llm = MonitoredLLM(model="gpt-4o-mini")
answer, trace = llm.chat(messages)
# trace contains: latency_ms, cost_usd, tokens, prompt_hash, response_hash
trace.log()   # → Elasticsearch / CloudWatch / stdout
```

Every call is automatically instrumented. The caller doesn't change — monitoring is transparent.

---

## Structured Trace (JSON log line)

```json
{
  "request_id": "a3f7b2c1",
  "timestamp": "2026-05-31T14:23:01Z",
  "model": "gpt-4o-mini",
  "prompt_tokens": 312,
  "completion_tokens": 84,
  "total_tokens": 396,
  "latency_ms": 743.2,
  "cost_usd": 0.0000972,
  "prompt_hash": "d4e2a1b9",
  "response_hash": "8f3c7d2e",
  "input_length": 187,
  "output_length": 412
}
```

`prompt_hash` and `response_hash` enable: cache analysis, dedup detection, repeated-query identification — without storing PII.

---

## PSI Drift Detection

Population Stability Index measures distribution shift between baseline and current window:

```
PSI < 0.10  → No significant drift (🟢)
PSI 0.10-0.20 → Minor drift — monitor (🟡)
PSI > 0.20  → Significant drift — investigate (🔴)
```

Applied to query word count: sudden spike → prompt template bug. Sudden drop → users sending shorter queries (UI change?).

---

## Prometheus Metrics + Grafana

```
# Panels to build:
llm_latency_seconds_bucket    → heatmap or P95 line chart
rate(llm_tokens_total[5m])    → tokens/sec by type (prompt vs completion)
rate(llm_cost_usd_total[1h]) * 24  → projected daily cost
llm_active_requests            → concurrency gauge
rate(llm_requests_total{status="error"}[5m])  → error rate
```

Scrape endpoint: `http://your-service:9090/metrics` → add to Prometheus config.

---

## Alert Rules

```yaml
# prometheus/alerts.yml
- alert: LLMHighLatency
  expr: histogram_quantile(0.95, llm_latency_seconds_bucket) > 3
  for: 5m
  annotations:
    summary: "P95 latency {{ $value }}s > 3s threshold"

- alert: LLMCostSpike
  expr: rate(llm_cost_usd_total[1h]) * 24 > 100
  annotations:
    summary: "Projected daily cost ${{ $value }} > $100 budget"
```

---

## Expected Output

```
Prometheus metrics: http://localhost:9090/metrics

── Running monitored LLM calls ──

{"request_id": "a3f7b2c1", "latency_ms": 743.2, "cost_usd": 0.0000972, ...}
  Q: What is the maximum LTV for a first-time buyer?
  A: First-time buyers are eligible for up to 95% LTV under Help to Buy...
  [743ms | 396 tok | $0.00010]

...

── Session summary ──
  total_requests: 5
  total_cost_usd: 0.000521
  avg_latency_ms: 831.4
  p95_latency_ms: 1203.7
  cost_per_req_usd: 0.000104

✓ All metrics within thresholds
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
python 06_llm_monitoring.py
```

Cost: ~$0.002 per run (5 API calls × gpt-4o-mini rates).

**Resume bullet:** "Built production LLM observability layer: Prometheus metrics (latency/cost/throughput) + structured JSON traces + PSI-based input drift detection; integrated with Grafana dashboards for real-time alerting."

# Session 6 — Production LLM Observability
# Task   : add latency, cost, quality, and drift monitoring to any LLM pipeline
# Shows  : Prometheus metrics, structured traces, Evidently drift detection
# Pattern: wrap any LLM call in MonitoredLLM — monitoring is transparent to callers
#
# Requires: prometheus-client, evidently (pip install prometheus-client evidently)
# Set OPENAI_API_KEY before running.

import os
import time
import json
import hashlib
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
from openai import OpenAI

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("prometheus-client not installed — metrics will be logged only (no HTTP endpoint)")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o-mini"


# ── Structured trace ──────────────────────────────────────────────────────────
@dataclass
class LLMTrace:
    request_id:      str
    timestamp:       str
    model:           str
    prompt_tokens:   int
    completion_tokens: int
    total_tokens:    int
    latency_ms:      float
    cost_usd:        float
    prompt_hash:     str    # for dedup / cache analysis (no PII)
    response_hash:   str
    input_length:    int    # prompt char length
    output_length:   int

    # Quality signals (filled later)
    faithfulness:    float | None = None
    user_rating:     int   | None = None   # 1-5 thumbs

    def log(self):
        print(json.dumps(asdict(self)))

    @property
    def cost_per_1k_tokens(self) -> float:
        return (self.cost_usd / self.total_tokens) * 1000 if self.total_tokens else 0.0


# ── Cost model (gpt-4o-mini prices — update as rates change) ──────────────────
COST_PER_1K_INPUT  = 0.000150   # $0.15 / 1M input tokens
COST_PER_1K_OUTPUT = 0.000600   # $0.60 / 1M output tokens


def compute_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * COST_PER_1K_INPUT / 1000 +
            completion_tokens * COST_PER_1K_OUTPUT / 1000)


# ── Prometheus metrics ────────────────────────────────────────────────────────
if PROMETHEUS_AVAILABLE:
    # Counter: monotonically increasing
    REQUEST_COUNT    = Counter("llm_requests_total",    "Total LLM requests", ["model", "status"])
    TOKEN_COUNT      = Counter("llm_tokens_total",       "Total tokens used",   ["model", "type"])
    COST_COUNTER     = Counter("llm_cost_usd_total",     "Total cost in USD",   ["model"])

    # Histogram: latency distribution (buckets in seconds)
    LATENCY_HIST     = Histogram("llm_latency_seconds",  "LLM call latency",    ["model"],
                                 buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])

    # Gauge: point-in-time value
    ACTIVE_REQUESTS  = Gauge("llm_active_requests",      "In-flight LLM requests")


# ── Monitored LLM wrapper ─────────────────────────────────────────────────────
class MonitoredLLM:
    """Wraps any LLM call with automatic observability instrumentation."""

    def __init__(self, model: str = MODEL):
        self.model  = model
        self._traces: list[LLMTrace] = []

    def chat(self, messages: list[dict], **kwargs) -> tuple[str, LLMTrace]:
        import uuid

        if PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.inc()

        prompt_text = " ".join(m.get("content", "") for m in messages)
        t0 = time.perf_counter()

        try:
            res = client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            answer     = res.choices[0].message.content.strip()
            usage      = res.usage

            trace = LLMTrace(
                request_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now(timezone.utc).isoformat(),
                model=self.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=round(latency_ms, 1),
                cost_usd=round(compute_cost(usage.prompt_tokens, usage.completion_tokens), 6),
                prompt_hash=hashlib.md5(prompt_text.encode()).hexdigest()[:8],
                response_hash=hashlib.md5(answer.encode()).hexdigest()[:8],
                input_length=len(prompt_text),
                output_length=len(answer),
            )

            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(model=self.model, status="success").inc()
                TOKEN_COUNT.labels(model=self.model, type="prompt").inc(usage.prompt_tokens)
                TOKEN_COUNT.labels(model=self.model, type="completion").inc(usage.completion_tokens)
                COST_COUNTER.labels(model=self.model).inc(trace.cost_usd)
                LATENCY_HIST.labels(model=self.model).observe(latency_ms / 1000)

            self._traces.append(trace)
            return answer, trace

        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(model=self.model, status="error").inc()
            raise
        finally:
            if PROMETHEUS_AVAILABLE:
                ACTIVE_REQUESTS.dec()

    def summary(self) -> dict:
        if not self._traces:
            return {}
        latencies = [t.latency_ms for t in self._traces]
        costs     = [t.cost_usd   for t in self._traces]
        tokens    = [t.total_tokens for t in self._traces]
        return {
            "total_requests":    len(self._traces),
            "total_cost_usd":    round(sum(costs), 6),
            "total_tokens":      sum(tokens),
            "avg_latency_ms":    round(statistics.mean(latencies), 1),
            "p95_latency_ms":    round(sorted(latencies)[int(0.95 * len(latencies))], 1),
            "avg_tokens":        round(statistics.mean(tokens), 1),
            "cost_per_req_usd":  round(statistics.mean(costs), 6),
        }


# ── Input drift detection ─────────────────────────────────────────────────────
class InputDriftDetector:
    """
    Track query length distribution — sudden shift signals a domain change
    or upstream bug (e.g. prompt template doubled in size).
    Uses Population Stability Index (PSI): PSI > 0.2 → significant drift.
    """

    def __init__(self, window: int = 50):
        self.window     = window
        self.baseline:  list[int] = []
        self.current:   list[int] = []

    def observe(self, text: str) -> None:
        length = len(text.split())
        if len(self.baseline) < self.window:
            self.baseline.append(length)
        else:
            self.current.append(length)
            if len(self.current) >= self.window:
                psi = self._psi(self.baseline, self.current)
                status = "🔴 DRIFT" if psi > 0.2 else "🟡 WARN" if psi > 0.1 else "🟢 OK"
                print(f"  Drift check (PSI={psi:.3f}): {status}")
                self.current = []   # reset window

    def _psi(self, expected: list[int], actual: list[int], n_bins: int = 10) -> float:
        all_vals = expected + actual
        edges    = np.percentile(all_vals, np.linspace(0, 100, n_bins + 1))
        edges    = np.unique(edges)
        if len(edges) < 2:
            return 0.0
        exp_pct  = np.histogram(expected, bins=edges)[0] / len(expected)
        act_pct  = np.histogram(actual,   bins=edges)[0] / len(actual)
        exp_pct  = np.clip(exp_pct, 1e-6, None)
        act_pct  = np.clip(act_pct, 1e-6, None)
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ── Alerting rules ────────────────────────────────────────────────────────────
def check_alerts(summary: dict) -> list[str]:
    alerts = []
    if summary.get("p95_latency_ms", 0) > 3000:
        alerts.append(f"⚠️  P95 latency {summary['p95_latency_ms']}ms > 3000ms threshold")
    if summary.get("avg_latency_ms", 0) > 1500:
        alerts.append(f"⚠️  Avg latency {summary['avg_latency_ms']}ms > 1500ms threshold")
    daily_cost = summary.get("cost_per_req_usd", 0) * 10_000   # estimate at 10k req/day
    if daily_cost > 100:
        alerts.append(f"⚠️  Projected daily cost ${daily_cost:.2f} > $100 budget")
    return alerts


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    llm     = MonitoredLLM(model=MODEL)
    drifter = InputDriftDetector(window=3)   # tiny window for demo

    # Start Prometheus HTTP server (scrape at :9090)
    if PROMETHEUS_AVAILABLE:
        start_http_server(9090)
        print("Prometheus metrics: http://localhost:9090/metrics\n")

    queries = [
        "What is the maximum LTV for a first-time buyer?",
        "How do I apply for personal finance as an expatriate?",
        "What is the early repayment charge in year 4?",
        "What is the minimum salary to get a credit card?",
        "How long does a SWIFT transfer take?",
    ]

    SYSTEM = "You are a bank assistant. Answer specifically with exact figures."

    print("── Running monitored LLM calls ──\n")
    for q in queries:
        drifter.observe(q)
        answer, trace = llm.chat([
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": q},
        ])
        trace.log()   # structured JSON log → Elasticsearch / CloudWatch
        print(f"  Q: {q[:60]}")
        print(f"  A: {answer[:100]}...")
        print(f"  [{trace.latency_ms:.0f}ms | {trace.total_tokens} tok | ${trace.cost_usd:.5f}]\n")

    # Summary
    print("── Session summary ──")
    summary = llm.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Alert check
    alerts = check_alerts(summary)
    if alerts:
        print("\n── Alerts ──")
        for a in alerts:
            print(f"  {a}")
    else:
        print("\n✓ All metrics within thresholds")

    print("\n── Recommended dashboards ──")
    print("  Grafana panels to build from Prometheus metrics:")
    print("  1. llm_latency_seconds (histogram) → P50/P95/P99 over time")
    print("  2. rate(llm_tokens_total[5m]) → tokens/sec by type")
    print("  3. rate(llm_cost_usd_total[1h]) * 24 → projected daily cost")
    print("  4. llm_active_requests → concurrency")
    print("  5. rate(llm_requests_total{status='error'}[5m]) → error rate")


if __name__ == "__main__":
    main()

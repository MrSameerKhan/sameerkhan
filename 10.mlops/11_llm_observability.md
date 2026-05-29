# 11 — LLM Observability — Capturing + Alerting Across the Stack

> Monitoring LLMs requires observing the *content* of responses, not just infra metrics.

---

## Why LLM Observability is Different

Traditional software observability: latency, error rate, CPU utilization. An HTTP 200 means success.

LLM observability: a 200 OK response may contain a hallucination, a refusal, or toxic content. The response IS the output — you need to observe its content, not just whether it arrived. Also unique to LLMs:

- Multi-step chains and agents (which step failed?)
- Token cost per request (can explode silently)
- Non-deterministic outputs (same input → different quality responses)
- PII in prompts (GDPR implications for logging)

---

## Quick Reference — LLM Observability Tools

```
| Tool        | Best for                           | Key feature                         |
|-------------|-----------------------------------|-------------------------------------|
| LangSmith   | LangChain apps, polished UI       | Auto-traces every chain/agent step  |
| LangFuse    | Open-source, any framework        | Cost tracking, multi-tenant, self-host |
| Phoenix     | RAG observability, RAG metrics    | RAGAS-style metrics live, OTel-based |
| Helicone    | Drop-in proxy, zero code change   | Change one URL, full observability  |
| Arize       | Enterprise MLOps, embedding drift | Explainability, drift detection     |
| W&B Weave   | Already use W&B                   | W&B-native trace + eval             |
```

Trace anatomy:

```
Request received
    ↓ [span: retrieval — 120ms, 5 chunks, cosine sim 0.82]
    ↓ [span: prompt build — 3ms, 1847 tokens]
    ↓ [span: LLM call — 1.2s, 312 output tokens, $0.004]
    ↓ [span: post-process — 5ms]
Response returned  [total: 1.33s, cost: $0.004]
```

---

## 3. Tool Deep Dives

### LangSmith

The default for LangChain-based applications. Every chain/agent step is auto-traced. UI shows nested calls, latencies, tokens, errors. Built-in datasets for offline eval.

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
# Every chain / agent step now traced — done.
```

### LangFuse

The default open-source choice. Works with any framework (LangChain, LlamaIndex, raw OpenAI SDK, custom). Strong on cost tracking (multi-tenant, multi-model). Self-host or use their cloud.

### Phoenix

Built by Arize. Strong on RAG observability — knows about retrieval steps, can compute RAGAS-style metrics live. Open-source, OpenTelemetry-based. Good for technical teams.

### Helicone

Different model: it's a PROXY. You change your OpenAI base URL to Helicone's; they record everything, forward to OpenAI. Zero code change beyond the URL. Great for adding observability to existing apps.

### Comparison for typical scenarios

```
You're building with LangChain              → LangSmith (built-in) or LangFuse (open)
You're using raw OpenAI/Anthropic SDK       → Helicone (drop-in) or LangFuse
You're doing RAG and need RAG-specific eval → Phoenix
You're in a large enterprise, need SLAs     → Arize AX or LangSmith Enterprise
You want vendor-neutral OpenTelemetry       → Traceloop or LangFuse with OTel exporter
You already use W&B                         → W&B Weave
```

---

## 4. Integration Patterns

### Pattern 1 — Drop-in proxy (Helicone)

```python
client = OpenAI(
    api_key=API_KEY,
    base_url="https://oai.helicone.ai/v1",   # ← only change
    default_headers={"Helicone-Auth": f"Bearer {HELICONE_KEY}"},
)
# All OpenAI calls now logged in Helicone dashboard
```

### Pattern 2 — Decorator (LangFuse / Phoenix)

```python
from langfuse.decorators import observe

@observe()
def my_rag_pipeline(query):
    chunks = retrieve(query)
    answer = generate(query, chunks)
    return answer

# Trace auto-captured with timing, input, output
```

### Pattern 3 — Native LangChain integration (LangSmith)

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
# Every chain / agent step now traced
```

### Pattern 4 — OpenTelemetry native (Traceloop)

```python
from traceloop.sdk import Traceloop
Traceloop.init(disable_batch=True)
# All OpenAI / Anthropic / Cohere calls auto-instrumented via OTel
# Goes to any OTel-compatible backend (Datadog, Honeycomb, etc.)
```

---

## 5. What to Log and What to Alert On

### Log every request

```
- Prompt (or hash, if PII concerns)
- Response
- Tokens in / out
- Latency (TTFT + total)
- Model name + version
- Per-step trace if chain/agent
- Cost ($-equivalent)
```

### Compute periodic aggregates

```
Hourly:  error rate, p99 latency, total cost
Daily:   refusal rate, response length distribution,
         hallucination rate (sampled LLM-judge)
Weekly:  feature usage breakdown, top expensive prompts
```

### Alert on

```
- 5xx error rate > 0.1%
- p99 latency > 2× baseline
- Cost per hour > budget × 1.5
- Refusal rate spike > 2× baseline  (suggests prompt regression or model change)
- Hallucination rate spike  (from sampled LLM-judge)
- Anomalous request patterns  (one user generating 100× normal traffic — possible abuse)
```

### Privacy considerations

```
Don't log raw prompts containing PII by default.
  - Sampling: log 1% of requests fully, hash the rest
  - Field-level redaction: strip emails, SSNs
  - Per-tenant encryption keys (in multi-tenant systems)
```

---

## 6. Failure Modes

**1. Tool overhead in critical path.** Helicone proxy adds 10-50ms. For high-throughput apps, async-only logging (background queue) is better.

**2. Vendor lock-in via integration tightness.** LangSmith is hard to leave once you're deep in. Mitigation: start with vendor-neutral (OTel + LangFuse) if you anticipate scale.

**3. Cost explosion in logs.** Every request logged with full prompts and responses can generate GB/day of trace data. Use sampling for non-error requests.

**4. Privacy leakage.** Prompts contain PII. Default-log behavior often violates GDPR. Set up redaction + retention policies BEFORE going live.

**5. Eval drift over time.** Same LLM-judge model gets updated by the vendor → scores shift even when your model didn't change. Version-pin the judge.

**6. Alerting fatigue.** Too many alerts → ignored. Start with 3-5 critical alerts; expand based on incidents.

---

## 7. Interview Questions

**Q1: How do you monitor an LLM in production?**

Five-layer stack: (1) Infra metrics — latency, throughput, GPU utilization (Prometheus + Grafana); (2) Cost — tokens per request, $-per-request, per-tenant (LangFuse, Helicone); (3) Quality — sampled LLM-judge on responses for hallucination / faithfulness; (4) Drift — input length distribution, refusal rate, response-length distribution; (5) Trace-level inspection for incident response (LangSmith / Phoenix / LangFuse trace UI).

**Q2: What's the difference between LangSmith and LangFuse?**

LangSmith: closed-source, built by LangChain Inc., best for LangChain-based apps, polished UI, paid. LangFuse: open-source, works with any framework, self-host or cloud option, strong cost tracking, growing community. Pick LangSmith if all-in on LangChain; LangFuse for flexibility or self-hosting.

**Q3: When would you use Helicone vs LangFuse?**

Helicone: drop-in proxy. Change one URL, get observability. Best for existing apps using raw OpenAI/Anthropic SDKs where you can't easily add logging. LangFuse: requires SDK integration (decorator or explicit logging). Best for new apps or chain/agent applications where you want full trace control. Helicone is "fastest to set up"; LangFuse is "most flexible."

**Q4: What's special about Phoenix's RAG observability?**

Phoenix understands the RAG pipeline structure — it knows there's a retrieval step and a generation step. Out of the box it computes RAG-specific metrics (context relevance, retrieval recall via gold sets, faithfulness via LLM-judge). Generic observability tools don't have this RAG-native view.

**Q5: How do you handle PII in LLM logs?**

Default-on PII logging violates GDPR/CCPA. Best practices: (1) Redact at ingestion — regex for emails/SSNs, NER for names; (2) Sample logging — store 1% of full prompts, hash the rest; (3) Per-tenant retention policies — delete after 30/90 days; (4) Encrypt at rest with per-tenant keys for multi-tenant systems. LangFuse and Phoenix both support custom redaction hooks; Helicone has a "redaction layer" feature.

---

## 8. Further Reading

- LangSmith docs: docs.smith.langchain.com
- LangFuse docs: langfuse.com/docs
- Phoenix docs: docs.arize.com/phoenix
- Helicone docs: docs.helicone.ai
- OpenTelemetry GenAI Semantic Conventions — emerging standard for LLM tracing
- OpenLLMetry (Traceloop): github.com/traceloop/openllmetry — vendor-neutral SDK

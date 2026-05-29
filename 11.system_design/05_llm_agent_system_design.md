# 05 LLM Agent System Design

## Problem Statement

Design a production agentic LLM system for a financial services company.

```
Requirements:
  - Users ask natural language questions: "Summarize my top 5 risk positions"
  - System uses multiple tools: database queries, calculations, document retrieval
  - 10K users, ~500 agent requests/hour, avg 8 tool calls per request
  - Latency: < 30 seconds per complete agent run
  - Auditability: every tool call must be logged (regulatory requirement)
  - Cost: < $0.50 per agent request
```

---

## 1. Architecture Overview

```
                    CLIENT LAYER
  Web App / Mobile App / Internal Dashboard / API Consumer
                         | HTTPS request
                         ↓
                    API GATEWAY
  Auth (JWT) · Rate limiting · Request routing · TLS termination
                         ↓
                    LLM GATEWAY
  Token budget enforcement · Model routing (cheap vs expensive)
  Prompt injection detection · Request/response logging
                         ↓
               AGENT ORCHESTRATOR
  Session manager · ReAct loop engine · Tool dispatcher
  Max-turns enforcer · Token budget tracker · State machine
       ↓           ↓            ↓              ↓
  [LLM       [Tool Registry  [Memory     [Audit Log
   Inference]  + Executor]    Store]      (Kafka)]
       ↓           ↓            ↓
  [Database  [RAG / Doc    [External
   Query      Retrieval     APIs
   Tool]       Tool]        (calc...)]
```

---

## 2. Component Deep Dive

### 2.1 API Gateway

Responsibilities:
- Authentication: validate JWT token, extract user_id + permissions
- Rate limiting: 10 agent requests / user / hour (prevent cost explosion)
- Request routing: /agent/* → Agent Orchestrator, /chat/* → simple LLM
- TLS termination: all external traffic over HTTPS

Tech: AWS API Gateway, Kong, or nginx + custom auth middleware

```python
import redis
import time

class RateLimiter:
    def __init__(self, redis_client, max_requests: int = 10, window_seconds: int = 3600):
        self.redis = redis_client
        self.max_requests = max_requests
        self.window = window_seconds

    def is_allowed(self, user_id: str) -> tuple[bool, int]:
        """Returns (allowed, requests_remaining)."""
        key = f"rate_limit:{user_id}"
        now = time.time()
        window_start = now - self.window

        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)   # remove old requests
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})               # add this request
        pipe.expire(key, self.window)
        _, count, _, _ = pipe.execute()

        if count >= self.max_requests:
            return False, 0
        return True, self.max_requests - count - 1
```

### 2.2 LLM Gateway

The LLM gateway sits between your application and LLM providers. It handles:

```
1. Token budget enforcement
   - Each user has a monthly token budget (e.g., 500K tokens/month)
   - Count input + output tokens per request
   - Reject if would exceed budget

2. Model routing
   - Simple questions → GPT-3.5 / Claude Haiku (cheap, fast)
   - Complex reasoning → GPT-4 / Claude Opus (expensive, better)
   - Routing rule: if estimated_tokens < 500 and task_type == "simple" → cheap model

3. Prompt injection detection
   - Before sending to LLM: scan for injection patterns
   - "ignore previous instructions" → flag and reject or sanitize

4. Caching
   - Cache identical prompts (SHA256 hash of prompt) → same response
   - Useful for FAQ-style agent tasks

5. Logging
   - Every LLM call: timestamp, user_id, model, tokens_in, tokens_out, latency, cost
```

```python
import hashlib
import time
from anthropic import Anthropic

class LLMGateway:
    def __init__(self, redis_client, audit_logger):
        self.client = Anthropic()
        self.redis = redis_client
        self.audit_logger = audit_logger
        self.COST_PER_1K = {
            "claude-haiku-4-5-20251001": {"input": 0.00025, "output": 0.00125},
            "claude-opus-4-6":           {"input": 0.015,   "output": 0.075},
        }

    def call(self, messages: list, model: str, max_tokens: int,
             user_id: str, session_id: str, tools: list = None) -> dict:
        # 1. Check token budget
        if not self._check_budget(user_id, max_tokens):
            raise BudgetExceededError(f"User {user_id} has exceeded monthly token budget")

        # 2. Check cache (only for tool-free simple calls)
        if not tools:
            cache_key = self._cache_key(messages, model)
            cached = self.redis.get(cache_key)
            if cached:
                return {"content": cached, "cached": True, "tokens": 0}

        # 3. Prompt injection check
        self._check_injection(messages)

        # 4. Call LLM
        start = time.time()
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)
        latency_ms = (time.time() - start) * 1000

        # 5. Calculate cost
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = (tokens_in / 1000 * self.COST_PER_1K[model]["input"] +
                tokens_out / 1000 * self.COST_PER_1K[model]["output"])

        # 6. Deduct from budget
        self._deduct_budget(user_id, tokens_in + tokens_out)

        # 7. Audit log
        self.audit_logger.log({
            "event": "llm_call", "user_id": user_id,
            "session_id": session_id, "model": model,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "cost_usd": round(cost, 6), "latency_ms": round(latency_ms, 1),
        })

        # 8. Cache if no tools
        if not tools:
            self.redis.setex(cache_key, 3600, str(response.content))

        return {"content": response.content, "usage": response.usage, "cost": cost}

    def _cache_key(self, messages: list, model: str) -> str:
        content = str(messages) + model
        return "llm_cache:" + hashlib.sha256(content.encode()).hexdigest()

    def _check_injection(self, messages: list):
        INJECTION_PATTERNS = [
            "ignore previous instructions",
            "ignore all instructions",
            "disregard your",
            "you are now",
            "new persona",
        ]
        for msg in messages:
            content = str(msg.get("content", "")).lower()
            for pattern in INJECTION_PATTERNS:
                if pattern in content:
                    raise PromptInjectionError(f"Injection pattern detected: '{pattern}'")
```

### 2.3 Agent Orchestrator

The core of the system — manages the ReAct loop per session.

```python
import uuid
from dataclasses import dataclass, field
from enum import Enum

class SessionStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_TURNS_REACHED = "max_turns_reached"

@dataclass
class AgentSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    query: str = ""
    messages: list = field(default_factory=list)
    tool_calls: list = field(default_factory=list)   # audit trail
    turns_used: int = 0
    tokens_used: int = 0
    cost_usd: float = 0.0
    status: SessionStatus = SessionStatus.RUNNING
    created_at: float = field(default_factory=time.time)

class AgentOrchestrator:
    MAX_TURNS = 15
    TOKEN_BUDGET_PER_REQUEST = 30_000   # tokens
    COST_BUDGET_PER_REQUEST  = 0.50    # USD

    def __init__(self, llm_gateway, tool_registry, memory_store, audit_logger):
        self.llm = llm_gateway
        self.tools = tool_registry
        self.memory = memory_store
        self.audit = audit_logger

    def run(self, user_id: str, query: str) -> dict:
        session = AgentSession(user_id=user_id, query=query)
        session.messages = [{"role": "user", "content": query}]

        # Inject relevant memories from past sessions
        relevant_memories = self.memory.recall(user_id, query, top_k=3)
        if relevant_memories:
            system_context = f"Relevant context from past sessions:\n{relevant_memories}"
            session.messages.insert(0, {"role": "user", "content": system_context})

        self.audit.log({"event": "session_start", "session_id": session.session_id,
                        "user_id": user_id, "query": query})

        try:
            result = self._run_loop(session)
        except Exception as e:
            session.status = SessionStatus.FAILED
            self.audit.log({"event": "session_error", "session_id": session.session_id,
                            "error": str(e)})
            raise

        # Save session summary to memory
        self.memory.save(user_id, query, result, session.tool_calls)

        self.audit.log({
            "event": "session_complete",
            "session_id": session.session_id,
            "turns": session.turns_used,
            "tokens": session.tokens_used,
            "cost_usd": session.cost_usd,
            "status": session.status.value,
        })

        return {"result": result, "session_id": session.session_id,
                "turns_used": session.turns_used, "cost": session.cost_usd}

    def _run_loop(self, session: AgentSession) -> str:
        while session.turns_used < self.MAX_TURNS:
            # Budget checks
            if session.tokens_used >= self.TOKEN_BUDGET_PER_REQUEST:
                session.status = SessionStatus.BUDGET_EXCEEDED
                return "Token budget exceeded for this request."
            if session.cost_usd >= self.COST_BUDGET_PER_REQUEST:
                session.status = SessionStatus.BUDGET_EXCEEDED
                return "Cost budget exceeded for this request."

            # LLM call
            response = self.llm.call(
                messages=session.messages,
                model="claude-opus-4-6",
                max_tokens=2048,
                user_id=session.user_id,
                session_id=session.session_id,
                tools=self.tools.get_schemas()
            )

            session.tokens_used += (response["usage"].input_tokens +
                                    response["usage"].output_tokens)
            session.cost_usd += response["cost"]
            session.turns_used += 1

            content = response["content"]

            # Check stop reason
            stop_reason = self._get_stop_reason(content)

            if stop_reason == "end_turn":
                session.status = SessionStatus.COMPLETED
                return self._extract_text(content)

            if stop_reason == "tool_use":
                tool_results = []
                for block in content:
                    if block.type == "tool_use":
                        # Validate + execute tool
                        result = self.tools.execute(
                            name=block.name,
                            inputs=block.input,
                            user_id=session.user_id,
                            session_id=session.session_id,
                        )
                        # Audit every tool call
                        session.tool_calls.append({
                            "turn": session.turns_used,
                            "tool": block.name,
                            "inputs": block.input,
                            "result_preview": str(result)[:200],
                        })
                        self.audit.log({
                            "event": "tool_call",
                            "session_id": session.session_id,
                            "tool": block.name,
                            "inputs": block.input,
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        })

                session.messages.append({"role": "assistant", "content": content})
                session.messages.append({"role": "user", "content": tool_results})

        session.status = SessionStatus.MAX_TURNS_REACHED
        return f"Reached maximum turns ({self.MAX_TURNS}). Partial result may be incomplete."
```

### 2.4 Tool Registry

Central registry of all tools available to agents. Enforces authorization.

```python
from pydantic import BaseModel, validator
from typing import Callable

class ToolResult:
    def __init__(self, success: bool, data: str, error: str = None):
        self.success = success
        self.data = data
        self.error = error

    def __str__(self):
        return self.data if self.success else f"Tool error: {self.error}"

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}   # name → {fn, schema, permissions}

    def register(self, name: str, fn: Callable, schema: dict,
                 required_permission: str = None):
        self._tools[name] = {
            "fn": fn,
            "schema": schema,
            "permission": required_permission,
        }

    def get_schemas(self) -> list[dict]:
        return [{"schema": t["schema"]} for t in self._tools.values()]

    def execute(self, name: str, inputs: dict,
                user_id: str, session_id: str) -> ToolResult:
        if name not in self._tools:
            return ToolResult(False, "", f"Tool '{name}' not found")

        tool = self._tools[name]

        # Permission check
        if tool["permission"]:
            if not self._user_has_permission(user_id, tool["permission"]):
                return ToolResult(False, "", f"User lacks permission: {tool['permission']}")

        # Input validation (Pydantic)
        try:
            validated_inputs = self._validate_inputs(name, inputs)
        except Exception as e:
            return ToolResult(False, "", f"Invalid inputs: {e}")

        # Execute with timeout
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(tool["fn"], **validated_inputs)
                result = future.result(timeout=10)   # 10s timeout per tool
            return ToolResult(True, str(result))
        except concurrent.futures.TimeoutError:
            return ToolResult(False, "", f"Tool '{name}' timed out after 10s")
        except Exception as e:
            return ToolResult(False, "", f"Tool execution failed: {e}")

    def _user_has_permission(self, user_id: str, permission: str) -> bool:
        user_permissions = fetch_user_permissions(user_id)
        return permission in user_permissions

# Register tools
registry = ToolRegistry()

registry.register(
    name="query_portfolio",
    fn=lambda portfolio_id, fields: db.query(
        "SELECT * FROM portfolios WHERE portfolio_id=*", portfolio_id
    ),
    schema={
        "name": "query_portfolio",
        "description": "Query a user's financial portfolio. Returns positions, values, P&L.",
        "input_schema": {
            "type": "object",
            "properties": {
                "portfolio_id": {"type": "string"},
                "fields": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["portfolio_id"],
        },
    },
    required_permission="read_portfolio",
)

registry.register(
    name="calculate",
    fn=lambda expression: str(eval(expression)),
    schema={
        "name": "calculate",
        "description": "Evaluate a math expression.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    required_permission=None,   # no permission required
)
```

### 2.5 Memory Store

Persists agent context across sessions per user.

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json

class AgentMemoryStore:
    def __init__(self, redis_client, faiss_index, embedding_model):
        self.redis = redis_client
        self.index = faiss_index
        self.encoder = embedding_model
        self.memories = []   # text backing store (use DB in production)

    def save(self, user_id: str, query: str, result: str, tool_calls: list):
        """Save session summary for future recall."""
        summary = f"Query: {query}\nResult: {result[:200]}\nTools used: {[t['tool'] for t in tool_calls]}"
        embedding = self.encoder.encode([summary]).astype(np.float32)
        self.index.add(embedding)
        self.memories.append({"user_id": user_id, "summary": summary})

    def recall(self, user_id: str, query: str, top_k: int = 3) -> str:
        """Retrieve relevant memories for a user's query."""
        if len(self.memories) == 0:
            return ""
        q_emb = self.encoder.encode([query]).astype(np.float32)
        distances, indices = self.index.search(q_emb, top_k)
        relevant = [
            self.memories[i]["summary"]
            for i in indices[0]
            if i < len(self.memories) and self.memories[i]["user_id"] == user_id
        ]
        return "\n---\n".join(relevant)
```

### 2.6 Audit Log (Kafka)

Every event → Kafka topic → consumed by: compliance store, monitoring, billing.

```python
from kafka import KafkaProducer
import json, time

class AuditLogger:
    def __init__(self, kafka_brokers: list[str]):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_brokers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        )

    def log(self, event: dict):
        event["timestamp"] = time.time()
        self.producer.send("agent-audit-log", value=event)
        # Fire and forget — don't block the agent loop

# Audit trail for one agent request looks like:
# {"event": "session_start",   "session_id": "abc123", "user_id": "u1", "query": "..."}
# {"event": "llm_call",        "session_id": "abc123", "tool": "query_portfolio", "tokens_in": 450, "tokens_out": 120, "cost_usd": 0.012}
# {"event": "tool_call",       "session_id": "abc123", "tool": "query_portfolio", "inputs": {...}}
# {"event": "tool_call",       "session_id": "abc123", "tool": "calculate",       "inputs": {...}}
# {"event": "session_complete","session_id": "abc123", "turns": 3, "cost_usd": 0.031, "status": "completed"}
```

---

## 3. Latency Budget

For a typical 3-turn agent request:

```
Total budget: 30 seconds

Turn 1:
  LLM call (input ~500 tokens):   ~2.0s
  Tool call (DB query):            ~0.1s
  Overhead (serialization, logging): ~0.1s
  Turn 1 total:                    ~2.2s

Turn 2:
  LLM call (input ~1000 tokens):  ~3.0s
  Tool call (calculate):           ~0.01s
  Overhead:                        ~0.1s
  Turn 2 total:                    ~3.1s

Turn 3 (final answer):
  LLM call (input ~1500 tokens):  ~4.0s
  No tool call
  Turn 3 total:                    ~4.0s

Total: ~9.3s for 3 turns — well within 30s budget
With 8 turns (complex request):
  Average ~3s/turn × 8 = ~26s — still within budget

If a single turn exceeds 30s: streaming helps — user sees tokens as they arrive
```

Streaming implementation:

```python
async def run_agent_streaming(user_id: str, query: str):
    """Stream intermediate results to client via SSE."""
    async for event in orchestrator.stream(user_id, query):
        if event["type"] == "thought":
            yield f"data: {json.dumps({'thought': event['content']})}\n\n"
        elif event["type"] == "tool_call":
            yield f"data: {json.dumps({'tool': event['name'], 'status': 'running'})}\n\n"
        elif event["type"] == "tool_result":
            yield f"data: {json.dumps({'tool': event['name'], 'status': 'done'})}\n\n"
        elif event["type"] == "final_answer":
            yield f"data: {json.dumps({'answer': event['content']})}\n\n"
```

---

## 4. Scaling

### 4.1 Horizontal Scaling

```
Agent Orchestrator is stateless → scale horizontally
Session state stored in Redis (not in-process memory)
Any orchestrator instance can handle any session

Orchestrator instances: 5 replicas behind load balancer
  Each handles ~100 concurrent sessions
  Total: 500 concurrent sessions

Auto-scaling:
  Scale up when: avg queue depth > 10 pending requests
  Scale down when: avg CPU < 20% for 5 minutes
```

### 4.2 Session State in Redis

```python
class SessionStore:
    TTL = 3600   # sessions expire after 1 hour of inactivity

    def __init__(self, redis_client):
        self.redis = redis_client

    def save(self, session: AgentSession):
        key = f"session:{session.session_id}"
        self.redis.setex(key, self.TTL, json.dumps({
            "messages": session.messages,
            "turns_used": session.turns_used,
            "tokens_used": session.tokens_used,
            "cost_usd": session.cost_usd,
            "status": session.status.value,
        }))

    def load(self, session_id: str) -> dict | None:
        key = f"session:{session_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None
```

### 4.3 Async Tool Execution

```python
import asyncio

async def execute_tools_parallel(tool_calls: list, registry: ToolRegistry) -> list:
    """Execute multiple independent tool calls in parallel."""
    tasks = [
        asyncio.to_thread(
            registry.execute,
            name=tc.name,
            inputs=tc.input,
            user_id=user_id,
            session_id=session_id,
        )
        for tc in tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return results

# If LLM calls 3 tools at once → run all 3 in parallel
# Cuts tool execution time from 3x to 1x if tools are independent
```

---

## 5. Observability

### 5.1 Metrics to Track

```
Per-request metrics:
  agent_request_duration_seconds (histogram)
  agent_turns_per_request (histogram)
  agent_tokens_per_request (histogram)
  agent_cost_per_request_usd (histogram)
  agent_tool_calls_per_request (histogram)

Per-tool metrics:
  tool_call_duration_seconds{tool_name} (histogram)
  tool_call_success_rate{tool_name} (gauge)
  tool_call_error_rate{tool_name} (counter)

System health:
  llm_gateway_latency_ms (histogram, by model)
  session_queue_depth (gauge)
  active_sessions (gauge)
  budget_exceeded_rate (counter)
  injection_detected_rate (counter)
```

### 5.2 Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: agent-alerts
    rules:
      - alert: AgentHighCostPerRequest
        expr: histogram_quantile(0.95, agent_cost_per_request_usd) > 0.40
        for: 5m
        annotations:
          summary: "P95 agent cost exceeding $0.40 (budget is $0.50)"

      - alert: AgentHighTurnsPerRequest
        expr: histogram_quantile(0.95, agent_turns_per_request) > 12
        for: 5m
        annotations:
          summary: "P95 turns approaching max limit of 15 — possible loop"

      - alert: ToolHighErrorRate
        expr: tool_call_error_rate > 0.05
        for: 3m
        annotations:
          summary: "Tool {{ $labels.tool_name }} error rate > 5%"

      - alert: LLMHighLatency
        expr: histogram_quantile(0.99, llm_gateway_latency_ms) > 15000
        for: 3m
        annotations:
          summary: "P99 LLM latency > 15s — check model provider status"
```

---

## 6. Key Trade-offs

**Stateless vs Stateful Orchestrator:**
```
Stateless (sessions in Redis): horizontally scalable, any pod handles any session
Stateful (sessions in-process): lower latency (no Redis round-trip), but sticky sessions required
→ Choose stateless for production at scale
```

**Single LLM vs Model Cascade:**
```
Single expensive model (Opus): higher quality, higher cost
Cascade (Haiku → Opus if complex): 70% of requests cheap, 30% expensive
  - Estimate complexity from query length + task type before choosing model
  - Saves ~60% on LLM costs with minimal quality loss
```

**Synchronous vs Async agent runs:**
```
Sync: user waits for complete response (simple, up to 30s)
Async: return job_id, poll for result or use webhooks (better UX for long runs)
→ Use async + SSE streaming for best UX
```

**Tool timeout strategy:**
```
Short timeout (3s): fast, but fails on slow DB queries
Long timeout (30s): handles everything, but slow agent
  - Per-tool timeouts: DB=5s, web_search=10s, calculate=1s
  - On timeout: return partial answer + tell agent "tool timed out, try a different approach"
```

**Memory retention:**
```
No memory: simpler, each request is fresh
Full memory: risk of injecting irrelevant old context, privacy concerns
  - Selective memory: only save and recall "important" interactions
  - Expire memory after 30 days
```

---

## 7. Interview Q&A

**Q: How do you prevent an agent going into an infinite loop in production?**
A: Three layers: (1) Max turns per request (e.g., 15) — hard stop, return best partial answer; (2) Token budget per request — hard cap if tokens exceed budget, prevents cost runaway; (3) Loop detection — if the same tool is called with identical inputs more than 2 times in a session, treat it as a loop and break with an error message. Monitor agent_turns_per_request P95 — if it trends toward the limit, investigate which tool or query type causes loops.

**Q: How do you control costs in an agentic system?**
A: Four levers: (1) Model cascade — route simple questions to cheap models (Haiku $0.00025/1K tokens vs Opus $0.015/1K); (2) Token budget per request — hard cap at 30K tokens prevents runaway; (3) Result caching — cache LLM responses for identical prompts (SHA256 hash), effective for FAQ-type queries; (4) Tool result truncation — limit tool outputs to 2000 chars before adding to context (prevents context from growing unboundedly). Monitor cost_per_request_usd — set alerts at 80% of budget.

**Q: How do you audit tool calls for regulatory compliance?**
A: Every tool call goes through the ToolRegistry which logs it to Kafka synchronously: session_id, user_id, tool name, inputs (sanitized — no PII), timestamp, and result preview. Kafka consumers write to immutable compliance storage (S3 + Athena or Elasticsearch). Logs are retained for 7 years (financial regulatory requirement). This gives a complete audit trail: "on April 9, user X ran an agent that queried portfolio Y and calculated Z." The LLM gateway also logs every LLM call (tokens, cost, model).

**Q: How would you handle a tool that is sometimes slow (>10s DB query)?**
A: Per-tool timeouts with a ToolResult with error="timed out after 10s" — the agent sees this and can try an alternative approach. (2) Async tools: for known slow tools, trigger async execution, get a job_id, have the agent poll with a check_status tool. (3) Caching: repeat queries within TTL skip the slow execution. (4) Query optimization.

---

## Key Takeaway

LLM agent system design = API gateway (auth, rate limiting) + LLM gateway (token budget, model routing, caching, injection detection) + orchestrator (ReAct loop, session state, budget enforcement) + tool registry (validation, permissions, timeout, audit) + memory store (cross-session context) + audit log (Kafka → compliance). Critical production concerns: cost explosion (model cascade + token budgets), infinite loops (max turns + loop detection), tool security (permission checks + input validation + sandboxing), and observability (trace every tool call, alert on P95 cost/turns/latency).

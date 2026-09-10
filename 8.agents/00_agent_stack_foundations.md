# Agent Stack Foundations — What Sits Where

> **Why this file exists.** Every other file in `8.agents/` explains a *technique* (ReAct, LangGraph, MCP, memory). None of them explain the *landscape* those techniques live in — what a "format" is, why LangChain and Ollama and Groq are three completely different kinds of thing, or where a framework stops and the protocol starts. That confusion is the single most common reason agent theory fails to stick. Read this first.
>
> **SSOT.** This file owns: the 5-layer stack, the wire-format comparison, the framework/product placement map, and the workflow-vs-agent distinction. Everything else cross-refs here rather than re-explaining.

---

## Quick Reference

| Term | One-line definition |
|------|--------------------|
| Model | The weights. `claude-opus-5`, `gpt-4.1`, `llama-3.3-70b`. The only part that is "the AI". |
| Runtime | Software that loads weights and runs inference. vLLM, llama.cpp, Ollama, TGI. |
| Host | Whoever owns the GPU. OpenAI, Anthropic, Groq, AWS Bedrock — or your laptop. |
| Format | The JSON contract sent over HTTP. Two matter: OpenAI-style and Anthropic-style. |
| SDK | A library that writes that HTTP for you. `openai`, `anthropic`. Convenience, not capability. |
| Framework | Optional layer above the SDK that adds composition, state, loops. LangChain, LangGraph. |
| Protocol | A standard for *serving tools* to any agent, cross-process. MCP. |

---

## 1. The Five Layers

```
Layer 6   PROTOCOL   tools served over a standard wire   →  MCP
Layer 5   FRAMEWORK  composition · state · loops         →  LangChain, LangGraph, CrewAI
Layer 4   SDK        a library you import                →  anthropic, openai
Layer 3   FORMAT     the JSON contract on HTTP           →  OpenAI-style · Anthropic-style
Layer 2   HOST       who owns the GPU                    →  OpenAI, Anthropic, Groq, Bedrock, your laptop
Layer 1   RUNTIME    what loads and runs the weights     →  vLLM, llama.cpp, Ollama
Layer 0   MODEL      the weights themselves              →  GPT-4.1, Claude Opus 5, Llama 3.3
```

**Only Layer 0 is the AI.** Layers 1–6 are plumbing to get bytes to it and back. Interviewers probe whether you know this, because engineers who don't tend to say things like "we use LangChain instead of OpenAI" — a category error that reveals the gap instantly.

### The one-sentence test

> *"Is X a model, a runtime, a host, a format, an SDK, a framework, or a protocol?"*

If you can answer that for any product name thrown at you, you have this layer.

---

## 2. Product Placement — Every Name You'll Hear

| Name | Layer | What it actually is |
|------|-------|--------------------|
| **OpenAI** | 0 + 2 | Company. Makes GPT models **and** hosts them. Its API shape became the industry default. |
| **Anthropic** | 0 + 2 | Company. Makes Claude and hosts it. Its own API shape. |
| **DeepSeek** | 0 + 2 | Lab (China). Makes V3/R1, weights open, also hosts an API. |
| **Mistral** | 0 + 2 | Lab (France). Makes Mistral/Mixtral, mostly open weights, also hosts. |
| **Meta (Llama)** | 0 only | Makes Llama weights and releases them. **Hosts nothing** — you or someone else must run them. |
| **Groq** | 2 | Host only. Custom LPU hardware. Serves *other people's* open models, very fast. Makes no models. |
| **Together / Fireworks** | 2 | Hosts. Rent GPUs, serve open-weight models over an API. |
| **AWS Bedrock / Google Vertex / Azure Foundry** | 2 | Cloud hosts that resell *other vendors'* models (incl. Claude) inside your cloud account. |
| **OpenRouter** | 2 (meta) | Aggregator. One key, one endpoint, routes to 300+ models across all the above. |
| **vLLM** | 1 | Runtime. Open-source inference server (PagedAttention, continuous batching). You install it on your own GPUs. → `../6.llms/05_vllm_internals.md` |
| **llama.cpp** | 1 | Runtime. C++ engine, quantized GGUF, runs on CPU/small GPUs. |
| **Ollama** | 1 (+ local 2) | Runtime + model manager wrapping llama.cpp. Also serves an **OpenAI-compatible HTTP endpoint** locally. |
| **LM Studio** | 1 (+ local 2) | Same job as Ollama with a desktop GUI. Also wraps llama.cpp. |
| **HuggingFace `transformers`** | 1 | Runs weights **in-process**, no HTTP at all. `model.generate()`. |
| **`openai` / `anthropic` pip packages** | 4 | SDKs. Wrap HTTP calls in typed Python. |
| **LangChain** | 5 | Framework. Composition (LCEL), integrations, parsers. → `03_langchain_primer.md` |
| **LangGraph** | 5 | Framework. Stateful agent graphs, checkpointing, HITL. → `04_langgraph_deep.md` |
| **CrewAI / AutoGen / Swarm / smolagents / Pydantic AI** | 5 | Frameworks, multi-agent flavoured. → `07_multi_agent_orchestration.md` |
| **MCP** | 6 | Protocol. Serves tools/resources/prompts cross-process to any host. → `08_mcp_protocol_deep.md` |

**Three traps this table defuses:**

- *Groq ≠ Grok.* Groq is a hosting company with custom silicon. Grok is xAI's model. They are unrelated.
- *Ollama is not a model.* `ollama run llama3.2` runs Meta's model on Ollama's runtime.
- *LangChain is not an LLM provider.* It calls providers. Removing LangChain does not remove your model.

### "Direct, Bedrock, Vertex" — three doors, same weights

| Access path | Runs on | Auth / billing |
|---|---|---|
| Direct | `api.anthropic.com` | Anthropic account, `ANTHROPIC_API_KEY` |
| AWS Bedrock | Amazon infra | Your AWS account, IAM |
| Google Vertex AI | Google infra | Your GCP project |

Identical model, identical quality. **Enterprises choose Bedrock/Vertex so spend, compliance, data residency and audit stay inside the cloud contract they already have.** That is a procurement and governance decision, not a capability one — and saying exactly that is the senior answer.

---

## 3. What a "Format" Is

A model is a pile of weights with no opinion about JSON. But inference usually runs on *someone else's machine*, so you talk to it over HTTP — and both sides must agree on the JSON shape. That agreement is the format. It is a REST contract, nothing deeper.

**Formats are a networking concern, not a model concern.** Run weights locally with `transformers` and there is no format at all, because there is no network.

### The two formats, side by side

**OpenAI** — `POST /v1/chat/completions`
```json
{ "model": "gpt-4.1", "messages": [{"role": "user", "content": "hi"}] }
```
```json
{ "choices": [ { "message": {"role": "assistant", "content": "hello"},
                 "finish_reason": "stop" } ] }
```

**Anthropic** — `POST /v1/messages`
```json
{ "model": "claude-opus-5", "max_tokens": 1024,
  "messages": [{"role": "user", "content": "hi"}] }
```
```json
{ "content": [ {"type": "text", "text": "hello"} ],
  "stop_reason": "end_turn" }
```

| | OpenAI style | Anthropic style |
|---|---|---|
| Endpoint | `/chat/completions` | `/messages` |
| `max_tokens` | optional | **required** |
| Answer lives at | `choices[0].message.content` — a **string** | `content[]` — a **list of typed blocks** |
| Stop signal | `finish_reason` (`stop` / `tool_calls` / `length`) | `stop_reason` (`end_turn` / `tool_use` / `max_tokens`) |
| System prompt | a message with `role: "system"` | a **top-level** `system` parameter |
| Tool request | `message.tool_calls`; **`content` becomes `None`** | a block with `type: "tool_use"` inside `content[]` |
| Tool result sent back | `{"role": "tool", "tool_call_id": ...}` | a `tool_result` block inside a **user** message |

**Who speaks which:**

| Format | Spoken by |
|---|---|
| **OpenAI style** | OpenAI · **Ollama** · vLLM · Groq · Together · Fireworks · DeepSeek · Mistral · OpenRouter · LM Studio · llama.cpp · (Gemini via its compat endpoint) |
| **Anthropic style** | Claude only — direct, Bedrock, Vertex |

**This is the payoff: it is two formats, not N models.** Learn both and you can drive essentially any model in existence. "Provider flexibility" is a two-item problem, not an open-ended one.

> **Interview trap.** In OpenAI style, `message.content` is `None` whenever the model makes a tool call. Printing it raw gives `None` and looks broken. In Anthropic style the equivalent trap is `content[0].text` — index 0 may be a `thinking` block, not text. Both bugs are extremely common; naming them unprompted signals real hands-on time.

---

## 4. SDK vs Framework vs Protocol

These three get conflated constantly. The distinction:

| | Gives you | Removing it means | Example |
|---|---|---|---|
| **SDK** | typed HTTP calls, retries, auth, streaming | you hand-write `requests.post` | `anthropic`, `openai` |
| **Framework** | composition, state, the loop, integrations | you write the `while` loop yourself | LangChain, LangGraph |
| **Protocol** | tools served from a *separate process*, discovered at runtime | you hardcode tool schemas in your file | MCP |

### Proof the SDK is optional

```python
import os, requests

r = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    },
    json={"model": "claude-opus-5", "max_tokens": 1024,
          "messages": [{"role": "user", "content": "What is LTV?"}]},
)
print(r.json()["content"])
```

`client.messages.create(...)` builds exactly that POST. Running this once permanently demystifies the SDK — and being able to say "the SDK is a convenience wrapper over one HTTP endpoint" is a small, cheap credibility signal.

### And no format at all

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")
print(pipe("What is LTV?")[0]["generated_text"])
```

No HTTP, no JSON, no key. Just a forward pass.

---

## 5. Workflows vs Agents — The Framing Question

This distinction comes from Anthropic's *Building Effective Agents* and has become the standard vocabulary. It is very likely to open an agent conversation.

| | Workflow | Agent |
|---|---|---|
| Control flow | **Predefined in your code** | **The model decides** at runtime |
| Path taken | Same every time | Varies per input |
| Cost / latency | Predictable | Unbounded until you cap it |
| Debuggability | High | Low — needs tracing |
| When | You can enumerate the steps | You genuinely cannot |

> **The senior answer:** *"Most things marketed as agents are workflows, and that's usually the right call. I reach for an agent only when the steps genuinely can't be enumerated ahead of time, because I'm trading predictable cost and debuggability for flexibility."*

### The five workflow patterns (Anthropic's taxonomy)

Know these by name — they are the shared vocabulary now.

| Pattern | Shape | Use when |
|---|---|---|
| **Prompt chaining** | LLM → check → LLM → check → LLM | Task decomposes into fixed sequential steps; gates validate between them |
| **Routing** | classify → dispatch to specialised prompt/model | Distinct input categories need different handling. **Also the cost lever** — route easy inputs to Haiku, hard ones to Opus |
| **Parallelization** | fan out → aggregate | Two flavours: **sectioning** (independent subtasks in parallel) and **voting** (same task N times, take consensus) |
| **Orchestrator-workers** | orchestrator decomposes → workers → synthesise | Like parallelization, but **subtasks are NOT predefined** — the orchestrator decides them per input. This is the boundary where workflow starts becoming agent |
| **Evaluator-optimizer** | generate → critique → revise → loop | Clear evaluation criteria exist and iterative refinement measurably helps |

**Parallelization vs orchestrator-workers is a favourite probe.** The difference is *who decides the subtasks*: in parallelization you do, in code, ahead of time; in orchestrator-workers the LLM does, at runtime, per input.

These five are complementary to — not competing with — the academic planning patterns (ReAct, Plan-and-Execute, ReWoo, LATS, Reflexion) in `06_planner_executor_patterns.md`. Anthropic's five describe *system architecture*; those describe *how a single agent plans*.

---

## 6. When NOT to Use a Framework

Interview panels now treat **framework-deletion judgment** as a senior hiring signal. Knowing when a plain Python function replaces the framework separates mid-level from senior.

| Situation | Reach for |
|---|---|
| One LLM call, no tools, no state | **The SDK.** Nothing else. |
| Fixed sequence of calls | **Plain Python functions.** A `for` loop is not technical debt. |
| Tool-calling loop, single agent | SDK + a `while` loop **or** LangGraph's prebuilt agent |
| Branching, retries, HITL, persistence, replay | **LangGraph** — this is what it's for |
| Latency-critical path (sub-second SLA) | **SDK direct.** LangChain adds ~100–300 ms per chain |
| Tools shipped across apps / by other vendors | **MCP** |

**What frameworks genuinely buy you** (say this, don't just bash them): checkpointing and resume-after-crash, human-in-the-loop interrupts, ~100+ pre-built provider integrations, and a visualizable topology for debugging. In an **enterprise** setting — audit trails, approval gates, compliance replay — those are load-bearing, not decoration.

**What they cost:** abstraction over the actual prompt, breaking changes across minor versions, a large dependency tree, and latency.

> **The balanced answer:** *"I start with the SDK and plain Python, and I add LangGraph at the point where I need checkpointing, HITL, or replay — because those are genuinely hard to hand-roll correctly. I don't add it for composition alone; a function already composes."*

---

## 7. How Agents, RAG and LLMs Connect

You'll be asked on all three. They are one stack, not three subjects:

```
LLM        the engine        → what a single call does        6.llms/
RAG        a tool            → retrieval, one capability      7.rag/
Agent      the loop          → decides WHICH tool, WHEN       8.agents/
```

- **RAG is a tool an agent can call.** Classic RAG always retrieves, then generates — a fixed pipeline, i.e. a *workflow*.
- **Agentic RAG** lets the model decide *whether* to retrieve, *what* to query, and *whether the result was good enough to retry*. That is the 2026 trend line, and the cleanest way to show you understand both topics at once.
- **The failure modes compound.** Bad retrieval poisons agent reasoning; an agent that retries retrieval multiplies RAG cost.

> **Connective answer to keep ready:** *"Classic RAG is a workflow — retrieve then generate, same path every time. Agentic RAG makes retrieval a tool the model chooses to call, so it can reformulate a weak query or skip retrieval entirely when it already knows the answer. You buy adaptivity and pay in latency, cost and evaluation difficulty — you're now evaluating a trajectory, not a single retrieval."*

---

## 8. Interview Q&A

**Q: What's the difference between a workflow and an agent?**

A workflow orchestrates LLM calls through **predefined code paths** — the sequence is fixed and lives in my code. An agent lets the **model direct its own process**, choosing tools and deciding when it's done, so the path varies per input. The tradeoff is concrete: workflows give predictable cost, latency and debuggability; agents give flexibility on tasks whose steps you genuinely can't enumerate. Most production systems marketed as agents are actually workflows, and that's usually correct — I only reach for a real agent when the step sequence can't be known ahead of time. Anthropic's five workflow patterns — prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer — cover most of what people build.

**Q: If I gave you a model name, how would you figure out how to run it?**

I'd separate the layers. First, **are the weights open or closed?** Closed (GPT, Claude) means exactly one path: the vendor's API or a cloud reseller like Bedrock/Vertex. Open (Llama, Mistral, Qwen, DeepSeek) means I choose a **runtime** — vLLM for GPU serving at throughput, llama.cpp/Ollama for local or CPU — or a **host** that already runs it, like Groq, Together or Fireworks. Then the only remaining question is the **wire format**, and there are only two: OpenAI-style, which almost everything speaks including Ollama and vLLM, or Anthropic-style for Claude. So "which model" collapses to three decisions: open or closed, who runs the GPU, and which of two JSON shapes.

**Q: OpenAI and Anthropic APIs — what actually differs?**

Structurally the same idea, different shapes. OpenAI returns `choices[0].message.content` as a **string** with `finish_reason`; Anthropic returns `content` as a **list of typed blocks** with `stop_reason`. The system prompt is a message role in OpenAI and a top-level parameter in Anthropic. `max_tokens` is optional in OpenAI and required in Anthropic. For tool calls, OpenAI puts them in `message.tool_calls` and sets `content` to `None`; Anthropic emits a `tool_use` block inside the content list. The two gotchas that bite in practice: reading `content` directly in OpenAI when a tool was called gives `None`, and indexing `content[0].text` in Anthropic can hit a `thinking` block instead of text.

**Q: Is LangChain necessary to build an agent?**

No. An agent is a `while` loop around one API call: ask the model, if it requested a tool run it, append the result, ask again. That's about fifteen lines with the raw SDK. Frameworks earn their place when you need things that are genuinely hard to hand-roll — checkpointing and crash resume, human-in-the-loop interrupts, replay/time-travel for debugging, or a hundred pre-built integrations. In an enterprise context those aren't nice-to-haves; approval gates and audit replay are usually requirements. What I won't do is add a framework purely for composition — plain functions already compose, and the abstraction cost is real: hidden prompts, breaking changes, and 100–300 ms of overhead per chain.

**Q: Where does MCP sit relative to all this?**

MCP is a layer above the framework, not a competitor to it. Tools defined with LangChain's `@tool` live **inside your process**; MCP tools live in a **separate server process** and are discovered at runtime over JSON-RPC. That's what makes them portable — one server works with Claude Desktop, Cursor, VS Code, or a custom LangGraph agent, which turns M hosts × N tools into M + N. Practically: in-app integrations stay as LangChain tools; anything crossing an app or vendor boundary — GitHub, Postgres, Slack, an internal platform team's service — is where MCP pays off.

**Q: Your agent needs to work with three different LLM providers. How do you architect that?**

I'd resist building an abstraction layer, because there are only two wire formats to support and a wrapper hides exactly the differences that matter — content blocks vs `tool_calls`, `stop_reason` vs `finish_reason`. Concretely: one code path on the OpenAI-compatible format covers OpenAI, Ollama, vLLM, Groq, Together and OpenRouter with nothing but a `base_url` swap, and a second, separate path handles Anthropic. Two explicit implementations beat one leaky abstraction. If I genuinely needed provider-agnostic routing in production I'd use OpenRouter or a gateway rather than hand-roll it, and I'd keep the per-provider quirks visible rather than averaged away.

---

## 9. Connections

| This file | Links to | Why |
|---|---|---|
| Agent fundamentals | [01_agents.md](01_agents.md) | The ReAct loop this stack runs |
| Worked end-to-end trace | [01b_agents_end_to_end.md](01b_agents_end_to_end.md) | Token/cost numbers for the loop |
| Reliability patterns | [02_agent_reliability_patterns.md](02_agent_reliability_patterns.md) | What breaks in production |
| LangChain | [03_langchain_primer.md](03_langchain_primer.md) | Layer 5, composition |
| LangGraph | [04_langgraph_deep.md](04_langgraph_deep.md) | Layer 5, state machines |
| Planning patterns | [06_planner_executor_patterns.md](06_planner_executor_patterns.md) | How a single agent plans |
| Multi-agent | [07_multi_agent_orchestration.md](07_multi_agent_orchestration.md) | Layer 5, multi-agent flavours |
| MCP | [08_mcp_protocol_deep.md](08_mcp_protocol_deep.md) | Layer 6, the protocol |
| vLLM internals | [../6.llms/05_vllm_internals.md](../6.llms/05_vllm_internals.md) | Layer 1, the runtime |
| RAG | [../7.rag/01_rag.md](../7.rag/01_rag.md) | Retrieval as an agent tool |

---

## Key Takeaway

Seven layers, and only the bottom one is the AI: **model → runtime → host → format → SDK → framework → protocol**. Product names that sound comparable usually sit on different layers — Groq is a host, Ollama a runtime, LangChain a framework, MCP a protocol. There are only **two wire formats**, OpenAI-style (spoken by nearly everything, including Ollama and vLLM) and Anthropic-style (Claude only), so provider flexibility is a two-item problem. A **workflow** has its control flow in your code; an **agent** lets the model choose — and most production systems should be workflows. Frameworks earn their place for checkpointing, HITL and replay, not for composition. Knowing when to *delete* the framework is the senior signal.

---

## Code Practice

- [../code_practice/08_agents/01_react_agent.py](../code_practice/08_agents/01_react_agent.py) — the loop with no framework
- [../code_practice/08_agents/02_tool_calling.py](../code_practice/08_agents/02_tool_calling.py) — provider-native tool calling
- [../code_practice/08_agents/03_langgraph_agent/](../code_practice/08_agents/03_langgraph_agent/) — the same loop as a state machine

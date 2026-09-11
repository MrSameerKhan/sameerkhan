# Agents — Interview Question Bank

> **How to use this.** Part A is a cold-recall drill — cover the right column, answer aloud, uncover. Part B holds the questions your theory files **don't** cover, answered in full. Part C indexes the 48 Q&As already written across `8.agents/` so you drill them where they live rather than reading duplicates. Part D is resume defence. Part E is what you ask them.
>
> **Answer out loud. Always.** Reading an answer you already agree with teaches you nothing.
>
> 🎯 **For the 20 highest-probability questions, use [AGENTS_SPOKEN_ANSWERS.md](AGENTS_SPOKEN_ANSWERS.md) instead** — full model answers written as spoken prose, with the four-move structure named. This file is the wider net; that one is what you drill.

---

# Part A — Rapid Fire (cover the right column)

| Question | Answer in one breath |
|---|---|
| What is an agent? | An LLM in a loop that can call tools, see results, and decide when it's done. |
| Workflow vs agent? | Workflow = control flow in **your code**. Agent = the **model decides**. Default to workflow. |
| Where does memory live? | In your process. The API is stateless — you resend the whole history every call. |
| What ends the agent loop? | `stop_reason` / `finish_reason` no longer asking for a tool. |
| How many wire formats matter? | Two. OpenAI-style (nearly everything) and Anthropic-style (Claude only). |
| Groq — what is it? | A **host** with custom silicon. Makes no models. |
| Ollama — what is it? | A local **runtime** wrapping llama.cpp, serving an OpenAI-compatible endpoint. |
| vLLM — what is it? | A self-hosted GPU **inference server**. PagedAttention, continuous batching. |
| LangChain vs LangGraph? | LangChain composes chains (LCEL). LangGraph runs **stateful graphs** with checkpointing + HITL. |
| What does LangGraph actually buy you? | Checkpointing, HITL interrupts, replay/time-travel, conditional routing. **Not** composition. |
| Checkpointer vs Store? | Checkpointer = **thread-scoped**. Store = **cross-thread**, keyed by user. You need both. |
| What's a reducer? | Merges a node's partial update into state. Without one, updates **overwrite** instead of append. |
| MCP in one line? | An open protocol serving tools/resources/prompts cross-process — turns M×N integrations into M+N. |
| MCP: tool vs resource? | Control dynamics: tools are **model**-controlled, resources **application**-controlled, prompts **user**-controlled. |
| MCP transports? | `stdio` for local subprocesses, **Streamable HTTP** for remote. HTTP+SSE is deprecated. |
| MCP auth? | stdio = process-level trust, no auth. Remote HTTP = **OAuth 2.1**. |
| Biggest MCP security risk? | Indirect prompt injection via **tool output**; plus **tool poisoning** via tool descriptions. |
| The 5 agent failure modes? | Tools-as-text · composition collapse · post-success wander · infinite loop · hallucinated tool/args. |
| Stop an agent looping forever? | Three layers: hard iteration cap, duplicate-call detection, explicit finish check. |
| Five production metrics? | Task success · tool-call accuracy · efficiency · cost per task · safety violations. |
| Memory tiers? | Working (context) → short-term (session) → long-term = **episodic / semantic / procedural**. |
| Episodic vs semantic? | Episodic = timestamped **events**. Semantic = de-tensed **facts**. |
| When multi-agent? | Tool overload (10+), distinct expertise, or clear phases. Otherwise don't. |
| #1 multi-agent failure? | **Lost-in-translation** — free-text handoffs. Fix with typed schemas at every boundary. |
| ReAct vs Plan-and-Execute? | ReAct interleaves and adapts. P&E plans upfront — auditable, but goes stale. |
| Anthropic's 5 workflow patterns? | Prompt chaining · routing · parallelization · orchestrator-workers · evaluator-optimizer. |
| Parallelization vs orchestrator-workers? | **Who decides the subtasks** — you in code, vs the LLM at runtime. |
| Classic vs agentic RAG? | Classic always retrieves (a workflow). Agentic makes retrieval a **tool the model chooses**. |
| Why isn't success rate enough? | Hides cost, latency, trajectory waste, variance, and safety. Report a **vector**. |
| 95% per-step over 10 steps? | ~60%. Multi-step accuracy **compounds** — long tasks fail more, by arithmetic. |
| Agent cost vs single call? | Often 10×+. Every turn resends the whole growing history. |

---

# Part B — Questions Your Theory Files Don't Cover

These came from 2026 interview-prep sources and current specs. They are the gaps.

### B1. What's the difference between a workflow and an agent?

A workflow orchestrates LLM calls through **predefined code paths** — I wrote the sequence, it's the same every run. An agent lets the **model direct its own process**: it picks tools, decides order, decides when it's done, so the path varies per input. The tradeoff is concrete — workflows give predictable cost, latency and debuggability; agents give flexibility on tasks whose steps genuinely can't be enumerated. Most systems marketed as agents are workflows, and that's usually the right call. I reach for a true agent only when I can't write the steps down in advance.

### B2. Name Anthropic's workflow patterns.

Five, and they compose. **Prompt chaining** — sequential calls with programmatic gates between steps. **Routing** — classify the input, dispatch to a specialised prompt or model; this is also the main cost lever, routing easy inputs to a cheap model and hard ones to a frontier model. **Parallelization** — two flavours: *sectioning* (independent subtasks in parallel) and *voting* (same task N times, take consensus). **Orchestrator-workers** — a central LLM decomposes dynamically, delegates, synthesises. **Evaluator-optimizer** — generate, critique, revise in a loop against clear criteria. The distinction people miss is parallelization vs orchestrator-workers: in parallelization *I* define the subtasks in code; in orchestrator-workers the *model* defines them per input — which is where a workflow starts becoming an agent.

### B3. LangGraph: Store vs Checkpointer?

Different scopes. The **checkpointer** is **thread-scoped** — it persists graph state for one conversation, keyed by `thread_id`, and is what makes multi-turn work and crash-resume possible. The **Store** is **cross-thread**, keyed by namespace plus a user id, and holds durable facts like preferences. The clean way to say it: *without the checkpointer, every invoke is a fresh conversation; without the Store, every new thread is a fresh relationship.* Production needs both. The classic bug is putting user preferences in the checkpointer — the user comes back tomorrow with a new `thread_id` and the agent has amnesia.

### B4. Two parallel branches update the same state key with no reducer. What happens?

Without a reducer the default is **replacement**, so the branches race and one silently overwrites the other — no error, just lost data, and it's non-deterministic so it won't reproduce reliably in testing. The fix is to declare a reducer on that key via `Annotated[list, add_messages]` or a custom merge function, so concurrent updates combine rather than clobber. This is the single most common LangGraph bug and it's worth naming as such.

### B5. MCP — tools vs resources vs prompts?

The real distinction is **control dynamics**, not data type. **Tools are model-controlled** — the LLM decides to invoke them, they take arguments and have side effects. **Resources are application-controlled** — the host decides what to pull into context; they're read-only data addressed by URI. **Prompts are user-controlled** — a human picks them from the host UI, typically a slash-command. So the design rule is: side effects or live computation → tool; static reference data → resource; a workflow a person triggers → prompt.

### B6. Walk me through an MCP connection lifecycle.

Three phases. **Initialization** — client and server exchange protocol version and negotiate capabilities, so each knows what the other supports. **Operation** — normal request/response over JSON-RPC 2.0: `tools/list` for discovery, `tools/call` to invoke, `resources/read` to pull data. **Shutdown** — clean teardown. Capability negotiation up front is what lets the protocol version without breaking older clients.

### B7. What is tool poisoning?

Malicious instructions hidden in a tool's **metadata** — the description or the JSON schema — rather than in its output. The model reads every tool description during tool selection, so a poisoned description is inside the trust boundary before any tool is even called. It's distinct from indirect prompt injection, which arrives in tool *results* after invocation. Mitigations: pin and review server versions rather than auto-updating, treat third-party tool descriptions as untrusted input, allowlist which servers can register tools, and — in an enterprise — run a registry with vetted servers instead of letting developers add arbitrary ones.

### B8. How does auth work in MCP, and what changed for enterprises?

It depends on transport. **stdio** servers run as a local subprocess of the host, so trust is process-level and there's no separate auth — the server inherits the user's credentials via environment. **Remote HTTP** servers use **OAuth 2.1**. The enterprise-relevant change is **Enterprise-Managed Authorization**, which moves authorization decisions to the organisation's identity provider instead of per-server user consent. That matters in a regulated environment because per-server consent doesn't scale and isn't auditable — you can't have every employee individually granting a server access to internal systems. With IdP-managed authorization you get central policy, group-based access and a real audit trail.

### B9. How would you scale an MCP deployment?

The constraint is that MCP sessions are **stateful at the connection level**, so you can't naively round-robin across instances. Options: sticky routing so a session always lands on the same instance, a shared session store so any instance can serve any session, or a gateway that terminates sessions and fans out. The second problem at scale is **token bloat** — many servers × many tools means the tool schemas alone can consume a large slice of context before the user's question arrives. Mitigations: deferred tool loading so schemas load on demand, server-side filtering to expose only relevant tools per user or task, and pagination on large results.

### B10. When would you delete the framework?

Whenever the framework isn't earning its cost. One LLM call with no tools and no state — use the SDK, nothing else. A fixed sequence of calls — plain Python functions; a `for` loop is not technical debt. A latency-critical path — go direct, since a chain layer adds meaningful per-call overhead. I keep the framework when I need something genuinely hard to hand-roll correctly: checkpointing with crash resume, human-in-the-loop interrupts, replay for debugging, or a large body of pre-built integrations. What I won't do is adopt a framework purely for composition — functions already compose, and the cost is real: hidden prompts, breaking changes across minor versions, and a heavy dependency tree.

### B11. Your agent must work across three LLM providers. Architecture?

I'd resist writing an abstraction layer. There are only two wire formats, and a wrapper hides exactly the differences that matter — content blocks vs `tool_calls`, `stop_reason` vs `finish_reason`, `content` going `None` on tool calls. Concretely: one code path against the OpenAI-compatible format already covers OpenAI, Ollama, vLLM, Groq, Together and OpenRouter with nothing but a `base_url` change, and a second explicit path handles Anthropic. Two honest implementations beat one leaky abstraction. If I needed real provider-agnostic routing in production I'd put a gateway or OpenRouter in front rather than hand-roll it, and I'd keep provider quirks visible instead of averaged away.

### B12. Classic RAG vs agentic RAG?

Classic RAG is a **workflow**: retrieve, then generate, same path every time. Predictable cost, easy to evaluate — you can score retrieval precision and answer quality separately. **Agentic RAG** makes retrieval a **tool the model chooses to call**, so it can decide *whether* to retrieve at all, reformulate a weak query and retry, or issue several targeted queries instead of one broad one. You buy adaptivity on multi-hop and ambiguous questions. You pay in latency, cost, and evaluation difficulty — you're now scoring a **trajectory** rather than a single retrieval, and a bad first query can cascade. I'd start classic, measure where it fails, and add agency only at the specific failure — usually query reformulation first.

### B13. How do you control cost in an agent system?

Four levers, roughly in order of return. **Model routing** — a cheap model for classification and routing, a frontier model only where judgment is needed. **Prompt caching** — the system prompt and tool schemas are identical every turn, so cache that prefix; it's the single biggest win in a loop that resends history. **Context discipline** — truncate tool outputs (a full CSV in a tool result blows the window), compact old turns, and namespace state per agent so workers don't each load the whole conversation. **Hard budgets** — cap iterations *and* cumulative tokens per task, and abort-and-escalate rather than letting a stuck agent burn. The thing to say explicitly: agents cost roughly an order of magnitude more than a single call because every turn resends a growing history, so cost per *completed task* is the metric, not cost per call.

### B14. What breaks first when you take an agent from demo to production?

In my reading of the failure taxonomy, the ordering is fairly consistent. **Loops** show up immediately — the demo had three happy-path queries, production has ambiguous ones where the model can't tell it's done. **Cost** is next and it's usually a surprise, because nobody priced the growing-context effect across turns. **Tool-call reliability** third — schema drift and hallucinated argument names that never appeared in testing. Then **prompt injection via tool output** once the agent touches anything user-supplied or web-sourced. And underneath all of it, **no evaluation harness**, so you can't tell whether a prompt change helped or hurt. The pattern is that demos test the happy path and production is all edge case.

### B15. Multi-step accuracy — why do long tasks fail more?

Because per-step accuracy **compounds multiplicatively**. At 95% per step, ten steps gives 0.95^10 ≈ 60%. That has two design consequences: fewer steps is a reliability feature, not just a cost one, so I'd rather have one well-designed tool than three chained ones; and you need checkpoints or verification mid-trajectory rather than only at the end. It also reframes evaluation — reporting a single end-to-end success rate on long tasks hides *where* the trajectory broke, so you want per-step tool-call accuracy alongside it.

---

# Part C — Index to the 48 Q&As You Already Have

Drill these **in their source file** — the surrounding context is what makes them stick.

| File | Q&A section | Count | Covers |
|---|---|---|---|
| [`00_agent_stack_foundations.md`](../../8.agents/00_agent_stack_foundations.md) | §8 | 6 | Workflow vs agent · running any model · API differences · framework necessity · MCP placement · multi-provider |
| [`01_agents.md`](../../8.agents/01_agents.md) | Interview Q&A | 4 | What is an agent · ReAct · MCP · preventing rogue agents |
| [`01b_agents_end_to_end.md`](../../8.agents/01b_agents_end_to_end.md) | §8 | 6 | Agent vs LLM call · ReAct vs CoT · LangGraph vs raw loop · rogue prevention · memory types · P&E vs ReAct |
| [`02_agent_reliability_patterns.md`](../../8.agents/02_agent_reliability_patterns.md) | §7 | 5 | **5 failure modes · loop prevention · planner/executor · tool hallucination · 5 metrics** |
| [`03_langchain_primer.md`](../../8.agents/03_langchain_primer.md) | §12 | 4 | LCEL · when NOT to use LangChain · LangChain vs LangGraph · Passthrough vs Lambda |
| [`04_langgraph_deep.md`](../../8.agents/04_langgraph_deep.md) | §14 | 5 | **Why LangGraph over AgentExecutor · reducers · HITL · stream modes · infinite loops** |
| [`05_agent_memory.md`](../../8.agents/05_agent_memory.md) | §11 | 5 | Memory types · what to write · Letta/MemGPT · episodic vs semantic · conflicts |
| [`06_planner_executor_patterns.md`](../../8.agents/06_planner_executor_patterns.md) | §11 | 4 | ReAct vs P&E · Reflexion · LATS · stopping Self-Refine |
| [`07_multi_agent_orchestration.md`](../../8.agents/07_multi_agent_orchestration.md) | §15 | 5 | When multi-agent · framework choice · **lost-in-translation** · cost control · code-as-action |
| [`08_mcp_protocol_deep.md`](../../8.agents/08_mcp_protocol_deep.md) | §13 | 5 | What/why MCP · vs LangChain tools · tool vs resource · transports · **security** |
| [`09_agent_evaluation.md`](../../8.agents/09_agent_evaluation.md) | §15 | 5 | End-to-end eval · offline vs online · why not just success · TAU-Bench · multi-agent eval |

**Bolded rows are the highest-yield for an enterprise conversational interview.**

---

# Part D — Resume Defence

Your projects, framed for an agent-focused conversation. **Verify every number against your current resume before the interview** — `02_INTERVIEW_PACK.md` is stale (see Day 4.1 of the plan).

### D1. Walk me through a production ML system you've built.

Lead with **Document AI at scale in regulated financial services**. Structure: the problem, the constraint that made it hard (regulatory, volume, or latency), what you built, the measured result, and what you'd do differently. Have the numbers exact — weighted F1 **0.959–0.972** and **0.975**; the P1 RCA at **5 services / 21,096 pages / 2.3% true failure rate**. The RCA story is your strongest bullet because it demonstrates production ownership, not just modelling.

### D2. Tell me about your RAG work.

**Rulebook-RAG**: 36 classes, hand-written **BM25 + RRF** fusion (+5.2 pts), **cross-encoder** reranking (+3.9 pts), **0 invalid outputs across 775**. Be ready for the follow-ups — *why hybrid rather than pure dense?* (lexical catches exact policy terms and identifiers that embeddings blur), *why a cross-encoder?* (precision at the top of the list, where the generator actually looks), *what does 0/775 prove?* (constrained output validity, not answer correctness — say that distinction yourself, it's the honest and impressive answer).

### D3. How would you make Rulebook-RAG agentic?

The question that fuses your resume with their agenda. Retrieval becomes a **tool** rather than a fixed first step, so the model can reformulate a query that returned weak results or skip retrieval when the answer is already grounded. Add a self-check tool for validation, HITL on any low-confidence classification, a hard iteration cap, and full trajectory logging for audit. Then — critically — **name what you'd lose**: determinism, predictable per-query cost, and a clean separable evaluation. In a bank, those losses may not be acceptable, and saying so demonstrates the judgment they're actually hiring for.

### D4. Have you deployed agents in production?

Be honest and precise. You have **built and run** four agent sessions — ReAct from scratch, provider-native tool calling, a LangGraph agent with checkpointed multi-turn state, and a multi-agent document workflow. Say plainly that this is **project work, not production deployment**, and that your production experience is Document AI under regulatory constraint. Then bridge: the reliability concerns you'd carry over — approval gates, audit trails, cost ceilings, per-class thresholds — are the *same* concerns, which is exactly why the enterprise agent problem is legible to you.

**Do not claim QLoRA / Mistral-7B fine-tuning.** Not on your resume, Phase 09 is parked, indefensible under follow-up.

---

# Part E — Questions To Ask Them

Asking these signals seniority, and the answers tell you which prepared depth to deploy.

1. **Are you building agents or integrating vendor tooling?** Decides whether they want primitives or framework fluency.
2. **What does your evaluation story look like today?** Most enterprises have none — if they admit it, your eval depth becomes the differentiator.
3. **Where does human-in-the-loop sit in your workflows?** Signals you think about approval gates before someone makes you.
4. **Is MCP on your roadmap, or are tools in-process today?** Shows you track the protocol layer, not just frameworks.
5. **What's the biggest reliability problem you've hit?** Their answer maps directly onto the five failure modes — and you can respond with the specific fix.
6. **How do you control cost per task?** Enterprise-native question; opens the routing/caching/budget discussion.

---

## Related

- 4-day schedule → [AGENTS_4DAY_PLAN.md](AGENTS_4DAY_PLAN.md)
- Stack foundations → [../../8.agents/00_agent_stack_foundations.md](../../8.agents/00_agent_stack_foundations.md)
- Agent theory folder → [../../8.agents/README.md](../../8.agents/README.md)
- Resume pack (**stale — rewrite Day 4**) → [../../02_INTERVIEW_PACK.md](../../02_INTERVIEW_PACK.md)

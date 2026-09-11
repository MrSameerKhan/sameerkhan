# Agents — 4-Day Interview Plan

> **Target:** interview ~**15 September 2026**. Topics named by the interviewer: **LLMs, RAG, Agents**.
> **Company:** **Lloyds Banking Group.** Rounds 1–2 technical (Hyderabad, face-to-face) — **done**. Next round is with the **UK team**.
> **Format:** conversational deep-dive — they ask, you explain out loud. No editor.
> **Budget:** 6–8 focused hours/day × 4 days ≈ 28 hours.
>
> 🎯 **Your drill file is [AGENTS_SPOKEN_ANSWERS.md](AGENTS_SPOKEN_ANSWERS.md)** — 20 model answers written to be *said*, each in the Define → Distinguish → Tradeoff → Judgment shape. A live mock on 11 Sep proved the gap is delivery, not knowledge: the "what is an agent" answer had the loop but no tools, no workflow contrast, no tradeoff. Work through that file aloud before anything else.
>
> ⚠️ **Read [LLOYDS_UK_ROUND.md](LLOYDS_UK_ROUND.md) before Day 1.** A UK retail bank round adds a layer the Hyderabad rounds almost certainly didn't test — **SS1/23 model risk** and **FCA Consumer Duty**. That layer, not agent theory, is what separates you from the rest of the pipeline. It also re-weights Day 3 (less MCP, more regulation) and adds 90 minutes to Day 4.

---

## The One Thing That Matters

**This is a speaking exam, not a reading exam.** Your theory is already strong — 3,400 lines across `8.agents/`, containing 48 written Q&As. Re-reading it will *feel* productive and will not move the needle.

Every block below ends with **SAY IT** — close the file, talk to the wall for 60–90 seconds. If you can't, you haven't learned it, you've recognised it. Recognition collapses under interview pressure.

> **Rule:** if a block runs over time, cut the reading, never the SAY IT.

---

## Your Positioning (read once, internalise)

You are **not** a fresh AI engineer. You are a **9-year Senior ML Engineer with production Document AI in regulated financial services** (ICE Data Services, Al Rajhi Bank), now moving into LLM/agent work. For a **UK retail bank** that is close to an ideal profile — they have compliance, audit, approval gates and cost ceilings, and most candidates interviewing for LLM roles have shipped a chatbot demo, not a regulated system.

**The connection to make early:** Lloyds' flagship GenAI product, **Athena**, is grounded RAG over ~13,000 authorised internal knowledge articles for customer-facing colleagues. That is structurally **your Rulebook-RAG project** at their scale. See [LLOYDS_UK_ROUND.md](LLOYDS_UK_ROUND.md) §2 and §4.

**Lead with the domain, support with the LLM depth.** When you hit something you haven't shipped, the honest bridge is strong:

> *"I haven't run that in production. What I have shipped is [X] under [regulatory/scale constraint], and the analogous problem there was [Y] — so here's how I'd approach it."*

**Do not claim:** QLoRA / Mistral-7B fine-tuning. Phase 09 is parked, it is not on your resume, and you cannot defend it. Saying it invites the one question that unravels the interview.

---

## Day 1 — Foundations + The Loop By Hand
**11 Sep · ~7h · Goal: you can place any product name on the stack, and write the agent loop from memory**

| Block | Time | Do |
|---|---|---|
| 1.1 | 60m | Read **`8.agents/00_agent_stack_foundations.md`** end to end. This is the file that fixes the confusion you named — where LangChain/LangGraph/Ollama/MCP actually sit. |
| 1.2 | 30m | **SAY IT** — for each of these, state the layer in one sentence, no notes: *Groq · Ollama · vLLM · LangChain · LangGraph · MCP · Bedrock · llama.cpp · OpenRouter · Together*. Get all 10 or repeat. |
| 1.3 | 45m | **SAY IT** — explain the two wire formats side by side. Then the two classic bugs: OpenAI `content=None` on tool calls, Anthropic `content[0]` being a thinking block. |
| 1.4 | 75m | Read **`01_agents.md`** + **`01b_agents_end_to_end.md`**. Focus on the ReAct trace and the token-count table — the cost arithmetic is an enterprise question. |
| 1.5 | 60m | **Hands-on.** Build the ladder: single call → multi-turn → tool schema → execute + feed back → `while` loop. Use `code_practice/12_agents_from_scratch/`. You must *feel* `stop_reason` flip to `tool_use`. |
| 1.6 | 45m | **SAY IT** — walk the full ReAct trace aloud for "population of Paris, then 10% of it". Every turn: what the model sees, what it emits, what you append. |
| 1.7 | 60m | **Workflows vs agents** (§5 of the foundations file). Learn Anthropic's five patterns **by name**: prompt chaining · routing · parallelization (sectioning/voting) · orchestrator-workers · evaluator-optimizer. |
| 1.8 | 30m | **SAY IT** — "Most things called agents are workflows, and that's usually right." Defend it. Then: what separates parallelization from orchestrator-workers? (*Who decides the subtasks, and when.*) |

**Day 1 checkpoint — answer all five aloud, no notes:**
1. What is an agent, and how is it different from an LLM call?
2. Where does memory live, and who owns it?
3. Walk me through one full ReAct turn.
4. Workflow vs agent — and which should I default to?
5. Name Anthropic's five workflow patterns.

*Short on time? Keep 1.1, 1.2, 1.5, 1.8.*

---

## Day 2 — Frameworks + What Breaks
**12 Sep · ~7h · Goal: framework judgment, and fluency in production failure — the hardest section in 2026 agent interviews**

| Block | Time | Do |
|---|---|---|
| 2.1 | 60m | Read **`03_langchain_primer.md`**. You need LCEL, Runnables, and §7 *When to use LangChain vs not*. Skip the legacy migration table. |
| 2.2 | 90m | Read **`04_langgraph_deep.md`**. Priority order: state + reducers → nodes/edges → **checkpointer** → **HITL interrupts** → conditional routing. `Send`/subgraphs are lower priority. |
| 2.3 | 45m | **Store vs Checkpointer** — not clearly separated in your notes, and it's a named interview question. Checkpointer = **thread-scoped**, one conversation, keyed by `thread_id`. Store = **cross-thread**, keyed by namespace + user id. *Without the checkpointer every call is a fresh conversation; without the Store every new thread is a fresh relationship.* Production needs both. |
| 2.4 | 45m | **Re-read your own** `code_practice/08_agents/03_langgraph_agent/`. With Day 1 in hand: `MemorySaver` = the checkpointer, `ToolNode` = your tool executor, `tools_condition` = your `while` condition. **Note honestly:** `with_hitl=False`, so the interrupt path was built but never run — say that if asked, don't oversell it. |
| 2.5 | 30m | **SAY IT** — "When would you NOT use LangChain?" Then the harder one: "What does LangGraph actually buy you?" (*checkpointing, HITL, replay, integrations — not composition; functions already compose.*) Framework-deletion judgment is an explicit senior signal. |
| 2.6 | 90m | Read **`02_agent_reliability_patterns.md`** — **the highest-value file in the folder for this interview.** The 5 failure modes, the fix per mode, and the 5-axis scorecard. |
| 2.7 | 45m | **SAY IT** — name all 5 failure modes and the fix for each, cold. Then the scorecard: task success · tool-call accuracy · efficiency · cost ceiling · safety failures. |
| 2.8 | 45m | **Enterprise angle.** Map each reliability pattern to a compliance concern: HITL → approval gates · audit log → traceability · tool allowlist → least privilege · budget cap → cost control · dedup → runaway spend. This mapping is *your* differentiator. |

**Day 2 checkpoint:**
1. LangChain vs LangGraph vs raw SDK — when each?
2. Store vs Checkpointer?
3. Five failure modes and their fixes.
4. How do you stop an agent looping forever? (*three layers, not one*)
5. What five metrics do you monitor in production?

*Short on time? Keep 2.3, 2.6, 2.7, 2.8.*

---

## Day 3 — MCP, Memory, Multi-Agent, Eval
**13 Sep · ~7h · Goal: cover the remaining surface at explain-depth, with MCP sharp because it's the hot topic**

| Block | Time | Do |
|---|---|---|
| 3.1 | 75m | Read **`08_mcp_protocol_deep.md`**. Then add these four, which your file doesn't yet frame and which separate senior answers: **(a) control dynamics** — tools are *model*-controlled, resources are *application*-controlled, prompts are *user*-controlled; that's the real distinction, not data type. **(b) lifecycle** — initialization (capability negotiation) → operation → shutdown. **(c) auth** — stdio has no auth (process-level trust); remote HTTP uses OAuth 2.1. **(d) tool poisoning** — malicious instructions hidden in tool *descriptions and schemas*, which the model reads during tool selection. Distinct from injection via tool *output*. |
| 3.2 | 30m | **Enterprise MCP.** Two specifics worth having: **Enterprise-Managed Authorization** (2026) moves authorization to the org's identity provider instead of per-server consent — exactly what a bank needs. And **scaling**: MCP sessions are stateful, so multi-instance deployments need sticky routing or a shared session store. Also **deferred tool loading** to stop a many-server deployment drowning the model in tool-schema tokens. |
| 3.3 | 30m | **SAY IT** — What is MCP and what problem does it solve? (*M×N → M+N*) Then: tool vs resource? Then: biggest security risk? (*indirect injection via tool output; tool poisoning via metadata*) |
| 3.4 | 60m | Read **`05_agent_memory.md`**. Learn the tiers cold: working → short-term → long-term, and long-term splits **episodic / semantic / procedural**. |
| 3.5 | 30m | **SAY IT** — the memory tiers, plus "how do you decide what's worth writing to long-term memory?" (*density filter, LLM judge, confidence threshold — write less than you think*). Note the namespacing trap: forgetting `user_id` leaks memories across users. In a bank that's a breach, and saying so lands. |
| 3.6 | 60m | Read **`06_planner_executor_patterns.md`** (ReAct / Plan-and-Execute / ReWoo / Reflexion — the others are recognition-only) and **`07_multi_agent_orchestration.md`** (the 5 coordination patterns + the anti-patterns table). |
| 3.7 | 30m | **SAY IT** — When does multi-agent beat single-agent? Then the sharp one: what's the #1 multi-agent failure mode? (*lost-in-translation — free-text handoffs between agents; fix with typed Pydantic contracts at every boundary*). |
| 3.8 | 75m | Read **`09_agent_evaluation.md`**. Outcome vs process eval; the metric vector; offline gate + online sampling. **SAY IT:** "How would you evaluate an agent?" — and never answer with a single number. |

**Day 3 checkpoint:**
1. MCP: what, why, and the three capabilities by control dynamics.
2. The memory tiers, with a use case each.
3. When multi-agent, and when not?
4. How do you evaluate an agent end to end?
5. Why is a single success-rate number insufficient?

*Short on time? Keep 3.1, 3.3, 3.5, 3.8.*

---

## Day 4 — Resume Defence, Connective Tissue, Mock
**14 Sep · ~7h · Goal: your projects, the LLM↔RAG↔Agent story, and a full dress rehearsal**

| Block | Time | Do |
|---|---|---|
| 4.1 | 90m | **Rewrite `02_INTERVIEW_PACK.md`.** It is stale against your current resume — it says "8 years / 94% accuracy / 60% RCA reduction" and describes the RAG project as FAISS + Streamlit + `llama3.2:1b`. Reality: **9 years**, weighted F1 **0.959–0.972** and **0.975**, P1 RCA = **5 services / 21,096 pages / 2.3% true failure rate**, and Rulebook-RAG = **36 classes, hand-written BM25 + RRF (+5.2 pts), cross-encoder (+3.9 pts), 0 invalid of 775**. Walking in with stale numbers is the most avoidable failure available to you. |
| 4.2 | 45m | **SAY IT** — 90-second walkthrough of **Rulebook-RAG**. Then defend the choices: why BM25 + RRF and not pure dense? why a cross-encoder? what does "0 invalid of 775" actually prove? |
| 4.3 | 45m | **Agentic RAG** — the bridge between two of their three topics. Classic RAG = always retrieve then generate = a *workflow*. Agentic RAG = retrieval is a **tool the model chooses**, so it can reformulate a weak query, retry, or skip retrieval entirely. You buy adaptivity; you pay in latency, cost and eval difficulty — you're now scoring a trajectory, not a single retrieval. **SAY IT** as one connected answer. |
| 4.4 | 45m | **SAY IT** — "How would you turn Rulebook-RAG into an agent?" This is the question that fuses your resume with their agenda. Retrieval becomes a tool; add a self-check tool; add HITL on any low-confidence classification; cap iterations; log every trajectory for audit. Name what you'd *lose*: determinism, predictable cost, and a simple eval story. |
| 4.5 | 60m | **Enterprise system design, out loud.** *"Design an agent that answers policy questions for bank staff."* Cover: tool surface, retrieval, HITL on anything advisory, audit logging, cost ceiling per query, eval set, injection defence, PII handling. 15 minutes of talking, no notes. |
| 4.6 | 90m | **Full mock.** Work `AGENTS_QA_BANK.md` end to end, out loud, timed. Mark every question that comes out shaky. |
| 4.7 | 45m | **Weak-spot pass.** Re-drill only the shaky ones. Do not re-read what already works. |
| 4.8 | 20m | **Prepare your questions for them.** *Are you building agents or integrating vendor tooling? What's your eval story today? Where does HITL sit? Is MCP on the roadmap?* Asking these signals seniority — and the answers tell you which of your prepared depth to deploy. |

**Day 4 checkpoint:** every question in the bank answered aloud without notes, and the Rulebook-RAG walkthrough clean in 90 seconds.

*Short on time? Keep 4.1, 4.2, 4.4, 4.6.*

---

## Interview Day — 15 Sep

**Morning, 30 minutes, nothing new.** Skim only:
- `00_agent_stack_foundations.md` § Key Takeaway
- The 5 failure modes
- Your Rulebook-RAG numbers

**Three habits that carry a conversational interview:**

1. **Structure before detail.** *"There are three parts to that — let me take them in order."* Buys thinking time and sounds organised.
2. **Name the tradeoff, always.** No technique is free. Saying what you'd *lose* is what separates senior from mid.
3. **Say "I haven't shipped that" and bridge.** Never bluff. Enterprise interviewers are calibrated for it, and the honest bridge from your Document AI experience is genuinely strong.

---

## Coverage Map

| Theory file | Day | Depth |
|---|---|---|
| `00_agent_stack_foundations.md` | 1 | **Master** — say it cold |
| `01_agents.md` · `01b_agents_end_to_end.md` | 1 | **Master** |
| `03_langchain_primer.md` | 2 | Explain + judgment |
| `04_langgraph_deep.md` | 2 | **Master** — most-probed framework |
| `02_agent_reliability_patterns.md` | 2 | **Master** — highest yield |
| `08_mcp_protocol_deep.md` | 3 | **Master** — hot topic |
| `05_agent_memory.md` | 3 | Explain |
| `06_planner_executor_patterns.md` | 3 | ReAct/P&E/ReWoo/Reflexion explain; rest recognise |
| `07_multi_agent_orchestration.md` | 3 | Explain + when-not |
| `09_agent_evaluation.md` | 3 | **Master** — enterprise cares |
| `02_INTERVIEW_PACK.md` | 4 | Rewrite, then master |

---

## What To Skip

Deliberately out of scope for four days — recognise the name, don't study:

- LATS, STORM, ADaPT (recognise only)
- CrewAI / AutoGen / Swarm internals — know *when* you'd pick each, not their APIs
- Computer Use / browser agents
- MCP sampling + elicitation (know they exist)
- Building an MCP server hands-on — explaining it is enough at 4 days
- LangGraph `Send` / subgraphs beyond "fan-out exists"

---

## Related

- Question bank → [AGENTS_QA_BANK.md](AGENTS_QA_BANK.md)
- Stack foundations → [../../8.agents/00_agent_stack_foundations.md](../../8.agents/00_agent_stack_foundations.md)
- Agent theory folder → [../../8.agents/README.md](../../8.agents/README.md)
- Your agent code → [../08_agents/](../08_agents/)
- Transformer/LLM board tracker → [MASTERY_PLAN.md](MASTERY_PLAN.md)

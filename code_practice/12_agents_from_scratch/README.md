# 12. Agents From Scratch — The Module Ladder

> Learn agents **by building**, one idea per module, with the flow visible.
>
> Derived by extracting **every section heading from all 11 files in [`8.agents/`](../../8.agents/)** and mapping each to a module — not from a remembered sketch. A 2026 practice check added two blocks the theory didn't have: **security/injection** and **observability**, both now treated as foundational.

**Teaching-shaped, not production-shaped.** The sessions in [`../08_agents/`](../08_agents/) are 180+ line files with policy databases and four-tool schemas — the mechanism is buried. These are the opposite: minimal, readable top to bottom, one concept each.

**Hard rule:** over ~90 lines means a module is teaching two things. Split it.

---

## Status

| | Built |
|---|---|
| Shared foundation | ✅ 3 / 3 |
| Modules | ✅ 1 / 37 |

## How to run

```bash
conda activate sameerkhan
cd code_practice/12_agents_from_scratch
python _providers.py              # health check — should print 3x OK
python 01_standard_llm_call.py
```

Every module also works cell-by-cell via `# %%` in the VS Code Interactive Window.

**Requires:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `ollama serve` running with `ollama pull llama3.2`.

---

## Shared foundation

| File | Status | Purpose |
|---|---|---|
| `_tools.py` | ✅ | Four tiny tools, exported in **both** wire envelopes from one source of truth. Kept identical across modules so the modules are **diffable** — only the machinery changes. |
| `_providers.py` | ✅ | `get_client("openai" \| "anthropic" \| "local")`. Run standalone for a health check. |
| `_fake_model.py` | ✅ | Scripted stand-in with the real client's interface. **Required from module 12** — a real model won't loop, hallucinate a tool name, or attempt an injection *on demand*, so without this the guards are untestable. |

---

## Block A — The wire

| # | File | The one idea | |
|---|---|---|---|
| 01 | `01_standard_llm_call.py` | Same prompt → OpenAI, Anthropic, local. **Two wire formats**: `choices[0].message.content` (string) vs `content[]` (typed blocks); `finish_reason` vs `stop_reason`. Ends with raw `requests.post` — the SDK is a wrapper, not a capability. | ✅ |
| 02 | `02_multi_turn_by_hand.py` | Carry the list yourself. **The API is stateless — memory *is* the list.** Tokens printed per turn. | ⬜ |
| 03 | `03_structured_output.py` | JSON mode, Pydantic, `strict`. **Why free-text parsing is a bug**, and how constrained output deletes a failure class. | ⬜ |

## Block B — Workflows *(deliberately before agents)*

| # | File | The one idea | |
|---|---|---|---|
| 04 | `04_workflow_prompt_chaining.py` | Sequential calls with a **programmatic gate** between steps. | ⬜ |
| 05 | `05_workflow_routing.py` | Classify → dispatch. **Cheap model for easy inputs** — the main cost lever. | ⬜ |
| 06 | `06_workflow_parallel_and_evaluator.py` | Sectioning, voting, and generate → critique → revise. | ⬜ |

*Built before any agent so "most things called agents are workflows" is **felt**, not read.*

## Block C — The loop

| # | File | The one idea | |
|---|---|---|---|
| 07 | `07_agent_without_sdk_react.py` | ReAct in prose — **you write the regex**. Local Llama buries the call in text and the parser misses it. **FM1, live.** | ⬜ |
| 08 | `08_agent_with_sdk_tool_calling.py` | Native tool calling, both envelopes. `finish_reason` flips; `content` is `None` on a tool turn. The regex from 07 is deleted. | ⬜ |
| 09 | `09_agent_parallel_tool_calls.py` | Several tool calls in **one** turn — return **all** results in a single message, or the model quietly stops parallelising. | ⬜ |
| 10 | `10_rag_as_a_tool.py` | Retrieval as a **tool the model chooses**. Classic vs **agentic RAG**. | ⬜ |
| 11 | `11_agent_cost_and_tokens.py` | Per-turn cost table. **Why an agent costs ~10× a single call.** | ⬜ |

## Block D — Reliability

| # | File | The one idea | |
|---|---|---|---|
| 12 | `12_failure_modes_reproduced.py` | All **five failure modes on demand** via `_fake_model`. | ⬜ |
| 13 | `13_guardrails_and_budgets.py` | Iteration cap, duplicate detection, Pydantic arg validation, token budget, abort-and-escalate. | ⬜ |
| 14 | `14_audit_log_and_hitl.py` | Append-only JSONL trace + **human gate** on the write tool. SS1/23 traceability, Consumer Duty approval. | ⬜ |

## Block E — Security

| # | File | The one idea | |
|---|---|---|---|
| 15 | `15_prompt_injection_via_tools.py` | A tool returns attacker text — and the agent **obeys it**. The attack, working. | ⬜ |
| 16 | `16_injection_defences.py` | Spotlighting, **capability isolation**, output validation, **dual-LLM (CaMeL)**. | ⬜ |

## Block F — Observability

| # | File | The one idea | |
|---|---|---|---|
| 17 | `17_tracing_and_observability.py` | Structured trace per run: intent, tool calls, args, latency, tokens, cost, approvals. **You cannot retrofit an audit trail.** | ⬜ |

## Block G — Frameworks

| # | File | The one idea | |
|---|---|---|---|
| 18 | `18_langchain_lcel.py` | LCEL, Runnables, `.invoke`/`.batch`/`.stream` — then the same job in three plain lines. | ⬜ |
| 19 | `19_langgraph_minimal_agent.py` | Module 08 as a state machine. **The `while` *is* the conditional edge.** | ⬜ |
| 20 | `20_langgraph_checkpoint_and_hitl.py` | `thread_id`, `interrupt()`, resume via `Command`, **time travel**. | ⬜ |
| 21 | `21_langgraph_parallel_subgraphs_streaming.py` | `Send`, subgraphs, the four `stream_mode`s, and the reducer-collision trap. | ⬜ |

## Block H — Memory

| # | File | The one idea | |
|---|---|---|---|
| 22 | `22_memory_short_term_strategies.py` | Full vs window vs summary, side by side — and **what the model forgets** under each. | ⬜ |
| 23 | `23_memory_entity_keyvalue.py` | Structured entity facts — the cheap tier people skip for vectors. | ⬜ |
| 24 | `24_memory_long_term_vector.py` | Recall into a **new** conversation. Episodic vs semantic. **Per-user namespacing** — a second user proves no leakage. | ⬜ |
| 25 | `25_memory_procedural_and_forgetting.py` | Reusable skills, plus **decay and eviction** — a store that only grows degrades. | ⬜ |

## Block I — Planning

| # | File | The one idea | |
|---|---|---|---|
| 26 | `26_plan_and_execute.py` | Plan upfront vs ReAct interleaved. Inspectable — and stale when step 1 surprises you. | ⬜ |
| 27 | `27_rewoo_parallel_planning.py` | All calls enumerated upfront, run in parallel, one reasoning pass. | ⬜ |
| 28 | `28_reflexion_and_self_refine.py` | Fail → critique → store → retry. Plus the **three stopping criteria**. | ⬜ |

## Block J — Multi-agent

| # | File | The one idea | |
|---|---|---|---|
| 29 | `29_multi_agent_pipeline.py` | Typed contracts at each hand-off — then a free-text one that silently misparses. **Lost-in-translation, live.** | ⬜ |
| 30 | `30_multi_agent_supervisor.py` | Dynamic routing, and the 5–10× token cost vs single-agent. | ⬜ |
| 31 | `31_multi_agent_debate_and_handoff.py` | Proposer / critic / judge, and Swarm-style handoff. | ⬜ |

## Block K — MCP

| # | File | The one idea | |
|---|---|---|---|
| 32 | `32_mcp_server_capabilities.py` | Tool, resource, prompt — and **control dynamics**: model-controlled vs application-controlled vs user-controlled. | ⬜ |
| 33 | `33_mcp_client_and_lifecycle.py` | initialize → capability negotiation → operation → shutdown. Runtime discovery. | ⬜ |
| 34 | `34_mcp_transports_and_mrtr.py` | `stdio` vs Streamable HTTP. The **2026-07-28 stateless spec and MRTR**. | ⬜ |
| 35 | `35_mcp_in_an_agent_and_security.py` | MCP tools into a LangGraph agent, then **tool poisoning** via a malicious description. | ⬜ |

## Block L — Evaluation

| # | File | The one idea | |
|---|---|---|---|
| 36 | `36_agent_eval_harness.py` | Score a **trajectory**, not an answer. Runs against modules 08 and 26. | ⬜ |
| 37 | `37_eval_variance_and_redteam.py` | ×5 seeds — **a bimodal 80% beside a uniform 80%**. Plus an injection red-team corpus. | ⬜ |

---

## Coverage map

| Theory file | Modules |
|---|---|
| `00_agent_stack_foundations.md` | 01, 02, 04, 05, 06, 10, 18 |
| `01_agents.md` · `01b_agents_end_to_end.md` | 07, 08, 11, 12, 15, 19, 22–30 |
| `02_agent_reliability_patterns.md` | 12, 13, 14, 26, 36 |
| `03_langchain_primer.md` | 03, 18 |
| `04_langgraph_deep.md` | 19, 20, 21 |
| `05_agent_memory.md` | 22, 23, 24, 25 |
| `06_planner_executor_patterns.md` | 07, 26, 27, 28 |
| `07_multi_agent_orchestration.md` | 29, 30, 31 |
| `08_mcp_protocol_deep.md` | 32, 33, 34, 35 |
| `09_agent_evaluation.md` | 17, 36, 37 |

**Recognition-only, not built:** LATS, STORM, ADaPT, Tree-of-Thought, CrewAI/AutoGen/smolagents internals — each named in the closing comment of its nearest module with a pointer to the theory.

> ⚠️ **MCP theory needs a fix first.** The **2026-07-28 spec** made MCP stateless and introduced **MRTR**, deprecating server-initiated `sampling`, `elicitation` and `roots`. [`08_mcp_protocol_deep.md`](../../8.agents/08_mcp_protocol_deep.md) §3 still presents them as current. Modules 32–35 are written against the new spec.

---

## Effort

**Full ladder ≈ 32 hours.** Blocks are independent once the three shared files exist — build in any order, pause between blocks.

**Minimum viable subset ≈ 6 hours:** 01, 07, 08, 12, 13, 14 — the wire, both loop forms, the reliability trio.

## Verification

Every module must **run top-to-bottom**, **print its own lesson** (visible in stdout, not merely implemented), **stay under ~90 lines**, and **work cell-by-cell**.

Cross-module:
- **07, 08, 19** produce the **same final answer** — only the machinery differs on screen. That equivalence is the point of the ladder.
- **13, 14** catch every scenario **12** produces.
- **16** defeats the exact attack **15** demonstrates.
- **37** shows a bimodal and a uniform 80% side by side.

---

## Related

- Theory → [`../../8.agents/`](../../8.agents/) · start at [`00_agent_stack_foundations.md`](../../8.agents/00_agent_stack_foundations.md)
- Spoken drill answers → [`../11_interview_drills/AGENTS_SPOKEN_ANSWERS.md`](../11_interview_drills/AGENTS_SPOKEN_ANSWERS.md)
- Production-shaped sessions → [`../08_agents/`](../08_agents/)

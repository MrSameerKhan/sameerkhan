# 12. Agents From Scratch — The Module Ladder

> Learn agents **by building**, one idea per module, with the flow visible.
>
> Derived by extracting **every section heading from all 11 files in [`8.agents/`](../../8.agents/)** and mapping each to a module — not from a remembered sketch. A 2026 practice check added two blocks the theory didn't have: **security/injection** and **observability**, both now treated as foundational.

**Teaching-shaped, not production-shaped.** The sessions in [`../08_agents/`](../08_agents/) are 180+ line files with policy databases and four-tool schemas — the mechanism is buried. These are the opposite: minimal, readable top to bottom, one concept each.

**Hard rule:** over ~90 lines means a module is teaching two things. Split it.

---

## Status

| | Built | Run live |
|---|---|---|
| Shared foundation | ✅ 3 / 3 | ✅ |
| Modules | **✅ 37 / 37** | ✅ 12 / 37 |

**Built** means the code is written. **Run live** means it executed end-to-end against
real providers and printed its own lesson — the bar set under *Verification* below.
A module is not done until both are ticked.

**Legend** (same badges as [`../../CLAUDE.md`](../../CLAUDE.md)): `✅ Run` · `🔧 Code-built` · ⬜ not written yet

### Verified runs

Captured output lives in each module's `_details.md`, not here — one canonical home per
[`../../RULES.md`](../../RULES.md).

| Block | Verified live | Notes |
|---|---|---|
| A wire | **01 · 02** | 03 needs an API key |
| B workflows | — | 04–06 need keys; 05 needs Ollama too |
| C loop | — | 07–11 need keys; 07 needs `llama3.2:1b` |
| D reliability | **12 · 13 · 14** | all three run on `_fake_model`, offline and free |
| E security | — | 15–16 need keys |
| F observability | — | 17 runs offline, not yet executed end-to-end |
| G frameworks | **20 · 21** | 18–19 need keys |
| H memory | **24 · 25** | 22–23 need keys |
| I planning | — | 26–28 need keys |
| J multi-agent | — | 29–31 need keys |
| K MCP | **34** | 32–33 run offline; 35 needs a key |
| L evaluation | **36 · 37** | 37's red-team section needs a key |

**Twelve modules are verified end-to-end** — every one that needs no API key. Captured
output and any fixes live in each module's `_details.md`, one canonical home per
[`../../RULES.md`](../../RULES.md).

**Three modules failed their own claims on first run** and were corrected:
[13](D_reliability/13_guardrails_and_budgets_details.md) had a guard that claimed a catch
it never performed; [20](G_frameworks/20_langgraph_checkpoint_and_hitl_details.md) had a
"fork" that silently replayed the recorded answer; and
[21](G_frameworks/21_langgraph_parallel_subgraphs_streaming_details.md) described a
collision as silent that LangGraph 1.2 now raises on. Each is written up as a finding.

## Folder layout

One folder per block. Shared files stay at the phase root because **notebooks cannot be
imported** — only plain `.py` can.

```
12_agents_from_scratch/
├── _providers.py  _tools.py  _fake_model.py   ← shared, importable, phase root
├── A_the_wire/          01 · 02 · 03
├── B_workflows/         04 · 05 · 06
├── C_the_loop/          07 · 08 · 09 · 10 · 11
├── D_reliability/       12 · 13 · 14
├── E_security/          15 · 16
├── F_observability/     17
├── G_frameworks/        18 · 19 · 20 · 21
├── H_memory/            22 · 23 · 24 · 25
├── I_planning/          26 · 27 · 28
├── J_multi_agent/       29 · 30 · 31
├── K_mcp/               32 · 33 · 34 · 35
└── L_evaluation/        36 · 37
```

Each module is a pair, per [`../../CLAUDE.md`](../../CLAUDE.md):
`NN_name.ipynb` + `NN_name_details.md`. The details file carries the status badge, the
theory links, the captured run output and any fixes found while running.

Because modules sit one level down, every module opens with:

```python
import sys; sys.path.insert(0, "..")   # reach the shared files at the phase root
```

## How to run

```bash
conda activate sameerkhan
cd code_practice/12_agents_from_scratch
python _providers.py              # health check — should print 3x OK
python A_the_wire/01_standard_llm_call.py   # module 01 predates the notebook switch
```

**Modules 02 onward are `.ipynb` notebooks.** Open one in VS Code, pick the
`sameerkhan` kernel, and run cells top to bottom. Notebooks keep their outputs in the
file, so a finished run is its own evidence — no separate details file to maintain.

The three `_shared.py` files stay plain Python: notebooks cannot be imported.

**Requires:** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and `ollama serve` running with `ollama pull llama3.2`.

---

## Shared foundation

| File | Status | Purpose |
|---|---|---|
| `_tools.py` | ✅ | Four tiny tools, exported in **both** wire envelopes from one source of truth. Kept identical across modules so the modules are **diffable** — only the machinery changes. |
| `_providers.py` | ✅ | `get_client("openai" \| "anthropic" \| "local")`. Run standalone for a health check. |
| `_fake_model.py` | ✅ | Scripted stand-in with the real client's interface. **Required from module 12** — a real model won't loop, hallucinate a tool name, or attempt an injection *on demand*, so without this the guards are untestable. |

---

## Block A — The wire · `A_the_wire/`

| # | File | The one idea | |
|---|---|---|---|
| 01 | `01_standard_llm_call.py` | Same prompt → OpenAI, Anthropic, local. **Two wire formats**: `choices[0].message.content` (string) vs `content[]` (typed blocks); `finish_reason` vs `stop_reason`. Ends with raw `requests.post` — the SDK is a wrapper, not a capability. | ✅ |
| 02 | `02_multi_turn_by_hand.ipynb` | Carry the list yourself. **The API is stateless — memory *is* the list.** Tokens printed per turn. | ✅ |
| 03 | `03_structured_output.ipynb` | JSON mode, Pydantic, `strict`. **Why free-text parsing is a bug**, and how constrained output deletes a failure class. | 🔧 |

## Block B — Workflows *(deliberately before agents)* · `B_workflows/`

| # | File | The one idea | |
|---|---|---|---|
| 04 | `04_workflow_prompt_chaining.ipynb` | Sequential calls with a **programmatic gate** between steps. | 🔧 |
| 05 | `05_workflow_routing.ipynb` | Classify → dispatch. **Cheap model for easy inputs** — the main cost lever. | 🔧 |
| 06 | `06_workflow_parallel_and_evaluator.ipynb` | Sectioning, voting, and generate → critique → revise. | 🔧 |

*Built before any agent so "most things called agents are workflows" is **felt**, not read.*

## Block C — The loop · `C_the_loop/`

| # | File | The one idea | |
|---|---|---|---|
| 07 | `07_agent_without_sdk_react.ipynb` | ReAct in prose — **you write the regex**. Local Llama buries the call in text and the parser misses it. **FM1, live.** | 🔧 |
| 08 | `08_agent_with_sdk_tool_calling.ipynb` | Native tool calling, both envelopes. `finish_reason` flips; `content` is `None` on a tool turn. The regex from 07 is deleted. | 🔧 |
| 09 | `09_agent_parallel_tool_calls.ipynb` | Several tool calls in **one** turn — return **all** results in a single message, or the model quietly stops parallelising. | 🔧 |
| 10 | `10_rag_as_a_tool.ipynb` | Retrieval as a **tool the model chooses**. Classic vs **agentic RAG**. | 🔧 |
| 11 | `11_agent_cost_and_tokens.ipynb` | Per-turn cost table. **Why an agent costs ~10× a single call.** | 🔧 |

## Block D — Reliability · `D_reliability/`

| # | File | The one idea | |
|---|---|---|---|
| 12 | `12_failure_modes_reproduced.ipynb` | All **five failure modes on demand** via `_fake_model`. | ✅ |
| 13 | `13_guardrails_and_budgets.ipynb` | Iteration cap, duplicate detection, Pydantic arg validation, token budget, abort-and-escalate. | ✅ |
| 14 | `14_audit_log_and_hitl.ipynb` | Append-only JSONL trace + **human gate** on the write tool. SS1/23 traceability, Consumer Duty approval. | ✅ |

## Block E — Security · `E_security/`

| # | File | The one idea | |
|---|---|---|---|
| 15 | `15_prompt_injection_via_tools.ipynb` | A tool returns attacker text — and the agent **obeys it**. The attack, working. | 🔧 |
| 16 | `16_injection_defences.ipynb` | Spotlighting, **capability isolation**, output validation, **dual-LLM (CaMeL)**. | 🔧 |

## Block F — Observability · `F_observability/`

| # | File | The one idea | |
|---|---|---|---|
| 17 | `17_tracing_and_observability.ipynb` | Structured trace per run: intent, tool calls, args, latency, tokens, cost, approvals. **You cannot retrofit an audit trail.** | 🔧 |

## Block G — Frameworks · `G_frameworks/`

| # | File | The one idea | |
|---|---|---|---|
| 18 | `18_langchain_lcel.ipynb` | LCEL, Runnables, `.invoke`/`.batch`/`.stream` — then the same job in three plain lines. | 🔧 |
| 19 | `19_langgraph_minimal_agent.ipynb` | Module 08 as a state machine. **The `while` *is* the conditional edge.** | 🔧 |
| 20 | `20_langgraph_checkpoint_and_hitl.ipynb` | `thread_id`, `interrupt()`, resume via `Command`, **time travel**. | ✅ |
| 21 | `21_langgraph_parallel_subgraphs_streaming.ipynb` | `Send`, subgraphs, the four `stream_mode`s, and the reducer-collision trap. | ✅ |

## Block H — Memory · `H_memory/`

| # | File | The one idea | |
|---|---|---|---|
| 22 | `22_memory_short_term_strategies.ipynb` | Full vs window vs summary, side by side — and **what the model forgets** under each. | 🔧 |
| 23 | `23_memory_entity_keyvalue.ipynb` | Structured entity facts — the cheap tier people skip for vectors. | 🔧 |
| 24 | `24_memory_long_term_vector.ipynb` | Recall into a **new** conversation. Episodic vs semantic. **Per-user namespacing** — a second user proves no leakage. | ✅ |
| 25 | `25_memory_procedural_and_forgetting.ipynb` | Reusable skills, plus **decay and eviction** — a store that only grows degrades. | ✅ |

## Block I — Planning · `I_planning/`

| # | File | The one idea | |
|---|---|---|---|
| 26 | `26_plan_and_execute.ipynb` | Plan upfront vs ReAct interleaved. Inspectable — and stale when step 1 surprises you. | 🔧 |
| 27 | `27_rewoo_parallel_planning.ipynb` | All calls enumerated upfront, run in parallel, one reasoning pass. | 🔧 |
| 28 | `28_reflexion_and_self_refine.ipynb` | Fail → critique → store → retry. Plus the **three stopping criteria**. | 🔧 |

## Block J — Multi-agent · `J_multi_agent/`

| # | File | The one idea | |
|---|---|---|---|
| 29 | `29_multi_agent_pipeline.ipynb` | Typed contracts at each hand-off — then a free-text one that silently misparses. **Lost-in-translation, live.** | 🔧 |
| 30 | `30_multi_agent_supervisor.ipynb` | Dynamic routing, and the 5–10× token cost vs single-agent. | 🔧 |
| 31 | `31_multi_agent_debate_and_handoff.ipynb` | Proposer / critic / judge, and Swarm-style handoff. | 🔧 |

## Block K — MCP · `K_mcp/`

| # | File | The one idea | |
|---|---|---|---|
| 32 | `32_mcp_server_capabilities.ipynb` | Tool, resource, prompt — and **control dynamics**: model-controlled vs application-controlled vs user-controlled. | 🔧 |
| 33 | `33_mcp_client_and_lifecycle.ipynb` | initialize → capability negotiation → operation → shutdown. Runtime discovery. | 🔧 |
| 34 | `34_mcp_transports.ipynb` | `stdio` vs Streamable HTTP — a **real subprocess**, a real handshake over real pipes. Transport as a *security* decision. | ✅ |
| 35 | `35_mcp_in_an_agent_and_security.ipynb` | MCP tools into a LangGraph agent, then **tool poisoning** via a malicious description. | 🔧 |

## Block L — Evaluation · `L_evaluation/`

| # | File | The one idea | |
|---|---|---|---|
| 36 | `36_agent_eval_harness.ipynb` | Score a **trajectory**, not an answer. Runs against modules 08 and 26. | ✅ |
| 37 | `37_eval_variance_and_redteam.ipynb` | ×5 seeds — **a bimodal 80% beside a uniform 80%**. Plus an injection red-team corpus. | ✅ |

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

> ⚠️ **MCP theory needs a fix first.** The **2026-07-28 spec** made MCP stateless and introduced **MRTR**, deprecating server-initiated `sampling`, `elicitation` and `roots`. [`08_mcp_protocol_deep.md`](../../8.agents/08_mcp_protocol_deep.md) §3 still presents them as current. **That revision could not be verified against a primary source here**, so modules 32–35 are built only on JSON-RPC 2.0 and the stable tools / resources / prompts core, and module 34 was renamed from `..._and_mrtr`. Confirm against the spec before adding MRTR content.

---

## Effort

**Full ladder ≈ 32 hours.** Blocks are independent once the three shared files exist — build in any order, pause between blocks.

**Minimum viable subset ≈ 6 hours:** 01, 07, 08, 12, 13, 14 — the wire, both loop forms, the reliability trio.

## Verification

Every module must **run top-to-bottom**, **print its own lesson** (visible in stdout, not merely implemented), and **stay under ~90 lines of code** — markdown cells and comments are free, the code is what must stay small. A notebook is done when its **outputs are saved in the file**.

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

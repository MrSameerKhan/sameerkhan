# 8. Agents

Scope: LLM agents — ReAct, tool use, frameworks (LangGraph / CrewAI / AutoGen / Swarm), memory, multi-agent, MCP, evaluation, reliability. **Tier: 2 (Theory).** The most comprehensive part of the LLM stack — 10 files cover the full agent lifecycle.

---

## Reading Order

| If you're learning... | Read in order |
|----------------------|---------------|
| Agent fundamentals | `01_agents` → `01b_agents_end_to_end` |
| Production reliability | `02_agent_reliability_patterns` (retries, loop detection, structured outputs, HITL, audit) |
| Frameworks | `03_langchain_primer` → `04_langgraph_deep` (the modern default) → `07_multi_agent_orchestration` |
| Memory | `05_agent_memory` (working / short / long-term split with subtypes) |
| Planning | `06_planner_executor_patterns` (ReAct / Plan&Execute / ReWoo / LATS / Reflexion) |
| Protocol layer | `08_mcp_protocol_deep` (Model Context Protocol) |
| Evaluation | `09_agent_evaluation` (success / tool-quality / trajectory / cost / safety) |

---

## Folder TOC

| File | Owns |
|------|------|
| `01_agents.md` | Agent fundamentals — ReAct, tool calling, MCP overview |
| `01b_agents_end_to_end.md` | Worked example — agent loop with tool calls |
| `02_agent_reliability_patterns.md` | SSOT: production hardening (retries, loop detection, structured outputs, HITL, audit log) |
| `03_langchain_primer.md` | LCEL, Runnables, output parsers, when to use vs not |
| `04_langgraph_deep.md` | SSOT: state machines, checkpointing, HITL interrupts, parallel branches, subgraphs |
| `05_agent_memory.md` | SSOT: working / short / long-term (episodic / semantic / procedural) memory architectures |
| `06_planner_executor_patterns.md` | SSOT: ReAct / Plan&Execute / ReWoo / LATS / Reflexion / Self-Refine / STORM / ADaPT |
| `07_multi_agent_orchestration.md` | SSOT: CrewAI / AutoGen / OpenAI Swarm / smolagents / Pydantic AI / LangGraph multi-agent |
| `08_mcp_protocol_deep.md` | SSOT: MCP architecture, capabilities (tools/resources/prompts/sampling/roots), transports |
| `09_agent_evaluation.md` | SSOT: success / tool-call / trajectory / cost / reliability / safety metrics + benchmarks |

---

## SSOT Topics Owned Here

- Agent reliability patterns → `02_agent_reliability_patterns.md`
- LangGraph deep dive → `04_langgraph_deep.md`
- Agent memory architectures → `05_agent_memory.md`
- Planner-executor patterns → `06_planner_executor_patterns.md`
- Multi-agent orchestration → `07_multi_agent_orchestration.md`
- MCP protocol → `08_mcp_protocol_deep.md`
- Agent evaluation → `09_agent_evaluation.md`

---

## Connections

- **LLM core** (prompting, fine-tuning): `../6.llms/`
- **RAG** (often the retrieval tool used by agents): `../7.rag/`
- **Tool authorization patterns** (security depth): `../11.system_design/09_tool_authorization_patterns.md`
- **LLM evaluation systems** (incl. agent eval at system level): `../11.system_design/11_llm_evaluation_systems.md`
- **LLM observability** (LangFuse / LangSmith / Phoenix): `../10.mlops/11_llm_observability_tools.md`
- **Structured outputs** (Pydantic + Instructor): `../4.nlp/04_applications/03_information_extraction.md`
- **Constrained decoding**: `../5.transformers/models/12_constrained_decoding.md`
- **Indirect prompt injection** (the #1 agent threat): `../7.rag/03_indirect_prompt_injection.md`
- **Agent system design** (capacity, multi-tenant, scaling): `../11.system_design/05_llm_agent_system_design.md`

---

## Practice

- Agents (10 sessions, all docs complete) → `code_practice/06_agents/`
- Each session pairs 1:1 with a file in this folder — see `code_practice/INDEX.md` § Phase 6.

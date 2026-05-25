# Multi-Agent Orchestration

> Why this matters: A single agent with too many tools and too long a context loses focus. Multi-agent systems decompose by **role** (specialist agents) or by **phase** (pipeline agents) — often producing better results than a monolithic agent on complex tasks.

---

## Quick Reference — Frameworks

| Framework | Style | Best for | License |
|-----------|-------|----------|---------|
| LangGraph (multi-agent mode) | Explicit state graph; agents as nodes/subgraphs | Production; full control | MIT |
| CrewAI | Role-based: each agent has a `role` + `goal` + `backstory` + `tools` | Sequential / hierarchical task decomposition | MIT |
| AutoGen (Microsoft) | Conversation-based: agents talk to each other in turn | Research; flexible group dynamics | MIT (Apache for v0.4) |
| OpenAI Swarm | Handoff primitive between agents | Lightweight; OpenAI-native | MIT |
| smolagents (HuggingFace) | Code-as-action: agents write Python instead of JSON tool calls | Code-execution tasks; minimal framework | Apache-2 |
| Pydantic AI | Type-safe, dependency-injected agents | Structured I/O agents | MIT |
| MetaGPT | Software-engineering simulation (PM → Architect → Engineer → ...) | Code generation pipelines | — |
| CAMEL | Role-playing dialogue between agents | Research / autonomous brainstorming | Apache-2 |

---

## 1. When to Reach for Multi-Agent

| Signal | Single agent | Multi-agent |
|--------|-------------|------------|
| 1-3 tools | Sufficient | overkill |
| 10+ tools | context bloat | Split by role |
| Sequential phases (research → write → review) | possible | Natural fit |
| Different expertise per step (legal vs technical) | hard | Specialist agents |
| Adversarial / critique (debate / red-team) | hard | Peer agents |
| Latency critical | Sufficient | Adds overhead |
| Cost critical | Sufficient | Adds tokens (each agent re-loads context) |

**Rule of thumb:** stay single-agent until you can articulate a clear reason for multi-agent. "It feels more sophisticated" is not a reason.

---

## 2. Coordination Patterns

```
1. SUPERVISOR / WORKER (hierarchical)
   Supervisor receives task → delegates sub-tasks → aggregates results
   ───────────────────────────────────────────────
   Pros: clear flow, easy to debug
   Cons: supervisor becomes bottleneck; serial unless explicitly parallel

2. PEER-TO-PEER (collaborative)
   Agents talk to each other directly; no fixed coordinator
   ───────────────────────────────────────────────
   Pros: emergent collaboration; mirrors real teams
   Cons: convergence issues; can loop forever; hard to evaluate

3. PIPELINE (sequential)
   Agent A → Agent B → Agent C → output
   ───────────────────────────────────────────────
   Pros: simple, deterministic
   Cons: errors compound; later agents can't influence earlier

4. BLACKBOARD (shared state)
   All agents read/write to a shared store; pick up work when triggered
   ───────────────────────────────────────────────
   Pros: flexible, decoupled
   Cons: synchronization complexity; race conditions

5. DEBATE / CRITIQUE (adversarial)
   Two+ agents argue different positions; a judge picks winner
   ───────────────────────────────────────────────
   Pros: surfaces blind spots; better factual accuracy
   Cons: 2-5× cost; needs good judge
```

Production agents most often use **Supervisor/Worker** (for clarity) or **Pipeline** (for predictability). Peer-to-peer is research-favored but hard to ship.

---

## 3. CrewAI — Role-Based

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Research Analyst",
    goal="Find authoritative information on {topic}.",
    backstory="A meticulous researcher who cites sources.",
    tools=[search_tool, browse_tool],
    llm=llm,
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, well-structured content based on research notes.",
    backstory="Award-winning technical writer for engineering audiences.",
    tools=[],
    llm=llm,
)

research_task = Task(
    description="Research the latest developments in {topic}. Cite 5+ sources.",
    expected_output="A research report in bullet points with sources.",
    agent=researcher,
)

write_task = Task(
    description="Using the research, write a 1000-word article on {topic}.",
    expected_output="A well-structured article.",
    agent=writer,
    context=[research_task],   # receives output of research_task
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
)

result = crew.kickoff(inputs={"topic": "vLLM speculative decoding"})
```

**Strengths:** very readable; clear role-task separation; quick prototyping. **Weaknesses:** less control vs LangGraph; debugging requires reading framework internals; recent versions added LangGraph-style flows but it's still simpler.

---

## 4. AutoGen — Conversation-Based

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

coder   = AssistantAgent(name="Coder",  llm_config={"model": "gpt-4o"})
critic  = AssistantAgent(name="Critic", llm_config={"model": "gpt-4o"},
                         system_message="Critique code for bugs and style.")
executor = UserProxyAgent(name="Executor", code_execution_config={"work_dir": "/tmp/coding"})

group_chat = GroupChat(agents=[coder, critic, executor], messages=[], max_round=10)
manager    = GroupChatManager(groupchat=group_chat, llm_config={"model": "gpt-4o"})

executor.initiate_chat(manager, message="Write a Python function that computes Fibonacci numbers, then test it.")
```

Agents take turns based on the manager's choice (which agent should speak next given the current state). Conversational dynamics are emergent.

**Strengths:** powerful for research and code-heavy tasks; widely cited in 2023-24 multi-agent papers; v0.4 (2024) is a major redesign with `core` + `agentchat` + ext packages. **Weaknesses:** high token cost; debugging multi-turn group dynamics is hard; production deployments rare.

---

## 5. OpenAI Swarm — Handoff Primitive

Swarm (released as a "cookbook" reference, not full framework) introduces a clean **handoff** pattern:

```python
from swarm import Swarm, Agent

def transfer_to_billing():
    return billing_agent

def transfer_to_tech_support():
    return tech_agent

triage_agent = Agent(
    name="Triage",
    instructions="Route the user to the correct department.",
    functions=[transfer_to_billing, transfer_to_tech_support],
)

billing_agent    = Agent(name="Billing",     instructions="Help with billing questions.")
tech_agent       = Agent(name="TechSupport", instructions="Help with technical issues.")

client = Swarm()
response = client.run(agent=triage_agent, messages=[{"role": "user", "content": "I can't log in"}])
# Triage runs → calls transfer_to_tech_support → tech_agent takes over
```

A handoff is just a tool call that returns a different Agent instance. Conversation continues with the new agent. Clean, minimal, OpenAI-tuned.

**Strengths:** smallest framework footprint; ideal for customer-support style routing. **Weaknesses:** reference implementation, not production-supported by OpenAI; less ecosystem.

---

## 6. smolagents — Code as Action

HuggingFace's smolagents (2024) makes a different bet: instead of emitting JSON tool calls, the agent emits **Python code** that calls tools as functions.

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

model = HfApiModel("meta-llama/Llama-3.1-8B-Instruct")
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
agent.run("What's the population of the largest city in Japan?")
```

Internally the agent generates Python like:

```python
results = web_search("largest city in Japan population")
print(results)
# (then thinks about output)
final_answer(results[0])
```

**Strengths:** one universal "tool" (code execution) handles everything; agent can compose tools naturally with Python; ~30% fewer LLM calls on benchmarks. **Weaknesses:** requires sandboxed code execution (security risk if not sandboxed); only works well with code-capable LLMs.

---

## 7. Pydantic AI — Type-Safe

```python
from pydantic_ai import Agent

class WeatherDeps(BaseModel):
    api_key: str

class WeatherReport(BaseModel):
    location: str
    temp_c: float
    conditions: str

agent = Agent(
    "gpt-4o-mini",
    deps_type=WeatherDeps,
    result_type=WeatherReport,
    system_prompt="You provide weather reports.",
)

@agent.tool
async def get_weather(ctx: RunContext[WeatherDeps], city: str) -> dict:
    # ctx.deps.api_key is available
    return await fetch_weather(city, ctx.deps.api_key)

result = await agent.run("Weather in Tokyo?", deps=WeatherDeps(api_key="..."))
# result.data is a typed WeatherReport instance
```

**Strengths:** type-safety end-to-end; dependency injection (clean separation of agent logic and config); great for production teams that already use Pydantic. **Weaknesses:** smaller ecosystem; newer (less battle-tested).

---

## 8. LangGraph for Multi-Agent (Most Production-Common)

LangGraph isn't a "multi-agent framework" per se but trivially expresses any pattern:

```python
# Supervisor pattern in LangGraph
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
    next: str  # which worker to call

def supervisor(state):
    decision = llm.invoke([
        *state["messages"], *
        ("Given messages: state['messages']), "
         "decide next worker: 'researcher', 'writer', or 'FINISH'")
    ])
    return {"next": parse_decision(decision)}

def researcher_agent(state):
    output = research_subgraph.invoke(state)
    return {"messages": [AIMessage(f"[Researcher]: {output}")]}

def writer_agent(state):
    output = writer_subgraph.invoke(state)
    return {"messages": [AIMessage(f"[Writer]: {output}")]}

def route(state):
    return state["next"] if state["next"] != "FINISH" else END

builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_agent)
builder.add_node("writer", writer_agent)
builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route,
    {"researcher": "researcher", "writer": "writer", END: END})
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
graph = builder.compile()
```

Each worker is a LangGraph subgraph. Supervisor decides routing. State is shared (or namespaced per worker).

**Strengths:** full control; same observability + checkpointing as single-agent LangGraph; can implement any multi-agent pattern. **Weaknesses:** more code than CrewAI for simple cases.

---

## 9. Communication Patterns Between Agents

| Pattern | Mechanism |
|---------|----------|
| Shared state | All agents read/write to the same state object (LangGraph default) |
| Message passing | Agents send messages to each other's queue (AutoGen) |
| Blackboard | Centralized store; agents publish updates; subscribers react |
| Direct call (handoff) | Agent A's output IS Agent B's input (Swarm, Pipeline) |
| Broadcast | One agent posts; all peers receive (rare in practice) |

For production, prefer **shared state with namespace per agent** — simpler than message passing, easier to debug.

---

## 10. Role Design — What Makes a Good Specialist

A specialist agent should have:

1. **A narrow, clear goal** ("Extract entities" not "Help the user")
2. **Limited tools** (3-5 max — too many = same context-bloat problem as single agent)
3. **A specific system prompt** that establishes voice + format
4. **A defined output schema** (Pydantic — what does this agent produce?)
5. **A defined input schema** (what does this agent consume?)
6. **No knowledge of orchestration** — should not "know" it's part of a multi-agent system

The orchestration logic lives in the coordinator (LangGraph supervisor / CrewAI process), not in the agents themselves.

---

## 11. Anti-Patterns

| Anti-pattern | Why it fails |
|-------------|-------------|
| 10+ agents with overlapping responsibilities | Convergence problems; same task done by multiple agents |
| Agents that "discuss" without converging | Token cost without progress; need explicit termination signal |
| Persona-heavy backstories ("you are a senior expert with 30 years...") | Doesn't actually change model behavior much; pure cargo cult |
| No success criteria | Can't tell when to stop; can't evaluate quality |
| One mega-state with everything | Agents see irrelevant info; back to context bloat |
| Reusing single-agent prompts as multi-agent prompts | Roles overlap; agents step on each other |

**The "Sequential Refinement" trap:** 3 agents in series, each "refining" the previous output, often produces worse results than 1 good agent because each refinement step introduces drift.

---

## 12. Cost & Latency

Multi-agent costs add up fast:

```
Single-agent task (50 tokens prompt, 500 tokens response, 5 turns):
→ ~2,750 tokens total

Multi-agent (3 agents, each gets full context, 5 supervisor cycles):
  Supervisor = 3 workers × 5 = 20 LLM calls
  Each call: ~2,000 token context
  Total: ~40,000 tokens (~14× more)
```

**Mitigations:** Namespace state so each agent only sees what it needs · **Cache** the system prompt (Anthropic prompt caching, OpenAI prefix caching) · **Smaller models for sub-agents** (only the supervisor needs frontier-class) · **Parallelize** independent workers via LangGraph `Send`.

---

## 13. Evaluation

Multi-agent eval is harder than single-agent. See `09_agent_evaluation.md`. Key extras:
- **Routing accuracy** (did supervisor pick the right worker?)
- **Communication efficiency** (token cost per task vs baseline)
- **Per-agent success rate** (which agent is the weak link?)
- **Convergence rate** (how often does the team finish without hitting the iteration cap?)

---

## 14. Gotchas

**Context explosion.** Each agent typically loads full conversation. With 4 agents × 10-turn conversation, you're sending 40 turns total. Namespace context per agent.

**Lost in translation between agents.** Agent A's output as Agent B's input — formatting mismatches cause silent failures. Use Pydantic schemas at every handoff.

**Supervisor too dumb.** If the supervisor LLM can't reliably route, the whole system breaks. Supervisor often needs a stronger model than workers.

**No clear termination.** AutoGen-style chats can loop indefinitely. Always set `max_round` / `recursion_limit`.

**Cargo-cult roles.** "Senior AI Architect" as a role doesn't measurably improve outputs. Pick roles based on real decomposition, not personas.

**Hidden retries.** Some frameworks (CrewAI, AutoGen) silently retry failed agents — masks real failures during debugging. Check logs.

---

## 15. Interview Q&A

**Q: When does multi-agent beat single-agent?**

Three clear cases: (1) **Tool overload** — a single agent with 10+ tools loses focus; splitting by tool-group gives each agent a manageable surface. (2) **Distinct expertise / personas** — legal review + technical implementation are different tasks with different evaluation criteria; one agent per role beats one agent juggling both. (3) **Phased workflows** — research → outline → draft → review, where each phase has different success criteria. **When it loses:** short tasks (overhead dominates), latency-critical (each agent adds round-trips), simple workflows (one agent with good tools is enough). Default to single-agent unless you can articulate WHY multi-agent helps.

**Q: How would you choose between CrewAI, AutoGen, LangGraph, and Swarm?**

**LangGraph** for production agents — full control, explicit state, checkpointing, HITL, debuggability. **CrewAI** for fast prototyping where roles map cleanly to your task. **AutoGen** for research/exploration — flexible conversation dynamics, but harder to productionize. **Swarm** for OpenAI-only customer-routing patterns (clean handoffs). **Don't pick the framework first** — design the orchestration pattern (supervisor / pipeline / peer), then choose the framework that expresses it cleanly.

**Q: What's the biggest production failure mode of multi-agent systems?**

**Lost-in-translation between agents.** Agent A says "Found 3 candidates: X, Y, Z" (free text); Agent B parses this as input. Tomorrow Agent A's output format changes to "Three candidates: X / Y / Z" → Agent B silently misparses → downstream agents act on wrong data → final output is wrong but plausible. Fix: **structured contracts at every hand-off** — Pydantic schemas, function-call args, JSON modes. The same discipline as service-to-service APIs in microservices. Free-text inter-agent communication looks elegant but is the #1 cause of silent failures.

**Q: How do you control cost in a multi-agent system?**

Four levers: (1) **Right-size models** — supervisor needs strong (GPT-4-class), workers can be cheaper (Haiku / mini). (2) **Namespace context** — each worker sees only what it needs, not the full conversation. (3) **Prompt caching** — Anthropic / OpenAI cache the system prompt; tokens after the first call are ~5-10% the cost. (4) **Hard iteration cap** with cost budget — track cumulative tokens; abort and escalate if a task exceeds budget. Multi-agent without these can be 10-50× the cost of single-agent for the same task; with them, 2-5×.

**Q: How is "code as action" (smolagents) different from JSON tool-calling?**

In JSON tool-calling, the LLM outputs `{"name": "search", "args": {"q": "..."}}` and the framework calls it. Each tool is a discrete call. In code-as-action, the LLM writes Python code that invokes tools as functions. Benefits: (1) **Composition** — the model can write `z = search(q); summarize(z[0].text)` in one step instead of two LLM calls. (2) **Control flow** — for-loops, conditionals, error handling all become "free." (3) **~30% fewer LLM calls** on benchmarks. Costs: (1) requires sandboxed execution (security risk); (2) only works well with code-capable LLMs (Llama-3 / GPT-4 / Claude 3.5+); (3) makes outputs harder to validate vs structured JSON. Best for tasks where compositional tool use is natural (data analysis, research with many web calls).

---

## 16. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| Agent fundamentals | `01_agents.md` | Conceptual background |
| LangGraph deep | `04_langgraph_deep.md` | The state-machine substrate |
| Planner-executor | `06_planner_executor_patterns.md` | Alternative: decompose by phase, not role |
| Agent memory | `05_agent_memory.md` | Shared blackboard / per-agent memory |
| Agent reliability | `02_agent_reliability_patterns.md` | Loop detection across agents |
| Tool authorization | `../11.system_design/09_tool_authorization_patterns.md` | Per-agent capability isolation |
| Agent evaluation | `09_agent_evaluation.md` | Multi-agent-specific metrics |
| Code practice | `code_practice/06_agents/07_multi_agent/` | Hands-on |

---

## Key Takeaway

Multi-agent helps when you can articulate **why** — too many tools, distinct expertise, or clear phases. Otherwise it adds cost without benefit. **Production stack in 2025:** LangGraph supervisor over namespaced subgraph-workers, Pydantic-typed contracts between agents, cached system prompts, smaller models for sub-agents, hard iteration cap. **Don't pick a framework first** — design the pattern (supervisor / pipeline / peer / blackboard), then pick the framework that expresses it cleanly. Free-text inter-agent communication is the #1 silent failure mode; use structured schemas. A good multi-agent team: supervisor + researcher + analyst + composer.

---

## Code Practice — Wired by Phase 6

- `code_practice/06_agents/07_multi_agent/` — supervisor + researcher + analyst + composer

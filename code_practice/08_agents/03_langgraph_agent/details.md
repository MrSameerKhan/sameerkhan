# Session 3 — LangGraph Production Workflow Agent
Status: `✅ Run`

Theory: [../../../8.agents/04_langgraph_deep.md](../../../8.agents/04_langgraph_deep.md) · [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md)

---

## Use Case

A prototype agent (sessions 01-02) crashes on edge cases, loses state between turns, and has no way to pause for human approval. LangGraph adds: typed state that persists across turns, checkpointing for crash recovery, and HITL interrupts.

---

## Why Sessions 01→02→03 Form a Progression

| Capability | Session 01 (ReAct) | Session 02 (Tool Calling) | Session 03 (LangGraph) |
|-----------|-------------------|--------------------------|----------------------|
| Tool execution | Manual loop | OpenAI-native | ToolNode (built-in) |
| State persistence | Messages list | Messages list | MemorySaver (disk/DB) |
| Multi-turn memory | Re-send all messages | Re-send all messages | Checkpointed by thread_id |
| HITL | Not supported | Not supported | `interrupt()` built-in |
| Crash recovery | Lost | Lost | Resume from last checkpoint |
| Streaming | Manual | Manual | `app.stream()` |

---

## Graph Architecture

```
[START]
    │
  llm_node ──── (has tool_calls?) ──── tools_condition
    │                                       │
    │◄──────────── ToolNode ◄───────── "tools"
    │
    └── (no tool_calls) ──── [END]
```

**State:** `Annotated[list[BaseMessage], operator.add]` — messages are appended, not overwritten. Every node adds to the history.

**Checkpointing:** `MemorySaver` stores the full state keyed by `thread_id`. Same thread_id across calls = continuous conversation.

---

## Multi-Turn Conversation (how state persists)

```python
config = {"configurable": {"thread_id": "thread-003"}}

# Turn 1: context stored in MemorySaver
app.invoke({"messages": [HumanMessage("I want to buy a £350k property")]}, config=config)

# Turn 2: previous messages loaded from checkpoint → agent remembers
app.invoke({"messages": [HumanMessage("I have £40k deposit. Is that enough?")]}, config=config)

# Turn 3: still has full context
app.invoke({"messages": [HumanMessage("What's the monthly payment?")]}, config=config)
```

Without LangGraph: you'd have to re-send the entire conversation history manually every turn.

---

## HITL Pattern (how to add it)

```python
from langgraph.types import interrupt, Command

# In a node:
def human_review(state):
    decision = interrupt({"message": "Approve this action?", "data": state["result"]})
    return {"approved": decision == "approved"}

# Compile with checkpointer (required for interrupt):
app = graph.compile(checkpointer=MemorySaver())

# Run until interrupt:
result = app.invoke(initial_state, config={"configurable": {"thread_id": "t1"}})
# result["__interrupt__"] contains the interrupt payload

# Resume:
result = app.invoke(Command(resume="approved"), config={"configurable": {"thread_id": "t1"}})
```

---

## Actual Output (Windows, gpt-4o-mini, 2026-06-25)

- Thread 001 (LTV): 95% first-time buyers ✓
- Thread 002 (payment calc): £1,236.61/month ✓
- Thread 003 (multi-turn): deposit eligibility remembered across 3 turns ✓ — agent knew £310k loan on turn 3 from turn 2 context
- Streaming: tool result repeated 3× before final answer (streaming artifact — each token chunk includes tool output prefix)

**Fix applied:** `POLICY_ALIASES` added to `tools.py` — "early repayment" now maps to "erc" key (same fix as S01).

---

## Expected Output

```
Thread: thread-003

User: I want to buy a £350,000 property as a first-time buyer.
Agent: As a first-time buyer, you can borrow up to 95% LTV under Help to Buy.
       On a £350,000 property that means a minimum deposit of £17,500 (5%).

User: I can put down a £40,000 deposit. Is that enough?
Agent: Yes — your £40,000 deposit gives an LTV of 88.6%, well within the 95% limit.
       That's actually better than the minimum required.

User: What would the monthly payment be at the 5-year fixed rate over 25 years?
Agent: Loan amount: £310,000 | Rate: 4.61% | Term: 25 years (300 months)
       Monthly payment: £1,733.41
       Total paid: £520,023 | Total interest: £210,023
```

---

## File Structure

```
03_langgraph_agent/
├── tools.py   — @tool decorated functions (search_policy, calculate_mortgage_payment, check_ltv)
├── graph.py   — StateGraph definition, llm_node, routing, MemorySaver compile
└── run.py     — multi-turn demo + streaming demo
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/08_agents/03_langgraph_agent
python run.py
```

Cost: ~$0.05 per run. First run downloads `langchain-openai` model wrappers.

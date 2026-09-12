# Module 19 — Module 08 as a State Machine
Status: `🔧 Code-built`

Theory: [../../../8.agents/04_langgraph_deep.md](../../../8.agents/04_langgraph_deep.md) §2-3 (minimal agent, the standard tool pattern) · §13 (the reducer-mismatch gotcha)

---

## Use Case

The third rung of the ladder's central equivalence: modules 07, 08 and 19 answer the same question with the same tools and land on the same number. Only the machinery differs on screen.

The claim being tested: **the `while` loop IS the conditional edge — LangGraph makes the loop inspectable, it does not add capability.**

---

## The Mechanism

```
module 08                          module 19
---------                          ---------
while True:                        b.add_edge("tools", "agent")
  r = call_model(messages)         node "agent"
  if finish != "tool_calls":       route() -> add_conditional_edges
      break
  run_tools(); append()            node "tools"
messages += x                      Annotated[list, operator.add]  (the reducer)
```

---

## Key Implementation Details

**No `langchain_openai` and no LangChain message classes.** Nodes are plain functions
holding the raw SDK dicts from module 08, so the diff against 08 is genuinely minimal.

**`draw_mermaid()`, not `draw_ascii()`.** The ASCII renderer needs `grandalf`, which is
not installed; mermaid is pure Python and renders natively in VS Code.

**The reducer comment is load-bearing.** Without `Annotated[list, operator.add]` the next
node's value *replaces* the list and history is lost silently. That is the most common
LangGraph bug and it is called out at the point it would bite.

---

## Fixes Applied (during run)

| Found | Fix |
|---|---|
| `draw_ascii()` raised `ImportError: Install grandalf to draw graphs` | Switched to `draw_mermaid()`, which needs no extra package |

---

## Actual Output

*Not yet run end-to-end (needs an API key); graph construction, compile, invoke and stream were smoke-tested offline and pass.*

---

## How to Run

Open `19_langgraph_minimal_agent.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY` only.

---

## Next

`20_langgraph_checkpoint_and_hitl.ipynb`

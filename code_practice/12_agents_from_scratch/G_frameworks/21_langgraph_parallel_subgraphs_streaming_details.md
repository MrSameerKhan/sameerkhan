# Module 21 — Send, Subgraphs, Streaming and the Reducer Trap
Status: `✅ Run (no API key)`

Theory: [../../../8.agents/04_langgraph_deep.md](../../../8.agents/04_langgraph_deep.md) §7 (Send), §8 (subgraphs), §9 (stream modes), §13 (the collision gotcha)

---

## Use Case

Four features that only make sense once the loop is a graph, plus the trap that eats parallel writes. `Send` is where module 06's fixed sectioning becomes runtime-decided orchestrator-workers.

The claim being tested: **any state key that can be written by more than one parallel branch needs a reducer.**

---

## The Mechanism

```
Send("worker", {...}) x N   -> N is computed FROM STATE at runtime
subgraph                    -> own state schema, nothing leaks implicitly
stream_mode                 -> values | updates | messages | debug
collision                   -> two branches, one key, no reducer
```

---

## Key Implementation Details

**`Send` returns a list built from state**, so the worker count is per-input. Let a
model produce that list and you have orchestrator-workers — the workflow/agent boundary.

**The subgraph maps both directions by hand.** That isolation is the feature: a worker
cannot accidentally read the orchestrator's context.

---

## Fixes Applied (during run)

| Found | Fix |
|---|---|
| The module claimed the reducer collision fails **silently** with last-write-wins, which is what `8.agents/04_langgraph_deep.md` §13 still says. **LangGraph 1.2.1 raises `InvalidUpdateError` and refuses to run.** | Rewrote the section to catch and display the real error, then fix it with `Annotated[int, operator.add]`. Both behaviours are now described, since older builds were silent. |
| The streaming cell reused the *broken* graph and inherited the same error. | Pointed it at the fixed graph. |

**The theory file is now stale on this point** and should be updated to say the collision
raises in 1.2+.

---

## Actual Output

```
InvalidUpdateError, as designed:
  At key 'count': Can receive only one value per step. Use an Annotated key ...
-> the graph REFUSED to run rather than lose a write. Good default.

with a reducer on count -> findings=3  count=3  CORRECT
```

---

## How to Run

Open `21_langgraph_parallel_subgraphs_streaming.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`../H_memory/22_memory_short_term_strategies.ipynb`

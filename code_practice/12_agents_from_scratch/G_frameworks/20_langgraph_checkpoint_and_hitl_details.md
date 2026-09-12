# Module 20 — Checkpointing, Interrupts and Time Travel
Status: `✅ Run (no API key — runs on `_fake_model`)`

Theory: [../../../8.agents/04_langgraph_deep.md](../../../8.agents/04_langgraph_deep.md) §4 (checkpointing, time travel) · §5 (HITL, both interrupt patterns)

---

## Use Case

The only honest argument for adding LangGraph. Module 14 built a human gate by hand in about forty lines and it worked — but it held state in a Python dict, so a restart lost the case.

The claim being tested: **checkpointing, interrupts and per-step history are genuinely hard to hand-roll, and they are what the framework is for.**

---

## The Mechanism

```
compile(checkpointer=MemorySaver())        <- HITL REQUIRES a checkpointer
cfg = {"configurable": {"thread_id": "case-1002"}}

invoke(...)            -> runs until interrupt(), returns __interrupt__ payload
                          process may now EXIT; state is in the checkpointer
invoke(Command(resume="approve"), cfg)  -> continues from the pause
get_state_history(cfg) -> every step, with what was pending at each
```

---

## Key Implementation Details

**The gate keys off `WRITE_TOOLS`**, same as module 14, so the two are directly
comparable — the difference is durability, not policy.

**The agent node derives its script position from state**, not from a module-level
counter. See the fixes below for why.

---

## Fixes Applied (during run)

| # | Found | Fix |
|---|---|---|
| 1 | The first draft "forked" an answered interrupt on the **same thread** with `Command(resume="deny")` and got **approve** back, with no error. Re-resuming a resolved interrupt **replays the recorded answer**; it does not branch. | Take the other branch on a **fresh `thread_id`**. `get_state_history` is used for inspection, which is what it reliably provides. |
| 2 | A module-level `FakeModel` with its own `self.calls` counter made replay meaningless — an old checkpoint resumed mid-script. | The agent node derives its script index from `state["audit"]`. Anything stateful **outside** graph state does not replay correctly. |

Both were found by running the notebook, not by reading the docs.

---

## Actual Output

```
paused: True
payload: {'tool': 'confirm_decision', 'args': {'reference': 'ERC-WAIVER-1002'}, ...}
graph is parked at node: ('tools',)

approved thread audit: [... 'gate:confirm_decision:approve', 'ran:confirm_decision', 'llm:stop']
denied   thread audit: [... 'gate:confirm_decision:deny', 'llm:stop']
committed? approved=True  denied=False

7 checkpoints recorded
```

Harmless msgpack warnings appear about `_fake_model` dataclasses — the checkpointer is
serialising types it does not know. In production you checkpoint plain dicts.

---

## How to Run

Open `20_langgraph_checkpoint_and_hitl.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`21_langgraph_parallel_subgraphs_streaming.ipynb`

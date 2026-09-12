# Module 30 — Supervisor / Worker, and What It Costs
Status: `🔧 Code-built`

Theory: [../../../8.agents/07_multi_agent_orchestration.md](../../../8.agents/07_multi_agent_orchestration.md) §2 (coordination patterns) · §12 (cost and latency) · §10 (role design)

---

## Use Case

The most common production multi-agent shape, measured against a single-agent baseline on the identical task.

The claim being tested: **a supervisor costs 5-10x a single agent, and you should be able to say what the routing bought.**

---

## The Mechanism

```
BASELINE   one agent, both tools, module-08 loop
MULTI      supervisor: ONE constrained routing call per cycle
           workers: narrow goal, SMALL tool list, own system prompt
           findings accumulate in the supervisor prompt -> input grows per cycle
```

---

## Key Implementation Details

**The baseline runs first**, so the multiplier is a measurement rather than a claim.

**The supervisor's route is a constrained `Literal`**, so it cannot name a worker with no dispatch entry — module 05's routing lesson one level up.

**`tool_loop` is shared** by the baseline and every worker, so the only difference measured is coordination overhead.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `30_multi_agent_supervisor.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. Roughly 10-14 calls.

---

## Next

`31_multi_agent_debate_and_handoff.ipynb`

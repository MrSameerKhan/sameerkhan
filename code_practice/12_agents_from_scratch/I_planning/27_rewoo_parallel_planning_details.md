# Module 27 — ReWOO: Reasoning Without Observation
Status: `🔧 Code-built`

Theory: [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md) §3

---

## Use Case

Enumerate every tool call up front, run the independent ones in parallel, reason once over all results.

The claim being tested: **ReWOO costs two LLM calls regardless of how many tools run, and buys parallelism by giving up adaptivity entirely.**

---

## The Mechanism

```
PLAN    1 call  -> Blueprint(evidence=[Ev(id, tool, arg-with-#E1-refs)])
EXECUTE 0 calls -> dependency waves, each wave in a ThreadPoolExecutor
SOLVE   1 call  -> one reasoning pass over all evidence

ReAct = N+1 LLM calls, serial. ReWOO = 2, parallel tools.
```

---

## Key Implementation Details

**Dependencies are parsed from `#E1` references** and executed in waves, so latency is the slowest tool rather than the sum.

**Tool results never pass through an LLM until SOLVE**, so they are not re-sent N times — the structural saving over ReAct.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `27_rewoo_parallel_planning.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. Exactly two calls.

---

## Next

`28_reflexion_and_self_refine.ipynb`

# Module 26 — Plan-and-Execute
Status: `🔧 Code-built`

Theory: [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md) §2

---

## Use Case

Decide the whole plan up front and you get an inspectable artefact — and something stale the moment step 1 surprises you.

The claim being tested: **the trade is adaptivity for inspectability.**

---

## The Mechanism

```
PHASE 1 PLAN     one constrained call -> Plan(steps=[Step(tool, arg, why), ...])
                 log it, cost it, gate it BEFORE anything runs
PHASE 2 EXECUTE  substitute #1, #2 placeholders; make NO decisions
break case       step 1 returns 'No policy matched' and step 2 multiplies by it
```

---

## Key Implementation Details

**The plan is a Pydantic object**, so it can be diffed against what actually ran and approved by a human before any side effect.

**The executor makes no decisions**, which is exactly why it can be a cheaper model — the module's practical payoff.

**The break case is deliberate and hand-written** rather than hoped for, so the staleness is demonstrated every run.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `26_plan_and_execute.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. Two calls plus tools.

---

## Next

`27_rewoo_parallel_planning.ipynb`

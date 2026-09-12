# Module 28 — Reflexion and Self-Refine
Status: `🔧 Code-built`

Theory: [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md) §5-6

---

## Use Case

The agent failed, writes down why, stores it, and does better next time — with no gradients.

The claim being tested: **Reflexion needs an EXTERNAL verifier. Without one you cannot reflect, only self-flatter.**

---

## The Mechanism

```
REFLEXION    attempt -> verify (EXTERNAL) -> write a lesson -> store -> retry
             the reflection PERSISTS into future tasks (module 25's store)
SELF-REFINE  generate -> self-critique -> revise
             no external signal; the critique dies with the task
STOP on: critic says APPROVED, OR outputs converge, OR a hard cap
```

---

## Key Implementation Details

**`verify()` is a plain function, not a model.** That is what separates Reflexion from self-flattery.

**All three stopping criteria are implemented as an OR**, because each fails alone: a strict critic never approves, a nitpicking critic never converges, and a cap catches everything else.

**`for ... else`** is the hard-cap branch — Python runs `else` only when the loop was not broken.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `28_reflexion_and_self_refine.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. Roughly 12-16 calls.

---

## Next

`../J_multi_agent/29_multi_agent_pipeline.ipynb`

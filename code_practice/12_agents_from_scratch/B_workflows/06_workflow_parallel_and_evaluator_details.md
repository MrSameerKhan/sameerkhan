# Module 06 — Workflow: Parallelization and Evaluator-Optimizer
Status: `🔧 Code-built`

Theory: [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §5 (patterns 3 and 5; the parallelization vs orchestrator-workers probe) · [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md) §6 (Self-Refine stopping criteria)

---

## Use Case

Closes Block B by covering the remaining workflow patterns and, more importantly, fixing
the boundary that the agent blocks then cross.

The claim being tested: **what separates parallelization from orchestrator-workers is
who decides the subtasks — you in code, or the model at runtime.**

---

## The Mechanism

```
SECTIONING   3 fixed topics -> ThreadPoolExecutor -> aggregate
             wall-clock = 1 call, not 3

VOTING       same question x5 at temp=1.0 -> Counter -> most_common
             buys reliability, NOT correctness

EVALUATOR    draft -> critique -> revise -> critique -> ...
             stops on APPROVED, or on the hard cap of 3
```

---

## Key Implementation Details

**`ThreadPoolExecutor`, not `asyncio`.** The SDK call is blocking I/O, so threads are
the right tool and the code stays readable. The wall-clock print is the evidence that
they really overlapped.

**Voting uses `temperature=1.0` deliberately.** At temperature 0 the five calls would be
near-identical and the vote would be theatre. Variance is what voting exists to absorb.

**The `for ... else` on the refine loop** is the hard-cap branch. Python runs `else` only
when the loop was not broken, which is exactly "the critic never said APPROVED". Two of
module 28's three stopping criteria are visible in eight lines.

**The LTV vote has a known answer, 89.1%.** A voting demo on an open-ended question
cannot show you anything.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Sectioning finishes in roughly the time of one call rather than three. The vote should be
unanimous or near-unanimous on 89.1. The refine loop usually approves in round 1 or 2;
if it never does, the `else` branch fires and proves the cap works.

---

## How to Run

Open `06_workflow_parallel_and_evaluator.ipynb`, select the `sameerkhan` kernel, run all
cells. Needs `OPENAI_API_KEY` only. Roughly 12-14 calls on the cheap model.

---

## Next

`../C_the_loop/07_agent_without_sdk_react.ipynb`

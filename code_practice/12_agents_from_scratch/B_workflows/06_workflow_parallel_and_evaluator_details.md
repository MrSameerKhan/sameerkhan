# Module 06 — Workflow: Parallelization and Evaluator-Optimizer
Status: `✅ Run` — **one prompt fix applied after the run**

Theory: [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §5 (patterns 3 and 5) · [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md) §6 (Self-Refine stopping criteria)

---

## Use Case

Closes Block B by covering the remaining workflow patterns and fixing the boundary the
agent blocks then cross.

The claim being tested: **what separates parallelization from orchestrator-workers is who
decides the subtasks — you in code, or the model at runtime.**

---

## The Mechanism

```
SECTIONING   3 fixed topics -> ThreadPoolExecutor -> aggregate
VOTING       same question x5 at temp=1.0 -> Counter -> most_common
EVALUATOR    draft -> critique -> revise, stopping on APPROVED or a hard cap
```

---

## Key Implementation Details

**`ThreadPoolExecutor`, not `asyncio`.** The SDK call is blocking I/O, so threads are the
right tool. The wall-clock print is the evidence they really overlapped.

**Voting uses `temperature=1.0` deliberately.** At temperature 0 the five calls would be
near-identical and the vote would be theatre.

**The `for ... else`** is the hard-cap branch — Python runs `else` only when the loop was
not broken.

---

## Fixes Applied (during run)

| Found | Fix |
|---|---|
| The refine loop's final output carried conversational wrapper text — *"Certainly! Here is the revised letter..."* and *"Let me know if you would like it tailored further!"* — around the actual letter. Harmless here, but in a chain the next step would parse that framing as content. | The revise prompt now says **"Reply with ONLY the letter text — no preamble, no closing offer."** Small, but it is the same discipline module 29 makes load-bearing at an agent hand-off. |

---

## Actual Output (macOS M1, `gpt-4.1-mini`, 2026-09-12)

**Sectioning ran concurrently**, three calls in the time of roughly one:

```
3 calls in 3.1s wall-clock — they ran concurrently, not back to back.
```

**Voting was unanimous and correct:**

```
votes : ['89.1', '89.1', '89.1', '89.1', '89.1']
tally : {'89.1': 5}
winner: 89.1  (true answer 89.1)
```

Unanimity at `temperature=1.0` means genuinely low variance on this task — which is the
point module 37 returns to: it means low variance, not correctness.

**The refine loop stopped on criterion 1**, the critic signalling done, in round 2:

```
ROUND 1 critique: The letter is clear and concise... However, it
ROUND 2 critique: APPROVED
  -> stopping criterion met
```

The hard cap was never reached, so the `for ... else` branch did not fire this run.

---

## How to Run

Open `06_workflow_parallel_and_evaluator.ipynb`, select the `sameerkhan` kernel, run all
cells. Needs `OPENAI_API_KEY` only. Roughly 12-14 calls on the cheap model.

---

## Next

`../C_the_loop/07_agent_without_sdk_react.ipynb`

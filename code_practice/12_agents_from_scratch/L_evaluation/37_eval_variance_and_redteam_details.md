# Module 37 — Variance and the Red-Team Corpus
Status: `✅ Run (statistics offline; red-team needs a key)`

Theory: [../../../8.agents/09_agent_evaluation.md](../../../8.agents/09_agent_evaluation.md) §8 (reliability / variance) · §10 (robustness to injection)

---

## Use Case

Module 36 ran each case once, which is an anecdote rather than an evaluation. Closes the ladder.

The claim being tested: **a bimodal 80% and a uniform 80% are different systems that report the same number.**

---

## The Mechanism

```
VARIANCE  N=5 trials per task -> report the DISTRIBUTION
          bimodal  4 tasks always pass, 1 always fails -> route around it
          uniform  every task 80% -> a coin flip; retries help

RED TEAM  standing, versioned corpus; two rates:
          payload acceptance (deviated)      target < 5%
          side-effect rate  (acted)          target < 1%
```

---

## Key Implementation Details

**The two distributions are constructed, not sampled**, so the contrast is unarguable and free.

**The red team runs against module 16's isolated reader**, so the near-zero side-effect rate is attributable to structure rather than to the model resisting — which the notebook says explicitly.

**Acceptance and side-effect are reported separately.** Acceptance is embarrassing; a side effect is an incident.

---

## Fixes Applied (during run)

None — statistics section ran clean on first execution.

---

## Actual Output

```
BIMODAL  mean success = 80%   deterministic tasks: 100%   flaky: none
UNIFORM  mean success = 80%   deterministic tasks:   0%   flaky: all 5

                      bimodal    uniform
mean success              80%        80%
task consistency         100%         0%
```

---

## How to Run

Open `37_eval_variance_and_redteam.ipynb`, select the `sameerkhan` kernel, run all cells. The statistics need no key. The red-team section needs `OPENAI_API_KEY`.

---

## Next

`— end of ladder —`

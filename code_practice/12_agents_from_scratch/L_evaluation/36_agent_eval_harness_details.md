# Module 36 — Agent Eval Harness
Status: `✅ Run (no API key)`

Theory: [../../../8.agents/09_agent_evaluation.md](../../../8.agents/09_agent_evaluation.md) §4-9 (the five metric classes) · [../../../8.agents/02_agent_reliability_patterns.md](../../../8.agents/02_agent_reliability_patterns.md) §4 (the scorecard)

---

## Use Case

Scores a trajectory rather than an answer, across the module-08 loop and the module-26 planner on identical cases.

The claim being tested: **no single metric is sufficient — both architectures score 100% success and one of them is unshippable.**

---

## The Mechanism

```
per run: success · tool_acc · efficiency (calls / optimal) · cost · safety
aggregate: means for the first four; safety as a COUNT, never a mean
```

---

## Key Implementation Details

**Safety is printed as a count, never averaged.** A 99% safety score means one run in a hundred did something forbidden — it is a gate, not a tradeoff.

**Both architectures answer every case correctly** by construction. That is the whole design: a single-metric harness reports them as equal.

**Runs on `_fake_model`**, so every axis is deterministic and verifiable.

---

## Fixes Applied (during run)

None — ran clean on first execution.

---

## Actual Output

```
architecture   success  tool_acc  effic    cost $  SAFETY
react(08)         100%      0.83   1.67  0.000240 2/    3
planner(26)       100%      1.00   1.00  0.000168 3/    3
```

Both 100% on success. `react` wandered (efficiency 1.67x) and fired a forbidden tool on one case.

---

## How to Run

Open `36_agent_eval_harness.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`37_eval_variance_and_redteam.ipynb`

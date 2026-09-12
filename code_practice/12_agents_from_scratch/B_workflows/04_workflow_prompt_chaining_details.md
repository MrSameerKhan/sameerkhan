# Module 04 — Workflow: Prompt Chaining
Status: `🔧 Code-built`

Theory: [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §5 (workflows vs agents, the five patterns)

---

## Use Case

A mortgage pre-check decomposes into fixed steps: extract, check eligibility, check
affordability, write the outcome. The sequence is known in advance, so the model should
not be choosing it.

The claim being tested: **a workflow's control flow lives in your code, and a gate can
halt the chain in a way the model cannot argue with.**

---

## The Mechanism

```
STEP 1  LLM    extract application       -> typed Loan object
GATE 1  YOU    ltv <= 95 ?               -> HALT on fail
STEP 2  LLM    summarise position
GATE 2  YOU    debt-to-income <= 45 ?    -> HALT on fail
STEP 3  LLM    write approval letter
```

Same path every run. Predictable cost, predictable latency, and the gates are unit
testable with no API key.

---

## Key Implementation Details

**The gates are arithmetic on typed fields**, which module 03 made possible. `ltv` is
computed from `loan.amount` and `loan.property_value` as floats — not regexed out of
prose. Chaining without structured output between steps is how the lost-in-translation
failure from `07_multi_agent_orchestration.md` gets in.

**`raise SystemExit` in gate 1** is deliberately blunt so the halt is visible in a
notebook. Production would return a rejection object.

**Step 2's LLM call is doing almost nothing.** That is honest: the affordability
*decision* is the ratio, and the model only writes prose about it. Noticing how little
work the model does in a workflow is part of the lesson.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

LTV computes to about 89.1% and passes. Debt-to-income is 950 / (74000/12), about 15.4%,
and passes. Both gates clear, so step 3 prints an approval note. To see a halt, raise
`monthly_debts` in `APPLICATION` above roughly 2775.

---

## How to Run

Open `04_workflow_prompt_chaining.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` only. Three calls.

---

## Next

`05_workflow_routing.ipynb`

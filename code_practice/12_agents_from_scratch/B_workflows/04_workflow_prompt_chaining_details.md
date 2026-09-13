# Module 04 — Workflow: Prompt Chaining
Status: `✅ Run`

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

---

## Key Implementation Details

**The gates are arithmetic on typed fields**, which module 03 made possible. `ltv` comes
from `loan.amount` and `loan.property_value` as floats, not regexed out of prose.

**Step 2's LLM call does almost no work.** That is honest: the affordability *decision* is
the ratio, and the model only writes prose about it. Noticing how little the model
decides in a workflow is part of the lesson.

---

## Fixes Applied (during run)

None — ran clean on first execution.

---

## Actual Output (macOS M1, `gpt-4.1-mini`, 2026-09-12)

```
STEP 1 extracted: {'applicant': 'Sarah Chen', 'income': 74000.0,
                   'monthly_debts': 950.0, 'amount': 285000.0,
                   'property_value': 320000.0}

GATE 1  LTV = 89.1%  (policy max 95.0%)      -> pass
GATE 2  debt-to-income = 15.4%  (policy max 45.0%)  -> pass

STEP 3 output:
 Approval is granted for Sarah Chen's loan request of $285,000 at an 89.1%
 loan-to-value ratio, subject to standard underwriting conditions.
```

Both gates computed exactly as predicted: 285000/320000 = 89.1%, and
950/(74000/12) = 15.4%. To see a halt, raise `monthly_debts` above roughly 2775.

---

## How to Run

Open `04_workflow_prompt_chaining.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` only. Three calls.

---

## Next

`05_workflow_routing.ipynb`

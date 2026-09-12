# Module 11 — Agent Cost and Tokens
Status: `🔧 Code-built`

Theory: [../../../8.agents/01b_agents_end_to_end.md](../../../8.agents/01b_agents_end_to_end.md) §2.2 (token count per turn, the 5-iteration dry run) · [../../../8.agents/09_agent_evaluation.md](../../../8.agents/09_agent_evaluation.md) §7 (cost thresholds)

---

## Use Case

Puts a measured number on the claim every agent article repeats without evidence. Closes
Block C by explaining why Blocks D through F exist at all.

The claim being tested: **an agent costs roughly 10x a single call, and the driver is
re-sent input, not generated output.**

---

## The Mechanism

```
single call     [question]                                   -> answer
                in = ~40

agent turn 1    [question] + SCHEMAS                          -> tool call
agent turn 2    [question] + SCHEMAS + asst + result          -> tool call
agent turn 3    [question] + SCHEMAS + asst + result + ...    -> answer
                                    ^ everything re-sent, every turn

total input grows ~QUADRATICALLY with turn count
```

---

## Key Implementation Details

**The baseline is a no-tools call on the same question.** It may answer wrongly from
parametric memory — that is fine and worth noting. It is the cost *floor*, not the
quality bar.

**Cost is computed per turn, not just in total**, so the growth curve is visible in the
table rather than inferred from one number.

**`msgs` is printed alongside tokens** to tie the token growth to the list growing, which
is module 02's lesson arriving with a price tag.

**Prices are `gpt-4.1-mini` rates.** Swap in Opus 5 rates and the multiplier rises
sharply because thinking is billed as output on every turn — module 01 measured 632
output tokens for a one-sentence answer.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Expect 3-4 turns. Input tokens should roughly double or triple from first turn to last.
The cost multiplier against the single call should land somewhere in the 5-15x range —
the exact figure varies with how many tools the model decides to call, and that variance
is itself part of the lesson.

---

## How to Run

Open `11_agent_cost_and_tokens.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` only. Under a cent.

---

## Next

`../D_reliability/12_failure_modes_reproduced.ipynb`

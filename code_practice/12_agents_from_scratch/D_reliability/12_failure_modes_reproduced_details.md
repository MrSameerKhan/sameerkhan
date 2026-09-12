# Module 12 — Five Failure Modes Reproduced
Status: `✅ Run` (no API key required — runs on `_fake_model`)

Theory: [../../../8.agents/02_agent_reliability_patterns.md](../../../8.agents/02_agent_reliability_patterns.md) §2 (the five modes, as observed) · §3 (the fixes, which land in module 13)

---

## Use Case

Every guard in Block D needs something to catch. Real models refuse to misbehave on
command, so the failures are scripted through `_fake_model.py`.

The claim being tested: **a guard you cannot trigger on demand is a guard you have never
verified.**

---

## The Mechanism

```
FakeModel([turn, turn, ...])  - same interface as client.chat.completions
                              - returns scripted turns in order
                              - repeats the LAST turn forever -> simulates FM4

loop()  = the naive module-08 loop with NO guards except max_steps
        = returns a trace, so module 13 can assert against the same five scripts
```

| mode | script | naive loop's response |
|---|---|---|
| FM1 | `raw_turn(json-in-content, None, "stop")` | treats it as a final answer |
| FM2 | `text_turn(prose plan)` | treats it as a final answer |
| FM3 | 4 tool turns, correct after 2 | executes all 4 |
| FM4 | 1 tool turn, `repeat_last=True` | runs to the step ceiling |
| FM5 | bad tool name, then bad arg name | `run_tool` returns ERROR strings |

---

## Key Implementation Details

**Nothing here raises.** Four of five failures return something plausible. That is the
entire reason they survive to production — a crash would have been caught in testing.

**`run_tool` returns structured errors rather than throwing**, by design in `_tools.py`.
An unknown tool is recoverable by the model on the next turn, so crashing would be the
wrong response.

**The malformed-JSON branch** wraps the raw string under `__malformed__` instead of
exploding, so FM5 stays inside the loop where module 13 can inspect it.

**`max_steps=8` is the only guard present**, and it is visibly doing all the work in FM3
and FM4. Noticing that is the setup for module 13.

---

## Fixes Applied (during run)

None — ran clean on the first execution.

---

## Actual Output (macOS M1, 2026-09-12)

```
FM1: 0 tool call(s), 0 error(s), ended=text
FM2: 0 tool call(s), 0 error(s), ended=text
FM3: 4 tool call(s), 0 error(s), ended=text
FM4: 8 tool call(s), 0 error(s), ended=capped
FM5: 2 tool call(s), 2 error(s), ended=text
```

---

## Expected Output

```
FM1: 0 tool call(s), 0 error(s), ended=text
FM2: 0 tool call(s), 0 error(s), ended=text
FM3: 4 tool call(s), 0 error(s), ended=text
FM4: 8 tool call(s), 0 error(s), ended=capped
FM5: 2 tool call(s), 2 error(s), ended=text
```

---

## How to Run

Open `12_failure_modes_reproduced.ipynb`, select the `sameerkhan` kernel, run all cells.
**No API key needed and no cost** — that is the point of the fake model.

---

## Next

`13_guardrails_and_budgets.ipynb`

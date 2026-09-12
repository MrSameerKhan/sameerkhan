# Module 13 — Guardrails and Budgets
Status: `✅ Run` (no API key required — runs on `_fake_model`)

Theory: [../../../8.agents/02_agent_reliability_patterns.md](../../../8.agents/02_agent_reliability_patterns.md) §3 (patterns A-F) · §5 (the combined production loop)

---

## Use Case

Module 12 produced five failures on demand. This module catches all five. Its pass
condition is exactly that, per the ladder README's cross-module checks.

The claim being tested: **each guard maps to a failure you can reproduce, and that
mapping is what makes the guard trustworthy.**

---

## The Mechanism

```
GUARD 1   rescue-from-content    regex JSON back out of content        -> FM1
GUARD 2a  plan-not-action        text turn with ZERO facts gathered
                                 -> nudge once, abort on repeat        -> FM2
GUARD 2b  no-progress stop       no NEW fact this turn -> stop         -> FM3
GUARD 3   duplicate detection    (tool, sorted-args) seen -> cache     -> FM4
GUARD 4   Pydantic validation    unknown tool / bad args -> ERROR      -> FM5
GUARD 5   token + iteration cap  hard ceiling -> ABORT AND ESCALATE    -> all
```

---

## Key Implementation Details

**Guard 1's regex is `\{\s*"name"`, with the backslash.** The copy in
`8.agents/02_agent_reliability_patterns.md` read `\{s*"name"` — zero-or-more literal
letter *s* — which matched compact JSON and silently missed pretty-printed JSON, which is
most of what a model emits into prose. Fixed in the theory file during this session.

**Every guard returns a structured error to the model, never an exception.** A
hallucinated tool name is recoverable next turn; raising discards that chance.

**Guard 2b tracks a set of facts, not a step count.** "Did the latest tool return data
not already in state?" is the real progress test. A step counter cannot tell FM3 apart
from legitimate multi-step work.

**Guard 5 aborts and escalates rather than returning a partial answer.**

---

## Fixes Applied (during run)

| # | Found | Fix |
|---|---|---|
| 1 | **Guard 2 did not catch FM2.** The first draft returned the prose plan as a final answer at step 1 — byte-identical behaviour to module 12's *unguarded* loop — while the summary table claimed a catch. | Added **guard 2a**: a text turn arriving with zero facts gathered is a plan, not an answer. Nudge once (`"Call the tool NOW"`), abort on a repeat. |
| 2 | FM5's label was then wrong. Both its tool calls fail validation, so nothing is ever grounded, and guard 2a correctly aborts on the ungrounded answer. | Relabelled to `guard 4, then 2a abort`. The layering is correct behaviour, not a regression. |

**Finding 1 is the module's own lesson landing on itself.** The notebook asserted a catch
it did not perform, and only re-running module 12's five scripts exposed it. A guard you
have not triggered is a guard you have not verified.

---

## Actual Output (macOS M1, 2026-09-12, `_fake_model`)

```
--- FM1 ---  step 1: GUARD1 rescued search_policy from content
             => Year-2 ERC is 10000.  (step 2)
--- FM2 ---  step 1: GUARD2 plan-not-action — nudging
             => ABORT: described a plan, never acted — escalate  (step 2)
--- FM3 ---  step 3: GUARD3 duplicate — serving cache
             => STOP: no new information this turn  (step 3)
--- FM4 ---  step 2: GUARD3 duplicate — serving cache
             => STOP: no new information this turn  (step 2)
--- FM5 ---  step 1: GUARD4 ERROR: unknown tool 'lookup_acct'
             step 2: GUARD4 ERROR: bad args for 'calculate': ('expression',)
             step 3: GUARD2 plan-not-action — nudging
             => ABORT: described a plan, never acted — escalate  (step 4)

mode  caught by                    steps
FM1   guard 1 rescue-from-content      2
FM2   guard 2a nudge, then abort       2
FM3   guard 2b + 3                     3
FM4   guard 3 duplicate detect         2
FM5   guard 4, then 2a abort           4
```

**Pass condition met: all five caught.** Compare module 12, where FM4 burned all 8 steps
and FM1/FM2 both returned a confident non-answer.

---

## Known Deviation

93 code lines against the ladder's ~90 guideline. Left as-is: the module teaches one idea
(guards map to reproducible failures) across five instances of it, and compressing
further costs more in readability than the three lines are worth.

---

## How to Run

Open `13_guardrails_and_budgets.ipynb`, select the `sameerkhan` kernel, run all cells.
**No API key needed and no cost.**

---

## Next

`14_audit_log_and_hitl.ipynb`

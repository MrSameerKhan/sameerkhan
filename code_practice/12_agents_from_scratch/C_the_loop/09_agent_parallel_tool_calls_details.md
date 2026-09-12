# Module 09 — Parallel Tool Calls
Status: `🔧 Code-built`

Theory: **none — this is a gap in `8.agents/`.** The folder never covers several tool
calls in one assistant turn, and the coverage map in the ladder README has no row for
module 09. Nearest context: [../../../8.agents/01_agents.md](../../../8.agents/01_agents.md) (tool calling).

---

## Use Case

Three independent policy lookups. A competent model asks for all three in one turn. The
orchestrator has to hand back every result together or the behaviour degrades silently.

The claim being tested: **return all tool results in a single message, or the model
stops parallelising — with no error raised.**

---

## The Mechanism

```
CORRECT                              BUG
assistant: [call1, call2, call3]     assistant: [call1, call2, call3]
user/tool: [res1, res2, res3]        user/tool: [res1] only
           ^ one round trip                     ^ model waits, re-asks, serialises
turns: 2                                        turns: 4+
```

Cost and latency scale with **turns**, not with tool calls, because every turn re-sends
the whole conversation (module 02).

---

## Key Implementation Details

**The bug is simulated by withholding results**, not by splitting them across messages,
because the OpenAI API rejects a follow-up that leaves a `tool_call_id` unanswered. The
withheld-result placeholder produces the same behavioural signal the real bug produces.

**`sequential_cost` counts turns deliberately.** Counting tool calls would show the two
runs as equal and miss the entire point.

**Only `search_policy` is exposed.** Restricting the tool surface to one tool makes the
turn structure legible; a second tool would let the model interleave and muddy the
comparison.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Run A should show a single step with 3 tool calls, then a text turn — `calls per turn:
[3]`. Run B should show more turns with fewer calls each, for example `[3, 1, 1]` or
`[1, 1, 1]`. The exact shape varies; the direction should not.

If Run A does not batch, the model chose not to — re-run, or make the three subtasks more
obviously independent.

---

## How to Run

Open `09_agent_parallel_tool_calls.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` only. Well under a cent.

---

## Next

`10_rag_as_a_tool.ipynb`

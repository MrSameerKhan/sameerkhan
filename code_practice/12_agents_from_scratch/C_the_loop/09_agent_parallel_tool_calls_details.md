# Module 09 — Parallel Tool Calls
Status: `✅ Run`

Theory: **none — this is a gap in `8.agents/`.** The folder never covers several tool
calls in one assistant turn, and the ladder README's coverage map has no row for module 09.

---

## Use Case

Three independent policy lookups. The orchestrator must hand back every result together
or the behaviour degrades silently.

The claim being tested: **return all tool results in a single message, or the model stops
parallelising — with no error raised.**

---

## Fixes Applied (during run)

None — ran clean on first execution, and the contrast was as sharp as the design intended.

---

## Actual Output (macOS M1, `gpt-4.1-mini`, 2026-09-12)

```
run                      calls/turn   turns
A batched (correct)             [3]       1
B withheld (bug)       [3, 1, 1, 1]       4
```

**Run A** asked for all three lookups in one turn and finished in a single round trip.

**Run B** opened identically — the model still asked for three — then, starved of two of
the three results, **serialised for the rest of the conversation**: one call per turn for
three more turns. Four round trips instead of one, for the same three lookups.

Nothing raised. The model simply learned, inside one conversation, that batching was
pointless. That is the failure mode in its entirety.

---

## How to Run

Open `09_agent_parallel_tool_calls.ipynb`, run all cells. Needs `OPENAI_API_KEY` only.

---

## Next

`10_rag_as_a_tool.ipynb`

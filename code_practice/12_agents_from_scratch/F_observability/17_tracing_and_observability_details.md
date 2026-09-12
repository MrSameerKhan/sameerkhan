# Module 17 — Tracing and Observability
Status: `🔧 Code-built`

Theory: [../../../8.agents/09_agent_evaluation.md](../../../8.agents/09_agent_evaluation.md) §6-7 (trajectory, cost and latency metrics) · §13 (the production eval pipeline) · [../../../10.mlops/11_llm_observability.md](../../../10.mlops/11_llm_observability.md)

---

## Use Case

Block F, like Block E, exists because a 2026 practice check found `8.agents/` treated
observability as a cross-reference rather than a foundation. Module 14 traced one run for
compliance; this generalises it to the six fields you need for every run.

The claim being tested: **you cannot retrofit an audit trail, and the same is true of
observability.**

---

## The Mechanism

```
Tracer(run_id, intent)
  .span(kind, **meta)   context manager - times the block, records metadata
                        kinds: llm · tool · approval · tool_blocked
  .close(outcome)       rolls up turns, tool calls, approvals, tokens, cost,
                        wall time; appends ONE json line per run

roll-up table -> the dashboard view
spans         -> the drill-down view
```

---

## Key Implementation Details

**`span` is a context manager, so latency is captured even if the body raises.** A
timing you only record on success hides exactly the slow failures you need to find.

**Three runs with deliberately different shapes** — a short one, a wanderer (module 12's
FM3), and one with a denied write. A single run cannot demonstrate a roll-up.

**Cost is computed, not logged by the provider.** Tokens times price, per run. That is
the number that gets an agent switched off, and it has to be attributable per run to be
actionable.

**The wanderer costs several times the short run for the same question**, and the module
points at it. That comparison is only possible because both were recorded.

**Runs on `_fake_model`** so the trace is deterministic and free. The tracer is provider
agnostic — swapping in a real client changes nothing above.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Three rows. `r-short` with 2 turns and 1 tool call; `r-wander` with 5 turns, 4 tool
calls and roughly 4x the cost; `r-write` with 1 approval, 0 writes executed and no
`COMMITTED` result. `traces.jsonl` is written into `F_observability/`.

---

## How to Run

Open `17_tracing_and_observability.ipynb`, select the `sameerkhan` kernel, run all cells.
**No API key needed and no cost.**

---

## Next

`../G_frameworks/18_langchain_lcel.ipynb`

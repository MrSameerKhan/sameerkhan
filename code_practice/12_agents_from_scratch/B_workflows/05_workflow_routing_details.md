# Module 05 — Workflow: Routing
Status: `🔧 Code-built`

Theory: [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §5 (routing is Anthropic's pattern 2, and the cost lever)

---

## Use Case

Not every question needs a frontier model. A rate lookup and a multi-policy
underwriting conflict arrive on the same endpoint and cost the same to serve unless you
split them.

The claim being tested: **classify-then-dispatch is the biggest cost lever in an LLM
system, and it is a workflow.**

---

## The Mechanism

```
question -> [cheap classifier, constrained to 3 values] -> tier
                                                            |
              lookup ---> local llama3.2 on the M1      free
              standard -> gpt-4.1-mini                  cheap
              complex --> claude-opus-5                 expensive
```

The classifier is one `gpt-4.1-mini` call with a `Literal` schema, so an unknown tier is
unrepresentable and the dispatch table can never miss.

---

## Key Implementation Details

**The `complex` branch uses the Anthropic envelope**, the other two use OpenAI. That is
module 01's lesson doing real work: one `if` for the format, and `local` shares the
OpenAI path with `gpt-4.1-mini` for free.

**Prices are per million tokens**, input and output, taken from current published rates.
`lookup` is priced at zero because it runs on your own hardware — the real cost is
electricity and the RAM it occupies.

**The saving is computed against an all-Opus baseline**, which is the honest comparison:
what you would pay if you skipped routing and sent everything to the best model.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

The three questions should classify as `lookup`, `standard`, `complex` in order. The
saving against all-Opus should land somewhere around 70-90%, driven mostly by the local
model absorbing question 1 and Opus's thinking tokens only being paid once.

Requires `ollama serve` with `llama3.2` pulled, or the `lookup` tier will fail.

---

## How to Run

Open `05_workflow_routing.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs both API keys **and** Ollama running. Six calls total, roughly 2 cents — the Opus
call dominates.

---

## Next

`06_workflow_parallel_and_evaluator.ipynb`

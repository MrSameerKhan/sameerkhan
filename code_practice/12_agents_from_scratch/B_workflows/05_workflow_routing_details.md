# Module 05 — Workflow: Routing
Status: `🔧 Code-built` — **re-run required after the fix below**

Theory: [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §5 (routing is pattern 2, and the cost lever)

---

## Use Case

Not every question needs a frontier model. A rate lookup and a multi-policy underwriting
conflict arrive on the same endpoint and cost the same to serve unless you split them.

The claim being tested: **classify-then-dispatch is the biggest cost lever in an LLM
system — and the size of the saving depends entirely on your traffic mix.**

---

## The Mechanism

```
question -> [cheap classifier, constrained to 3 values] -> tier
              lookup ---> local llama3.2 on the M1      free
              standard -> gpt-4.1-mini                  cheap
              complex --> claude-opus-5                 expensive
```

---

## Key Implementation Details

**The `complex` branch uses the Anthropic envelope**, the other two use OpenAI. Module
01's lesson doing real work: one `if` for the format, and `local` shares the OpenAI path
with `gpt-4.1-mini` for free.

**The classifier is constrained to a `Literal`**, so an unknown tier is unrepresentable
and the dispatch table can never miss.

---

## Fixes Applied (during run)

| # | Found | Fix |
|---|---|---|
| 1 | **The router was free.** `classify()` discarded `r.usage` entirely, so a module about cost accounting never charged for its own classifier — one `gpt-4.1-mini` call per request, omitted from the total. | `ROUTER` accumulator added; router cost is now a row in the table and is included in the routed total, with its share of the bill printed. |
| 2 | **The predicted saving was wrong.** This file previously expected 70-90%. The measured saving was **18%**. | Rewrote the lesson to say what actually drives it: the saving is proportional to how much traffic you can move off the expensive model. One of three questions genuinely needed Opus, and that one question dominated the bill. |
| 3 | The `complex` tier returned `out = 2000` exactly — it **hit the `max_tokens` ceiling**. On Opus 5 that budget is shared with thinking (module 01's trap), so the answer may be truncated and you paid for all of it. | Called out explicitly in the lesson text. |

**Finding 2 is the useful one.** A routing demo that claims a large saving on three
cherry-picked questions is misleading. The honest version tells you to measure your own
mix, and notes that on mostly-hard traffic routing saves almost nothing.

---

## Actual Output (macOS M1, 2026-09-12) — **before the fix**

```
lookup    <- What is the standard variable rate?
standard  <- Summarise the early repayment charges for a 5-year fix.
complex   <- A self-employed applicant with 2 years of accounts wants 92% LTV...

tier           in    out     cost $
lookup         32    183    0.00000
standard       21    258    0.00042
complex        52   2000    0.05026
routed                      0.05068
all-opus                    0.06155

routing saved 18% on these three questions
```

Classification was correct on all three. **The routed figure above excludes the router**,
which is exactly the defect fixed. Re-run to get the corrected total.

---

## How to Run

Open `05_workflow_routing.ipynb`, select the `sameerkhan` kernel, run all cells. Needs
both API keys **and** Ollama running with `llama3.2`. Six calls plus three classifier
calls; the Opus call dominates at roughly 5 cents.

---

## Next

`06_workflow_parallel_and_evaluator.ipynb`

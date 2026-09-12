# Module 29 — Pipeline Hand-offs and Lost-in-Translation
Status: `🔧 Code-built`

Theory: [../../../8.agents/07_multi_agent_orchestration.md](../../../8.agents/07_multi_agent_orchestration.md) §14 (context explosion, lost in translation) · §15 Q3

---

## Use Case

The #1 production failure of multi-agent systems is not reasoning. It is formatting drift at a hand-off.

The claim being tested: **every agent boundary needs a typed contract, exactly like a service boundary in microservices.**

---

## The Mechanism

```
TYPED      A -> Extracted(loan_amount: float, ...) -> B
           a drift in phrasing is REJECTED at the boundary, loudly

FREE TEXT  A -> prose -> regex in B
           'GBP 285,000' instead of '285000' -> B misparses -> NO EXCEPTION
           downstream agents act on wrong data; the answer is plausible
```

---

## Key Implementation Details

**The v1/v2 pair changes only the formatting**, not the content — a prompt tweak a human would not notice.

**Validation happens on receipt, not on send.** That is the microservices discipline and it is what makes the failure loud.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `29_multi_agent_pipeline.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. One extraction call.

---

## Next

`30_multi_agent_supervisor.ipynb`

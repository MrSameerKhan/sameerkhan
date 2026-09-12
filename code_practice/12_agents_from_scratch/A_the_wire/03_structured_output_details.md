# Module 03 — Structured Output
Status: `🔧 Code-built`

Theory: [../../../8.agents/03_langchain_primer.md](../../../8.agents/03_langchain_primer.md) §4 (output parsers, and why they fail 5-15%) · [../../../5.transformers/02_models/12_constrained_decoding.md](../../../5.transformers/02_models/12_constrained_decoding.md)

---

## Use Case

An agent's tool call is structured data. If you cannot reliably get structured data out
of a model, every tool call is a coin flip. This module establishes the guarantee that
modules 08 and 13 depend on.

The claim being tested: **constrained output removes a failure class rather than
catching it.**

---

## The Mechanism

```
1. FREE TEXT          "Here's the JSON you asked for: ```json {...}```"
   + json.loads       -> breaks on the preamble, the fence, a renamed key
                         You are writing a parser for an adversary who changes daily.

2. JSON MODE          response_format={"type": "json_object"}
                      -> ALWAYS parses. Shape is still whatever the model felt like.
                         {"name": ...} instead of {"applicant": ...} still "works".

3. STRICT SCHEMA      response_format={"type": "json_schema",
                                       "json_schema": {..., "strict": True}}
                      -> the decoder is constrained. Off-schema is UNREPRESENTABLE.
```

---

## Key Implementation Details

**`additionalProperties: False` is mandatory in strict mode.** Pydantic's
`model_json_schema()` does not add it, so the module sets it by hand. Without it the API
rejects the request.

**Attempt 1 may well succeed on the day you run it.** That is the point, not a flaw in
the demo — a failure mode that appears one run in twenty is worse than one that appears
every time, because you ship it. If attempt 1 passes, the lesson is that nothing
*guaranteed* it.

**The LTV line at the end is deliberate.** It does arithmetic on `loan.amount`, a typed
float, not on a substring. That is the payoff of the whole module.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Attempt 1 prints raw prose, possibly fenced, and may parse or fail. Attempt 2 always
parses. Attempt 3 prints a clean typed object and an LTV of about 89.1%
(285000 / 320000).

---

## How to Run

Open `03_structured_output.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` only. Three calls, well under a cent.

---

## Next

`../B_workflows/04_workflow_prompt_chaining.ipynb`

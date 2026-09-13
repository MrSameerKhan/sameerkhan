# Module 03 — Structured Output
Status: `✅ Run`

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
1. FREE TEXT       "```json { ... } ```"
   + json.loads    -> breaks on the fence, the preamble, a renamed key

2. JSON MODE       response_format={"type": "json_object"}
                   -> ALWAYS parses. Shape is still whatever the model chose.

3. STRICT SCHEMA   response_format={"type": "json_schema",
                                    "json_schema": {..., "strict": True}}
                   -> the decoder is constrained. Off-schema is UNREPRESENTABLE.
```

---

## Key Implementation Details

**`additionalProperties: False` is mandatory in strict mode.** Pydantic's
`model_json_schema()` does not add it, so the module sets it by hand. Without it the API
rejects the request.

**The LTV line at the end is deliberate.** It does arithmetic on `loan.amount`, a typed
float, not on a substring. That is the payoff of the whole module.

---

## Fixes Applied (during run)

None — ran clean on first execution, and every attempt failed or succeeded exactly where
the module predicted.

---

## Actual Output (macOS M1, `gpt-4.1-mini`, 2026-09-12)

**Attempt 1 failed for real**, on a markdown code fence — the predicted failure actually
occurred rather than passing by luck:

```
RAW:
 ```json
{ "applicant": "Sarah Chen", "annual_income": 74000, ... }
```

parsed -> FAILED: 1 validation error for Loan
  Invalid JSON: expected value at line 1 column 1
```

**Attempt 2 parsed, and still did not match.** JSON mode removed the fence but the model
chose its own field names — `annual_income` and `loan_amount` where the schema wants
`income` and `amount`:

```
json.loads -> OK
matches MY schema -> NO: 2 validation errors for Loan
```

That is the exact "renamed key" failure the module warns about, demonstrated live. Note
the model also invented a sixth field, `loan_term`, in both attempts.

**Attempt 3 produced a typed object**, and the arithmetic runs on a float:

```json
{ "applicant": "Sarah Chen", "income": 74000.0, "amount": 285000.0,
  "property_value": 320000.0, "first_time_buyer": true }
```

```
LTV = 89.1%   (arithmetic on a TYPED field, not on a guess)
```

285000 / 320000 = 89.1%, matching the expected value exactly.

**All three attempts behaved as designed**, which makes this the cleanest demonstration in
Block A: the progression is not argued, it is observed.

---

## How to Run

```bash
conda activate sameerkhan
cd code_practice/12_agents_from_scratch/A_the_wire
```

Open `03_structured_output.ipynb`, select the `sameerkhan` kernel, run all cells. Needs
`OPENAI_API_KEY` only. Three calls, well under a cent.

---

## Next

`../B_workflows/04_workflow_prompt_chaining.ipynb` — sequential calls with a
programmatic gate between steps.

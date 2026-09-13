# Module 08 — Native Tool Calling
Status: `✅ Run`

Theory: [../../../8.agents/01_agents.md](../../../8.agents/01_agents.md) (tool calling, both APIs) · [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §3 (both envelopes, both traps)

---

## Use Case

The same question module 07 answered with a hand-written parser, answered again with the
provider's structured tool-call channel — written as a diff against 07.

The claim being tested: **native tool calling buys format reliability, not judgment.**

---

## Key Implementation Details

**`content` is printed on every OpenAI step** so the `None` is seen rather than read about.

**`.arguments` is a JSON string, `.input` is a dict.** OpenAI needs `json.loads`,
Anthropic does not. Forgetting this is the commonest port-between-envelopes bug.

---

## Fixes Applied (during run)

None — ran clean on first execution.

---

## Actual Output (macOS M1, 2026-09-12)

**Both envelopes reached £10,000 in 4 steps.** Both traps appeared exactly as documented:

```
OpenAI     step 1: finish_reason=tool_calls  content=None      <- THE TRAP
Anthropic  step 1: stop_reason=tool_use  blocks=['text', 'tool_use']
           step 2: stop_reason=tool_use  blocks=['thinking', 'text', 'tool_use']
```

Note step 1 on Anthropic had **no thinking block** and step 2 did — `content[0]` is not
reliably any particular type, which is precisely why filtering by `.type` is mandatory.

**Both models self-corrected.** Each first searched a natural-language phrase, got
`No policy matched. Try one of: ltv, erc, ...`, and retried with `erc`:

```
-> search_policy({'query': 'early repayment charge year 2'}) = No policy matched...
-> search_policy({'query': 'erc'}) = Early repayment charge: 5% yr1, 4% yr2, ...
-> calculate({'expression': '250000 * 0.04'}) = 10000.0
```

That recovery is the argument for `_tools.py` returning a **structured error listing the
valid keys** rather than raising — the model fixed itself on the next turn.

**4 steps, not the 3 this file originally predicted.** The extra step is the failed first
search. Worth keeping: it is a more honest trace than a clean 3-step run.

Claude's final answer also volunteered two caveats nobody asked for — whether the
percentage applies to the original advance or the outstanding balance, and whether a
penalty-free overpayment allowance applies first. Both are real underwriting questions the
policy string does not answer.

---

## How to Run

Open `08_agent_with_sdk_tool_calling.ipynb`, run all cells. Needs both API keys. The
Anthropic loop costs a few cents because thinking is billed on every turn.

---

## Next

`09_agent_parallel_tool_calls.ipynb`

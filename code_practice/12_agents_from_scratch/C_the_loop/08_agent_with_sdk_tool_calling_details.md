# Module 08 — Native Tool Calling
Status: `🔧 Code-built`

Theory: [../../../8.agents/01_agents.md](../../../8.agents/01_agents.md) (tool calling, both APIs) · [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §3 (the two envelopes and both traps)

---

## Use Case

The same question module 07 answered with a hand-written parser, answered again with the
provider's structured tool-call channel. Written as a diff against 07 so the deletion is
visible.

The claim being tested: **native tool calling replaces your parser with the provider's
contract — and buys format reliability, not judgment.**

---

## The Mechanism

```
OpenAI                                  Anthropic
------                                  ---------
tools=openai_schemas(...)               tools=anthropic_schemas(...)
finish_reason == "tool_calls"           stop_reason == "tool_use"
msg.tool_calls[i].function.name         block.type == "tool_use" -> block.name
        .arguments  (JSON STRING)               .input      (already a dict)
append msg, then                        append {"role":"assistant","content":r.content}
{"role":"tool", "tool_call_id":...}     then {"role":"user", "content":[tool_result...]}
```

Both loops are the same five lines of logic in two shapes.

---

## Key Implementation Details

**`content` is printed explicitly on every OpenAI step** so the `None` is seen rather
than read about. Printing it raw in production code is what makes this trap bite.

**`.arguments` is a JSON string, `.input` is a dict.** OpenAI needs `json.loads`,
Anthropic does not. Forgetting this is the most common port-between-envelopes bug.

**The assistant turn is appended verbatim** in both. Dropping it, or reconstructing it
from text, breaks the `tool_use_id` / `tool_call_id` pairing and the next call 400s.

**`_tools.py` supplies both schema shapes from one source**, so the two loops are
genuinely diffable — only the machinery differs.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Both loops should call `search_policy` then `calculate` and land on **10000** (250000 at
4% for year 2), in 3 steps each. OpenAI should print `content=None` on steps 1 and 2.
Anthropic should print `blocks=['thinking', 'tool_use']` or similar, never a bare
`['tool_use']` on Opus 5.

---

## How to Run

Open `08_agent_with_sdk_tool_calling.ipynb`, select the `sameerkhan` kernel, run all
cells. Needs both API keys. The Anthropic loop costs a few cents because thinking is
billed on every turn — see module 11.

---

## Next

`09_agent_parallel_tool_calls.ipynb`

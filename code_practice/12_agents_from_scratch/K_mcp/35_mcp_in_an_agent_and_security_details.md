# Module 35 — MCP in an Agent, and Tool Poisoning
Status: `🔧 Code-built`

Theory: [../../../8.agents/08_mcp_protocol_deep.md](../../../8.agents/08_mcp_protocol_deep.md) §7 (MCP in an agent) · §11 (security) · [../../../7.rag/03_indirect_prompt_injection.md](../../../7.rag/03_indirect_prompt_injection.md)

---

## Use Case

MCP tools drop into module 08's loop in eight lines. Then the new attack surface: the tool description itself.

The claim being tested: **runtime discovery is MCP's value AND its vulnerability — a third party injects text into your context on every request.**

---

## The Mechanism

```
module 15   payload in a tool RESULT       reaches the model only after a call
module 35   payload in a tool DESCRIPTION  reaches it on EVERY request, before any call

defence: pin a fingerprint of (name, description, schema). Any drift blocks the session.
```

---

## Key Implementation Details

**Honest and poisoned servers differ only in the description.** Same name, same schema — and no client UI shows a tool description.

**The rug pull is the real threat**: a server can serve honest descriptions for a week, then switch. Your agent was audited against a schema it no longer receives.

**The fingerprint defence is cheap and catches it**, which is why it is the one recommended outright.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `35_mcp_in_an_agent_and_security.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY` for the live agent sections.

---

## Next

`../L_evaluation/36_agent_eval_harness.ipynb`

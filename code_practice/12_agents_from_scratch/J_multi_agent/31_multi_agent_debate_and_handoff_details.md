# Module 31 — Debate and Swarm-style Handoff
Status: `🔧 Code-built`

Theory: [../../../8.agents/07_multi_agent_orchestration.md](../../../8.agents/07_multi_agent_orchestration.md) §2 (debate) · §5 (Swarm handoff) · §15 Q5

---

## Use Case

Two coordination shapes at opposite ends of the cost curve, closing Block J.

The claim being tested: **a handoff is just a tool call that returns a different agent — the cheapest multi-agent there is.**

---

## The Mechanism

```
DEBATE   proposer -> critic -> judge        3-5 calls, best on CONTESTABLE questions
HANDOFF  triage --transfer_to_policy--> policy specialist
         the conversation CONTINUES; no supervisor, no re-loaded context
```

---

## Key Implementation Details

**The debate question is contestable on purpose.** On a lookup question debate adds nothing, because there is nothing to contest — and the single-agent control makes that visible.

**Handoff tools take no parameters.** The transfer *is* the side effect; the payload is the conversation itself.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `31_multi_agent_debate_and_handoff.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. Roughly 6-8 calls.

---

## Next

`../K_mcp/32_mcp_server_capabilities.ipynb`

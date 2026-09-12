# Module 33 — MCP Client, Handshake and Runtime Discovery
Status: `🔧 Code-built`

Theory: [../../../8.agents/08_mcp_protocol_deep.md](../../../8.agents/08_mcp_protocol_deep.md) §2 (architecture) · §6 (client) — see the version note in module 32

---

## Use Case

Why MCP exists at all: the agent does not know what a server can do until it asks.

The claim being tested: **runtime discovery is the point — every earlier module IMPORTED its schemas and had to be redeployed to gain a tool.**

---

## The Mechanism

```
initialize            -> protocol version + capabilities negotiated
notifications/initialized (NO id — notifications expect no reply)
tools/list            -> DISCOVERY; the agent learns its own tool surface
tools/call            -> operation

operations before `initialized` are refused with -32002
```

---

## Key Implementation Details

**Order is enforced by the server**, so the session is a state machine rather than a bag of endpoints.

**The `to_openai` adapter is eight lines** and is the whole of `langchain-mcp-adapters`. The value of MCP is the agreement, not the code.

**Version mismatch fails at the handshake** with a clear error, rather than as a confusing failure three calls later.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `33_mcp_client_and_lifecycle.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`34_mcp_transports.ipynb`

# Module 32 — MCP Server Capabilities and Control Dynamics
Status: `🔧 Code-built`

Theory: [../../../8.agents/08_mcp_protocol_deep.md](../../../8.agents/08_mcp_protocol_deep.md) §3 (capabilities) · §13 Q3 — **but see the version note below**

---

## Use Case

The three capabilities an MCP server exposes, and the distinction that actually gets probed: who controls each one.

The claim being tested: **tool vs resource is a CONTROL question, not a read/write one.**

---

## The Mechanism

```
Tool      -> the MODEL decides to call it        needs a schema, needs a gate if it writes
Resource  -> the APPLICATION attaches it        read-only by construction
Prompt    -> the USER picks it from a menu      the server ships a workflow

all of it is JSON-RPC 2.0 over a byte stream
```

---

## Key Implementation Details

**Built from scratch — the `mcp` package is not installed** and is not needed. The protocol is JSON, so implementing it is the clearest way to show there is no magic.

**Unknown methods and resources return JSON-RPC errors, not exceptions**, matching real server behaviour.

**Version note:** this block is built on JSON-RPC 2.0 and the tools/resources/prompts core, which are stable. The ladder README cites a 2026-07-28 revision adding MRTR and making MCP stateless. **That could not be verified against a primary source**, so nothing here depends on it.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `32_mcp_server_capabilities.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`33_mcp_client_and_lifecycle.ipynb`

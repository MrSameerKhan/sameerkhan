# Module 34 — MCP Transports: stdio over Real Pipes
Status: `✅ Run (no API key)`

Theory: [../../../8.agents/08_mcp_protocol_deep.md](../../../8.agents/08_mcp_protocol_deep.md) §4 (transports) · §11 (security) — **see the version note below**

---

## Use Case

The protocol is identical either way. The transport decides who can reach the server and who holds the credentials.

The claim being tested: **the transport is a security decision wearing a performance costume.**

---

## The Mechanism

```
stdio  server is a SUBPROCESS of the host
       newline-delimited JSON on stdin/stdout — that loop IS the transport
       credentials INHERITED; no trust boundary to cross
       the process lifetime IS the session lifetime

HTTP   real trust boundary -> auth per request, session isolation, blast radius
```

---

## Key Implementation Details

**A real subprocess is spawned and a real handshake runs over real pipes.** The server is written to `_stdio_server.py`, used, then deleted.

**Closing stdin ends the session** — no shutdown RPC needed, which makes the state point concrete.

**Version note.** The ladder README cites a 2026-07-28 revision making MCP stateless and introducing MRTR, deprecating `sampling`, `elicitation` and `roots`. **I could not verify that against a primary source**, so this module does not describe MRTR and does not depend on it. Confirm against the spec before teaching those details. The module was renamed from `34_mcp_transports_and_mrtr` for that reason.

---

## Fixes Applied (during run)

None — ran clean on first execution after avoiding a nested triple-quoted docstring in the generated server file.

---

## Actual Output

```
wrote _stdio_server.py (27 lines)
subprocess pid=76914 — a CHILD of this kernel, sharing its credentials
initialize  -> {'name': 'stdio-policy', 'version': '0.1'}
tools/list  -> ['search_policy']
tools/call  -> ERC: 5% yr1, 4% yr2, 3% yr3, 2% yr4, 1% yr5.

3 round trips over real pipes in 23.0 ms
server exited, returncode = 0
```

---

## How to Run

Open `34_mcp_transports.ipynb`, select the `sameerkhan` kernel, run all cells. **No API key needed and no cost.**

---

## Next

`35_mcp_in_an_agent_and_security.ipynb`

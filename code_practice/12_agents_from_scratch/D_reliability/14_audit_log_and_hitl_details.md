# Module 14 — Audit Log and Human-in-the-Loop
Status: `✅ Run` (no API key required — runs on `_fake_model`)

Theory: [../../../8.agents/02_agent_reliability_patterns.md](../../../8.agents/02_agent_reliability_patterns.md) §3 (HITL, audit) · [../../../8.agents/04_langgraph_deep.md](../../../8.agents/04_langgraph_deep.md) §5 (interrupts — the framework version of this gate)

---

## Use Case

A lending decision is a regulated action. It needs a reconstructable trace and a named
human behind any state change. This module builds both by hand so that LangGraph's
`interrupt()` in module 20 reads as a convenience rather than a capability.

The claim being tested: **you cannot retrofit an audit trail.**

---

## The Mechanism

```
emit()  -> append one JSON object per line to run_trace.jsonl, flushed immediately
           run_start · llm_turn · approval_requested · approval_decision
           tool_call · tool_blocked · run_end

gate    -> if tool in WRITE_TOOLS:  ask a human FIRST
           approved -> run it, log it
           denied   -> log tool_blocked, feed the refusal back as an observation
```

Reads run freely. Writes need a name against them.

---

## Key Implementation Details

**The gate keys off `WRITE_TOOLS` in `_tools.py`, not off the prompt.** A model can be
argued out of a prompt instruction. It cannot be argued out of a Python `if`. This is the
capability-isolation idea that module 16 leans on.

**A denial is an observation, not an exception.** The model is told it was refused and
can respond sensibly. Raising would discard the run.

**`approve_fn` is injected** so the notebook runs unattended and both branches can be
demonstrated. Production swaps in a queue or a UI; the call site is unchanged.

**Latency is recorded per LLM turn and per tool call.** It cannot be reconstructed after
the fact, which is the concrete form of the "cannot retrofit" claim.

**`list(SCRIPT)` is passed per run** because `FakeModel` holds its own call counter.
Sharing one instance across both runs would resume mid-script.

---

## Fixes Applied (during run)

None — ran clean on the first execution.

---

## Actual Output (macOS M1, 2026-09-12)

```
Waiver recorded.
run          event                detail                                  
--------------------------------------------------------------------------
run-approve  run_start            step                                    
run-approve  llm_turn             step 1                                  
run-approve  tool_call            fetch_customer_note                     
run-approve  llm_turn             step 2                                  
run-approve  approval_requested   confirm_decision                        
run-approve  approval_decision    confirm_decision approved=True by s.khan
run-approve  tool_call            confirm_decision                        
run-approve  llm_turn             step 3                                  
run-approve  run_end              answered                                
run-deny     run_start            step                                    
run-deny     llm_turn             step 1                                  
run-deny     tool_call            fetch_customer_note                     
...
18 events · 1 write(s) executed · 2 gate decision(s)
```

---

## Expected Output

Run 1 logs `approval_requested` then `approval_decision approved=True` then a
`tool_call` with `write=True` and a `COMMITTED:` result. Run 2 logs the request, an
`approved=False` decision, a `tool_blocked`, and **no** committed write anywhere.

The table should end with roughly 14-18 events, 1 write executed, 2 gate decisions.
`run_trace.jsonl` is written into `D_reliability/` and deleted on each re-run.

---

## How to Run

Open `14_audit_log_and_hitl.ipynb`, select the `sameerkhan` kernel, run all cells.
**No API key needed and no cost.**

---

## Next

`../E_security/15_prompt_injection_via_tools.ipynb`

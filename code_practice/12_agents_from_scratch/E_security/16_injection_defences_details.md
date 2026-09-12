# Module 16 — Injection Defences
Status: `🔧 Code-built`

Theory: [../../../7.rag/03_indirect_prompt_injection.md](../../../7.rag/03_indirect_prompt_injection.md) (defence stack) · [../../../11.system_design/09_tool_authorization_patterns.md](../../../11.system_design/09_tool_authorization_patterns.md) (capability isolation depth)

---

## Use Case

Defeats the exact attack module 15 demonstrates — the ladder README's cross-module check
for Block E.

The claim being tested: **only structural defences hold. Assume the model obeys the
attacker, and make that not matter.**

---

## The Mechanism

```
1 SPOTLIGHTING       wrap untrusted text, tell the model it is data
                     -> persuasion. A model that ignores your system prompt
                        ignores this too.

2 OUTPUT VALIDATION  check the result after the fact
                     -> too late. The write already fired.

3 CAPABILITY         reader agent's tool list does NOT contain the write tool
  ISOLATION          -> the payload is obeyed in spirit and reaches nothing.
                        STRUCTURAL.

4 DUAL-LLM (CaMeL)   quarantined LLM reads untrusted text, may only emit a
                     fixed Pydantic schema - never free text, never a tool call
                     -> untrusted text never reaches anything that can act.
                        STRUCTURAL.
```

---

## Key Implementation Details

**Defence 2 is described but deliberately not implemented.** Implementing it would
suggest it works. Its failure is temporal — the side effect has already happened by the
time you validate — so the honest demonstration is the comparison table.

**Defence 4 reuses module 03's strict schema.** That is the payoff of building structured
output early: it is the quarantine boundary here, not a convenience.

**`contains_suspected_instructions` is a field on the schema**, so the quarantined model
reports the attack as data rather than acting on it. That flag feeds module 14's audit
trail and module 37's red-team corpus.

**The reader agent's isolation is visible in one argument**: `openai_schemas(["fetch_customer_note"])`.
The write tool is absent from the list. That single line is the whole control.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Defence 3 should show `write tools fired: NONE — structurally impossible`. Defence 4
should print a typed object, with `contains_suspected_instructions` ideally `True`.
The comparison table is fixed text.

---

## How to Run

Open `16_injection_defences.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` only. Roughly 5 calls, under a cent.

---

## Next

`../F_observability/17_tracing_and_observability.ipynb`

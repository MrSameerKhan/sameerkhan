# Module 22 — Short-term Memory Strategies
Status: `🔧 Code-built`

Theory: [../../../8.agents/05_agent_memory.md](../../../8.agents/05_agent_memory.md) §2 (compaction strategies)

---

## Use Case

Module 02 established that memory is the list. The list grows without bound, so eventually you must throw some of it away. This module plants a client reference in turn 1 and asks for it in turn 8.

The claim being tested: **there is no free compaction — you are choosing which failure you get.**

---

## The Mechanism

```
full     everything      -> nothing forgotten, input grows every turn
window   last N messages -> forgets ABRUPTLY when a turn slides out
summary  precis + tail   -> forgets GRADUALLY, and can summarise wrongly
```

---

## Key Implementation Details

**The probe is a client reference, not a topic.** A summariser may keep the gist of a topic while losing an identifier, which is exactly the failure worth showing.

**The summary prompt has to beg for identifiers.** That instruction is load-bearing and it is still only a request — which is the argument for module 23.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `22_memory_short_term_strategies.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. Roughly 24 calls, a cent or two.

---

## Next

`23_memory_entity_keyvalue.ipynb`

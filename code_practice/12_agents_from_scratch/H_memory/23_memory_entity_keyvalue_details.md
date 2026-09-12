# Module 23 — Entity / Key-Value Memory
Status: `🔧 Code-built`

Theory: [../../../8.agents/05_agent_memory.md](../../../8.agents/05_agent_memory.md) §3 (semantic memory) · §4 (write/retrieve/forget)

---

## Use Case

The cheapest and most reliable memory tier, and the one most people skip on the way to vectors.

The claim being tested: **for exact, structured facts a dict beats a vector store on every axis that matters.**

---

## The Mechanism

```
STORE[entity][attribute] = {value, confidence, source}

extract  -> constrained Literal attributes + confidence threshold
read     -> O(1), exact, no model in the path
conflict -> 'latest wins' for preferences, 'source priority' for ground truth
```

---

## Key Implementation Details

**A closed `Literal` set of attributes** stops the store filling with invented fields. Writing too much pollutes retrieval, which is the month-three failure.

**Every fact carries value, confidence and source**, so 'why does the agent believe this?' is answerable — module 14's requirement applied to memory.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `23_memory_entity_keyvalue.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY`. Four extraction calls.

---

## Next

`24_memory_long_term_vector.ipynb`

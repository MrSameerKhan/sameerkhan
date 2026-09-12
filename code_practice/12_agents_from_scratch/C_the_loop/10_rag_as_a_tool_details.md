# Module 10 — RAG as a Tool
Status: `🔧 Code-built`

Theory: [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §7 (how LLM, RAG and agents connect; classic vs agentic RAG) · [../../../7.rag/01_rag.md](../../../7.rag/01_rag.md)

---

## Use Case

Connects the `7.rag` arc to the `8.agents` arc. Retrieval is one capability; the agent is
the loop that decides when to use it. This is the cleanest way to show you understand
both topics at once, which is why it recurs in interviews.

The claim being tested: **classic RAG is a workflow, agentic RAG is an agent, and the
difference is who decides to retrieve.**

---

## The Mechanism

```
CLASSIC   q -> retrieve(q) -> stuff context -> generate
          retrievals = 1, always, for every question

AGENTIC   q -> model sees search_policy + calculate as TOOLS
            -> may call neither, either, or one of them repeatedly
            -> may reformulate the query after a weak result
          retrievals = 0..N, decided at runtime
```

---

## Key Implementation Details

**The retriever is deliberately naive** — keyword overlap over the `_POLICY` dict in
`_tools.py`, no embeddings. Retrieval quality is the subject of `7.rag`; this module is
only about who invokes it. A good retriever here would distract from that.

**The arithmetic question is the control.** "What is 17 times 4?" has nothing to do with
mortgage policy, so classic RAG visibly retrieves irrelevant context while agentic RAG
reaches for `calculate` instead. Without a question outside the corpus the two look
identical.

**`search_policy` doubles as the retrieval tool.** Reusing the shared tool keeps this
module diffable against 08 and 09 — the machinery is the only difference.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

Classic RAG reports `retrievals=1` for both questions, including the arithmetic one.
Agentic RAG should show `['search_policy('erc')']` for the policy question and
`['calculate('17*4')']` — or no calls at all — for the arithmetic one. Year 3 ERC is 3%.

---

## How to Run

Open `10_rag_as_a_tool.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` only. Roughly 6-8 calls, under a cent.

---

## Next

`11_agent_cost_and_tokens.ipynb`

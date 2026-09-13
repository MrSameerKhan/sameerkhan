# Module 10 — RAG as a Tool
Status: `✅ Run` — with an instructive partial failure, see below

Theory: [../../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §7 (classic vs agentic RAG) · [../../../7.rag/01_rag.md](../../../7.rag/01_rag.md)

---

## Use Case

Connects the `7.rag` arc to the `8.agents` arc. Retrieval is one capability; the agent is
the loop that decides when to use it.

The claim being tested: **classic RAG is a workflow, agentic RAG is an agent, and the
difference is who decides to retrieve.**

---

## Key Implementation Details

**The retriever is deliberately naive** — keyword overlap over the `_POLICY` dict, no
embeddings. Retrieval quality belongs to `7.rag`; this module is about who invokes it.

**The arithmetic question is the control.** Without a question outside the corpus the two
approaches look identical.

---

## Fixes Applied (during run)

None. But see the partial failure below — it is worth keeping rather than fixing.

---

## Actual Output (macOS M1, `gpt-4.1-mini`, 2026-09-12)

**The control worked perfectly.** Classic RAG retrieved mortgage policy in order to answer
"What is 17 times 4?" — it has no choice. Agentic RAG skipped the corpus entirely and
reached for `calculate`:

```
CLASSIC   Q: What is 17 times 4?   retrievals=1   A: 17 times 4 is 68.
AGENTIC   Q: What is 17 times 4?   tool calls: ["calculate('17 * 4')"]
```

**The policy question went the other way, and that is the interesting part.** Classic RAG
answered it; agentic RAG did not:

```
CLASSIC   retrievals=1  A: The early repayment charge in year 3 is 3%.
AGENTIC   tool calls: ["search_policy('early repayment charge year 3')"]
          A: I could not find specific information for the early repayment charge
             in year 3. Could you please specify which...
```

The agent chose a natural-language query. The naive retriever only matches the literal
key `erc`, so it returned nothing — **and the agent did not reformulate.** Classic RAG
succeeded precisely because it never had to choose a query; it stuffed the whole top-k in.

**This is an honest result, not a broken demo.** The module's own lesson claims agentic
RAG "can reformulate a weak query" — *can*, not *will*. Notice what module 08 did in the
same situation: it got `No policy matched. Try one of: ltv, erc, ...` and retried. The
difference is that `search_policy` returns that helpful list while this module's
`retrieve()` returns silence. **A retriever that fails informatively lets an agent
recover; one that fails quietly does not.** That is the practical lesson, and it is
better than the one the module set out to teach.

---

## How to Run

Open `10_rag_as_a_tool.ipynb`, run all cells. Needs `OPENAI_API_KEY` only.

---

## Next

`11_agent_cost_and_tokens.ipynb`

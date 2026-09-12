# Module 18 — LangChain and LCEL
Status: `🔧 Code-built`

Theory: [../../../8.agents/03_langchain_primer.md](../../../8.agents/03_langchain_primer.md) §1-2 (LCEL, Runnables) · §7-8 (when not to use it, the honest critique)

---

## Use Case

LangChain is the most-used and most-criticised agent framework. This module extracts the one good idea and then shows the same job done without it, so the trade is explicit rather than tribal.

The claim being tested: **LCEL buys one uniform interface and integrations. It does not buy capability, and plain functions already compose.**

---

## The Mechanism

```
RunnableLambda(f) | RunnableLambda(g) | RunnableLambda(h)
         -> a new Runnable with .invoke / .batch / .stream / .ainvoke for free

RunnableParallel(a=..., b=..., original=RunnablePassthrough())
         -> fan out on one input, return a dict

plain(topic) = shorten(llm(template({"topic": topic})))
         -> identical output, three lines, no dependency
```

---

## Key Implementation Details

**`langchain_openai` is deliberately NOT imported.** The steps are `RunnableLambda`s
wrapping the raw SDK call from module 02. That is the honest picture: a framework composes
your code, it does not supply the capability. It also means this module needs no package
beyond `langchain-core`, which is already installed.

**`.batch` runs concurrently** with no extra code. That is the clearest single argument
for the Runnable protocol — module 06 needed `ThreadPoolExecutor` for the same effect.

**The `plain()` function is the punchline** and is deliberately last. Identical output,
no abstraction over the prompt, no 100-300ms overhead.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## How to Run

Open `18_langchain_lcel.ipynb`, select the `sameerkhan` kernel, run all cells. Needs `OPENAI_API_KEY` only. Roughly 7 calls, under a cent.

---

## Next

`19_langgraph_minimal_agent.ipynb`

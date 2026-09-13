# Module 07 — ReAct Without SDK Support
Status: `🔧 Code-built` — **re-run required after the fix below**

Theory: [../../../8.agents/01_agents.md](../../../8.agents/01_agents.md) (the ReAct loop) · [../../../8.agents/02_agent_reliability_patterns.md](../../../8.agents/02_agent_reliability_patterns.md) §2 (FM1)

---

## Use Case

Shows what an agent was before native tool calling, so module 08's `tools=[...]` reads as
a *removal* of work rather than magic.

The claim being tested: **without a structured tool-call channel the integration is a
regex, and its reliability is the model's willingness to hold a text format.**

---

## The Mechanism

```
prompt teaches a TEXT FORMAT (Thought / Action / Action Input)
loop: call model with the whole transcript
      stop=["Observation:"]     <- stop it before it invents the tool result
      ACTION_RE.search(text)    <- YOU mine the tool call out of prose
      run tool, append "Observation: ..."
until a Final Answer, or the parser misses
```

---

## Fixes Applied (during run)

| Found | Fix |
|---|---|
| **The loop reported success on an answer computed from nothing.** `gpt-4.1-mini` emitted an `Action` *and* a `Final Answer` in the same turn. The code checked `FINAL_RE` **before** `ACTION_RE`, so it returned the Final Answer — which was the literal string **"The early repayment charge in year 2 on a 250000 loan is `[calculated amount]`"** — and reported `ok=True, steps=1`. No tool ever ran. | An `Action` now takes precedence over a `Final Answer` found beside it: both are matched, and the answer is only accepted when there is no pending action. |

**This is the module's own lesson landing on the module.** FM1 is "the model called a tool
in prose and your orchestrator could not see it". Here the orchestrator *did* see the
action and discarded it in favour of a placeholder — and reported success. `stop=["Observation:"]`
did not help, because the model never wrote the word Observation.

---

## Actual Output (macOS M1, 2026-09-12) — **before the fix**

```
gpt-4.1-mini  parsed=True    steps=1     <- FALSE PASS, see above
llama3.2:1b   parsed=False   steps=1
```

The 1B behaved exactly as intended — it wrote a paragraph of prose with no `Action:` line
at all, and the parser missed:

```
To find the early repayment charge (ERC) on a $250,000 loan with interest, I will
first search for the bank's loan policy...
  !! PARSER MISS — no Action/Action Input found
```

That is FM1, live, on the model kept specifically to produce it.

---

## How to Run

Open `07_agent_without_sdk_react.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` and Ollama with **both** `llama3.2` and `llama3.2:1b`.

Expected after the fix: `gpt-4.1-mini` should take 3 steps and reach **10000**
(250000 x 4% for year 2). The 1B should still miss.

---

## Next

`08_agent_with_sdk_tool_calling.ipynb`

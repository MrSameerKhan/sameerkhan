# Module 07 — ReAct Without SDK Support
Status: `🔧 Code-built`

Theory: [../../../8.agents/01_agents.md](../../../8.agents/01_agents.md) (the ReAct loop) · [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md) §1 · [../../../8.agents/02_agent_reliability_patterns.md](../../../8.agents/02_agent_reliability_patterns.md) §2 (FM1)

---

## Use Case

Shows what an agent was before providers shipped native tool calling, so that module
08's `tools=[...]` parameter reads as a *removal* of work rather than magic.

Two claims tested: **the model now chooses the sequence**, and **without a structured
tool-call channel the integration is a regex whose reliability is the model's.**

---

## The Mechanism

```
prompt teaches a TEXT FORMAT:
    Thought: ...
    Action: search_policy
    Action Input: erc

loop:
    call model with the whole transcript
    stop=["Observation:"]        <- stop the model before it invents the tool result
    ACTION_RE.search(text)       <- YOU mine the tool call out of prose
    run the tool
    append "Observation: ..." to the transcript
until FINAL_RE matches, or the parser misses
```

---

## Key Implementation Details

**`stop=["Observation:"]`** is load-bearing. Without it the model happily writes the
Observation *itself*, hallucinating the tool result, and the loop never executes a real
tool. This is the single most common bug when hand-rolling ReAct.

**The whole transcript is one user message**, rebuilt each turn. That is module 02's
lesson: the list is the memory, and here it is literally a growing string.

**The 1B override is scoped to one cell** and put back immediately, so later modules
still get the 3B. Per `_providers.py`, changing the model is a one-line change.

**A parser miss returns `ok=False` rather than raising.** That is the honest simulation:
in production FM1 does not crash, it silently produces an answer with no tool behind it.

---

## Fixes Applied (during run)

*Not yet run.*

---

## Actual Output

*Not yet run.*

---

## Expected Output

`gpt-4.1-mini` should hold the format, call `search_policy("erc")`, then `calculate`, and
reach a Final Answer in 3 steps. Year 2 ERC is 4%, so 250000 x 0.04 = **10000**.

The 1B is expected to fail, most likely with a parser miss. If it happens to succeed,
re-run it — the point is the variance, and that variance IS the failure mode.

Needs `ollama serve` with **both** `llama3.2` and `llama3.2:1b` pulled.

---

## How to Run

Open `07_agent_without_sdk_react.ipynb`, select the `sameerkhan` kernel, run all cells.
Needs `OPENAI_API_KEY` and Ollama. Under a cent.

---

## Next

`08_agent_with_sdk_tool_calling.ipynb`

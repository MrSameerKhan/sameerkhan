# Session 1 — ReAct Agent from Scratch
Status: `✅ Run`

Theory: [../../../8.agents/01_agents.md](../../../8.agents/01_agents.md) · [../../../8.agents/06_planner_executor_patterns.md](../../../8.agents/06_planner_executor_patterns.md)

---

## Use Case

Multi-step research assistant: a single LLM call can't search, then calculate, then synthesize. The ReAct loop can — it runs as many rounds as needed to answer correctly.

---

## The Loop

```
User question
    │
    LLM generates:
    Thought: I need to look up the 5-year fixed rate, then calculate the payment.
    Action: lookup_rate
    Action Input: 5-year fixed
    │
    System executes tool → appends Observation:
    Observation: 5-year fixed: 4.61% fixed
    │
    LLM continues:
    Thought: Now I can calculate. £250,000 over 25 years at 4.61%.
    Action: calculate
    Action Input: 250000 * (0.0461/12) * (1 + 0.0461/12)**300 / ((1 + 0.0461/12)**300 - 1)
    │
    Observation: 1395.23
    │
    Thought: I have enough to answer.
    Final Answer: Monthly payment: £1,395.23...
```

---

## Key Implementation Details

**Parsing with regex:** ReAct from scratch requires parsing the LLM output:
```python
thought      = re.search(r"Thought:\s*(.+?)(?=\nAction)", text, re.S)
action       = re.search(r"Action:\s*(\w+)", text)
action_input = re.search(r"Action Input:\s*(.+?)(?=\n|$)", text, re.S)
final        = re.search(r"Final Answer:\s*(.+)", text, re.S)
```

**Why OpenAI tool calling (session 02) is better:** no regex, no parsing, structured JSON — the LLM returns a typed `tool_calls` object. ReAct from scratch is educational; in production, use function calling.

**Max iterations guard:** without `MAX_ITERATIONS = 8`, a confused LLM can loop forever. Always cap.

---

## Fixes Applied (bugs found during run)

1. **Quoted action inputs**: LLM wrapped inputs in quotes → `calculate('"expr"')` evaluated to a string, `round(string)` crashed. Fix: `clean()` helper strips surrounding quotes in `parse_action`.
2. **search_policy too strict**: LLM used natural phrases ("minimum deposit first time buyer"), DB keys are acronyms ("ltv"). Fix: `POLICY_ALIASES` dict maps common phrases to DB keys before keyword scan.
3. **Missing Action Input**: LLM occasionally wrote `lookup_rate(None)` inline instead of using the `Action Input:` line. Fix: guard in `run_react` returns helpful error message instead of crashing.

## Actual Output (Windows, gpt-4o-mini, 2026-06-25)

```
Q1: Monthly payment £250k / 25yr / 5-year fixed
  Step 1: lookup_rate → 4.61%
  Step 2: calculate → 1405.236286
  Final Answer: ~£1,405.24 ✓

Q2: First-time buyer £320k — deposit + documents
  Step 1: search_policy(deposit) → LTV 95% for first-time buyers
  Step 2: calculate(320000 * 0.05) → £16,000
  Final Answer: £16,000 deposit ✓ (docs from general knowledge — policy alias overlap)

Q3: Max personal finance SAR 12,000 salary
  Step 1: search_policy → 36× salary for nationals
  Final Answer: SAR 432,000 ✓
```

All 3 questions answered correctly. Q1 took 4 steps (1 bad step from missing Action Input, then recovered).

---

## Expected Output

```
Question: What would the monthly payment be on a £250,000 mortgage over 25 years
          at the 5-year fixed rate?

── Step 1 ──
Thought: I need to find the 5-year fixed rate first.
Action: lookup_rate(product='5-year fixed')
Observation: 5-year fixed: 4.61% fixed

── Step 2 ──
Thought: Now calculate monthly payment: £250,000, 4.61%, 300 months.
Action: calculate(expression='250000 * (0.0461/12) * (1+0.0461/12)**300 / ((1+0.0461/12)**300 - 1)')
Observation: 1395.23

── Step 3 ──
Thought: I have all the information needed.
Final Answer: The monthly payment on a £250,000 mortgage over 25 years at the
5-year fixed rate of 4.61% would be approximately £1,395.23.
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/08_agents
python 01_react_agent.py
```

Cost: ~$0.03 per run (3 questions × 3-4 LLM calls each).

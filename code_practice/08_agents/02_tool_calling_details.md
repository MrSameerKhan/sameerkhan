# Session 2 — OpenAI Function Calling
Status: `✅ Run`

Theory: [../../../8.agents/01_agents.md](../../8.agents/01_agents.md)

---

## Use Case

Financial data agent: connect the LLM to live tools (mortgage calculator, exchange rates, account lookup, eligibility checker) without writing a custom parser. OpenAI handles the JSON schema and argument extraction.

---

## How Function Calling Works

```
1. Send messages + tools (JSON schemas) to OpenAI
2. LLM decides which tool(s) to call → returns tool_calls (not text)
3. Execute each tool → get result
4. Append tool results to messages (role="tool")
5. Send again → LLM generates final text answer
```

**Key difference from ReAct:** no text parsing. The LLM returns a structured object:
```python
msg.tool_calls[0].function.name      # "get_mortgage_payment"
msg.tool_calls[0].function.arguments  # '{"principal": 280000, "annual_rate_pct": 4.61, "months": 300}'
```

**Parallel tool calls:** for the query "What are my balances in SAR?", the LLM calls `get_account_balance(ACC001)` and `get_account_balance(ACC002)` AND `get_exchange_rate(GBP, SAR)` simultaneously in one round-trip.

---

## Tool Schema Pattern

```python
{
    "type": "function",
    "function": {
        "name": "get_mortgage_payment",
        "description": "Calculate monthly payment, total paid, total interest",
        "parameters": {
            "type": "object",
            "properties": {
                "principal":       {"type": "number",  "description": "Loan amount"},
                "annual_rate_pct": {"type": "number",  "description": "Annual rate % e.g. 4.61"},
                "months":          {"type": "integer", "description": "Term in months"},
            },
            "required": ["principal", "annual_rate_pct", "months"],
        },
    },
}
```

**Description is critical:** the LLM reads the description to decide which tool to call. Be specific about units (percentage not decimal), field types, and examples.

---

## Actual Output (Windows, gpt-4o-mini, 2026-06-25)

```
Q1 (monthly payment £280k / 25yr / 4.61%):
  → get_mortgage_payment({principal:280000, rate:4.61, months:300})
  Answer: £1,573.86/month, total £472,159.39 ✓

Q2 (balances ACC001 + ACC002):
  → get_account_balance(ACC001) + get_account_balance(ACC002)  ← parallel
  Answer: ACC001 = 45,230.5 SAR, ACC002 = 12,800.0 GBP ✓

Q3 (eligibility + payment £270k on £300k property):
  → check_eligibility(...) + get_mortgage_payment(...)  ← parallel
  Answer: eligible (LTV 90%, DSR 5.45%), monthly £1,534.66 ✓
```

- Parallel tool calls confirmed on Q2 and Q3 — both tools called in single round-trip
- No parsing errors — structured JSON arguments extracted cleanly by OpenAI function calling

---

## Expected Output

```
Q: A customer wants to borrow £270,000 on a £300,000 property. Monthly income £5,500,
   existing debts £300/month. Eligible? If yes, monthly payment at 4.72% over 25 years?

  → check_eligibility({'loan_amount': 270000, 'property_value': 300000,
                        'monthly_income': 5500, 'monthly_debts': 300,
                        'is_first_time_buyer': False})
  → get_mortgage_payment({'principal': 270000, 'annual_rate_pct': 4.72, 'months': 300})

A: The customer is eligible for the mortgage:
   • LTV: 90.0% — exactly at the 90% limit for standard borrowers
   • Debt-service ratio: 5.5% — well within the 45% limit
   
   Monthly payment at 4.72% over 25 years: £1,517.43
   Total paid: £455,229 | Total interest: £185,229
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/08_agents
python 02_tool_calling.py
```

Cost: ~$0.04 per run (3 queries × 2-3 LLM calls each + tool calls).

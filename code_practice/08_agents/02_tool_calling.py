# Session 2 — OpenAI Function Calling (Structured Tool Use)
# Task   : financial data agent — LLM calls tools to answer multi-part queries
# Shows  : tool schema definition, tool-call loop, parallel tool calls, error handling
# Differs from session 01: OpenAI handles parsing — no regex, no format enforcement
#
# Set OPENAI_API_KEY before running.

import os
import json
import math
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o-mini"


# ── Tool implementations ───────────────────────────────────────────────────────

def get_mortgage_payment(principal: float, annual_rate_pct: float, months: int) -> dict:
    """Monthly payment using the standard amortisation formula."""
    r = annual_rate_pct / 100 / 12
    if r == 0:
        payment = principal / months
    else:
        payment = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    total    = payment * months
    interest = total - principal
    return {
        "monthly_payment":   round(payment, 2),
        "total_paid":        round(total, 2),
        "total_interest":    round(interest, 2),
        "principal":         principal,
        "rate_pct":          annual_rate_pct,
        "months":            months,
    }


def get_exchange_rate(from_currency: str, to_currency: str) -> dict:
    """Mock exchange rates — replace with real FX API in production."""
    rates = {
        ("GBP", "SAR"): 4.74, ("SAR", "GBP"): 0.211,
        ("USD", "SAR"): 3.75, ("SAR", "USD"): 0.267,
        ("EUR", "SAR"): 4.07, ("SAR", "EUR"): 0.246,
        ("GBP", "USD"): 1.265,("USD", "GBP"): 0.791,
    }
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if not rate:
        return {"error": f"Rate for {from_currency}/{to_currency} not available."}
    return {"from": from_currency.upper(), "to": to_currency.upper(), "rate": rate}


def get_account_balance(account_id: str) -> dict:
    """Mock account lookup — replace with real DB query in production."""
    accounts = {
        "ACC001": {"balance": 45230.50, "currency": "SAR", "type": "current",  "status": "active"},
        "ACC002": {"balance": 12800.00, "currency": "GBP", "type": "savings",  "status": "active"},
        "ACC003": {"balance":  3400.00, "currency": "USD", "type": "current",  "status": "active"},
    }
    if account_id not in accounts:
        return {"error": f"Account {account_id} not found."}
    return {"account_id": account_id, **accounts[account_id]}


def check_eligibility(
    loan_amount: float,
    property_value: float,
    monthly_income: float,
    monthly_debts: float,
    is_first_time_buyer: bool,
) -> dict:
    """Check mortgage eligibility against policy rules."""
    ltv          = loan_amount / property_value * 100
    max_ltv      = 95.0 if is_first_time_buyer else 90.0
    dsr          = (monthly_debts / monthly_income) * 100   # debt-service ratio
    max_dsr      = 45.0

    issues = []
    if ltv > max_ltv:
        issues.append(f"LTV {ltv:.1f}% exceeds maximum {max_ltv}% — deposit too small")
    if dsr > max_dsr:
        issues.append(f"Debt-service ratio {dsr:.1f}% exceeds 45% affordability limit")

    return {
        "eligible":         len(issues) == 0,
        "ltv_pct":          round(ltv, 2),
        "max_ltv_pct":      max_ltv,
        "debt_service_pct": round(dsr, 2),
        "max_dsr_pct":      max_dsr,
        "issues":           issues,
        "recommendation":   "Eligible for mortgage" if not issues else "; ".join(issues),
    }


# ── Tool schemas (JSON Schema for OpenAI function calling) ────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_mortgage_payment",
            "description": "Calculate monthly mortgage payment, total paid, and total interest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal":       {"type": "number",  "description": "Loan amount in currency"},
                    "annual_rate_pct": {"type": "number",  "description": "Annual interest rate as a percentage (e.g. 4.61)"},
                    "months":          {"type": "integer", "description": "Loan term in months"},
                },
                "required": ["principal", "annual_rate_pct", "months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get the current exchange rate between two currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "ISO currency code (e.g. GBP, SAR, USD)"},
                    "to_currency":   {"type": "string", "description": "ISO currency code"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_balance",
            "description": "Retrieve account balance and status for a given account ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {"type": "string", "description": "Account ID (e.g. ACC001)"},
                },
                "required": ["account_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "Check if a customer is eligible for a mortgage based on LTV and affordability.",
            "parameters": {
                "type": "object",
                "properties": {
                    "loan_amount":          {"type": "number",  "description": "Loan amount requested"},
                    "property_value":       {"type": "number",  "description": "Property valuation"},
                    "monthly_income":       {"type": "number",  "description": "Gross monthly income"},
                    "monthly_debts":        {"type": "number",  "description": "Existing monthly debt payments"},
                    "is_first_time_buyer":  {"type": "boolean", "description": "Whether applicant is a first-time buyer"},
                },
                "required": ["loan_amount", "property_value", "monthly_income", "monthly_debts", "is_first_time_buyer"],
            },
        },
    },
]

TOOL_MAP = {
    "get_mortgage_payment": get_mortgage_payment,
    "get_exchange_rate":    get_exchange_rate,
    "get_account_balance":  get_account_balance,
    "check_eligibility":    check_eligibility,
}


# ── Tool-call loop ────────────────────────────────────────────────────────────

def run_agent(question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": "You are a financial assistant. Use tools to answer accurately. For multi-part questions, call multiple tools."},
        {"role": "user",   "content": question},
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL, temperature=0, tools=TOOLS, messages=messages
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            return msg.content   # no more tool calls → final answer

        # Execute all tool calls (may be parallel)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args    = json.loads(tc.function.arguments)

            if verbose:
                print(f"  → {fn_name}({args})")

            result = TOOL_MAP[fn_name](**args)

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      json.dumps(result),
            })


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    queries = [
        "What would be the monthly payment for a £280,000 mortgage over 25 years at 4.61% annual rate?",
        "I have accounts ACC001 and ACC002. What are the balances in SAR?",
        "A customer wants to borrow £270,000 on a £300,000 property. Their monthly income is £5,500 and they have £300 in existing monthly debt payments. Is this eligible? If yes, what would the monthly payment be at 4.72% over 25 years?",
    ]

    for q in queries:
        print(f"\n{'═' * 68}")
        print(f"Q: {q}\n")
        answer = run_agent(q, verbose=True)
        print(f"\nA: {answer}")


if __name__ == "__main__":
    main()

# tools.py — Tool implementations for the LangGraph agent (session 03)

import math
from langchain_core.tools import tool

POLICY_DB = {
    "ltv":           "Maximum LTV: 90% standard borrowers, 95% first-time buyers (Help to Buy).",
    "erc":           "Early repayment charges: 5% yr1, 4% yr2, 3% yr3, 2% yr4, 1% yr5. None after fixed period.",
    "income":        "Required: payslips (3 months), SA302 (self-employed, 2 yrs), pension statements (retired).",
    "affordability": "Monthly debts ≤ 45% of gross monthly income. Stress-tested at rate + 3%.",
    "personal":      "Personal finance: max 36× salary (nationals) / 24× (expatriates). Max tenor 60/36 months.",
    "rates":         "SVR: 7.24%. Fixed terms: 2yr 4.85%, 3yr 4.72%, 5yr 4.61%, 10yr 4.89%.",
    "savings":       "Savings: 2.8% p.a. Accounts < SAR 500 → SAR 10/month maintenance fee.",
    "credit":        "Credit card limit: 3× salary (entry), up to 12× (premium). Min salary SAR 5,000.",
}


POLICY_ALIASES = {
    "early repayment": "erc",
    "overpay":         "erc",
    "deposit":         "ltv",
    "first time":      "ltv",
    "loan to value":   "ltv",
    "document":        "income",
    "payslip":         "income",
    "proof":           "income",
    "debt":            "affordability",
    "expatriate":      "personal",
    "expat":           "personal",
    "national":        "personal",
    "fixed":           "rates",
    "tracker":         "rates",
    "svr":             "rates",
}


@tool
def search_policy(query: str) -> str:
    """Search bank policy documents for eligibility rules, LTV limits, and documentation requirements."""
    q = query.lower()
    for alias, key in POLICY_ALIASES.items():
        if alias in q:
            return POLICY_DB[key]
    for keyword, answer in POLICY_DB.items():
        if keyword in q or any(w in q for w in keyword.split("_")):
            return answer
    return "No specific policy found. Rephrase with keywords: ltv, erc, income, affordability, personal, rates, savings, credit."


@tool
def calculate_mortgage_payment(principal: float, annual_rate_pct: float, months: int) -> str:
    """Calculate the monthly mortgage payment given principal, annual rate percentage, and term in months."""
    r = annual_rate_pct / 100 / 12
    if r == 0:
        payment = principal / months
    else:
        payment = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
    return (
        f"Monthly payment: £{payment:,.2f} | "
        f"Total paid: £{payment * months:,.2f} | "
        f"Total interest: £{(payment * months - principal):,.2f}"
    )


@tool
def check_ltv(loan_amount: float, property_value: float, is_first_time_buyer: bool = False) -> str:
    """Check if a loan-to-value ratio is within policy limits."""
    ltv     = loan_amount / property_value * 100
    max_ltv = 95.0 if is_first_time_buyer else 90.0
    status  = "APPROVED" if ltv <= max_ltv else "REJECTED"
    return (
        f"LTV: {ltv:.1f}% | Max allowed: {max_ltv:.1f}% | "
        f"Status: {status} | "
        f"{'Eligible.' if ltv <= max_ltv else f'Deposit shortfall: £{(ltv - max_ltv) / 100 * property_value:,.0f}'}"
    )


TOOLS = [search_policy, calculate_mortgage_payment, check_ltv]

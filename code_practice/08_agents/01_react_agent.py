# Session 1 — ReAct Agent from Scratch
# Task   : multi-step research assistant — search + calculate + synthesize in a loop
# Pattern: Thought → Action → Observation → ... → Final Answer
# Shows  : how agents work before using a framework — build the loop manually
#
# ReAct paper: Yao et al. 2022 — "ReAct: Synergizing Reasoning and Acting in LLMs"
# Set OPENAI_API_KEY before running.

import os
import re
import json
import math
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o-mini"

MAX_ITERATIONS = 8


# ── Tools ─────────────────────────────────────────────────────────────────────

POLICY_DB = {
    "ltv":           "Maximum LTV is 90% for standard borrowers and 95% for first-time buyers.",
    "erc":           "Early repayment charges: 5% year 1, 4% year 2, 3% year 3, 2% year 4, 1% year 5.",
    "income":        "Proof of income required: payslips (3 months), SA302 (self-employed), pension statements (retired).",
    "affordability": "Total monthly debts must not exceed 45% of gross monthly income at a stressed rate 3% above product rate.",
    "tenor":         "Mortgage tenors: 5, 10, 15, 20, 25, 30 years. Maximum age at end of term: 75.",
    "personal":      "Personal finance: max 36x monthly salary for nationals, 24x for expatriates. Max tenor 60 months.",
    "rates":         "SVR currently 7.24%. Fixed-rate products: 2, 3, 5, 10 year terms available.",
    "savings":       "Savings account returns 2.8% per annum. Accounts below SAR 500 pay SAR 10 monthly maintenance fee.",
}


POLICY_ALIASES = {
    "deposit":    "ltv",
    "first-time": "ltv",
    "first time": "ltv",
    "loan to value": "ltv",
    "document":   "income",
    "payslip":    "income",
    "proof":      "income",
    "early repayment": "erc",
    "overpay":    "erc",
    "debt":       "affordability",
    "national":   "personal",
    "expatriate": "personal",
    "expat":      "personal",
    "fixed":      "rates",
    "tracker":    "rates",
}


def search_policy(query: str) -> str:
    """Search bank policy documents for the given query."""
    q = query.lower()
    for alias, key in POLICY_ALIASES.items():
        if alias in q:
            return POLICY_DB[key]
    for keyword, answer in POLICY_DB.items():
        if keyword in q or any(w in q for w in keyword.split()):
            return answer
    return f"No policy found. Try one of these keywords: {', '.join(POLICY_DB.keys())}"


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression."""
    try:
        # Allow only safe math operations
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(round(result, 6))
    except Exception as e:
        return f"Calculation error: {e}"


def lookup_rate(product: str) -> str:
    """Look up the current rate for a product."""
    if not product:
        return "Please specify a product. Available: 2-year fixed, 3-year fixed, 5-year fixed, 10-year fixed, tracker, svr, savings, personal"
    rates = {
        "2-year fixed":  "4.85% fixed",
        "3-year fixed":  "4.72% fixed",
        "5-year fixed":  "4.61% fixed",
        "10-year fixed": "4.89% fixed",
        "tracker":       "Bank of England base rate + 0.99% (currently 6.24%)",
        "svr":           "7.24%",
        "savings":       "2.8% per annum",
        "personal":      "5.5% to 9.5% APR depending on salary band and credit profile",
    }
    p = product.lower()
    for k, v in rates.items():
        if k in p or p in k:
            return f"{product}: {v}"
    return f"Rate for '{product}' not found. Available: {', '.join(rates.keys())}"


TOOLS = {
    "search_policy": search_policy,
    "calculate":     calculate,
    "lookup_rate":   lookup_rate,
}

TOOL_DESCRIPTIONS = """
- search_policy(query: str) → str
    Search bank policy documents. Use for eligibility rules, LTV limits, documentation requirements.
    Example: search_policy("maximum LTV first time buyer")

- calculate(expression: str) → str
    Evaluate a math expression. Supports standard Python math operators and math module functions.
    Example: calculate("200000 * 0.0485 / 12 / (1 - (1 + 0.0485/12)**-240)")

- lookup_rate(product: str) → str
    Look up current rates for mortgage or savings products.
    Example: lookup_rate("5-year fixed")
"""

SYSTEM_PROMPT = f"""You are a financial assistant that answers customer questions using tools.

Respond using EXACTLY this format:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Action Input: [the single string argument to pass to the tool]

When you have enough information to answer:
Thought: I now have all the information needed.
Final Answer: [your complete answer to the customer]

Rules:
- Always use Thought before Action
- Action must be one of: search_policy, calculate, lookup_rate
- Action Input is a plain string (not JSON)
- Use Final Answer only when the question is fully answered
- If a tool returns an error, try a different approach

Available tools:{TOOL_DESCRIPTIONS}"""


# ── ReAct loop ────────────────────────────────────────────────────────────────

def parse_action(text: str) -> tuple[str | None, str | None, str | None]:
    """Extract (thought, action, action_input) from LLM output. Returns Final Answer too."""
    thought      = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer))", text, re.S)
    action       = re.search(r"Action:\s*(\w+)", text)
    action_input = re.search(r"Action Input:\s*(.+?)(?=\n|$)", text, re.S)
    final        = re.search(r"Final Answer:\s*(.+)", text, re.S)

    def clean(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
            s = s[1:-1]
        return s

    if final:
        return (thought.group(1).strip() if thought else ""), "FINAL", final.group(1).strip()
    return (
        thought.group(1).strip() if thought else None,
        action.group(1).strip() if action else None,
        clean(action_input.group(1)) if action_input else None,
    )


def run_react(question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": question},
    ]

    for step in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL, temperature=0, messages=messages
        )
        text = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": text})

        thought, action, action_input = parse_action(text)

        if verbose:
            print(f"\n── Step {step} ──")
            if thought:
                print(f"Thought: {thought}")

        if action == "FINAL":
            if verbose:
                print(f"Final Answer: {action_input}")
            return action_input

        if action not in TOOLS:
            observation = f"Unknown tool '{action}'. Available: {list(TOOLS.keys())}"
        elif not action_input:
            observation = "Action Input was missing. Add 'Action Input: <value>' on its own line."
        else:
            if verbose:
                print(f"Action: {action}({action_input!r})")
            observation = TOOLS[action](action_input)

        if verbose:
            print(f"Observation: {observation}")

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Max iterations reached without a final answer."


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    questions = [
        "What would the monthly payment be on a £250,000 mortgage over 25 years at the 5-year fixed rate?",
        "I'm a first-time buyer wanting to buy a £320,000 property. What's the minimum deposit I need, and what documents will the bank require?",
        "What is the maximum personal finance amount for a Saudi national with a monthly salary of SAR 12,000?",
    ]

    for q in questions:
        print(f"\n{'═' * 68}")
        print(f"Question: {q}")
        answer = run_react(q, verbose=True)
        print(f"\n{'─' * 68}")
        print(f"Answer: {answer}")


if __name__ == "__main__":
    main()

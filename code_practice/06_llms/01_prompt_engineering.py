# Session 1 — Prompt Engineering
# Task   : domain-aware customer support bot for a bank
# Model  : gpt-4o-mini (OpenAI API) — swap for any OpenAI-compatible endpoint
# Shows  : zero-shot → few-shot → CoT progression on identical queries
# Metric : qualitative comparison + response length and confidence tracking
#
# Set OPENAI_API_KEY in your environment before running.

import os
import time
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

MODEL      = "gpt-4o-mini"
TEMPERATURE = 0.2   # low for deterministic support responses


# ── Prompting strategies ───────────────────────────────────────────────────────

def zero_shot(query: str) -> str:
    """No examples — model uses general knowledge only."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": "You are a helpful banking assistant."},
            {"role": "user",   "content": query},
        ],
    )
    return response.choices[0].message.content


def few_shot(query: str) -> str:
    """
    3 domain-specific examples in the system prompt.
    Shows the model the exact format, tone, and depth expected.
    """
    examples = """
You are a customer support specialist for Al Rajhi Bank. Answer clearly and specifically.
Always mention the relevant product, include actual figures where known, and end with
a next step the customer can take.

Example 1:
Q: What documents do I need for a home loan?
A: For an Al Rajhi home finance application you will need: (1) National ID or Iqama,
(2) salary certificate or employment letter, (3) last 3 months' bank statements,
(4) property valuation report, and (5) down payment proof (minimum 10% of property value).
Visit any branch or apply via the mobile app to start the process.

Example 2:
Q: What is the profit rate on a personal finance product?
A: Al Rajhi personal finance is Sharia-compliant. The profit rate (not interest) ranges
from 5.5% to 9.5% annually depending on your salary, term, and credit profile.
Calculate your monthly instalment using the finance calculator in our mobile app.

Example 3:
Q: Can I open a savings account online?
A: Yes. Download the Al Rajhi mobile app, tap "Open Account", complete the biometric
ID verification, and your account is live within 10 minutes. No branch visit required.
""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": examples},
            {"role": "user",   "content": query},
        ],
    )
    return response.choices[0].message.content


def chain_of_thought(query: str) -> str:
    """
    Instruct the model to reason through the answer before responding.
    Most effective for complex multi-part questions or policy lookups.
    """
    system = """You are an Al Rajhi Bank specialist. For each customer question:
1. First identify what the customer actually needs (restated simply)
2. List the key facts relevant to their question
3. Compose a clear, specific answer

Use this format:
NEED: [one sentence]
FACTS: [bullet points]
ANSWER: [response to customer]"""

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
    )
    return response.choices[0].message.content


def self_consistency(query: str, n: int = 3) -> str:
    """
    Sample n independent answers and return the one that appears most consistent.
    For factual QA: pick the answer whose key claim appears in ≥ 2 responses.
    Simplified here: return all n answers for comparison.
    """
    responses = []
    for _ in range(n):
        responses.append(few_shot(query))
        time.sleep(0.5)   # avoid rate limiting
    return responses


# ── Structured output (JSON mode) ─────────────────────────────────────────────
def json_response(query: str) -> str:
    """Force JSON output — reliable for downstream parsing."""
    system = """You are an Al Rajhi Bank assistant. Respond ONLY in valid JSON:
{
  "answer": "...",
  "confidence": "high|medium|low",
  "next_step": "...",
  "escalate_to_human": true|false
}"""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": query},
        ],
    )
    return response.choices[0].message.content


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    queries = [
        "How do I increase my credit card limit?",
        "My salary transfer was delayed — what should I do?",
    ]

    for query in queries:
        print(f"\n{'═' * 70}")
        print(f"QUERY: {query}")
        print("═" * 70)

        print("\n[Zero-shot]")
        print(zero_shot(query))

        print("\n[Few-shot]")
        print(few_shot(query))

        print("\n[Chain-of-Thought]")
        print(chain_of_thought(query))

        print("\n[JSON mode]")
        print(json_response(query))

        time.sleep(1)   # rate limit buffer

    # Decision tree: which technique to use
    print("\n\n── Technique decision guide ──")
    guide = {
        "Simple FAQ (account balance, branch hours)":    "zero-shot",
        "Domain format critical (regulatory, compliance)": "few-shot (3-5 domain examples)",
        "Complex policy lookup (multi-condition eligibility)": "chain-of-thought",
        "High-stakes decision (loan approval, fraud query)": "self-consistency (vote N=3-5)",
        "Downstream parsing needed (API, database write)": "json_mode",
    }
    for task, technique in guide.items():
        print(f"  {task}")
        print(f"    → {technique}\n")


if __name__ == "__main__":
    main()

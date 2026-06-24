# Session 1 — Prompt Engineering
# Task   : domain-aware customer support bot for a bank
# Shows  : zero-shot -> few-shot -> CoT -> JSON mode progression on identical queries
#
# Change PROVIDER to switch backends. Nothing else needs to change.
#   "openai"  → needs OPENAI_API_KEY env var
#   "claude"  → needs ANTHROPIC_API_KEY env var
#   "ollama"  → needs Ollama running locally (ollama serve)

import os
import time

PROVIDER = "ollama"   # "openai" | "claude" | "ollama"

if PROVIDER == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    MODEL  = "gpt-4o-mini"

elif PROVIDER == "claude":
    import anthropic
    client = anthropic.Anthropic()
    MODEL  = "claude-haiku-4-5-20251001"

elif PROVIDER == "ollama":
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    MODEL  = "llama3.2"


def ask(system: str, query: str, json_mode: bool = False) -> str:
    if PROVIDER == "claude":
        sys_text = system + ("\n\nReturn valid JSON only. No markdown." if json_mode else "")
        r = client.messages.create(
            model=MODEL, max_tokens=512, temperature=0.2,
            system=sys_text,
            messages=[{"role": "user", "content": query}],
        )
        return r.content[0].text
    else:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = client.chat.completions.create(
            model=MODEL, max_tokens=512, temperature=0.2,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": query}],
            **extra,
        )
        return r.choices[0].message.content


# ── Prompting strategies ───────────────────────────────────────────────────────

def zero_shot(query: str) -> str:
    return ask("You are a helpful banking assistant.", query)


def few_shot(query: str) -> str:
    system = """You are a customer support specialist for Al Rajhi Bank. Answer clearly and specifically.
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
ID verification, and your account is live within 10 minutes. No branch visit required."""
    return ask(system, query)


def chain_of_thought(query: str) -> str:
    system = """You are an Al Rajhi Bank specialist. For each customer question:
1. First identify what the customer actually needs (restated simply)
2. List the key facts relevant to their question
3. Compose a clear, specific answer

Use this format:
NEED: [one sentence]
FACTS: [bullet points]
ANSWER: [response to customer]"""
    return ask(system, query)


def self_consistency(query: str, n: int = 3) -> list:
    responses = []
    for _ in range(n):
        responses.append(few_shot(query))
        time.sleep(0.3)
    return responses


def json_response(query: str) -> str:
    system = """You are an Al Rajhi Bank assistant. Respond ONLY in valid JSON:
{
  "answer": "...",
  "confidence": "high|medium|low",
  "next_step": "...",
  "escalate_to_human": true|false
}"""
    return ask(system, query, json_mode=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Provider: {PROVIDER} | Model: {MODEL}\n")

    queries = [
        "How do I increase my credit card limit?",
        "My salary transfer was delayed -- what should I do?",
    ]

    for query in queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")
        print("=" * 70)

        print("\n[Zero-shot]")
        print(zero_shot(query))

        print("\n[Few-shot]")
        print(few_shot(query))

        print("\n[Chain-of-Thought]")
        print(chain_of_thought(query))

        print("\n[JSON mode]")
        print(json_response(query))

        time.sleep(0.5)

    print("\n\n-- Technique decision guide --")
    guide = {
        "Simple FAQ (account balance, branch hours)":           "zero-shot",
        "Domain format critical (regulatory, compliance)":      "few-shot (3-5 domain examples)",
        "Complex policy lookup (multi-condition eligibility)":  "chain-of-thought",
        "High-stakes decision (loan approval, fraud query)":    "self-consistency (vote N=3-5)",
        "Downstream parsing needed (API, database write)":      "json_mode",
    }
    for task, technique in guide.items():
        print(f"  {task}")
        print(f"    -> {technique}\n")


if __name__ == "__main__":
    main()

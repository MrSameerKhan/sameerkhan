# Session 3 — LLM Evaluation: A/B Testing Two Prompt Strategies
# Task   : compare zero-shot vs few-shot on a financial QA benchmark
# Metrics: ROUGE-L (automated) + LLM-as-judge (correctness, completeness, grounding)
# Shows  : evaluation pipeline every ML engineer runs before shipping a model change
#
# Change PROVIDER to switch backends. Nothing else needs to change.
#   "openai"  → needs OPENAI_API_KEY env var
#   "claude"  → needs ANTHROPIC_API_KEY env var
#   "ollama"  → needs Ollama running locally (ollama serve)

import os
import json
from dataclasses import dataclass, field

PROVIDER = "openai"   # "openai" | "claude" | "ollama"

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
            model=MODEL, max_tokens=1024, temperature=0,
            system=sys_text,
            messages=[{"role": "user", "content": query}],
        )
        return r.content[0].text
    else:
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = client.chat.completions.create(
            model=MODEL, max_tokens=1024, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": query}],
            **extra,
        )
        return r.choices[0].message.content.strip()


# ── Evaluation dataset ─────────────────────────────────────────────────────────
EVAL_SET = [
    {
        "question":  "What is the minimum down payment for a residential mortgage in Saudi Arabia?",
        "reference": "The minimum down payment for a residential mortgage is 10% of the property value for Saudi nationals.",
    },
    {
        "question":  "Can an expatriate obtain a home loan in Saudi Arabia?",
        "reference": "Expatriates can obtain home loans from some Saudi banks, but eligibility, LTV ratios, and terms differ from those for Saudi nationals.",
    },
    {
        "question":  "What is the maximum loan tenor for personal finance at Al Rajhi Bank?",
        "reference": "The maximum tenor for personal finance at Al Rajhi Bank is typically 60 months (5 years).",
    },
    {
        "question":  "What Sharia compliance principle governs Islamic bank lending?",
        "reference": "Islamic banks use Murabaha (cost-plus sale), Ijara (leasing), or Diminishing Musharakah instead of interest-based loans to comply with the prohibition on Riba.",
    },
    {
        "question":  "What documents are typically required to open a corporate bank account?",
        "reference": "Typically required: commercial registration, articles of association, board resolution authorising signatories, valid IDs for all signatories, and proof of business address.",
    },
]


# ── Two systems under test ─────────────────────────────────────────────────────

SYSTEM_A = "You are a banking assistant. Answer the question clearly and concisely."

SYSTEM_B = """You are a specialist at Al Rajhi Bank with expertise in Saudi banking regulations,
Islamic finance, and retail/corporate banking products. When answering:
- Be specific: include actual figures, percentages, or product names where known
- State if something varies by customer segment or bank policy
- If unsure, say so rather than guessing

Example:
Q: What is the profit rate on personal finance?
A: Al Rajhi personal finance (Murabaha-based) carries an annual profit rate typically
between 5.5% and 9.5%, varying by salary band, tenor, and credit risk. Use the bank's
finance calculator for a personalised quote."""


def call_llm(system: str, question: str) -> str:
    return ask(system, question)


# ── ROUGE-L ───────────────────────────────────────────────────────────────────

def rouge_l(prediction: str, reference: str) -> float:
    pred = prediction.lower().split()
    ref  = reference.lower().split()
    m, n = len(pred), len(ref)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if pred[i-1] == ref[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p   = lcs / m
    r   = lcs / n
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an impartial evaluator of banking assistant responses.
Score the response on three dimensions from 1 (poor) to 5 (excellent):

- correctness:   Is the factual content accurate? Are numbers/policies correct?
- completeness:  Does it address all parts of the question?
- specificity:   Does it give concrete details (figures, product names) vs vague generalities?

Question: {question}
Reference answer: {reference}
Response to evaluate: {response}

Return ONLY valid JSON matching this schema:
{{"correctness": int, "completeness": int, "specificity": int, "reasoning": str}}"""


@dataclass
class JudgeScore:
    correctness:  int = 0
    completeness: int = 0
    specificity:  int = 0
    reasoning:    str = ""

    @property
    def average(self) -> float:
        return (self.correctness + self.completeness + self.specificity) / 3


def llm_judge(question: str, reference: str, response: str) -> JudgeScore:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, response=response)
    raw    = ask("You are an impartial JSON evaluator.", prompt, json_mode=True)
    data   = json.loads(raw)
    return JudgeScore(
        correctness=data.get("correctness", 0),
        completeness=data.get("completeness", 0),
        specificity=data.get("specificity", 0),
        reasoning=data.get("reasoning", ""),
    )


# ── Evaluation loop ────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    question:   str
    reference:  str
    response_a: str
    response_b: str
    rouge_a:    float = 0.0
    rouge_b:    float = 0.0
    judge_a:    JudgeScore = field(default_factory=JudgeScore)
    judge_b:    JudgeScore = field(default_factory=JudgeScore)


def run_eval(n_examples: int = len(EVAL_SET)) -> list[EvalResult]:
    results = []
    for i, ex in enumerate(EVAL_SET[:n_examples]):
        print(f"  Evaluating {i+1}/{n_examples}: {ex['question'][:60]}...")
        resp_a = call_llm(SYSTEM_A, ex["question"])
        resp_b = call_llm(SYSTEM_B, ex["question"])
        r = EvalResult(
            question=ex["question"],
            reference=ex["reference"],
            response_a=resp_a,
            response_b=resp_b,
            rouge_a=rouge_l(resp_a, ex["reference"]),
            rouge_b=rouge_l(resp_b, ex["reference"]),
            judge_a=llm_judge(ex["question"], ex["reference"], resp_a),
            judge_b=llm_judge(ex["question"], ex["reference"], resp_b),
        )
        results.append(r)
    return results


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(results: list[EvalResult]) -> None:
    print("\n" + "=" * 70)
    print("EVALUATION REPORT: System A (zero-shot) vs System B (few-shot)")
    print("=" * 70)

    mean = lambda lst: sum(lst) / len(lst) if lst else 0.0

    rouge_a = [r.rouge_a for r in results]
    rouge_b = [r.rouge_b for r in results]
    judge_a = [r.judge_a.average for r in results]
    judge_b = [r.judge_b.average for r in results]

    print(f"\n{'Metric':<30} {'System A':>10} {'System B':>10} {'Winner':>10}")
    print("-" * 62)
    print(f"{'ROUGE-L (avg)':<30} {mean(rouge_a):>10.3f} {mean(rouge_b):>10.3f} {'B' if mean(rouge_b) > mean(rouge_a) else 'A':>10}")
    print(f"{'Judge correctness':<30} {mean([r.judge_a.correctness for r in results]):>10.2f} {mean([r.judge_b.correctness for r in results]):>10.2f} {'B' if mean([r.judge_b.correctness for r in results]) > mean([r.judge_a.correctness for r in results]) else 'A':>10}")
    print(f"{'Judge completeness':<30} {mean([r.judge_a.completeness for r in results]):>10.2f} {mean([r.judge_b.completeness for r in results]):>10.2f} {'B' if mean([r.judge_b.completeness for r in results]) > mean([r.judge_a.completeness for r in results]) else 'A':>10}")
    print(f"{'Judge specificity':<30} {mean([r.judge_a.specificity for r in results]):>10.2f} {mean([r.judge_b.specificity for r in results]):>10.2f} {'B' if mean([r.judge_b.specificity for r in results]) > mean([r.judge_a.specificity for r in results]) else 'A':>10}")
    print(f"{'Judge overall avg':<30} {mean(judge_a):>10.2f} {mean(judge_b):>10.2f} {'B' if mean(judge_b) > mean(judge_a) else 'A':>10}")

    print(f"\n{'-' * 70}")
    print("Per-question breakdown:")
    for i, r in enumerate(results):
        winner = "B" if r.judge_b.average > r.judge_a.average else "A"
        print(f"\n  Q{i+1}: {r.question[:65]}")
        print(f"        ROUGE  A={r.rouge_a:.3f}  B={r.rouge_b:.3f}")
        print(f"        Judge  A={r.judge_a.average:.2f}  B={r.judge_b.average:.2f}  -> {winner} wins")
        print(f"        Reasoning: {r.judge_b.reasoning[:120]}")

    print(f"\n{'=' * 70}")
    if mean(judge_b) > mean(judge_a) + 0.3:
        print("DECISION: Ship System B -- meaningfully better across all judge dimensions.")
    elif mean(judge_b) > mean(judge_a):
        print("DECISION: B is marginally better. Run human eval on 50+ examples to confirm.")
    else:
        print("DECISION: No significant difference. Stick with simpler System A.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Provider: {PROVIDER} | Model: {MODEL}")
    print(f"Eval set: {len(EVAL_SET)} questions")
    print(f"Metrics : ROUGE-L + LLM-as-judge (3 dimensions)\n")

    results = run_eval()
    print_report(results)


if __name__ == "__main__":
    main()

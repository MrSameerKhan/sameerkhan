# Session 1 — Prompt Engineering
Status: `🔧 Code-built`

Theory: [../../../6.llms/01_prompting.md](../../../6.llms/01_prompting.md)

---

## Use Case

Domain-aware customer support bot: a generic LLM says "please contact your bank" — a well-prompted LLM says "Al Rajhi personal finance carries a profit rate of 5.5%–9.5%, apply via the mobile app." Few-shot examples inject domain knowledge instantly, without fine-tuning.

---

## Key Concepts Demonstrated

### Progression: Zero-shot → Few-shot → CoT → JSON mode

| Technique | How | When it wins |
|-----------|-----|-------------|
| Zero-shot | System role only | Simple tasks with instruction-tuned models |
| Few-shot | 3 domain examples in system prompt | Complex format / domain vocabulary |
| Chain-of-Thought | Explicit NEED / FACTS / ANSWER structure | Multi-condition eligibility, policy lookup |
| Self-consistency | Sample N=3, vote | High-stakes facts where one pass may hallucinate |
| JSON mode | `response_format={"type": "json_object"}` | Downstream parsing (API, database, UI) |

### Why Few-shot Works

The model is a next-token predictor. By showing 3 examples of the format/depth you expect, you prime the model to continue in that style. The examples shift the probability distribution of the output toward your preferred format.

```
System prompt without examples → generic output style
System prompt with 3 domain examples → model predicts: "the next response should look like these"
```

### CoT Pattern (NEED / FACTS / ANSWER)

Forces the model to decompose before answering. Reduces hallucination on multi-step questions because the model commits its reasoning to the context before generating the answer.

---

## API Call Structure

```
client.chat.completions.create(
    model      = "gpt-4o-mini",
    temperature = 0.2,           ← low for deterministic support responses
    messages   = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query},
    ]
)
```

**temperature=0** for evaluation/extraction. **temperature=0.7–0.9** for creative generation. **temperature=0.2** for support — consistent but not robotic.

---

## Expected Output (sample)

```
══════════════════════════════════════════════════════════════════════
QUERY: How do I increase my credit card limit?
══════════════════════════════════════════════════════════════════════

[Zero-shot]
To increase your credit card limit, you can contact your bank's customer
service or log into your online banking portal...

[Few-shot]
To request a credit limit increase on your Al Rajhi credit card: (1) Log
into the Al Rajhi mobile app → Cards → Request Limit Increase, (2) Submit
your latest payslip for income verification. Decisions are made within 2
business days. Alternatively visit any branch with your National ID.

[Chain-of-Thought]
NEED: Customer wants to raise their credit card spending limit.
FACTS:
  • Al Rajhi limit increases require income re-verification
  • Can be requested via mobile app, phone (920002470), or branch
  • Minimum 6 months card tenure typically required
ANSWER: Log into Al Rajhi app → Cards → Request Limit Increase...

[JSON mode]
{
  "answer": "Log into Al Rajhi app → Cards → Request Limit Increase...",
  "confidence": "high",
  "next_step": "Open Al Rajhi mobile app and navigate to Cards section",
  "escalate_to_human": false
}
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
python 01_prompt_engineering.py
```

Cost: ~$0.01–0.02 per run (5 API calls × 2 queries × gpt-4o-mini rates).
No local model download needed — runs entirely via API.

**Swap to Anthropic:**
```python
import anthropic
client = anthropic.Anthropic()
# Replace client.chat.completions.create() with client.messages.create()
```

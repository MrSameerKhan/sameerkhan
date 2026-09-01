# Session 3 — Synthetic Dataset Preparation
Status: `🔧 Code-built`

Theory: [../../../6.llms/07_dataset_preparation.md](../../6.llms/07_dataset_preparation.md)

---

## Use Case

You have 10K raw bank policy documents and zero labeled instruction pairs. This pipeline converts raw text → usable fine-tuning dataset automatically, using an LLM to generate synthetic Q&A pairs.

---

## Pipeline

```
Raw documents (7 policy paragraphs)
    │
    LLM generates 3 Q&A pairs per doc  →  21 raw pairs
    │
    Quality filter (length, no-question-response, no-duplicate) → ~18 pairs
    │
    Format as ChatML or Alpaca text
    │
    80/10/10 split → train / validation / test
    │
    HuggingFace Dataset → save_to_disk()
    │
    Optional: generate DPO preference pairs (chosen + rejected per prompt)
```

---

## Output Formats

### ChatML (TinyLlama / Llama-2-chat / Mistral-Instruct)
```
<|system|>
You are a knowledgeable bank assistant...</s>
<|user|>
What is the maximum LTV for a first-time buyer?</s>
<|assistant|>
First-time buyers are eligible for up to 95% LTV under the Help to Buy
scheme. This requires a minimum 5% deposit...</s>
```

### Alpaca (OPT / older models)
```
Below is an instruction.

### Instruction:
What is the maximum LTV for a first-time buyer?

### Response:
First-time buyers are eligible for up to 95% LTV...
```

### DPO format (for session 04)
```json
{
  "prompt": "What is the maximum LTV for a first-time buyer?",
  "chosen": "First-time buyers can borrow up to 95% LTV under Help to Buy...",
  "rejected": "You can get a mortgage as a first-time buyer with various options..."
}
```

---

## Quality Filtering Rules

| Filter | Catches |
|--------|---------|
| `len(instruction) < 15` | Degenerate one-word questions |
| `response.endswith("?")` | LLM returned a question instead of answer |
| `instruction[:30] in response` | Copy-paste (response echoes the question) |

Add domain-specific filters: e.g. filter responses that don't contain any numbers if the instruction asks about a percentage.

---

## Expected Output

```
Step 1: Generating instruction pairs from raw documents...
  Doc 1/7: The maximum loan-to-value (LTV) ratio for standard...
    → 3 pairs kept
  ...
  Doc 7/7: Credit cards are available to customers...
    → 2 pairs kept

Total pairs before filter: 18
Total pairs after formatting: 18
  Saved train:      14 examples → data/09_finetuning/banking_instructions/train
  Saved validation:  2 examples
  Saved test:        2 examples

Step 2: Generating DPO preference pairs...
  ✓ 'What is the maximum LTV for first-time buyers?...'
  ✓ 'What documents do I need to apply for a mortgage?...'
  ...

DPO dataset: 5 preference pairs → data/09_finetuning/banking_instructions/dpo_train
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
python 03_dataset_prep.py
```

Cost: ~$0.05 per run (21 LLM calls for Q&A generation + 5 for DPO pairs).
Runtime: ~60 seconds.

**Scale up:** for 10K policy documents, run in parallel using `asyncio` + `AsyncOpenAI`. Expect ~$5–10 for 10K documents at 3 pairs each.

# Session 6 — GPT-2 Causal Language Modeling + Text Generation
Status: `✅ Run`

Theory: [../../../5.transformers/02_models/02_gpt_family.md](../../5.transformers/02_models/02_gpt_family.md)

---

## Use Case

Synthetic training data generator: you have 500 labelled customer complaint examples but need 10,000 to train a classifier. Fine-tune GPT-2 on your domain corpus (emails, complaints, policy docs), then generate thousands of realistic synthetic examples with varied prompts.

---

## Key Concept — Causal Language Modeling

Unlike BERT (bidirectional — attends to all tokens), GPT-2 is causal — each token can only attend to previous tokens. Training target: predict the next token.

```
Input : [The] [bank] [approved] [the]
Labels: [bank] [approved] [the] [loan]   ← shifted left by 1

GPT-2 shifts labels internally — you pass input_ids as labels.
Loss = cross-entropy averaged over all token positions.
```

**Key difference from BERT fine-tuning:** there are no `[SEP]` or `[CLS]` tokens, no sentence pairs. You chunk raw text into fixed-length windows and predict the next token at every position.

---

## Tensor Shapes

```
Text → tokenize → concatenate all token ids → chunk at MAX_LEN=256

Per chunk:
  input_ids: shape (256,)
  labels:    shape (256,) — same tensor (GPT-2 shifts right internally)

Model output:
  logits: shape (256, 50257)  — score for each vocab token at each position
  loss:   scalar — cross-entropy averaged over all 256 positions

Perplexity = exp(loss) — lower = model is more confident about next token
  - Untrained GPT-2: ~30-50 on Wikipedia
  - After fine-tuning: ~15-25 on in-domain text
  - Domain-adapted model on domain test: <10 (excellent)
```

---

## Sampling Strategies

| Strategy | How | Output |
|----------|-----|--------|
| Greedy | argmax at each step | Deterministic, often repetitive |
| Top-k | Sample from top k tokens | Diverse but controlled (k=50 is standard) |
| Top-p (nucleus) | Sample from top-p cumulative prob mass | Dynamic — adapts to prediction sharpness |
| Temperature | Scale logits before softmax | >1 = more random; <1 = more confident |

**Top-p is the default production choice.** At each step, sort tokens by probability, keep the smallest set whose cumulative probability ≥ p (e.g., 0.92). Sample from that set. When the model is confident (peaked distribution), the set is small. When uncertain, it's larger — natural adaptation.

---

## Input Data — WikiText-2

| Property | Value |
|----------|-------|
| Source | Curated Wikipedia articles (good writing, varied vocabulary) |
| Size | ~2M tokens total |
| Script subset | 2,000 examples → ~150k tokens → ~585 chunks at MAX_LEN=256 |
| Download | `load_dataset("wikitext", "wikitext-2-raw-v1")` |

**Production swap:** replace WikiText with your domain corpus (financial reports, customer emails) — same code, different `texts` list. The more domain-specific the corpus, the better the generated synthetic examples.

---

## Expected Output

```
Device: mps

Epoch 1/3 | loss: 3.2841 | perplexity: 26.71
Epoch 2/3 | loss: 2.9103 | perplexity: 18.35
Epoch 3/3 | loss: 2.7241 | perplexity: 15.24

Saved to models/05_transformers/gpt2_generation

── Generation — same prompt, different strategies ──

  [greedy]
  The bank's loan approval process requires the applicant to provide proof of income
  income income income...     ← repetition without diversity

  [top_k]
  The bank's loan approval process requires the applicant to submit documentation
  including tax returns, bank statements, and employment verification letters.

  [top_p]
  The bank's loan approval process requires the applicant to demonstrate financial
  stability through three months of payslips and a valid credit history report.

── Synthetic training data examples ──
  → Customer complaint: the interest rate on my account has been changed without notice,
    and I was not informed of the new rate before the change took effect.

  → Loan application status update: your file has been reviewed by our credit assessment
    team and we require additional documentation to proceed with your request.
```

---

## Actual Output (Windows, GTX 1650 Ti, 2026-06-25)

```
Device: cuda

Epoch 1/3 | loss: 3.6503 | perplexity: 38.49
Epoch 2/3 | loss: 3.2548 | perplexity: 25.91
Epoch 3/3 | loss: 3.1254 | perplexity: 22.77

Saved to models/05_transformers/gpt2_generation

-- Generation — same prompt, different strategies --
  [greedy]  repetitive — "bank...bank...bank obligations" loop (expected greedy behavior)
  [top_k]   diverse, domain-relevant (state inspection, emergency guarantees)
  [top_p]   fluent financial prose (application fee, annual report submission)

-- Synthetic training data examples --
  → Customer complaint: interest rate fixed at $3/year, could not retrieve funds
  → Loan application status update: approved for review, submit within weeks
  → Mortgage terms: fixed rate 15-20% first two years, 12% premium thereafter
```

- Perplexity dropped 38.5 → 25.9 → 22.8 — model adapting to domain
- All 3 generation strategies produced distinct outputs — sampling working correctly
- Warning: `attention_mask` not set (pad=eos token) — cosmetic, does not affect output

---

## How to Run

```powershell
python code_practice\05_transformers\06_gpt2_generation.py
```

First run: downloads `gpt2` (~548 MB) + WikiText-2 (~8 MB).
CUDA training: ~3–5 min for 3 epochs on 2k examples.

**Note:** `tokenizer.pad_token = tokenizer.eos_token` is required — GPT-2 has no pad token.

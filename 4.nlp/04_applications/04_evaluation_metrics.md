# NLP Evaluation Metrics

---

## Quick Reference

| Task | Primary Metric | Secondary | Notes |
|------|---------------|-----------|-------|
| Text Classification | F1 (macro/weighted) | ROC-AUC, PR-AUC | See ML/fundamentals/04 |
| NER | Entity-level F1 (seqeval) | Per-type F1 | Exact span + type match |
| Machine Translation | BLEU | chrF, COMET | N-gram overlap |
| Summarization | ROUGE-1/2/L | BERTScore | Recall-oriented |
| QA (extractive) | EM + F1 | SQuAD F1 | Token overlap |
| QA (generative) | BERTScore | ROUGE-L | Semantic similarity |
| Language Models | Perplexity | BPC | Lower = better |
| Generation (general) | BERTScore | Human eval | Semantic not n-gram |

---

## Classification Metrics (Recap)

Covered in depth in `ML/fundamentals/04_model_evaluation.md`. NLP-specific additions:

```python
from sklearn.metrics import classification_report, f1_score

# Multi-class: always check per-class breakdown
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

# For imbalanced NLP datasets (rare entity types, rare categories)
# → macro F1 penalizes poor performance on rare classes
# → weighted F1 is dominated by common classes
# → report both

# Multi-label: samples-averaged F1 is most informative
f1_samples = f1_score(y_true, y_pred, average='samples')
```

---

## NER Metrics: seqeval

### Entity-level F1 (Standard)

```python
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score
)

# seqeval expects list of lists of IOB tags (one list per sentence)
y_true = [
    ['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG'],
    ['O', 'B-LOC', 'O', 'O'],
]
y_pred = [
    ['B-PER', 'I-PER', 'O', 'B-ORG', 'O'],    # missed last token of ORG
    ['O', 'B-LOC', 'O', 'O'],
]

# Entity-level: entire span must match (type + start + end)
print(classification_report(y_true, y_pred))
# PER: P=1.00, R=1.00, F1=1.00
# ORG: P=0.00, R=0.00, F1=0.00  → span boundary missed = 0

f1 = f1_score(y_true, y_pred)   # macro by default
print(f"Entity-level F1: {f1:.4f}")
```

### Why Entity-level > Token-level

```
Ground truth: "New York City" = B-LOC I-LOC I-LOC
Prediction:   "New York"      = B-LOC I-LOC O

Token-level F1: 2 of 3 tokens correct → F1 ≈ 0.80
Entity-level F1: span doesn't fully match → F1 = 0.00

Entity-level is stricter but more meaningful for downstream use:
if you extract "New York" instead of "New York City", you'll fail
to find it in a knowledge base or pass it to the next pipeline stage.
```

---

## BLEU (Bilingual Evaluation Understudy)

**What it measures:** N-gram precision of generated text vs. reference(s)

**Formula:**
```
BLEU = BP × exp(Σ w_n × log p_n)

where:
  p_n = clipped n-gram precision (1 ≤ n ≤ N)
  w_n = weight (usually 1/4 for n=1,2,3,4)
  BP  = brevity penalty (penalizes too-short outputs)

Brevity Penalty:
  BP = 1         if len(output) >= len(reference)
  BP = exp(1-r/c) if len(output) < len(reference)

Clipped precision (prevents padding hack):
  Count n-gram matches but clip to max count in any reference
```

```python
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# Corpus BLEU (standard for MT evaluation)
references  = [[['the', 'cat', 'sat', 'on', 'the', 'mat']]]   # list of list of tokens per sentence
hypotheses  = [['the', 'cat', 'is', 'on', 'the', 'mat']]

bleu4 = corpus_bleu(references, hypotheses)
print(f"BLEU-4: {bleu4:.4f}")

# Sentence-level BLEU (smoothed — use SmoothingFunction for short sentences)
smoother = SmoothingFunction()
score    = sentence_bleu(
    [['the', 'cat', 'sat']],
    ['the', 'cat', 'sits'],
    smoothing_function=smoother.method1
)

# sacrebleu: standardized, reproducible BLEU (use this in papers/production)
import sacrebleu

bleu = sacrebleu.corpus_bleu(hypotheses=['the cat is on the mat'],
                              references=[['the cat sat on the mat']])
print(bleu)   # BLEU = 51.15 ...
```

### BLEU Limitations

```
- N-gram matching: "The bank on the river bank" vs "The bank is by the river" — same n-grams, different meaning
- No semantic similarity; synonyms penalized
- Reference-dependent: quality limited by reference quality
- Short text fails: very short outputs get random scores
- Not good for summarization: too precision-focused

→ Use BLEU for machine translation comparison
→ Avoid for open-ended generation, summarization, dialogue
```

---

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**What it measures:** N-gram recall of generated summary vs. reference(s)

**Variants:**
```
ROUGE-1: Unigram overlap (individual words)
ROUGE-2: Bigram overlap (adjacent word pairs)
ROUGE-L: Longest Common Subsequence (handles word order without exact contiguity)
ROUGE-S: Skip-gram overlap (pairs with gaps allowed)
```

**Formula:**
```
ROUGE-N Recall    = |matched n-grams| / |reference n-grams|
ROUGE-N Precision = |matched n-grams| / |hypothesis n-grams|
ROUGE-N F1        = 2 × P × R / (P + R)
```

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

reference  = "The cat sat on the mat near the window."
hypothesis = "A cat is sitting on a mat."

scores = scorer.score(reference, hypothesis)
print(f"ROUGE-1: P={scores['rouge1'].precision:.3f}, "
      f"R={scores['rouge1'].recall:.3f}, "
      f"F={scores['rouge1'].fmeasure:.3f}")
print(f"ROUGE-2: F={scores['rouge2'].fmeasure:.3f}")
print(f"ROUGE-L: F={scores['rougeL'].fmeasure:.3f}")

# Corpus-level (for evaluating multiple summaries)
from rouge_score import scoring

aggregator = scoring.BootstrapAggregator()
for ref, hyp in zip(references, hypotheses):
    aggregator.add_scores(scorer.score(ref, hyp))
result = aggregator.aggregate()
print(f"ROUGE-1 F1 (mean): {result['rouge1'].mid.fmeasure:.4f}")
```

### ROUGE Limitations

```
- Surface form only: synonyms fail ("automobile" vs "car" = 0 overlap)
- Reference quality matters: bad references → misleading scores
- Abstractive summaries penalized: paraphrasing = low ROUGE but high quality
- No coherence, fluency, or factual accuracy measurement

→ ROUGE-1/2/L are still the standard for abstractive summarization benchmarks
→ Pair with BERTScore and human evaluation for production
```

---

## BERTScore

**What it measures:** Semantic similarity between generated and reference text using contextual embeddings.

**Formula:**
```
For each token in hypothesis, find max cosine similarity with any reference token.
Aggregate (precision, recall, F1) using these max-similarity scores.

P_BERT = avg max cosine-sim(hypothesis tokens × reference tokens)
R_BERT = avg max cosine-sim(reference tokens × hypothesis tokens)
F_BERT = harmonic mean of P_BERT and R_BERT
```

```python
from bert_score import score

candidates = ["The cat is sitting on a mat."]
references  = ["A cat sat on the mat."]

P, R, F1 = score(candidates, references,
                 lang="en",
                 model_type="microsoft/deberta-xlarge-mnli",   # best model for BERTScore
                 rescale_with_baseline=True)   # scale to more intuitive range

print(f"BERTScore F1: {F1.mean():.4f}")

# Batch evaluation
all_P, all_R, all_F1 = score(all_candidates, all_references, lang="en", batch_size=64)
print(f"Mean BERTScore F1: {all_F1.mean():.4f}")
```

### BERTScore vs BLEU/ROUGE

```
BERTScore advantages:
  ✓ Captures semantic similarity (synonyms score well)
  ✓ Handles paraphrasing
  ✓ Correlates better with human judgment than BLEU/ROUGE

BERTScore disadvantages:
  ✗ Slower (requires BERT forward pass)
  ✗ Less interpretable than n-gram overlap
  ✗ Still reference-dependent

Rule of thumb: use BERTScore where form can vary (abstractive summarization,
generation tasks). Use BLEU when surface accuracy matters (MT, code generation).
```

---

## QA Metrics: Exact Match (EM) and F1

**Extractive QA (SQuAD format):** The answer is a span in the context.

```python
import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text, remove punctuation, articles, extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def exact_match(prediction, ground_truth):
    """Exact match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def token_f1(prediction, ground_truth):
    """Token-level F1 between prediction and reference (SQuAD official metric)."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens   = normalize_answer(ground_truth).split()

    common    = Counter(pred_tokens) & Counter(gt_tokens)
    num_same  = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# Handle multiple valid answers (SQuAD v1.1 has multiple annotations)
def squad_metrics(prediction, ground_truths):
    em = max(exact_match(prediction, gt) for gt in ground_truths)
    f1 = max(token_f1(prediction, gt) for gt in ground_truths)
    return {'EM': em, 'F1': f1}

# Example
pred = "in 1976"
gts  = ["1976", "In 1976", "the year 1976"]
print(squad_metrics(pred, gts))   # {'EM': True, 'F1': 1.0}
```

---

## Perplexity

**What it measures:** How well a language model predicts a text corpus. Lower = better.

```
Perplexity = exp(-1/N × Σ log P(w_i | w_1...w_{i-1}))

Interpretation:
  PP = 10  → model is as confused as uniformly choosing among 10 words
  PP = 100 → very uncertain model
Good LMs achieve PP < 50 on standard benchmarks
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_perplexity(text, model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    tokens    = tokenizer(text, return_tensors='pt')
    input_ids = tokens.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss = mean negative log-likelihood per token
        nll = outputs.loss.item()

    perplexity = torch.exp(torch.tensor(nll)).item()
    return perplexity

# Perplexity for long texts (sliding window to avoid truncation)
def compute_perplexity_sliding(text, model, tokenizer, max_length=512, stride=256):
    encodings  = tokenizer(text, return_tensors='pt')
    max_length = min(max_length, model.config.max_position_embeddings)
    seq_len    = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc    = min(begin_loc + max_length, seq_len)
        trg_len    = end_loc - prev_end_loc
        input_ids  = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100   # ignore prefix

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            nll     = outputs.loss * trg_len

        nlls.append(nll)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()
```

---

## Modern Evaluation Frameworks (2024-2025)

The metrics above measure individual outputs. **Modern frameworks** wrap them into standardized benchmarks for LLMs, embeddings, and RAG systems.

### MTEB — Massive Text Embedding Benchmark

The standard for embedding model comparison: 56 datasets across retrieval, reranking, clustering, classification, STS, summarization.

```python
from mteb import MTEB
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
evaluation = MTEB(tasks=["MSMARCO", "STSBenchmark", "Banking77Classification"])
results = evaluation.run(model, output_folder="results/")
```

Used to compare BGE, E5, Nomic, OpenAI embeddings. Public leaderboard at huggingface.co/spaces/mteb/leaderboard. **BEIR** is the older retrieval-focused subset, still widely cited.

### RAGAS — RAG Pipeline Evaluation

Evaluates retrieval-augmented generation by isolating each stage. All metrics use an LLM judge internally.

| Metric | What it measures |
|--------|-----------------|
| faithfulness | Generated answer is grounded in the retrieved context |
| answer_relevancy | Answer addresses the question |
| context_precision | Are retrieved chunks ranked correctly? |
| context_recall | Were the right chunks retrieved at all? |
| context_entity_recall | Coverage of expected entities in retrieved context |
| noise_sensitivity | Robustness to irrelevant retrieved content |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

result = evaluate(
    dataset=eval_dataset,   # {question, ground_truth, contexts, answer}
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
```

Diagnostic by design — low `context_recall` says fix retrieval; low `faithfulness` says fix prompting/grounding. Deeper RAG eval coverage: `05_rag_pipeline.md` and `06_generative_eval.md`.

### LLM Benchmarks — lm-eval-harness, MT-Bench, Arena-Hard, AlpacaEval

| Tool | Purpose |
|------|---------|
| lm-eval-harness (EleutherAI) | Standardized eval on 100+ benchmarks (MMLU, HellaSwag, GSM8K, TruthfulQA, ARC...); industry standard for model release reports |
| MMLU / MMLU-Pro | Multi-task knowledge (57 subjects, ~14K questions); Pro version harder |
| GSM8K / MATH | Math reasoning |
| HumanEval / MBPP / SWE-bench | Code generation (function-level → repo-level) |
| TruthfulQA | Avoiding misconceptions |
| MT-Bench | LLM-as-judge multi-turn conversation eval (GPT-4 grader) |
| Arena-Hard / AlpacaEval 2 | LLM-as-judge alignment with human preferences |
| Chatbot Arena (lmsys) | Live crowdsourced pairwise comparisons → Elo ratings |
| HELM (Stanford) | Holistic Evaluation of Language Models — accuracy + robustness + fairness + efficiency + bias |

```bash
# lm-eval-harness — evaluate any HF model on standard benchmarks
pip install lm-eval
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
  --tasks mmlu,hellaswag,arc_challenge,truthfulqa_mc2 --batch_size 8
```

### Long Context Evaluation

Standard benchmarks saturate quickly with 100K+ context windows. Use:

| Benchmark | What it tests |
|-----------|---------------|
| Needle-in-a-Haystack (NIAH) | Recall a fact placed at varying depths in long context |
| RULER (NVIDIA, 2024) | 13 tasks: needle, multi-needle, multi-hop QA, aggregation, frequent-word extraction |
| LongBench / LongBench-v2 | Long-document QA, summarization, code completion |
| InfiniteBench | 100K+ context tasks (math, retrieval, dialog) |
| L-Eval / ZeroSCROLLS | Earlier long-context benchmarks |

### LLM-as-Judge — Caveats

Most modern eval relies on a stronger LLM judging outputs. Known biases:

```
- Position bias    — judges prefer the first option presented; mitigation: swap order, average
- Length bias      — longer answers rated higher even when same content
- Self-preference  — models often rate their own outputs higher
- Style over substance — fluent wrong answers can beat awkward right ones
```

Best practices: use a stronger model as judge (GPT-4-class), randomize order, calibrate with human annotations on a sample.

---

## Human Evaluation Rubrics

For production NLP systems, automated metrics have limits. Key human evaluation dimensions:

```
Fluency:       Is the text grammatical and readable? (1-5 scale)
Adequacy:      Does the output convey all the information in the source? (1-5)
Coherence:     Does the text flow logically? (1-5)
Faithfulness:  For summarization — is the output factually consistent with source?
               (binary or 1-5 scale, most critical for news summarization)
Relevance:     Does the response answer the question? (1-5)

For document extraction:
Accuracy:      Are extracted fields correct? (per-field accuracy)
Completeness:  Are all required fields present? (recall)
```

### Setting Up Human Evaluation

```python
# Inter-annotator agreement — critical for trustworthy human eval
from sklearn.metrics import cohen_kappa_score
import numpy as np

annotator1_scores = [4, 3, 5, 2, 4, 3, 4, 3, 2]
annotator2_scores = [4, 4, 5, 2, 3, 3, 4, 4, 2]

# Cohen's Kappa (agreement beyond chance)
kappa = cohen_kappa_score(annotator1_scores, annotator2_scores)
print(f"Cohen's Kappa: {kappa:.3f}")
# κ > 0.60: substantial agreement (acceptable for NLP annotation)
# κ > 0.80: almost perfect agreement
```

---

## Choosing the Right Metric

| Task | Primary | Why |
|------|---------|-----|
| Text Classification | F1 (macro) | Class imbalance; all classes matter |
| Imbalanced binary | PR-AUC | ROC-AUC misleading |
| NER | Entity-level F1 | Full span must match |
| Machine Translation | BLEU + chrF | Surface accuracy matters |
| Abstractive Summary | ROUGE-L + BERTScore | Surface + semantic |
| Extractive QA | EM + F1 | SQuAD standard |
| Document extraction | Field-level EM | Exact value matters |
| LM training | Perplexity | Probabilistic model quality |
| Open-ended generation | BERTScore + human | N-gram metrics inadequate |

---

## Gotchas

**BLEU-4 on short sentences:** BLEU-4 requires 4-gram matches; short sentences may have no 4-grams — score = 0. Use smoothed BLEU or BLEU-1/2 for sentence-level evaluation.

**ROUGE as sole metric for summarization:** A model that copies the first 3 sentences of the article often beats abstractive summaries on ROUGE. Add faithfulness evaluation (FactCC, QAFactEval) for production.

**seqeval with custom entity types:** seqeval splits entity types from label strings using `-` as separator. If your labels contain hyphens (B-INVOICE-NUM), they'll break. Rename to underscores (B-INVOICE_NUM).

**BERTScore layer selection:** Different layers capture different information. By default uses the best layer for each model. Using a too-early layer → a different kind of semantic similarity.

**EM for QA is brutal:** "in 1976" vs "1976" — different EM (0) but same F1 (0.67). Always report both EM and F1 for QA tasks.

**Perplexity is model-specific:** Perplexities of different model types aren't comparable. GPT-2 perplexity ≠ BERT perplexity (BERT is masked LM, not causal LM).

---

## Interview Q&A

**Q: Why is BLEU not sufficient for evaluating summarization?**

A: BLEU is precision-oriented and n-gram based. For summarization, you want recall (did the summary capture the key information?), not just precision. ROUGE addresses this by measuring recall. Additionally, abstractive summaries use different words than the reference — a perfectly valid paraphrase gets zero n-gram overlap. BERTScore handles this via semantic similarity. Finally, BLEU/ROUGE don't measure factual accuracy — a summary that contradicts the source article can still score highly on ROUGE if it shares words with the reference.

**Q: A model gets high ROUGE-1 but low ROUGE-2. What does this indicate?**

A: The model captures the right individual words (high unigram recall) but not the right word sequences (low bigram recall). This suggests the summary is extracting the right topics/keywords but not preserving the correct phrasing or word ordering — characteristic of a model that's good at topic detection but poor at fluency or copying relevant phrases. Could indicate an extractive model that's picking the right sentences but in the wrong order, or a generative model that paraphrases heavily.

**Q: How do you evaluate a NER model in production when you can't label all test data?**

A: (1) Sample-based evaluation — label a random sample of production outputs (confidence predictions). (2) Active learning — have annotators focus on low-confidence predictions. (3) Business metric proxies — downstream accuracy (if NER feeds into a database, check database fill rate and query accuracy). (4) Error rate from human review — if humans flag extraction errors, track error rate per document type. (5) Distribution monitoring — track entity type distributions; unexpected shifts signal model degradation.

**Q: BLEU score dropped from 0.35 to 0.28 after model update. Is this bad?**

A: Context matters: (1) BLEU absolute values aren't intuitive — 0.35 could be excellent or mediocre depending on the task and language pair. Was it measured on the same test set with the same tokenization? (Tokenization differences alone can cause large BLEU changes — always use sacrebleu for reproducibility.) (3) Does this translate to human evaluation? Sometimes BLEU drops while human quality improves (model is more fluent/creative but less word-for-word similar to reference). Always complement BLEU with at least one other metric.

**Q: For a real-time customer service chatbot, which metrics would you track?**

A: Automated: BERTScore for response relevance, intent classification accuracy (does it understand the request?), entity extraction F1 (getting product names right). Business: resolution rate (did the customer's problem get solved?), containment rate (did they need to escalate to human agent?), response latency. Human evaluation sample: appropriateness, helpfulness, tone (sampled 1-5% of conversations). The automated metrics are proxies — the business metrics are what actually matter.

---

## Connections

- **Text Classification:** `01_text_classification.md` — F1, ROC-AUC, PR-AUC covered in detail there
- **NER:** `02_ner_and_tagging.md` — seqeval entity-level F1 is the primary NER metric
- **Information Extraction:** `03_information_extraction.md` — Field-level EM for document extraction
- **RAG pipeline + RAGAS depth:** `05_rag_pipeline.md`
- **Generative eval depth:** `06_generative_eval.md`
- **LLM evaluation systems:** `../11_system_design/11_llm_evaluation_systems.md`
- **MTEB / embedder choice:** `../02_embeddings/02_sentence_embeddings.md`
- **Conformal prediction (calibrated uncertainty):** `../../1.machine learning/01_fundamentals/04865-conformal-prediction-distribution-free-uncertainty`
- **Transformers:** `../../2_deep_learning/02_architectures/04_transformer.md` — BERTScore uses contextual embeddings
- **ML Model Evaluation:** `../../1.machine learning/01_fundamentals/04_model_evaluation.md` — Core metric concepts (precision/recall/F1/ROC) defined there
- **Production LLM observability tooling:** `../../10.mlops/11_llm_observability_tools.md`

---

## Key Takeaway

No single metric captures everything. Use metrics appropriate to the task: seqeval entity F1 for NER, BLEU/ROUGE for generation (with BERTScore as sanity check), field-level EM for document extraction. Always complement automated metrics with human evaluation on a sample — they disagree more often than you'd expect. In production, business metrics (resolution rate, processing time, error rate) matter more than academic benchmarks.

---

## Code Practice — Wired by Phase 6

- `code_practice/09_llms/13_lm_eval_harness/` — ARC + HellaSwag + TruthfulQA
- `code_practice/05_rag/08_rag_eval/` — RAGAS-style eval

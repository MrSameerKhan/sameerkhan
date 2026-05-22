# Interview Pack — Technical Q&A + Behavioral STAR

> **One file. Everything interview-related lives here.** Replaces
> `04_interview_prep_behavioral_star_answers_barraiser`. Interview-Q bank from
> `06_mental_model` consolidated here.

---

## Table of Contents

**Part A — Technical Q&A (top 20 detailed answers)**

1. [Transformer & Architecture (Q1-Q4)](#1-transformer--architecture)
2. [RAG Pipelines (Q5-Q8)](#2-rag-pipelines)
3. [Fine-tuning & Alignment (Q9-Q12)](#3-fine-tuning--alignment)
4. [Agentic AI (Q13-Q15)](#4-agentic-ai)
5. [Your Production System (Q16-Q17)](#5-your-production-system)
6. [MLOps & System Design (Q18-Q20)](#6-mlops--system-design)

**Part B — Extended Question Bank (no full answers — practice out loud)**

7. [Phase-by-Phase Question Bank](#7-phase-by-phase-question-bank)

**Part C — Behavioral STAR (BarRaiser)**

8. [What BarRaiser Tests](#8-what-barraiser-tests)
9. [The STAR Method](#9-the-star-method)
10. [10 Behavioral Questions with Full Answers](#10-10-behavioral-questions-with-full-answers)
11. [BarRaiser-Specific Tips](#11-barraiser-specific-tips)
12. [Your 5 Core Stories — Template](#12-your-5-core-stories--template)
13. [Day-Before Checklist](#13-day-before-checklist)

---

## Part A — Technical Q&A

> Top 20 questions for LLM / GenAI Engineer roles. Target: 2-3 minutes per answer. **Practice aloud, not in your head.**

---

### 1. Transformer & Architecture

**Q1: Explain self-attention mathematically.**

`Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V` where Q=XW_Q, K=XW_K, V=XW_V. The query (what I am looking for) dots with each key (what each position offers) to produce attention weights, scaled by √d_k to prevent softmax saturation at high dimensions, then used to weight the values. Multi-head repeats this h times with different learned projections, concatenates and projects — each head captures different patterns (syntax, coreference, semantics). You use this in your BERT and Donut architectures at ICE.

---

**Q2: What is KV Cache and why does it matter?**

During autoregressive generation, each new token must attend to all previous tokens. Without caching, we recompute K and V for all prior tokens every step — O(n²). KV Cache stores K and V tensors for all previous tokens; each step only computes K/V for the new token and appends to cache, making inference O(n) per step. The cost: GPU memory grows linearly with context length. This is exactly what vLLM's PagedAttention solves — treating KV cache like virtual memory with non-contiguous page allocation.

---

**Q3: BERT vs GPT architecturally — and where you have used each.**

BERT = encoder-only, bidirectional attention — reads full sequence at once, MLM pretraining, ideal for classification/NER/embeddings. You used BERT at ICE to push page-level accuracy from 93% to 94%. GPT = decoder-only, causal left-to-right attention, autoregressive next-token prediction, ideal for generation. Donut (your OCR-free pipeline) uses Swin Transformer encoder + BART decoder — an encoder-decoder architecture. You have hands-on experience with all three architectural families.

---

**Q4: What is Flash Attention?**

Standard attention materialises the full n×n attention matrix in HBM (slow GPU memory), creating an O(n²) memory bottleneck. Flash Attention tiles the computation into blocks that fit in fast SRAM, fusing the softmax into the tiled computation to avoid slow round-trips. Produces identical results to standard attention but is 2-4× faster and uses O(n) memory. Critical for long-context models and fine-tuning on large financial documents — directly relevant to your ICE pipeline.

---

### 2. RAG Pipelines

**Q5: Walk me through your RAG pipeline end-to-end.**

From your project: PDF/text ingestion → fixed-size chunking (500 chars, 50 overlap) → sentence-transformers `all-MiniLM-L6-v2` embeddings → FAISS `IndexFlatIP` (exact cosine via L2-normalised inner product) → top-K=5 retrieval → prompt assembly with source attribution → Ollama `llama3.2:1b` (local) or HF InferenceClient (cloud). FastAPI backend (`/ingest`, `/query`, `/evaluate`), Streamlit UI, RAGAS-style evaluation (P@K, R@K, MRR=1.0). Key design decision: `IndexFlatIP` over IVF because corpus is small (<100K docs) — exact search with no recall loss, under 5ms latency.

---

**Q6: What is hybrid search and why is it better than pure vector search?**

Vector search captures semantic similarity but misses exact keyword matches — product codes, ISIN numbers, document IDs, proper nouns. BM25 (sparse retrieval) is great at exact matches but misses paraphrases. Hybrid search combines both using Reciprocal Rank Fusion: each method's rank is converted to 1/(k+rank), scores summed. Consistently outperforms either alone — especially on financial documents where exact terms (ISIN codes, contract numbers) AND semantic context both matter. Direct extension of your current FAISS system.

---

**Q7: What is HyDE?**

Hypothetical Document Embeddings: instead of embedding the short user query directly, prompt the LLM to generate a hypothetical answer first, then embed that hypothetical answer for retrieval. The hypothesis is in the same semantic space as the real documents, so retrieval is more precise. Works especially well when queries are short and documents are long — common in Q&A over annual reports, regulatory filings, and financial disclosures.

---

**Q8: How do you evaluate a RAG system?**

RAGAS framework: (1) **Faithfulness** — does the answer contain only information from retrieved context, no hallucination? (2) **Answer Relevancy** — is the answer actually addressing the question? (3) **Context Precision** — are retrieved chunks relevant? (4) **Context Recall** — are all ground-truth relevant chunks being retrieved? Build an eval dataset of 50-100 expert question-answer-context triplets from your domain. Run RAGAS, identify the weakest metric, fix that specific component. Your project has MRR=1.0 and P@5=1.0 on the ML domain test set — speak to this directly.

---

### 3. Fine-tuning & Alignment

**Q9: Explain LoRA. What is the math?**

Instead of updating the full weight matrix W (d×k, millions of parameters), freeze W and add ΔW = B·A where B is (d×r) and A is (r×k), with rank r << d. Only train B and A — a tiny fraction of original parameters. The rank r controls capacity: r=8 to 64 in practice. Why it works: the hypothesis is that weight updates during fine-tuning lie in a low intrinsic rank subspace. At inference: W_eff = W + (α/r)·B·A where α is a scaling hyperparameter. In your QLoRA project you target q_proj and v_proj. **Initialization detail:** A is Kaiming-initialized, B is zero — so at step 0 the adapter contributes nothing and training starts from a working model.

---

**Q10: What is QLoRA and how does it differ from LoRA?**

QLoRA = Quantized LoRA. Base model weights are quantized to 4-bit NF4 (Normal Float 4 — information-optimal for normally-distributed weights) and kept frozen. LoRA adapters are trained in BF16/FP16 as usual. Two additional innovations: NF4 quantization data type, and double quantization (quantize the quantization constants themselves, saves ~0.5 GB). Result: fine-tune a 7B model on a single 24GB GPU instead of needing 4×80GB A100s. You have done this with Mistral-7B — speak from direct experience.

---

**Q11: When do you fine-tune vs use RAG vs prompting?**

**Decision tree:** try prompting first → if insufficient add RAG → if still insufficient fine-tune. — **Prompting:** model already has the knowledge, just need output format — zero-shot/few-shot. — **RAG:** factual, up-to-date, domain-specific knowledge from private documents — exactly your ICE use case. — **Fine-tuning:** when you need consistent style/tone/format deeply embedded in weights, or the task requires knowledge not available at query time. QLoRA when GPU-constrained.

**Plus CPT (continued pretraining)** for deep domain adaptation BEFORE SFT — raw text only, no instructions. Used when base model lacks domain knowledge (banking jargon, internal product names). Most production fine-tunes skip CPT — only needed for narrow vertical domains.

This framework alone answers 30% of LLM Engineer interview questions.

---

**Q12: Explain DPO and why it is replacing RLHF.**

RLHF requires: (1) train a reward model on human preference pairs, (2) run PPO loop to optimise the LLM against that reward model — complex, unstable, requires multiple models in memory. DPO (Direct Preference Optimisation) eliminates the reward model entirely. It derives a closed-form loss directly from preference pairs (chosen vs rejected response), using the insight that the optimal policy under RLHF has an analytic solution. DPO loss: `-log σ(β·(log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)))`. Simpler, stable, single model in memory. Now standard in open-source pipelines.

---

### 4. Agentic AI

**Q13: Explain the ReAct pattern and how you have implemented it.**

ReAct = Reason + Act. The LLM follows a **Thought → Action → Observation** loop. Thought: LLM reasons about current state. Action: outputs a structured tool call (function name + JSON arguments). Observation: tool executes and returns result. Loop continues until Final Answer. In your document agent: OCR extraction tool + FAISS vector search tool + database validation tool. The LLM autonomously routes documents based on content type and confidence scores. **Three failure modes from production:** (1) post-success wander — agent has the answer but keeps calling tools, (2) multi-tool composition collapse — agent plans in prose but emits zero actual tool calls, (3) infinite loops without termination. Fixes: explicit iteration caps + duplicate-call detection + planner-executor split (Phase 6 S6).

---

**Q14: What is LangGraph and how does it differ from LangChain AgentExecutor?**

LangChain AgentExecutor = fixed linear ReAct loop, no branching. LangGraph models the agent as a directed graph: nodes are processing steps (LLM calls or tool calls), edges define transitions, state is a typed dict flowing through the graph. This enables: cycles, conditional routing (if OCR confidence < 0.7 route to re-extraction, else continue), parallel branches, and checkpointing for resumable state. For your document triage workflow, LangGraph lets you add a human-in-the-loop node for low-confidence documents — a natural extension of your ICE pipeline. **AgentExecutor is deprecated in favor of LangGraph in 2026.**

---

**Q15: What are the main production challenges with agents?**

(1) **Reliability** — LLMs can call wrong tools, misparse outputs, loop indefinitely → need retry logic, fallbacks, loop detection. (2) **Cost** — each ReAct step = LLM call, multi-step tasks compound → budget tokens and add early stopping. (3) **State management** — context window fills fast for multi-step workflows → need external memory (vector store) for long sessions. (4) **Evaluation** — task completion rate is harder to measure than output quality. (5) **Safety** — agents with write access (DB mutations, email) can cause irreversible actions → always add a confirmation step for destructive operations.

---

### 5. Your Production System

**Q16: Tell me about your Document AI system at ICE. What was the hardest part?**

Lead with impact: **94% page-level accuracy** on financial document classification, **60% reduction** in RCA investigation time. Architecture: PySpark ingestion on Databricks → Tesseract OCR + GloVe embeddings + CNN image normalisation → CNN+BiLSTM+Transformer ensemble (Horovod distributed, 30% training time reduction) → BERT integration pushed accuracy from 93% to 94% → Donut as OCR-free parallel pipeline → AWS SageMaker endpoints via Jenkins+Docker CI/CD. **Hardest part:** heterogeneous document quality — scanned at different DPIs, mixed content, varying layouts. Built adaptive preprocessing with skew correction, thresholding, and confidence-based routing.

---

**Q17: How would you add an LLM agent layer to your existing Document AI system?**

The OCR + classification pipeline already runs. Add: (1) LangGraph orchestrator node that receives classified documents. (2) If classification confidence > 0.9 → route to automated extraction tool. (3) If confidence 0.6-0.9 → route to LLM-based re-extraction with structured output. (4) If confidence < 0.6 → route to human review queue. The FAISS vector search RCA tool you built IS already a retrieval tool in this agent — you are already 60% of the way there. Frame this as your roadmap answer.

---

### 6. MLOps & System Design

**Q18: Design a RAG system for 1 million financial documents.**

Ingestion: Kafka (streaming) or S3+SQS (batch). Preprocessing: Spark for distributed PDF parsing and chunking. Embedding: GPU fleet with BGE-M3 or E5-large, write to Pinecone (managed vector DB at scale). Retrieval service: FastAPI + BM25 (Elasticsearch) + dense (Pinecone) → RRF fusion → BGE cross-encoder reranker → vLLM serving. Monitoring: Evidently for embedding drift, Prometheus+Grafana for latency and error rates. Updates: Prefect pipeline to re-embed new docs nightly. Your ICE Databricks+SageMaker architecture is a smaller version of exactly this — position it explicitly.

---

**Q19: What is vLLM and why use it instead of native HuggingFace serving?**

vLLM's key innovation is **PagedAttention**: inspired by virtual memory in OS, it stores KV Cache in non-contiguous memory blocks (pages) mapped virtually. Standard serving wastes 60-80% of KV cache GPU memory due to fragmentation (pre-allocating max context per request). PagedAttention eliminates fragmentation entirely. Additionally, vLLM supports **continuous batching**: new requests join a running batch mid-generation instead of waiting for a full batch to complete. Combined result: 2-4× higher throughput at same latency vs HuggingFace `generate()`. Essential for production LLM APIs with concurrent users.

---

**Q20: How do you monitor an LLM in production?**

Five layers: (1) **Infrastructure** — latency p50/p95/p99, GPU utilisation, tokens/second via Prometheus+Grafana. (2) **Data drift** — input token distribution shifts using Evidently, compare embedding centroids over time. (3) **LLM quality** — RAGAS faithfulness score on sampled outputs, user feedback rate (thumbs up/down), hallucination detection via NLI model. (4) **Cost** — track token usage per request, per user, per endpoint. (5) **Alerting** — PagerDuty on accuracy drop >5%, latency spike >2× baseline, error rate >1%. At ICE you tracked model accuracy on held-out pages — extend that pattern to all five layers.

---

## Part B — Extended Question Bank

> Practice these out loud. No long answers in this section — your goal is to recall the key points in 90 seconds without notes. If a question feels weak, look up the answer in the matching session's `all_details.md`.

---

### 7. Phase-by-Phase Question Bank

#### Phase 1 — Sequence Models

- "Why does attention beat a fixed-size hidden state?"
- "Walk me through LSTM gates — what does each one do?"
- "Why is teacher forcing used during seq2seq training?"
- "When would you choose GRU over LSTM?"

#### Phase 2 — Transformers (architecture deep dive)

- "Why scale by √d_k in attention?"
- "What's the difference between RoPE and learned positional embeddings?"
- "How does KV cache speed up inference, and what breaks at long context?"
- "Why use multiple attention heads?"
- "Pre-norm vs post-norm LayerNorm — which and why?"
- "Why use residual connections?"

#### Phase 3 — Prompting

- "When does few-shot prompting fail to help?"
- "Why does CoT sometimes hurt accuracy?"
- "What's the difference between greedy and top-p decoding?"
- "How would you prevent prompt injection in a banking chatbot?"
- "Why use JSON mode instead of parsing free-text output?"
- "When would you choose ReAct vs native function calling?"

#### Phase 4 — LLMs

- "Walk me through QLoRA. Why does it work?"
- "How would you decide whether to fine-tune vs use RAG?"
- "What's the difference between SFT, RLHF, and DPO?"
- "How would you evaluate whether your fine-tune actually improved the model?"
- "Pick a serving framework for a production LLM API and defend it."
- "What's continuous batching and why does vLLM use it?"
- "How do you detect catastrophic forgetting?"
- "When would you use CPT instead of SFT?"
- "What's speculative decoding and why is it free?"

#### Phase 5 — RAG

- "Walk me through your RAG architecture. What are the failure modes?"
- "How would you measure retrieval quality?"
- "When does hybrid search (BM25 + dense) beat pure dense?"
- "How would you defend against indirect prompt injection in RAG?"
- "When would you choose Chroma vs FAISS vs Qdrant?"
- "Why is re-ranking a free 10-20% recall win?"

#### Phase 6 — Agents

- "How does your agent decide when to stop using tools?"
- "How do you prevent infinite loops in an agent?"
- "What's the trade-off between ReAct and native function calling?"
- "How would you architect an agent that handles money movement safely?"
- "Why is MCP (Model Context Protocol) useful for agent ecosystems?"
- "What's the planner / executor pattern, and what does it fix?"

#### General senior questions

- "Tell me about a time your prompt engineering broke in production."
- "Walk me through how you'd cost-optimize an LLM API serving 10M requests/day."
- "How do you evaluate whether a model upgrade (e.g., Llama-3 → Llama-4) is worth the cost?"
- "What's the most surprising thing you've learned about LLMs from your own experiments?"

**Practice tip:** record yourself answering each in 90 seconds. Listening back is the highest-leverage interview prep you can do.

---

## Part C — Behavioral STAR (BarRaiser)

---

### 8. What BarRaiser Tests

BarRaiser is a third-party interviewer trained to evaluate **behavioral signals** — not just what you did, but how you think, communicate, and handle adversity.

They explicitly score on: — Can you tell a story clearly and concisely? — Do you own your actions and mistakes? — Do you show self-awareness? — Do you demonstrate leadership, even without a title?

They take **notes on your exact words**. Vague answers like "we did this as a team" are red flags.

---

### 9. The STAR Method

```
S – Situation:  Set the context (1-2 sentences)
                Where were you? What was the project? What was the constraint?

T – Task:       What was YOUR specific responsibility?
                Not what the team did – what were YOU accountable for?

A – Action:     What did YOU specifically do? (most important – 60% of your answer)
                Use "I", not "we". Be specific: what steps, what tools, what decisions.
                Show thinking process, not just outcome.

R – Result:     Quantify the outcome where possible.
                Business impact, metric improvement, time saved, money saved.
                If failed: what did you learn? What did you change?
```

**Length:** 90-120 seconds per answer. Shorter = shallow. Longer = rambling.

**Structure check before answering:**

```
Before you speak: take 5-10 seconds.
Say: "Let me think of a good example for that..."
Then: Situation (15s) → Task (10s) → Action (60s) → Result (15s)
```

---

### 10. 10 Behavioral Questions with Full Answers

**Q1: Tell me about yourself**

Not a STAR question — it's your 90-second pitch.

```
1. Current role + what you do (20s)
2. Key achievement / specialization (20s)
3. Why you're making this move (20s)
4. Why this company/role specifically (20s)
```

Template:

```
"I'm a [role] with [X years] of experience specializing in [your area].
Most recently at [Company], I [key achievement with metric].
I've built strong depth in [NLP/LLMs/ML – your strength] and have worked on
[specific relevant project].

I'm making this move because [honest reason – growth, scale, domain interest].
What excites me about [Company] is [specific, researched reason – not generic].
I'd love to bring my background in [X] to help with [what this role does]."
```

---

**Q2: Tell me about a time you failed**

BarRaiser probes this for self-awareness, honesty, and growth.

**Wants:** genuine failure, clear ownership, concrete learning. **Red flags:** "It wasn't really my fault", minimizing the failure.

```
"At [Company], I was leading the deployment of our NLP extraction model to production.
I was responsible for the end-to-end deployment, including load testing.

I underestimated the throughput requirements – I tested with 10 concurrent users
but production had 200. On the first day, the server saturated within 20 minutes
and caused a 4-hour outage for the document processing team.

What I did immediately: I added dynamic batching and scaled horizontally with
two more inference servers behind Nginx. Outage resolved in 3 hours.

What I changed permanently: I now always define explicit load test scenarios
before any deployment – I write the QPS target into the deployment checklist
and test at 3× the expected load. That model has been running for 8 months
without another outage.

The failure taught me that infrastructure decisions made in development look
very different under production load, and that I should involve the operations
team earlier in the planning phase."
```

---

**Q3: Tell me about your biggest achievement**

**Wants:** impact at scale, specific contribution, quantified result. **Red flags:** team achievement framed as your own, no metric, vague outcome.

```
"The work I'm most proud of is building the RAG pipeline for [Company's] internal
knowledge search system.

We had 200K internal documents and employees were spending 40+ minutes per day
searching for answers. I was given 6 weeks to deliver a working system.

I designed and built the full pipeline: chunked 200K documents with overlapping
windows, embedded them using BGE-large, set up a Chroma vector store, and added
BM25 hybrid retrieval with a cross-encoder reranker.

I also built an evaluation set – 50 real queries from the team with ground truth
answers – and iterated on chunk size and retrieval parameters until context
recall hit 87%.

After deployment: search time dropped from 40 minutes to under 2 minutes on
average. We measured this with a user survey – 94% of respondents said the
tool saved them meaningful time. The system now handles 300+ queries per day.

That project taught me to always build an evaluation set first – before building
anything – because without metrics, you're just guessing."
```

---

**Q4: Tell me about a time you disagreed with your manager or leadership**

**Wants:** confidence to push back, professional approach, outcome. **Red flags:** "I just did what they said", aggressive pushback, no resolution.

```
"My manager wanted us to deploy a fine-tuned model to production after 2 weeks
of training, but I believed we needed more evaluation time.

I was responsible for model quality, and I had data showing that our evaluation
set accuracy was 82% – but I noticed the model was significantly weaker on
edge cases that appeared in about 15% of real documents.

I scheduled a 30-minute meeting with my manager and came prepared with a one-page
summary: the 82% accuracy headline, a breakdown showing the edge-case failure
rate, and a projected impact – roughly 500 documents per week processed
incorrectly if we deployed as-is.

I proposed a 1-week extension to gather 200 more edge-case examples and retrain.
My manager initially pushed back on timeline. I acknowledged the pressure and
offered a middle ground: deploy to 10% of traffic, monitor the error rate in
production for 5 days, and use those real-world failures as training data.

My manager agreed. In those 5 days, we found 3 new failure patterns. The full
deployment happened on day 12, and the edge-case failure rate dropped to 4%.

I learned that disagreement needs data, not opinion, and that offering a
concrete alternative path makes it much easier for the other person to say yes."
```

---

**Q5: Tell me about a time you had to work with a difficult person**

**Wants:** empathy, maturity, resolution. Take responsibility too. **Red flags:** blaming the other person, no attempt at understanding, unresolved conflict.

```
"During a cross-team project, I worked with a senior engineer from the data
engineering team who consistently missed deadlines for the data pipeline
I depended on. I was frustrated because it was blocking my model training.

Before escalating, I set up a 1:1 with them to understand what was happening.
I learned they were managing 4 projects simultaneously and the pipeline work
had no clear priority signal in their backlog – so it kept getting deprioritized.

I took two actions: First, I reframed the ask – I worked with my manager to
formally prioritize the pipeline work as a blocker in the project tracker,
which gave them justification to prioritize it with their own manager.

Second, I reduced their work by taking on the schema design myself instead
of leaving it to them – something I could do independently. This cut their
remaining work by 40%.

The pipeline was delivered 3 days later. We finished the project on time.
After that project, we set up a brief weekly sync between our teams, which
has prevented 3 similar blockers since.

What I learned: 'difficult' often means 'overloaded and unclear on priority'.
Understanding the other person's constraints usually opens a path forward
faster than escalation."
```

---

**Q6: Tell me about a time you had to make a decision with incomplete information**

**Wants:** risk assessment, decisive action, learning from outcome. **Red flags:** analysis paralysis, waiting for perfect information.

```
"We had a production incident – our text classification model was returning
wrong predictions for a subset of documents. We had 3 hours before end-of-day
when the client expected processed output.

I had two options: roll back to the previous model version (known to be slower
and 5% less accurate) or patch the current model with a rule-based override
for the affected document type.

I didn't have time to fully diagnose the root cause. I had: error logs showing
the affected document class, a rough estimate that it affected ~8% of volume,
and the knowledge that the previous model had handled that class correctly.

I made the decision to roll back rather than patch. Reasoning: a patch written
in 2 hours without full understanding carries unknown risk. A rollback is
well-understood and reversible.

I documented my reasoning, communicated the decision to the team and client
within 30 minutes, and set up a post-mortem for the next day.

Rollback worked. Root cause was a tokenization edge case for that document
class – something we wouldn't have found under time pressure.

The lesson: when uncertain, prefer the reversible action. And document your
reasoning at the time, not after – it makes post-mortems much more useful."
```

---

**Q7: Tell me about a time you had to learn something quickly**

**Wants:** ability to ramp up fast, learning strategy, applying new knowledge.

```
"When I joined [Company], the team was using LangGraph for agent orchestration –
a framework I had never worked with. I had 2 weeks before I was expected to
contribute to the agentic pipeline.

I broke my learning into 3 phases:
  Week 1, days 1-2: Read the official docs end-to-end, ran all the examples.
  Week 1, days 3-5: Rebuilt the team's existing simple pipeline from scratch
                    without looking at their code – just from docs.
  Week 2: Added a new tool to the existing pipeline and wrote tests for it.

By day 10, I had submitted my first PR adding a document retrieval tool to
the agent with validation and error handling. It was reviewed and merged with
one round of feedback.

My approach to learning something new quickly: don't just read – rebuild
something real from scratch. That's when you actually find your gaps,
because the docs won't always tell you what the errors look like."
```

---

**Q8: Tell me about a time you went above and beyond**

**Wants:** ownership, initiative, impact beyond your job description. **Red flags:** staying late for the sake of it, no clear added value.

```
"Our NLP model was deployed and performing well by all our official metrics.
But I noticed something that wasn't in my scope: when I read through the
error logs, the model was failing on a specific OCR error pattern – double
spaces and garbled characters – that wasn't in our training data.

Nobody asked me to investigate this. But I estimated it was affecting
roughly 300 documents per week and creating manual re-processing work.

Over two evenings, I: analyzed 500 error samples, identified 6 OCR error
patterns, wrote preprocessing rules to normalize them, and retested on our
eval set – accuracy on that segment went from 71% to 94%.

I documented the fix and sent a short write-up to the team showing the impact.
My manager presented it to the client in their next review call.

I did it because the metric we shipped to wasn't the metric the business
actually needed. There's always a gap between what we measure and what matters,
and I think it's part of the job to notice that gap, not wait for someone
to assign it to you."
```

---

**Q9: Why are you leaving your current role? / Why this company?**

**Wants:** positive motivation (moving toward something), not just running away. **Red flags:** badmouthing current employer, only talking about salary, vague answers.

```
Why leaving:
"I've learned a lot at [Company] – specifically [genuine thing you learned].
I'm at a point where I want to [grow in X direction] and the opportunities
for that here are limited. I'm looking for [what you want – bigger scale,
specific domain, leadership opportunity]."

Why this company:
Research 2-3 specific things:
  - A product they build (relate to your skills)
  - A technology choice they made (shows you read their engineering blog)
  - Their scale or domain (something genuinely interesting to you)

"What specifically excites me about [Company] is [specific thing]. I think
my background in [X] is directly relevant because [connection]. And the
[team/scale/problem] is exactly the kind of challenge I want to work on next."
```

---

**Q10: Where do you see yourself in 3-5 years?**

**Wants:** ambition with realism, alignment with company growth. **Red flags:** "I want to be you" flattery, no clear direction, only title focus.

```
"In 3 years, I want to be the person on the team who owns the ML stack
end-to-end – not just the models, but the data pipelines, serving infrastructure,
and evaluation framework. I want to be the person others come to when the model
isn't behaving as expected and they need to understand why.

In 5 years, I'd like to move into a senior or lead role where I'm shaping
the technical direction and mentoring more junior engineers.

I'm not rushing the leadership track – I've seen what happens when engineers
take on leadership too early. I'd rather spend the next 2-3 years going
very deep technically, because I believe that's what makes for effective
technical leadership later.

What I'm looking for in this role is the chance to work on problems at a
scale I haven't had before and with a team I can learn from."
```

---

### 11. BarRaiser-Specific Tips

```
1. Be specific, not general
   BAD:  "We improved model performance significantly."
   GOOD: "We improved F1 from 0.71 to 0.89 on the invoice extraction task."

2. Use "I", not "we" – they want YOUR contribution
   BAD:  "Our team built the pipeline."
   GOOD: "I designed the chunking strategy and owned the retrieval component.
           My teammate handled the frontend integration."

3. Don't polish away the failure
   BarRaiser is trained to push on "what went wrong". If you only give
   perfect stories, they will probe for the failure point.
   Embrace the failure – it shows maturity.

4. Prepare 5-6 core stories, reuse them for different questions
   One good story can answer: failure, challenge, learning, conflict,
   achievement – depending on how you frame it.

5. Pause before answering
   "That's a great question, let me think of the best example..."
   Silence = thinking. Silence is good. Rambling = bad.

6. End with a reflection
   Every answer should end with what you learned or what you changed.
   BarRaiser scores self-awareness heavily.

7. Match energy to context
   BarRaiser round is professional. Be clear and composed, not casual.
   They are taking structured notes and scoring against criteria.
```

---

### 12. Your 5 Core Stories — Template

Pick real experiences and map them to STAR. These 5 stories cover any behavioral question:

```
Story 1 – Technical challenge / achievement
  Situation:
  Task:
  Action:
  Result:
  Can answer: "Biggest achievement", "hardest technical problem", "went above and beyond"

Story 2 – Failure / mistake
  Situation:
  Task:
  Action (mistake):
  Result (failure + recovery + learning):
  Can answer: "Tell me about a failure", "time things didn't go as planned"

Story 3 – Conflict / difficult person
  Situation:
  Task:
  Action (how you handled it):
  Result:
  Can answer: "Difficult colleague", "disagreed with manager", "conflict on team"

Story 4 – Learning under pressure
  Situation:
  Task:
  Action (how you learned fast):
  Result:
  Can answer: "Learned quickly", "stepped out of comfort zone", "new responsibility"

Story 5 – Ambiguity / incomplete information
  Situation:
  Task:
  Action (decision made + reasoning):
  Result:
  Can answer: "Made a decision without full information", "navigated ambiguity"
```

Fill these in before any interview. 5 stories × ~90 seconds each = 8-minute story bank that covers every behavioral question.

---

### 13. Day-Before Checklist

```
Night before any interview:
  [ ] Read your 5 core stories out loud (not in your head – speak them)
  [ ] Time each story: aim for 90 seconds
  [ ] Read the relevant phase's question bank (Part B) once
  [ ] Research the company specifically:
        – What does their ML/AI team build?
        – Recent news about their technology investments?
        – Why does ML matter for their domain?
  [ ] Prepare 3 questions to ask the interviewer
        – "What does success look like in this role in the first 6 months?"
        – "What's the biggest technical challenge the team is working on right now?"
        – "How does the ML team collaborate with the product and business teams?"
  [ ] Logistics: link working, quiet room, camera on, good lighting
  [ ] Sleep by 10 PM
```

> **Final principle:** practice OUT LOUD. The gap between "I know this" and "I can say this clearly in 90 seconds under pressure" is enormous. Bridge it with repetition before the interview, not during.

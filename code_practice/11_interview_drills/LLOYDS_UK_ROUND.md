# Lloyds Banking Group — UK Round Addendum

> **Situation:** Rounds 1–2 technical, face-to-face, Hyderabad — **done**. Next round is with the **UK team**, expected to go deep on **LLMs, RAG, Agents**.
> **Read this alongside** [AGENTS_4DAY_PLAN.md](AGENTS_4DAY_PLAN.md). It doesn't replace the plan — it re-weights it and adds the layer the Hyderabad rounds almost certainly didn't test.
>
> *All external facts below are from public reporting as of Sept 2026. Use them to show you did homework — not to recite figures at someone who works there.*

---

## 1. Why This Round Is Different

Two technical rounds have already established that you can do the work. A UK-team round at this stage is usually **architecture, judgment and environment fit** — can you ship this *inside a UK retail bank*, safely, and explain it to people who will be personally accountable for it.

The gap most candidates fall into: they can explain ReAct, RAG and LangGraph perfectly, and have **no idea that a customer-facing LLM agent is a regulated model** in this environment. That gap is your opening.

**Your unfair advantage:** you have already shipped ML in regulated financial services — Al Rajhi Bank and ICE Data Services, Document AI, production ownership including a P1 RCA. Most candidates for LLM roles have shipped a chatbot demo. Lead with the regulated-environment experience and let the agent theory support it, not the other way round.

---

## 2. What Lloyds Actually Runs

| Fact | Why it matters to you |
|---|---|
| **Google Cloud Vertex AI is the technology spine** — migrated 2024, 300+ data scientists, 18+ GenAI systems in production | **Not** Azure/OpenAI. Know Vertex vocabulary. And note: **Claude and 200+ models are served on Vertex** — this is exactly the "direct vs Bedrock vs Vertex" distinction in [`00_agent_stack_foundations.md`](../../8.agents/00_agent_stack_foundations.md) §2, and now you can explain *why* a bank picks that door: spend, data residency, IAM and audit stay inside an existing cloud contract. |
| **Athena** — first large-scale GenAI product. A knowledge assistant for customer-facing colleagues, searching **~13,000 authorised internal knowledge articles**, grounded to that corpus **rather than the open web**. Cut average search time ~66%, 59s → 20s. Serves an org supporting ~28m customers. | **This is your Rulebook-RAG project, at Lloyds scale.** Same shape: retrieval over an authorised policy corpus, grounded, answering staff questions where a wrong answer has regulatory consequence. This is the single strongest connection you can draw in the room. |
| **50+ GenAI solutions and 80 ML use cases into production inside a year** | They are past experimenting. Questions will be about *operating* these, not building a first one. |
| **~£50m value in 2025, targeting £100m+ in 2026** | They measure AI in delivered value. Framing answers around business outcome, not model quality, matches how they talk. |
| **Scaling agentic AI across the organisation is the stated 2026 priority**, plus an AI financial assistant in the mobile app expanding into savings, borrowing and investments | **This is why you're being interviewed on agents.** You are being assessed against a live 2026 programme, not a hypothetical. |
| **Rohit Dhawan** (ex-AWS) is Group Director of AI & Advanced Analytics, running a centralised **AI Centre of Excellence** | Centralised CoE + federated delivery. Reasonable to ask how the CoE and product teams split responsibility. |
| **Vertex AI Agent Builder / Agent Engine** — managed agent runtime that supports **LangGraph** and CrewAI, with ADK, grounding API, Vertex AI Search, IAM-bound execution, persistent sessions, trace viewer | Your Phase 08 LangGraph work **runs on their platform**. Say that. "My LangGraph experience ports directly onto Agent Engine" is concrete and checkable. |

**A quiet gift:** the tools in your own `08_agents/03_langgraph_agent/tools.py` are **UK mortgage policy** — 95% LTV for first-time buyers, Help to Buy, the ERC taper, SVR, affordability stress-tested at rate + 3%. You have been building toy Lloyds systems already. Use those examples when you need to make an agent concept concrete; it will land as domain fluency.

---

## 3. The Regulatory Layer — Your Differentiator

This is the section that wins the round. **Study it properly.**

### SS1/23 — PRA Model Risk Management Principles

In force **17 May 2024**. Deliberately **technology-neutral**, and it **expressly brings AI and ML into scope** — including, in terms, *LLMs used as components within agentic workflows*. Scope is the **entire firm**, not just credit and market risk: an AI chatbot talking to customers is in scope.

A UK bank must evidence **five things for every model**:

| # | Principle | What it means for an LLM agent |
|---|---|---|
| 1 | **Model identification + risk tiering** | Is your agent in the model inventory? Almost certainly yes. What tier — does it influence a customer outcome, or just summarise for an employee? Tiering drives how much validation is required. |
| 2 | **Governance** | Board and senior-management accountability. Under SM&CR a **named individual** carries this. Someone's personal regulatory record is attached to your agent. |
| 3 | **Development, implementation and use** | Documented and justified. **A prompt change is a model change** — it needs versioning, change control and re-testing. So is a tool-schema change, or swapping the underlying model version. |
| 4 | **Independent validation with effective challenge** | The hardest one for GenAI, and the best question you can be asked. A second line must be able to challenge the model independently. |
| 5 | **Mitigants for anything not fully validated** | If you can't fully validate it, you must compensate: HITL gates, restricted scope, tighter monitoring, human fallback, kill switch. |

Stated PRA concerns for AI models specifically: **explainability, data provenance, fairness, accountability.**

### FCA Consumer Duty

Four outcomes — **products and services · price and value · consumer understanding · consumer support**. For AI, two bite hardest:

- **Consumer understanding** — AI-generated explanations must let the customer make an informed decision.
- **Consumer support** — AI must not become a barrier between the customer and the firm. An agent that traps someone in a loop instead of escalating is a Duty failure, not just bad UX.

**Explainability is at the individual decision level.** You must be able to explain *why this customer got this answer* — to the customer, to the FCA, and to the **Financial Ombudsman Service** if challenged. And the Duty expects **outcomes monitoring**: a measurement plan, a monitoring frequency, a defined drift tolerance, and a documented intervention path when outcomes deteriorate.

### The synthesis to have ready

> *"An LLM agent that touches a customer outcome is a model under SS1/23, so it needs inventory entry, risk tiering, documented development, independent validation and mitigants — and separately, under Consumer Duty, I have to be able to explain any individual decision to the customer, the FCA and the Ombudsman. The genuinely hard part is Principle 4. Independent validation of a deterministic scoring model is a known exercise. Validating a stochastic agent means validating a **trajectory distribution**, not a function — so in practice you validate the components you can (retrieval precision, tool-call accuracy, refusal behaviour), you hold out a golden set and measure variance across repeated runs rather than a single pass, you red-team it, and where you cannot get to full validation you fall back to Principle 5 and compensate with human-in-the-loop and scope restriction."*

Say that and you will be a different candidate from everyone else in the pipeline.

---

## 4. Questions To Expect, With Angles

### Q. "How would you build something like Athena?"

Highest-probability question. Answer as the RAG engineer you actually are, then add the bank layer.

Structure: **hybrid retrieval** over the authorised corpus — lexical plus dense, because policy documents are full of exact product names, rates and clause identifiers that embeddings blur, and that's precisely what you found building Rulebook-RAG. **Rerank** with a cross-encoder for top-of-list precision, since the generator only really sees the top few. **Ground hard** — answer only from retrieved articles, cite the article, and refuse when retrieval is weak. Then the bank layer: **an unanswerable question must produce "I don't know, here's who to ask", not a plausible sentence**, because a colleague repeating a hallucinated policy to a customer is a Consumer Duty incident. Version the corpus so you can reconstruct what the system would have said on the date of a complaint. Log every answer with its retrieved sources for audit.

Then name your numbers: **36 classes, BM25 + RRF worth +5.2 points, cross-encoder +3.9, 0 invalid outputs across 775** — and be honest that 0/775 measures *output validity*, not answer correctness.

### Q. "We're scaling agentic AI in 2026. Where would you start?"

Resist the temptation to design the most sophisticated thing. **Start with the workflow/agent distinction** — most of what gets called agentic should be a workflow, because workflows have predictable cost and are far easier to validate under SS1/23. Then: start **internal-facing, not customer-facing**, because the regulatory burden is dramatically lighter and you learn the failure modes on colleagues before customers. Pick a use case with a **verifiable outcome**, so you can build the eval harness before you scale. Put **HITL on anything advisory or irreversible** from day one, not retrofitted. And instrument trajectories from the first day, because you cannot retrofit an audit trail.

### Q. "How do you know your agent is working?"

Never a single number. Outcome metrics *and* process metrics — task success, tool-call accuracy, trajectory efficiency, cost per task, safety violations. Then the bank-specific additions: **variance across repeated runs** (an 80% mean that's bimodal is unusable and unvalidatable), **refusal correctness** (does it decline when it should?), **citation faithfulness** (does the answer actually follow from the retrieved source?), and **outcomes monitoring** with a defined drift tolerance and intervention path, because Consumer Duty asks for exactly that. Offline gates each release; online sampling tracks live quality.

### Q. "What worries you about deploying agents in a bank?"

Lead with **indirect prompt injection**, because it is the one that turns a quality problem into a security incident: content in a retrieved document or a tool result is untrusted text, and if the same agent can both read customer data and take an action, injected instructions become unauthorised actions. Mitigation is **capability isolation** — never give one agent both a broad read tool and a consequential write tool without a human gate. Then: **cost non-determinism**, since an agent's spend per query is unbounded until you cap turns and tokens. Then **explainability under challenge** — when the Ombudsman asks why this customer got this answer six months from now, you need the trajectory, the retrieved sources and the prompt version, which means logging designed in from the start. And **silent degradation** — a model version change or a corpus update can move behaviour with no code change at all, which is exactly why prompt and corpus versioning belong under change control.

### Q. "Have you deployed agents in production?"

Be precise and don't inflate. You have **built and run** four agent sessions — ReAct from scratch, provider-native tool calling, a LangGraph agent with checkpointed multi-turn state, and a multi-agent document workflow. That is project work. Your **production** experience is Document AI in regulated financial services, at volume, including owning a P1 incident RCA across five services and 21,096 pages. Then bridge — the disciplines that make agents shippable in a bank are the disciplines you already practise: per-class thresholds, human review on low confidence, audit trails, change control, and knowing what your false-positive rate costs someone downstream.

### Q. "Why Vertex AI rather than calling OpenAI directly?"

Governance, not capability. The weights are the same wherever they're served; what changes is that spend, IAM, data residency, VPC controls and audit logging stay inside a cloud contract the bank has already assured. For a UK bank that's the difference between a procurement conversation and a two-year one. Add that Agent Engine gives you a managed runtime with request-scoped tracing and IAM-bound execution, which is most of an audit trail you'd otherwise build yourself — and that it runs LangGraph, so agent code isn't locked to the platform.

---

## 5. Where This Changes The 4-Day Plan

Keep the plan's structure. Make these swaps:

| Day | Change |
|---|---|
| **1** | Unchanged. Foundations still first. Add 15 min: in `00_agent_stack_foundations.md` §2, re-read the direct/Bedrock/Vertex row — you now have a live reason to care. |
| **2** | Unchanged and now **more** important. Reliability patterns are the spine of every regulatory answer — Principle 5 mitigants *are* the reliability patterns. When you do block 2.8, map each pattern to **SS1/23 and Consumer Duty** specifically, not generic compliance. |
| **3** | **Cut MCP depth to ~45 min.** Lloyds is on Vertex/Agent Engine; MCP is worth knowing, but it is not their substrate. **Reallocate the saved time to §3 of this file.** Keep agent eval at full depth — it maps straight onto independent validation. |
| **4** | **Add 90 min: the Athena answer and the SS1/23 mapping, out loud.** Rehearse §4 Q1 and the §3 synthesis until they're fluent. Keep 4.1 (rewrite the stale `02_INTERVIEW_PACK.md`) — non-negotiable. |

**New must-haves before you walk in:**
1. The **Athena answer**, 2 minutes, fluent, with your Rulebook-RAG numbers in it.
2. The **SS1/23 five principles**, named, and what each means for an agent.
3. **Consumer Duty explainability** — individual decision level, defensible to customer, FCA and Ombudsman.
4. **Why Vertex**, answered as governance rather than capability.

---

## 6. Questions To Ask Them

Sharper than the generic set, because you know their programme:

1. **"Athena is grounded to internal knowledge articles — how do you handle it when retrieval comes back weak? Refuse, or answer with lower confidence?"** Shows you understand that refusal design is the hard part of grounded RAG in a regulated setting.
2. **"As you scale agentic AI this year, how are agents being treated under SS1/23 — are they going into the model inventory, and how are you approaching independent validation?"** This is the question a senior engineer in a bank actually argues about. It will change how they see you.
3. **"Where's the line between the AI Centre of Excellence and the product teams — does the CoE own the platform and patterns, or the delivery too?"**
4. **"For the customer-facing assistant, where does human-in-the-loop sit today?"**
5. **"Agent Engine supports LangGraph — is that the house pattern, or is ADK preferred for new builds?"** Signals you know their stack at implementation level.

---

## 7. Tone Notes For A UK Panel

- **Understate, then evidence.** "I've done a fair amount of this" followed by specifics reads better here than a confident global claim. Overclaiming is penalised harder in UK interviews than in many markets.
- **Say what you'd lose.** Every design choice has a cost. Naming it unprompted is the single clearest seniority signal.
- **Be blunt about limits.** *"I haven't run that in production"* is a strong sentence when followed by a real bridge. In a bank, someone who overstates their experience is a risk, and the panel is calibrated to detect it.
- **Domain fluency travels.** You know what LTV, affordability stress-testing and early repayment charges are. Use the vocabulary naturally — it signals you won't need six months of domain onboarding.

---

## Related

- 4-day plan → [AGENTS_4DAY_PLAN.md](AGENTS_4DAY_PLAN.md)
- Question bank → [AGENTS_QA_BANK.md](AGENTS_QA_BANK.md)
- Stack foundations (Vertex/Bedrock/direct) → [../../8.agents/00_agent_stack_foundations.md](../../8.agents/00_agent_stack_foundations.md)
- Reliability patterns (= SS1/23 Principle 5 mitigants) → [../../8.agents/02_agent_reliability_patterns.md](../../8.agents/02_agent_reliability_patterns.md)
- Agent evaluation (= independent validation) → [../../8.agents/09_agent_evaluation.md](../../8.agents/09_agent_evaluation.md)
- Indirect prompt injection → [../../7.rag/03_indirect_prompt_injection.md](../../7.rag/03_indirect_prompt_injection.md)
- Production RAG ops → [../../10.mlops/13_production_rag_ops.md](../../10.mlops/13_production_rag_ops.md)
- Multi-tenant RAG → [../../11.system_design/10_multi_tenant_rag.md](../../11.system_design/10_multi_tenant_rag.md)

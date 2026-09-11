# Agents — 20 Spoken Answers

> **This file is for your mouth, not your eyes.** Every answer below is written the way you'd *say* it, at roughly 60–90 seconds. Read one, close the file, say it in your own words, then check what you dropped.
>
> Companion files: [AGENTS_QA_BANK.md](AGENTS_QA_BANK.md) for one-liners and wider coverage · [LLOYDS_UK_ROUND.md](LLOYDS_UK_ROUND.md) for the regulatory layer · [AGENTS_4DAY_PLAN.md](AGENTS_4DAY_PLAN.md) for the schedule.

---

## The four moves

Every strong answer below does the same four things in the same order. Learn the **pattern**, not the wording — it works on questions nobody predicted.

| Move | What it does |
|---|---|
| **Define** | One crisp sentence. No preamble. |
| **Distinguish** | What it *isn't*. This is where you take the opening the question gives you. |
| **Tradeoff** | What you give up. **Skipping this is the single most common downgrade from senior to mid.** |
| **Judgment** | What you'd actually do. Makes you sound like someone who has shipped, not read. |

**Do not memorise these verbatim.** Interviewers hear recitation instantly. Own the four moves and use your own language.

---

# Block A — Agents

## A1. When you say "agent", what do you actually mean?

> For me an agent is **an LLM in a loop, with tools.** Three parts: the model decides what to do next, it can call tools to actually do things in the world, and the loop keeps going until the model decides it's finished.
>
> The reason the word does so much work is that it gets used for two genuinely different things. A **workflow** has its control flow in my code — I wrote the sequence, it runs the same path every time. An **agent** lets the model direct its own process: it chooses the tools, the order, and when to stop. So the path varies per input.
>
> The tradeoff is concrete. Workflows give you predictable cost, predictable latency, and something you can debug. Agents give you flexibility on tasks where you genuinely can't enumerate the steps up front.
>
> In practice most things marketed as agents are workflows — and that's usually the right call. I'd only reach for a real agent when I can't write the steps down in advance, because I'm trading away cost predictability and auditability to get there. In a regulated setting that trade isn't always available to you.

**Why it works** — Define (LLM + tools + loop, *tools* said out loud). Distinguish (workflow vs agent). Tradeoff (predictability vs flexibility). Judgment (most things should be workflows). The last line shows you understand *their* constraints unprompted.

**Follow-up:** *"So is a three-step LLM script an agent?"* → A2.

---

## A2. I have a script: classify a complaint, draft a response, check it against tone-of-voice policy. Three LLM calls, tools in between. Is that an agent?

> No — that's a workflow, and I'd say that's the right design for it.
>
> The test isn't how many LLM calls there are or whether tools are involved. It's **who decides the sequence.** In your example I decided it: classify, then draft, then check. It runs that path every time, in that order. The model is doing work inside each step, but it isn't choosing the steps.
>
> It'd become an agent if the model could decide — say, look at the complaint and choose whether to pull the customer's history, whether to escalate, whether it needs a second draft. Then the path varies per complaint and I can't predict it in advance.
>
> And for a complaints flow I'd keep it as the workflow. I know the steps, and the fixed path is what lets me tell you the cost per complaint and show a regulator the same process ran every time. That's worth more here than flexibility.
>
> Specifically, what you've described is Anthropic's **prompt chaining** pattern — sequential calls with a check between steps. The check before sending is doing real work.

**Why it works** — Gives the discriminator (*who decides the sequence*), not a definition. Then names the pattern, which signals you know the taxonomy. The refusal to upgrade it to an agent is the judgment.

**Follow-up:** *"When would you upgrade it?"* → when the step sequence genuinely can't be enumerated, and only after measuring that the fixed path is failing.

---

## A3. Walk me through what actually happens in the agent loop.

> Start from a single call, because everything else is that call plus one idea.
>
> You send a list of messages and a list of tool schemas. The model replies, and the important field is **`stop_reason`** — or `finish_reason` on the OpenAI side. If it says the turn ended, you're done, that's the answer.
>
> If instead it says the model wants a tool, the reply contains a tool call: a name and a set of arguments. The model **hasn't run anything** — it can't. It's asking you to. So you look the name up in your registry, run the function, and append the result back onto the message list as a tool-result message.
>
> Then you call again with the longer list. The model now sees its own request and the result, and decides again: another tool, or an answer. That repeats until it stops asking.
>
> Two things people get wrong. First, **memory is that list** — the API is stateless, so you resend the whole history every turn, which is also why cost grows superlinearly across a long run. Second, the model never executes anything itself; every side effect happens in your process, which is exactly where you get to put your guardrails.

**Why it works** — Narrates the mechanism rather than naming it. The "hasn't run anything" point is the one most candidates miss and it's the foundation of the whole security answer.

**Follow-up:** *"Where would you put a human approval step?"* → between the tool request and your execution of it, because that's the only place where the action hasn't happened yet.

---

## A4. What actually breaks when you put an agent in production?

> Five things, and I'd order them by how early they bite.
>
> **Loops** show up immediately. Your demo had three happy-path queries; production has ambiguous ones where the model can't tell it's finished, so it keeps calling. Often the same tool with the same arguments.
>
> **Cost** is next and it's usually a surprise, because nobody priced the growing context. Every turn resends the whole history, so a ten-step run isn't ten times a single call — it's worse.
>
> **Tool-call reliability** third. Hallucinated tool names, argument names that don't exist, sometimes the model writing the JSON into the text instead of the tool-call field so your orchestrator never sees it.
>
> Then **prompt injection through tool output**, once the agent touches anything user-supplied or web-sourced.
>
> And underneath all of it, **no evaluation harness** — so you can't tell whether a prompt change helped or hurt.
>
> The pattern is that demos test the happy path and production is all edge case. The fixes are cheap individually — an iteration cap, deduplicating repeated calls, validating arguments against a schema, a token budget — but they have to be designed in, because you can't retrofit an audit trail.

**Why it works** — Ordered by when they bite, which reads as experience rather than a memorised list. Ends with the generalisation. This is the highest-yield agent answer in 2026 — teams want people who've seen what breaks.

**Follow-up:** *"How do you stop the loop specifically?"* → A5.

---

## A5. How do you stop an agent looping forever?

> Three layers, because no single one is sufficient.
>
> The outer one is a **hard iteration cap** — a fixed maximum number of turns. That's the safety net, not the solution. When it fires you've already failed; you just failed cheaply. And you log every cap hit, because a rising rate tells you which prompts confuse the agent.
>
> The middle layer is **duplicate detection.** Keep a set of the tool calls you've already made — the name plus the arguments, normalised. If the model asks for one it's already made, don't run it again; return the cached result with a note saying it was already called. That signals the model to either use what it has or finish.
>
> The inner layer is a **completion check**: if a turn produced no new information — no tool returned data you didn't already have — you're almost certainly stuck or done, so force a stop.
>
> Together those catch nearly everything. What I'd add on top in a bank is that the cap and the budget need to be **per-task and enforced outside the model**, because anything you ask the model to enforce about itself is advisory.
>
> Being honest: my own tool-calling session has a bare `while True` with no cap — I spotted it going back through the code. It's the first thing I'd fix.

**Why it works** — Three named layers with a reason each. The last paragraph is the strongest part: volunteering a flaw in your own code reads as genuine engineering judgment, and it pre-empts them finding it.

**Follow-up:** *"What would you log?"* → the full trajectory: every tool call, arguments, result, and the prompt version — because six months later you need to reconstruct it.

---

## A6. LangChain, LangGraph, or neither?

> Depends on what I need, and the honest default is **less framework than people assume.**
>
> One LLM call with no tools and no state — just the SDK. A fixed sequence of calls — plain Python functions. A `for` loop is not technical debt, and a function already composes, so I don't take on a framework purely for composition.
>
> Where LangGraph genuinely earns its place is the things that are hard to hand-roll correctly: **checkpointing with crash resume, human-in-the-loop interrupts, and replay** so you can go back to step five and see what the state was. In an enterprise those aren't nice-to-haves — an approval gate and an audit replay are usually requirements.
>
> What it costs you is real: abstraction over the actual prompt the model sees, breaking changes across minor versions, a large dependency tree, and a few hundred milliseconds per chain, which matters if you're under a sub-second SLA.
>
> So: start with the SDK and plain Python, add LangGraph at the point where I need checkpointing, HITL or replay. And I'd rather be able to delete the framework than be unable to explain what it's doing.

**Why it works** — Answers with a decision rule, not a preference. "Framework-deletion judgment" is an explicit senior signal in 2026 hiring. Names both what it buys and what it costs — most candidates only do one.

**Follow-up:** *"What's the difference between the checkpointer and the store?"* → checkpointer is thread-scoped, one conversation, keyed by `thread_id`; the store is cross-thread, keyed by user. Without the checkpointer every call is a fresh conversation; without the store every new thread is a fresh relationship. You need both.

---

## A7. When does multi-agent actually beat a single agent?

> Three cases where I can articulate why, and otherwise I wouldn't.
>
> **Tool overload** — once a single agent has ten or more tools it starts picking badly, and the schemas themselves eat a serious slice of the context before the user's question arrives. Splitting by tool group gives each agent a surface it can actually reason about.
>
> **Genuinely different expertise** — a legal review and a technical implementation have different success criteria, so one agent juggling both does neither well.
>
> **Phased workflows** where each phase has its own definition of done.
>
> Where it loses: short tasks, where coordination overhead dominates. Latency-critical paths, because every hand-off is another round trip. And cost — each agent typically reloads context, so you can be five or ten times a single agent for the same job.
>
> The failure mode I'd watch hardest is **lost-in-translation.** Agent A emits free text, Agent B parses it. Tomorrow A's phrasing shifts slightly, B misparses, and everything downstream acts on wrong data — producing an answer that's wrong but completely plausible. The fix is typed contracts at every hand-off, Pydantic schemas, the same discipline you'd apply between microservices. Free-text between agents looks elegant and is the number one silent failure.
>
> Default is single-agent until I can say *why* not.

**Why it works** — Three cases *with reasons*, then where it loses, then the specific failure mode. The microservices analogy lands well with enterprise engineers.

**Follow-up:** *"How would you control the cost?"* → right-size the models (supervisor strong, workers cheap), namespace context so workers don't load the whole conversation, cache the system prompt, hard budget per task.

---

## A8. How would you evaluate an agent?

> Never with one number — that's the main thing.
>
> There are two sides. **Outcome**: did the user get what they wanted? That's task success against a held-out set, rule-based where the task has a checkable answer, LLM-as-judge where it's open-ended. **Process**: how did it get there? Tool-call accuracy — right tool, right arguments. Trajectory efficiency — step count against the minimum, and whether it looped. Cost per task. Latency.
>
> You need both, because a ninety-percent-success agent at five pounds a task is unshippable, and so is an efficient one that's wrong forty percent of the time.
>
> The one people skip is **variance.** Run each task five or ten times with different seeds. Eighty percent mean can be "eighty percent of the time on every task", which is usable, or "always passes half, always fails half", which is not. Those are completely different systems and a single average hides it.
>
> Two more for a bank: **refusal correctness** — does it decline when it should? — and **citation faithfulness**, does the answer actually follow from the source it cites.
>
> Operationally, offline gates each release, online sampling tracks live quality, and disagreement between them tells you the test set has gone stale.

**Why it works** — Two axes, then the variance point almost nobody raises, then two domain-specific additions. The bimodal-vs-uniform distinction is a genuine discriminator.

**Follow-up:** *"Why can't you just measure success rate?"* → it hides cost, latency, trajectory waste, variance and safety. Production reports a vector with a threshold per dimension.

---

# Block B — RAG

## B1. How would you build something like Athena — a knowledge assistant over our internal policy documents?

> I've built close to this, so let me describe what I did and then what I'd change for your scale.
>
> Retrieval is **hybrid** — lexical and dense together, fused with reciprocal rank fusion. That's not a default I picked up from a blog; policy documents are full of exact product names, rates and clause references, and dense embeddings blur precisely those. On my own rulebook project, adding BM25 and RRF over dense-only was worth about five points. Then a **cross-encoder reranker** on the top candidates, worth another four — because the generator only really reads the top few, so precision at the top is what matters, not recall at fifty.
>
> Then the part that matters more in a bank than the retrieval: **grounding and refusal.** Answer only from retrieved passages, cite the article, and when retrieval comes back weak, say so rather than generate. A colleague repeating a hallucinated policy to a customer isn't a quality bug, it's a Consumer Duty incident.
>
> Around it: version the corpus, so you can reconstruct what the system would have said on the date of a complaint. Log every answer with its retrieved sources. And evaluate retrieval separately from generation, because otherwise you can't tell which half is failing.
>
> At your scale the thing I'd want to understand first is permissioning — whether every colleague should see every article, because that becomes a filter at retrieval time, not a post-hoc check.

**Why it works** — Leads with experience and real numbers, justifies each choice with a *reason* rather than naming a technique, then pivots to the regulated-domain concerns. The closing question shows you're thinking about their problem.

**Follow-up:** *"What does 'weak retrieval' mean concretely — what's your threshold?"* → be honest: reranker score below a tuned cutoff, calibrated on a labelled set, and you tune it against the cost of a wrong answer versus an unnecessary escalation.

---

## B2. Why hybrid retrieval? Why not just use embeddings?

> Because they fail on different things, and in a policy corpus the failures are the cases that matter.
>
> Dense retrieval is strong on meaning. Someone asks about "penalties for paying off early" and the document says "early repayment charge" — different words, same concept, embeddings handle that.
>
> What embeddings are bad at is **exact tokens**: product codes, clause numbers, a specific rate, a named scheme. Those get smeared into the neighbourhood of similar strings. BM25 is the opposite — it has no idea that "car" and "automobile" are related, but if the user types an identifier it finds the document containing that identifier.
>
> So you run both and fuse the rankings with RRF. The thing I like about RRF is that it uses only **ranks, never scores**, which is why it needs no normalisation between two retrievers with completely incompatible score scales. And a document that one retriever nearly missed gets rescued by the other — that promotion is the entire value.
>
> Cost is one extra index and a few milliseconds; BM25 is cheap. Typical gain on standard benchmarks is a few points of recall, though it's very dataset-dependent — it's largest exactly where the corpus is full of identifiers, which is the case here.

**Why it works** — Explains the *mechanism* of each failure rather than asserting "hybrid is better". The rank-not-score point about RRF is a detail that signals you implemented it rather than imported it.

**Follow-up:** *"How would you weight them?"* → RRF with k=60 as the default; tune on a labelled set if one retriever is clearly stronger on your corpus.

---

## B3. What does reranking actually buy you?

> It converts recall into precision, and it's the cheapest big win in a RAG pipeline.
>
> First-stage retrieval uses a **bi-encoder**: the query and every document are embedded separately, so document vectors precompute offline and search is fast over millions. But the query and the document never interact — the model never sees them together.
>
> A **cross-encoder** puts the query and one document through a single forward pass, so attention runs across both. Far more accurate, because it can see that this specific passage answers this specific question. The cost is one model call per candidate at query time, so you can't run it over the whole corpus.
>
> Hence two stages: bi-encoder retrieves fifty cheaply, cross-encoder reranks those down to three or five. You get close to cross-encoder accuracy at close to bi-encoder latency.
>
> Why it matters more than it looks: the generator only reads the top few passages. Being right at rank forty is worth nothing. On my rulebook project the reranker was worth about four points, which was the second-largest single improvement after hybrid retrieval.
>
> I'd skip it if the latency budget were very tight or top-one retrieval were already above ninety percent — measured, not assumed.

**Why it works** — Mechanism first (separately vs jointly encoded), then the architecture it forces, then why it matters, then when you'd skip it. That last part is the judgment move.

**Follow-up:** *"What's your latency budget for that?"* → know your numbers; a cross-encoder over 50 candidates is tens of milliseconds on GPU, more on CPU.

---

## B4. What's the difference between RAG and agentic RAG?

> Classic RAG is a **workflow**: retrieve, then generate, same path every time. That predictability is a feature — you can evaluate retrieval and generation separately, and you can tell me the cost per query.
>
> **Agentic RAG** makes retrieval a **tool the model chooses to call.** So it can decide whether to retrieve at all, reformulate a query that came back weak and try again, or issue several targeted queries instead of one broad one.
>
> What you buy is adaptivity on multi-hop and ambiguous questions — the ones where a single query was never going to work.
>
> What you pay is significant. Latency and cost both become non-deterministic. And evaluation gets much harder, because you're now scoring a **trajectory** rather than a single retrieval — a bad first query can cascade, and your neat separation of retrieval quality from answer quality is gone.
>
> So I'd start classic, measure where it actually fails, and add agency at that specific failure — usually query reformulation first, which gets most of the benefit for one extra call. In a regulated setting I'd also want to know that the non-determinism is acceptable before I introduce it.

**Why it works** — Reuses the workflow/agent frame from A1, which makes you sound coherent across the whole interview. The "start classic, measure, add agency at the failure" line is the judgment.

**Follow-up:** *"How would you evaluate the agentic version?"* → trajectory metrics: number of retrieval calls, whether reformulation improved the result, and end-to-end faithfulness — plus variance across repeated runs.

---

## B5. How do you evaluate a RAG pipeline?

> Separately, at both stages — otherwise you can't tell which half is broken.
>
> **Retrieval on its own** needs a labelled set of query-to-relevant-document pairs. Recall at k tells you whether the answer was even available; that's the ceiling on everything downstream, because generation cannot produce what wasn't retrieved. MRR and NDCG tell you whether it was near the top.
>
> **Generation** I'd frame with the RAGAS four: **faithfulness** — is every claim in the answer supported by the retrieved context, which is your hallucination detector. **Answer relevancy** — did it address the question asked. **Context precision** — were the retrieved chunks actually useful. **Context recall** — did you retrieve everything needed.
>
> The reason those four are worth knowing as a set is that each points at a different fix. Faithfulness low means tighten the prompt or add a verifier. Context recall low means increase k or fix chunking. You're not just scoring, you're diagnosing.
>
> If there's no labelled data — which is normal — generate synthetic question-answer pairs from your own documents and hand-review a sample.
>
> For a bank I'd add refusal correctness and the ability to reproduce any historical answer, which means versioning the corpus, not just the code.

**Why it works** — Separates the two stages, then explains why the four metrics are a *set* (each maps to a fix), which is more than reciting them. The no-labelled-data answer is practical.

**Follow-up:** *"What's your faithfulness threshold?"* → gate around 0.80 as a starting point, tuned against how costly a wrong answer is in your setting.

---

# Block C — LLMs

## C1. RAG, fine-tuning, or long context?

> They solve different problems, and the mistake is treating them as competing options.
>
> **RAG is for knowledge** — facts that change, documents, private data, anything where you need to show where the answer came from. You update a corpus instead of retraining, and you get attribution for free, which in a regulated setting is often the deciding factor on its own.
>
> **Fine-tuning is for behaviour** — format, tone, domain vocabulary, refusal patterns. Things you'd struggle to specify in a prompt reliably. It teaches the model *how to respond*, not *what's true*.
>
> **Long context** is the one people forget. If the whole corpus fits in the window and the cost is acceptable, pasting it in is simpler than any retrieval stack, and simpler is worth a lot. It stops working on corpus size and repeated-query cost, and you lose attribution.
>
> The rule I'd use: knowledge problem, RAG. Behaviour problem, fine-tune. Small corpus, just paste it. And they compose — a common production shape is fine-tuning for format on top of RAG for facts.
>
> The failure I'd flag is fine-tuning to inject knowledge. It sort of works, it's expensive, and you can't update or cite it.

**Why it works** — Frames as different problems, not a ranking. Including long context shows you're not defaulting to the fashionable answer. The final warning is a real mistake you've clearly thought about.

**Follow-up:** *"Where's the cutover from long context to RAG?"* → when the corpus stops fitting, or when per-query cost of resending it beats the cost of building retrieval.

---

## C2. What's LoRA and why does it work?

> LoRA is parameter-efficient fine-tuning. Instead of updating a weight matrix directly, you freeze it and learn a **low-rank decomposition of the update** — two thin matrices whose product is the change. You train only those.
>
> The reason it works is the empirical claim underneath: fine-tuning updates have **low intrinsic rank.** Pretraining made the model general, so the weights span a broad space, but adapting to one narrow task only needs to shift things along a few directions. If you take a fully fine-tuned model and decompose the difference, most of the signal is in the first handful of singular values.
>
> The practical consequences are what matter. Trainable parameters drop by orders of magnitude, so optimiser state drops with it — that's where the memory actually goes. The base weights are frozen, which limits catastrophic forgetting. And you can hold one base model and swap small adapters per task.
>
> Two details worth knowing: **B initialises to zero** so the update is exactly zero at step one and you start from pretrained behaviour — but A must be non-zero or no gradient ever flows. And merging back is **mathematically exact**, so there's no inference overhead once merged.
>
> QLoRA then quantises the frozen base to four-bit, which is what brings a 7B into consumer-GPU range.

**Why it works** — Mechanism, then *why* the mechanism is valid, then consequences. The B-zero/A-nonzero asymmetry is a detail that only comes from implementing it.

**Follow-up:** *"Have you run it?"* → be exact: implemented end to end and worked the arithmetic by hand; the training runs are blocked on a torch/CUDA version conflict on a GTX 1650 Ti. Do not claim a completed production fine-tune.

---

## C3. Why isn't one evaluation metric enough?

> Because every single number hides a different failure, and they're not correlated.
>
> Take success rate. Ninety percent sounds shippable. It hides **cost** — if that's five pounds a task, it isn't. It hides **latency** — if p99 is sixty seconds, it fails the UX. It hides **trajectory** — succeeding via fifty redundant calls is fragile and expensive even when the answer is right. It hides **variance** — ninety percent could be uniform, or it could be a system that always passes some tasks and always fails others, which are completely different products. And it hides **safety** — one forbidden action in a thousand is still a release blocker.
>
> So production evaluation reports a **vector** with an explicit threshold per dimension, and a regression on any of them blocks the release.
>
> The other half is that automatic metrics measure the wrong thing on open-ended output. BLEU and ROUGE score surface overlap, so a correct paraphrase scores badly. LLM-as-judge is better but has known biases — position, verbosity, self-preference — so you randomise order and calibrate against human labels on a sample.
>
> And whatever you use, hold out a set that never touches prompt tuning, or your metric quietly becomes a training signal.

**Why it works** — Takes one metric and dismantles it five ways, which is far more convincing than listing metrics. Ends with the leakage point, which is the mistake most teams actually make.

**Follow-up:** *"How would you catch contamination?"* → held-out proprietary sets over public benchmarks, and watch for a suspiciously large gap between public benchmark scores and your own.

---

# Block D — Enterprise

## D1. We put an agent in front of customers. What governance applies?

> Two regimes, and they ask for different things.
>
> Under the **PRA's SS1/23** an LLM agent that touches a customer outcome is a **model**, so it lands in the model inventory with a risk tier. The supervisory statement is deliberately technology-neutral and explicitly covers AI and machine learning — including LLMs used as components inside agentic workflows. Which means the five principles apply: identification and tiering, governance with named senior accountability, documented development and use, independent validation with effective challenge, and mitigants where you can't fully validate.
>
> Under **Consumer Duty** the requirement is different — it's about outcomes. You have to be able to explain an **individual decision** in terms the customer understands, and defend it to the customer, to the FCA, and to the Ombudsman if it's challenged. Plus outcomes monitoring with a defined drift tolerance and a documented intervention path.
>
> The genuinely hard one is **independent validation.** Validating a deterministic scoring model is a known exercise. A stochastic agent means validating a **trajectory distribution**, not a function. In practice you validate the components you can — retrieval precision, tool-call accuracy, refusal behaviour — you measure variance across repeated runs rather than a single pass, you red-team it, and where you can't reach full validation you fall back to Principle 5 and compensate: human-in-the-loop, restricted scope, tighter monitoring.
>
> One practical consequence people miss: a **prompt change is a model change.** It needs versioning, change control and re-testing, same as a coefficient.

**Why it works** — This is the answer almost no candidate has. Names both regimes precisely, then goes straight to the hardest part instead of stopping at the list. The prompt-change line is the detail that proves you've thought it through rather than read a summary.

**Follow-up:** *"Who signs it off?"* → under SM&CR a named senior manager carries personal accountability, which is why the approval gate is a control and not a UX nicety.

---

## D2. How do you control cost in an agent system?

> Four levers, roughly in order of return.
>
> **Model routing** first. Not every step needs a frontier model. Classification and routing go to something cheap; the expensive model only handles the steps that need judgment. That's usually the biggest single win and it's mostly free.
>
> **Prompt caching** second. In an agent loop the system prompt and the tool schemas are byte-identical every single turn, and you're resending them every turn. Cache that prefix and the repeated portion costs a fraction. This is the one people forget, and in a loop it's substantial.
>
> **Context discipline** third. Truncate tool outputs — a full CSV coming back from a tool will blow the window on its own. Compact old turns. If it's multi-agent, namespace the state so each worker sees what it needs rather than the whole conversation.
>
> **Hard budgets** fourth: cap iterations *and* cumulative tokens per task, and abort-and-escalate rather than letting a stuck agent burn. That's a control, not an optimisation.
>
> The framing I'd use is that agents cost roughly an order of magnitude more than a single call because every turn resends a growing history. So the metric that matters is **cost per completed task**, not cost per call — a cheaper model that needs three more turns isn't cheaper.

**Why it works** — Ordered by return, with a reason each. The cost-per-completed-task reframe at the end is the senior insight and it's the kind of thing a budget-holder repeats back to their boss.

**Follow-up:** *"What's your cost per task target?"* → depends on volume; under about ten pence for high-volume internal, more for low-volume high-value. Say you'd measure before committing.

---

# Block E — Your Own Work

> ⚠️ **Verify every number against your current resume before you walk in.** `02_INTERVIEW_PACK.md` is stale.

## E1. Tell me about your RAG work.

> The project is a rulebook assistant — retrieval over a policy corpus, answering questions across **thirty-six classes** of query.
>
> I wrote the retrieval layer rather than importing one, deliberately, because I wanted to understand where it failed. It's hybrid: **BM25 fused with dense retrieval using reciprocal rank fusion**, which was worth about **five points**, then a **cross-encoder reranker** on top, worth about **four more**. Both of those gains came from the same underlying cause — policy text is full of exact identifiers and rates that embeddings blur.
>
> On output validity, **zero invalid outputs across seven hundred and seventy-five** evaluated cases. I'd be precise about what that measures though: it's constrained output validity — the answer always conformed to the expected structure. It is not a claim about answer correctness, which is a separate and harder measurement.
>
> What I'd do differently: I evaluated retrieval and generation together for too long before separating them, which made it hard to tell which half was costing me. And I'd version the corpus from day one — I added that later, and reconstructing what the system would have answered on a given date is much harder retrofitted.

**Why it works** — Real numbers, each attached to a reason. The voluntary precision about what 0/775 *doesn't* prove is the strongest move in the whole answer — it's the kind of honesty that makes an interviewer trust everything else you said.

**Follow-up:** *"How would you make it agentic?"* → retrieval becomes a tool the model chooses; add a self-check tool and HITL on low-confidence classifications. Then name what you'd lose: determinism, predictable cost, and the clean separable evaluation.

---

## E2. Have you deployed agents in production?

> Not agents, no — and I'd rather be precise about that than blur it.
>
> What I've built and run is four agent implementations: a ReAct loop written from scratch with manual parsing, native tool calling, a LangGraph agent with checkpointed multi-turn state, and a multi-agent document workflow. That's project work, on my own machine.
>
> My **production** experience is Document AI in regulated financial services — at ICE and Al Rajhi Bank. Document classification at volume, with per-class thresholds and human review on low-confidence cases. Including owning a P1 incident root-cause analysis across five services and twenty-one thousand pages, where the real failure rate turned out to be two point three percent.
>
> The reason I don't think that's a weak answer is that the disciplines that make agents shippable in a bank are the ones I've already had to practise: approval gates, audit trails, change control, knowing what a false positive costs someone downstream, and thresholds tuned to the cost of being wrong rather than to a benchmark.
>
> The agent-specific engineering is new to me. The environment it has to survive in is not.

**Why it works** — Answers "no" in the first four words, which buys credibility for everything after. Then pivots to the real experience and explicitly names the transfer. The last two sentences are the whole answer; land them.

**Follow-up:** *"What would worry you most about shipping one here?"* → indirect prompt injection through tool output, because it turns a quality problem into a security incident — and the mitigation is capability isolation, never giving one agent both a broad read tool and a consequential write tool without a human gate.

---

# Before you walk in

**Three habits that carry a conversational round:**

1. **Structure before detail.** *"There are three parts to that — let me take them in order."* Buys thinking time and sounds organised.
2. **Always name the tradeoff.** No technique is free. Saying what you'd lose is the clearest seniority signal available to you.
3. **Say "I haven't shipped that" and bridge.** Never bluff. Enterprise interviewers are calibrated for it, and your bridge from regulated Document AI is genuinely strong.

**Drill method:** read one answer, close the file, say it aloud, then check what you dropped. The gap between reading and saying is the entire exercise. Do the ones you feel *most* confident about too — those are where overconfidence hides.

---

## Related

- [AGENTS_QA_BANK.md](AGENTS_QA_BANK.md) — 32 rapid-fire one-liners + wider coverage
- [LLOYDS_UK_ROUND.md](LLOYDS_UK_ROUND.md) — Lloyds specifics, SS1/23 and Consumer Duty in full
- [AGENTS_4DAY_PLAN.md](AGENTS_4DAY_PLAN.md) — the schedule
- [../../8.agents/00_agent_stack_foundations.md](../../8.agents/00_agent_stack_foundations.md) — the stack, formats, workflows vs agents

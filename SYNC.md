# SYNC — Cross-Machine Handoff

> **Update this file before every `git push`. Read it after every `git pull`.**
> This is the only file that communicates state between Mac and Windows.

---

## Last Session

| Field | Value |
|-------|-------|
| **Machine** | Windows |
| **Date** | 11 September 2026 |
| **What I did** | **Numerical audit of the whole boards 1–6b sequence-model arc — found real bugs in 3 of 5 files, all now script-verified.** (1) `02_rnn`: single arithmetic slip — `a₁` row 1 written `0.400+0.000 = 0.500`. The ENTIRE file (forward, BPTT, weight update, 2nd forward) was built on it. All regenerated. (2) `03_lstm`: gate values were **narrated, not computed** — did not follow from the stated weights at all. Claimed ŷ=0.411 / 69% gradient retention; actual with those weights was ŷ=0.149 / 14.7%. REDESIGNED the weight matrices so the forget gate genuinely saturates, regenerated every number. Now ŷ=0.718, L=0.040, **79.2%** retention. (3) `04_gru`: same class of failure — real retention with the stated weights was **8.6%** (RNN is 9%) vs a claimed 66%. Redesigned `Wz_x` positive / `Wz_h` negative so `z` closes once `h` is non-zero. Now ŷ=0.501, L=0.124, **71.4%**. (4) `05_attention`: math was CORRECT (forward+backward+update all verified) — one typo, `A[sat,sat] 0.206→0.244` (its row didn't sum to 1). (5) `06_transformer`: correct except **a conceptual backward-pass bug** — `∂L/∂V` used a *column* of `A` instead of the *row* feeding `c_mat`; same bug was baked into the Python (`A[:, 3:4]` → `A[3, :]`). It corrupted Wv/Wq/Wk gradients and the whole 2nd-forward section. Fixed: loss 0.454→**0.428**. Also `S[on,cat] 0.485→0.406`. (6) **Granular-arithmetic expansion** — every cell shown, no `≈` shortcuts — across `05_attention`, `06_transformer`, `06b_encoder`, `06c_decoder`. (7) RENAMED `06c_transformer_decoder_end_to_end.md` → `06c_transformer_decoder_cross_attention_end_to_end.md` (git mv; 25 refs across 15 files updated; verified 0 stale). (8) Taught calculus from zero (derivative → chain rule → partials → backprop) — was a self-identified gap. |
| **Files changed** | REWRITTEN (numbers regenerated): `4.nlp/03_sequence_models/02_rnn_end_to_end.md`, `03_lstm_end_to_end.md`, `04_gru_end_to_end.md`. EXPANDED + fixed: `05_attention_end_to_end.md`, `06_transformer_end_to_end.md`, `06b_transformer_encoder_multihead.md`. RENAMED + expanded: `06c_transformer_decoder_end_to_end.md` → `06c_transformer_decoder_cross_attention_end_to_end.md`. REF UPDATES (rename): 11 files in `5.transformers/`, `06b_...md`, `07_decoding_strategies.md`, `MASTERY_PLAN.md`, `SYNC.md`. |

---

## Session 2 — 11 September 2026 (Windows) — AGENTS ARC STARTED

| Field | Value |
|-------|-------|
| **Context** | **INTERVIEW ~15 SEP 2026 — LLOYDS BANKING GROUP.** Rounds 1-2 technical face-to-face in Hyderabad **DONE**; next round is the **UK team**, expected to grill on LLMs/RAG/Agents (per a recently-joined contact). Conversational deep-dive. Budget 6-8 h/day × 4 days. |
| **Lloyds recon** (public sources, Sep 2026) | **Google Cloud Vertex AI is the technology spine** (migrated 2024; 300+ data scientists; 18+ GenAI systems in prod) — NOT Azure/OpenAI. **Athena** = first large-scale GenAI product: grounded RAG over **~13,000 authorised internal knowledge articles** for customer-facing colleagues, search time 59s→20s (-66%), deliberately grounded to internal corpus **not** the open web. **50+ GenAI solutions + 80 ML use cases to production in one year.** ~£50m value 2025 → **£100m+ target 2026**. **Scaling agentic AI is the stated 2026 priority**, plus an AI financial assistant in the mobile app expanding into savings/borrowing/investments. Rohit Dhawan (ex-AWS) is Group Director of AI & Advanced Analytics, running a centralised AI CoE. **Vertex AI Agent Engine runs LangGraph** — Phase 08 work ports directly. |
| **The two connections that matter** | (1) **Athena ≈ Rulebook-RAG.** Same shape — retrieval over an authorised policy corpus, grounded, staff-facing, wrong answers have regulatory consequence. Lead with this. (2) The tools in `08_agents/03_langgraph_agent/tools.py` are **literally UK mortgage policy** (95% LTV first-time buyer, Help to Buy, ERC taper, SVR, affordability at rate+3%). Toy Lloyds systems already built. |
| **What I did** | (1) Walked the existing `08_agents/03_langgraph_agent/` session line by line — found `route_after_llm` is DEAD CODE (never passed to `add_conditional_edges`; the graph uses built-in `tools_condition`) and `app = build_graph(with_hitl=False)`, so the HITL/`interrupt()` path was **built but never exercised**. Do not oversell it in interview. (2) Started `code_practice/12_agents_from_scratch/` — a step-by-step ladder (single call → memory → tool schema → round-trip → loop → multi-tool). Step 1 written and run. (3) **AUDITED all 3,400 lines of `8.agents/`** — content quality is high, 48 interview Q&As already written; the gap was never content. (4) Wrote 3 new files (below). (5) Fixed **34 stale cross-refs** across all 10 files in `8.agents/`. |
| **Files created** | `8.agents/00_agent_stack_foundations.md` (NEW SSOT — the 7-layer stack: model/runtime/host/format/SDK/framework/protocol; the two wire formats side by side; product placement map; **workflows vs agents** + Anthropic's 5 workflow patterns; when to delete the framework). `code_practice/11_interview_drills/AGENTS_4DAY_PLAN.md` (day-by-day, every block ends in SAY IT). `code_practice/11_interview_drills/AGENTS_QA_BANK.md` (32 rapid-fire + 15 new full answers + index to the existing 48 + resume defence). |
| **Refs fixed** | `code_practice/06_agents/` → `08_agents/` with REAL filenames (the 10-session folder never existed; there are 4). `5.llms` → `6.llms`/`7.rag`, `7.mlops` → `10.mlops`, `8.system_design`/`9.system_design` → `11.system_design`. Sessions that genuinely don't exist (MCP, long-term memory, agent eval, production hardening) now say **"not yet built"** instead of pointing at nothing. Verified: **77 links checked, 0 broken.** |
| **Gaps research found** (not in repo, now written) | **Workflows vs agents** — the #1 framing question, absent entirely. **Anthropic's 5 workflow patterns** (prompt chaining / routing / parallelization / orchestrator-workers / evaluator-optimizer) — repo had only the academic planners. **LangGraph Store vs Checkpointer** (cross-thread vs thread-scoped) — a named interview question. **MCP control dynamics** — tools are *model*-controlled, resources *application*-controlled, prompts *user*-controlled; that's the real distinction, not data type. **Tool poisoning** (injection via tool *metadata*, distinct from tool *output*). **MCP auth**: stdio = process trust, remote = OAuth 2.1; **Enterprise-Managed Authorization** (2026) moves authz to the org IdP — directly relevant to a bank. **Agentic RAG** — the bridge between two of their three named topics. |

### 7.rag + 6.llms AUDITED (same session)

**Verdict: both are strong. RAG had 9 real defects, LLMs had 1.** Theory quality is not the gap in any of the three folders — recall under pressure is.

| File | Found | Fixed |
|---|---|---|
| `7.rag/02_rag_pipeline.md` | **RRF worked example was numerically wrong AND pedagogically broken** — it used *identical* rankings for dense and sparse (so nothing fused), and claimed `D3=D2=0.03200`. Script-verified truth: D3=0.03226, D2=0.03175, **not equal**. | Rewrote with rankings that genuinely disagree; every term shown; verified `D1 0.032522 > D2 0.032266 > D3 0.031754 > D4 0.031498`. Added the teaching point: D2 was 3rd in dense, 1st in BM25, and RRF lifts it to 2nd — *that promotion is the whole value of hybrid.* |
| `7.rag/06_production_rag.md` | **Python `SyntaxError`** — `incremental_index()` had non-default `vectorstore` after a defaulted param. Code could never run. | Reordered params. |
| `7.rag/01_rag.md` | Embedding table garbled: a row labelled `BAAI/bge` with "1K context" was actually describing **BGE-M3** (8192 ctx); `intfloat/e5-mistral-70b-instruct` **does not exist** — it is **7b**, and the same row said "7B params", self-contradicting. | Both corrected. |
| `7.rag/01_rag.md` | Reranking Q&A opened with a garbled sentence ("ANN retrieval scores speed using bi-encoder"). Also `claude-opus-4-6` model ref. | Rewritten; model ref updated. |
| `7.rag/01_rag.md` vs `02_rag_pipeline.md` | **Cross-file contradiction** on hybrid lift: "5-15%" vs "3-8%" recall@10. | Harmonised to **+3-8% on BEIR**, explicitly marked dataset-dependent, with a note to quote the range not a point figure (an interviewer who knows BEIR will push). |
| `7.rag/05_rag_evaluation.md` | Recall@k / Precision@k formulas used `\|` inside backticks **inside a markdown table** → the rows render broken. | Switched to `#(...)` notation. |
| `6.llms/05_vllm_internals.md` | Same waste figure quoted **three different ways**: 60-80% (Q1), 85.4% (§2, measured), ~99% (diagram caption). | Harmonised to the measured **85.4%** batch figure, with a note to quote that rather than the 98.8% single-request worst case. Also `250+` vs `256+` typo. |

> 🔴 **`7.rag/01b_rag_end_to_end.md` WAS THE WORST FILE — and I nearly missed it.** 690 lines of narrated step-by-step numbers, i.e. **exactly the failure class this file's own 11-Sep warning describes.** Script-verified; **four independent arithmetic faults**:
> 1. **BM25 IDF wrong, and it poisoned every BM25 score in the file.** Wrote `log(2.5/2.5 + 1) = log(1.60) = 0.470`. But `2.5/2.5 + 1 = 2`, so it is **log(2) = 0.69315**. Every score was ~32% low. Regenerated: D1 0.966→**1.42400**, D2 0.750→**1.10640**, D3 0.483→**0.71200**.
> 2. **Cosine(Q,D2) had two errors compounding.** `0.240 + 0.688` was written as **0.880** (it is 0.928), and `0.86²` was written as **0.60** (it is 0.7396). Result 0.911 → **0.90055**.
> 3. **Cosine(Q,D3) had three slips** — `0.8×0.79` as 0.624 (=0.632), `0.8×0.68` as 0.504 (=0.544), `0.79²` as 0.6084 (=0.6241). Final rounded to 0.997 *by luck*.
> 4. **Softmax gave the SAME logit two different values.** Logits `cat=0.2, sat=0.2`, yet `e^0.2` appears as both **1.350** and **1.221**. 1.350 is `e^0.3`. Sum 12.386 → **12.2577**; P(mat) 0.659 → **0.6662**; P(cat) 0.109 → **0.0996**.
>
> **Every ranking survived** (dense D1>D3>D2>D4, BM25 D1>D2>D3>D4, RRF tie D2/D3, greedy picks "mat"), so the narrative was sound and only the arithmetic needed regenerating. Added three teaching notes the errors revealed: IDF is a common factor so it **cannot** change BM25 ranking; the D2/D3 RRF tie is **exact and structural** (ranks 2+3 vs 3+2, addition commutes) because RRF reads ranks not scores; and **equal logits must yield equal probabilities** — a free self-check on any hand-computed softmax.

**Both previously-flagged 6.llms items turned out to be properly FIXED, not left hanging:** the Llama-3 KV-cache figure is correctly 0.50 GiB (GQA) with an explicit note that 2.00 GiB is the MHA/Llama-2-7B number people misattach; and the GSM8K "18%→70%" claim carries an honest CAUTION block explaining the Kojima/Wei conflation rather than substituting an unverified number.

### ALL 13 READMEs FIXED — the "READMEs won't display" bug found and killed

**Root cause was ONE bug repeated in 4 files:** the closing ``` fence of the mermaid block had **text appended on the same line**, e.g. ``` ``` **Tier: 2 (Theory).** …``` . The fence never closes, so the diagram fails to render *and* the sentence gets swallowed into the code block. Hit `7.rag`, `8.agents`, `9.multimodal`, `11.system_design`. Fixed by moving the prose to its own line after the fence.

Other defects found in the same sweep:

| Problem | Files |
|---|---|
| **H1 title didn't match the folder number** | `9.multimodal/README.md` was titled **"# 11. Multimodal"**; `10.mlops/README.md` was **"# 8. MLOps"**. Both now correct — all 10 numbered folders verified title-vs-folder. |
| **Renamed files still referenced by old name** | `10.mlops` listed `03_feature_store.md` (**never existed**; real file is `03_monitoring_and_drift.md`, which was missing from the TOC entirely). `1.machine learning` listed `03_unsupervised.md` (real: `03_unsupervised_learning.md`). |
| **Dead `code_practice/` paths** | `10.mlops` pointed at `code_practice/04_llms/` and `code_practice/05_rag/` — **neither folder exists**. Repointed at the real `06_llms/`, `07_rag/05_production_rag/`, `09_finetuning/` with ✅/⏸ status. |
| **`1.machine_learning` (underscore) — the folder name has a SPACE** | Same bug class as the `2.deep_learning` fix from the prior arc. 4 files: `10.mlops/README.md` ×2, `11.system_design/README.md`, `6.llms/README.md`, `4.nlp/03_sequence_models/01_rnn_to_attention.md` (which also had a wrong subfolder, `05_algorithms/` → `02_algorithms/`). Now written as URL-encoded `%20` links, which render correctly on GitHub and in VS Code. |
| Stale file count | `8.agents/README.md` said "10 files"; it is now **11** (after `00_agent_stack_foundations.md`). |

**Verification:** fence/mermaid lint **0 issues across all 13 READMEs**; README link + TOC audit **clean**; repo-wide **218 links, 0 broken**; all 10 numbered folder titles match. *(Lint scripts are in the session scratchpad — the fence check is worth rebuilding if READMEs are ever edited by hand again, it took seconds and found what eyeballing missed.)*

---

### 6.llms COMPLETED — all 12 files read. **Do-not-claim rule LIFTED at user's instruction.**

User decided to claim fine-tuning/alignment after learning it properly, so the 8 deferred files were audited in full. **Found 11 more defects, several serious.** The honest framing agreed for interview: **claim the knowledge, not the run** — *"I've implemented LoRA/QLoRA and DPO end to end and worked the arithmetic by hand; the training runs are blocked on a torch/cu121 conflict on a GTX 1650 Ti."* That survives follow-up (dataset size, loss curve, what broke); "I fine-tuned Mistral-7B in production" does not.

| File | Verdict |
|---|---|
| `02b_finetuning_end_to_end.md` | ⚠️ **6 faults.** (1) Reward model computed `r_A` using **h_B's value** — wrote `0.9×0.4 = 0.480` (0.9×0.4=0.36; correct first term is 1.2×0.4). (2) **Bradley-Terry loss had a sign error in the exponent** — `1/1.934` uses `e^(+0.66)`; and it reported **σ(0.66)=0.659 *as the loss***. Correct `L = 0.4166`. (3) §9 LoRA verification carried **three mutually inconsistent number sets** and a `W_Q` contradicting §3.3; regenerated `q_on [0.0052, 0.0736] → [0.0165, 0.0798]`. (4) DPO loss `0.558 → 0.5596`. (5) KL example listed `P(on)` twice. (6) PyTorch snippet fed **3 features into a 2-feature layer**. |
| `02_finetuning.md` | ⚠️ **LoRA trainable params off by 7.6×** — claimed `20,971,520` for r=64 across all 7 modules; correct is **159,907,840** (**2.32%**, not 0.31%), all-params `6,898,323,456`. Base verified against published `6,738,415,616`. Also `"Mistral / Mistral"`, `"add k_proj, k_proj"`, a backwards bf16 comment, and **train/val split listed under things to AVOID**. |
| `07_dataset_preparation.md` | ⚠️ **Said ChatML is "used by Llama-3, Mistral"** — it is not; Llama-3 uses `<\|start_header_id\|>`/`<\|eot_id\|>`, Mistral uses `[INST]`. Wrong in 2 places and **contradicts `02c` §4**. Also **contradicted `02c` on multi-turn masking** — claimed "mask all but the final assistant turn"; correct is **every assistant turn is a target**, `02c` was right. Magpie snippet sliced `[0]` off a **string** (one character, not a header cut). |
| `03_alignment.md` | ⚠️ KL target `6-10 nats` **contradicted `03b`** (healthy 2-6); `odds()` mislabelled `odds_ratio()`; **2 refs to files that don't exist** (`10_alignment_end_to_end.md`, `6.llms/06_evaluation`). |
| `06_alignment_follow_ups.md` | ⚠️ IPO failure mode contradicted its own §5 (said loss is *small* past the target margin; it *rises* — that is the intended penalty). |
| `03b_alignment_end_to_end.md` | ✅ **Clean.** All four dry-runs independently re-derived (BT `0.153→0.149`, PPO clip `0.568→0.372`, DPO `0.404→0.371`, ORPO `0.517`). Only `∇L` written as `ΔL`, a garbled expectation subscript, 1 dead ref. |
| `02c_sft_end_to_end.md` | ✅ **Clean.** Masked/unmasked loss `2.178202 → 2.051061` (diff `0.127141`) and all 10 template-overhead percentages verified. |
| `03c_dpo_end_to_end.md` | ✅ **Clean and genuinely excellent.** Independently re-derived the closed form (`Z=5.524786`, π* to 6dp), the `Z(x)` cancellation, `L_BT = L_DPO = 0.167786`, the β-sweep KL, and the memory math (239.3 vs 119.7 GiB = 2.00×). |
| `01_prompting.md` · `04_evaluation.md` · `05_vllm_internals.md` | ✅ audited earlier this session |

> **Deliberately NOT changed: dated model names in illustrative code** (`gpt-4o`, `claude-sonnet-4-6`) across 6.llms/7.rag/8.agents. They are example snippets, not claims. **One must not be touched casually:** `8.agents/01b_agents_end_to_end.md:324` quotes `claude-sonnet-4-6 at $3/$15 per MTok` — that is *correct pricing for that model*, and the whole `$0.022/run` cost walkthrough is built on it. Renaming the model without regenerating the arithmetic would introduce an error.

**Repo-wide after all fixes: 214 links across 37 files, 0 broken.**

---

### AUDIT LEDGER — what was actually verified, and by whom

| Folder | Read by this session | Status |
|---|---|---|
| `8.agents/` | **11 of 11** | ✅ Complete |
| `7.rag/` | **7 of 7** | ✅ Complete |
| `6.llms/` | **12 of 12** | ✅ Complete |

~~**The 8 unread `6.llms` files are the fine-tuning + alignment cluster**~~ **— SUPERSEDED, now all read. See the 6.llms section above.** Original reasoning retained below for the record:

**The 8 then-unread `6.llms` files were the fine-tuning + alignment cluster** (`02`, `02b` 754L, `02c`, `03`, `03b` 571L, `03c`, `06`, `07` — ~2,812 lines). Deliberately deprioritised for THIS interview: it is exactly the area covered by the **do-not-claim rule** (Phase 09 parked, not on the resume), so it will be discussed conceptually at most. They also carry script-verified evidence from the previous session recorded above — e.g. `03c` (`cancellation 0.000e+00`, `L_DPO == L_BT 0.167786` both ways) and `02b` (`B=[[0.0],[0.0]]` fixed at source, NF4 claim corrected to the measured 2.14×). **Trusted, not re-verified.** If time opens up post-interview, `02b` and `03b` are the two worth re-checking first — they are the largest narrated-number files left in the repo, and that is the exact class that failed in `7.rag/01b`.

**Extra defects found while closing the gaps:** `7.rag/03_indirect_prompt_injection.md` had a **Python `SyntaxError`** — the `signals` detector was written as a `list` with `"key": value` entries, then read via `signals.values()`; now a `dict`. `6.llms/04_evaluation.md` had **BERTScore F1 printed as ≈ −0.92** (cannot be negative there; it is 0.92), a **garbled `pass@k` example** whose two lines were both labelled `pass@1` — fixed at source to `pass@1 = 0.4000` / `pass@10 = 0.9996` for n=20,c=8, with the point that made it worth fixing (one sample passes 40% of the time; ten attempts solve it essentially always — which is why a bare "pass@k" without both n and k is meaningless) — and 4× dated `claude-opus-4-6` refs.

**SEQUENCING FIXED — the arc is `6.llms` → `7.rag` → `8.agents`, and all three READMEs now say so** with the same three-line diagram (engine → tool → loop) and an explicit "do not start at RAG". `6.llms/README.md` had **no single linear path** (topic-indexed only) and its Folder TOC was **missing 3 files that its own Reading Order references** — `02c_sft_end_to_end`, `03c_dpo_end_to_end`, `04b_evaluation_end_to_end` (same defect class as the 5.transformers TOC). Added a straight-through path plus the missing rows. `7.rag/README.md` gained the framing decision (RAG vs fine-tuning vs long-context vs agentic RAG) it was missing at the entry point — the content existed but was buried in `01b` §9/§13.

**No new foundations files written for RAG or LLMs — deliberately.** The agents one earned its place because there was a *named* gap (what a format/SDK/framework even is). No equivalent gap exists here: the 7-layer stack in `8.agents/00_agent_stack_foundations.md` already covers the LLM layer and is cross-referenced from all three READMEs, and RAG's orientation need was a **sequencing** problem, not a missing-content problem. Writing more would have duplicated and violated SSOT.

**Dead refs fixed in both folders too.** `code_practice/03_prompting/`, `05_rag/`, `09_llms/`, `04_5_advanced/`, `02_transformers/` **never existed** — real phases are 05-12. Phase 09 refs now carry ⏸ **"code-built, not run"** so the parked status is visible at the point of use. Rulebook-RAG (`07_rag/06_rulebook_rag/`) is now linked from `01_rag.md`, `04_advanced_rag.md`, `05_rag_evaluation.md` and the README as the portfolio project. **Repo-wide check: 184 links across 37 files, 0 broken.**

---

> 🇬🇧 **THE DIFFERENTIATOR for the UK round — regulation, written up in `LLOYDS_UK_ROUND.md` §3.** **SS1/23** (PRA Model Risk Management, in force 17 May 2024) is technology-neutral and **expressly covers AI/ML, including LLMs used as components inside agentic workflows**; scope is the whole firm, not just credit/market risk. Five principles: model identification + risk tiering · governance (SM&CR named accountability) · documented development/implementation/use (**a prompt change IS a model change**) · **independent validation with effective challenge** · mitigants where not fully validated. **FCA Consumer Duty** adds individual-decision explainability — defensible to the customer, the FCA **and the Financial Ombudsman Service** — plus outcomes monitoring with a drift tolerance and documented intervention path. **The synthesis to rehearse:** validating a stochastic agent means validating a *trajectory distribution*, not a function — so validate components (retrieval precision, tool-call accuracy, refusal behaviour), measure variance across repeated runs not a single pass, red-team, and fall back to Principle 5 mitigants (HITL, scope restriction) where full validation is unreachable. Almost no other candidate will have this.

> ⚠️ **Do NOT claim QLoRA / Mistral-7B in the interview.** Not on the resume, Phase 09 parked, indefensible under follow-up. Already flagged in the Open Finding below.

> ⚠️ **`02_INTERVIEW_PACK.md` IS STALE** and is scheduled for rewrite on Day 4 of the plan. It still says "8 years / 94% accuracy / 60% RCA reduction" and describes the RAG project as FAISS + Streamlit + `llama3.2:1b`. Real: 9 years, F1 0.959-0.972 and 0.975, P1 RCA 5 services / 21,096 pages / 2.3%, Rulebook-RAG 36 classes / BM25+RRF (+5.2) / cross-encoder (+3.9) / 0 invalid of 775.

---

> ⚠️ **Pattern worth knowing:** the three broken files (RNN/LSTM/GRU) all failed the same way — plausible-looking numbers that were *written* rather than *computed*. The two correct ones (attention/transformer) are single-shot parallel computations, which are harder to fake. **Any file whose numbers were narrated step-by-step should be assumed unverified until a script reproduces it.**

> **Correction (25 Aug):** the blue placeholders in Naukri's draft (`Please mention if any`, `DD'MM'YY`) are **deliberate fill-in markers** — their cover email asks the customer to complete them. Earlier I called this a QC failure. It is not. Do not raise it.

> ⚠️ Reminder from 22 Aug: three commits shipped without updating this file. **Update it in the same commit as the work, not later.**

---

## Next Task

```
DIRECTION: user will choose the starting point next session. Open threads below,
           highest-value first. Nothing is blocked on anything else.

>>> DECISION OPEN (paused mid-session 11 Sep) <<<
  06b_transformer_encoder_multihead.md  §14-15 (backward + weight update)
  The file NEVER shows W1, W2 (FFN weights) or the target T. Consequence:
    - W1/W2: reconstructible only by pseudo-inverse. h1 CANNOT be inverted exactly
      -- a LayerNorm output has every row summing to 0, so it is always rank-deficient.
    - T: unrecoverable, full stop (16 unknowns, 1 equation = the scalar loss).
  So §14-15's numbers (loss 0.395052 -> 0.354209, all gradient magnitudes) cannot be
  verified without INVENTING a target. A reconstruction was built and checked:
    invented T (role vectors: bank=[1,1,-1,-1], loan=[-1,-1,1,1], others 0)
    -> loss 0.770069 -> 0.599646, Wv 0.2407 / Wo 0.2682 / Wq 0.0387
    -> qualitative story SURVIVES (Wv,Wo dominate; Wq,Wk ~7-10x smaller; loss drops)
    -> but every number in §14-15 changes.
  Blast radius checked: ZERO. Those numbers are cited in no other file, and OUTPUT
  (which 06c consumes as MEMORY) is unaffected -- it is forward-pass only.
  NOT YET DONE. Decide: rewrite §14-15 with the reconstruction, or leave as-is.

AGENTS (asked about 11 Sep, nothing started):
  Phase 08 is COMPLETE -- all 4 sessions ✅ Run, but back in JUNE (3 months cold):
    01_react_agent.py · 02_tool_calling.py · 03_langgraph_agent/ · 04_document_agent/
  MASTERY_PLAN.md contains ZERO agent drills (grepped -- none).
  So the gap is recall/defence, not build. Three options put to user, none chosen:
    (a) build an agent drill in 11_interview_drills (LangGraph from scratch, cold)
    (b) re-run + walk the existing 4
    (c) new session for untouched theory: MCP (08), multi-agent (07), agent eval (09)
  NOTE: "LangGraph 46% / Agents 23%" in 00_HUB are % OF JOB DESCRIPTIONS,
        not mastery levels. Do not misread these as skill gaps.

Board:  5.transformers/whiteboard/6.decoder.jpg      (to draw, 25 min)
Source: 4.nlp/03_sequence_models/06c_transformer_decoder_cross_attention_end_to_end.md

The board must show:
  - three sub-layers: masked self-attn -> cross-attn -> FFN, each + Add & Norm
  - the causal triangle, -inf marked BEFORE the softmax box
  - cross-attn as a RECTANGLE (3 x 4), K,V arrows in from the encoder, Q up from
    the decoder  <- this is the one people draw backwards
  - teacher forcing (one pass, all rows) vs autoregressive (L passes, last row only)
  - where the KV cache sits; cross-attn K,V built ONCE

Reading order after that (board numbers, REORDERED 28 Aug):
   7 tokenization -> 8 BERT -> 9 GPT -> 10 T5 -> 11 decoding -> 12 attention at
   scale -> 13 modern block -> 14 long ctx -> 15 MoE -> 16 speculative
   -> 17-21 pretraining / SFT / LoRA / RLHF / eval

Boards 8 and 9 theory rewritten and verified. Board 9 is now TWO files, no mixing:
    `05_bert_end_to_end.md`      (BERT)
    `06_gpt1_end_to_end.md`      (GPT-1: post-LN, 116,534,784 params, fine-tuned per task)
    `06b_gpt2_end_to_end.md`     (GPT-2: pre-LN + ln_f, 1/sqrt(N) init, 124,439,808, zero-shot)
    `06c_gpt3_end_to_end.md`     (GPT-3: sparse attn, 8-model ladder, in-context learning)
  Board 16 theory done: `5.transformers/02_models/13_speculative_decoding.md` REWRITTEN
    (the 'proof' was literally `= ... = p(y)` deferring to the paper; 'lossless', 'residual' and
     'max(0' appeared ZERO times -- the residual norm(max(0,p-q)) IS the mechanism; and the
     speedup table did not match the formula printed above it, e.g. a=0.7,K=5 claimed 2.7 vs 2.9412)
    Lossless now proved 3 ways: elementwise 0.000e+00, closed form 0.000e+00, 10M-draw MC 8.64e-05.
    Corrected the central conflation: expected tokens E(a,K) is NOT speedup; speedup = E/(1+K*c).
    At a=0.5, c=0.2, K=8 the real figure is 0.7677 -- a SLOWDOWN.
  FULL-REPO VERIFICATION PASS -- run because "is it all correct?" deserves a check, not an assertion.
    346 markdown links checked repo-wide.
      26 were broken, NONE in files touched this arc -- all pre-existing:
        22 x off-by-one depth in code_practice/ (`../../../` -> `../../`)
         2 x pointed at 03_encoder_decoder_family.md (real file: 03_encoder_decoder.md)
         1 x RULES.md had a stray leading ../
         1 x CLAUDE.md's cross-ref EXAMPLE pointed at 01_tokenization.md (real file: 03_)
      ALL FIXED. Now 0 broken (the CLAUDE.md line is an illustration in backticks, ../ intended).
    Cross-file number consistency checked on 9 key parameter counts.
      FOUND A REAL ERROR IN MY OWN WORK: BERT-base was quoted as 108,770,304, which MIXES
      simplified blocks (4d^2 + 2*d*d_ff, no biases/LN) with FULL embeddings incl. segment.
      Not a consistent accounting. Correct: 108,891,648 encoder / 109,482,240 with pooler
      (= HF bert-base-uncased). Fixed in 3 files; 108,770,304 no longer appears anywhere.
      The 21.9% / 78.1% / two-thirds proportions still hold exactly against the corrected total.
    All other 8 figures agree across every file that cites them.

  6.llms/ AUDITED file by file (9 files; 4 were already written this arc).
    REPO-WIDE FIX: 9 broken paths using `2.deep_learning` (underscore) -- the folder is
      `2.deep learning` (space). 3 had further errors: missing 01_fundamentals/ subfolder,
      07_training_stability -> 03_, architectures/ -> 02_architectures/, 08_arch_comparison -> 00_.
      Touched 4.nlp x2, 5.transformers x2, 6.llms, 9.multimodal. All 9 now resolve.
    01_prompting.md    "GSM8K ~18% -> ~70%" conflates Kojima zero-shot-CoT with Wei few-shot CoT
                       on a different model. Flagged rather than substituting unverified numbers.
    02_finetuning.md   broken arch-comparison path (see repo-wide fix).
    02b_finetuning...  FIXED AT SOURCE the two errors I had only flagged before: B=[[0.8],[0.0]]
                       -> [[0.0],[0.0]] (and the forward pass that used it), and the fabricated
                       "NF4 6x better" -> the real measurement 2.14x. Also removed a stale
                       "From the BERT file: x_cls=[1.386, 2.019]" -- BERT now runs at d_model=4.
    03_alignment.md    Added the missing Z(x) CANCELLATION (same gap as 03b) + the beta sweep.
                       Softened "same quality, much simpler" (2 places) -- DPO is offline and
                       leaves no reusable RM.
    03b_alignment...   AUDITS CLEAN. All 20 values verified (Bradley-Terry, PPO clipping, DPO).
    04_evaluation.md   pass@k correction note added earlier stands.
    05_vllm_internals  "Llama-3 8B, 4096 ctx: ~2 GB per request" is 4x too high -- it is GQA,
                       so 536,870,912 bytes = 0.50 GiB. ~2 GB is the MHA (Llama-2-7B) figure.
                       Same error class as 02_gpt_family but INVERTED.
    06_alignment_follow_ups  clean (citations and arXiv IDs check out).
    README.md          missing 02c/03c/04b entirely; reading order now carries board numbers.
                       Removed a stale note about 00_roadmap.md, which does not exist here.

  5.transformers/README.md REWRITTEN -- reading order was stale and file order != learning order.
    Now states explicitly that FILE NUMBERS ARE NOT READING ORDER, with the two breaks named:
      04b_attention_at_scale is board 12 but files at 4 (needs a model behind it)
      09/09b are board 19 (adaptation) but file inside the architecture run 13->15
    Reading Order now split into Stage 1 (architecture, boards 7-16) / Stage 2 (17-21) / off-path,
    every row carrying its board number. Folder TOC rebuilt in true file order with all 20 files
    (04b, 06c, 07b, 08b, 09b were missing or misplaced).
    Fixed 4 broken paths: README used `2.deep_learning` but the folder is `2.deep learning`.
    Removed a stale note about `00_roadmap.md`, which does not exist in this folder.
    DECIDED NOT TO RENUMBER the files: renaming would touch ~20 files and every cross-ref across
    5 folders for no gain -- MASTERY_PLAN + this README are the ordering artifacts.

  5.transformers/02_models/ AUDITED -- 6 remaining files (12 were already done this arc).
    DELETED `04_efficient_transformers copy.md` (verified strict subset, 0 unique lines,
      referenced nowhere) -- ticks the Pending Cleanup item.
    01_bert_family.md      "10x more steps (500K vs 1M)" self-contradictory AND wrong both ways:
                           RoBERTa used HALF the steps (500K vs 1M) and 16x the TOKENS
                           (2.048e15 vs 1.311e11). Batch 2048->8000, vocab ->50,265.
    02_gpt_family.md       KV cache "~0.5GB" for Llama-2-7B is 4x too small -- it is 2.00 GiB
                           (the claim silently assumed GQA on an MHA model).
                           "3.5GB (8-bit)" is FOUR-bit; int8 is 7.0 GB.
    03_encoder_decoder.md  T5-XXL and T5-11B listed as two sizes (same model).
                           Flan-T5 attributed to Wei et al. 2022 -> it is CHUNG et al. 2022;
                           Wei et al. 2021 is the original FLAN paper.
    09_parameter_efficient_tuning.md  Two "trainable%" figures with no config stated (both right:
                           r=8 q,v = 4,194,304 / r=16 attn+MLP = 39,976,960). 112GB vs 98GB
                           header/body mismatch = different precision assumptions, now stated.
    12_constrained_decoding.md  Claimed mask-building is O(vocab x depth) -- contradicts its OWN
                           citation (Outlines precomputes an FSM index -> O(1) lookup).
                           Added: constrained decoding is NOT distribution-preserving (deletes
                           63.8% of belief here), unlike KV cache / Flash / speculative.
    14_reasoning_models.md "o3 gets ~96x" -> ~96%. Added the cost/latency table and the RLVR
                           connection to board 20 (a verifier cannot be reward-hacked).

  5.transformers/01_fundamentals/ AUDITED FILE BY FILE (all 5) -- see below.
    01_attention_mechanism.md   Flash framed backwards ("trades compute for memory" -> it is
                                FASTER; the trade is FLOPs for memory TRAFFIC). sqrt(d_k) had no
                                numbers: added var(q.k)=d_k measured + the saturation it prevents
                                (Jacobian mass collapses 6.9x unscaled, flat when scaled).
    02_transformer_architecture.md  REAL MATH ERROR: LayerNorm written (x-mu)/(sigma+eps).
                                Correct is /sqrt(sigma^2+eps). At var=1e-8 the wrong form gives
                                0.9091 vs correct 0.0316 -- 28.8x, and it defeats the whole point
                                of eps. Also the Pre-LN line was incoherent; added the final
                                LayerNorm and the warmup connection.
    03_tokenization.md          Two tables CONTRADICTED each other: GPT-3 listed as 100,277
                                (it is 50,257, same as GPT-2), LLaMA-1/2 as byte-level BPE (it is
                                SentencePiece), LLaMA-3 as SentencePiece (it is tiktoken BPE).
                                Fertility rule was BACKWARDS on Chinese ("1 token per 1-2 chars"
                                -> it is 1-2 TOKENS PER CHAR, 4-8x worse than English).
                                SSOT flagged, NOT deleted -- 4.nlp owns tokenization and there are
                                now 3 files on it. User's call.
    04_pretraining_objectives.md  Poisson(lambda=3) attributed to T5; it is BART's. T5 uses a MEAN
                                span length of 3. Also the "40% masking is too hard" claim is
                                contradicted by Wettig et al. 2023 -- flagged as received wisdom.
    05_vision_transformers.md   Two broken paths (81_fundamentals / 81_bert_family -> 01_).
                                Swin window count 6,278 -> 6,269. Added verified ViT param table
                                and a new SECTION 10: Donut = Swin encoder + BART decoder, since
                                that is a live resume claim and each half is already hand-computed.

  Board 21 theory done: `6.llms/04b_evaluation_end_to_end.md` NEW  --  LLM TRACK COMPLETE.
    04_evaluation.md never mentions PERPLEXITY (0 hits) despite it being the board's first
    killer question, and its pass@k example is garbled: two lines both labelled pass@1 giving
    0.48 and 0.95. Correct for n=20,c=8: pass@1 = 0.4000, pass@10 = 0.9996. Flagged in-place.
    Perplexity tokenizer-dependence shown exactly: ONE document, PPL 1.28 to 6.02.
    Caught an error in my own draft: claimed 27% MMLU is "within noise" -- it is z=5.47,
    statistically above chance. Corrected to the real point: 2.7% of the usable 25-100% range.
  Board 20 theory done: `6.llms/03c_dpo_end_to_end.md` NEW
    The alignment files state r = beta*log(pi*/pi_ref) + beta*log Z(x) and then jump straight to
    the DPO loss, never explaining that beta*log Z(x) CANCELS. That cancellation IS DPO.
    Now derived + verified: closed-form recovery 5.551e-17, cancellation 0.000e+00,
    and L_DPO == L_BT (0.167786 both ways).
    beta sweep shows what the KL term stops: KL 0.0020 at beta=10 -> 0.9155 at beta=0.05
    (collapse onto argmax reward = reward hacking).
    PPO holds 4 models (239.3 GiB at 8B), DPO holds 2 (119.7). Also wrote what DPO GIVES UP
    (offline, no reusable RM, margin over-optimisation) -- it won the open ecosystem, not the argument.
  Board 19 theory done: `5.transformers/02_models/09b_lora_qlora_end_to_end.md` NEW
    Found TWO errors in 02b_finetuning_end_to_end.md and flagged them in-place:
      - LoRA init dry-run says B=0 but writes B=[[0.8],[0.0]], computes x@B=0.800 from the
        non-zero B, then claims the result is [0,0]. With that B it is [0.16, 0.088].
      - "NF4 gives 6x better representation" is derived from stipulated numbers. Measured
        against the actual NF4 levels on 2M Gaussian samples: 2.14x lower MSE.
    Also caught an inconsistency in MY OWN draft: §8 used a LoRA count ignoring GQA
    (8,388,608) contradicting §4's table (6,815,744). Fixed; optimizer ratio 957x -> 1,178x.
    Merge verified exact: 4.441e-16, zero inference overhead.
  Board 18 theory done: `6.llms/02c_sft_end_to_end.md` NEW
    (07_dataset_preparation.md covers ChatML + the -100 masking RULE well and needs no fix,
     but nothing anywhere computed what masking DOES. That gap is what the new file fills.)
    Masked vs unmasked loss 2.178202 -> 2.051061, and the gradients differ in DIRECTION
    (cosine 0.890171 on EMB) -- masking is not a rescale.
    Template overhead ~5 tokens/turn = 34% of a short 4-turn chat, <3% with long turns.
    Flagged the two distinct template failures; the missing-stop-token one is the common bug.
  Board 17 theory done: `4.nlp/03_sequence_models/08_scaling_laws_emergent.md` REWRITTEN
    (was 245 lines; used the symbol C for BOTH the irreducible loss constant and for compute;
     never stated 'power law' or 'irreducible'; and section 5 duplicated boards 18/20 -- trimmed
     to cross-refs)
    Derived the compute-optimal exponents from the fit: 0.4516 / 0.5484, NOT 0.5/0.5.
    FLAGGED: the published Chinchilla Approach-3 parameters do NOT reproduce the paper's own
    20:1 headline -- they imply 93:1 at DeepMind's own budget. Besiroglu et al. 2024 found that
    fit describes the data poorly. File uses 20:1 as the operational rule and says why.
    Emergence: demonstrated p^20 going 0.000798 -> 0.038760 -> 0.290106 -> 0.817907 from a
    perfectly SMOOTH p. The discontinuity is in the metric (Schaeffer et al. 2023).
  Board 7 theory done: `4.nlp/01_fundamentals/05_embedding_lookup_end_to_end.md` NEW
    (04_tokenization_end_to_end.md AUDITS CLEAN -- I reproduced its BPE training programmatically
     and the merges/counts/ties/final state match exactly. But 'weight tying' and 'embedding
     matrix' appeared ZERO times across both tokenization files, so board 7's second half had
     no home. That is what the new file covers.)
    Embedding share: GPT-2 small 31.0% vs GPT-3 175B 0.4% on the SAME 50,257 vocab.
    Tying saves 21.1% (GPT-1) / 23.7% (GPT-2); Llama 3 declines 6.5% and unties.

  ALL BOARDS 6-21 THEORY COMPLETE (architecture 6-16 + stage 2 17-21).
  NOTHING has passed a gate yet: G1 draw 0/16, G3 code 0/16. Whiteboard folder still empty.
  Drill 01 built 14 Aug, still never attempted. Drill 04 still not built. Next: 17-21 (pretraining/SFT/LoRA/RLHF/eval),
  
  Board 15 theory done: `5.transformers/02_models/10_mixture_of_experts.md` REWRITTEN
    (had only 8 numeric lines in 374; attention breakdown said ~2.3B, actual 1.34B; gates were
     renormalised from its own ROUNDED probs giving 0.588/0.412 instead of 0.598688/0.401312)
    Mixtral exact: 46,702,792,704 total / 12,879,925,248 active = 3.63x, memory IDENTICAL.
    L_aux minimum = alpha = 0.010000 balanced -> 0.068000 collapsed.
  Board 14 theory done: `5.transformers/02_models/11_long_context_scaling.md` REWRITTEN
    (was 204 lines with 4 numeric lines; 'slope' and 'wavelength' appeared ZERO times despite
     being ALiBi's defining feature and the reason PI/NTK/YaRN differ)
    Key result: at d=128/base=10k/L=4096, exactly 18 of 64 RoPE pairs never complete a rotation
    -- and those are EXACTLY the 18 YaRN fully interpolates. PI stretches all wavelengths 8x;
    NTK stretches 1.00x..8.00x graded. SWA: 32 layers x 4096 = 131,072.
  Board 13 theory done: `5.transformers/02_models/08b_llama3_end_to_end.md` NEW
    (08_modern_llm_architecture.md AUDITS CLEAN -- RMSNorm/RoPE/SwiGLU arithmetic all correct,
     the only file in this sweep that needed no fixing. Left as the mechanisms file + scope note.)
    Llama 3 params exact: 8B=8,030,261,248  70B=70,553,706,496  405B=405,853,388,800.
    RoPE base 10k->500k = 47x longer wavelength. 15T tokens = 93x past Chinchilla, deliberately.
  Board 12 theory done: `5.transformers/02_models/04b_attention_at_scale_end_to_end.md` NEW
    (04_efficient_transformers.md had ZERO worked numbers and 0 mentions of memory-bound /
     arithmetic intensity; it also duplicates boards 15 and 19. Left as a survey + scope note.)
    KV cache verified EXACT (0.000e+00) and Flash online softmax verified EXACT (1.665e-16).
    Killer question answered: 7B/4k/batch8 = 16.00 GiB = 123% of the weights.
  Board 11 theory done: `4.nlp/03_sequence_models/07_decoding_strategies.md` REWRITTEN
    (old one had a beam-search addition error -2.813 + -0.412 = -3.206, should be -3.225,
     and a temperature table whose T=0.5 row summed to 1.002 over an unstated vocabulary)
  Board 10 theory also done, split in two:
    `07_t5_end_to_end.md`        (T5: relative position bias, RMSNorm, span corruption)
    `07b_bart_end_to_end.md`     (BART: denoising, full-document target, post-LN)
  All three were numerically broken before; every value is now audited + torch-checked.

Also open:
  Board 6b whiteboard — theory written, board not drawn (20 min).
  Drill 01 — built 14 Aug, NEVER ATTEMPTED, now ~4 WEEKS COLD. 20 min, no reference, 13/13.
  code_practice/11_interview_drills/01_multihead_attention.py
    (concepts were walked through 11 Sep — shapes, -inf-before-softmax, .contiguous()
     before .view() — but the file was never filled in. Still 5 x NotImplementedError.)
  Drill 04 (cross-attention) — not built; board 6 is reached so it is due.
  Gates: G1 draw 0/16 · G3 code 0/16. Whiteboard folder still empty.
  02_INTERVIEW_PACK.md still stale (see "Open Finding" below) — days 12-14.
```

> **Verification tooling note (11 Sep):** every fix this session was checked by rebuilding the
> forward + backward pass in numpy and diffing against the file. Scripts are in the session
> scratchpad (not committed). Rebuilding one takes ~10 min and is the only way to trust a
> hand-written numeric walkthrough — recommend doing this for any remaining end-to-end file
> whose numbers were never script-checked.


---

## ACTIVE ARC — Interview Prep (started 14 Aug 2026)

**Context:** self-assessed weak in all four areas: resume defence, LLM/GenAI theory, ML system design, behavioural/STAR. Format chosen: **coding drills as the spine**, other three folded in around them.

**Timeline (rev. 22 Aug):** nothing scheduled yet. The original "1–2 weeks" clock is off. Days below are an *order*, not dates — runway is intact, so depth over speed. But drills have slipped 8 days with zero attempted; theory is not a substitute for saying it out loud or typing it cold.

### ACTIVE TRACK — LLM Architecture (scope set 25 Aug)

Full tracker: **`code_practice/11_interview_drills/MASTERY_PLAN.md`**

**REORDERED 28 Aug.** Two moves + the parked half added as boards 17–21. Rationale lives in
MASTERY_PLAN.md → "Ordering logic". Theory already exists for every board; the gap is recall,
not reading.

Each board passes 4 gates: **G1 draw · G2 hand-compute · G3 code in 20 min · G4 defend aloud.**
No board advances until all four pass.

| # | Board | G1 | Status |
|---|-------|----|--------|
| 1–5 | RNN → LSTM → GRU → Attention → Transformer **encoder** | — | ✅ drawn · G3 outstanding |
| 6 | **Transformer Decoder** — theory written 28 Aug (`06c`), board not drawn ← NEXT | 25 | 📄 |
| 6b | Encoder w/ multi-head, d_model=4 — **theory written 28 Aug**, board not drawn | 20 | 📄 |
| 7 | Tokenization → Embedding | 15 | ⬜ |
| 8 | BERT | 20 | ⬜ |
| 9 | GPT | 20 | ⬜ |
| 10 | T5 / BART *(= the Donut decoder)* | 25 | ⬜ |
| 11 | **Decoding** — greedy/beam/top-k/top-p/temperature ← *moved up from 15* | 20 | ⬜ |
| 12 | **Attention at scale** — KV cache, Flash, PagedAttention ← *now before the modern block* | 25 | ⬜ |
| 13 | **Modern LLM block** — pre-LN, RMSNorm, SwiGLU, RoPE, GQA ← *was 11* | 25 | ⬜ |
| 14 | Long context — RoPE scaling, ALiBi, SWA | 20 | ⬜ |
| 15 | Mixture of Experts | 20 | ⬜ |
| 16 | **Speculative decoding** ← *split out of old 15* | 15 | ⬜ |
| 17 | **Pretraining + scaling laws** *(was out of scope)* | 20 | ⬜ |
| 18 | **SFT / instruction tuning** *(was out of scope)* | 20 | ⬜ |
| 19 | **LoRA / QLoRA** *(was out of scope)* | 20 | ⬜ |
| 20 | **RLHF → DPO** *(was out of scope)* | 25 | ⬜ |
| 21 | **Evaluation** *(was out of scope)* | 20 | ⬜ |

**Why the two moves:** GQA cannot be justified without KV-cache arithmetic, so 12 now precedes 13.
And board 10 leaves you holding a probability row with no way to pick a token — answering that at
board 15 was backwards.

Boards 1–5 live in `4.nlp/03_sequence_models/whiteboard/`.
Boards 6–21 go in **`5.transformers/whiteboard/`**, continuing the numbering.

**Done means:** given any model card (Llama / Mistral / DeepSeek / Qwen), draw the whole model
from memory — tokens → vectors, each block, what changed since 2017 and why, how generation runs,
where the memory goes.

Every drill is a literal "I hand-wrote / I built from scratch" claim on the resume — drilling them **is** resume defence.

### Naukri resume — draft 3 reviewed 28 Aug
> **Handled in a SEPARATE session.** Theory/transformer sessions: skip this block.

Files: `resume/CustCopy2.pdf` (draft 1) · `resume/CustCopy3.pdf` (draft 3) ·
`resume/RESUME_CHANGE_REQUEST.md`+`.pdf` (what was sent) · `resume/resume_sameer_khan_v2.*` (my layout).

Draft 3 is 219 words SHORTER than draft 1.

| Ask | Result |
|-----|--------|
| S1 fix the inverted F1 sentence | ✅ fixed, both instances, my exact wording |
| S5 front-load hard numbers | ✅ first number at word ~79 (was ~370) |
| S8 remove Certifications / DOB / Career Timeline | ✅ all gone |
| S2 present tense on ICE | ⚠️ KRA bullets yes; Highlights still past tense |
| S6 remove hedges | ⚠️ "growing focus"+"strong understanding" gone; "exposure" ×2 remains, 3 new hedges added |
| S3 remove Core Competencies | ❌ ignored — still 11 items incl. "AI Solution Architecture & Technical Leadership" |
| S4 restore 7 specific phrases | ❌ 0 of 7 |
| S7 ATS parser test / S10 send .docx | ⏳ no written answer yet |

**REGRESSION — they cut length by deleting evidence:**
- **entire P1 RCA bullet gone** (21,096 pages · five services · Java validation defect · 260K regression · 2.3%) ← strongest bullet on the page
- TF 1.x→2.x migration, change control, rollback — gone
- `0.974` gone, so "outperforming the ensemble" now has no number behind a 0.975-vs-0.974 margin

**Two things got worse:** first verb a recruiter reads is now *"**Supported** a mortgage document
classification platform"* (draft 1 said "Owned"; the KRA section two inches below says "Own" —
the resume contradicts itself).

**Round-3 asks, in priority order:**
1. Restore the P1 RCA bullet — same standing as the F1 fix had
2. "Supported" → "Own"
3. Put `0.974` back, reframed: *"matched the OCR-based production ensemble's 0.974 while removing OCR from the inference path"*
4. Core Competencies — asked once with evidence, ignored. Drop it and spend goodwill on #1.

Not yet written: the round-3 reply. Keep it SHORT (4 asks), lead with what they got right.
Reply to `resumeservice@naukri.com`, subject unchanged, must keep `#VR#260817TS43026084_43453892#`.

### Resume layout files — RESOLVED 24 Aug

Hybrid built. `resume/resume_hybrid_MASTER.md` is the **content SSOT** — edit that, then mirror into the HTML.

| File | Use |
|---|---|
| `resume_sameer_khan_v2.html` | **Primary.** Designed two-column w/ sidebar + metrics. Send this. |
| `resume_hybrid_v1.html` | Single-column ATS-safe fallback for strict Workday/Greenhouse portals |
| `resume_hybrid_MASTER.md` | Plain-text content source — hand this to Naukri |

**To make the PDF — scripted (preferred):**
```bash
cd resume
/private/tmp/.../scratchpad/pdfenv/bin/python make_pdf.py resume_sameer_khan_v2.html
```
`make_pdf.py` is committed in `resume/`. It needs a playwright venv:
```bash
python3 -m venv pdfenv && ./pdfenv/bin/pip install playwright pypdf
./pdfenv/bin/playwright install chromium
```
(the scratchpad venv is temporary — recreate it anywhere, incl. Windows.)

**Manual fallback:** open the .html in Safari → File → Print → tick **"Print backgrounds"**
→ A4, scale **100%** → PDF ▾ → Save as PDF. Without that checkbox the navy band and tinted
sidebar render white.

**Verified 25 Aug** by rendering and inspecting: 2 pages, page-1 main slack 19.7mm /
sidebar 12.9mm, page-2 slack 16.0mm. PDF text layer extracts main content BEFORE sidebar
(correct ATS reading order), 719 words selectable on page 1, **0 ligature glyphs** —
`font-variant-ligatures:none` is set so "first" does not extract as "ﬁrst".
Old files kept for reference only: `resume_sameer_khan_sr_ML.pdf`, `CustCopy2.pdf` (Naukri draft 1).

### Open Finding — 02_INTERVIEW_PACK.md is stale

Written against an older resume. Must be rewritten before any interview. Confirmed mismatches:

| Pack says | Final resume says |
|-----------|-------------------|
| "8 years", "94% accuracy", "60% RCA reduction" | 9 years; weighted F1 0.959–0.972 and 0.975; P1 RCA = 5 services / 21,096 pages / 2.3% true failure rate |
| RAG project = FAISS + Streamlit + `llama3.2:1b` | **Rulebook-RAG**: 36 classes, hand-written BM25 + RRF (+5.2 pts), cross-encoder (+3.9 pts), 0 invalid of 775 |
| "You have done QLoRA with Mistral-7B" | Not on the resume — **and Phase 09 is parked, so it cannot be defended. Do not claim this.** |

Zero coverage in the pack for: Rulebook-RAG, ONNX export, 643 classes, dedup pipeline, Union-Find, per-class thresholds, Textract, L40S. Scheduled for days 12–14.

---

## Machine Differences (matters for drills)

| | Mac (here) | Windows |
|---|---|---|
| **torch** | 2.12.0, CPU | 2.5.1 + cu121, GTX 1650 Ti |
| **Good for** | Drills 1–8 (all CPU, tiny tensors, seconds to run) | GPU sessions, Phase 05/10 |
| **Known block** | — | Phase 09 fine-tuning parked: torch 2.6 not on cu121, trl 1.6 meta-tensor bug |

All interview drills are **deliberately CPU-only and seed-fixed** — they run identically on both machines. No GPU needed, no environment drift.

---

## Active Learning Arc

| Layer | Current Focus | Status |
|-------|--------------|--------|
| Theory | All 11 folders complete · `4.nlp/03_sequence_models/` end-to-end arc added 22 Aug (RNN→LSTM→GRU→Attention→Transformer, hand-computed) | ✅ Done |
| Code Practice | Phase 05 Transformers (S1-S7 exist) → Phase 06-10 also exist | 🔧 Need to run sessions |
| Root Packs | 00_HUB, 01_CAREER, 02_INTERVIEW, 03_LEARNING | ✅ Done |

---

## Code Practice Phase Status

| Phase | Topic | Sessions | Confirmed Run | Notes |
|-------|-------|----------|--------------|-------|
| 05 | Transformers | S01-S07 | ✅ All 7 run | GPU, GTX 1650 Ti |
| 06 | LLMs | S01-S03 | ✅ All 3 run | OpenAI gpt-4o-mini; S01 also tested on Ollama |
| 07 | RAG | S01-S05 | ✅ All 5 run | faiss-cpu + rank-bm25 installed |
| 08 | Agents | S01-S04 | ✅ All 4 run | langchain-openai + langgraph installed |
| 09 | Fine-tuning | S01-S06 | ⏸ Parked | torch 2.6 not on cu121; trl 1.6 meta tensor bug with 2.5.1 |
| 10 | Document AI | S01-S04 | unknown | Check _details.md badges |

---

## Pending Cleanup

- [x] Delete `5.transformers/02_models/04_efficient_transformers copy.md` — DONE. Verified a strict
      subset of the original (0 unique lines) and referenced nowhere before removing.
- [ ] Move scripts out of `junk/` → root or `scripts/`
- [ ] Update `progress.md` to reflect actual session state

---

## How to Update This File

After any work session, before `git push`, update only these blocks:

**Last Session** — overwrite machine, date, what you did, files changed.

**Next Task** — overwrite with exactly what to do next (be specific: file name, command, topic).

**Drill Schedule status column** — while the interview arc is active. Use `🔧 Built` → `✅ Passed` (record your time and score, e.g. `✅ 13/13 in 24 min`).

That's it. Keep everything else as-is until it changes.

**On the other machine:** `git pull`, open this file, read **Next Task**, go. Nothing else to reconstruct.

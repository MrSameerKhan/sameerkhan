# Mastery Plan — Sequence Models → LLMs

> Goal stated 25 Aug 2026: **perfection up to and including LLMs.**
> Theory is already written for every topic below. This file does not re-explain anything —
> it links to the canonical file (Tier 2) and defines how each topic is *tested*.
> Status lives here; cross-machine handoff stays in `SYNC.md`.

---

## The standard — what "perfect" means

A topic is **not done** when you have read it. It is done when all four pass, on the same day,
without notes:

| Gate | Test |
|------|------|
| **G1 Draw** | Whiteboard the full forward path from memory, in the time budget |
| **G2 Compute** | Hand-compute one numeric step with real numbers (the `_end_to_end.md` files already give you the numbers) |
| **G3 Code** | Type the core mechanism in ≤20 min, no reference, tests green |
| **G4 Defend** | Answer the three killer questions out loud, unprompted |

**Rule: no topic advances until all four gates pass.** One topic done to G4 beats three read.

---

## How to run ONE topic — the repeatable loop

Budget: **one topic per sitting, ~2.5 hours.** Do not split a topic across days — the four
gates must pass on the same day or the recall claim is not real.

**Step 1 · Read (30 min).** Open the canonical file. Read once, straight through, no notes.
You are refreshing, not learning — this material is already written.

**Step 2 · G1 Draw (time-budgeted).** Close the file. Blank paper. Draw the full forward path
from memory inside the budget in the topic table. Photograph it into the whiteboard folder.
*If you open the file mid-draw, the attempt is void — restart at Step 1 tomorrow.*

**Step 3 · Mark the draw (10 min).** Reopen the file. Mark your board against it in red:
what did you omit, misorder, or get dimensionally wrong? **The red marks are the topic** —
everything you got right was already known.

**Step 4 · G2 Compute (20 min).** Take the worked numbers from the `_end_to_end.md` file and
redo one step by hand. Match to 2 decimal places. This is what catches "I know the shape of it
but not the mechanism."

**Step 5 · G3 Code (20 min, hard stop).** Run the drill. No reference, no autocomplete help,
timer visible. Stop at 20:00 whether or not tests are green — an overrun is a fail, and the
fail is the information.

**Step 6 · G4 Defend (15 min).** Answer the three killer questions **out loud, standing up**.
Record on your phone. Play it back. If you hear "um, basically, kind of", you do not own it yet.

**Step 7 · Log (5 min).** Tick the Progress table below. Update `SYNC.md`. Commit both with the
work — same commit, not later.

**On a fail:** do not retry the same day. Re-read only the section you missed, and re-run the
failed gate at the start of the next sitting. A topic may take two sittings; that is normal
and is not a reason to move on.

---

## Stage 0 — CLOSED

RNN → LSTM → GRU → Attention → Transformer.
Theory: `4.nlp/03_sequence_models/02..06_*_end_to_end.md`
Whiteboards: `4.nlp/03_sequence_models/whiteboard/1..5` ✅
**G3 outstanding:** Drill 01 (multi-head attention) built 14 Aug, never attempted.

---

## THE TRACK — LLM Architecture (scope set 25 Aug)

**Two stages.** Boards 6–16 are **architecture**: how the model is built. Boards 17–21 are
**what makes it an LLM**: how a next-token predictor becomes something that follows instructions.
Architecture alone does not get you there, so 17–21 are on the track, not parked.
*(Reordered 28 Aug — see "Ordering logic" for the two moves and why.)*

**Done means:** given any model card — Llama, Mistral, DeepSeek, Qwen — you can read its config
and draw the whole model from memory: tokens → vectors, what each block does, what changed since
the 2017 paper and why, how generation runs, and where the memory goes.

Boards 1–5 (RNN → Transformer encoder) are complete: `4.nlp/03_sequence_models/whiteboard/`.

| # | Board | What is NEW on this board | G1 | Theory (canonical) |
|---|-------|---------------------------|----|--------------------|
| 6 | **Transformer Decoder** | Masked self-attention · **cross-attention** · teacher forcing vs autoregressive | 25 | `4.nlp/03_sequence_models/06c_transformer_decoder_cross_attention_end_to_end.md` |
| 7 | **Tokenization → Embedding** | BPE merges · vocab → embedding matrix · weight tying · fertility economics | 15 | `4.nlp/01_fundamentals/04_tokenization_end_to_end.md` (tokens) + `05_embedding_lookup_end_to_end.md` (vectors) |
| 8 | **BERT** | Encoder-only · MLM · `[CLS]`/`[SEP]` · segment embeddings | 20 | `5.transformers/02_models/05_bert_end_to_end.md` |
| 9 | **GPT** | Causal LM · weight tying · **no cross-attention** · post-LN vs pre-LN | 20 | `5.transformers/02_models/06_gpt1_end_to_end.md` + `06b_gpt2_end_to_end.md` + `06c_gpt3_end_to_end.md` |
| 10 | **T5 / BART** | T5: relative position bias · RMSNorm · span corruption. BART: denoising, full-document target. **BART's decoder is Donut's.** | 25 | `5.transformers/02_models/07_t5_end_to_end.md` + `07b_bart_end_to_end.md` |
| 11 | **Decoding** ← *moved up from 15* | Greedy / beam / top-k / top-p / min-p / typical · repetition + length penalty | 20 | `4.nlp/03_sequence_models/07_decoding_strategies.md` |
| 12 | **Attention at scale** ← *now before the modern block* | **KV cache growth** · Flash Attention (tiling, SRAM vs HBM) · PagedAttention · memory- vs compute-bound | 25 | `5.transformers/02_models/04b_attention_at_scale_end_to_end.md` (arithmetic) + `04_efficient_transformers.md` (survey) + `6.llms/05_vllm_internals.md` |
| 13 | **Modern LLM block / Llama 3** ← *was 11* | **pre-LN vs post-LN** · RMSNorm · SwiGLU · RoPE · MHA→MQA→GQA · reading a model card | 25 | `5.transformers/02_models/08_modern_llm_architecture.md` (mechanisms) + `08b_llama3_end_to_end.md` (Llama 3 configs + arithmetic) |
| 14 | **Long context** | RoPE scaling — PI / NTK / YaRN · ALiBi · sliding window | 20 | `5.transformers/02_models/11_long_context_scaling.md` |
| 15 | **Mixture of Experts** | Router · top-k experts · load-balancing loss | 20 | `5.transformers/02_models/10_mixture_of_experts.md` |
| 16 | **Speculative decoding** ← *split out of old 15* | Draft model · verify / reject · why it needs a KV cache | 15 | `5.transformers/02_models/13_speculative_decoding.md` |

### Stage 2 — base model → useful model (added 28 Aug, was "out of scope")

| # | Board | What is NEW on this board | G1 | Theory (canonical) |
|---|-------|---------------------------|----|--------------------|
| 17 | **Pretraining + scaling laws** | Next-token objective at scale · Chinchilla · what "emergence" is | 20 | `5.transformers/01_fundamentals/04_pretraining_objectives.md` + `4.nlp/03_sequence_models/08_scaling_laws_emergent.md` |
| 18 | **SFT / instruction tuning** | Prompt masking in the loss · chat templates · dataset shape | 20 | `6.llms/02c_sft_end_to_end.md` (arithmetic) + `07_dataset_preparation.md` (formats) |
| 19 | **LoRA / QLoRA** | Low-rank ΔW · where it attaches · rank/alpha · merging · what NF4 quantizes | 20 | `5.transformers/02_models/09b_lora_qlora_end_to_end.md` (arithmetic) + `09_parameter_efficient_tuning.md` (landscape) |
| 20 | **RLHF → DPO** | Reward model · PPO loop · **the Z(x) cancellation** · why DPO deletes it | 25 | `6.llms/03c_dpo_end_to_end.md` (derivation) + `03_alignment.md` + `06_alignment_follow_ups.md` |
| 21 | **Evaluation** | Perplexity vs benchmarks · pass@k · MC floors · LLM-as-judge · contamination | 20 | `6.llms/04b_evaluation_end_to_end.md` (arithmetic) + `04_evaluation.md` (metrics) |

**Ordering logic:** 6 completes the original paper, so 8/9/10 become *subtractions* from something
already owned rather than new constructions. 9 must follow 6 so that "GPT is decoder-only but has
no cross-attention" is concrete, not memorised — it is a standard interview trap. 13 needs 9 as the
baseline it diffs against. 14–16 are inference-time and only make sense once 13 exists.

**Two moves made 28 Aug:**

1. **Decoding 15 → 11.** Board 10 leaves you holding a probability row over the vocabulary with no
   way to turn it into a token. Answering that four boards later was backwards. It sits after the
   BERT/GPT/T5 triad rather than splitting it, since those three are one idea.
2. **Attention-at-scale 12 → before the modern block (now 13).** The old board 11 asked *"what does
   GQA trade away, and why is it worth it?"* — unanswerable without KV-cache arithmetic, which was
   board 12. The board that asks the question now comes after the board that sizes the cache.
   Speculative decoding split out to 16, because it needs the cache too.

Boards go in `5.transformers/whiteboard/` numbered `6.decoder` … `21.evaluation`, continuing the
existing numbering rather than restarting.

### Killer questions per board (G4)

| # | Ask yourself, out loud |
|---|---|
| 6 | Where do K and V come from in cross-attention vs self-attention? Why is training parallel but inference sequential? |
| 7 | Why does BPE beat word-level on OOV, and why can byte-level BPE never emit `<UNK>`? Why is token count ≠ word count for billing? What fraction of GPT-2 small is the embedding table, and of GPT-3? Why does Llama 3 decline a 6.5% saving by untying? |
| 8 | Why mask 15%? Why is BERT useless for generation? Where does `[CLS]` meaning come from? |
| 9 | What exactly does the LM head tie to, and why? What does the causal mask actually sever? Pre-LN vs post-LN — which is GPT-2, and what else changed with it? |
| 10 | Draw Donut on top of this board. Which half is Swin, which is BART? How does T5's target differ from BART's, and why does that make T5's decoder cheaper? |
| 11 | Why does beam search hurt open-ended generation? What does temperature do to the logits, precisely (and what does it NOT change)? Which method can drop the most-probable token, and why can't top-p? Does length_penalty > 0 reward long or short outputs? |
| 12 | Compute KV cache size for 7B, 4k context, batch 8 — and is it bigger or smaller than the weights? Does Flash Attention change the output? Why is decode memory-bound but prefill compute-bound? |
| 13 | Why did pre-LN remove the need for LR warmup? What does GQA trade away, and why is it worth it? Given a config.json, derive param count, d_head, KV cache and the tokenizer family. Why is Llama 3's d_ff 3.5x d_model and not 8/3? |
| 14 | Why does RoPE extrapolate when learned positions cannot? What breaks first at 128k — and how many dimensions is it? Why does NTK beat PI? What does YaRN's ramp key off? |
| 15 | Why is a 47B MoE cheaper to serve than 47B dense — cheaper in WHAT exactly, and what is unchanged? Why is 8x7B not 56B? Why is L_aux a PRODUCT f_i*P_i? |
| 16 | Why is speculative decoding lossless rather than an approximation — write the identity. What does the residual distribution do? Why is expected-tokens NOT the speedup, and when does K=8 make things SLOWER? |
| 17 | Why is 7B on 3T better than 70B on 300B — better in WHAT, and cheaper how? What are the Chinchilla exponents (not the rounded ones)? What exactly "emerges", and why is p^K the counter-argument? |
| 18 | Why mask the prompt out of the loss — and is masking just a rescale? Why -100 and not 0? What are the TWO distinct template failures, and which one is the most common SFT bug? |
| 19 | What does LoRA add, and to which matrices? Why is B initialised to zero and A not — what breaks if both are zero? Why does rank 8 suffice, and when does it NOT? What is QLoRA's biggest memory win — weights or optimizer state? |
| 20 | Derive DPO: why does beta*log Z(x) cancel, and what property of Bradley-Terry makes that work? What is the KL term stopping, and what happens as beta -> 0? How many models does PPO hold vs DPO? What does DPO GIVE UP? |
| 21 | Give THREE independent reasons perplexity is a bad headline metric. Write pass@k and say what it rewards. What does 27% on MMLU actually mean? How do you detect contamination with no training-data access? |

---

## Drill queue (G3)

| Drill | Covers | Board | Status |
|-------|--------|-------|--------|
| 01 | Multi-head attention + causal mask | 5–6 | 🔧 built · **not attempted since 14 Aug** |
| 02 | BPE: learn merges, encode, decode | 7 | ⬜ |
| 03 | Causal LM loss: shift, cross-entropy, perplexity | 9 | ⬜ |
| 04 | Cross-attention: encoder K/V, decoder Q | 6, 10 | ⬜ |
| 05 | RMSNorm + SwiGLU + RoPE, verify relative-position property | 13 | ⬜ |
| 06 | KV cache: naive vs cached, assert identical logits | 12 | ⬜ |
| 07 | GQA: reshape K/V heads, param + cache savings | 13 | ⬜ |
| 08 | MoE router: top-k gate, load-balance loss | 15 | ⬜ |
| 09 | Sampling: greedy / top-k / top-p / temperature | 11 | ⬜ |

All CPU-only and seed-fixed — identical on Mac and Windows. Each is built when its board is reached,
so it is fresh rather than stale.

---

## Progress

| Board | G1 draw | G2 compute | G3 code | G4 defend |
|-------|---------|------------|---------|-----------|
| 1–5 Sequence models → Encoder | ✅ | ✅ | ⬜ 0/1 | ⬜ |
| 6 Transformer Decoder | ⬜ | ⬜ | ⬜ | ⬜ |
| 7 Tokenization | ⬜ | ⬜ | ⬜ | ⬜ |
| 8 BERT | ⬜ | ⬜ | — | ⬜ |
| 9 GPT | ⬜ | ⬜ | ⬜ | ⬜ |
| 10 T5 / BART | ⬜ | ⬜ | ⬜ | ⬜ |
| 11 Decoding | ⬜ | ⬜ | ⬜ | ⬜ |
| 12 Attention at scale | ⬜ | ⬜ | ⬜ | ⬜ |
| 13 Modern LLM block | ⬜ | ⬜ | ⬜ | ⬜ |
| 14 Long context | ⬜ | ⬜ | — | ⬜ |
| 15 MoE | ⬜ | ⬜ | ⬜ | ⬜ |
| 16 Speculative decoding | ⬜ | ⬜ | — | ⬜ |
| 17 Pretraining + scaling laws | ⬜ | ⬜ | — | ⬜ |
| 18 SFT / instruction tuning | ⬜ | ⬜ | ⬜ | ⬜ |
| 19 LoRA / QLoRA | ⬜ | ⬜ | ⬜ | ⬜ |
| 20 RLHF → DPO | ⬜ | ⬜ | ⬜ | ⬜ |
| 21 Evaluation | ⬜ | ⬜ | — | ⬜ |

Update this table, then `SYNC.md`, in the same commit as the work.

---

## Out of scope — deliberately parked

> **Changed 28 Aug.** Adaptation, alignment, pretraining and evaluation are **no longer parked** —
> they are boards 17–21. "Perfect in LLMs" is not reachable from architecture alone. What remains
> below is genuinely out of scope.

**Distributed training.** DDP / FSDP / ZeRO, multi-node, gradient checkpointing. You will not train
at that scale and no interview for this profile will go there before you steer it there.

**Fine-tuning as a practice** — *still open, and boards do not close it.* `code_practice/09_finetuning/`
is 6 sessions, all `🔧 Code-built`, **zero run** — parked since June on torch 2.6 / cu121 / trl.
Meanwhile the résumé skills line claims *LoRA / QLoRA / PEFT, SFT / instruction tuning, RLHF-DPO*.
Boards 18–20 let you *explain* them; "what LR and rank did you use, and how did you pick them?"
still has no answer behind it. One real QLoRA run — tiny model, 200 examples — closes that. It is an
environment fix, not study, and it is independent of the board track.

**Quantization + reasoning models** — one file each, no board allocated:
`2.deep learning/02_architectures/10_quantization_theory.md` (read alongside board 12) and
`5.transformers/02_models/14_reasoning_models.md` (read after 21). Add boards later if either
turns out to matter for a specific role.

## Also out of scope — but do not forget it

**ViT → Swin → Donut** is not on this path, and it is the single most prominent claim on the
résumé (*"Proposed and built the OCR-free successor… Donut / Swin encoder, 75M parameters"*).
Boards 6 and 10 cover its **decoder** (cross-attention, then BART), so it is half-covered —
the Swin encoder half has no board.

Run it as a parallel track after board 16, or accept that the most likely deep-dive question
has no board behind it. Theory: `9.multimodal/03_vision_transformers.md`, `05_donut_end_to_end.md`.

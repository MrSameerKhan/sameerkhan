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

**Scope: architecture only.** LoRA/QLoRA and RLHF/DPO are *adaptation* and *alignment*, not
architecture — they are parked below in "Out of scope". Fine-tuning as a practice is a separate
problem (see the note at the bottom).

**Done means:** given any model card — Llama, Mistral, DeepSeek, Qwen — you can read its config
and draw the whole model from memory: tokens → vectors, what each block does, what changed since
the 2017 paper and why, how generation runs, and where the memory goes.

Boards 1–5 (RNN → Transformer encoder) are complete: `4.nlp/03_sequence_models/whiteboard/`.

| # | Board | What is NEW on this board | G1 | Theory (canonical) |
|---|-------|---------------------------|----|--------------------|
| 6 | **Transformer Decoder** | Masked self-attention · **cross-attention** · teacher forcing vs autoregressive | 25 | `4.nlp/03_sequence_models/06_transformer_end_to_end.md` |
| 7 | **Tokenization → Embedding** | BPE merges · vocab → embedding matrix · weight tying | 15 | `5.transformers/01_fundamentals/03_tokenization.md` |
| 8 | **BERT** | Encoder-only · MLM · `[CLS]`/`[SEP]` · segment embeddings | 20 | `5.transformers/02_models/05_bert_end_to_end.md` |
| 9 | **GPT** | Causal LM · weight tying · **no cross-attention** | 20 | `5.transformers/02_models/06_gpt_end_to_end.md` |
| 10 | **T5 / BART** | Nothing new — full assembly. **This is the Donut decoder.** | 25 | `5.transformers/02_models/07_t5_end_to_end.md` |
| 11 | **Modern LLM block** | **pre-LN vs post-LN** · RMSNorm · SwiGLU · RoPE · MHA→MQA→GQA | 25 | `5.transformers/02_models/08_modern_llm_architecture.md` |
| 12 | **Attention at scale** | Flash Attention (tiling, SRAM vs HBM) · KV cache growth · PagedAttention | 25 | `5.transformers/02_models/04_efficient_transformers.md` + `6.llms/05_vllm_internals.md` |
| 13 | **Long context** | RoPE scaling — PI / NTK / YaRN · ALiBi · sliding window | 20 | `5.transformers/02_models/11_long_context_scaling.md` |
| 14 | **Mixture of Experts** | Router · top-k experts · load-balancing loss | 20 | `5.transformers/02_models/10_mixture_of_experts.md` |
| 15 | **Decoding** | Greedy / beam / top-k / top-p / temperature · speculative decoding | 20 | `4.nlp/03_sequence_models/07_decoding_strategies.md` + `02_models/13_speculative_decoding.md` |

**Ordering logic:** 6 completes the original paper, so 8/9/10 become *subtractions* from something
already owned rather than new constructions. 9 must follow 6 so that "GPT is decoder-only but has
no cross-attention" is concrete, not memorised — it is a standard interview trap. 11 needs 9 as the
baseline it diffs against. 12–15 are inference-time and only make sense once 11 exists.

Boards go in `5.transformers/whiteboard/` numbered `6.decoder` … `15.decoding`, continuing the
existing numbering rather than restarting.

### Killer questions per board (G4)

| # | Ask yourself, out loud |
|---|---|
| 6 | Where do K and V come from in cross-attention vs self-attention? Why is training parallel but inference sequential? |
| 7 | Why does BPE beat word-level on OOV? Why is token count ≠ word count for billing? |
| 8 | Why mask 15%? Why is BERT useless for generation? Where does `[CLS]` meaning come from? |
| 9 | What exactly does the LM head tie to, and why? What does the causal mask actually sever? |
| 10 | Draw Donut on top of this board. Which half is Swin, which is BART? |
| 11 | Why did pre-LN remove the need for LR warmup? What does GQA trade away, and why is it worth it? |
| 12 | Compute KV cache size for 7B, 4k context, batch 8. Does Flash Attention change the output? |
| 13 | Why does RoPE extrapolate when learned positions cannot? What breaks first at 128k? |
| 14 | Why is a 47B MoE cheaper to serve than 47B dense? What is load-balancing loss preventing? |
| 15 | Why does beam search hurt open-ended generation? What does temperature do to the logits, precisely? |

---

## Drill queue (G3)

| Drill | Covers | Board | Status |
|-------|--------|-------|--------|
| 01 | Multi-head attention + causal mask | 5–6 | 🔧 built · **not attempted since 14 Aug** |
| 02 | BPE: learn merges, encode, decode | 7 | ⬜ |
| 03 | Causal LM loss: shift, cross-entropy, perplexity | 9 | ⬜ |
| 04 | Cross-attention: encoder K/V, decoder Q | 6, 10 | ⬜ |
| 05 | RMSNorm + SwiGLU + RoPE, verify relative-position property | 11 | ⬜ |
| 06 | KV cache: naive vs cached, assert identical logits | 12 | ⬜ |
| 07 | GQA: reshape K/V heads, param + cache savings | 11 | ⬜ |
| 08 | MoE router: top-k gate, load-balance loss | 14 | ⬜ |
| 09 | Sampling: greedy / top-k / top-p / temperature | 15 | ⬜ |

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
| 11 Modern LLM block | ⬜ | ⬜ | ⬜ | ⬜ |
| 12 Attention at scale | ⬜ | ⬜ | ⬜ | ⬜ |
| 13 Long context | ⬜ | ⬜ | — | ⬜ |
| 14 MoE | ⬜ | ⬜ | ⬜ | ⬜ |
| 15 Decoding | ⬜ | ⬜ | ⬜ | ⬜ |

Update this table, then `SYNC.md`, in the same commit as the work.

---

## Out of scope — deliberately parked

**Adaptation & alignment.** LoRA / QLoRA (`02_models/09_parameter_efficient_tuning.md`) and
RLHF → DPO (`6.llms/03b_alignment_end_to_end.md`). These are not architecture. Pick them up after
board 15.

**Fine-tuning as a practice.** No board fixes this. `code_practice/09_finetuning/` is 6 sessions,
all `🔧 Code-built`, **zero run** — parked since June on torch 2.6 / cu121 / trl. Meanwhile the
résumé skills line claims *LoRA / QLoRA / PEFT, SFT / instruction tuning, RLHF-DPO*. Boards let you
explain them; "what LR and rank did you use, and how did you pick them?" still has no answer behind
it. One real QLoRA run — tiny model, 200 examples — closes that. It is an environment fix, not study.

**Pretraining, distributed training, evaluation.** Scaling laws, DDP/FSDP/ZeRO, MMLU / LLM-as-judge.
Out of the architecture scope by choice.

## Also out of scope — but do not forget it

**ViT → Swin → Donut** is not on this path, and it is the single most prominent claim on the
résumé (*"Proposed and built the OCR-free successor… Donut / Swin encoder, 75M parameters"*).
Stage B3 (cross-attention) is its decoder, so it is half-covered.

Run it as a parallel track after Stage B, or accept that the most likely deep-dive question
has no board behind it. Theory: `9.multimodal/03_vision_transformers.md`, `05_donut_end_to_end.md`.

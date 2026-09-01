# 02c — SFT / Instruction Tuning: End-to-End

> Board 18. This is where a **base model** — a next-token predictor
> ([06_gpt1_end_to_end.md](../5.transformers/02_models/06_gpt1_end_to_end.md)) — becomes something
> you can talk to.
>
> Data *formats* (ChatML, dataset sizing, synthetic generation) live in
> [07_dataset_preparation.md](07_dataset_preparation.md), which covers them well. **LoRA/QLoRA is
> board 19** ([02b_finetuning_end_to_end.md](02b_finetuning_end_to_end.md)); **RLHF/DPO is board 20**
> ([03_alignment.md](03_alignment.md)). Neither is repeated here.
>
> This file is the part with arithmetic: **what changes in the loss**, and **what breaks when the
> template is wrong**.

---

## 1. SFT is not a new objective

```
PRETRAINING   next-token prediction on raw text
SFT           next-token prediction on (prompt, response) pairs
                                        ^^^^^^^^^^^^^^^^^^^^^
```

**The loss function does not change.** It is the same cross-entropy over the same vocabulary with
the same causal mask. Three things change:

1. **The data** — curated instruction/response pairs instead of scraped text.
2. **The masking** — the prompt is excluded from the loss (§2).
3. **The formatting** — special tokens mark turn boundaries (§4).

That is the whole of SFT. It is a data and masking discipline, not an architectural one — which is
why it is cheap (hours, not months) and why 1,000 good examples can beat 50,000 mediocre ones.

---

## Table of Contents

1. SFT is not a new objective
2. **Prompt masking — computed**
3. What masking does to the gradient
4. Chat templates
5. **The template mismatch failure**
6. Dataset shape
7. Quick reference

---

## 2. Prompt masking — computed

Take a minimal "conversation" through the model from
[06_gpt1_end_to_end.md](../5.transformers/02_models/06_gpt1_end_to_end.md):

```
input :   bank  approved  │  the   loan
          └── prompt ─────┘  └─ response ─┘
target:  approved   the      loan  <eos>
```

Per-position loss (the same values verified in that file):

```
  pos   predicts       part        loss
    0   approved     PROMPT    2.559624      <- masked out
    1        the   response    2.097782
    2       loan   response    1.498288
    3      <eos>   response    2.557114
```

```
UNMASKED loss = mean over all 4 positions       = 2.178202
MASKED   loss = mean over the 3 response only   = 2.051061
difference                                        +0.127141
```

Both reproduce exactly in torch with `ignore_index=-100`.

**Position 0 is the model predicting `approved` — a *prompt* token.** Training on it teaches the
model to generate user turns. Do that at scale and you get the classic symptom: the model answers,
then writes the *next question* to itself.

```python
labels = input_ids.clone()
labels[:prompt_len] = -100        # -100 = ignore_index, excluded from CrossEntropyLoss
```

**Why `-100` and not `0`:** `0` is a valid token ID. PyTorch's `CrossEntropyLoss` reserves `-100` as
the sentinel that skips a position entirely — it contributes nothing to the numerator *or the
denominator* of the mean.

> **The nuance worth knowing.** Masking the prompt is standard but not universal. Some recipes train
> on the full sequence deliberately, on the theory that predicting the prompt is a useful auxiliary
> signal — and for *single-turn, short-prompt* data the difference is small. For multi-turn chat,
> where prompts are most of the tokens, masking is clearly correct. If asked, say "mask by default,
> and know why: without it you are training a user simulator alongside an assistant."

---

## 3. What masking does to the gradient

It is tempting to assume masking just scales the loss down. It does not:

```
  tensor    |grad| unmasked   |grad| masked    cosine similarity
  EMB             1.154689        1.441038             0.890171
  W_o             0.721807        0.989066             0.996408
  W_v             0.575374        0.785135             0.996498
  W1              0.081765        0.099131             0.976518
```

**Cosine similarity of `0.890` on the embedding table means the gradients point in genuinely
different directions** — not a rescaling. Masking changes *what the model is being pulled toward*,
which is the point.

The magnitudes are *larger* under masking because the mean is over 3 positions instead of 4 while
the retained per-position gradients are unchanged.

---

## 4. Chat templates

A template is a bijection between a list of messages and a token sequence. Two dominant families:

```
ChatML  (Qwen, many open models)
  <|im_start|>system\n You are helpful. <|im_end|>\n
  <|im_start|>user\n What is 2+2? <|im_end|>\n
  <|im_start|>assistant\n 4 <|im_end|>\n

Llama 3
  <|begin_of_text|>
  <|start_header_id|>system<|end_header_id|>\n\n You are helpful. <|eot_id|>
  <|start_header_id|>user<|end_header_id|>\n\n What is 2+2? <|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>\n\n 4 <|eot_id|>
```

Both cost about **5 non-content tokens per turn**:

```
  turns   overhead    % of a chat with 10-token msgs    with 200-token msgs
      1          6                        37.5%                       2.9%
      2         11                        35.5%                       2.7%
      4         21                        34.4%                       2.6%
      8         41                        33.9%                       2.5%
     16         81                        33.6%                       2.5%
```

**A 4-turn exchange of short messages is 34% scaffolding.** With substantial turns it is under 3%.
The overhead matters for many-tiny-turn workloads and is negligible otherwise.

**Always use `tokenizer.apply_chat_template()`.** Hand-writing the format is how the mismatch in §5
happens — the special tokens must be the *same token IDs* the model was trained on, and typing
`<|im_start|>` as text may tokenize into several ordinary tokens rather than the single special one.

---

## 5. The template mismatch failure

The board's killer question. Two distinct failures:

### 5.1 Wrong template → the model doesn't know it's its turn

```
trained on:   <|im_start|>assistant\n   4 <|im_end|>
inferred as:  <|start_header_id|>assistant<|end_header_id|>\n\n
```

The model has **never seen the inference-time prefix**. The tokens that reliably preceded "now you
speak" during training are absent. Symptoms, in order of how often they are misdiagnosed:

- the model continues the *user's* turn instead of answering
- it answers, then writes the next user question itself
- it produces plausible but oddly-framed text — the failure looks like "bad quality", not "wrong
  format", which is why it gets blamed on the fine-tune

**Nothing errors.** The tokens are all valid; they are just a sequence the model has no experience
with.

### 5.2 Missing EOS → the model never stops

If the training target omits the turn-end token (`<|im_end|>` / `<|eot_id|>`), the model is never
shown that responses terminate. At inference it generates until `max_new_tokens`, then gets cut
mid-sentence.

**This is the single most common SFT bug**, and it is entirely a data-preparation error — the model
learned exactly what it was shown.

```
train target must include the stop token:   ... 4 <|im_end|>
                                                  ^^^^^^^^^^ inside the LOSS, not just the text
```

If the stop token sits in the sequence but is masked out of the loss, the model still never learns
to emit it. **The stop token must be an unmasked label.**

---

## 6. Dataset shape

Covered in depth in [07_dataset_preparation.md](07_dataset_preparation.md); the three things worth
carrying:

```
style / persona only        ~100-500 examples     (LIMA: 1K curated beat 52K Alpaca)
narrow task                 ~1K-10K
strong general instructor   ~10K-100K
```

**Quality dominates quantity**, which is unusual in deep learning and is the practical headline of
the LIMA result. The mechanism is plausible: pretraining already installed the capabilities, and SFT
is teaching *format and selection*, not knowledge — so a small, clean, consistent set of
demonstrations is enough, and contradictory examples actively hurt.

Two mechanics:

- **Packing** — concatenating short examples to fill the context rather than padding. Improves
  throughput substantially; requires care that attention does not cross example boundaries.
- **Masking multi-turn** — in a `k`-turn conversation, *every* assistant turn is a training target
  and *every* user turn is masked. Not just the last one.

---

## 7. Quick reference

```
SFT = same next-token loss, on (prompt, response) pairs, with the prompt MASKED

labels[:prompt_len] = -100        -100 = ignore_index, skipped entirely
loss                               unmasked 2.178202  ->  masked 2.051061
gradient                           cosine 0.890 on EMB -- a DIFFERENT direction, not a rescale

template  ~5 non-content tokens/turn; use tokenizer.apply_chat_template()
stop tok  must be present AND unmasked, or the model never stops
```

**The seven things to be able to say cold:**

1. **SFT changes no objective.** Same cross-entropy, same causal mask. What changes is the data, the
   masking, and the formatting.
2. **Mask the prompt** — otherwise you are training a user simulator alongside the assistant. Here
   the masked-out position was the model predicting `approved`, a prompt token.
3. **`-100`, not `0`.** `0` is a real token ID; `-100` is PyTorch's `ignore_index` and drops the
   position from both numerator and denominator.
4. **Masking is not a rescaling** — cosine similarity `0.890` on the embedding gradient. It changes
   the direction of the update.
5. **Templates cost ~5 tokens per turn** — 34% of a short 4-turn chat, under 3% with substantial
   turns. Always use `apply_chat_template()`, never hand-written strings.
6. **Wrong template = the model doesn't know its turn began.** It continues the user turn or
   interviews itself. Nothing errors, and it gets misdiagnosed as poor quality.
7. **The stop token must be present *and unmasked*.** Omit it and the model generates until
   `max_new_tokens` — the most common SFT bug, and purely a data error.

---

## See also

- [07_dataset_preparation.md](07_dataset_preparation.md) — ChatML in full, dataset sizing, synthetic data, tool traces
- [02_finetuning.md](02_finetuning.md) — the fine-tuning landscape
- [02b_finetuning_end_to_end.md](02b_finetuning_end_to_end.md) — board 19: LoRA, QLoRA, NF4
- [03_alignment.md](03_alignment.md) — board 20: what comes after SFT
- [../5.transformers/02_models/06_gpt1_end_to_end.md](../5.transformers/02_models/06_gpt1_end_to_end.md) — the per-position losses §2 uses
- [../4.nlp/03_sequence_models/08_scaling_laws_emergent.md](../4.nlp/03_sequence_models/08_scaling_laws_emergent.md) — board 17: what pretraining already installed

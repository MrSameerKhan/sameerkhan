# Module 01 — One Call, Three Providers, Two Formats
Status: `✅ Run`

Theory: [../../8.agents/00_agent_stack_foundations.md](../../../8.agents/00_agent_stack_foundations.md) §3 (what a format is) · §4 (SDK vs framework vs protocol)

---

## Use Case

Before any loop, know exactly what goes over the wire and what comes back. Every later
module is this one call plus one idea. The claim being tested: there are only **two wire
formats**, not N providers — so "provider flexibility" is a two-item problem.

---

## The Two Formats

```
OpenAI-style                          Anthropic-style
POST /v1/chat/completions             POST /v1/messages
  ↓                                     ↓
choices[0].message.content            content[]
  = a STRING                            = a LIST OF TYPED BLOCKS
finish_reason: stop                   stop_reason: end_turn
system = a message role               system = a top-level parameter
max_tokens optional                   max_tokens REQUIRED

spoken by: OpenAI · Ollama · vLLM     spoken by: Claude only
           Groq · Together · DeepSeek             (direct, Bedrock, Vertex)
```

Local Llama speaks OpenAI, so Cell 4 reuses Cell 2's parsing verbatim. Only `base_url`
changes. That is the whole reason provider-switching is cheap.

---

## Key Implementation Details

**Never index `content[0]`.** On a thinking model index 0 is a `thinking` block with no
`.text` attribute, so `r.content[0].text` raises `AttributeError`. Always filter by type:

```python
text = "".join(b.text for b in r.content if b.type == "text")
```

**`max_tokens` is a ceiling, not a target,** and on a thinking model the reasoning is
billed against the same budget. Set it to 200 and the thinking consumes all of it:
`stop_reason` comes back `max_tokens`, `content[]` holds a thinking block with no text
block, and you get an empty answer with no error raised.

**Cell 6 drops the SDK entirely** — one `requests.post` produces the identical result.
The SDK adds typing, retries and auth handling. It adds no capability.

---

## Fixes Applied (during run)

| Change | Why |
|---|---|
| Cell 5 gained `shown~` / `unseen~` columns and a Lesson 2 block | The token table already exposed the thinking-token gap but never named it. `unseen~` is clamped at 0 because the 4-chars-per-token estimate can overshoot on short answers. |

---

## Actual Output (macOS M1, 2026-09-12)

```
providers available: {'openai': True, 'anthropic': True, 'local': True}

--- OPENAI ---
  answer lives at : choices[0].message.content
  type            : str
  stop signal     : finish_reason = stop

--- ANTHROPIC ---
  answer lives at : content[] -> the block with type == 'text'
  type            : list of ['thinking', 'text']
  stop signal     : stop_reason = end_turn

--- LOCAL (ollama) ---
  format          : OpenAI — identical parsing to Cell 2

provider    format      stop signal       in   out
openai      openai      stop              24    29
anthropic   anthropic   end_turn          33   632
local       openai      stop              42    68

--- RAW HTTP, NO SDK ---
  block types : ['thinking', 'text']
```

**Both claims held.** Anthropic returned `['thinking', 'text']`, so the `content[0]` trap
is confirmed rather than theoretical. And 632 output tokens for a one-sentence answer is
~22× OpenAI's — a thinking model bills its reasoning as output and you never see it. At
Opus 5 rates that call cost ~$0.016 against ~$0.002 without thinking, roughly 8×.

`MAX_TOK = 2000` was adequate: 632 used, about a third of budget.

---

## How to Run

```bash
conda activate sameerkhan
cd code_practice/12_agents_from_scratch
python _providers.py              # expect 3 × OK
python 01_standard_llm_call.py
```

Needs `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` in `~/.zshenv` (not `~/.zshrc` — a
notebook kernel never reads that), plus Ollama running with `llama3.2` pulled.

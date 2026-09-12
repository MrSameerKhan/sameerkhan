"""
Module 01 — One call, three providers, two formats.

THE ONE IDEA: there are only TWO wire formats in the world, not N providers.
Learn both and you can drive essentially any model.

Run:  python 01_standard_llm_call.py
 or:  Shift+Enter each `# %%` cell in the VS Code Interactive Window.
"""

# %%
# ── Cell 1: setup ─────────────────────────────────────────────────────────────
import sys; sys.path.insert(0, "..")   # shared _providers.py lives in the phase root
from _providers import get_client, available

QUESTION = "What is the maximum LTV for a first-time buyer? Answer in one sentence."

# max_tokens is a CEILING, not a target — you pay for what is generated, not this.
# It must be generous on a thinking model: Claude Opus 5 reasons before it answers,
# and that reasoning is billed against the SAME budget. Set this to 200 and the
# thinking consumes all of it, stop_reason comes back 'max_tokens', and content[]
# holds a thinking block with NO text block. Cell 3 proves it.
MAX_TOK = 2000

print("providers available:", {k: v for k, v in available().items()})
results = {}          # provider -> (text, stop_signal, tokens_in, tokens_out)


# %%
# ── Cell 2: OpenAI format ─────────────────────────────────────────────────────
# The answer is a STRING at choices[0].message.content, and the stop signal is
# called finish_reason.

client, model, _ = get_client("openai")
r = client.chat.completions.create(
    model=model, max_tokens=MAX_TOK,
    messages=[{"role": "user", "content": QUESTION}],
)

print("\n--- OPENAI ---")
print("  answer lives at : choices[0].message.content")
print("  type            :", type(r.choices[0].message.content).__name__)
print("  stop signal     : finish_reason =", r.choices[0].finish_reason)
print("  text            :", r.choices[0].message.content.strip()[:90])
results["openai"] = (r.choices[0].message.content, r.choices[0].finish_reason,
                     r.usage.prompt_tokens, r.usage.completion_tokens)


# %%
# ── Cell 3: Anthropic format ──────────────────────────────────────────────────
# The answer is a LIST OF TYPED BLOCKS at content[], and the stop signal is
# called stop_reason. There is no .text on the response.

client, model, _ = get_client("anthropic")
r = client.messages.create(
    model=model, max_tokens=MAX_TOK,
    messages=[{"role": "user", "content": QUESTION}],
)

text = "".join(b.text for b in r.content if b.type == "text")
print("\n--- ANTHROPIC ---")
print("  answer lives at : content[] -> the block with type == 'text'")
print("  type            :", type(r.content).__name__,
      "of", [b.type for b in r.content])
print("  stop signal     : stop_reason =", r.stop_reason)
print("  text            :", text.strip()[:90])
results["anthropic"] = (text, r.stop_reason, r.usage.input_tokens, r.usage.output_tokens)

# ⚠️ TWO traps live in that one call, and both bite real code:
#
#  1. content[0] is NOT the text. Opus 5 thinks first, so index 0 is a THINKING
#     block. `r.content[0].text` raises. Always filter by .type — never index.
#
#  2. Thinking is billed against max_tokens. Re-run this cell with MAX_TOK = 200
#     and you get stop_reason='max_tokens', content = ['thinking'] only, and an
#     EMPTY answer — the model never reached the text. Nothing errors; you just
#     silently get nothing back.
if not text.strip():
    print("  ⚠️  EMPTY — thinking consumed the whole budget. Raise MAX_TOK.")


# %%
# ── Cell 4: local Llama via Ollama ────────────────────────────────────────────
# Same code path as Cell 2. Ollama serves an OpenAI-compatible endpoint, so the
# ONLY change is base_url. This is why "provider flexibility" is cheap.

client, model, _ = get_client("local")
r = client.chat.completions.create(
    model=model, max_tokens=MAX_TOK,
    messages=[{"role": "user", "content": QUESTION}],
)

print("\n--- LOCAL (ollama) ---")
print("  format          : OpenAI — identical parsing to Cell 2")
print("  text            :", r.choices[0].message.content.strip()[:90])
results["local"] = (r.choices[0].message.content, r.choices[0].finish_reason,
                    r.usage.prompt_tokens, r.usage.completion_tokens)


# %%
# ── Cell 5: the lesson, side by side ──────────────────────────────────────────
print("\n" + "=" * 74)
print(f"{'provider':11} {'format':11} {'stop signal':14} {'in':>5} {'out':>5} {'shown~':>6} {'unseen~':>7}")
print("-" * 74)
for name, (txt, stop, tin, tout) in results.items():
    fmt = "anthropic" if name == "anthropic" else "openai"
    # ~4 chars per token is a rough but honest estimate for English prose.
    shown = round(len(txt.strip()) / 4)
    unseen = max(0, tout - shown)          # clamp: the estimate can overshoot slightly
    print(f"{name:11} {fmt:11} {stop:14} {tin:5} {tout:5} {shown:6} {unseen:7}")
print("=" * 74)
print("LESSON 1 — TWO formats across THREE providers. Local speaks OpenAI, so it")
print("shares a code path with hosted OpenAI. Only Anthropic needs different parsing.")
print()
print("LESSON 2 — read the 'unseen' column. All three answered in one sentence, but")
print("a thinking model BILLS ITS REASONING AS OUTPUT. Those tokens never reach your")
print("screen and you pay full output rate for them. That is why the same one-line")
print("answer costs ~8x more here. Module 11 shows this compounding once per turn.")


# %%
# ── Cell 6: proof the SDK is just a wrapper ───────────────────────────────────
# No SDK at all — one HTTP POST. client.messages.create() builds exactly this.
import os, requests

resp = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
             "anthropic-version": "2023-06-01",
             "content-type": "application/json"},
    json={"model": "claude-opus-5", "max_tokens": MAX_TOK,
          "messages": [{"role": "user", "content": QUESTION}]},
)
blocks = resp.json()["content"]
print("\n--- RAW HTTP, NO SDK ---")
print("  block types :", [b["type"] for b in blocks])
print("  text        :", "".join(b["text"] for b in blocks if b["type"] == "text")[:90])
print("\nThe SDK adds typing, retries and auth handling. It adds no capability.")

# And with weights on your own machine there is NO format at all — no HTTP, no
# JSON, just a forward pass:
#     from transformers import pipeline
#     pipeline("text-generation", model="meta-llama/Llama-3.2-3B-Instruct")(QUESTION)
# Not run here: 4 GB VRAM won't hold it at full precision. That is what Ollama
# solves — it quantises and then puts an HTTP server in front.

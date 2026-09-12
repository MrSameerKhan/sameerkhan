"""
Step 1 — A single LLM call.

Goal: know exactly what goes over the wire and exactly what comes back.
No tools, no loop, no framework. Every later step is this call plus one idea.

Run: Shift+Enter on each `# %%` cell (VS Code Interactive Window),
     or straight through with `python 01_single_call.py`.
"""

# %%
# ── Cell 1: the client ────────────────────────────────────────────────────────
import os
import anthropic

assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not visible to this interpreter"

client = anthropic.Anthropic()      # reads ANTHROPIC_API_KEY from the environment
MODEL = "claude-opus-5"

print("client ready:", MODEL)


# %%
# ── Cell 2: the simplest possible call ────────────────────────────────────────
# `messages` is a LIST of turns, each {"role": ..., "content": ...}.
# That list is the ONLY memory the model has. Internalise this now — it is the
# whole reason steps 2-6 look the way they do.
#
# max_tokens is a hard CEILING on the response, not a target. Too low and the
# reply is cut mid-sentence (stop_reason becomes "max_tokens"). ~16000 is a sane
# default in real non-streaming code; 1024 just keeps these demos snappy.

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is the maximum LTV for a first-time buyer?"}
    ],
)

print(response)          # look at the whole raw object before we dissect it


# %%
# ── Cell 3: anatomy of the response ───────────────────────────────────────────
# The response is NOT a string. Three fields carry the whole ladder.

print("stop_reason :", response.stop_reason)
print("content     :", len(response.content), "block(s)")
for i, block in enumerate(response.content):
    print(f"   [{i}] type = {block.type}")
print("usage       :", response.usage.input_tokens, "in /",
                       response.usage.output_tokens, "out")

# Text lives INSIDE a block. There is no response.text.
text = "".join(b.text for b in response.content if b.type == "text")
print("\n" + "─" * 60 + "\n" + text)


# %%
# ── Cell 4: prove the API is stateless ────────────────────────────────────────
# Nothing is stored server-side. A second call knows nothing about the first.

followup = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    messages=[{"role": "user", "content": "What did I just ask you?"}],
)

print("".join(b.text for b in followup.content if b.type == "text"))

# %%

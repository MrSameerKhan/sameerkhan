"""
A scripted stand-in for a real LLM client.

WHY THIS EXISTS — this is the important bit.

From module 12 onward we need to demonstrate that guardrails actually fire. But
you cannot ask a real model to misbehave on command:

    "please loop forever"                 -> it won't
    "please hallucinate a tool name"      -> it won't
    "please emit malformed JSON"          -> it won't

A good model is precisely the thing that makes failure modes hard to reproduce.
So the failures get scripted instead. Same interface as the OpenAI client, but
the responses come from a list you wrote.

This is not a testing shortcut — it IS the technique. Any guard you cannot
trigger on demand is a guard you have never actually verified.
"""

import json
from dataclasses import dataclass, field


# ── Minimal shapes matching the OpenAI response object ────────────────────────

@dataclass
class _Function:
    name: str
    arguments: str            # JSON *string*, exactly as the real API returns it


@dataclass
class _ToolCall:
    id: str
    function: _Function
    type: str = "function"


@dataclass
class _Message:
    role: str = "assistant"
    content: str | None = None
    tool_calls: list[_ToolCall] | None = None


@dataclass
class _Choice:
    message: _Message
    finish_reason: str


@dataclass
class _Usage:
    prompt_tokens: int = 100
    completion_tokens: int = 20
    total_tokens: int = 120


@dataclass
class _Response:
    choices: list[_Choice]
    usage: _Usage = field(default_factory=_Usage)


# ── Script builders ───────────────────────────────────────────────────────────

def tool_turn(name: str, args: dict, call_id: str = "call_1") -> _Response:
    """A turn where the model asks for a tool.
    Note content is None — that is what the real API does on a tool turn."""
    return _Response([_Choice(
        _Message(tool_calls=[_ToolCall(call_id, _Function(name, json.dumps(args)))]),
        "tool_calls",
    )])


def text_turn(text: str) -> _Response:
    """A turn where the model answers and stops."""
    return _Response([_Choice(_Message(content=text), "stop")])


def raw_turn(content: str | None, tool_calls, finish_reason: str) -> _Response:
    """Escape hatch for malformed turns the helpers above won't produce."""
    return _Response([_Choice(_Message(content=content, tool_calls=tool_calls), finish_reason)])


# ── The fake client ───────────────────────────────────────────────────────────

class FakeModel:
    """Drop-in for `client.chat.completions`. Returns scripted turns in order.

        fake = FakeModel([tool_turn("calculate", {"expression": "2+2"}),
                          text_turn("The answer is 4.")])
        resp = fake.create(model="...", messages=[...])

    When the script runs out it repeats the LAST turn forever — which is exactly
    how you simulate an agent that will not stop.
    """

    def __init__(self, script: list[_Response], *, repeat_last: bool = True):
        self.script = script
        self.repeat_last = repeat_last
        self.calls = 0

    def create(self, **_kwargs) -> _Response:
        self.calls += 1
        if self.calls <= len(self.script):
            return self.script[self.calls - 1]
        if self.repeat_last and self.script:
            return self.script[-1]
        raise StopIteration("FakeModel script exhausted")

    # Lets the fake sit where `client.chat.completions` sits in real code
    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self


if __name__ == "__main__":
    fake = FakeModel([
        tool_turn("search_policy", {"query": "ltv"}),
        text_turn("Maximum LTV is 95% for first-time buyers."),
    ])
    for i in range(3):
        r = fake.create()
        m = r.choices[0].message
        kind = f"tool:{m.tool_calls[0].function.name}" if m.tool_calls else f"text:{m.content!r}"
        print(f"call {i + 1}: finish_reason={r.choices[0].finish_reason:11} {kind}")
    print("\nNote call 3 repeated call 2 — that is how a runaway loop is simulated.")

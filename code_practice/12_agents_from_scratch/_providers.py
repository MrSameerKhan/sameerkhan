"""
One client factory so modules 01-08 switch provider with a single constant.

Three backends, two wire formats:

    openai      OpenAI hosted          OpenAI format
    local       Ollama on your GPU     OpenAI format  <- same code path as above
    anthropic   Claude hosted          Anthropic format

That "local speaks OpenAI" line is the whole reason provider-switching is cheap:
Ollama exposes an OpenAI-compatible endpoint, so swapping base_url is enough.
Only Anthropic needs a different code path, because only Anthropic has a
different format. Two formats, not N providers.
"""

import os

OLLAMA_BASE = "http://localhost:11434/v1"

CONFIG = {
    "openai":    {"model": "gpt-4.1-mini",  "format": "openai"},
    "local":     {"model": "llama3.2",      "format": "openai"},
    "anthropic": {"model": "claude-opus-5", "format": "anthropic"},
}


def get_client(provider: str):
    """Return (client, model_name, wire_format) for the named provider."""
    if provider not in CONFIG:
        raise ValueError(f"Unknown provider '{provider}'. Pick one of: {', '.join(CONFIG)}")

    cfg = CONFIG[provider]

    if provider == "anthropic":
        import anthropic
        _require("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(), cfg["model"], cfg["format"]

    from openai import OpenAI

    if provider == "local":
        # api_key is ignored by Ollama but the SDK requires a non-empty string.
        return OpenAI(base_url=OLLAMA_BASE, api_key="ollama"), cfg["model"], cfg["format"]

    _require("OPENAI_API_KEY")
    return OpenAI(), cfg["model"], cfg["format"]


def _require(var: str) -> None:
    if not os.environ.get(var):
        raise RuntimeError(
            f"{var} is not visible to this interpreter.\n"
            f"  - Check you selected the 'sameerkhan' conda env as the interpreter.\n"
            f"  - Env vars are read once at process start: after setx, RESTART the kernel."
        )


def available() -> dict[str, bool]:
    """Which providers can actually run right now. Useful at the top of a module
    so it degrades gracefully instead of dying on the first call."""
    import urllib.request

    ok = {
        "openai":    bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "local":     False,
    }
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        ok["local"] = True
    except Exception:
        pass          # Ollama not running — `ollama serve`
    return ok


if __name__ == "__main__":
    print("provider availability:")
    for name, ready in available().items():
        mark = "OK " if ready else "-- "
        note = "" if ready else "  (key missing, or `ollama serve` not running)"
        print(f"  {mark} {name:10} {CONFIG[name]['model']:16} {CONFIG[name]['format']}{note}")

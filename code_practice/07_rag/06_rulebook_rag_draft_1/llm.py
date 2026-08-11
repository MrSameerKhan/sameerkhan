"""LLM client + StubLLM + response cache + cost estimator. Per §22.2.

Every real call goes through Anthropic's Messages API using output_config.format for
structured output (not tool-use) since the spec's schemas are plain JSON, not tool calls.
temperature is never sent — see config.py's note on why (Opus 5 / Sonnet 5 reject it).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import anthropic

import config

# USD per 1M tokens: (input, output). Standard rates; Sonnet 5 has an introductory
# $2/$10 rate through 2026-08-31 not reflected here — treat estimates as an upper bound.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _estimate_tokens(text: str) -> int:
    """Rough char/4 heuristic for pre-call dry-run estimates only. Real usage always
    comes from response.usage — this is never used to compute an actual bill."""
    return max(1, len(text) // 4)


def estimate_cost(n_calls: int, avg_in: int, avg_out: int, model: str) -> float:
    price_in, price_out = PRICING.get(model, PRICING["claude-opus-5"])
    return n_calls * (avg_in * price_in + avg_out * price_out) / 1_000_000


# Older models (Haiku 4.5, Sonnet 4.5, ...) reject the `effort` parameter outright (400).
# Only pass it to models that actually support it.
_MODELS_WITHOUT_EFFORT = {"claude-haiku-4-5-20251001", "claude-haiku-4-5"}


def _output_config(schema: Optional[dict], model: str, effort: str) -> dict:
    cfg = {} if model in _MODELS_WITHOUT_EFFORT else {"effort": effort}
    if schema is not None:
        cfg["format"] = {"type": "json_schema", "schema": schema}
    return cfg


def _cache_key(model: str, system: str, user: str, schema: Optional[dict]) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "user": user, "schema": schema},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RefusalError(RuntimeError):
    """Raised when Claude's safety classifiers decline a request (stop_reason == 'refusal')."""


class SchemaValidationError(RuntimeError):
    """Raised when the response fails schema validation twice in a row."""


class LLM:
    def __init__(self, use_cache: bool = True):
        self._client = anthropic.Anthropic()
        self.use_cache = use_cache
        config.PATHS.llm_cache_dir.mkdir(parents=True, exist_ok=True)

    def call(
        self,
        system: str,
        user: str,
        schema: Optional[dict] = None,
        model: Optional[str] = None,
        max_tokens: int = 8000,
        effort: str = "high",
        dry_run: bool = False,
    ):
        model = model or config.LLM_MODEL

        if dry_run:
            in_tok = _estimate_tokens(system) + _estimate_tokens(user)
            out_tok = min(max_tokens, 500)
            print(f"[dry-run] model={model} effort={effort}")
            print(f"[dry-run] estimated input tokens: ~{in_tok}")
            print(f"[dry-run] projected cost @ ~{out_tok} output tokens: "
                  f"${estimate_cost(1, in_tok, out_tok, model):.4f}")
            return None

        cache_path = None
        if self.use_cache:
            key = _cache_key(model, system, user, schema)
            cache_path = config.PATHS.llm_cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                return cached["parsed"] if schema else cached["text"]

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        output_config = _output_config(schema, model, effort)
        if output_config:
            kwargs["output_config"] = output_config

        response = self._client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise RefusalError(
                f"Model declined the request (category={getattr(response.stop_details, 'category', None)})"
            )

        text = next((b.text for b in response.content if b.type == "text"), "")

        if schema is not None:
            result = self._parse_with_retry(text, system, user, schema, model, max_tokens, effort)
        else:
            result = text

        if cache_path is not None:
            cache_path.write_text(json.dumps({"text": text, "parsed": result if schema else None}))

        return result

    def _parse_with_retry(self, text, system, user, schema, model, max_tokens, effort):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # output_config.format normally guarantees valid JSON; a failure here means a
        # refusal-adjacent or truncated response. Retry once before raising.
        retry = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=_output_config(schema, model, effort),
        )
        retry_text = next((b.text for b in retry.content if b.type == "text"), "")
        try:
            return json.loads(retry_text)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(f"Schema validation failed twice for model={model}") from e


class StubLLM:
    """Deterministic, schema-valid fake responses. Zero cost, no network. §21.2."""

    def call(
        self,
        system: str,
        user: str,
        schema: Optional[dict] = None,
        model: Optional[str] = None,
        max_tokens: int = 8000,
        effort: str = "high",
        dry_run: bool = False,
    ):
        if dry_run:
            print(f"[dry-run/stub] model={model or config.LLM_MODEL} effort={effort} — no call made")
            return None
        if schema is None:
            return "[stub response]"
        return self._fake_for_schema(schema)

    @staticmethod
    def _fake_for_schema(schema: dict) -> dict:
        # Only the shapes this project's schemas actually need (§6.5 decision, §6.2
        # rulebook entry). Falls back to a type-driven default for anything else.
        props = schema.get("properties", {})
        out = {}
        for name, spec in props.items():
            out[name] = StubLLM._fake_value(name, spec)
        return out

    @staticmethod
    def _fake_value(name: str, spec: dict):
        t = spec.get("type")
        if name == "abstain":
            return False
        if name in ("confidence", "llm_confidence"):
            return 0.5
        if name == "citations":
            return ["stub.inc.1"]
        if t == "string":
            return "stub"
        if t == "boolean":
            return False
        if t == "number":
            return 0.5
        if t == "integer":
            return 1
        if t == "array":
            item_spec = spec.get("items", {"type": "string"})
            return [StubLLM._fake_value(name, item_spec) for _ in range(2)]
        if t == "object":
            return {}
        return None


def get_llm(use_stub: bool = False, use_cache: bool = True):
    if use_stub:
        return StubLLM()
    return LLM(use_cache=use_cache)

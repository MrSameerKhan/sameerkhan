"""Anthropic client + StubLLM + disk cache + real cost/latency counter (spec §3, §7).

Model-specific facts this file depends on (verified against the current API):
  - Haiku 4.5 ACCEPTS `temperature`  (Opus 5 / Sonnet 5 reject it with a 400)
  - Haiku 4.5 REJECTS `output_config.effort` with a 400 — never send it
  - Haiku 4.5 supports structured outputs via `output_config.format`
"""

import hashlib
import json
import statistics
import time
from typing import Optional

import anthropic

import config

class RefusalError(RuntimeError):
    """Claude's safety classifiers declined the request (stop_reason == 'refusal')."""

class SchemaValidationError(RuntimeError):
    """Response failed JSON parsing twice in a row."""

def _cache_key(model: str, system: str, user: str, schema: Optional[dict]) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "user": user, "schema": schema},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

class UsageLog:
    """Real token counts and latencies. Cache hits keep their original token
    counts (so list-price cost stays honest) but are excluded from latency
    percentiles — a 1ms disk read is not an API latency sample."""

    def __init__(self):
        self.records: list[dict] = []
        config.PATHS.results_dir.mkdir(parents=True, exist_ok=True)
        self._path = config.PATHS.results_dir / "usage.jsonl"

    def add(self, model, input_tokens, output_tokens, latency_s, cached):
        rec = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": round(latency_s, 3),
            "cached": cached,
        }
        self.records.append(rec)
        with self._path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    @property
    def cost_usd(self) -> float:
        """List-price cost of the work done, counting cached calls at what they
        originally cost. This is the number RESULTS.md reports."""
        total = 0.0
        for r in self.records:
            price_in, price_out = config.PRICING.get(r["model"], (0.0, 0.0))
            total += (r["input_tokens"] * price_in + r["output_tokens"] * price_out) / 1e6
        return total

    @property
    def billed_usd(self) -> float:
        """What you actually spend this run — cache hits are free."""
        total = 0.0
        for r in self.records:
            if r["cached"]:
                continue
            price_in, price_out = config.PRICING.get(r["model"], (0.0, 0.0))
            total += (r["input_tokens"] * price_in + r["output_tokens"] * price_out) / 1e6
        return total

    def latency_percentiles(self) -> tuple[float, float]:
        live = sorted(r["latency_s"] for r in self.records if not r["cached"])
        if not live:
            return (0.0, 0.0)
        p50 = statistics.median(live)
        p95 = live[min(len(live) - 1, int(0.95 * len(live)))]
        return (p50, p95)

    def summary(self) -> str:
        p50, p95 = self.latency_percentiles()
        n_live = sum(1 for r in self.records if not r["cached"])
        return (
            f"{len(self.records)} calls ({n_live} live, {len(self.records) - n_live} cached) | "
            f"list ${self.cost_usd:.4f} | billed ${self.billed_usd:.4f} | "
            f"p50 {p50:.2f}s p95 {p95:.2f}s"
        )

USAGE = UsageLog()

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
        max_tokens: Optional[int] = None,
    ):
        model = model or config.LLM_MODEL
        max_tokens = max_tokens or config.LLM_MAX_TOKENS

        cache_path = None
        if self.use_cache:
            key = _cache_key(model, system, user, schema)
            cache_path = config.PATHS.llm_cache_dir / f"{key}.json"
            if cache_path.exists():
                cached = json.loads(cache_path.read_text())
                u = cached["usage"]
                USAGE.add(model, u["input_tokens"], u["output_tokens"], 0.0, cached=True)
                return cached["parsed"] if schema else cached["text"]

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=config.LLM_TEMPERATURE,   # valid on Haiku 4.5; would 400 on Opus 5
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        t0 = time.monotonic()
        response = self._client.messages.create(**kwargs)
        latency = time.monotonic() - t0

        USAGE.add(model, response.usage.input_tokens, response.usage.output_tokens,
                  latency, cached=False)

        if response.stop_reason == "refusal":
            raise RefusalError(f"Model declined the request (model={model})")

        text = next((b.text for b in response.content if b.type == "text"), "")
        result = self._parse(text, model) if schema is not None else text

        if cache_path is not None:
            cache_path.write_text(json.dumps({
                "text": text,
                "parsed": result if schema else None,
                "usage": {"input_tokens": response.usage.input_tokens,
                          "output_tokens": response.usage.output_tokens},
            }))
        return result

    @staticmethod
    def _parse(text: str, model: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # output_config.format normally guarantees valid JSON; a failure here
            # means truncation (hit max_tokens) or a refusal-adjacent response.
            raise SchemaValidationError(
                f"model={model} returned unparseable JSON ({len(text)} chars)"
            ) from e


class StubLLM:
    """Schema-valid canned responses. $0, no network. Built on day 0 so every
    later day is debuggable without spending money (§4)."""

    def call(self, system, user, schema=None, model=None, max_tokens=None):
        USAGE.add("stub", 0, 0, 0.0, cached=True)
        if schema is None:
            return "[stub response]"
        return {name: self._fake(name, spec)
                for name, spec in schema.get("properties", {}).items()}

    @staticmethod
    def _fake(name: str, spec: dict):
        if name == "abstain":
            return False
        if name in ("confidence", "llm_confidence"):
            return 0.5
        if name == "citations":
            return ["CLASS_000.inc.1"]
        if name == "class_id":
            return "CLASS_000"
        t = spec.get("type")
        if t == "string":
            return "stub"
        if t == "boolean":
            return False
        if t == "number":
            return 0.5
        if t == "integer":
            return 1
        if t == "array":
            return [StubLLM._fake(name, spec.get("items", {"type": "string"}))]
        if t == "object":
            return {}
        return None


def get_llm(use_stub: bool = False, use_cache: bool = True):
    return StubLLM() if use_stub else LLM(use_cache=use_cache)

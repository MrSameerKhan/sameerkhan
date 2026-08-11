"""corpus/sources.yaml -> data/pdfs/. §22.5.

Individual fetch failures are logged and the run continues — only exit non-zero if
*zero* sources succeeded. §19.4: 404/timeout must never stop the run.
"""

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import requests
import yaml

import config


@dataclass
class Source:
    class_id: str
    url: str
    kind: str


@dataclass
class FetchResult:
    class_id: str
    url: str
    status: str  # "fetched" | "cached" | "failed"
    bytes: int = 0
    path: Optional[str] = None
    error: Optional[str] = None


def load_sources(path: Path = config.PATHS.sources_yaml) -> tuple[list[Source], dict]:
    raw = yaml.safe_load(path.read_text())
    sources = [Source(**s) for s in raw["sources"]]
    return sources, raw["fetch_policy"]


def _url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def fetch_one(src: Source, policy: dict, cache_dir: Path) -> FetchResult:
    dest = cache_dir / f"{_url_hash(src.url)}.pdf"
    if policy.get("skip_if_cached", True) and dest.exists():
        return FetchResult(src.class_id, src.url, "cached", bytes=dest.stat().st_size, path=str(dest))

    headers = {"User-Agent": policy.get("user_agent", "rulebook-rag/1.0")}
    last_error = None
    for attempt in range(policy.get("retries", 2) + 1):
        try:
            resp = requests.get(src.url, headers=headers, timeout=policy.get("timeout_seconds", 30))
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("application/pdf"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(resp.content)
                return FetchResult(src.class_id, src.url, "fetched", bytes=len(resp.content), path=str(dest))
            last_error = f"HTTP {resp.status_code} ({resp.headers.get('content-type', '?')})"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}: {e}"
        time.sleep(policy.get("delay_seconds", 1.0))
    return FetchResult(src.class_id, src.url, "failed", error=last_error)


def fetch_all(sources: list[Source], policy: dict, limit: Optional[int] = None) -> list[FetchResult]:
    cache_dir = config.PATHS.pdfs_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for src in sources[:limit] if limit else sources:
        results.append(fetch_one(src, policy, cache_dir))
    return results


def cmd_fetch_corpus(args: argparse.Namespace) -> int:
    sources, policy = load_sources()
    if args.limit:
        sources = sources[: args.limit]

    if args.dry_run:
        cache_dir = config.PATHS.pdfs_dir
        n_cached = sum(1 for s in sources if (cache_dir / f"{_url_hash(s.url)}.pdf").exists())
        print(f"{len(sources)} sources: {n_cached} cached, {len(sources) - n_cached} to download")
        for s in sources:
            cached = (cache_dir / f"{_url_hash(s.url)}.pdf").exists()
            print(f"  [{'cached' if cached else 'fetch '}] {s.class_id:35s} {s.url}")
        return 0

    results = fetch_all(sources, policy)

    fetched = [r for r in results if r.status == "fetched"]
    cached = [r for r in results if r.status == "cached"]
    failed = [r for r in results if r.status == "failed"]

    index = {r.class_id: asdict(r) for r in results if r.status != "failed"}
    config.PATHS.pdfs_dir.mkdir(parents=True, exist_ok=True)
    (config.PATHS.pdfs_dir / "index.json").write_text(json.dumps(index, indent=2))

    report_lines = [
        "# Fetch report", "",
        f"- fetched: {len(fetched)}", f"- cached: {len(cached)}", f"- failed: {len(failed)}", "",
        "## Failures", "",
    ]
    for r in failed:
        report_lines.append(f"- `{r.class_id}` — {r.url} — {r.error}")
    config.PATHS.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.PATHS.reports_dir / "fetch_report.md").write_text("\n".join(report_lines))

    print(f"fetched {len(fetched)} · cached {len(cached)} · failed {len(failed)} -> reports/fetch_report.md")

    if not fetched and not cached:
        print("FAILED: zero sources succeeded")
        return 3
    return 0

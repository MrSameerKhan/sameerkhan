"""sources.yaml -> data/pdfs/. Cache by url hash, be polite, never die on a 404 (section 5.2)."""

import hashlib
import json
import time

import requests
import yaml

import config

HEADERS = {"User-Agent": "rulebook-rag/1.0 (personal learning project)"}
DELAY_S = 1.0      # be polite to irs.gov
TIMEOUT_S = 30


def url_to_path(url: str):
    return config.PATHS.pdfs_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.pdf"


def iter_targets(limit=None):
    doc = yaml.safe_load(config.PATHS.sources_yaml.read_text())
    for entry in doc["classes"][:limit]:
        yield entry["slug"], "form", entry["form_url"]
        yield entry["slug"], "instructions", entry["instructions_url"]


def probe(session, url):
    """HEAD the url. Returns (status, content_type). status 0 == transport error."""
    try:
        r = session.head(url, headers=HEADERS, timeout=TIMEOUT_S, allow_redirects=True)
        if r.status_code in (403, 405):        # some servers refuse HEAD
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT_S, stream=True)
            r.close()
        return r.status_code, r.headers.get("Content-Type", "")
    except requests.RequestException as e:
        return 0, type(e).__name__


def cmd_fetch_corpus(args):
    config.PATHS.pdfs_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    status = {}          # slug -> {"form": bool, "instructions": bool}
    index = {}

    for slug, kind, url in iter_targets(args.limit):
        entry = status.setdefault(slug, {})
        path = url_to_path(url)

        if not args.dry_run and path.exists():
            print(f"  {slug:10} {kind:12} cached")
            entry[kind] = True
            index[url] = {"slug": slug, "kind": kind, "path": path.name}
            continue

        code, ctype = probe(session, url)
        ok = code == 200 and "pdf" in ctype.lower()
        entry[kind] = ok
        print(f"  {slug:10} {kind:12} {code or ctype}{'' if ok else '   <- unusable'}")

        if ok and not args.dry_run:
            r = session.get(url, headers=HEADERS, timeout=TIMEOUT_S)
            path.write_bytes(r.content)
            index[url] = {"slug": slug, "kind": kind, "path": path.name,
                          "bytes": len(r.content)}
        time.sleep(DELAY_S)

    forms = sum(1 for v in status.values() if v.get("form"))
    instr = sum(1 for v in status.values() if v.get("instructions"))
    both = sorted(s for s, v in status.items() if v.get("form") and v.get("instructions"))
    missing = sorted(s for s, v in status.items() if v.get("form") and not v.get("instructions"))

    print(f"\n  forms available ......... {forms}/{len(status)}")
    print(f"  instructions available .. {instr}/{len(status)}")
    print(f"  USABLE (both) ........... {len(both)}/{len(status)}   <- the real taxonomy size")
    if missing:
        print(f"\n  form but no instructions ({len(missing)}):")
        print("    " + ", ".join(missing))

    if not args.dry_run:
        (config.PATHS.pdfs_dir / "index.json").write_text(json.dumps(index, indent=2))
        print(f"\n  wrote {len(index)} pdfs + index.json")

    return 0


def build_subparser(sub):
    p = sub.add_parser("fetch-corpus", help="Download form + instructions PDFs.")
    p.add_argument("--dry-run", action="store_true", help="HEAD only, download nothing.")
    p.add_argument("--limit", type=int, default=None, help="First N classes only.")
    p.set_defaults(func=cmd_fetch_corpus)

"""Command dispatch. One subparser per command; new ones land as each day ships (section 3)."""

import argparse
import importlib
import os
import shutil
import sys

import config
from fetch import build_subparser as fetch_subparser
from index import build_subparsers as index_subparsers
from retrieve import build_subparser as retrieve_subparser
from decide import build_subparser as decide_subparser
from evaluate import build_subparser as eval_subparser
from extract import build_subparsers as extract_subparsers
from rules import build_subparser as rules_subparser


IMPORT_NAME = {
    "anthropic": "anthropic",
    "sentence-transformers": "sentence_transformers",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "pymupdf": "fitz",
    "pyyaml": "yaml",
    "requests": "requests",
    "matplotlib": "matplotlib",
}


def _check(label: str, ok: bool, detail: str = "") -> bool:
    pad = "." * max(1, 34 - len(label))
    suffix = f"  ({detail})" if detail else ""
    print(f"  {label} {pad} {'PASS' if ok else 'FAIL'}{suffix}")
    return ok


def cmd_doctor(args: argparse.Namespace) -> int:
    results = []

    v = sys.version_info
    results.append(_check("python >= 3.10", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}"))

    missing = []
    for pip_name, import_name in IMPORT_NAME.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    results.append(_check(f"imports ({len(IMPORT_NAME) - len(missing)}/{len(IMPORT_NAME)})",
                          not missing, ", ".join(missing)))

    try:
        import torch
        results.append(_check("torch is CPU", not torch.cuda.is_available(), torch.__version__))
    except ImportError:
        results.append(_check("torch is CPU", False, "not installed"))

    # Both models must load from the local HF cache - no network at run time.
    for label, model_id, kind in [
        ("embed model cached", config.EMBED_MODEL, "bi"),
        ("reranker cached", config.RERANK_MODEL, "cross"),
    ]:
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            (CrossEncoder if kind == "cross" else SentenceTransformer)(model_id)
            results.append(_check(label, True, model_id.split("/")[-1]))
        except Exception as e:  # noqa: BLE001 - doctor reports, never crashes
            results.append(_check(label, False, f"{type(e).__name__}"))

    key = os.environ.get(config.ANTHROPIC_API_KEY_ENV, "")
    has_key = key.startswith("sk-ant-")
    results.append(_check("ANTHROPIC_API_KEY", has_key, f"{len(key)} chars" if key else "unset"))

    if has_key:
        try:
            from llm import get_llm
            reply = get_llm(use_cache=False).call(
                system="Reply with one word.", user="Say OK.", max_tokens=10)
            results.append(_check("LLM smoke call", bool(reply),
                                  f"{config.LLM_MODEL} -> {reply!r}"))
        except Exception as e:  # noqa: BLE001
            results.append(_check("LLM smoke call", False, f"{type(e).__name__}: {e}"))
    else:
        results.append(_check("LLM smoke call", False, "skipped, no key"))

    created = []
    for d in [config.PATHS.pdfs_dir, config.PATHS.llm_cache_dir, config.PATHS.rulebook_dir,
              config.PATHS.index_dir, config.PATHS.results_dir]:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(d.name)
    results.append(_check("directories", True,
                          f"created {', '.join(created)}" if created else "all present"))

    free_gb = shutil.disk_usage(config.ROOT).free / (1024 ** 3)
    results.append(_check("free disk", free_gb > config.MIN_FREE_DISK_GB, f"{free_gb:.0f} GB"))

    ok = all(results)
    print(f"\n  -> {'ALL PASS' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Verify the environment is ready.")
    p_doctor.set_defaults(func=cmd_doctor)

    fetch_subparser(sub)
    extract_subparsers(sub)
    rules_subparser(sub)
    index_subparsers(sub)
    retrieve_subparser(sub)
    decide_subparser(sub)
    eval_subparser(sub)


    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

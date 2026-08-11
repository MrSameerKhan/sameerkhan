"""Every CLI command. One subparser per command, wired to config.PATHS. §22.3.

Commands are added incrementally as each build phase lands — see §11.5 for the full
target list. Only `doctor` exists so far (Phase 0 prerequisite, built first per §0).
"""

import argparse
import importlib
import os
import shutil
import sys

import config
from taxonomy.build_taxonomy import cmd_build_taxonomy
from corpus.fetch_public import cmd_fetch_corpus
from corpus.extract_text import cmd_extract_text
from corpus.generate_synthetic import cmd_generate_synthetic

# pip package name -> import name, for packages where they differ
IMPORT_NAME = {
    "anthropic": "anthropic",
    "sentence-transformers": "sentence_transformers",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "pymupdf": "fitz",
    "pyyaml": "yaml",
    "requests": "requests",
    "matplotlib": "matplotlib",
    "pytest": "pytest",
}


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    pad = "." * max(1, 40 - len(label))
    suffix = f"  ({detail})" if detail else ""
    print(f"  {label} {pad} {status}{suffix}")
    return ok


def cmd_doctor(args: argparse.Namespace) -> int:
    results = []

    v = sys.version_info
    results.append(_check("python version", v >= (3, 10), f"{v.major}.{v.minor}.{v.micro}"))

    missing = []
    for pip_name, import_name in IMPORT_NAME.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    results.append(_check(f"imports ({len(IMPORT_NAME) - len(missing)}/{len(IMPORT_NAME)})",
                           not missing, ", ".join(missing) if missing else ""))

    try:
        import torch
        is_cpu_build = "+cpu" in torch.__version__ or not torch.cuda.is_available()
        results.append(_check("torch is CPU build", is_cpu_build, torch.__version__))
    except ImportError:
        results.append(_check("torch is CPU build", False, "torch not installed"))

    for model_label, model_id, loader in [
        ("embedding model cached", config.EMBED_MODEL, "sentence_transformers"),
        ("reranker cached", config.RERANK_MODEL, "sentence_transformers"),
    ]:
        try:
            if loader == "sentence_transformers":
                from sentence_transformers import CrossEncoder, SentenceTransformer
                if "reranker" in model_label:
                    CrossEncoder(model_id)
                else:
                    SentenceTransformer(model_id)
            results.append(_check(model_label, True, model_id))
        except Exception as e:  # noqa: BLE001 - doctor must never crash, only report
            results.append(_check(model_label, False, f"run §20.5 pre-download: {type(e).__name__}"))

    api_key_present = bool(os.environ.get(config.ANTHROPIC_API_KEY_ENV))
    results.append(_check("ANTHROPIC_API_KEY present", api_key_present))

    if api_key_present:
        try:
            from llm import LLM
            LLM(use_cache=False).call(
                system="Reply with one word.", user="Say OK.", max_tokens=10,
            )
            results.append(_check("LLM smoke call", True))
        except Exception as e:  # noqa: BLE001
            results.append(_check("LLM smoke call", False, f"{type(e).__name__}: {e}"))
    else:
        results.append(_check("LLM smoke call", False, "skipped, no API key"))

    expected_dirs = [
        config.PATHS.pdfs_dir, config.PATHS.out_of_taxonomy_dir, config.PATHS.llm_cache_dir,
        config.PATHS.rulebook_entries, config.PATHS.rulebook_evidence, config.PATHS.index_dir,
        config.PATHS.baseline_dir, config.PATHS.security_dir, config.PATHS.results_dir,
        config.PATHS.reports_dir, config.PATHS.logs_dir,
    ]
    missing_dirs = [str(d) for d in expected_dirs if not d.exists()]
    results.append(_check("directories", not missing_dirs,
                           f"{len(missing_dirs)} missing" if missing_dirs else ""))

    free_gb = shutil.disk_usage(config.ROOT).free / (1024 ** 3)
    results.append(_check("free disk", free_gb > config.MIN_FREE_DISK_GB, f"{free_gb:.0f} GB"))

    all_pass = all(results)
    print(f"  -> {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}")
    return 0 if all_pass else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor", help="Verify the environment is ready to build/run.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_build_tax = sub.add_parser("build-taxonomy", help="seed_taxonomy.yaml -> taxonomy.json")
    p_build_tax.add_argument("--dry-run", action="store_true")
    p_build_tax.set_defaults(func=cmd_build_taxonomy)

    p_fetch = sub.add_parser("fetch-corpus", help="Download real public PDFs.")
    p_fetch.add_argument("--limit", type=int, default=None)
    p_fetch.add_argument("--dry-run", action="store_true")
    p_fetch.set_defaults(func=cmd_fetch_corpus)

    p_extract = sub.add_parser("extract-text", help="PDFs -> data/real/documents.jsonl")
    p_extract.add_argument("--limit", type=int, default=None)
    p_extract.add_argument("--dry-run", action="store_true")
    p_extract.set_defaults(func=cmd_extract_text)

    p_synth = sub.add_parser("generate-synthetic", help="Generate synthetic filled documents.")
    p_synth.add_argument("--per-class", type=int, default=8)
    p_synth.add_argument("--limit", type=int, default=None)
    p_synth.add_argument("--dry-run", action="store_true")
    p_synth.add_argument("--stub-llm", action="store_true")
    p_synth.set_defaults(func=cmd_generate_synthetic)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

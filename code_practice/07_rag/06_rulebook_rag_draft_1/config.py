"""Central configuration for rulebook-rag. Every tunable lives here — no magic numbers elsewhere."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    taxonomy_seed: Path = ROOT / "taxonomy" / "seed_taxonomy.yaml"
    taxonomy_json: Path = ROOT / "taxonomy" / "taxonomy.json"

    sources_yaml: Path = ROOT / "corpus" / "sources.yaml"
    pdfs_dir: Path = ROOT / "data" / "pdfs"
    real_docs: Path = ROOT / "data" / "real" / "documents.jsonl"
    synthetic_docs: Path = ROOT / "data" / "synthetic" / "documents.jsonl"
    out_of_taxonomy_dir: Path = ROOT / "data" / "out_of_taxonomy"
    splits_json: Path = ROOT / "data" / "splits.json"
    llm_cache_dir: Path = ROOT / "data" / "llm_cache"

    rulebook_entries: Path = ROOT / "rulebook" / "entries"
    rulebook_evidence: Path = ROOT / "rulebook" / "evidence"
    rulebook_rejected: Path = ROOT / "rulebook" / "rejected.jsonl"

    index_dir: Path = ROOT / "index"
    index_vectors: Path = ROOT / "index" / "vectors.npy"
    index_ids: Path = ROOT / "index" / "ids.json"
    index_bm25: Path = ROOT / "index" / "bm25.pkl"
    index_manifest: Path = ROOT / "index" / "manifest.json"

    baseline_dir: Path = ROOT / "baseline"
    security_dir: Path = ROOT / "security"
    injection_cases: Path = ROOT / "security" / "injection_cases.jsonl"

    eval_set: Path = ROOT / "eval" / "eval_set.jsonl"
    results_dir: Path = ROOT / "results"
    reports_dir: Path = ROOT / "reports"
    logs_dir: Path = ROOT / "logs"


PATHS = Paths()

# --- Retrieval ---
K1 = 5  # stage-1 documentType candidates
K2 = 8  # stage-2 leaf-class candidates
K3 = 4  # nearest exemplars retrieved alongside rules
RRF_K = 60  # RRF damping constant
BM25_K1 = 1.5
BM25_B = 0.75

# --- Decision / abstention ---
ABSTAIN_THRESHOLD = 0.55
# Spec calls for temperature=0 at decide time for determinism, but Claude Opus 5 / Sonnet 5
# reject the `temperature` parameter outright (removed starting with the Opus 4.7 generation) —
# only Haiku 4.5 still accepts it. Confirmed with user: drop temperature everywhere, rely on
# output_config.format (structured outputs) for consistent shape and EFFORT_DECIDE for
# consistent behavior instead. See llm.py.
EFFORT_DECIDE = "low"
EFFORT_GENERATE = "high"
MAX_DOC_TOKENS = 1500  # document being classified (head+tail truncation)
MAX_EXEMPLAR_TOKENS = 300  # per exemplar shown in the prompt
PROMPT_TOKEN_BUDGET = 12000

# --- Models ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.environ.get("RULEBOOK_LLM_MODEL", "claude-opus-5")
LLM_MODEL_BULK = os.environ.get("RULEBOOK_LLM_MODEL_BULK", "claude-haiku-4-5-20251001")

# --- Taxonomy / scale (REDUCED version per spec §5.4 — confirmed with user) ---
TAXONOMY_VERSION = "1.0.0"
RULEBOOK_VERSION = "1.0.0"
TARGET_N_CATEGORIES = 8
TARGET_N_DOCTYPES = 40
TARGET_N_LEAF_CLASSES = 150
MIN_LEAF_CLASSES = 100  # non-negotiable floor per §5.4 — below this the project loses its reason to exist
N_HELD_BACK = 20  # fixed regardless of scale, per §5.3(4) / §13
N_OUT_OF_TAXONOMY_DOCS = 20
DEEP_FANOUT_MIN_CLASSES = 40  # one documentType must hold at least this many leaf classes
MIN_NEAR_DUPLICATE_PAIRS = 6
MIN_CONTEXT_DEPENDENT_PAIRS = 3
TAXONOMY_SEED = 42  # fixed seed for held-back class selection — must be committed, not re-rolled

# --- Eval set (REDUCED version per spec §10.1) ---
EVAL_SET_SIZE = 200
EVAL_SET_SEED = 42

# --- Validation thresholds (rulebook/validate.py) ---
VALIDATE_MIN_PRECISION = 0.80
VALIDATE_MIN_COVERAGE = 0.10
DEFECT_NAME_DISTANCE_THRESHOLD = 0.15
DEFECT_MIN_LOG_ODDS = 1.5

# --- Term statistics (rulebook/term_stats.py) ---
TERM_STATS_ALPHA = 0.5
TERM_STATS_TOP_K = 20

# --- Cost guardrails ---
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
MIN_FREE_DISK_GB = 2

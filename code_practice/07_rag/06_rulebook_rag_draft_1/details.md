# Session 6 — Rulebook-RAG

Status: `🔧 Code-built` (Phase 0 scaffolding only — taxonomy/corpus/inference/data/eval phases not started)

Spec: [RULEBOOK_RAG_PROJECT_SPEC.md](../../../RULEBOOK_RAG_PROJECT_SPEC.md) (repo root — full build spec, read this first)

---

## Use Case

Portfolio-grade RAG system that classifies documents into a hierarchical taxonomy by
**retrieving written rules and citing them**, instead of training a classifier. Two
engines: inference-time RAG (classify a document, citing the rule that produced the
decision) and data-time RAG (generate and statistically validate the rulebook itself
from labeled examples). Full detail, build order, and acceptance criteria live in the
spec linked above — this file tracks build status only.

**Scope decision (confirmed with user):** reduced version — 150 leaf classes / 40
documentTypes / 8 categories, 200-case eval set (spec §5.4, §10.1 explicitly sanction
this as "still valid").

---

## Deviations from the spec (confirmed with user)

- **Determinism at decide-time:** spec requires `temperature = 0` for classification
  calls. Claude Opus 5 / Sonnet 5 reject the `temperature` parameter outright (removed
  starting with the Opus 4.7 generation) — only Haiku 4.5 still accepts it. Resolution:
  drop `temperature` everywhere, rely on `output_config.format` (structured outputs)
  for consistent shape and low effort for consistent behavior. See `config.py`
  (`EFFORT_DECIDE`) and `llm.py`.

---

## File Structure (current)

```
06_rulebook_rag/
├── config.py            — every tunable in one place (§22.1)
├── llm.py                — Anthropic client + StubLLM + cache + cost estimator (§22.2)
├── cli.py                 — command dispatch; only `doctor` wired so far (§22.3)
├── requirements.txt
├── .gitignore
├── taxonomy/ corpus/ rulebook/ index/ retrieve/ decide/ baseline/
│   security/ eval/ data/ results/ logs/ reports/ tests/   — empty, phase-gated
```

---

## Phase status (per spec §12 build order)

- [x] Phase 0 (prerequisite) — `config.py`, `llm.py`, `cli.py` + `doctor` command
- [ ] Phase 1 — taxonomy + corpus (target 5 days, hard cap)
- [ ] Phase 2 — inference path
- [ ] Phase 3 — data path
- [ ] Phase 4 — evaluation

## Next step

`python -m cli doctor` to verify the environment (needs `pip install -r
requirements.txt` and `ANTHROPIC_API_KEY` first), then Phase 1: hand-author
`taxonomy/seed_taxonomy.yaml` — the spec calls this "the highest-judgment task in the
project," so it gets its own review pass with the user before moving to Phase 2.

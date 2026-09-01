# RULES.md — Repository Maintenance Rules

Going-forward conventions for keeping this repo bible-quality as it evolves. Companion: STRUCTURE.md (SSOT map) · PLAN.md (active migration plan).

---

## Rule 1 — Grep Before Adding

Before adding **any** new technical content to a `.md` file:

```
1. Grep the repo for the topic name
2. If found at similar depth in another file + 1-line cross-ref, no duplication
3. If found at shallower depth or nowhere + add the depth in the SSOT file
   (per the role contract below); all other mentions become 1-line cross-refs
4. NEVER let the same explanation grow in 3+ files
```

This rule exists because the repo previously accumulated the same content (FlashAttention, DPO, modern LLMs, etc.) in 5-10 files each — see STRUCTURE.md SSOT-Map.

---

## Rule 2 — Tier Discipline

The repo has 4 tiers. Each tier has a strict contract.

| Tier | Files | Owns | Doesn't own |
|------|-------|------|-------------|
| 1. Personal-Org | `00_HUB.md`, `01_CAREER_PACK.md`, `02_INTERVIEW_PACK.md`, `03_LEARNING_PACK.md` | Action plan, job market data, interview answers, study navigation | Technical explanations |
| 2. Theory | Folders `1.machine learning/` through `11.system_design/` | All technical explanations, math derivations, comparison tables | Daily actions, code |
| 3. Practice | `code_practice/` | Implementations | Conceptual explanations |
| 4. Audit/Meta | `README.md`, `STRUCTURE.md`, `PLAN.md`, `RULES.md`, `archive/` | Repo navigation, structure rules, history | Anything concept-specific |

Tier 1 references Tier 2. Tier 1 never explains. Tier 3 references Tier 2. Tier 3 never explains. **Tier 4 describes structure. Tier 4 never explains concepts.**

---

## Rule 3 — Naming Conventions

| Element | Convention |
|---------|------------|
| Theory files | `NN_topic_name.md` — two-digit zero-padded |
| Worked-example pair | `02b_topic_end_to_end.md` — paired with `02_topic.md` |
| `README.md` | `README.md` (NOT `00_roadmap.md` — the old pattern is legacy) |
| Root meta-files | `README.md`, `STRUCTURE.md`, `RULES.md`, `PLAN.md` (during migration only) |
| Folder names | `N.topic_name/` — lowercase, snake_case after the dot |
| Section anchors in links | `[file.md#section-heading-kebab](file.md#section-heading-kebab)` |

---

## Rule 4 — Cross-Reference Format

| Pattern | When | Example |
|---------|------|---------|
| Relative path with extension | Always | `[../9.multimodal/02_document_ai.md](9.multimodal/02_document_ai.md)` |
| Anchor link for sub-section | When pointing into a specific section | `[file.md#65-conformal-prediction](file.md#65-conformal-prediction)` |
| "Connections" table | At the bottom of each theory file | See any current theory file for example |
| Bare URLs | Never (use markdown link syntax) | ✗ `https://...` |

Cross-references are how the bible knits together. **Never write a path without verifying it resolves.**

---

## Rule 5 — Status Tracking

Each `code_practice/<phase>/<session>/all_details.md` carries one of three status badges in its header.

| Badge | Meaning |
|-------|---------|
| ✅ Run | Code executed end-to-end on target hardware; results captured in `all_details.md` |
| 🔧 Code-built | Code complete, tested locally, awaiting target-hardware run |
| 📄 Docs-only | Specification written, code not yet implemented |

Each theory folder's `README.md` states its reading order explicitly.

---

## Rule 6 — When in Doubt

The **SSOT map in `STRUCTURE.md`** is the source of truth. If two files conflict, update STRUCTURE.md first to pick the canonical home, then propagate. If a concept doesn't appear in STRUCTURE.md and is being introduced, add it to the SSOT map AT THE SAME TIME as you write the content.

---

## Rule 7 — Subtractive Mindset

The repo previously suffered from additive bias — every new topic became a new section in 3-5 places.

**Default to:** trim, consolidate, cross-ref. **Avoid:** dump entire explanations into multiple files because "they're related."

If a topic deserves depth, it gets one canonical home. Every other mention is a 1-2 line cross-ref.

---

## Rule 8 — No Hidden Drift

When a path changes (file moved, folder renamed): 1. Update all references in the same commit (use grep + sed) 2. Update SSOT map in STRUCTURE.md 3. Verify with a final grep — zero stale paths should remain.

The Phase 3 cross-reference pass in PLAN.md is what happens when this rule is broken. Don't break it again.

---

## Rule 9 — Update Checklist for New Files

Every new file requires a checklist. Do these in the same session as the file creation.

### New theory file (`NN_topic_name.md` in a theory folder)

| Step | What to update | What to add |
|------|---------------|-------------|
| 1 | **Folder `README.md`** | Add to Reading Order + Folder TOC + SSOT list |
| 2 | **`STRUCTURE.md`** | Add to SSOT map — which file owns this topic |
| 3 | Root `README.md` | Only if the overall Structure table changes (rare) |

### New code practice file (`code_practice/<phase>/`)

| Step | What to update | What to add |
|------|---------------|-------------|
| 1 | Root `README.md` | Update session status badge (📄 → 🔧 → ✅) in the practice plan table |
| 2 | Phase `README.md` | Add session row if it exists |

### Never needed for routine file creation

- `00_HUB.md` — only update when a **portfolio milestone** is hit or career status changes
- `01_CAREER_PACK.md` / `02_INTERVIEW_PACK.md` — only update when job search strategy changes
- `RULES.md` / `STRUCTURE.md` folder contract section — only update when a folder's scope changes

"""instructions text -> rulebook/entries/CLASS_NNN.json. One LLM call per class (section 5.5).

Two invariants, both load-bearing:
  1. Reads ONLY the instructions booklet. Never a form page, never an eval page (1.1).
  2. Its OUTPUT must not name the form. Eval pages may name themselves harmlessly, but
     a rulebook that names the form completes the class_id -> real form mapping and
     lets the model answer from pretrained memory instead of the rules (1.2).
"""

import json
import re

import numpy as np

import config
from extract import load_taxonomy
from llm import USAGE, get_llm

DOCUMENT_TYPES = ["tax_return", "tax_schedule", "information_return",
                  "certificate", "application", "worksheet"]

NAME_LEAK = re.compile(
    r"\b(?:1040|1065|1098|1099|1120|2106|2441|2555|3903|4562|4797|5695|"
    r"8582|8606|8825|8863|8889|8949|8995)\b"
    r"|\bW-[289]\b|\bSS-4\b|\bK-1\b"
    r"|\b[Ff]orm\s+(?:\d|[A-Z])"
    r"|\b[Ss]chedule\s+[A-Z]\b"
)

# "Line 12b. Section 179 expense deduction." -> captures the caption after the number.
# Line-anchored ($ with re.M): without it the capture runs past the caption into the
# prose that follows and gets cut mid-word, which is what produced 0% grounding.
CAPTION = re.compile(
    r"^ *(?:Lines?|Items?|Box(?:es)?|Parts?)\s+[\dA-Za-z][\w ,&-]{0,18}\.[ \t]*(.{4,70})$",
    re.M)
# Fallback for booklets with no "Line N." structure (f1099div mined 0 captions):
# a standalone short title-case line is almost always a printed heading.
HEADING = re.compile(r"^ *([A-Z][A-Za-z'()/-]*(?: [A-Za-z0-9'()/,.-]+){1,7})$", re.M)
# The fallback can re-match a "Line N." line whole; strip the numbering so it dedupes
# against the primary hits instead of appearing twice.
NUM_PREFIX = re.compile(r"^(?:Lines?|Items?|Box(?:es)?|Parts?)\s+[\dA-Za-z][\w ,&-]{0,18}\.\s*",
                        re.I)

_CLAUSE = {"type": "object",
           "properties": {"text": {"type": "string"}, "quote": {"type": "string"}},
           "required": ["text", "quote"], "additionalProperties": False}
_DISC = {"type": "object",
         "properties": {"vs_class_id": {"type": "string"},
                        "text": {"type": "string"}, "quote": {"type": "string"}},
         "required": ["vs_class_id", "text", "quote"], "additionalProperties": False}

SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string", "enum": DOCUMENT_TYPES},
        "definition": {"type": "string"},
        "includes": {"type": "array", "items": _CLAUSE},
        "excludes": {"type": "array", "items": _CLAUSE},
        "discriminators": {"type": "array", "items": _DISC},
        "aliases": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["document_type", "definition", "includes", "excludes",
                 "discriminators", "aliases"],
    "additionalProperties": False,
}

SYSTEM = f"""You write rulebook entries. A downstream classifier sees ONE page of a
BLANK PRINTED FORM and a set of these entries, and must decide which class the page
belongs to. It never sees examples - your entry is its only knowledge.

You are reading an INSTRUCTIONS BOOKLET. The classifier never sees this booklet; it
sees the form. Your job is to mine the booklet for what is PRINTED ON THE FORM.

You are given a FIELD CAPTIONS list extracted from the booklet's line-by-line section.
Those strings are the actual printed captions on the form. PREFER THEM for `quote`.

`quote` must be a string appearing verbatim ON THE BLANK FORM: a line caption, field
label, box title, column header, section heading. 3-10 words. Copy it exactly.

NEVER quote the booklet's explanatory prose. Sentences like "this return is used to
report..." or "the entity does not pay tax on..." are ABOUT the form and appear nowhere
ON it. A quote that cannot be found on the blank form is a dead rule.

ABSOLUTE RULE - no identifiers anywhere in your output, including quotes and aliases.
Never write a form number (1065, W-2, 1099-MISC, SS-4...), never "Form X" or
"Schedule X", and never the document's PROPER TITLE - the title line printed at the top
of the form is banned even though it appears there, because it hands the classifier the
answer instead of making it read the content. Prefer interior field labels over
anything in the page header.

`includes`: 2-4 clauses. Field labels or headings that MUST be present on this form.
    Their quotes are checked against this form, so they must be captions of THIS form.
`excludes`: 1-3 clauses. Content a confusable sibling has that this one does not. The
    quote here is the SIBLING's caption - the string whose presence rules this class out.
`discriminators`: one per sibling given. Reference the sibling by opaque id in
    `vs_class_id`; state the observable on-page difference. A rule that does not
    discriminate is worthless.
`aliases`: alternate NON-IDENTIFYING wordings only. Usually empty.

Worked example of the naming rule, for a discriminator against a sibling:

  BAD  "text": "Partnership return allocates to partners via Schedule K-1; Schedule C
                is for sole proprietor business income"
  GOOD "text": "Allocates income among several partners, each receiving a separate
                statement of their share; the sibling reports the business income of
                one self-employed individual"

The same rule governs `definition`: describe what the page contains, never what it is
called.

document_type must be one of: {', '.join(DOCUMENT_TYPES)}"""


STOP = {"of", "the", "and", "for", "to", "a", "an", "or", "on", "in",
        "from", "with", "by", "at", "as", "your", "this", "you"}


def _norm(s: str) -> str:
    s = (s.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2013", "-").replace("\u2014", "-"))
    return re.sub(r"\s+", " ", s).strip().lower()


def _tokens(s: str) -> list:
    """Content tokens for fuzzy grounding. The booklet's rendering of a caption is
    rarely byte-identical to the form's printed one - it adds parentheticals ("(EIN)"),
    reorders ("Partner's Capital Account Analysis" vs "Analysis of Partners' Capital
    Accounts") and changes plurals. We cannot snap quotes to the form text to fix that:
    that would be writing rules from the eval document (1.1). So we state a matcher
    instead."""
    s = re.sub(r"\([^)]*\)", " ", s)          # drop parentheticals
    out = []
    for t in re.findall(r"[a-z0-9]+", _norm(s)):
        if t in STOP or len(t) < 3:
            continue
        if len(t) > 3 and t.endswith("s"):    # crude singularisation
            t = t[:-1]
        out.append(t)
    return out


MAX_QUOTE_TOKENS = 12   # a printed caption; the prompt asks for 3-10 words
MIN_QUOTE_TOKENS = 2


def is_grounded(quote: str, form_tokens: set, thresh: float = 0.8) -> bool:
    """A quote grounds if it is caption-shaped AND its content tokens are on the form.

    The length bound is not cosmetic. A 20-token sentence of generic tax vocabulary
    ("Use Part I to combine the net income and net loss from all passive activities")
    clears an 80% overlap threshold by chance against almost any tax form, which
    silently inflated the metric. Capping length removes that false-positive channel.
    """
    qt = _tokens(NUM_PREFIX.sub("", quote.strip()))   # "Box 1." is a booklet
    if not (MIN_QUOTE_TOKENS <= len(qt) <= MAX_QUOTE_TOKENS):   # convention, not print
        return False
    return sum(1 for t in qt if t in form_tokens) / len(qt) >= thresh


def quote_too_long(quote: str) -> bool:
    return len(_tokens(NUM_PREFIX.sub("", quote.strip()))) > MAX_QUOTE_TOKENS


# A printed caption is a noun phrase. Booklet headings that open with an imperative
# are instructions about the form, not text on it - f8889 produced "Report
# contributions to your HSA" and "Figure your HSA deduction", grounding 0/3.
IMPERATIVE = re.compile(
    r"^(?:Report|Figure|Use|Enter|Complete|Attach|See|Do|Don't|If|When|Include|Check|"
    r"Write|Add|Subtract|Multiply|Divide|Note|Caution|Example|You|We|Your|Where|How|"
    r"What|Who|Why|Purpose|General|Specific|Instructions?)\b", re.I)


def mine_captions(text: str, budget: int = 7000) -> str:
    """Printed field captions from the booklet's line-by-line section."""
    cands = [m.group(1) for m in CAPTION.finditer(text)]
    if len(cands) < 8:                       # booklet has no "Line N." structure
        cands += [m.group(1) for m in HEADING.finditer(text)]
    seen, out, used = set(), [], 0
    for c in cands:
        cap = NUM_PREFIX.sub("", re.sub(r"\s+", " ", c)).strip(" .,;:")
        key = cap.lower()
        if not (5 <= len(cap) <= 70) or key in seen or IMPERATIVE.match(cap):
            continue
        seen.add(key)
        if used + len(cap) + 3 > budget:
            break
        out.append(cap)
        used += len(cap) + 3
    return "\n".join(f"- {c}" for c in out)


def stratified(s: str, n: int, windows: int = 8) -> str:
    """Evenly spaced windows across the whole booklet - head+tail misses the middle,
    which is exactly where the line-by-line instructions live."""
    if len(s) <= n:
        return s
    w = n // windows
    step = (len(s) - w) / (windows - 1)
    return "\n\n[...]\n\n".join(s[int(i * step): int(i * step) + w]
                                for i in range(windows))


def load_instruction_text() -> dict:
    by_class = {}
    for line in config.PATHS.pages_jsonl.open():
        r = json.loads(line)
        if r["kind"] == "instructions" and not r["image_only"]:
            by_class.setdefault(r["class_id"], []).append(r)
    return {c: " ".join(p["text"] for p in sorted(v, key=lambda x: x["page_id"]))
            for c, v in by_class.items()}


def load_form_text() -> dict:
    """class_id -> (normalised text, content-token set). Used ONLY to audit quotes
    after the fact - never fed to the model (1.1)."""
    out = {}
    for line in config.PATHS.pages_jsonl.open():
        r = json.loads(line)
        if r["kind"] == "form" and not r["image_only"]:
            out.setdefault(r["class_id"], []).append(_norm(r["text"]))
    return {c: (" ||| ".join(v), set(_tokens(" ".join(v)))) for c, v in out.items()}


def nearest_siblings(texts: dict, k: int) -> dict:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBED_MODEL)
    ids = sorted(texts)
    vecs = np.asarray(model.encode([texts[i][:2000] for i in ids],
                                   normalize_embeddings=True))
    sim = vecs @ vecs.T
    np.fill_diagonal(sim, -1.0)
    return {cid: [ids[j] for j in np.argsort(-sim[i])[:k]] for i, cid in enumerate(ids)}


def title_of(real_name: str) -> str:
    """'Form 1065 - U.S. Return of Partnership Income' -> the title half."""
    return _norm(real_name.split(" - ", 1)[-1])


def is_leaky(s: str, title: str) -> bool:
    return bool(NAME_LEAK.search(s)) or (len(title) > 12 and title in _norm(s))


def audit(entry: dict, form_norm: str, form_tokens: set, title: str) -> dict:
    """Grounding + leak audit.

    Only `includes` must ground on the class's OWN form. `excludes`/`discriminators`
    legitimately quote a SIBLING's caption (5b's own example does exactly that).

    Leaks are checked on `text`/`definition`/`aliases` only, NOT on `quote`. A quote
    naming a sibling ("Profit or Loss From Business") says "if you see this, it is not
    this class" without revealing which class it IS - no mapping is exposed. A
    discriminator's `text` naming a form is different: `vs_class_id` sits right beside
    it, so the prose hands over that sibling's class_id -> form mapping outright.
    """
    res = {"inc_exact": 0, "inc_fuzzy": 0, "inc_total": 0, "inc_toolong": 0,
           "oth_fuzzy": 0, "oth_total": 0, "leaks": []}
    for field in ("includes", "excludes", "discriminators"):
        for c in entry.get(field, []):
            q = _norm(c["quote"])
            exact = bool(q) and q in form_norm
            # An exact substring of the form IS on the form - the length bound guards
            # the fuzzy path against generic-vocabulary false positives, and must not
            # override direct proof.
            fuzzy = exact or is_grounded(c["quote"], form_tokens)
            if field == "includes":
                res["inc_total"] += 1
                res["inc_exact"] += exact
                res["inc_fuzzy"] += fuzzy
                res["inc_toolong"] += quote_too_long(c["quote"])
            else:
                res["oth_total"] += 1
                res["oth_fuzzy"] += fuzzy
            if is_leaky(c["text"], title):
                res["leaks"].append(c["rule_id"])
    if is_leaky(entry.get("definition", ""), title):
        res["leaks"].append(f"{entry['class_id']}.definition")
    for a in entry.get("aliases", []):
        if is_leaky(a, title):
            res["leaks"].append(f"{entry['class_id']}.alias")
    return res


REPAIR_SCHEMA = {
    "type": "object",
    "properties": {"fixes": {"type": "array", "items": {
        "type": "object",
        "properties": {"rule_id": {"type": "string"}, "text": {"type": "string"}},
        "required": ["rule_id", "text"], "additionalProperties": False}}},
    "required": ["fixes"], "additionalProperties": False,
}

REPAIR_SYSTEM = """You rewrite clause texts to remove document identifiers.

Return one fix per clause given. Keep the meaning and the level of detail exactly;
change only the naming. Remove every form number, every "Form X" / "Schedule X", and
every proper document title. Refer to another class only by its opaque CLASS_NNN id,
or describe it ("the sibling", "a return filed by one individual").

Example:
  IN   "Partnership return allocates via Schedule K-1; Schedule C is for sole
        proprietor business income"
  OUT  "Allocates income among several partners, each receiving a separate statement
        of their share; the sibling reports the business income of one self-employed
        individual\""""


# Last-resort deterministic replacement. Ordered: the "Form N"/"Schedule X" patterns
# run first so they consume the number, otherwise the bare-number rule fires again on
# the same span and yields "that document that document".
SCRUBS = [
    (re.compile(r"\b[Ff]orms?\s+\d[\w./-]*"), "that document"),
    (re.compile(r"\b[Ff]orms?\s+[A-Z]{1,2}-?\d?\b"), "that document"),
    (re.compile(r"\b[Ss]chedules?\s+[A-Z](?:-\d)?\b"), "that schedule"),
    (re.compile(r"\bW-8BEN\b|\bW-[289]\b"), "that certificate"),
    (re.compile(r"\bSS-4\b"), "that application"),
    (re.compile(r"\bK-1\b"), "that per-owner statement"),
    (re.compile(r"\b(?:1040|1065|1098|1099|1120|2106|2441|2555|3903|4562|4797|"
                r"5695|8582|8606|8825|8863|8889|8949|8995)\b"), "that document"),
]


def scrub(s: str, title: str) -> str:
    """Guarantee the invariant when the repair call will not. Readability suffers a
    little; a contaminated anon run would cost the whole headline number."""
    for pat, repl in SCRUBS:
        s = pat.sub(repl, s)
    if len(title) > 12:
        # Absorb a preceding article so we get "this document", not "the this document".
        s = re.compile(r"\b(?:the|a|an)\s+" + re.escape(title), re.I).sub(
            "this document", s)
        s = re.compile(re.escape(title), re.I).sub("this document", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def repair_leaks(entry: dict, leaks: list, client, title: str) -> tuple:
    """Strip identifiers the main prompt failed to suppress. Returns (repaired,
    scrubbed, aliases dropped). Prompting alone did not hold across three revisions,
    so the scrub backstop makes the invariant structural rather than aspirational."""
    class_id = entry["class_id"]
    def_id = f"{class_id}.definition"

    slots = {}
    for field in ("includes", "excludes", "discriminators"):
        for c in entry.get(field, []):
            slots[c["rule_id"]] = c

    # Aliases are optional (5b's example has none) and carry no rule, so a leaking
    # alias is dropped outright rather than rewritten.
    dropped = 0
    if entry.get("aliases"):
        kept = [a for a in entry["aliases"] if not is_leaky(a, title)]
        dropped = len(entry["aliases"]) - len(kept)
        entry["aliases"] = kept

    targets = []
    for rid in dict.fromkeys(leaks):
        if rid in slots:
            targets.append((rid, slots[rid]["text"]))
        elif rid == def_id:
            targets.append((rid, entry.get("definition", "")))

    repaired = 0
    if targets:
        listing = "\n".join(f'{rid}: "{txt}"' for rid, txt in targets)
        out = client.call(system=REPAIR_SYSTEM,
                          user=f"Rewrite these texts:\n\n{listing}",
                          schema=REPAIR_SCHEMA)
        for f in out.get("fixes", []):
            rid, new = f.get("rule_id", ""), f.get("text", "")
            if not new or is_leaky(new, title):
                continue
            if rid in slots:
                slots[rid]["text"] = new
                repaired += 1
            elif rid == def_id:
                entry["definition"] = new
                repaired += 1

    # Backstop: anything still leaking gets replaced deterministically.
    scrubbed = 0
    for c in slots.values():
        if is_leaky(c["text"], title):
            c["text"] = scrub(c["text"], title)
            scrubbed += 1
    if is_leaky(entry.get("definition", ""), title):
        entry["definition"] = scrub(entry["definition"], title)
        scrubbed += 1

    return repaired, scrubbed, dropped


def cmd_build_rules(args):
    tax = {c["class_id"]: c for c in load_taxonomy()}
    texts = load_instruction_text()
    form_text = load_form_text()

    source_doc = {}
    for line in config.PATHS.pages_jsonl.open():
        r = json.loads(line)
        if r["kind"] == "instructions":
            source_doc[r["class_id"]] = r["source_doc"]

    print("  computing nearest siblings (local embeddings, free)...")
    siblings = nearest_siblings(texts, config.N_SIBLINGS)

    client = get_llm(use_stub=args.stub_llm)
    config.PATHS.rulebook_dir.mkdir(parents=True, exist_ok=True)

    targets = sorted(texts)[:args.limit]
    n_clauses = n_dropped = n_repaired = n_scrubbed = n_alias_dropped = 0
    T = {"inc_exact": 0, "inc_fuzzy": 0, "inc_total": 0, "inc_toolong": 0,
         "oth_fuzzy": 0, "oth_total": 0}
    all_leaks = []

    for class_id in targets:
        sib_lines = "\n".join(
            f"  {s} = {tax[s]['real_name']}" for s in siblings[class_id])
        captions = mine_captions(texts[class_id])
        user = (
            f"Write the rulebook entry for {class_id}.\n\n"
            f"Its {config.N_SIBLINGS} nearest siblings (write one discriminator per "
            f"sibling; their names are given so you know what you are separating from, "
            f"but never write a name into your output):\n{sib_lines}\n\n"
            f"--- FIELD CAPTIONS PRINTED ON THIS FORM (prefer these for quotes) ---\n"
            f"{captions}\n\n"
            f"--- INSTRUCTIONS BOOKLET FOR {class_id} (sampled across the document) ---\n"
            f"{stratified(texts[class_id], config.MAX_INSTRUCTION_CHARS)}"
        )
        if args.dry_run:
            print(f"  {class_id}  {len(captions.splitlines()):3} captions, "
                  f"{len(user):,} chars")
            continue

        entry = client.call(system=SYSTEM, user=user, schema=SCHEMA)
        entry["class_id"] = class_id
        entry["source_doc"] = source_doc.get(class_id, "")

        for field, tag in (("includes", "inc"), ("excludes", "exc"),
                           ("discriminators", "disc")):
            kept, n = [], 0
            for c in entry.get(field, []):
                if not c.get("quote", "").strip():
                    n_dropped += 1
                    continue
                n += 1
                c["rule_id"] = f"{class_id}.{tag}.{n}"
                kept.append(c)
            entry[field] = kept
            n_clauses += len(kept)

        title = title_of(tax[class_id]["real_name"])
        fnorm, ftok = form_text.get(class_id, ("", set()))

        a = audit(entry, fnorm, ftok, title)
        if a["leaks"] and not args.no_repair:
            rep, scr, drp = repair_leaks(entry, a["leaks"], client, title)
            n_repaired += rep
            n_scrubbed += scr
            n_alias_dropped += drp
            a = audit(entry, fnorm, ftok, title)   # re-audit after repair

        (config.PATHS.rulebook_dir / f"{class_id}.json").write_text(
            json.dumps(entry, indent=2))

        for k in T:
            T[k] += a[k]
        all_leaks.extend(a["leaks"])
        print(f"  {class_id}  {len(entry['includes'])}inc "
              f"{len(entry['excludes'])}exc {len(entry['discriminators'])}disc"
              f"  | includes grounded {a['inc_fuzzy']}/{a['inc_total']}"
              f" (exact {a['inc_exact']})"
              f"{'  LEAKS: ' + ','.join(a['leaks']) if a['leaks'] else ''}")

    if args.dry_run:
        return 0

    pct = 100 * T["inc_fuzzy"] / T["inc_total"] if T["inc_total"] else 0
    exact_pct = 100 * T["inc_exact"] / T["inc_total"] if T["inc_total"] else 0
    print(f"\n  clauses kept {n_clauses}, dropped for missing quote {n_dropped}")
    print(f"  INCLUDES grounded on own form: {T['inc_fuzzy']}/{T['inc_total']} "
          f"({pct:.0f}%)  <- the metric")
    print(f"    of which exact substring matches: {T['inc_exact']} ({exact_pct:.0f}%)")
    print(f"    rejected as not caption-shaped (>{MAX_QUOTE_TOKENS} tokens): "
          f"{T['inc_toolong']}")
    print(f"  excludes/disc grounded on own form: {T['oth_fuzzy']}/{T['oth_total']} "
          f"(expected low - they quote siblings)")
    print(f"  identifier fixes: {n_repaired} rewritten by model, "
          f"{n_scrubbed} scrubbed deterministically, {n_alias_dropped} aliases dropped")
    print(f"  name leaks remaining: {len(all_leaks)}   (must be 0)")
    zero = [c for c in sorted(config.PATHS.rulebook_dir.glob('CLASS_*.json'))]
    print(f"  rulebook entries written: {len(zero)}")
    print(f"  {USAGE.summary()}")
    return 0


def build_subparser(sub):
    p = sub.add_parser("build-rules", help="instructions -> rulebook entries")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stub-llm", action="store_true")
    p.add_argument("--no-repair", action="store_true",
                   help="Skip the identifier-repair call (measures raw leak rate).")
    p.set_defaults(func=cmd_build_rules)

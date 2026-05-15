"""
SHARED DATASET — used across all coding practice phases.

Domain: Acme Financial Services (FICTIONAL — 100% synthetic, no real data).
Memory-safe: text only, deterministic via fixed seed.

Functions exposed:
    get_corpus(target_chars=50_000)         → long text string (Phase 1-2 language modeling)
    get_documents(n_extra=80)               → list of dicts (Phase 5 RAG, Phase 6 agents)
    get_classification_data(n_total=200)    → (text, label) pairs (Phase 1 BiLSTM, Phase 2 transformer)
    get_ner_data(n_total=100)               → (tokens, BIO tags) (Phase 1 BiLSTM tagger)
    get_qa_pairs()                          → curated (question, answer) for eval (Phase 5)
    get_instruction_pairs(n_total=100)      → (instruction, response) for fine-tuning (Phase 4)

Sizing knobs let each phase pick the right scale. Default values are CPU-friendly.
"""

import random


# ======================================================================
# Hand-curated core facts (FICTIONAL — used as ground truth for QA eval)
# ======================================================================
PRODUCTS = [
    {"name": "Premium Checking",   "category": "checking", "rate": 2.5, "fee": 0,  "min_balance": 5000},
    {"name": "Basic Checking",     "category": "checking", "rate": 0.5, "fee": 5,  "min_balance": 100},
    {"name": "High Yield Savings", "category": "savings",  "rate": 4.2, "fee": 0,  "min_balance": 1000},
    {"name": "Personal Loan",      "category": "loan",     "rate": 8.5, "max_amount": 50000,  "max_years": 7},
    {"name": "Home Mortgage",      "category": "loan",     "rate": 6.8, "max_amount": 500000, "max_years": 30},
    {"name": "Auto Loan",          "category": "loan",     "rate": 5.5, "max_amount": 40000,  "max_years": 7},
]

EMPLOYEES = [
    {"name": "Sarah Chen",    "role": "Head of Loans",           "dept": "Loans"},
    {"name": "Mike Johnson",  "role": "Customer Service Manager", "dept": "Service"},
    {"name": "Priya Sharma",  "role": "Risk Analyst",            "dept": "Risk"},
    {"name": "David Lee",     "role": "IT Director",             "dept": "IT"},
    {"name": "Anna Martinez", "role": "Compliance Officer",      "dept": "Compliance"},
]

POLICIES = [
    "All personal loans require credit score above 650.",
    "Home mortgages have a 30 day cooling off period.",
    "High Yield Savings accounts require minimum balance of 1000 dollars.",
    "Premium Checking accounts include free wire transfers.",
    "Auto loans require proof of valid insurance.",
    "Loan applications take 5 to 7 business days to process.",
    "Account closures must be requested in writing.",
    "Customer service is available from 9 am to 6 pm on weekdays.",
]


# ======================================================================
# Variety pools — extra synthetic entities for data expansion
# ======================================================================
EXTRA_PRODUCTS = [
    {"name": "Student Loan",      "category": "loan",    "rate": 4.5, "max_amount": 75000,  "max_years": 10},
    {"name": "Business Loan",     "category": "loan",    "rate": 7.2, "max_amount": 250000, "max_years": 15},
    {"name": "Credit Line",       "category": "loan",    "rate": 9.0, "max_amount": 25000,  "max_years": 5},
    {"name": "Money Market",      "category": "savings", "rate": 3.8, "fee": 0,  "min_balance": 2500},
    {"name": "Retirement Savings","category": "savings", "rate": 5.0, "fee": 0,  "min_balance": 500},
    {"name": "Travel Card",       "category": "checking","rate": 1.0, "fee": 10, "min_balance": 0},
    {"name": "Student Checking",  "category": "checking","rate": 0.8, "fee": 0,  "min_balance": 50},
]

EXTRA_EMPLOYEES = [
    {"name": "John Park",        "role": "Senior Analyst",    "dept": "Risk"},
    {"name": "Maria Garcia",     "role": "Branch Manager",    "dept": "Operations"},
    {"name": "Wei Chang",        "role": "Software Engineer", "dept": "IT"},
    {"name": "Aisha Khan",       "role": "Marketing Director","dept": "Marketing"},
    {"name": "Carlos Rivera",    "role": "Loan Officer",      "dept": "Loans"},
    {"name": "Emma Wilson",      "role": "Accountant",        "dept": "Finance"},
    {"name": "Hiroshi Tanaka",   "role": "Data Scientist",    "dept": "IT"},
    {"name": "Olivia Brown",     "role": "HR Manager",        "dept": "HR"},
    {"name": "Kenji Yamamoto",   "role": "Investment Advisor","dept": "Wealth"},
    {"name": "Fatima Al-Rashid", "role": "Compliance Analyst","dept": "Compliance"},
]


# ======================================================================
# Templates for generating sentence variations
# ======================================================================
LOAN_TEMPLATES = [
    "The {name} offers interest rate of {rate} percent. Maximum amount is {max_amount} dollars over {max_years} years.",
    "Apply for the {name} at {rate} percent annual rate for up to {max_amount} dollars.",
    "{name} provides loans up to {max_amount} dollars at {rate} percent over {max_years} years.",
    "Our {name} starts at {rate} percent interest with a maximum of {max_amount} dollars.",
    "The {name} has a rate of {rate} percent and a maximum loan size of {max_amount} dollars.",
]

ACCOUNT_TEMPLATES = [
    "The {name} account earns {rate} percent annual interest. Monthly fee is {fee} dollars and minimum balance is {min_balance} dollars.",
    "Open a {name} with {min_balance} dollars minimum at {rate} percent interest.",
    "Our {name} offers {rate} percent rate with a {fee} dollar monthly fee.",
    "The {name} pays {rate} percent interest and requires {min_balance} dollars minimum balance.",
    "{name} accounts feature {rate} percent yield and {fee} dollar monthly maintenance.",
]

EMPLOYEE_TEMPLATES = [
    "{name} works as {role} in the {dept} department.",
    "{name} serves as our {role} leading the {dept} team.",
    "Our {role} is {name}, based in the {dept} department.",
    "{role} {name} oversees operations in {dept}.",
    "{name} is the {role} responsible for {dept}.",
]


# ======================================================================
# Core helpers
# ======================================================================
def _all_products():
    return PRODUCTS + EXTRA_PRODUCTS


def _all_employees():
    return EMPLOYEES + EXTRA_EMPLOYEES


def _format_product(p: dict, template: str) -> str:
    if p["category"] == "loan":
        return template.format(name=p["name"], rate=p["rate"],
                               max_amount=p["max_amount"], max_years=p["max_years"])
    return template.format(name=p["name"], rate=p["rate"],
                           fee=p["fee"], min_balance=p["min_balance"])


# ======================================================================
# Documents (RAG / agent knowledge base)
# ======================================================================
def get_documents(n_extra: int = 80, seed: int = 42) -> list:
    """Curated docs + N synthetic variations. Default 80 extras → ~120 total docs."""
    rng = random.Random(seed)
    docs = []

    # Curated core (always first 19 docs)
    for i, p in enumerate(PRODUCTS):
        tmpl = LOAN_TEMPLATES[0] if p["category"] == "loan" else ACCOUNT_TEMPLATES[0]
        docs.append({"id": f"prod_{i}", "title": p["name"],
                     "content": _format_product(p, tmpl), "category": p["category"]})

    for i, e in enumerate(EMPLOYEES):
        docs.append({"id": f"emp_{i}", "title": e["name"],
                     "content": EMPLOYEE_TEMPLATES[0].format(**e),
                     "category": "employee"})

    for i, p in enumerate(POLICIES):
        docs.append({"id": f"pol_{i}", "title": f"Policy {i+1}",
                     "content": p, "category": "policy"})

    # Synthetic variations from extended pools + templates
    for i in range(n_extra):
        kind = rng.choice(["product", "employee"])
        if kind == "product":
            p = rng.choice(_all_products())
            tmpl = rng.choice(LOAN_TEMPLATES if p["category"] == "loan" else ACCOUNT_TEMPLATES)
            docs.append({"id": f"syn_prod_{i}", "title": p["name"],
                         "content": _format_product(p, tmpl), "category": p["category"]})
        else:
            e = rng.choice(_all_employees())
            tmpl = rng.choice(EMPLOYEE_TEMPLATES)
            docs.append({"id": f"syn_emp_{i}", "title": e["name"],
                         "content": tmpl.format(**e), "category": "employee"})

    return docs


# ======================================================================
# Corpus — long text string for char/word language modeling
# ======================================================================
def get_corpus(target_chars: int = 50_000, seed: int = 42) -> str:
    """Generate a corpus of approximately `target_chars` characters.
    Phase 1 RNN scratch: 50K is plenty (CPU-friendly).
    Phase 2 mini-GPT:   pass target_chars=200_000 or more."""
    rng = random.Random(seed)
    chunks = []
    total = 0
    while total < target_chars:
        roll = rng.random()
        if roll < 0.4:  # product
            p = rng.choice(_all_products())
            tmpl = rng.choice(LOAN_TEMPLATES if p["category"] == "loan" else ACCOUNT_TEMPLATES)
            s = _format_product(p, tmpl)
        elif roll < 0.7:  # employee
            e = rng.choice(_all_employees())
            s = rng.choice(EMPLOYEE_TEMPLATES).format(**e)
        else:  # policy
            s = rng.choice(POLICIES)
        chunks.append(s)
        total += len(s) + 1
    return " ".join(chunks)


# ======================================================================
# Classification (text → category)
# ======================================================================
def get_classification_data(n_total: int = 200, seed: int = 42) -> list:
    """Balanced (text, label) for sequence classification.
    Labels: 'checking', 'savings', 'loan', 'employee', 'policy'."""
    rng = random.Random(seed)
    data = []

    # All curated docs first
    for d in get_documents(n_extra=0, seed=seed):
        data.append((d["content"], d["category"]))

    # Top up with variations
    while len(data) < n_total:
        kind = rng.choice(["product", "employee", "policy"])
        if kind == "product":
            p = rng.choice(_all_products())
            tmpl = rng.choice(LOAN_TEMPLATES if p["category"] == "loan" else ACCOUNT_TEMPLATES)
            data.append((_format_product(p, tmpl), p["category"]))
        elif kind == "employee":
            e = rng.choice(_all_employees())
            data.append((rng.choice(EMPLOYEE_TEMPLATES).format(**e), "employee"))
        else:
            data.append((rng.choice(POLICIES), "policy"))

    rng.shuffle(data)
    return data[:n_total]


# ======================================================================
# NER (token-level BIO tagging)
# ======================================================================
def get_ner_data(n_total: int = 100, seed: int = 42) -> list:
    """(tokens, BIO tags) for sequence tagging.
    Entities: PERSON, PRODUCT, MONEY (raw numbers)."""
    rng = random.Random(seed)
    examples = []

    def tag_tokens(tokens: list, entities: dict) -> list:
        """entities maps {span_first_token: label}. Tag others O."""
        tags = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            matched = False
            for first, label in entities.items():
                span = first.split()
                if tokens[i: i + len(span)] == span:
                    tags.append(f"B-{label}")
                    for _ in span[1:]:
                        tags.append(f"I-{label}")
                    i += len(span)
                    matched = True
                    break
            if matched:
                continue
            if tok.isdigit():
                tags.append("B-MONEY")
            else:
                tags.append("O")
            i += 1
        return tags

    while len(examples) < n_total:
        kind = rng.choice(["employee", "product"])
        if kind == "employee":
            e = rng.choice(_all_employees())
            sent = rng.choice(EMPLOYEE_TEMPLATES).format(**e)
            tokens = sent.replace(".", " .").split()
            tags = tag_tokens(tokens, {e["name"]: "PERSON"})
        else:
            p = rng.choice(_all_products())
            tmpl = rng.choice(LOAN_TEMPLATES if p["category"] == "loan" else ACCOUNT_TEMPLATES)
            sent = _format_product(p, tmpl)
            tokens = sent.replace(".", " .").split()
            tags = tag_tokens(tokens, {p["name"]: "PRODUCT"})
        if len(tokens) == len(tags):
            examples.append((tokens, tags))

    return examples


# ======================================================================
# QA pairs (handcrafted ground truth — for eval)
# ======================================================================
def get_qa_pairs() -> list:
    """Curated (question, expected_answer_substring) for RAG eval."""
    return [
        ("What is the interest rate on Premium Checking?",        "2.5"),
        ("Who manages the Loans department?",                      "Sarah Chen"),
        ("What is the minimum credit score for personal loans?",   "650"),
        ("What is the maximum amount for a Home Mortgage?",        "500000"),
        ("How long does loan processing take?",                    "5 to 7"),
        ("Who is the IT Director?",                                "David Lee"),
        ("What rate does High Yield Savings offer?",               "4.2"),
        ("Is wire transfer free on Premium Checking?",             "free"),
        ("What is the rate on Auto Loan?",                         "5.5"),
        ("Who handles compliance?",                                "Anna Martinez"),
    ]


# ======================================================================
# Instruction pairs (Phase 4 fine-tuning)
# ======================================================================
def get_instruction_pairs(n_total: int = 100, seed: int = 42) -> list:
    """(instruction, response) pairs derived from facts. Format: {'instruction', 'response'}."""
    rng = random.Random(seed)
    pairs = []

    # Product-based instructions
    for p in _all_products():
        if p["category"] == "loan":
            pairs.append({
                "instruction": f"What is the interest rate on {p['name']}?",
                "response": f"The {p['name']} has an interest rate of {p['rate']} percent.",
            })
            pairs.append({
                "instruction": f"What is the maximum amount I can borrow with {p['name']}?",
                "response": f"You can borrow up to {p['max_amount']} dollars with {p['name']} over {p['max_years']} years.",
            })
        else:
            pairs.append({
                "instruction": f"What is the interest rate on {p['name']}?",
                "response": f"The {p['name']} earns {p['rate']} percent annual interest.",
            })
            pairs.append({
                "instruction": f"What is the minimum balance for {p['name']}?",
                "response": f"The {p['name']} requires a minimum balance of {p['min_balance']} dollars.",
            })

    # Employee-based instructions
    for e in _all_employees():
        pairs.append({
            "instruction": f"Who is {e['name']}?",
            "response": f"{e['name']} is our {e['role']} in the {e['dept']} department.",
        })
        pairs.append({
            "instruction": f"Who works in the {e['dept']} department?",
            "response": f"{e['name']} works in {e['dept']} as {e['role']}.",
        })

    rng.shuffle(pairs)
    return pairs[:n_total]


# ======================================================================
# Backward-compat helpers (used by Phase 1 char-LM sessions)
# ======================================================================
def get_corpus_as_string(target_chars: int = 50_000, seed: int = 42) -> str:
    return get_corpus(target_chars=target_chars, seed=seed)


def get_char_vocab(target_chars: int = 50_000, seed: int = 42):
    """Build character vocabulary from corpus. Returns (char2idx, idx2char, vocab_size)."""
    text = get_corpus(target_chars=target_chars, seed=seed)
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(chars)


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SHARED DATASET — Acme Financial Services (synthetic)")
    print("=" * 60)

    corpus = get_corpus()
    print(f"\nCorpus (default 50K): {len(corpus):,} chars · {len(set(corpus))} unique chars")
    print(f"  First 200 chars: {corpus[:200]}...")

    docs = get_documents()
    from collections import Counter
    print(f"\nDocuments (default ~120): {len(docs)}")
    print(f"  By category: {dict(Counter(d['category'] for d in docs))}")

    classif = get_classification_data()
    print(f"\nClassification examples: {len(classif)}")
    print(f"  Label balance: {dict(Counter(l for _, l in classif))}")

    ner = get_ner_data()
    print(f"\nNER examples: {len(ner)}")
    toks, tags = ner[0]
    print(f"  Sample tokens: {toks}")
    print(f"  Sample tags:   {tags}")

    qa = get_qa_pairs()
    print(f"\nQA pairs (curated): {len(qa)}")

    pairs = get_instruction_pairs()
    print(f"\nInstruction pairs: {len(pairs)}")
    print(f"  Sample: {pairs[0]}")

    print("\nAll functions accept size knobs:")
    print(f"  get_corpus(target_chars=200_000) → {len(get_corpus(200_000)):,} chars")
    print(f"  get_documents(n_extra=200)       → {len(get_documents(200))} docs")

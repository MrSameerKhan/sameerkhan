# Session 2 — Structured Extraction with Pydantic + Instructor
# Task   : parse unstructured invoice/contract text -> validated Python objects
# Shows  : Pydantic schema, instructor retry-on-validation-fail, nested models
#
# Change PROVIDER to switch backends. Nothing else needs to change.
#   "openai"  → needs OPENAI_API_KEY env var
#   "claude"  → needs ANTHROPIC_API_KEY env var
#   "ollama"  → needs Ollama running locally (ollama serve)

import os
import json
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import instructor

PROVIDER   = "openai"   # "openai" | "claude" | "ollama"
MAX_TOKENS = 1024

if PROVIDER == "openai":
    from openai import OpenAI
    client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
    MODEL  = "gpt-4o-mini"

elif PROVIDER == "claude":
    import anthropic
    client = instructor.from_anthropic(anthropic.Anthropic())
    MODEL  = "claude-haiku-4-5-20251001"

elif PROVIDER == "ollama":
    from openai import OpenAI
    client = instructor.from_openai(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))
    MODEL  = "llama3.2"


def _extract(response_model, system: str, content: str):
    """Single call wrapper — handles Claude vs OpenAI/Ollama message format."""
    if PROVIDER == "claude":
        return client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            response_model=response_model, max_retries=3,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
    else:
        return client.chat.completions.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            response_model=response_model, max_retries=3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": content},
            ],
        )


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class LineItem(BaseModel):
    description: str
    quantity:    float
    unit_price:  float
    total:       float

    @field_validator("total")
    @classmethod
    def total_must_match(cls, v, info):
        expected = round(info.data.get("quantity", 0) * info.data.get("unit_price", 0), 2)
        if abs(v - expected) > 0.02:
            raise ValueError(f"total {v} does not match quantity x unit_price = {expected}")
        return v


class Invoice(BaseModel):
    invoice_number:  str
    vendor_name:     str
    customer_name:   str
    invoice_date:    str
    due_date:        Optional[str] = None
    currency:        str = Field(default="USD")
    line_items:      list[LineItem]
    subtotal:        float
    tax_rate_pct:    Optional[float] = None
    tax_amount:      Optional[float] = None
    total_amount:    float
    payment_terms:   Optional[str] = None


class ContractClause(BaseModel):
    clause_type:  str
    summary:      str
    key_dates:    list[str]
    key_amounts:  list[str]
    obligations:  list[str]


class Contract(BaseModel):
    parties:        list[str]
    effective_date: str
    expiry_date:    Optional[str] = None
    contract_type:  str
    governing_law:  Optional[str] = None
    clauses:        list[ContractClause]
    risk_flags:     list[str]


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_invoice(raw_text: str) -> Invoice:
    return _extract(
        Invoice,
        system=(
            "Extract all invoice fields from the text. "
            "Compute missing totals from quantity x unit_price. "
            "If a field is missing, use null."
        ),
        content=raw_text,
    )


def extract_contract(raw_text: str) -> Contract:
    return _extract(
        Contract,
        system=(
            "Extract structured data from this contract. "
            "Identify risk flags: unusual indemnification, unlimited liability, "
            "auto-renewal without notice, non-standard governing law."
        ),
        content=raw_text,
    )


# ── Test documents ─────────────────────────────────────────────────────────────

SAMPLE_INVOICE = """
INVOICE

Invoice Number: INV-2024-00847
Date: November 15, 2024
Due Date: December 15, 2024
Payment Terms: Net 30

From:
DataSync Solutions Ltd.
VAT Reg: GB 123 456 789

To:
Al Rajhi Capital
Riyadh, Saudi Arabia

Services Rendered:
  API Integration Services     4 days    @ $1,200/day     $4,800.00
  Data Migration Consulting    2 days    @ $950/day        $1,900.00
  Training Sessions (remote)   3 hours   @ $350/hour       $1,050.00

Subtotal: $7,750.00
VAT (15%): $1,162.50
TOTAL DUE: $8,912.50

Currency: USD
"""

SAMPLE_CONTRACT = """
MASTER SERVICE AGREEMENT

This Master Service Agreement ("Agreement") is entered into as of January 1, 2025,
between Acme Data Corp ("Vendor") and FintechBank SA ("Client").

PAYMENT TERMS: Client shall pay all invoices within 30 days of receipt.
Late payments accrue interest at 1.5% per month. Annual contract value: $120,000.

TERMINATION: Either party may terminate with 90 days written notice.
Vendor may terminate immediately upon Client's material breach if not cured within 14 days.

LIABILITY: Vendor's total liability under this Agreement shall not exceed the fees
paid in the 12 months preceding the claim. HOWEVER, this limitation does not apply
to gross negligence or wilful misconduct, for which liability is unlimited.

CONFIDENTIALITY: Both parties agree to maintain strict confidentiality for 5 years
following termination. Violations subject to $500,000 liquidated damages.

AUTO-RENEWAL: This Agreement automatically renews annually unless either party
provides 60 days written notice before the renewal date.

GOVERNING LAW: This Agreement is governed by the laws of the Cayman Islands.

IP OWNERSHIP: All work product created under this Agreement is owned by Vendor
unless explicitly assigned in a separate written instrument.
"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Provider: {PROVIDER} | Model: {MODEL}\n")

    print("-- Invoice Extraction --\n")
    invoice = extract_invoice(SAMPLE_INVOICE)
    print(invoice.model_dump_json(indent=2))

    print("\n-- Validation check --")
    for item in invoice.line_items:
        expected = round(item.quantity * item.unit_price, 2)
        status   = "OK" if abs(item.total - expected) < 0.02 else "MISMATCH"
        print(f"  {item.description}: {item.quantity} x {item.unit_price} = {expected} | extracted: {item.total} [{status}]")
    print(f"  Grand total: {invoice.total_amount}")

    print("\n\n-- Contract Extraction --\n")
    contract = extract_contract(SAMPLE_CONTRACT)
    print(json.dumps(contract.model_dump(), indent=2))

    print("\n-- Risk Flags --")
    for flag in contract.risk_flags:
        print(f"  [!] {flag}")


if __name__ == "__main__":
    main()

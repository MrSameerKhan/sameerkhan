# Session 2 — Structured Extraction with Pydantic + Instructor
# Task   : parse unstructured invoice/contract text → validated Python objects
# Model  : gpt-4o-mini via Instructor (wraps OpenAI function calling)
# Shows  : Pydantic schema, instructor retry-on-validation-fail, nested models
#
# Instructor forces the LLM to return valid JSON matching your Pydantic model.
# If the output fails validation, Instructor automatically retries with the
# validation error injected back into the prompt (up to max_retries).
#
# Install: pip install instructor (already in environment.yml)
# Set OPENAI_API_KEY before running.

import os
import json
from datetime import date
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import instructor
from openai import OpenAI

raw_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
client     = instructor.from_openai(raw_client)   # patches OpenAI client

MODEL = "gpt-4o-mini"


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
        if abs(v - expected) > 0.02:   # allow small float rounding
            raise ValueError(f"total {v} does not match quantity × unit_price = {expected}")
        return v


class Invoice(BaseModel):
    invoice_number:  str
    vendor_name:     str
    customer_name:   str
    invoice_date:    str                    # keep as string — dates vary in format
    due_date:        Optional[str] = None
    currency:        str = Field(default="USD")
    line_items:      list[LineItem]
    subtotal:        float
    tax_rate_pct:    Optional[float] = None
    tax_amount:      Optional[float] = None
    total_amount:    float
    payment_terms:   Optional[str] = None


class ContractClause(BaseModel):
    clause_type:     str   # e.g. "payment", "termination", "liability", "confidentiality"
    summary:         str   # 1-2 sentence plain English summary
    key_dates:       list[str]
    key_amounts:     list[str]
    obligations:     list[str]   # who must do what


class Contract(BaseModel):
    parties:         list[str]
    effective_date:  str
    expiry_date:     Optional[str] = None
    contract_type:   str
    governing_law:   Optional[str] = None
    clauses:         list[ContractClause]
    risk_flags:      list[str]   # unusual or high-risk terms for human review


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_invoice(raw_text: str) -> Invoice:
    return client.chat.completions.create(
        model=MODEL,
        response_model=Invoice,
        max_retries=3,   # retry if Pydantic validation fails
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract all invoice fields from the text. "
                    "Compute missing totals from quantity × unit_price. "
                    "If a field is missing, use null."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
    )


def extract_contract(raw_text: str) -> Contract:
    return client.chat.completions.create(
        model=MODEL,
        response_model=Contract,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract structured data from this contract. "
                    "Identify risk flags: unusual indemnification, unlimited liability, "
                    "auto-renewal without notice, non-standard governing law."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
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
    print("── Invoice Extraction ──\n")
    invoice = extract_invoice(SAMPLE_INVOICE)
    print(invoice.model_dump_json(indent=2))

    print("\n── Validation check ──")
    for item in invoice.line_items:
        expected = round(item.quantity * item.unit_price, 2)
        status   = "✓" if abs(item.total - expected) < 0.02 else "✗ MISMATCH"
        print(f"  {item.description}: {item.quantity} × {item.unit_price} = {expected} | extracted: {item.total} {status}")
    print(f"  Grand total: {invoice.total_amount}")

    print("\n\n── Contract Extraction ──\n")
    contract = extract_contract(SAMPLE_CONTRACT)
    print(json.dumps(contract.model_dump(), indent=2))

    print("\n── Risk Flags ──")
    for flag in contract.risk_flags:
        print(f"  ⚠  {flag}")


if __name__ == "__main__":
    main()

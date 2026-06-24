# Session 2 — Structured Extraction with Pydantic + Instructor
Status: `✅ Run`

Theory: [../../../4.nlp/04_applications/03_information_extraction.md](../../../4.nlp/04_applications/03_information_extraction.md)

---

## Use Case

Invoice and contract parser: raw PDF text → validated Python objects. Every Document AI pipeline needs reliable structured extraction — Instructor makes it bulletproof by enforcing the schema and auto-retrying on validation failure.

---

## Key Concept — How Instructor Works

```
Your Pydantic model (schema)
    │
    instructor.from_openai(client)
    │
    LLM generates JSON matching schema
    │
    Pydantic validates the JSON
    │
    Pass → return typed Python object ✓
    Fail → inject validation error into prompt → retry (up to max_retries)
```

Without Instructor: JSON parsing fails silently or throws unhandled exceptions.
With Instructor: you always get a typed, validated Python object or a clean error after N retries.

---

## Pydantic Schema Design

```python
class LineItem(BaseModel):
    description: str
    quantity:    float
    unit_price:  float
    total:       float

    @field_validator("total")
    @classmethod
    def total_must_match(cls, v, info):
        expected = round(info.data["quantity"] * info.data["unit_price"], 2)
        if abs(v - expected) > 0.02:
            raise ValueError(f"total {v} != quantity × unit_price = {expected}")
        return v
```

The `@field_validator` runs after the LLM fills in all fields. If `total` doesn't match `quantity × unit_price`, Pydantic raises a `ValueError`. Instructor catches it, injects the error message into the prompt, and asks the LLM to fix it.

---

## Nested Models

```
Invoice
├── invoice_number: str
├── vendor_name: str
├── line_items: list[LineItem]   ← nested list of Pydantic models
│       ├── description: str
│       ├── quantity: float
│       ├── unit_price: float
│       └── total: float         ← validated against quantity × unit_price
├── subtotal: float
└── total_amount: float

Contract
├── parties: list[str]
├── clauses: list[ContractClause]   ← nested
│       ├── clause_type: str
│       ├── summary: str
│       ├── key_dates: list[str]
│       ├── key_amounts: list[str]
│       └── obligations: list[str]
└── risk_flags: list[str]           ← LLM identifies unusual terms
```

---

## Risk Flag Extraction

The contract extractor asks the LLM to identify high-risk clauses. From `SAMPLE_CONTRACT`:

```
Expected risk_flags:
  ⚠  Unlimited liability for gross negligence/wilful misconduct
  ⚠  Auto-renewal with only 60 days notice window
  ⚠  Governing law: Cayman Islands (non-standard for Saudi entity)
  ⚠  IP ownership defaults to Vendor, not Client
  ⚠  $500,000 liquidated damages for confidentiality breach
```

These flags surface in human review before contract execution — exactly what a Document AI system at a bank needs.

---

## Actual Output (Windows, gpt-4o-mini, 2026-06-25)

All 3 line items validated OK (quantity × unit_price matched exactly). Contract extracted 7 clauses. Risk flags correctly identified: unlimited liability, auto-renewal without notice, non-standard governing law (Cayman Islands). Invoice grand total $8,912.50 correct.

---

## Expected Output

```
── Invoice Extraction ──

{
  "invoice_number": "INV-2024-00847",
  "vendor_name": "DataSync Solutions Ltd.",
  "customer_name": "Al Rajhi Capital",
  "invoice_date": "November 15, 2024",
  "due_date": "December 15, 2024",
  "currency": "USD",
  "line_items": [
    {"description": "API Integration Services", "quantity": 4, "unit_price": 1200, "total": 4800},
    {"description": "Data Migration Consulting", "quantity": 2, "unit_price": 950, "total": 1900},
    {"description": "Training Sessions (remote)", "quantity": 3, "unit_price": 350, "total": 1050}
  ],
  "subtotal": 7750.0,
  "tax_rate_pct": 15.0,
  "tax_amount": 1162.5,
  "total_amount": 8912.5,
  "payment_terms": "Net 30"
}

── Validation check ──
  API Integration Services: 4 × 1200 = 4800 | extracted: 4800 ✓
  Data Migration Consulting: 2 × 950 = 1900 | extracted: 1900 ✓
  Training Sessions (remote): 3 × 350 = 1050 | extracted: 1050 ✓
  Grand total: 8912.5

── Risk Flags ──
  ⚠  Unlimited liability for gross negligence or wilful misconduct
  ⚠  Auto-renewal clause with only 60 days cancellation notice
  ⚠  Governing law: Cayman Islands — non-standard for Saudi entities
  ⚠  IP ownership defaults to Vendor without explicit assignment
  ⚠  $500,000 liquidated damages for confidentiality breach
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
pip install instructor   # if not already installed
python 02_structured_extraction.py
```

Cost: ~$0.02–0.04 per run (2 API calls, long context).

**Production pattern:** wrap in a FastAPI endpoint:
```python
@app.post("/extract/invoice", response_model=Invoice)
async def extract(file: UploadFile):
    text = await parse_pdf(file)   # PyMuPDF / pdfplumber
    return extract_invoice(text)
```

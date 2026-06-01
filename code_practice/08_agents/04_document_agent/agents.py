# agents.py — Specialist agent functions for the mortgage document processing pipeline
#
# Each function is a pure specialist — takes state, does one job, returns updates.
# The supervisor in graph.py decides the routing order.

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL  = "gpt-4o-mini"


# ── Shared LLM helper ─────────────────────────────────────────────────────────
def _llm(system: str, user: str, json_mode: bool = False) -> str:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    res = client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        **kwargs,
    )
    return res.choices[0].message.content.strip()


# ── Policy store (replaces vector DB for this demo) ───────────────────────────
POLICY = {
    "ltv":           "Max LTV: 90% standard, 95% first-time buyers. Buy-to-let: 75%.",
    "income":        "Required docs: payslips (3 months), SA302 (self-employed, 2 yrs), pension statements.",
    "affordability": "Total monthly debts ≤ 45% of gross income. Stress rate: product rate + 3%.",
    "deposit":       "Minimum deposit: 10% standard, 5% first-time buyer (Help to Buy).",
    "erc":           "ERC: 5%/4%/3%/2%/1% for years 1-5. None after fixed period.",
}

def retrieve_policy(topics: list[str]) -> str:
    relevant = {k: v for k, v in POLICY.items() if any(t in k for t in topics)}
    if not relevant:
        relevant = POLICY
    return "\n".join(f"• {k}: {v}" for k, v in relevant.items())


# ── Specialist agents ─────────────────────────────────────────────────────────

def classify_document(state: dict) -> dict:
    """Classify what type of document was submitted."""
    doc = state["document_text"]
    result = _llm(
        system=(
            "Classify this bank document. Return JSON with:\n"
            '{"doc_type": str, "confidence": float, "key_indicators": [str]}\n'
            "doc_type options: mortgage_application, income_proof, property_valuation, "
            "bank_statement, identity_document, other"
        ),
        user=f"Document text:\n{doc}",
        json_mode=True,
    )
    data = json.loads(result)
    return {
        "doc_type":          data["doc_type"],
        "classification":    data,
        "steps_completed":   state.get("steps_completed", []) + ["classify"],
    }


def extract_fields(state: dict) -> dict:
    """Extract structured fields from the document."""
    doc      = state["document_text"]
    doc_type = state.get("doc_type", "unknown")

    result = _llm(
        system=(
            f"Extract all relevant fields from this {doc_type}. Return JSON with ALL applicable fields:\n"
            '{"applicant_name": str|null, "applicant_id": str|null, '
            '"property_address": str|null, "property_value": float|null, '
            '"loan_amount": float|null, "monthly_income": float|null, '
            '"employment_status": str|null, "monthly_debts": float|null, '
            '"deposit_amount": float|null, "is_first_time_buyer": bool|null, '
            '"loan_term_months": int|null, "currency": str}'
        ),
        user=f"Document:\n{doc}",
        json_mode=True,
    )
    data = json.loads(result)
    return {
        "extracted_fields": data,
        "steps_completed":  state.get("steps_completed", []) + ["extract"],
    }


def retrieve_policy_agent(state: dict) -> dict:
    """Retrieve relevant policy sections based on doc type and extracted fields."""
    doc_type = state.get("doc_type", "")
    fields   = state.get("extracted_fields", {})

    topics = ["ltv", "income", "affordability", "deposit"]
    if "mortgage" in doc_type:
        topics = ["ltv", "income", "affordability", "deposit", "erc"]

    policy_text = retrieve_policy(topics)
    return {
        "policy_context": policy_text,
        "steps_completed": state.get("steps_completed", []) + ["policy_retrieval"],
    }


def check_eligibility(state: dict) -> dict:
    """Compare extracted fields against retrieved policy — make a preliminary decision."""
    fields  = state.get("extracted_fields", {})
    policy  = state.get("policy_context", "")
    doc_type = state.get("doc_type", "")

    result = _llm(
        system=(
            "You are a mortgage underwriter. Based on the extracted application data and "
            "policy rules, assess eligibility. Return JSON:\n"
            '{"decision": "approved"|"rejected"|"refer_to_underwriter", '
            '"ltv_pct": float|null, "dsr_pct": float|null, '
            '"issues": [str], "conditions": [str], "summary": str}'
        ),
        user=(
            f"Document type: {doc_type}\n\n"
            f"Extracted data:\n{json.dumps(fields, indent=2)}\n\n"
            f"Policy rules:\n{policy}"
        ),
        json_mode=True,
    )
    data = json.loads(result)
    return {
        "eligibility":     data,
        "decision":        data["decision"],
        "steps_completed": state.get("steps_completed", []) + ["eligibility_check"],
    }


def generate_report(state: dict) -> dict:
    """Generate the final decision report for the customer file."""
    fields      = state.get("extracted_fields", {})
    eligibility = state.get("eligibility", {})
    human_note  = state.get("human_note", "")
    approved    = state.get("human_approved", None)

    result = _llm(
        system=(
            "Write a professional mortgage assessment report. Include: applicant summary, "
            "LTV and affordability analysis, decision with rationale, any conditions or "
            "next steps. Be specific with numbers. Max 250 words."
        ),
        user=(
            f"Applicant data: {json.dumps(fields, indent=2)}\n\n"
            f"Eligibility assessment: {json.dumps(eligibility, indent=2)}\n\n"
            f"Human reviewer note: {human_note or 'None'}\n"
            f"Final approval status: {'Approved by manager' if approved else 'Pending/Rejected by manager' if approved is False else 'Automated decision'}"
        ),
    )
    return {
        "final_report":    result,
        "steps_completed": state.get("steps_completed", []) + ["report"],
    }

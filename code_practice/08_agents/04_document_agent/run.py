# run.py — Portfolio demo: Mortgage Document Processing Agent
#
# Demonstrates:
#   1. Automatic pipeline: classify → extract → policy → eligibility → report
#   2. HITL flow: borderline case pauses for manager, then resumes
#
# Run: python run.py
# Set OPENAI_API_KEY before running.

from langgraph.types import Command
from graph import app


# ── Sample documents ───────────────────────────────────────────────────────────

MORTGAGE_APPLICATION = """
MORTGAGE APPLICATION FORM

Applicant: Sarah Mitchell
Date of Birth: 12 March 1988
National ID: GB-44729183
Employment: Software Engineer, TechCorp Ltd (permanent, 4 years)
Monthly Gross Salary: £6,200
Monthly Existing Debts: £450 (car loan £320, credit card minimum £130)

Property Details:
Address: 14 Elmwood Close, Manchester, M20 4PQ
Purchase Price: £385,000
Property Valuation: £390,000 (independent surveyor report attached)

Loan Request:
Amount: £320,000
Term: 25 years
Product: 5-year fixed rate
First-time buyer: Yes (no prior property ownership)

Documents Submitted: Payslips (3 months), P60 (latest), bank statements (3 months)
Deposit Source: Personal savings £65,000
"""

BORDERLINE_APPLICATION = """
MORTGAGE APPLICATION FORM

Applicant: James Okafor
Date of Birth: 5 September 1990
National ID: GB-58831029
Employment: Freelance Consultant (self-employed, 18 months)
Annual Net Profit (last year): £52,000 (monthly equiv: £4,333)
Monthly Existing Debts: £1,800 (personal loan £1,200, credit card £600)

Property Details:
Purchase Price: £295,000
Property Valuation: £290,000 (below purchase price — surveyor note: market decline risk)

Loan Request:
Amount: £260,000
Term: 30 years
First-time buyer: No

Deposit: £35,000
"""


# ── Run helpers ────────────────────────────────────────────────────────────────

def run_automatic(document: str, thread_id: str, label: str) -> None:
    """Process document through the full pipeline automatically."""
    print(f"\n{'═' * 68}")
    print(f"PROCESSING: {label}")
    print(f"Thread: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"document_text": document}, config=config)

    print(f"\n✓ Steps completed: {result.get('steps_completed', [])}")
    print(f"✓ Document type:   {result.get('doc_type', '—')}")
    print(f"✓ Decision:        {result.get('decision', '—').upper()}")

    fields = result.get("extracted_fields", {})
    print(f"\nExtracted:")
    for k, v in fields.items():
        if v is not None:
            print(f"  {k}: {v}")

    elig = result.get("eligibility", {})
    if elig.get("issues"):
        print(f"\nIssues: {elig['issues']}")

    print(f"\n{'─' * 68}")
    print("FINAL REPORT:")
    print(result.get("final_report", "No report generated."))


def run_with_hitl(document: str, thread_id: str, label: str) -> None:
    """Process borderline document — handles HITL interrupt."""
    print(f"\n{'═' * 68}")
    print(f"PROCESSING (with HITL): {label}")
    print(f"Thread: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    # First run — will interrupt at human_approval_gate if borderline
    result = app.invoke({"document_text": document}, config=config)

    # Check if we hit an interrupt
    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        print(f"\n⚠️  INTERRUPT: Manager review required")
        print(f"   Applicant: {interrupt_data.get('applicant')}")
        print(f"   Decision:  {interrupt_data.get('decision')}")
        print(f"   Issues:    {interrupt_data.get('issues')}")
        print(f"   Summary:   {interrupt_data.get('summary')}")

        # Simulate manager decision
        manager_decision = {"approved": True, "note": "Acceptable risk — freelancer with strong track record. Proceed with standard terms."}
        print(f"\n   Manager decision: {manager_decision}")

        # Resume the graph
        result = app.invoke(
            Command(resume=manager_decision),
            config=config,
        )

    print(f"\n✓ Steps completed: {result.get('steps_completed', [])}")
    print(f"✓ Decision:        {result.get('decision', '—').upper()}")
    print(f"✓ Human approved:  {result.get('human_approved')}")
    print(f"✓ Manager note:    {result.get('human_note')}")
    print(f"\n{'─' * 68}")
    print("FINAL REPORT:")
    print(result.get("final_report", "No report generated."))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Case 1: Clean application — automated pipeline, no HITL needed
    run_automatic(MORTGAGE_APPLICATION, "thread-mortgage-001", "Standard mortgage application")

    # Case 2: Borderline — triggers HITL manager approval
    run_with_hitl(BORDERLINE_APPLICATION, "thread-mortgage-002", "Borderline / refer case")


if __name__ == "__main__":
    main()

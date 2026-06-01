# graph.py — LangGraph supervisor for the mortgage document processing agent
#
# Pipeline: classify → extract → policy_retrieval → eligibility_check
#                                                          │
#                                               [HITL: manager approval]
#                                                          │
#                                                    generate_report → END
#
# The HITL interrupt fires only for borderline cases (decision = "refer_to_underwriter").
# Clean approvals go straight to report generation.

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agents import (
    classify_document,
    extract_fields,
    retrieve_policy_agent,
    check_eligibility,
    generate_report,
)


# ── State ─────────────────────────────────────────────────────────────────────
class DocumentState(TypedDict):
    # Input
    document_text:   str

    # Populated by agents
    doc_type:        str
    classification:  dict
    extracted_fields: dict
    policy_context:  str
    eligibility:     dict
    decision:        str          # "approved", "rejected", "refer_to_underwriter"
    final_report:    str
    steps_completed: list[str]

    # HITL
    human_approved:  Optional[bool]
    human_note:      Optional[str]


# ── HITL node ─────────────────────────────────────────────────────────────────
def human_approval_gate(state: DocumentState) -> dict:
    """
    Pause and wait for a manager to review borderline cases.
    LangGraph serialises state to MemorySaver; resumed with Command(resume={...}).
    """
    eligibility = state.get("eligibility", {})
    response = interrupt({
        "type":        "human_review",
        "message":     "⚠️  Borderline case — manager review required",
        "applicant":   state["extracted_fields"].get("applicant_name", "Unknown"),
        "decision":    eligibility.get("decision"),
        "issues":      eligibility.get("issues", []),
        "summary":     eligibility.get("summary", ""),
        "prompt":      "Enter {'approved': true/false, 'note': 'your comment'}",
    })
    return {
        "human_approved": response.get("approved", False),
        "human_note":     response.get("note", ""),
    }


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_eligibility(state: DocumentState) -> str:
    """Send borderline cases to human review; clear approvals/rejections go to report."""
    if state.get("decision") == "refer_to_underwriter":
        return "human_approval_gate"
    return "generate_report"


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph() -> object:
    graph = StateGraph(DocumentState)

    graph.add_node("classify",            classify_document)
    graph.add_node("extract",             extract_fields)
    graph.add_node("policy_retrieval",    retrieve_policy_agent)
    graph.add_node("eligibility_check",   check_eligibility)
    graph.add_node("human_approval_gate", human_approval_gate)
    graph.add_node("generate_report",     generate_report)

    # Linear pipeline
    graph.set_entry_point("classify")
    graph.add_edge("classify",          "extract")
    graph.add_edge("extract",           "policy_retrieval")
    graph.add_edge("policy_retrieval",  "eligibility_check")

    # Conditional: clear decision → report | borderline → HITL
    graph.add_conditional_edges(
        "eligibility_check",
        route_after_eligibility,
        {
            "human_approval_gate": "human_approval_gate",
            "generate_report":     "generate_report",
        },
    )

    graph.add_edge("human_approval_gate", "generate_report")
    graph.add_edge("generate_report",     END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


app = build_graph()

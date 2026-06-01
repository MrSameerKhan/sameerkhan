# Session 4 — Mortgage Document Processing Agent (Portfolio Project)
Status: `🔧 Code-built`

Theory: [../../../8.agents/07_multi_agent_orchestration.md](../../../8.agents/07_multi_agent_orchestration.md) · [../../../8.agents/04_langgraph_deep.md](../../../8.agents/04_langgraph_deep.md)

**Portfolio milestone:** after running this, update your resume with:
> "Built LangGraph mortgage document processing agent: classify → extract → policy retrieval → eligibility check → HITL manager approval → report generation. Handles borderline cases with interrupt-based human review."

---

## Use Case

A mortgage application arrives as a document. One agent handles the full pipeline: classify what document it is, extract all fields, check policy eligibility, pause for manager review if borderline, then generate a professional decision report.

Real deployments: ICE Data Services, Nanonets, any bank's document intake pipeline.

---

## Architecture — Specialist Agents + Conditional HITL

```
[START]
    │
  classify ──────────────────────── doc_type
    │
  extract ──────────────────────── extracted_fields
    │
  policy_retrieval ───────────────── policy_context
    │
  eligibility_check ──────────────── decision
    │
  route_after_eligibility?
    │
    ├── "approved" / "rejected" ──────────── generate_report ── [END]
    │
    └── "refer_to_underwriter" ── human_approval_gate
                                         │ (interrupt — wait for manager)
                                         ▼
                                   generate_report ── [END]
```

---

## Specialist Agent Pattern

Each agent in `agents.py` is a pure function: `state → partial_state_update`. The supervisor graph (LangGraph) orchestrates the order. This is cleaner than one monolithic agent:

| Agent | Input | Output | Why separate |
|-------|-------|--------|-------------|
| `classify_document` | raw text | doc_type | Different prompts for different doc types |
| `extract_fields` | raw text + doc_type | extracted_fields | Schema varies by document type |
| `retrieve_policy_agent` | doc_type + fields | policy_context | Can swap RAG retriever here |
| `check_eligibility` | fields + policy | decision, issues | Pure underwriting logic |
| `generate_report` | all of the above | final_report | Separate formatting from logic |

---

## HITL Flow Detail

```python
# In human_approval_gate node:
response = interrupt({
    "message": "⚠️ Borderline — manager review required",
    "applicant": state["extracted_fields"]["applicant_name"],
    "issues": state["eligibility"]["issues"],
})
# LangGraph pauses here, serialises state to MemorySaver

# Later, manager resumes:
result = app.invoke(
    Command(resume={"approved": True, "note": "Acceptable risk"}),
    config={"configurable": {"thread_id": "thread-002"}},
)
# Graph resumes from human_approval_gate → generate_report → END
```

The state at interrupt time is fully serialised. The graph can be resumed hours later — critical for real workflows where manager review takes time.

---

## Expected Output

```
PROCESSING: Standard mortgage application
Thread: thread-mortgage-001

✓ Steps completed: ['classify', 'extract', 'eligibility_check', 'report']
✓ Document type:   mortgage_application
✓ Decision:        APPROVED

Extracted:
  applicant_name: Sarah Mitchell
  property_value: 390000.0
  loan_amount: 320000.0
  monthly_income: 6200.0
  is_first_time_buyer: True
  loan_term_months: 300

──────────────────────────────────────────────────────────────────────
FINAL REPORT:
MORTGAGE ASSESSMENT — Sarah Mitchell

LTV Analysis: £320,000 / £390,000 = 82.1% — within the 95% limit for
first-time buyers. Affordability: estimated monthly payment approx £1,788
at 5-year fixed rate of 4.61%, representing 28.8% of gross income — well
within the 45% threshold. Decision: APPROVED...

══════════════════════════════════════════════════════════════════════
PROCESSING (with HITL): Borderline / refer case
Thread: thread-mortgage-002

⚠️  INTERRUPT: Manager review required
   Applicant: James Okafor
   Decision:  refer_to_underwriter
   Issues:    ['LTV 89.7% near limit', 'Self-employed under 2 years', 'Property valued below purchase price']

   Manager decision: {'approved': True, 'note': 'Acceptable risk — strong track record'}

✓ Human approved:  True
✓ Manager note:    Acceptable risk — strong track record
FINAL REPORT: [manager-approved decision with conditions]
```

---

## File Structure

```
04_document_agent/
├── agents.py  — 5 specialist agents (classify, extract, policy, eligibility, report)
├── graph.py   — LangGraph supervisor: pipeline + conditional HITL routing
└── run.py     — demo: automatic case + HITL borderline case
```

---

## How to Run

```bash
export OPENAI_API_KEY="sk-..."
cd code_practice/08_agents/04_document_agent
python run.py
```

Cost: ~$0.15 per run (2 documents × ~5 LLM calls each + report generation).
Runtime: ~30-45 seconds.

**Resume bullet:**
> Built LangGraph multi-agent document processing pipeline (classify → extract → policy RAG → eligibility → HITL → report) for mortgage applications; reduced manual review time by routing clear cases automatically and surfacing borderline cases to underwriters via interrupt-based approval.

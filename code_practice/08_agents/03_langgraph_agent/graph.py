# graph.py — LangGraph state machine for the production workflow agent (session 03)
#
# Architecture:
#   [START] → llm_node → (has tool_calls?) → tool_node → llm_node → ...
#                                          ↘ (no tool_calls) → [END]
#
# Features:
#   - MemorySaver checkpointer: conversation persists across .invoke() calls (thread_id)
#   - interrupt_before=["human_review"]: pause for human approval before sensitive actions
#   - tools_condition: built-in LangGraph router for tool-call vs end

import os
from typing import Annotated
import operator

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from tools import TOOLS

# ── State ─────────────────────────────────────────────────────────────────────
from typing import TypedDict

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    # operator.add means each node's returned messages are APPENDED (not overwritten)


# ── LLM with tools bound ──────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ["OPENAI_API_KEY"],
)
llm_with_tools = llm.bind_tools(TOOLS)

SYSTEM = SystemMessage(content=(
    "You are a bank assistant with access to tools. "
    "Use tools to answer precisely. When an action requires manager approval "
    "(e.g. confirming a loan decision), route to human_review."
))


# ── Nodes ─────────────────────────────────────────────────────────────────────
def llm_node(state: AgentState) -> dict:
    """Call the LLM. Prepend system message on first turn only."""
    msgs = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in msgs):
        msgs = [SYSTEM] + msgs
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


def human_review(state: AgentState) -> dict:
    """
    HITL gate: pause here and wait for human to approve or reject.
    LangGraph serialises state to checkpointer; resumed via Command(resume=...).
    """
    last = state["messages"][-1]
    decision = interrupt({
        "type":    "human_review",
        "message": "Agent wants to confirm a loan decision. Approve? (type 'approved' or 'rejected')",
        "context": last.content if hasattr(last, "content") else str(last),
    })
    return {"messages": [HumanMessage(content=f"Human decision: {decision}")]}


# ── Routing ───────────────────────────────────────────────────────────────────
def route_after_llm(state: AgentState) -> str:
    """Route based on whether the last message has tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        # Check if any tool call needs human review (e.g. confirm_loan)
        for tc in last.tool_calls:
            if tc["name"] in ("confirm_loan_decision",):
                return "human_review"
        return "tools"
    return END


# ── Graph ─────────────────────────────────────────────────────────────────────
def build_graph(with_hitl: bool = False) -> object:
    graph = StateGraph(AgentState)

    graph.add_node("llm",          llm_node)
    graph.add_node("tools",        ToolNode(TOOLS))

    if with_hitl:
        graph.add_node("human_review", human_review)
        graph.add_edge("human_review", "llm")

    graph.set_entry_point("llm")

    # Standard tool routing
    graph.add_conditional_edges(
        "llm",
        tools_condition,      # built-in: "tools" if tool_calls else END
    )
    graph.add_edge("tools", "llm")   # after tool → back to LLM

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


app = build_graph(with_hitl=False)

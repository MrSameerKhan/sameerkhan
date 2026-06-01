# run.py — Entry point for session 03 LangGraph agent
#
# Demonstrates:
#   1. Basic multi-turn conversation (thread_id persists state)
#   2. How checkpointing lets you resume mid-conversation
#   3. Streaming output (token-by-token)
#
# Run: python run.py
# Set OPENAI_API_KEY before running.

from langchain_core.messages import HumanMessage, AIMessage
from graph import app


def print_messages(result: dict) -> None:
    """Print the last assistant message from the result."""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            print(f"Agent: {msg.content}")
            return


def run_conversation(thread_id: str, turns: list[str]) -> None:
    """Run a multi-turn conversation on a single thread (persistent state)."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n{'═' * 68}")
    print(f"Thread: {thread_id}")

    for user_msg in turns:
        print(f"\nUser: {user_msg}")
        result = app.invoke(
            {"messages": [HumanMessage(content=user_msg)]},
            config=config,
        )
        print_messages(result)


def demo_streaming(question: str) -> None:
    """Stream agent output token by token."""
    print(f"\n{'═' * 68}")
    print(f"[Streaming] {question}")
    print("Agent: ", end="", flush=True)

    config = {"configurable": {"thread_id": "stream-demo"}}
    for chunk in app.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages",
    ):
        if chunk and hasattr(chunk[0], "content") and chunk[0].content:
            print(chunk[0].content, end="", flush=True)
    print()


def main():
    # ── Demo 1: Single-turn queries ───────────────────────────────────────────
    run_conversation("thread-001", [
        "What is the maximum LTV for first-time buyers?",
    ])

    run_conversation("thread-002", [
        "Calculate the monthly payment on a £220,000 mortgage at 4.61% over 300 months.",
    ])

    # ── Demo 2: Multi-turn — context persists within same thread ──────────────
    run_conversation("thread-003", [
        "I want to buy a £350,000 property as a first-time buyer.",
        "I can put down a £40,000 deposit. Is that enough?",
        "What would the monthly payment be at the 5-year fixed rate over 25 years?",
    ])

    # ── Demo 3: Streaming ─────────────────────────────────────────────────────
    demo_streaming("What are the early repayment charges and when do they apply?")


if __name__ == "__main__":
    main()

"""
pipeline.py — Full RAG pipeline: query → retrieve → prompt Ollama → stream answer

Prerequisites:
  brew install ollama
  ollama pull llama3.2
  ollama serve          (runs automatically after brew install)

Run:  python pipeline.py
      python pipeline.py --query "what is a transformer?"
"""

import argparse

import ollama

from config import OLLAMA_MODEL, TOP_K
from retriever import Retriever

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions
using only the provided context.

Rules:
- Answer using information from the context only.
- If the context does not contain enough information, say "I don't have enough
  information in the provided documents to answer this."
- Be concise and direct. Cite the source file when relevant.
- Do not make up facts."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} | relevance={chunk['score']:.3f}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def ask(query: str, retriever: Retriever) -> str:
    # Step 1 — retrieve
    chunks = retriever.retrieve(query, k=TOP_K)
    if not chunks:
        return "No relevant documents found in the index."

    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] score={c['score']:.4f}  {c['source']}")

    # Step 2 — build prompt
    user_message = build_prompt(query, chunks)

    # Step 3 — call Ollama (streaming)
    print("\n--- Answer ---\n")
    full_response = ""
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        stream=True,
    )
    for chunk in stream:
        text = chunk["message"]["content"]
        print(text, end="", flush=True)
        full_response += text

    print("\n")
    return full_response


def main(query: str | None = None) -> None:
    # Quick check that Ollama is reachable and model is available
    try:
        models = [m.model for m in ollama.list().models]
    except Exception:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure it is running:\n"
            "  brew install ollama\n"
            "  ollama serve\n"
            "  ollama pull llama2"
        )

    if not any(OLLAMA_MODEL in m for m in models):
        raise RuntimeError(
            f"Model '{OLLAMA_MODEL}' not found locally.\n"
            f"Run: ollama pull {OLLAMA_MODEL}"
        )

    retriever = Retriever()

    if query:
        ask(query, retriever)
    else:
        # Interactive loop
        print(f"\nRAG System ready (model: {OLLAMA_MODEL}). Type your question (or 'quit' to exit).\n")
        while True:
            try:
                query = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            ask(query, retriever)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="Single query (optional)")
    args = parser.parse_args()
    main(args.query)

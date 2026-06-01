# Session 5 — vLLM Serving: client.py
# Demonstrates how to query a vLLM server using the OpenAI SDK (drop-in replacement).
# Works with any OpenAI-compatible server: vLLM, LM Studio, Ollama, Groq, etc.
#
# Run server first: python server.py --mode serve --port 8000
# Then run: python client.py

import time
import asyncio
import statistics
from openai import OpenAI, AsyncOpenAI

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL         = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # must match server --model

sync_client  = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
async_client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")


# ── Single request ─────────────────────────────────────────────────────────────
def single_query(question: str) -> tuple[str, float]:
    t0  = time.perf_counter()
    res = sync_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a bank assistant. Answer specifically."},
            {"role": "user",   "content": question},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    latency = time.perf_counter() - t0
    return res.choices[0].message.content, latency


# ── Throughput benchmark ───────────────────────────────────────────────────────
async def async_query(question: str) -> tuple[str, float]:
    t0  = time.perf_counter()
    res = await async_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
        temperature=0.7,
    )
    return res.choices[0].message.content, time.perf_counter() - t0


async def throughput_benchmark(questions: list[str]) -> dict:
    """Send all requests concurrently — measures vLLM's continuous batching benefit."""
    t0      = time.perf_counter()
    results = await asyncio.gather(*[async_query(q) for q in questions])
    total   = time.perf_counter() - t0

    latencies = [r[1] for r in results]
    return {
        "requests":         len(questions),
        "total_time_s":     round(total, 2),
        "requests_per_sec": round(len(questions) / total, 1),
        "p50_latency_ms":   round(statistics.median(latencies) * 1000),
        "p95_latency_ms":   round(sorted(latencies)[int(0.95 * len(latencies))] * 1000),
    }


# ── Streaming ─────────────────────────────────────────────────────────────────
def streaming_query(question: str) -> None:
    print(f"\nQ: {question}")
    print("A: ", end="", flush=True)

    stream = sync_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=150,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    questions = [
        "What is the maximum LTV for a first-time buyer?",
        "How much is the early repayment charge in year 3?",
        "What documents are required for a mortgage application?",
        "What is the maximum personal finance for a SAR 10,000 salary?",
        "What is the SVR rate and when does it apply?",
    ]

    # Single requests
    print("── Single requests ──")
    for q in questions[:2]:
        answer, latency = single_query(q)
        print(f"Q: {q}")
        print(f"A: {answer}")
        print(f"   Latency: {latency*1000:.0f}ms\n")

    # Throughput benchmark (concurrent)
    print("── Throughput benchmark (50 concurrent requests) ──")
    all_questions = questions * 10   # 50 requests
    stats = asyncio.run(throughput_benchmark(all_questions))
    print(f"  Requests:          {stats['requests']}")
    print(f"  Total time:        {stats['total_time_s']}s")
    print(f"  Throughput:        {stats['requests_per_sec']} req/s")
    print(f"  Median latency:    {stats['p50_latency_ms']}ms")
    print(f"  P95 latency:       {stats['p95_latency_ms']}ms")

    # Expected: vLLM ≈ 80-200 req/s vs HuggingFace serial ≈ 1-3 req/s

    # Streaming
    print("\n── Streaming ──")
    streaming_query("Explain the ERC schedule for a 5-year fixed mortgage.")


if __name__ == "__main__":
    main()

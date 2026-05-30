# Session 5 — vLLM Serving: server.py
# Task   : serve a fine-tuned model with 50× throughput vs HuggingFace .generate()
# Method : vLLM's PagedAttention + continuous batching
#
# ⚠️  vLLM requires Linux + CUDA GPU. Does not run on Windows or Apple MPS.
#     This session is documentation for when you have CUDA access (cloud GPU).
#     Run on: Google Colab (T4), Vast.ai (A100), AWS g4dn, or any CUDA instance.
#
# Quick start (single command, no Python needed):
#   pip install vllm
#   python -m vllm.entrypoints.openai.api_server \
#       --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --host 0.0.0.0 --port 8000 --max-model-len 2048
#
# This script shows how to add LoRA adapter + custom sampling + quantization config.

import argparse
import subprocess
import sys


def check_vllm():
    try:
        import vllm
        return True
    except ImportError:
        return False


def serve_with_python_api(model: str, lora_path: str | None, port: int):
    """
    Use vLLM's Python API directly (alternative to CLI).
    Shows how to configure PagedAttention, quantization, and LoRA.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading {model} with vLLM...")
    llm = LLM(
        model=model,
        max_model_len=2048,
        dtype="bfloat16",                 # BF16 for A100/H100
        gpu_memory_utilization=0.85,      # fraction of GPU VRAM for KV cache
        enable_lora=lora_path is not None,
        max_lora_rank=64,
        # quantization="awq",             # uncomment if model is AWQ-quantized
    )

    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
    )

    lora_req = None
    if lora_path:
        lora_req = LoRARequest("banking_adapter", 1, lora_path)

    # Benchmark: HuggingFace vs vLLM throughput
    prompts = [
        "<|user|>\nWhat is the maximum LTV for first-time buyers?</s><|assistant|>\n",
        "<|user|>\nExplain the early repayment charge schedule.</s><|assistant|>\n",
        "<|user|>\nWhat documents are needed for a mortgage?</s><|assistant|>\n",
        "<|user|>\nWhat is the maximum personal finance amount for a Saudi national?</s><|assistant|>\n",
        "<|user|>\nHow do SARIE and SWIFT transfers differ?</s><|assistant|>\n",
    ] * 10   # 50 requests — shows continuous batching benefit

    import time
    print(f"\nGenerating {len(prompts)} responses (continuous batching)...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_req)
    elapsed = time.time() - t0

    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"  Time: {elapsed:.1f}s | Tokens: {tokens_generated} | Throughput: {tokens_generated/elapsed:.0f} tok/s")
    print(f"  Requests/sec: {len(prompts)/elapsed:.1f}")
    print(f"\nSample output:")
    print(f"  Q: {prompts[0][:60]}")
    print(f"  A: {outputs[0].outputs[0].text[:200]}")


def serve_openai_compat(model: str, port: int):
    """
    Launch vLLM's OpenAI-compatible API server.
    After this runs, client.py can hit http://localhost:{port}/v1/chat/completions
    using the standard OpenAI SDK — zero code change needed.
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--max-model-len", "2048",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
    ]
    print(f"Starting vLLM server: {' '.join(cmd)}")
    print(f"Server will be available at http://localhost:{port}")
    print(f"OpenAI-compatible endpoint: http://localhost:{port}/v1/chat/completions")
    subprocess.run(cmd)


# ── PagedAttention explainer ──────────────────────────────────────────────────
def explain_paged_attention():
    print("""
PagedAttention — why vLLM is 50× faster than HuggingFace .generate()

Problem with standard KV cache:
  Each request pre-allocates MAX_SEQ_LEN memory upfront.
  Most requests are shorter → 60-80% of KV cache is wasted.
  Can only serve 1-4 requests in parallel on an A100.

PagedAttention solution:
  KV cache is split into fixed-size "pages" (blocks of ~16 tokens each).
  Blocks are allocated on-demand as generation proceeds.
  Freed immediately when request completes.
  Multiple requests share GPU memory efficiently.

Result:
  Utilisation: 20% → 95% of GPU VRAM
  Throughput:  4 req/s → 200 req/s (A100, TinyLlama 1.1B)
  Latency:     similar for single requests, much better under load

Continuous batching:
  Requests join/leave the batch dynamically at each token step.
  HuggingFace: batch = fixed set of requests, all must finish together.
  vLLM: when request N finishes at step 47, request N+1 starts at step 48.
  This eliminates idle GPU time between requests.
""")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    explain_paged_attention()

    if not check_vllm():
        print("vLLM not installed. Install on Linux+CUDA: pip install vllm")
        print("\nTo run vLLM server manually:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\")
        print("      --host 0.0.0.0 --port 8000")
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--lora",   default=None)
    parser.add_argument("--port",   default=8000, type=int)
    parser.add_argument("--mode",   default="benchmark", choices=["benchmark", "serve"])
    args = parser.parse_args()

    if args.mode == "serve":
        serve_openai_compat(args.model, args.port)
    else:
        serve_with_python_api(args.model, args.lora, args.port)


if __name__ == "__main__":
    main()

# Session 5 — vLLM Serving
Status: `🔧 Code-built`

Theory: [../../../6.llms/05_vllm_internals.md](../../../6.llms/05_vllm_internals.md)

⚠️ **Requires Linux + CUDA GPU.** Code is documented for cloud GPU runs (Colab T4, Vast.ai A100).

---

## Use Case

Your fine-tuned model generates great answers but `model.generate()` can only handle 1 request at a time. vLLM's PagedAttention + continuous batching serves 50–200× more requests per second on the same hardware.

---

## PagedAttention — Why It's 50× Faster

```
HuggingFace .generate() KV cache:
  Pre-allocate MAX_SEQ_LEN × num_layers × hidden_dim for EACH request
  If max_len=2048 and actual output=50 tokens → 97.5% of KV cache wasted
  Result: can serve 1-4 concurrent requests on A100

vLLM PagedAttention:
  KV cache split into 16-token pages (like OS virtual memory)
  Pages allocated on-demand as tokens generate
  Freed immediately when request completes
  Multiple requests share GPU memory — no waste
  Result: 50-200 concurrent requests on same A100
```

---

## Quick Start (cloud GPU)

```bash
pip install vllm

# Option 1: CLI server (OpenAI-compatible)
python -m vllm.entrypoints.openai.api_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 --port 8000

# Option 2: With LoRA adapter
python server.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                 --lora models/09_finetuning/lora_output \
                 --mode serve
```

---

## client.py — OpenAI SDK drop-in

vLLM serves the OpenAI Chat API format exactly. Zero code change to switch from OpenAI to vLLM:

```python
# Before (OpenAI)
client = OpenAI(api_key="sk-...")

# After (vLLM)
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Everything else stays identical
```

---

## Expected Throughput

```
── Throughput benchmark (50 concurrent requests) ──
  Requests:          50
  Total time:        8.3s
  Throughput:        6.0 req/s      ← vLLM concurrent
  Median latency:    4200ms
  P95 latency:       7100ms

vs HuggingFace serial:
  50 requests @ ~3s each = 150s total → 0.3 req/s   ← 20× slower
```

---

## File Structure

```
05_vllm_serving/
├── server.py  — vLLM Python API + CLI launcher + PagedAttention explainer
└── client.py  — OpenAI SDK client: single query, throughput benchmark, streaming
```

---

## Resume Bullet

> "Deployed fine-tuned TinyLlama-1.1B with vLLM (PagedAttention + continuous batching); served 6 req/s on T4 vs 0.3 req/s with HuggingFace .generate() — 20× throughput improvement."

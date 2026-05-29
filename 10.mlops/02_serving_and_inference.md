# 02 — Model Serving & Inference

> From trained model to production API. ONNX export, FastAPI serving, vLLM for LLMs, dynamic batching. The optimization stack.

---

## Quick Reference

| Stack | Best For | Throughput |
|---|---|---|
| FastAPI + ONNX | Custom models, low latency | High |
| TorchServe | PyTorch models, batching | Medium-High |
| vLLM | LLM inference | SOTA LLM throughput |
| TGI (HuggingFace) | HuggingFace LLMs | High |
| Triton Inference Server | Multi-framework, GPU batching | Highest |
| BentoML | ML frameworks agnostic | Medium |

---

## Core Concepts

### Inference Optimization Stack

```
Raw PyTorch model
  ↓ [Export]
ONNX / TorchScript / TensorRT
  ↓ [Quantization]
INT8 / FP16 / BF16
  ↓ [Batching]
Dynamic batching (server-side)
  ↓ [Caching]
KV cache (LLMs) / Result cache
  ↓ [Scaling]
Horizontal scaling + load balancing
```

---

## ONNX Export & Inference

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "microsoft/deberta-v3-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=9)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# — Export to ONNX ————————————————————————————
dummy_inputs = tokenizer(
    "Sample text for export",
    return_tensors="pt",
    padding="max_length",
    max_length=512,
    truncation=True,
)

torch.onnx.export(
    model,
    args=(dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    f="model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,  # fold constants for optimization
)

# — Optimize ONNX (optional) ——————————————————
from onnxruntime.transformers import optimizer as ort_optimizer

opt_model = ort_optimizer.optimize_model(
    "model.onnx",
    model_type="bert",
    num_heads=12,
    hidden_size=768,
    optimization_options=ort_optimizer.FusionOptions("bert"),
)
opt_model.convert_float_to_float16()  # FP16 for GPU inference
opt_model.save_model_to_file("model_opt_fp16.onnx")

# — ONNX Runtime Inference —————————————————————
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession(
    "model_opt_fp16.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

# Warm up
_ = session.run(None, {
    "input_ids": np.zeros((1, 512), dtype=np.int64),
    "attention_mask": np.zeros((1, 512), dtype=np.int64),
})

# Inference
def predict(texts: list[str]) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    logits = session.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    )[0]
    return logits.argmax(axis=-1)
```

---

## FastAPI Serving

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import time
from contextlib import asynccontextmanager
import onnxruntime as ort
import numpy as np

# — Lifespan: load model once at startup ——————
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.session = ort.InferenceSession(
        "model.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.state.tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    yield
    # Cleanup on shutdown (if needed)

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    texts: list[str]
    max_length: int = 512

class PredictResponse(BaseModel):
    predictions: list[int]
    probabilities: list[list[float]]
    latency_ms: float

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if len(request.texts) > 32:
        raise HTTPException(status_code=400, detail="Max batch size is 32")

    start = time.perf_counter()

    inputs = app.state.tokenizer(
        request.texts,
        padding=True,
        truncation=True,
        max_length=request.max_length,
        return_tensors="np",
    )

    logits = app.state.session.run(
        None,
        {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
    )[0]

    latency_ms = (time.perf_counter() - start) * 1000

    probs = softmax(logits, axis=-1)
    preds = logits.argmax(axis=-1).tolist()

    return PredictResponse(
        predictions=preds,
        probabilities=probs.tolist(),
        latency_ms=latency_ms,
    )

@app.get("/health")
async def health():
    return {"status": "healthy"}

# — Dynamic Batching (micro-batching) —————————
# For high-throughput APIs: collect requests within a time window, batch them

class BatchingService:
    def __init__(self, max_batch_size=32, max_wait_ms=10):
        self.queue = asyncio.Queue()
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms / 1000

    async def predict_single(self, text: str):
        future = asyncio.get_event_loop().create_future()
        await self.queue.put((text, future))
        return await future

    async def process_batches(self):
        while True:
            batch = []
            futures = []

            # Wait for first item
            item, future = await self.queue.get()
            batch.append(item)
            futures.append(future)

            # Collect more items (up to max_batch within max_wait time)
            deadline = asyncio.get_event_loop().time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item, future = await asyncio.wait_for(
                        self.queue.get(), timeout=remaining
                    )
                    batch.append(item)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break

            # Process batch
            results = run_model(batch)
            for future, result in zip(futures, results):
                future.set_result(result)
```

---

## LLM Serving with vLLM

```python
# vLLM: PagedAttention + continuous batching = SOTA LLM throughput
# 24× throughput vs naive HuggingFace generate on A100

from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,      # number of GPUs
    gpu_memory_utilization=0.9,  # fraction of GPU memory to use
    max_model_len=8096,
    dtype="bfloat16",
    enforce_eager=False,         # use CUDA graph (faster)
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["</s>", "[INST]"],
)

# Batch inference (automatically handles dynamic batching)
prompts = [
    "[INST] What is machine learning? [/INST]",
    "[INST] Explain transformers in 2 sentences. [/INST]",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)

# — vLLM as OpenAI-compatible server ——————————
# python -m vllm.entrypoints.openai.api_server \
#   --model meta-llama/Llama-2-7b-chat-hf \
#   --served-model-name llama2 \
#   --tensor-parallel-size 1 \
#   --port 8000

# Then use OpenAI client:
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="ignored")
response = client.chat.completions.create(
    model="llama2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### PagedAttention (vLLM key innovation)

```
Standard KV cache: allocate max_seq_len memory upfront
  - Wastes memory for short sequences
  - Limits concurrent requests

PagedAttention: KV cache in fixed-size pages (like OS virtual memory)
  - Allocates pages on demand as sequence grows
  - Shares pages between sequences that have common prefix
  - 3-24× more concurrent requests + 24× higher throughput
```

---

## Quantization for Serving

```python
# — Dynamic INT8 quantization (PyTorch) ———————
from torch.quantization import quantize_dynamic

model_int8 = quantize_dynamic(
    model,
    {torch.nn.Linear},  # only quantize linear layers
    dtype=torch.qint8
)
# ~4× smaller, ~2× faster on CPU, minimal accuracy loss

# — TorchScript for deployment ————————————————
traced = torch.jit.trace(model, (input_ids, attention_mask))
torch.jit.save(traced, "model_traced.pt")

# Load and use
loaded = torch.jit.load("model_traced.pt")
output = loaded(input_ids, attention_mask)

# — TensorRT for maximum GPU throughput ————————
# Usually done via ONNX + TensorRT
import tensorrt as trt
trtexec_cmd = """
trtexec \
  --onnx=model.onnx \
  --saveEngine=model.trt \
  --fp16 \
  --minShapes=input_ids:1x1,attention_mask:1x1 \
  --optShapes=input_ids:8x256,attention_mask:8x256 \
  --maxShapes=input_ids:32x512,attention_mask:32x512 \
  --workspace=4096
"""
# Results in 3-5× speedup vs ONNX Runtime on the same GPU
```

---

## Latency vs Throughput

```
Latency: time for single request (P50, P95, P99)
Throughput: requests per second (across all concurrent requests)

They trade off:
  - Large batch = high throughput, high latency (wait time in queue)
  - Small batch = low latency, low throughput (GPU underutilized)

Optimize for:
  Latency-sensitive (real-time API): small batch, aggressive optimization
  Throughput-sensitive (offline batch): large batch, GPU utilization priority

Key latency bottlenecks:
  1. Model forward pass (GPU compute)
  2. Data transfer CPU → GPU (H2D latency)
  3. Tokenization (CPU)
  4. Network (client → server)
```

### Benchmarking

```python
import time
import numpy as np

def benchmark_latency(predict_fn, inputs, n_runs=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        predict_fn(inputs)

    # Measure
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        predict_fn(inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "mean_ms": np.mean(latencies),
        "std_ms": np.std(latencies),
    }

# Target SLAs (typical):
# Text classification API: P95 < 50ms
# LLM generation: P95 < 2s for first token (TTFT)
# Document extraction: P95 < 300ms
```

---

## Gotchas

**Cold start latency:** First inference is always slow (CUDA initialization, model loading). Keep models warm with periodic dummy requests. Use `torch.jit.optimize_for_inference(model)` after tracing.

**Batch padding inefficiency:** Dynamic padding (pad to longest in batch) is much better than padding to max_length for all sequences. Reduces wasted compute significantly for variable-length inputs.

**GPU memory fragmentation:** Loading/unloading models repeatedly fragments GPU memory. Prefer keeping models in memory; use `torch.cuda.empty_cache()` only when necessary.

**ONNX opset compatibility:** Not all PyTorch ops are supported in all ONNX opsets. Use `opset_version=17` (latest stable). Custom ops need custom ONNX plugins.

**vLLM prefix caching:** vLLM automatically caches KV states for identical prompt prefixes. For RAG, putting the system prompt first (consistent across requests) enables prefix cache hits → major latency reduction.

---

## Interview Q&A

**Q: How would you optimize a BERT model for production inference?**

Step-by-step: (1) Baseline: measure p50/p95 latency and throughput with representative inputs. (2) ONNX export: `torch.onnx.export` with dynamic axes, verify outputs match within floating-point tolerance. (3) ONNX Runtime: typically 3-4× speedup over PyTorch on CPU without any accuracy loss. (4) FP16 conversion — halves memory and speeds up GPU compute. (5) Profile with `torch.profiler` to find actual bottlenecks. (6) Server-side dynamic batching — collect requests within 10ms window, process as a batch. (7) Horizontal scaling with load balancer for high concurrency. Typical result: 4-8× improvement over naive PyTorch.

**Q: What is PagedAttention in vLLM and why does it matter?**

Standard LLM serving pre-allocates a contiguous KV cache block for each request sized for max sequence length. If max_seq_len=4096 but you serve 100 users, 400KB × 100 = 40MB is allocated — most of this is wasted for short responses and limits concurrent requests. PagedAttention stores KV cache in fixed-size non-contiguous pages (like OS virtual memory paging). Pages are allocated on demand and can be shared across sequences sharing a common prefix. Result: no zero-page fragmentation, 3-5× more concurrent requests, same GPU, same latency — massive throughput improvement at the same cost.

---

## Connections

- Efficient Transformers (`transformers/models/04`): Quantization, Flash Attention covered architecturally
- MLOps Pipelines (`7.mlops/03`): Monitor latency/throughput metrics post-deployment
- MLOps Pipelines (`7.mlops/04`): CI/CD deploys the serving infrastructure
- System Design (`8.system_design`): Serving is a key component of ML system design

## Key Takeaway

```
Serving stack: ONNX export → FP16 → ONNX Runtime for BERT-class models.
vLLM for LLMs — PagedAttention gives 24× throughput vs naive HuggingFace.
Always batch before optimizing — bottleneck is rarely where you expect.
Key metrics: P95 latency and throughput (req/s).
SLA targets: <50ms for classification, <2s TTFT for LLM.
```

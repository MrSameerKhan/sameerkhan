# Serving Optimization — End-to-End

> **Workflow:** PyTorch model → ONNX export → Quantization → Benchmark → Deploy

---

## Why Optimize Serving?

```
Problem: GBM classifier fine, but PyTorch model deployed for invoice classification.
  Training model: 89ms p95 latency (GPU batch forward pass)
  Serving requirement: <20ms p95 (100K QPS, real-time API)

Optimization path:
  PyTorch (89ms) → ONNX export (42ms) → INT8 quantization (11ms) → TensorRT (6ms)
                                                                    ↑
                                        For ONNX-only deploy, 11ms often sufficient
```

---

## Step 1: Baseline — Measure Before Optimizing

```python
import torch
import time
import numpy as np

# Load production model
model = torch.load("invoice_classifier.pt")
model.eval()

def benchmark(model_fn, input_data, n_warmup=50, n_runs=500):
    """Measure p50, p95, p99 latency and throughput."""
    device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'

    # Warmup (fill GPU cache, JIT compile)
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model_fn(input_data)

    # Synchronize GPU before timing
    if str(device) != 'cpu':
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_runs):
        if str(device) != 'cpu':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            _ = model_fn(input_data)

        if str(device) != 'cpu':
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    return {
        "p50_ms":     round(np.percentile(latencies, 50), 2),
        "p95_ms":     round(np.percentile(latencies, 95), 2),
        "p99_ms":     round(np.percentile(latencies, 99), 2),
        "mean_ms":    round(latencies.mean(), 2),
        "throughput": round(1000 / latencies.mean(), 1),   # requests/sec (batch_size=1)
    }

# Batch size 1 (online serving)
x = torch.randn(1, 128).cuda()
baseline = benchmark(model, x)
print("PyTorch FP32:", baseline)
# {'p50_ms': 8.1, 'p95_ms': 12.3, 'p99_ms': 18.7, 'mean_ms': 8.4, 'throughput': 119.0}
```

---

## Step 2: ONNX Export

```python
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

# Export to ONNX
dummy_input = torch.randn(1, 128)  # batch_size=1, features=128

torch.onnx.export(
    model,
    dummy_input,
    "invoice_classifier.onnx",
    opset_version=17,
    input_names=["features"],
    output_names=["logits"],
    dynamic_axes={
        "features": {0: "batch_size"},   # allow variable batch size
        "logits":   {0: "batch_size"},
    },
    do_constant_folding=True,  # fold constant ops at export time
    export_params=True,        # store trained parameters in model file
)
print("ONNX export complete.")

# Verify the export
onnx_model = onnx.load("invoice_classifier.onnx")
onnx.checker.check_model(onnx_model)
print(f"ONNX model: {len(onnx_model.graph.node)} nodes, opset {onnx_model.opset_import[0].version}")

# Compare outputs: PyTorch vs ONNX
x_np = np.random.randn(1, 128).astype(np.float32)
x_pt = torch.FloatTensor(x_np)

with torch.no_grad():
    pt_out = model(x_pt).numpy()

ort_session = ort.InferenceSession("invoice_classifier.onnx",
                                    providers=["CPUExecutionProvider"])
onnx_out = ort_session.run(["logits"], {"features": x_np})[0]

max_diff = np.abs(pt_out - onnx_out).max()
print(f"Max output difference PyTorch vs ONNX: {max_diff:.6f}")
# Max output difference: 0.000002  ← within floating point precision, OK
```

### ONNX Runtime Benchmark

```python
# Benchmark ONNX Runtime (CPU)
ort_session_cpu = ort.InferenceSession(
    "invoice_classifier.onnx",
    providers=["CPUExecutionProvider"],
    sess_options=ort.SessionOptions()
)

x_np_batch1 = np.random.randn(1, 128).astype(np.float32)

def onnx_infer(x):
    return ort_session_cpu.run(["logits"], {"features": x})[0]

onnx_metrics = benchmark(onnx_infer, x_np_batch1)
print("ONNX Runtime (CPU):", onnx_metrics)
# {'p50_ms': 2.1, 'p95_ms': 3.4, 'p99_ms': 5.1, 'mean_ms': 2.2, 'throughput': 455.0}
# → 3.6× speedup vs PyTorch FP32 on same CPU
```

---

## Step 3: Quantization

### Dynamic Quantization (easiest, CPU only)

```python
import torch.quantization

# Dynamic quantization: weights stored as INT8, activations quantized at runtime
# No calibration data needed — fastest to apply
model_dynamic_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},     # quantize these layer types
    dtype=torch.qint8
)

# Benchmark
dynamic_metrics = benchmark(model_dynamic_int8, torch.randn(1, 128))
print("Dynamic INT8:", dynamic_metrics)
# {'p50_ms': 3.2, 'p95_ms': 4.8, 'p99_ms': 6.1, 'mean_ms': 3.3, 'throughput': 303.0}

# Size comparison
import os
torch.save(model.state_dict(), "model_fp32.pt")
torch.save(model_dynamic_int8.state_dict(), "model_int8.pt")

fp32_size = os.path.getsize("model_fp32.pt") / 1e6
int8_size  = os.path.getsize("model_int8.pt") / 1e6
print(f"FP32: {fp32_size:.1f} MB → INT8: {int8_size:.1f} MB ({int8_size/fp32_size:.1%})")
# FP32: 12.4 MB → INT8: 3.2 MB (25.8% — 4× compression)
```

### Static Quantization (better accuracy, needs calibration data)

```python
import torch.quantization
from torch.quantization import prepare, convert, QConfig, default_qconfig

# Step 1: Prepare model for quantization (insert observers)
model_fp32 = torch.load("invoice_classifier.pt")
model_fp32.eval()

# Set quantization config
model_fp32.qconfig = torch.quantization.get_default_qconfig('x86')
model_prepared = torch.quantization.prepare(model_fp32)

# Step 2: Calibrate on representative data
# Feed 1K samples so observers collect activation statistics
calibration_loader = get_calibration_data(n_samples=1000)
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

# Step 3: Convert — replace float ops with INT8 ops
model_int8_static = torch.quantization.convert(model_prepared)

# Verify accuracy preserved
x_cal = next(iter(calibration_loader))
with torch.no_grad():
    fp32_out = model_fp32(x_cal)
    int8_out  = model_int8_static(x_cal)

cosine_sim = torch.nn.functional.cosine_similarity(fp32_out, int8_out, dim=-1).mean()
print(f"Cosine similarity FP32 vs INT8: {cosine_sim:.4f}")
# 0.9997 → practically identical outputs

static_metrics = benchmark(model_int8_static, torch.randn(1, 128))
print("Static INT8:", static_metrics)
# {'p50_ms': 1.8, 'p95_ms': 2.9, 'p99_ms': 3.8, 'mean_ms': 1.9, 'throughput': 526.0}
```

### ONNX INT8 Quantization

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize ONNX model to INT8 (dynamic)
quantize_dynamic(
    model_input="invoice_classifier.onnx",
    model_output="invoice_classifier_int8.onnx",
    weight_type=QuantType.QInt8,
)

# Load and benchmark
ort_int8 = ort.InferenceSession("invoice_classifier_int8.onnx",
                                 providers=["CPUExecutionProvider"])

def onnx_int8_infer(x):
    return ort_int8.run(["logits"], {"features": x})[0]

onnx_int8_metrics = benchmark(onnx_int8_infer, x_np_batch1)
print("ONNX INT8:", onnx_int8_metrics)
# {'p50_ms': 1.1, 'p95_ms': 1.7, 'p99_ms': 2.3, 'mean_ms': 1.2, 'throughput': 833.0}
```

---

## Step 4: Benchmark Comparison

```
Full benchmark dry run — invoice classifier, batch_size=1, CPU (Intel Xeon)

Method                    p50     p95     p99    Throughput    Size
─────────────────────────────────────────────────────────────────────
PyTorch FP32 (baseline)   8.1ms  12.3ms  18.7ms   119 req/s   12.4 MB
PyTorch Dynamic INT8      3.2ms   4.8ms   6.1ms   303 req/s    3.2 MB  (2.5× faster)
PyTorch Static INT8       1.8ms   2.9ms   3.8ms   526 req/s    3.2 MB  (4.2× faster)
ONNX FP32                 2.1ms   3.4ms   5.1ms   455 req/s   12.2 MB  (3.6× faster)
ONNX INT8                 1.1ms   1.7ms   2.3ms   833 req/s    3.1 MB  (7.2× faster)

Target: <20ms p95 ✓ — all methods meet target
Recommendation: ONNX INT8 for production (7.2× speedup, 4× size reduction)
```

---

## Step 5: Accuracy Validation After Quantization

```python
from sklearn.metrics import roc_auc_score

def evaluate_model(predict_fn, X_test, y_test):
    y_prob = predict_fn(X_test)
    auc = roc_auc_score(y_test, y_prob)
    return auc

# Load test set
X_test_np = X_test.numpy().astype(np.float32)

# FP32 baseline AUC
fp32_auc = evaluate_model(
    lambda x: model(torch.FloatTensor(x)).detach().numpy(),
    X_test_np, y_test
)

# ONNX INT8 AUC
int8_auc = evaluate_model(
    lambda x: ort_int8.run(["logits"], {"features": x})[0],
    X_test_np, y_test
)

print(f"FP32 AUC: {fp32_auc:.4f}")
print(f"INT8 AUC: {int8_auc:.4f}")
print(f"AUC drop: {(fp32_auc - int8_auc)*100:.3f}%")
# FP32 AUC: 0.9231
# INT8 AUC: 0.9218
# AUC drop: 0.013%  ← acceptable (< 0.1% threshold)
```

---

## Step 6: Production Serving with Optimized Model

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
import onnxruntime as ort
import numpy as np
import time

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load optimized model at startup
    app.state.session = ort.InferenceSession(
        "invoice_classifier_int8.onnx",
        providers=["CPUExecutionProvider"],
        sess_options=configure_ort_session(),
    )
    yield
    # Cleanup on shutdown (if needed)

def configure_ort_session() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4    # parallelism within single inference
    opts.inter_op_num_threads = 1    # parallelism between ops
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True   # optimize memory allocation
    return opts

app = FastAPI(lifespan=lifespan)

@app.post("/v1/predict")
async def predict(request: dict):
    t0 = time.perf_counter()

    features = np.array(request["features"], dtype=np.float32).reshape(1, -1)
    outputs  = app.state.session.run(["logits"], {"features": features})[0]
    prob     = float(outputs[0])

    latency_ms = (time.perf_counter() - t0) * 1000

    return {
        "probability":  prob,
        "label":        int(prob > 0.42),
        "latency_ms":   round(latency_ms, 2),
        "model_version": "invoice-classifier/5/int8",
    }
```

---

## LLM-Specific: vLLM and PagedAttention

For LLM serving (GPT, LLaMA, Mistral), the bottleneck is different — it's memory bandwidth, not compute.

```python
# Standard HuggingFace serving (inefficient — one request at a time)
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Problem:
# - KV cache for one request: 2 × layers × heads × seq_len × head_dim × dtype
# - Mistral-7B: 2 × 32 × 8 × 4096 × 128 × 2 bytes = 512 MB per request
# - 80GB GPU → only 160 concurrent requests max
# - Naive allocation: fragments memory → can only fit ~40 requests
```

```python
# vLLM: 3-5× throughput improvement via PagedAttention + continuous batching
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-v0.1",
    tensor_parallel_size=1,    # 1 GPU
    gpu_memory_utilization=0.9,  # use 90% of GPU VRAM for KV cache
    dtype="float16",
)

# PagedAttention: KV cache stored in non-contiguous "pages" (like OS virtual memory)
# → eliminates internal fragmentation
# → 160 requests fit instead of 40 (4× more concurrent requests)

# Continuous batching: new requests join the batch mid-generation
# → GPU never idles waiting for slow requests to finish

sampling_params = SamplingParams(temperature=0.8, max_tokens=256)

# Batch inference
prompts = [
    "Extract the invoice number from: INV-2024-0432, dated March 14...",
    "Classify this document as invoice, contract, or receipt: ...",
]
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text)
```

**PagedAttention key insight:**
```
Traditional KV cache:
  Request 1: allocates 512MB contiguous block (for max_seq_len=4096)
  Request 2: needs 512MB but only 400MB available → rejected even though
             request 2 only uses 100 tokens (100MB actual)

PagedAttention (like OS virtual memory paging):
  KV cache divided into fixed-size "pages" (16 tokens = ~2KB each)
  Request 1 gets pages 0,1,2... as needed (no pre-allocation)
  Request 2 gets pages 40,41,42... from available pool
  Fragmentation near zero → 3-5× more concurrent requests
```

---

## Optimization Decision Guide

```
Model type?
│
├── Sklearn / GBM / XGBoost
│   └── Export to ONNX → quantize → ORT serving
│       Speedup: 3-7×, no accuracy loss
│
├── Small neural network (MLP, small transformer, <1B params)
│   ├── CPU serving: PyTorch → ONNX INT8
│   │   Speedup: 5-10×, <0.1% AUC drop
│   └── GPU serving: TorchScript + TensorRT
│       Speedup: 10-20×
│
└── LLM (>1B params, text generation)
    ├── Throughput critical → vLLM (PagedAttention + continuous batching)
    ├── Cost critical → AWQ/GPTQ 4-bit quantization
    └── Edge/CPU → llama.cpp (GGUF format, 4-bit)

Quantization type?
  Latency requirement relaxed (< 50ms): dynamic quantization (quick, no calibration)
  Latency critical (< 5ms):             static quantization or ONNX INT8 (needs calibration)
  Acceptable accuracy drop:             < 0.1% AUC / < 1 BLEU point for NLP
```

---

## LLM Quantization (for reference)

```python
# AWQ 4-bit quantization (best quality/size tradeoff for LLMs)
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model = AutoAWQForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Calibration: AWQ finds important weights and quantizes conservatively
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_samples)

model.save_quantized("mistral-7b-awq-4bit")
# FP16: 14GB → AWQ 4-bit: 3.8GB (4× compression)
# Throughput: ~2× vs FP16 on same GPU
# Quality: perplexity within 1% of FP16
```

---

## Gotchas

**Always benchmark AFTER warmup.** First inference on GPU/ONNX session hits JIT compilation, memory allocation, CUDA context init. Always warmup ≥20 iterations before measuring. Otherwise you're measuring initialization, not inference.

**Quantization accuracy validation is non-negotiable.** INT8 quantization can hurt accuracy on edge cases: very large values, sparse activations. Always measure AUC/BLEU/accuracy on your test set after quantization. Don't ship if accuracy drop > 0.1%.

**Batch size matters for throughput, not latency.** Single-item latency (online serving) improves with ONNX/INT8. Throughput (batch serving) improves dramatically with larger batches. Measure both: p95 for SLA, throughput for cost.

**ONNX opset version compatibility.** Export with the highest opset your runtime supports. ONNX Runtime 1.17 supports up to opset 18. Exporting opset 19 with ORT 1.16 → runtime error.

**PagedAttention memory math.** KV cache per token for Mistral-7B: 2 (K+V) × 32 layers × 8 heads × 128 head_dim × 2 bytes (FP16) = 131KB per token. 4096 token context = 512MB per request. This is why LLM serving is memory-bound, not compute-bound.

---

## Interview Q&A

**Q: Walk me through how you'd optimize a PyTorch model for production serving.**
A: (1) Baseline: measure p50/p95 latency and throughput with representative inputs. (2) ONNX export: `torch.onnx.export()` with dynamic axes, verify outputs match within floating point tolerance. (3) ONNX Runtime: typically 3-4× speedup over PyTorch on CPU without any accuracy loss. (4) Quantization: dynamic INT8 for quick wins, static INT8 with calibration data for maximum speedup. (5) Validate: measure AUC/accuracy after each step — reject if drop > 0.1%. (6) Serve with FastAPI + ORT session configured with optimal thread counts.

**Q: What is PagedAttention and why does it matter?**
A: Standard LLM serving pre-allocates a contiguous KV cache block for each request sized for max sequence length. If max_seq_len=4096 but the actual request uses 200 tokens, 95% of the allocation is wasted — fragmentation means you can serve far fewer concurrent requests than the GPU memory would allow. PagedAttention stores KV cache in small fixed-size pages (like OS virtual memory paging). Pages are allocated on demand and can be non-contiguous. Result: near-zero fragmentation, 3-5× more concurrent requests, same GPU, same latency — massive throughput improvement at the same cost.

**Q: What is the tradeoff of INT8 quantization?**
A: INT8 uses 1 byte per weight vs FP32 (4 bytes) or FP16 (2 bytes). Benefits: 2-4× size reduction, 2-7× speedup (faster memory bandwidth, INT8 multiply-accumulate is faster than FP ops on modern CPUs). Cost: 8-bit representation has 256 values vs FP32's ~16M — small rounding errors accumulate. In practice for most models: AUC drop < 0.1%, which is acceptable. For models with sparse activations or large dynamic range, accuracy can drop more — always validate.

---

## Connections

- **Model registry:** `8.mlops/08_model_registry_end_to_end.md` — track optimized model versions
- **Serving and inference reference:** `8.mlops/02_serving_and_inference.md` — vLLM, FastAPI details
- **Monitoring:** `8.mlops/09_monitoring_end_to_end.md` — monitor latency after deployment

## Key Takeaway

Optimization pipeline: **PyTorch → ONNX → INT8** gives 5-10× speedup with <0.1% accuracy loss on CPU. Always measure baseline first, validate accuracy after each step, never ship without benchmarking. For LLMs: **vLLM** (PagedAttention + continuous batching) gives 3-5× throughput improvement by eliminating KV cache fragmentation. Standard path: ONNX INT8 for sklearn/small neural nets; vLLM + AWQ 4-bit for LLMs. Rule: measure → optimize → validate → repeat.

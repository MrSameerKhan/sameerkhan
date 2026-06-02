import torch
import time

print("=" * 50)
print("Phase 4 — CUDA Availability")
print("=" * 50)
print("PyTorch version :", torch.__version__)
print("CUDA available  :", torch.cuda.is_available())
print("CUDA version    :", torch.version.cuda)
print("GPU name        :", torch.cuda.get_device_name(0))
print("VRAM (GB)       :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

print()
print("=" * 50)
print("Phase 5 — Tensor Ops on GPU")
print("=" * 50)
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y
print("Tensor device   :", z.device)
print("Shape           :", z.shape)
print("VRAM used (MB)  :", round(torch.cuda.memory_allocated() / 1e6, 1))

print()
print("=" * 50)
print("Phase 6 — CPU vs GPU Benchmark")
print("=" * 50)

def benchmark(device_name):
    device = torch.device(device_name)
    x = torch.randn(2048, 2048, device=device)
    y = torch.randn(2048, 2048, device=device)
    if device_name == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        z = x @ y
    if device_name == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"{device_name.upper():4s}: {elapsed:.3f}s for 100 iterations")

benchmark("cpu")
benchmark("cuda")

print()
print("=" * 50)
print("Phase 6 — HuggingFace Transformer on GPU")
print("=" * 50)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
device = torch.device("cuda")
model = model.to(device)

inputs = tokenizer("Testing GPU training", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print("Logits          :", outputs.logits)
print("Device          :", outputs.logits.device)
print()
print("ALL CHECKS PASSED — GPU is ready for training.")

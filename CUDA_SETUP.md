# CUDA Setup Plan — GTX 1650 Ti on Windows 11

**Machine:** Dell G3 3500 | Intel i7-10750H | 16GB RAM | GTX 1650 Ti 4GB  
**Current state:** Driver 512.72 (CUDA 11.6 max) | PyTorch 2.12.0+cpu  
**Goal:** PyTorch with CUDA 12.1 → verified GPU training

---

## Phase 1 — Verify Laptop Config

### 1.1 Confirm GPU is recognized by OS
```powershell
nvidia-smi
```
**Expect:** GPU name, driver version, CUDA version in table  
**If fails:** Device Manager → Display Adapters → check GTX 1650 Ti is listed without yellow warning icon. If warning: reinstall driver from scratch (DDU method, see Phase 2 fallback).

### 1.2 Confirm GPU is not disabled
```powershell
Get-PnpDevice -Class Display | Select-Object Status, FriendlyName
```
**Expect:** Status = OK for GeForce GTX 1650 Ti  
**If fails:** Right-click in Device Manager → Enable device

### 1.3 Check current driver version
```powershell
nvidia-smi --query-gpu=driver_version --format=csv,noheader
```
**Current:** 512.72  
**Need:** ≥ 527.41 for CUDA 12.x  
**If already ≥ 527:** Skip Phase 2, go to Phase 3.

---

## Phase 2 — Update NVIDIA Driver

### 2.1 Primary method — GeForce Experience (easiest)
1. Open **GeForce Experience** (search in Start)
2. Click **Drivers** tab
3. Click **Check for Updates**
4. Download → **Express Installation**
5. Reboot when prompted

**Verify after reboot:**
```powershell
nvidia-smi
```
Expect `Driver Version: 5xx.xx` and `CUDA Version: 12.x`

---

### 2.2 If GeForce Experience fails or not installed — Manual Driver
1. Go to: https://www.nvidia.com/Download/index.aspx
2. Select:
   - Product Type: **GeForce**
   - Series: **GeForce GTX 16 Series (Notebooks)**
   - Product: **GeForce GTX 1650 Ti**
   - OS: **Windows 11 64-bit**
   - Download Type: **Game Ready Driver**
3. Download `.exe` → Run → Express Installation → Reboot

---

### 2.3 If standard install fails — DDU Clean Install (nuclear option)
Use this if driver install errors, nvidia-smi still shows old version, or display glitches after install.

1. Download **Display Driver Uninstaller (DDU)** from guru3d.com
2. Boot into **Safe Mode** (Settings → Recovery → Advanced startup → Troubleshoot → Advanced → Startup Settings → Restart → press 4)
3. In Safe Mode: run DDU → select GPU → **Clean and restart**
4. After reboot (normal mode): install fresh driver from Step 2.2

---

## Phase 3 — Install PyTorch with CUDA 12.1

### 3.1 Uninstall CPU-only PyTorch
```powershell
conda activate sameerkhan
pip uninstall torch torchvision torchaudio -y
```

### 3.2 Install CUDA-enabled PyTorch
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
This downloads ~2.5GB. Use a stable connection.

---

### 3.3 If download times out or fails
Use conda instead:
```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3.4 If CUDA 12.1 build is incompatible — fall back to CUDA 11.8
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Requires driver ≥ 522.06. Works with driver versions 522–526.

---

## Phase 4 — Verify CUDA is Working

### 4.1 Quick sanity check
```python
python -c "
import torch
print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())
print('CUDA version    :', torch.version.cuda)
print('GPU name        :', torch.cuda.get_device_name(0))
print('VRAM (GB)       :', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
"
```
**Expect:**
```
PyTorch version : 2.x.x+cu121
CUDA available  : True
CUDA version    : 12.1
GPU name        : NVIDIA GeForce GTX 1650 Ti
VRAM (GB)       : 4.29
```

### 4.2 If CUDA available = False after correct driver + install

Check 1 — wrong PyTorch build installed:
```python
python -c "import torch; print(torch.__version__)"
# Must show +cu121 not +cpu
```

Check 2 — CUDA toolkit version mismatch:
```powershell
nvcc --version   # should show 12.1 (if CUDA toolkit installed separately)
```

Check 3 — environment issue, try fresh:
```powershell
conda deactivate
conda activate sameerkhan
python -c "import torch; print(torch.cuda.is_available())"
```

Check 4 — multiple Python installs conflict:
```powershell
where python
# Should point to conda env, not system Python
```

---

## Phase 5 — Smoke Test GPU Tensor Operations

```python
import torch

# Move tensor to GPU
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = x @ y  # matrix multiply on GPU

print("Tensor device:", z.device)          # should say cuda:0
print("Shape:", z.shape)
print("VRAM used (MB):", round(torch.cuda.memory_allocated() / 1e6, 1))
```

**If `RuntimeError: CUDA out of memory`:** Tensor too large. Reduce to 100x100 for 4GB VRAM — shouldn't happen at 1000x1000.  
**If `RuntimeError: no kernel image available`:** PyTorch build doesn't match GPU compute capability. GTX 1650 Ti is compute 7.5 — all cu118/cu121 builds support it.

---

## Phase 6 — Test Model Training on GPU

### 6.1 Minimal BERT fine-tune on GPU
Run the existing BERT session with GPU:

```python
# At top of any training script, add:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Move model and inputs to GPU:
model = model.to(device)
input_ids = input_ids.to(device)
labels = labels.to(device)
```

Check GPU utilization during training:
```powershell
# In a separate terminal, watch GPU usage:
nvidia-smi -l 1   # refresh every 1 second
```
**Expect:** GPU-Util jumps to 30–90% during forward/backward pass.

---

### 6.2 Benchmark CPU vs GPU speed

```python
import torch
import time

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
    print(f"{device_name.upper()}: {elapsed:.3f}s for 100 iterations")

benchmark("cpu")
benchmark("cuda")
```
**Expect:** GPU is 10–50x faster for matrix ops.

---

### 6.3 Test transformers library on GPU (HuggingFace)
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda")
model = model.to(device)

inputs = tokenizer("Testing GPU training", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

print("Logits:", outputs.logits)
print("Device:", outputs.logits.device)  # must be cuda:0
```

---

### 6.4 VRAM budget guide for GTX 1650 Ti (4GB)

| Task | VRAM needed | Fits? |
|------|-------------|-------|
| BERT-base inference | ~500MB | Yes |
| BERT-base fine-tuning (batch=8) | ~2GB | Yes |
| GPT-2 (117M) fine-tuning | ~1.5GB | Yes |
| DistilBERT fine-tuning (batch=16) | ~1.5GB | Yes |
| LLaMA-7B (4-bit QLoRA) | ~6GB | No — exceeds 4GB |
| LLaMA-7B (4-bit QLoRA, CPU offload) | ~3.5GB | Marginal |
| LLaMA-3.2-1B (4-bit QLoRA) | ~1.5GB | Yes |

For QLoRA on larger models: use `device_map="auto"` + `load_in_4bit=True` with bitsandbytes to offload layers to CPU RAM.

---

## Summary Checklist

- [ ] Phase 1: `nvidia-smi` shows GTX 1650 Ti
- [ ] Phase 2: Driver updated to ≥ 527.41, CUDA 12.x shown in `nvidia-smi`
- [ ] Phase 3: `pip install torch ... --index-url .../cu121` completed
- [ ] Phase 4: `torch.cuda.is_available()` returns `True`
- [ ] Phase 5: Tensor ops run on `cuda:0` without error
- [ ] Phase 6: Transformer model loads and runs forward pass on GPU
- [ ] Phase 6: `nvidia-smi -l 1` shows GPU-Util > 0% during training

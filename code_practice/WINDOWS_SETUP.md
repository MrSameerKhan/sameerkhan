# Running on Windows

All Python code is cross-platform. This file covers the two differences you'll hit on Windows.

---

## 1. Run commands — PowerShell equivalents

The `_details.md` files show bash-style commands. Use these PowerShell equivalents instead.

### Setting environment variables

```bash
# Bash (Mac/Linux) — shown in details files
export OPENAI_API_KEY="sk-..."
```

```powershell
# PowerShell (Windows) — use this instead
$env:OPENAI_API_KEY = "sk-..."
python script.py
```

### `KMP_DUPLICATE_LIB_OK=TRUE` prefix

```bash
# Bash — shown in some details files (Mac-only workaround)
KMP_DUPLICATE_LIB_OK=TRUE python script.py
```

**On Windows: just run `python script.py` directly.** This env var is a Mac OpenMP workaround not needed on Windows.

---

## 2. Device selection — automatic

Every script detects your hardware automatically:

```python
DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"
```

On Windows:
- `mps` is never available → skipped
- `cuda` → used if you have an NVIDIA GPU
- `cpu` → fallback (slower but works for all phases except 09/05 vLLM)

---

## 3. Per-phase notes

| Phase | Windows status | Notes |
|-------|---------------|-------|
| 05 Transformers | ✅ Full support | CPU runs fine for small models (opt-125m, MiniLM) |
| 06 LLMs Core | ✅ Full support | API-only, no GPU needed |
| 07 RAG | ✅ Full support | FAISS CPU works on Windows |
| 08 Agents | ✅ Full support | API-only for sessions 01-02; LangGraph runs on CPU |
| 09 LoRA (01) | ✅ Full support | Runs on CPU (slow) or CUDA (fast) |
| 09 QLoRA (02) | ⚠️ CUDA only | `bitsandbytes` needs CUDA; skips quantization on CPU |
| 09 Dataset (03) | ✅ Full support | API-only |
| 09 DPO (04) | ✅ Full support | Runs on CPU or CUDA |
| 09 vLLM (05) | ❌ Linux+CUDA only | Code is documentation; run on cloud GPU |
| 09 Monitoring (06) | ✅ Full support | API-only |
| 10 Document AI | ✅ Full support | LayoutLM + Donut run on CPU; ColPali falls back to CLIP |

---

## 4. Quick start for any session

```powershell
# Activate environment
conda activate sameerkhan

# Set API key (if session uses OpenAI)
$env:OPENAI_API_KEY = "sk-..."

# Run
cd code_practice\06_llms
python 01_prompt_engineering.py
```

No `KMP_DUPLICATE_LIB_OK`, no `export`. That's it.

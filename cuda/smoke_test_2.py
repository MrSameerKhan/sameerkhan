import os
import torch
from huggingface_hub import constants
import datasets
import transformers

print("=" * 50)
print("Cache Location Verification")
print("=" * 50)

checks = {
    "HF_HOME"           : constants.HF_HOME,
    "Datasets cache"    : str(datasets.config.HF_DATASETS_CACHE),
    "Transformers cache": transformers.utils.TRANSFORMERS_CACHE,
    "Torch hub cache"   : torch.hub.get_dir(),
}

all_pass = True
for name, path in checks.items():
    on_s = path.startswith("S:\\") or path.startswith("s:\\")
    status = "PASS" if on_s else "FAIL"
    if not on_s:
        all_pass = False
    print(f"  [{status}] {name}: {path}")

print()
if all_pass:
    print("ALL CACHES ON S: DRIVE — setup complete.")
else:
    print("SOME CACHES STILL ON C: — restart terminal and re-run.")

import torch

if torch.backends.mps.is_available():
    print("✅ MPS (Metal Performance Shaders) is available!")
    print("Using device:", torch.device("mps"))
else:
    print("❌ MPS is not available. PyTorch is using CPU.")


import torch
import time

device = torch.device("mps")

size = 10000
a = torch.rand(size, size, dtype=torch.float16, device=device)
b = torch.rand(size, size, dtype=torch.float16, device=device)

start = time.time()
c = torch.matmul(a, b)
torch.mps.synchronize()
end = time.time()

print("🚀 PyTorch M1 GPU Computation Time (float16):", end - start)


import torch
print(torch.__version__)


print(a.device)  # Should print: "mps"
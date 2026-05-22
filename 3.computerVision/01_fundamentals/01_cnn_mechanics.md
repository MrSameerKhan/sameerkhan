# 01 — CNN Mechanics

> **Quick Reference:** Output size: W_out = (W_in - K + 2P) / S + 1. For Same padding: P = (K-1)/2, W_out = W_in/S.

---

## 1. Why CNN Over Fully Connected

A 224×224×3 image has 150,528 pixels. A fully connected layer to 1024 units = 150,528 × 1024 = **154M parameters** for the first layer alone. This overfits catastrophically.

CNN solution: apply a small filter (e.g., 3×3×3) that slides across the image. Same 3×3×3 = **27 parameters** per filter, shared across all spatial positions. Scale to 64 filters: **1,728 parameters** — 100,000× fewer than FC.

Three properties that make CNN work:
- **Parameter sharing** — same filter used everywhere
- **Local connectivity** — each neuron sees only a K×K patch
- **Hierarchical features** — edges → textures/parts → objects

---

## 2. Convolution Operation

A filter W of shape (K, K, C_in) slides over the input with stride S, computing a dot product at each position:

```
output[i,j] = Σ_{k,l,c} input[i*S+k, j*S+l, c] * W[k, l, c] + bias
```

**What filters learn by layer:**
- Layer 1: Gabor-like edge detectors (horizontal, vertical, diagonal)
- Layer 2–3: Textures, simple shapes, color blobs
- Layer 4–5: Object parts, faces, wheels, etc.

**Output size formula:**
```
W_out = floor((W_in - K + 2P) / S) + 1

Example (224×224, K=3, P=1, S=1): (224 - 3 + 2) / 1 + 1 = 224  (same)
Example (224×224, K=3, P=0, S=2): (224 - 3) / 2 + 1 = 112  (half)
```

---

## 3. Stride

Stride controls how many pixels the filter moves per step.

| Stride | Effect | Use case |
|--------|--------|----------|
| S=1 | Output same spatial size (with P=1 for K=3) | Feature extraction |
| S=2 | Halves spatial dimensions | Downsampling (learnable) |

**Stride-2 conv vs Max Pooling for downsampling:**

| | Stride-2 Conv | MaxPool |
|---|---|---|
| Learnable | Yes | No |
| Preserves all info | Partially (learned) | Takes max, discards rest |
| Preferred in | Modern CNNs (ResNet, EfficientNet) | Older architectures (VGG) |
| Parameters | +K²·C_in·C_out | 0 |

---

## 4. Padding

| Type | Formula | Effect |
|------|---------|--------|
| Valid (no padding) | P=0 | Output shrinks each layer |
| Same padding | P=(K-1)/2 | Output = input size (at S=1) |

**Rule:** For K=3, P=1. For K=5, P=2. For K=7, P=3. Only odd K allows integer P for same padding.

---

## 5. Feature Maps and Channels

For a conv layer with C_in input channels, C_out filters of size K×K:
- **Parameters:** K × K × C_in × C_out (+ C_out biases, often disabled with BN)
- **Example:** 3×3×3×64 = 1,728 params (first conv of a small CNN on RGB input)

**Feature hierarchy:**
```
Input:  3 channels (RGB)
Layer 1 (stride 1): 64 feature maps — edges, colors, simple textures
Layer 2 (stride 2): 128 feature maps, 112×112 — corners, curves
Layer 3 (stride 2): 256 feature maps, 56×56 — parts (eyes, wheels)
Layer 4 (stride 2): 512 feature maps, 28×28 — object-level features
Layer 5 (stride 2): 512 feature maps, 14×14 — high-level representations
```

### 1×1 Convolution

A 1×1 conv is a linear projection across channels — applies C_in → C_out transformation at each spatial location independently.

Uses:
- **Channel reduction:** 512 → 64 (bottleneck in ResNet)
- **Non-linearity injection:** apply ReLU after 1×1 to add capacity
- **Cross-channel mixing:** cheapest way to mix information across channels
- Parameters: 1 × 1 × C_in × C_out (no spatial info used)

---

## 6. Pooling

### Max Pooling

2×2 max pool with stride 2: take maximum value in each 2×2 window.

```
Input 4×4:          After 2×2 MaxPool (stride 2):
1  3  2  4          3  4
5  6  1  2    →     6  4
7  8  3  1          8  4
2  1  4  3          2  4
```

Properties: reduces spatial size by 2×, provides local translation invariance, discards 75% of activations (takes max).

### Average Pooling

Takes mean instead of max. Used for: feature visualization (less sharp), sometimes in lightweight models.

### Global Average Pooling (GAP)

Reduces each feature map to a single scalar (spatial mean):

```python
gap = nn.AdaptiveAvgPool2d(1)  # output is (B, C, 1, 1)
# then flatten: x = x.view(x.size(0), -1)  → (B, C)
```

GAP vs Flatten before classifier:
- **GAP:** H×W×C → C (tiny FC head, good for transfer learning)
- **Flatten:** H×W×C → huge FC vector (overfits, not transferable across image sizes)

Modern CNNs (ResNet, EfficientNet) always use GAP before the final FC.

---

## 7. ReLU and Batch Normalization

### ReLU

```
ReLU(x) = max(0, x)
```

- Kills negative activations (sparsity)
- Dying ReLU: if a neuron's pre-activation is always negative, gradient = 0 forever

Variants:
| Activation | Formula | Use |
|-----------|---------|-----|
| ReLU | max(0,x) | Default, fast |
| Leaky ReLU | max(0.01x, x) | Fixes dying ReLU |
| GELU | x·Φ(x) | Transformers, modern CNNs |
| SiLU/Swish | x·σ(x) | EfficientNet, MobileNet v3 |

### Batch Normalization in CNNs

Normalizes each channel across (N, H, W):

```
For channel c: μ_c = mean over all (batch, H, W) positions
               σ_c = std over all (batch, H, W) positions
               x_norm = (x - μ_c) / (σ_c + ε)
               output = γ_c · x_norm + β_c  (learnable scale and shift)
```

Benefits: stabilizes training, allows higher LR, reduces sensitivity to initialization.

**Critical:** `model.train()` uses batch statistics; `model.eval()` uses running mean/variance. Always call `model.eval()` before inference.

### Standard Modern CNN Block

```
Conv2d(in_ch, out_ch, K, stride, padding, bias=False)
→ BatchNorm2d(out_ch)
→ ReLU(inplace=True)
[→ MaxPool2d(2, 2)]   ← only when downsampling
```

---

## 8. Receptive Field

The receptive field (RF) is the region of the input image that influences one output neuron.

```
Single 3×3 conv:   RF = 3
Two 3×3 convs:     RF = 5  (3 + 2)
Three 3×3 convs:   RF = 7  (3 + 2 + 2)
General:           RF_L = 1 + L × (K - 1)   (stride 1, no dilation)
With stride-2:     RF grows much faster — e.g., stride-2 every 2 layers:
                   after 4 layers → RF covers 28×28 input region
```

**Why it matters:** Small RF → local features (edges, textures). Large RF → global features (object shape, scene). Dilated convolutions artificially increase RF without more parameters.

### Dilated (Atrous) Convolution

Insert (r−1) zeros between kernel elements. Same number of parameters, much larger receptive field.

```
Regular 3×3 (dilation=1):  looks at 3×3 patch — 9 parameters
Dilated 3×3 (dilation=2):  looks at 5×5 patch using same 9 parameters
Dilated 3×3 (dilation=4):  looks at 9×9 patch using same 9 parameters

Effective RF with dilation r:  RF = 1 + 2·r·(K - 1)
```

```python
conv_dilated = nn.Conv2d(128, 128, kernel_size=3, padding=2, dilation=2)
# RF = 1 + 2·2·(3-1) = 9, but only 9 weights per channel pair
```

**Where used:** segmentation (DeepLab series — ASPP module stacks rates 6/12/18); audio (WaveNet stacks rates 1/2/4/8/...). **Gotcha:** stacked equal-rate dilations create "gridding artifacts" (checkerboard patterns); mix rates to cover all positions evenly.

---

## 9. Depthwise Separable Convolution (MobileNet)

Standard conv: K×K×C_in filter applied to all channels simultaneously. Depthwise separable: split into two steps.

```
Step 1 — Depthwise:  K×K×1 filter per channel (each channel independently)
  Params: K × K × C_in

Step 2 — Pointwise:  1×1×C_in filter to combine channels
  Params: C_in × C_out

Total params: K²·C_in + C_in·C_out   vs standard: K²·C_in·C_out

Speed gain: ~8-9× fewer operations for K=3 (used in MobileNet)
```

---

## 10. When to Use What

| Situation | Choice | Reason |
|-----------|--------|--------|
| Standard feature extraction | 3×3 conv, stride 1, same padding | Universal — used in every modern CNN |
| Downsampling | Stride-2 conv OR MaxPool | Stride-2 is learnable; MaxPool is standard |
| Channel reduction | 1×1 conv | Cheap cross-channel mixing |
| Edge/mobile deployment | Depthwise separable | 8-9× fewer FLOPs |
| Classification head | Global Average Pool → FC | Fewer params than flattening |
| Very deep network | BatchNorm after every conv | Stabilizes training |

---

## 11. Gotchas

**Never forget: convolution output size depends on K, P, S — check formula.** Mismatched spatial dims are the most common CNN shape error. Always verify with a forward pass on dummy data.

**BatchNorm behavior differs train vs eval.** `model.train()` uses batch statistics. `model.eval()` uses running mean/variance. Forgetting `.eval()` during inference → noisy/wrong predictions. Always call `model.eval()` before inference.

**Padding=1 for K=3, padding=2 for K=5 — memorize this.** Same padding formula: P = (K-1)/2. For odd K only. Even kernel sizes avoided precisely because of this.

**Feature maps after ReLU can be all zeros (dying ReLU).** If you see NaN loss or zero activations: check initialization (use He init for ReLU), check LR isn't too high. Switch to Leaky ReLU as a diagnostic.

**GAP vs Flatten before classifier.** Flatten: H×W×C → H·W·C dimensional vector → huge FC layer → overfits easily. GAP: H×W×C → C → small FC layer → much better for transfer learning.

**Parameter count grows with depth AND width.** Doubling channels (width) → 4× parameters in conv layers. Depth is cheaper per-layer than width.

---

## 12. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Shape mismatch error | Wrong padding/stride | Verify formula; use `padding='same'` in TF |
| Loss NaN from start | Bad init or too high LR | He init; reduce LR; check for zero inputs |
| All feature maps zero after 3 layers | Dying ReLU | Use Leaky ReLU; lower LR; check BN |
| Train loss drops, val doesn't | Overfitting | More augmentation; add Dropout; more BatchNorm |
| Wrong predictions in eval | Forgot model.eval() | Always call model.eval() before inference |
| Model too large for device | Too many channels | Use depthwise separable; reduce channels |
| Output size doesn't match expected | Stride/padding config | Print shapes at each layer |

---

## 13. Code Reference

```python
import torch
import torch.nn as nn

# Basic CNN block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

# Verify output shapes
model = ConvBlock(3, 64)
x = torch.randn(4, 3, 224, 224)   # batch=4, channels=3, H=224, W=224
out = model(x)
print(out.shape)   # torch.Size([4, 64, 224, 224])

# Strided conv for downsampling (replaces MaxPool)
downsample = ConvBlock(64, 128, stride=2, padding=1)
out2 = downsample(out)
print(out2.shape)  # torch.Size([4, 128, 112, 112])

# Depthwise separable conv
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)  # per-channel
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)                           # 1×1 mix
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# Parameter count
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

conv_std = ConvBlock(64, 128, 3, padding=1)
conv_dws = DepthwiseSeparable(64, 128)
print(f"Standard conv: {count_params(conv_std):,}")   # 73,728
print(f"Depthwise sep: {count_params(conv_dws):,}")   # 8,576 = 8.6× fewer
```

---

## 14. Interview Q&A (Senior Level)

**Q: Why does a CNN have translational equivariance, not invariance?**

A: CNNs are equivariant to translation — if the input shifts by (Δx, S), the feature maps shift by (Δx/S, Δy/S) where S is the cumulative stride. The pattern is detected at a different spatial location. True invariance (same output regardless of position) comes from global average pooling, which collapses spatial dimensions entirely. MaxPool provides local invariance within a pooling window. This distinction matters for detection tasks (need equivariance to locate objects) vs classification (can afford full invariance via GAP).

**Q: Why use BatchNorm before or after ReLU — does order matter?**

A: Original ResNet uses Conv → BN → ReLU (post-activation). Pre-activation ResNet (He et al. 2016, ResNet v2) uses BN → ReLU → Conv and shows slightly better performance for very deep networks. The argument for pre-activation: BN before ReLU normalizes inputs to the nonlinearity, ensuring no extreme values. Practically, for most production use cases (ResNet-50 depth), the difference is minor. Modern architectures (EfficientNet, ConvNeXt) have moved toward LayerNorm and GELU — the canonical "best" order is less settled now.

**Q: What's the difference between two 3×3 convs twice and one 5×5?**

A: Same receptive field (5×5), but two 3×3 convs have 2×(3×3×C) = 18C parameters vs 5×5×C = 25C for one 5×5. Two 3×3s also have two ReLU nonlinearities — more expressive. This is why VGG replaced larger kernels with stacked 3×3s, and essentially all modern CNNs use 3×3 as the standard building block. The tradeoff: two ops vs one op (speed), but modern hardware handles this well.

**Q: What happens to gradient flow in a very deep CNN without skip connections?**

A: Gradients vanish. Each layer multiplies by its local gradient — even if each is slightly < 1 (e.g., 0.9), after 100 layers: 0.9^100 = 0.00003. The early layers receive near-zero gradients and barely update. This is the degradation problem ResNet solved. Note: this is distinct from the vanishing gradient in RNNs (across time steps) — here it's across depth. BatchNorm helps by keeping activations normalized, but deep networks (>30 layers) still need skip connections for reliable training.

---

## 15. Connections

| This file | Links to | Why |
|-----------|---------|-----|
| CNN used in DL fundamentals | `../../2.deep learning/02_architectures/02_cnn.md` | Full architecture file with ResNet, transfer learning |
| BatchNorm vs LayerNorm | `../../2.deep learning/01_fundamentals/03_training_stability.md` | Normalization comparison |
| He initialization for ReLU | `../../2.deep learning/01_fundamentals/03_training_stability.md` | Weight init strategies |
| CNN architectures evolution | `02_cnn_architectures.md` | LeNet → ConvNeXt |
| Vision Transformer (deep dive) | `04_vision_transformer_deep.md` | ViT, Swin, DeiT, hierarchical patch attention |
| CNN for detection/segmentation | `../02_applications/02_object_detection.md` | CNN backbone for downstream tasks |
| Self-supervised vision (DINOv2, MAE, I-JEPA) | `../02_applications/05_self_supervised_vision.md` | Modern CV pretraining |

---

## Key Takeaway

```
The CNN pipeline: Conv (detect local patterns) → BN (stabilize) → ReLU (nonlinearity)
→ Pooling/Stride (downsample) → Repeat → GAP → FC → Softmax

Three properties that make CNN work:
  Parameter sharing (same filter everywhere)
  Local connectivity (see small patches)
  Hierarchical features (edges → parts → objects)

Output size formula (memorize): (W - K + 2P) / S + 1
  For Same: P=(K-1)/2, W_out = W/S

Parameter count: K×K×C_in×C_out per conv layer
  1×1 conv is the cheapest channel transformation
  Depthwise separable = 8-9× fewer FLOPs than standard conv
```

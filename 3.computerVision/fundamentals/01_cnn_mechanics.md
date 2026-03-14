# 01 — CNN Mechanics

## Quick Reference

| Operation | Formula | Output Size |
|-----------|---------|------------|
| Convolution | W_out = (W_in − K + 2P) / S + 1 | Depends on K, P, S |
| Same padding | P = (K−1)/2 | W_out = W_in / S |
| Valid padding | P = 0 | W_out = W_in − K + 1 |
| Max Pooling | - | W_out = W_in / pool_size |
| Params per conv layer | K×K×C_in×C_out + C_out (bias) | - |

**W=width, K=kernel size, P=padding, S=stride, C=channels**

---

## 1. Why CNN Over Fully Connected?

### The Fully Connected Problem
```
ImageNet image: 224×224×3 = 150,528 pixels
FC layer with 1000 neurons: 150,528 × 1000 = 150M parameters

Problems:
  1. Too many parameters → overfits immediately
  2. No spatial awareness → pixel (10,10) and pixel (200,200) treated equally
  3. No translation invariance → cat in top-left ≠ cat in bottom-right
```

### What CNN Solves
```
Parameter sharing:  one filter (3×3×3 = 27 params) reused across entire image
Local connectivity: each neuron sees a small local patch, not all pixels
Translation invariance: same filter detects the pattern wherever it appears
Hierarchical features: early layers → edges, later layers → shapes → objects

Example:
  FC:  150M parameters for first layer alone
  CNN: 64 filters × 3×3 × 3 channels = 1,728 parameters → same output depth
```

**The core insight:** natural images have local structure. A filter that detects a vertical edge is useful everywhere — not just at one location.

---

## 2. Convolution Operation

### What Happens
A filter (kernel) slides across the input, computing a dot product at each position. The result is a feature map.

```
Input patch × filter = one output pixel

      Input (4×4):          Filter (2×2):
      [1  2  3  4]          [1  0]
      [5  6  7  8]          [0  1]
      [9  10 11 12]
      [13 14 15 16]

Position (0,0): [1×1 + 2×0 + 5×0 + 6×1] = 7
Position (0,1): [2×1 + 3×0 + 6×0 + 7×1] = 9
Position (1,0): [5×1 + 6×0 + 9×0 + 10×1] = 15
...

Feature Map (3×3):
      [7   9  11]
      [15  17 19]
      [23  25 27]
```

**What filters learn:**
- Edge detectors (horizontal, vertical, diagonal)
- Texture detectors (smooth, rough, periodic)
- Color detectors (red channel, green channel)
- Complex part detectors in deeper layers (eyes, wheels, corners)

### Output Size Formula
```
W_out = (W_in − K + 2P) / S + 1
H_out = (H_in − K + 2P) / S + 1

Example: 32×32 input, K=3, P=1, S=1
  W_out = (32 − 3 + 2) / 1 + 1 = 32   (same size → that's Same padding)

Example: 32×32 input, K=3, P=0, S=1
  W_out = (32 − 3 + 0) / 1 + 1 = 30   (shrinks → Valid padding)
```

---

## 3. Stride

Controls how many pixels the filter moves at each step.

```
Stride = 1: filter moves 1 pixel at a time → large output
Stride = 2: filter moves 2 pixels → output ≈ halved (downsampling)

Input 8×8, K=3, S=1: W_out = (8−3)/1 + 1 = 6
Input 8×8, K=3, S=2: W_out = (8−3)/2 + 1 = 3.5 ≈ 3 (floor)
```

**Stride 2 vs Max Pooling for downsampling:**
```
Modern CNNs (ResNet, EfficientNet) use stride-2 conv instead of pooling
  Advantage: learned downsampling (not hand-designed max operation)
  Disadvantage: more parameters than pooling (but usually worth it)
```

---

## 4. Padding

Why: repeated convolution without padding shrinks feature maps until they vanish.

```
Valid (P=0):   output smaller than input → feature maps shrink each layer
Same (P=(K-1)/2): output same size as input → required for very deep networks

For K=3: P=1 → same size
For K=5: P=2 → same size
For K=7: P=3 → same size
```

**Edge pixels:** without padding, corners and edges contribute to fewer output pixels → features at image edges under-represented. Padding fixes this.

```python
# PyTorch
conv_valid = nn.Conv2d(3, 64, kernel_size=3, padding=0)   # shrinks by 2 each dim
conv_same  = nn.Conv2d(3, 64, kernel_size=3, padding=1)   # preserves size
```

---

## 5. Feature Maps and Channels

### Channel Dimension
```
RGB input:     3 channels (R, G, B) — each a 2D array of pixel values
After conv layer with 64 filters: output has 64 channels (feature maps)

Convolution across all channels:
  Input:  H × W × C_in
  Filter: K × K × C_in      (depth must match input channels — single filter)
  Output: H' × W' × 1        (one scalar per spatial position)
  With 64 filters: H' × W' × 64
```

### Parameter Calculation
```
Conv2d(in_channels=3, out_channels=64, kernel_size=3):
  Weights: 3 × 3 × 3 × 64 = 1,728
  Biases:  64
  Total:   1,792 parameters

Compare to FC on 224×224 image with 64 neurons:
  224 × 224 × 3 × 64 = 9,633,792 parameters → 5000× more
```

### Feature Hierarchy
```
Layer 1 (7×7 conv): edges — horizontal, vertical, diagonal
Layer 2-3:          textures — stripes, grids, gradients
Layer 4-5:          parts — eyes, wheels, door handles
Layer 6+:           objects — faces, cars, dogs

This hierarchy emerges from training — not hand-designed
```

### 1×1 Convolution
```
K=1 conv applies an independent linear combination across channels at each spatial location.

Uses:
  1. Channel reduction (bottleneck): 256 → 64 channels before expensive 3×3 conv → fewer params
  2. Channel expansion: 64 → 256 channels after bottleneck
  3. Adds nonlinearity (with ReLU) without changing spatial dimensions
  4. Point-wise cross-channel interaction

Standard in: ResNet bottleneck, Inception, MobileNet depthwise separable conv
```

---

## 6. Pooling

### Max Pooling
```
Select maximum value in each pooling window.

Input (4×4):          2×2 MaxPool → Output (2×2):
[1  3  2  4]          [3  4]
[5  6  1  2]    →     [6  4]
[3  2  4  1]
[1  3  2  4]

Top-left  2×2: max(1,3,5,6)=6    wait — top-left: max(1,3,5,6)... let me redo
Top-left  2×2: max(1,3,5,6)=6 → top-left output
Top-right 2×2: max(2,4,1,2)=4 → top-right output
...
```

**Why max pooling:** retains the strongest feature activations. A feature detected anywhere in the window passes through → local translation invariance.

### Average Pooling
Takes mean instead of max. Used less in CNNs (max pooling usually better), but:
- Global Average Pooling (GAP): collapses entire spatial dim to 1×1 per channel → used before final classifier

```python
# GAP: used in ResNet, EfficientNet instead of large FC layer
gap = nn.AdaptiveAvgPool2d(1)  # H×W → 1×1, keeps channels
# 512×7×7 feature map → 512×1×1 → flatten → 512 → FC → n_classes
```

### Pooling vs Stride
```
Max Pool: non-learnable, deterministic — takes max
Stride-2 Conv: learnable — learns what to keep

Modern trend: stride-2 conv replacing max pool
  EfficientNet, ViT: no max pooling
  Classic (LeNet, VGG, older ResNet): max pooling
```

---

## 7. ReLU and Batch Normalization

### ReLU in CNN Context
```
ReLU(x) = max(0, x)

Applied elementwise to each feature map activation after convolution.

Why: nonlinearity — without it, stacked conv layers = one linear transform
Why ReLU (not sigmoid/tanh):
  1. No vanishing gradient for positive activations (gradient = 1)
  2. Sparse activations — many zeros → efficient
  3. Fast to compute

Variants:
  Leaky ReLU: max(0.01x, x) — prevents dying ReLU for negative inputs
  GELU:       x · Φ(x)      — used in transformers and modern CNNs (smooth)
  SiLU/Swish: x · σ(x)      — used in EfficientNet
```

**Dying ReLU problem:**
```
If a neuron's pre-activation is always negative → output always 0 → gradient always 0
→ neuron "dies" — never updates again
Cause: large LR, bad initialization → weights driven very negative

Fix: Leaky ReLU, ELU, careful initialization (He init for ReLU), proper LR
```

### Batch Normalization in CNN
```
Normalizes feature map activations across the batch dimension.

For CNN: normalize across (N, H, W), keeping C separate
  i.e., for each channel, compute statistics over batch + spatial dims

BatchNorm forward:
  μ_c = mean over (N, H, W) for channel c
  σ²_c = variance over (N, H, W) for channel c
  x̂ = (x − μ_c) / √(σ²_c + ε)
  y = γ · x̂ + β         ← γ, β are learned per-channel

At inference: use running mean/variance (accumulated during training)
```

**Benefits:**
```
1. Reduces internal covariate shift → allows higher LR
2. Acts as mild regularizer (adds noise from batch statistics)
3. Reduces sensitivity to weight initialization
4. Enables much deeper networks to train stably
```

### Standard Modern CNN Block
```
Input
  → Conv2d (K×K, C_out, stride=S, padding=P)
  → BatchNorm2d(C_out)
  → ReLU (or SiLU/GELU in modern nets)
  [→ MaxPool2d (if downsampling here, not via stride)]

This order (Conv → BN → ReLU) is standard in ResNet.
Pre-activation ResNet uses (BN → ReLU → Conv) — slightly better in theory.
```

---

## 8. Receptive Field

How large a region of the input does one output neuron "see"?

```
After 1 conv layer (K=3, S=1): receptive field = 3×3
After 2 conv layers (K=3, S=1): RF = 5×5
After 3 conv layers (K=3, S=1): RF = 7×7

With stride-2 (or pooling) between layers, RF grows faster:
  Layer 1 (K=3, S=1): RF = 3×3
  MaxPool (2×2):       RF = 4×4
  Layer 2 (K=3, S=1): RF = 8×8
  MaxPool (2×2):       RF = 10×10

Deep CNN (e.g., ResNet-50, 224×224 input): final layer neurons see entire image
```

**Why this matters:**
- Small RF → local features (edges, textures)
- Large RF → global features (object shape, scene)
- Dilated convolutions artificially increase RF without more parameters

---

## 9. Depthwise Separable Convolution (MobileNet)

Standard conv: K×K×C_in filter applied to all channels simultaneously
Depthwise separable: split into two steps

```
Step 1 — Depthwise: K×K×1 filter per channel (each channel independently)
  Params: K × K × C_in

Step 2 — Pointwise: 1×1×C_in filter to combine channels
  Params: C_in × C_out

Total params: K²·C_in + C_in·C_out  vs  standard: K²·C_in·C_out

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

**Never forget: convolution output size depends on K, P, S — check formula.**
Mismatched spatial dims are the most common CNN shape error. Always verify with a forward pass on dummy data.

**BatchNorm behavior differs train vs eval.**
`model.train()` uses batch statistics. `model.eval()` uses running mean/variance. Forgetting `.eval()` during inference → noisy/wrong predictions. Always call `model.eval()` before inference.

**Padding=1 for K=3, padding=2 for K=5 — memorize this.**
Same padding formula: P = (K-1)/2. For odd K only. Even kernel sizes avoided precisely because of this.

**Feature maps after ReLU can be all zeros (dying ReLU).**
If you see NaN loss or zero activations: check initialization (use He init for ReLU), check LR isn't too high. Switch to Leaky ReLU as a diagnostic.

**GAP vs Flatten before classifier.**
Flatten: H×W×C → H·W·C dimensional vector → huge FC layer → overfits easily.
GAP: H×W×C → C → small FC layer → much better for transfer learning.

**Parameter count grows with depth AND width.**
Doubling channels (width) → 4× parameters in conv layers. Depth is cheaper per-layer than width.

---

## 12. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Shape mismatch error | Wrong padding/stride | Verify with formula; use `padding='same'` in TF |
| Loss NaN from start | Bad init or too high LR | Use He init; reduce LR; check for zero inputs |
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
print(out.shape)   # → torch.Size([4, 64, 224, 224])

# Strided conv for downsampling (replaces MaxPool)
downsample = ConvBlock(64, 128, stride=2, padding=1)
out2 = downsample(out)
print(out2.shape)   # → torch.Size([4, 128, 112, 112])

# Depthwise separable conv
class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)   # per-channel
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)                            # 1×1 mix
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# Parameter count
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

conv_std = nn.Conv2d(64, 128, 3, padding=1)
conv_dws = DepthwiseSeparable(64, 128)
print(f"Standard conv: {count_params(conv_std):,}")     # 73,728
print(f"Depthwise sep: {count_params(conv_dws):,}")     # 8,576 → 8.6× fewer
```

---

## 14. Interview Q&A (Senior Level)

**Q: Why does a CNN have translational equivariance, not invariance?**
A: CNNs are equivariant to translation — if the input shifts by (Δx, Δy), the feature maps shift by (Δx/S, Δy/S) where S is the cumulative stride. The pattern is detected at a different spatial location. True invariance (same output regardless of position) comes from global average pooling, which collapses spatial dimensions entirely. Max pooling provides local invariance within a pooling window. This distinction matters for detection tasks (need equivariance to locate objects) vs classification (can afford full invariance via GAP).

**Q: Why use BatchNorm before or after ReLU — does order matter?**
A: Original ResNet uses Conv → BN → ReLU (post-activation). Pre-activation ResNet (He et al. 2016, ResNet v2) uses BN → ReLU → Conv and shows slightly better performance for very deep networks. The argument for pre-activation: BN before ReLU normalizes inputs to the nonlinearity, ensuring no extreme values. Practically, for most production use cases (ResNet-50 depth), the difference is minor. Modern architectures (EfficientNet, ConvNeXt) have moved toward LayerNorm and GELU — the canonical "best" order is less settled now.

**Q: What's the difference between 3×3 conv twice and one 5×5 conv?**
A: Same receptive field (5×5), but two 3×3 convs have 2×(3×3×C) = 18C parameters vs 5×5×C = 25C for one 5×5. Two 3×3s also have two ReLU nonlinearities → more expressive. This is why VGG replaced larger kernels with stacked 3×3s, and essentially all modern CNNs use 3×3 as the standard building block. The tradeoff: two ops vs one op (speed), but modern hardware handles this well.

**Q: What happens to gradient flow in a very deep CNN without skip connections?**
A: Gradients vanish. Each layer multiplies by its local gradient — even if each is slightly < 1 (e.g., 0.9), after 100 layers: 0.9^100 ≈ 0.00003. The early layers receive near-zero gradients and barely update. This is the degradation problem ResNet solved. Note: this is distinct from the vanishing gradient in RNNs (across time steps) — here it's across depth. BatchNorm helps by keeping activations normalized, but deep networks (>30 layers) still need skip connections for reliable training.

---

## 15. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| CNN used in DL fundamentals | `../../1.deep learning/architectures/02_cnn.md` | Full architecture file with ResNet, transfer learning |
| BatchNorm vs LayerNorm | `../../1.deep learning/fundamentals/03_training_stability.md` | Normalization comparison |
| He initialization for ReLU | `../../1.deep learning/fundamentals/03_training_stability.md` | Weight init strategies |
| CNN architectures evolution | `02_cnn_architectures.md` | LeNet → ConvNeXt |
| CNN for detection/segmentation | `../applications/02_object_detection.md` | CNN backbone for downstream tasks |

---

## Key Takeaway

**The CNN pipeline:** Conv (detect local patterns) → BN (stabilize) → ReLU (nonlinearity) → Pooling/Stride (downsample) → Repeat → GAP → FC → Softmax.

**Three properties that make CNN work:** parameter sharing (same filter everywhere), local connectivity (see small patches), hierarchical features (edges → parts → objects).

**Output size formula (memorize):** `(W − K + 2P) / S + 1`. For Same: P=(K-1)/2, W_out = W/S.

**Parameter count:** K×K×C_in×C_out per conv layer. 1×1 conv is the cheapest channel transformation.

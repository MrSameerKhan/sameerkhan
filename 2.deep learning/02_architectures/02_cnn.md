# 02 — CNN (Convolutional Neural Network)

## Quick Reference (30-sec scan)

- **Core idea:** learn local spatial patterns via shared weight filters — translation invariant
- **Convolution:** filter slides over input, computes dot product at each position
- **Receptive field:** grows with depth — early layers see edges, deep layers see objects
- **Key ops:** Conv → BatchNorm → ReLU → Pooling → repeat → Flatten → MLP head
- **Why efficient:** filter weights shared across all spatial positions — 3×3×64 filter = 576 params regardless of input size
- **Gotcha:** standard CNNs lose global context — each neuron only sees local region (fixed by attention in ViT)

---

## Why CNN Over MLP for Images

MLP on 224×224 RGB image:

```
Flatten = 224×224×3 = 150,528 inputs
First layer (1000 neurons): 150M parameters  → impractical
No spatial awareness: pixel (0,0) and pixel (223,223) treated identically
```

CNN on same image:

```
3×3 conv filter (64 filters): 3×3×3×64 = 1,728 parameters
Same filter applied everywhere: captures local patterns + translation invariant
224×224 image → still ~1728 parameters for this layer
```

**Inductive bias:** CNNs assume nearby pixels are related and patterns can appear anywhere in the image. This prior dramatically reduces the data needed to learn visual features.

---

## Core Operations

### Convolution

A filter (kernel) slides across the input, computing a dot product at each position.

```
Input patch (3×3):    Filter (3×3):    Output (1 value):
[1, 0, 1]             [1, 0, -1]
[0, 1, 0]         ×   [1, 0, -1]   = sum of element-wise products
[1, 0, 1]             [1, 0, -1]

= 1×1 + 0×0 + 1×(-1) + 0×1 + 1×0 + 0×(-1) + 1×1 + 0×0 + 1×(-1) = 0
```

This single filter slides across the entire image, producing one **feature map**.

Multiple filters → multiple feature maps (channels in output).

**Key parameters:** · **Kernel size:** 3×3 (most common), 5×5, 1×1 · **Stride:** how many pixels to move per step (stride=1 → dense, stride=2 → halves resolution) · **Padding:** same (output = input size), valid (output shrinks by kernel_size-1) · **Filters (out_channels):** how many feature maps to learn

**Output size formula:**

```python
output_size = floor((input_size + 2×padding - kernel_size) / stride) + 1

Example: input=224, kernel=3, padding=1, stride=1
output = (224 + 2 - 3) / 1 + 1 = 224   (same padding preserves size)
```

### Pooling

Reduces spatial dimensions, keeps dominant features.

```python
Max Pooling (2×2, stride=2):
[5, 2, 4]    [5, 4]
[1, 2, 4]  →          → takes max in each 2×2 region
[1, 0, 3, 6]
[2, 3, 1, 2]
```

- **Max pooling:** keeps strongest activation — preserves sharp features
- **Average pooling:** smooths activations — used in global average pooling at end
- **Global Average Pooling (GAP):** collapses entire feature map to single value per channel — replaces flatten → large FC layers in modern CNNs

### 1×1 Convolution

Looks useless but isn't — applies a linear combination across channels at each spatial position.

Uses: · Channel reduction (bottleneck in ResNet): 256 channels → 64 channels → 256 channels · Adding non-linearity without changing spatial size · Dimension matching in residual shortcuts

---

## Receptive Field — What Each Neuron Sees

The receptive field is the region of the input that influences a given neuron.

```
Layer 1 (3×3 conv): each neuron sees a 3×3 region
Layer 2 (3×3 conv): each neuron sees a 5×5 region of original input
Layer 3 (3×3 conv): each neuron sees a 7×7 region
...
Layer n:  sees (2n+1) × (2n+1) region

This is how CNNs build hierarchy: · Early layers: edges, corners, textures
· Middle layers: shapes, parts · Deep layers: objects, semantic concepts
```

Deeper network → larger receptive field → more context.

---

## Standard CNN Architecture Pattern

```
Input Image [H × W × C]
     ↓
[Conv → BN → ReLU] × N    # feature extraction block
     ↓
MaxPool or stride=2        # spatial downsampling
     ↓
[Conv → BN → ReLU] × N    # deeper features
     ↓
 ...repeat...
     ↓
Global Average Pooling     # collapse spatial dims
     ↓
Linear + Softmax           # classification head
```

---

## CNN Architectures Evolution

| Architecture | Year | Key Innovation |
|-------------|------|---------------|
| LeNet | 1998 | First practical CNN (MNIST) |
| AlexNet | 2012 | Deep CNN + ReLU + Dropout + GPU — started deep learning boom |
| VGG | 2014 | Very deep (16-19 layers), only 3×3 convs — simplicity wins |
| GoogLeNet/Inception | 2014 | Inception modules — parallel conv filters of different sizes |
| ResNet | 2015 | Residual connections — enabled 50-152 layer training |
| DenseNet | 2017 | Each layer connects to all previous layers |
| EfficientNet | 2019 | Compound scaling (width + depth + resolution together) |
| ConvNeXt | 2022 | CNN redesigned with ViT training recipe — matches Swin |
| ConvNeXt V2 | 2023 | Adds Fully-Convolutional MAE pretraining + GRN normalization |
| Vision Mamba (Vim) | 2024 | SSM-based vision backbone — O(n) compute on patches, competitive with ViT |
| MambaVision (NVIDIA) | 2024 | Hybrid Mamba + Self-Attention — SOTA accuracy/throughput on ImageNet |

**For interviews:** know AlexNet (historical), VGG (simplicity), ResNet (key innovation), EfficientNet (compound scaling), ConvNeXt (modern CNN baseline), and a one-line answer on Mamba-style vision backbones (efficient alternative to ViT).

---

## Convolution Variants (Efficiency)

| Variant | Idea | Where used |
|---------|------|-----------|
| Dilated (atrous) | Insert holes in kernel: receptive field = 1 + 2·dilation·(k-1) | Segmentation (DeepLab), audio (WaveNet) |
| Depthwise | One filter per input channel (no cross-channel mixing) | MobileNet, EfficientNet (always paired with pointwise) |
| Pointwise (1×1) | Mixes channels only, no spatial | Bottlenecks; "channel attention" in SE blocks |
| Depthwise-separable | Depthwise + Pointwise — ~8-9× fewer params than standard conv | MobileNet, Xception, EfficientNet |
| Grouped convolution | Split channels into G groups, conv within each | ResNeXt, ShuffleNet, GPU memory-friendly |
| Deformable conv | Kernel positions are learned, not fixed grid | Object detection (DCN, Deformable DETR) |

```python
# Depthwise-separable conv block (~9× fewer params than nn.Conv2d for 3×3)
class DepSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))
```

**Senior interview answer:** "If memory or latency is the binding constraint, I'd reach for depthwise-separable convolutions (MobileNet pattern) — typically 8-9× fewer parameters than full conv for the same receptive field, with minimal accuracy loss. For long-range receptive fields without spatial reduction (e.g., semantic segmentation), I'd use dilated convolutions instead of stacking convs."

---

## ResNet — The Most Important CNN

Introduced skip connections to solve degradation problem.

```
Residual Block:
Input x
  ├─────── identity shortcut ────────┐
  |                                  |
Conv → BN → ReLU                     |
  |                                  |
Conv → BN                            |
  |                                  |
  └──────────── + ───────────────────┘
  |
ReLU
  |
Output = F(x) + x
```

output = F(x) + x — if layers are useless, F(x) → 0 and output = x (identity).

Enabled networks with 50, 101, 152 layers. Standard backbone for most CV tasks.

**ResNet variants:** · ResNet-18/34: basic blocks (two 3×3 convs) · ResNet-50/101/152: bottleneck blocks (1×1 → 3×3 → 1×1) — more efficient

---

## Key Hyperparameters

| Hyperparameter | Effect | Typical Values |
|---------------|--------|---------------|
| Kernel size | Larger = more context per layer | 3×3 (default), 1×1, 7×7 (first layer) |
| Number of filters | More = richer features, more params | 32, 64, 128, 256, 512 |
| Stride | Controls spatial downsampling | 1 (preserve), 2 (halve) |
| Depth | More layers = larger receptive field | 18-152 (ResNet family) |
| Pooling type | Max (sharp), Avg (smooth), Global Avg | Global avg pooling at end |

---

## When to Use CNN

| Task | CNN? | Notes |
|------|------|-------|
| Image classification | Yes | Standard choice, ResNet/EfficientNet |
| Object detection | Yes | Backbone (ResNet) + detection head |
| Semantic segmentation | Yes | Encoder-decoder (U-Net, DeepLab) |
| Document image classification | Yes | ResNet or EfficientNet backbone |
| Document layout analysis | Yes | CNN + spatial features |
| Long-range image relationships | Partial | Use ViT or hybrid CNN+Transformer |
| Text sequences | No | Transformer |
| Tabular data | No | MLP |

---

## Gotchas

**1. CNNs are not globally aware.** Each neuron has a fixed receptive field. Without enough depth, two distant image regions can't interact. This is why ViT (global attention) sometimes beats CNNs on tasks requiring global context understanding.

**2. Pooling loses spatial information.** Max pooling takes the strongest activation but discards WHERE it was. This is why CNNs trained on classification fail at detection without architectural changes (replace pooling with strided convs or feature pyramids).

**3. Batch size matters more for BatchNorm in CNNs.** CNNs standardly use BatchNorm. Small batch sizes (< 8) make BN statistics unreliable. Use GroupNorm or SyncBN for very small batches.

**4. More filters ≠ always better.** Doubling filters doubles parameters AND memory. EfficientNet showed that scaling width, depth, and resolution together (compound scaling) is more efficient than scaling one alone.

**5. Data augmentation is not optional for CNNs.** CNNs have translation invariance via shared filters but NOT rotation, scale, or color invariance. These must come from augmentation (random crop, flip, color jitter). Without it, CNNs overfit heavily on small datasets.

---

## Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Feature maps all zero after few layers | Dying ReLU + bad init | Use He init; check LR |
| Overfit on training, poor val accuracy | Too many params, no augmentation | Add augmentation, dropout after GAP |
| BN behaves differently train vs eval | Forgot `model.eval()` | Always call eval() at inference |
| OOM during training | Feature maps too large | Add stride=2 earlier, reduce resolution |
| Model learns slowly | LR too low for SGD | Use LR range test, try 0.01-0.1 for SGD |
| Poor performance on small dataset | Training from scratch | Use pretrained backbone + fine-tune |

---

## Code Reference

```python
import torch.nn as nn

# Basic CNN Block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ResNet residual block
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)  # residual connection

# Using pretrained ResNet (most common approach)
import torchvision.models as models
backbone    = models.resnet50(weights='IMAGENET1K_V1')
# Replace classification head for custom task
num_classes = 5
backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)

# Fine-tune: freeze backbone, train only head
for param in backbone.parameters():
    param.requires_grad = False
for param in backbone.fc.parameters():
    param.requires_grad = True

# Unfreeze last block for better fine-tuning
for param in backbone.layer4.parameters():
    param.requires_grad = True

# EfficientNet (modern efficient baseline)
backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
```

---

## Interview Q&A

**Q: Why do CNNs use 3×3 filters instead of larger ones?**

Two 3×3 conv layers have the same receptive field as one 5×5 layer (both see 5×5 region), but with fewer parameters (2×3×3 = 18 vs 5×5 = 25) and an extra non-linearity between them. VGG popularized this — all conv filters are 3×3. The extra non-linearity makes the model more expressive per parameter. 1×1 convolutions are used separately for channel mixing without spatial interaction.

**Q: What is the difference between stride=2 and max pooling for downsampling?**

Both halve spatial resolution. Max pooling takes the maximum in each region — it's not learned and preserves the strongest activations. Stride=2 conv is learned — the network decides what to preserve. Modern architectures (ResNet variants, EfficientNet) increasingly prefer strided convolutions over pooling as they give the model more control. Global average pooling at the end remains standard.

**Q: How would you adapt a pretrained ResNet for document image classification?**

Standard transfer learning: freeze backbone weights (pretrained on ImageNet), replace the final FC layer with a new one matching your class count. Fine-tune in two stages: (1) train only the new head for a few epochs with high LR; (2) unfreeze the last 1-2 residual blocks and train with very low LR (1e-5). For documents specifically, also consider adjusting input resolution — documents often benefit from higher resolution (448×448 or 512×512) than the standard 224×224.

**Q: Why does CNN need data augmentation but MLP doesn't as much?**

CNNs have translation invariance (filters share weights spatially) but not rotation, scale, or illumination invariance. These must be learned from data. Without augmentation, a CNN trained on upright cats will fail on rotated cats. Data augmentation artificially creates this variation. MLPs have no spatial assumptions at all — they're equally poor at all geometric transformations — so augmentation helps less (the problem is the architecture, not the data).

---

## Connections

- Builds on: `../01_fundamentals/01_foundations.md` — convolution is a specialized linear layer
- Builds on: `../01_fundamentals/03_training_stability.md` — BatchNorm critical for CNN training
- Builds on: `../01_fundamentals/05_modern_components.md` — ResNet uses residual connections
- Leads to: `04_transformer.md` — ViT applies transformer directly to image patches, replacing CNN
- Leads to: CV domain — object detection (FPN, YOLO), segmentation (U-Net, DeepLab)

---

## Key Takeaway

```
CNN = learn local patterns via shared weight filters + translation invariant
3×3 conv: most efficient — 2 stacked = 5×5 receptive field with extra non-linearity
Hierarchy: edges → shapes → parts → objects as depth increases
ResNet:    standard backbone — residual connections enabled deep (50-152 layer) training
Use pretrained: always fine-tune from ImageNet weights, never train from scratch on small data
Limitation: local receptive field only — no global context (fixed by ViT/attention)
```

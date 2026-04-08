# 01 — MLP (Multi-Layer Perceptron)

---

## Quick Reference *(30-sec scan)*

- **What**: fully connected layers stacked — every neuron connects to every neuron in next layer
- **Formula per layer**: `a = f(Wx + b)` — linear transform → activation
- **Depth**: each layer learns increasingly abstract features
- **Problem**: doesn't scale to images (1000×1000 image → 1M params per neuron) or sequences
- **Use today**: classification head on top of CNN/Transformer, tabular data, embedding projection
- **Gotcha**: MLPs on raw images ignore spatial structure — pixels far apart treated same as adjacent

---

## What Is an MLP

A stack of fully connected (dense) layers where every input neuron connects to every output neuron.

```
Input (784)  →  Hidden1 (256)  →  Hidden2 (128)  →  Output (10)
  ↓                ↓                  ↓                 ↓
x vector        a1 = ReLU(W1x+b1)  a2 = ReLU(W2a1+b2)  ŷ = softmax(W3a2+b3)
```

Parameters per layer = `input_size × output_size + output_size (bias)`

Example: 784 → 256 → 128 → 10
```
Layer 1: 784×256 + 256  = 200,960
Layer 2: 256×128 + 128  =  32,896
Layer 3: 128×10  + 10   =   1,290
Total:                    235,146 parameters
```

---

## How It Works

### Forward Pass
```
z1 = W1 × x  + b1       # linear transform
a1 = ReLU(z1)            # activation
z2 = W2 × a1 + b2
a2 = ReLU(z2)
ŷ  = softmax(W3 × a2 + b3)   # output layer
```

### What Each Layer Learns
```
Raw input   → Layer 1: simple patterns (pixel combinations)
Layer 1 out → Layer 2: combinations of simple patterns
Layer 2 out → Layer 3: abstract concepts
```

Each layer is a feature extractor — later layers build on earlier ones.

### Universal Approximation Theorem
A single hidden layer MLP with enough neurons can approximate any continuous function. In practice: deeper (more layers) beats wider (more neurons per layer) for the same parameter count.

---

## Key Properties

| Property | Value |
|----------|-------|
| Connectivity | Every neuron → every neuron in next layer |
| Parameters | Grows quadratically with layer width |
| Spatial awareness | None — treats all inputs equally |
| Sequential awareness | None — no memory |
| Inductive bias | None — learns everything from scratch |

**No inductive bias = needs more data than CNN/Transformer for same task.**

---

## When to Use MLP

| Situation | Use MLP? | Why |
|-----------|---------|-----|
| Tabular / structured data | ✅ Yes | No spatial/sequential structure needed |
| Classification head (on CNN/ViT features) | ✅ Yes | Features already extracted |
| Embedding projection | ✅ Yes | Simple linear + nonlinear mapping |
| Raw image classification | ❌ No | CNN much more efficient |
| Sequence/text tasks | ❌ No | Transformer handles order |
| Very large inputs | ❌ No | Parameter count explodes |

---

## MLP vs CNN vs Transformer

| | MLP | CNN | Transformer |
|--|-----|-----|-------------|
| Input type | Tabular, embeddings | Images, spatial data | Sequences, patches |
| Spatial awareness | No | Yes (local) | Yes (global via attention) |
| Sequential awareness | No | No | Yes |
| Parameter efficiency | Low | High | Medium |
| Inductive bias | None | Translation invariance | None (needs positional embed) |
| Typical use | Head layer, tabular | Vision backbone | Language, vision (ViT) |

---

## Gotchas

**1. MLPs on images ignore spatial structure**
Flattening a 28×28 image to 784 loses all spatial information. A pixel in the top-left corner has no special relationship to its neighbor — the MLP treats them all equally. This is why CNNs are far more parameter-efficient for images.

**2. Parameter explosion with large inputs**
Input size 1000 → hidden 1000: already 1M parameters in one layer. For 224×224 RGB image (150K inputs) → 1000 neurons: 150M parameters in one layer alone. Completely impractical.

**3. Depth without residuals degrades**
Deep MLPs (>5–6 layers) suffer vanishing gradients without residual connections. Either use shallow MLPs or add skip connections.

**4. Universal approximation ≠ practical**
The theorem says one wide hidden layer is enough — but this requires exponentially many neurons. In practice, depth + regularization + proper architecture beats width.

---

## Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| MLP underfits on tabular data | Too few neurons/layers | Increase width or depth |
| MLP overfits on tabular data | Too many params for dataset size | Dropout, L2, reduce size |
| MLP performs poorly on images | Wrong architecture for task | Switch to CNN |
| Deep MLP not learning | Vanishing gradients | Add residual connections, use ReLU |
| Training loss ≠ val loss early | Learning rate too high | Reduce LR |

---

## Code Reference

```python
import torch.nn as nn

# Standard MLP for classification
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [nn.Linear(prev_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Usage
model = MLP(input_dim=784, hidden_dims=[256, 128], num_classes=10)

# MLP classification head on top of CNN features
class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.head(x)
```

---

## Interview Q&A

**Q: What is the universal approximation theorem and what does it NOT guarantee?**
> It states that a single hidden layer MLP with enough neurons can approximate any continuous function to arbitrary precision. It does NOT guarantee that: (1) we can find those weights via gradient descent, (2) the required number of neurons is practical, or (3) the network generalizes. It's a theoretical existence result, not a practical design guide.

**Q: Why do CNNs outperform MLPs on images despite MLPs being universal approximators?**
> MLPs have no inductive bias — they must learn that nearby pixels are related from scratch, requiring huge amounts of data. CNNs embed translation invariance and local connectivity as architectural priors — they share weights spatially (conv filters), so they need far fewer parameters to learn the same visual features. An MLP on MNIST works. An MLP on ImageNet is impractical.

**Q: Where do MLPs still appear in modern architectures?**
> Everywhere — as sub-components. The FFN (feed-forward network) inside every transformer block is a 2-layer MLP. Classification heads on ViT, BERT, ResNet are MLPs. Projection layers in CLIP, recommendation systems, and tabular models are MLPs. The MLP is not replaced — it's embedded inside larger architectures.

---

## Connections

- **Builds on**: `fundamentals/01_foundations.md` — neuron equation, activations, loss
- **Leads to**: `02_cnn.md` — CNN adds spatial structure to fix MLP's image limitation
- **Leads to**: `04_transformer.md` — FFN inside transformer is a 2-layer MLP
- **Relevant in**: every architecture — MLP appears as head/projection in CNN, ViT, BERT, GPT

---

## Key Takeaway

```
MLP = stack of Linear → Activation layers, fully connected
Good for: tabular data, classification heads, embedding projection
Bad for:  raw images (use CNN), sequences (use Transformer)
Lives inside: transformer FFN block, every model's output head
Limitation: no spatial/sequential awareness, parameter-heavy for large inputs
```
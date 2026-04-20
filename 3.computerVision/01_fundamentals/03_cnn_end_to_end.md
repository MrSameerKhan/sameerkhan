# 03 — CNN End-to-End: Forward Pass, Backprop, and Training Trace

## What This File Does

Complete numerical trace of a CNN: convolution arithmetic → activation → pooling → fully connected → softmax → cross-entropy loss → backprop → weight update. Uses a 2-layer toy CNN on a 4×4 grayscale image.

---

## Setup

```
Input: 4×4 grayscale image (1 channel)
Task:  binary classification (cat vs dog)

Architecture:
  Conv1: 1 filter, kernel 3×3, padding=0, stride=1  → 2×2 feature map
  ReLU
  Flatten → 4 neurons
  FC1:   4 → 2 (hidden)
  ReLU
  FC2:   2 → 1
  Sigmoid → probability

Input image X (4×4):
  [1, 0, 2, 1]
  [2, 1, 0, 3]
  [0, 3, 1, 2]
  [1, 2, 0, 1]
```

---

## Step 1: Convolution (Conv1)

Filter W (3×3):
```
  [1,  0, -1]
  [0,  1,  0]
  [-1, 0,  1]
```
Bias b = 0.

Output size = (4 - 3 + 0×2)/1 + 1 = **2×2**

**Computing each output position:**

```
Z[0,0] = X[0:3, 0:3] ⊙ W
  = 1×1 + 0×0 + 2×(-1)
  + 2×0 + 1×1 + 0×0
  + 0×(-1) + 3×0 + 1×1
  = 1 + 0 - 2 + 0 + 1 + 0 + 0 + 0 + 1 = 1

Z[0,1] = X[0:3, 1:4] ⊙ W
  = 0×1 + 2×0 + 1×(-1)
  + 1×0 + 0×1 + 3×0
  + 3×(-1) + 1×0 + 2×1
  = 0 + 0 - 1 + 0 + 0 + 0 - 3 + 0 + 2 = -2

Z[1,0] = X[1:4, 0:3] ⊙ W
  = 2×1 + 1×0 + 0×(-1)
  + 0×0 + 3×1 + 1×0
  + 1×(-1) + 2×0 + 0×1
  = 2 + 0 + 0 + 0 + 3 + 0 - 1 + 0 + 0 = 4

Z[1,1] = X[1:4, 1:4] ⊙ W
  = 1×1 + 0×0 + 3×(-1)
  + 3×0 + 1×1 + 2×0
  + 2×(-1) + 0×0 + 1×1
  = 1 + 0 - 3 + 0 + 1 + 0 - 2 + 0 + 1 = -2

Feature map Z after Conv1:
  [[ 1, -2],
   [ 4, -2]]
```

---

## Step 2: ReLU

```
A = ReLU(Z) = max(0, Z)

  [[ max(0,1),  max(0,-2)],   =  [[1, 0],
   [ max(0,4),  max(0,-2)]]       [4, 0]]
```

Neurons at Z=-2 are **dead** (output 0, gradient 0 — they won't update in backprop).

---

## Step 3: Flatten

```
Flatten A:  [1, 0, 4, 0]   (4-dimensional vector, row-major)
```

---

## Step 4: FC1 (4 → 2)

Weights W1 and bias b1:
```
W1 = [[0.5, -0.3],
      [0.2,  0.1],
      [0.4, -0.5],
      [-0.1, 0.3]]   # shape [4, 2]

b1 = [0.0, 0.0]
```

```
h = flatten @ W1 + b1
  = [1, 0, 4, 0] @ W1 + [0, 0]

h[0] = 1×0.5 + 0×0.2 + 4×0.4 + 0×(-0.1) = 0.5 + 0 + 1.6 + 0 = 2.1
h[1] = 1×(-0.3) + 0×0.1 + 4×(-0.5) + 0×0.3 = -0.3 + 0 - 2.0 + 0 = -2.3

h (pre-ReLU) = [2.1, -2.3]

After ReLU: a = [2.1, 0.0]   ← second neuron is dead
```

---

## Step 5: FC2 (2 → 1)

```
W2 = [[0.6],
      [-0.4]]   # shape [2, 1]

b2 = [0.1]

logit = a @ W2 + b2
      = [2.1, 0.0] @ [[0.6],[-0.4]] + [0.1]
      = 2.1×0.6 + 0.0×(-0.4) + 0.1
      = 1.26 + 0 + 0.1 = 1.36
```

---

## Step 6: Sigmoid + Loss

```
p̂ = σ(1.36) = 1 / (1 + e^{-1.36}) = 1 / (1 + 0.257) = 0.795

True label y = 1 (this is a cat image)

Binary cross-entropy loss:
  L = -[y log(p̂) + (1-y) log(1-p̂)]
    = -[1 × log(0.795) + 0 × log(0.205)]
    = -log(0.795)
    = 0.230
```

The model predicts 79.5% probability of cat with loss 0.230. Correct but not confident.

---

## Step 7: Backpropagation

**Gradient through sigmoid + BCE loss:**
```
dL/d(logit) = p̂ - y = 0.795 - 1.0 = -0.205
```

**Gradient through FC2:**
```
dL/dW2 = a^T × dL/d(logit)
        = [[2.1], [0.0]] × (-0.205)
        = [[-0.431], [0.0]]     ← W2[0] gets gradient, W2[1] gets 0 (dead neuron)

dL/db2 = dL/d(logit) = -0.205

dL/da = dL/d(logit) × W2^T
      = -0.205 × [0.6, -0.4]
      = [-0.123, 0.082]
```

**Gradient through FC1 ReLU:**
```
ReLU gate: a[0] = 2.1 > 0 → pass gradient; a[1] = 0 → block gradient

dL/dh = dL/da ⊙ 1[h > 0]
      = [-0.123, 0.082] ⊙ [1, 0]
      = [-0.123, 0.0]
```

**Gradient through FC1 weights:**
```
dL/dW1 = flatten^T × dL/dh
        = [[1], [0], [4], [0]] × [-0.123, 0.0]
        = [[-0.123, 0.0],
           [ 0.0,   0.0],
           [-0.492, 0.0],
           [ 0.0,   0.0]]

Neurons corresponding to flatten[1]=0 and flatten[3]=0 → zero gradient (zero input = dead for weight update)
```

**Gradient through Flatten and ReLU back to Conv:**
```
dL/d(flatten) = dL/dh × W1^T
              = [-0.123, 0.0] × W1^T
              = [-0.123×0.5 + 0.0×(-0.3),
                 -0.123×0.2 + 0.0×0.1,
                 -0.123×0.4 + 0.0×(-0.5),
                 -0.123×(-0.1) + 0.0×0.3]
              = [-0.0615, -0.0246, -0.0492, 0.0123]

Reshape to 2×2: [[-0.0615, -0.0246],
                 [-0.0492,  0.0123]]

ReLU gate from Step 2: [[1>0, -2≤0], [4>0, -2≤0]] = [[1,0],[1,0]]

dL/dZ = dL/dA ⊙ 1[Z>0]
      = [[-0.0615, -0.0246],    ⊙    [[1, 0],
         [-0.0492,  0.0123]]          [1, 0]]
      = [[-0.0615, 0.0],
         [-0.0492, 0.0]]
```

**Gradient through Conv filter:**
```
dL/dW_conv = cross-correlation of input X with dL/dZ

dL/dW[i,j] = Σ dL/dZ[r,c] × X[r+i, c+j]

Using only the non-zero gradients (dL/dZ[0,0]=-0.0615 and dL/dZ[1,0]=-0.0492):

dL/dW[0,0] = -0.0615×X[0,0] + -0.0492×X[1,0] = -0.0615×1 + -0.0492×2 = -0.1599
dL/dW[0,1] = -0.0615×X[0,1] + -0.0492×X[1,1] = -0.0615×0 + -0.0492×1 = -0.0492
dL/dW[0,2] = -0.0615×X[0,2] + -0.0492×X[1,2] = -0.0615×2 + -0.0492×0 = -0.1230
... (all 9 filter weights get gradients)

dL/db_conv = Σ dL/dZ = -0.0615 + 0 + (-0.0492) + 0 = -0.1107
```

---

## Step 8: Weight Update (SGD, lr=0.01)

```
W_conv ← W_conv - 0.01 × dL/dW_conv
W2     ← W2     - 0.01 × dL/dW2
W2[0]  = 0.6  - 0.01 × (-0.431) = 0.6  + 0.00431 = 0.60431   (increases)
W2[1]  = -0.4 - 0.01 × 0.0     = -0.4             (unchanged — dead neuron)

b2 ← b2 - 0.01 × (-0.205) = 0.1 + 0.00205 = 0.10205
```

After update: network is slightly more confident on cat images.

---

## Convolution Output Size Formulas (memorize)

```
Spatial output: W_out = ⌊(W_in + 2P - K) / S⌋ + 1

Same padding:  P = (K-1)/2,  W_out = W_in (for S=1)
Valid padding: P = 0,         W_out = W_in - K + 1

Parameters per Conv layer: K × K × C_in × C_out + C_out
Examples:
  Conv 3×3, C_in=3, C_out=64:  3×3×3×64 + 64 = 1,792 params
  Conv 3×3, C_in=64, C_out=128: 3×3×64×128 + 128 = 73,856 params
  Conv 1×1, C_in=256, C_out=64: 1×1×256×64 + 64 = 16,448 params (bottleneck)
```

---

## MaxPool Forward and Backward

```
MaxPool 2×2, stride=2 on feature map:
  Input:  [[3, 1, 2, 0],
           [4, 2, 1, 3],    Window [0:2, 0:2]: max=4 (position (1,0))
           [1, 0, 3, 1],    Window [0:2, 2:4]: max=3 (position (1,3))
           [2, 3, 0, 2]]    Window [2:4, 0:2]: max=3 (position (3,1))
                            Window [2:4, 2:4]: max=3 (position (3,3))
  Output: [[4, 3],
           [3, 3]]

Backward: gradient flows ONLY to the max position in each window.
  All non-max positions get gradient = 0 (sub-gradient of max operation).
```

---

## Common CNN Architecture Pattern

```
Input (224×224×3)
→ Conv 7×7, s=2, 64 filters → 112×112×64
→ MaxPool 3×3, s=2         → 56×56×64
→ ResBlock × 3             → 56×56×64
→ ResBlock × 4 (s=2)       → 28×28×128
→ ResBlock × 6 (s=2)       → 14×14×256
→ ResBlock × 3 (s=2)       → 7×7×512
→ GlobalAvgPool            → 512
→ FC 512 → 1000
→ Softmax

Total params (ResNet-34): ~21.8M
```

---

## Data Augmentation Dry Run

Standard augmentation for image classification training:

```python
import torchvision.transforms as T

train_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.08, 1.0)),    # random crop + resize
    T.RandomHorizontalFlip(p=0.5),                  # mirror image
    T.ColorJitter(brightness=0.4, contrast=0.4,
                  saturation=0.4, hue=0.1),          # color distortion
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],         # ImageNet mean/std
                std=[0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
```

**Why normalize with ImageNet mean/std?** Pre-trained models were trained with this normalization. Mismatching shifts the activation distributions → degraded transfer learning.

---

## Evaluation Metrics for CV Tasks

| Task | Metric | Formula / Notes |
|------|--------|-----------------|
| Image classification | Top-1 accuracy | % correct predictions |
| Image classification | Top-5 accuracy | % where correct is in top 5 |
| Object detection | mAP@50 | mean Average Precision at IoU≥0.5 |
| Object detection | mAP@50:95 | mAP averaged over IoU 0.5:0.05:0.95 |
| Segmentation | mIoU | mean Intersection over Union across classes |
| Face verification | AUC / EER | ROC curve area; Equal Error Rate |

**IoU (Intersection over Union):**
```
IoU = |A ∩ B| / |A ∪ B|
    = intersection_area / (area_A + area_B - intersection_area)

IoU = 0 → no overlap, IoU = 1 → perfect match
IoU > 0.5 → conventionally "good" detection
```

---

## Gotchas

**Vanishing gradients in deep CNNs.** Without residual connections, gradients shrink exponentially with depth (sigmoid/tanh saturation, repeated multiplication by small weights). Solution: ResNet skip connections let gradients flow directly through the identity path.

**Dead filters.** If a filter's output is always negative → all ReLU outputs = 0 → zero gradients → filter never updates. Solution: use Leaky ReLU, careful weight initialization (He init for ReLU), and batch normalization.

**He initialization for ReLU layers.**
```python
# He init: W ~ N(0, sqrt(2 / fan_in))
# PyTorch default for Conv2d uses Kaiming uniform (He variant)
nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
```
Using Xavier (for sigmoid/tanh) with ReLU causes variance to halve each layer → dead neurons.

**Batch normalization placement.** Standard: Conv → BN → ReLU. Pre-activation ResNets use BN → ReLU → Conv for slightly better gradient flow. Don't apply BN after final FC layer in generation tasks.

---

## Interview Q&A

**Q: Walk me through the full forward pass of a CNN.**
A: Input image → Conv layers apply learned filters via cross-correlation, each filter learning to detect a specific pattern (edges, textures, shapes). Output is feature maps. → Activation function (ReLU: kills negatives, introduces non-linearity). → Pooling (MaxPool: downsamples, provides translation invariance, reduces parameters). → Flatten spatial features into a 1D vector. → FC layers combine features for final prediction. → Softmax/Sigmoid for class probabilities. Key formula: output size = ⌊(W + 2P - K)/S⌋ + 1.

**Q: Why do we use MaxPool instead of stride=2 convolution for downsampling?**
A: MaxPool is parameter-free (no learned weights) and provides explicit translation invariance for its window. Stride=2 convolution also downsamples but with learned parameters — it can learn more complex downsampling but has more parameters. Modern architectures (ResNet, EfficientNet) often prefer stride=2 convolution as it's learnable. MaxPool is still used where strict translation invariance matters (early pooling in VGG-style nets).

**Q: What does backprop through MaxPool look like?**
A: MaxPool has no parameters (no weight gradient). During forward pass, we record the position of the maximum in each window. During backward pass, the gradient flows only to that max position — all other positions get gradient 0. This is the sub-gradient of the max operation.

**Q: What's the receptive field and why does it matter?**
A: The receptive field of a neuron is the region of the input image that influences its output. A neuron in a deeper layer has a larger receptive field (sees a bigger region of the original image). For tasks like object classification, you need neurons with large receptive fields to capture whole-object context. Formula for stacked 3×3 convs (no pooling): receptive field = 1 + 2×num_layers (for 3×3 kernels). With pooling or stride=2: grows exponentially. Deep CNNs stack many small filters (3×3) rather than one large filter (9×9) because: same receptive field with fewer parameters and more non-linearity.

---

## Connections

| Topic | File |
|-------|------|
| CNN architectures (VGG, ResNet, EfficientNet) | `02_cnn_architectures.md` |
| Transfer learning details | `../02_applications/01_transfer_learning.md` |
| Object detection (YOLO, Faster R-CNN) | `../02_applications/02_object_detection.md` |
| ViT (attention replaces convolution) | `5.transformers/01_fundamentals/05_vision_transformers.md` |
| LayoutLM (CNN + transformer for documents) | `7.multimodal/06_layoutlm_end_to_end.md` |

---

## Key Takeaway

**Forward:** Conv (cross-correlation) → ReLU (nonlinearity, kills negatives) → Pool (downsample, invariance) → FC (decision). **Backward:** gradients flow through ReLU gates (zero where Z≤0), Pool routes gradient to max position only, Conv filter gradient = correlate input with upstream gradient. Dead neurons (always-zero ReLU) give zero gradient → don't update → use He initialization and batch norm to prevent.

# 03 — Image Segmentation

## Quick Reference

| Task | Definition | Output | Key Models |
|------|-----------|--------|-----------|
| Semantic | Label every pixel with class | H×W class map | FCN, DeepLabV3+, SegFormer |
| Instance | Separate each object instance | H×W instance mask | Mask R-CNN |
| Panoptic | Semantic + Instance unified | H×W with instance IDs | Panoptic FPN, Mask2Former |
| Promptable | Segment anything with prompt | Arbitrary masks | SAM |

**Core difference from detection:** detection draws a bounding box; segmentation assigns a label to every single pixel.

---

## 1. Semantic vs Instance vs Panoptic

```
Original Image:
  Two cats, one dog, grass background

Semantic Segmentation:
  Each pixel → class label (not distinguishing instances)
  [cat cat cat dog dog grass grass]
  Both cats same color → can't tell them apart

Instance Segmentation:
  Each pixel → object instance
  [cat1 cat1 cat2 cat2 dog1 dog1 ___]
  Each cat gets unique color → distinct objects

Panoptic Segmentation:
  Semantic for background (grass, sky, road) +
  Instance for foreground objects (cats, dogs, people)
  Best of both worlds
```

**Evaluation metrics:**
```
Semantic:  mIoU (mean Intersection over Union across classes)
Instance:  mAP at IoU thresholds (same as detection but for masks)
Panoptic:  PQ = SQ × RQ (Panoptic Quality = Segmentation Quality × Recognition Quality)
```

---

## 2. FCN (Fully Convolutional Network) — Foundation

The original deep learning segmentation approach (Long et al., 2015).

**Key insight:** Replace FC layers with 1×1 convolutions → network accepts any input size → outputs per-pixel predictions.

```
FCN pipeline:
  Image → VGG/ResNet backbone → spatial feature maps
  → 1×1 conv (instead of FC) → coarse prediction map (e.g., 14×14)
  → Bilinear upsampling to original image size → per-pixel class scores

Problem: 14×14 → 224×224 via bilinear upsampling loses fine detail
Solution: FCN-8s skip connections:
  Add feature maps from pool3 and pool4 before upsampling
  → finer boundary predictions
```

This is the foundation. Every modern architecture (U-Net, DeepLab) builds on this idea.

---

## 3. U-Net — The Medical Imaging Standard

**Designed for:** biomedical image segmentation with limited labeled data.
**Authors:** Ronneberger et al., 2015.

### Architecture — The "U" Shape
```
Encoder (Contracting Path):           Decoder (Expanding Path):
  Input 572×572×1                        ←─── Skip connection from encoder
        ↓
  [Conv3×3→ReLU]×2 + MaxPool           Upsample (transposed conv)
  256×256×64                →→→→→→→→→   256×256×128 (cat + conv×2)
        ↓                    skip           ↓
  [Conv3×3→ReLU]×2 + MaxPool           Upsample
  128×128×128              →→→→→→→→→   128×128×64 (cat + conv×2)
        ↓                    skip           ↓
  [Conv3×3→ReLU]×2 + MaxPool           Upsample
  64×64×256                →→→→→→→→→   64×64×32 (cat + conv×2)
        ↓                    skip           ↓
  [Conv3×3→ReLU]×2 + MaxPool           Upsample
  32×32×512                →→→→→→→→→   32×32×16 (cat + conv×2)
        ↓                    skip           ↓
  Bottleneck: [Conv]×2                  1×1 Conv → num_classes
  16×16×1024                           → sigmoid/softmax → mask
```

### Skip Connections (Why They're Critical)
```
Encoder (deep): high-level semantics, small spatial resolution
  → "What is where in a coarse sense"
Decoder (expanding): must recover fine-grained spatial details
  → "Exactly which pixels belong to this object"

Skip connections concatenate encoder feature maps to decoder at each level
  → decoder gets both semantic context AND spatial detail
  → fine boundaries possible even after heavy downsampling

Without skip connections: decoder blurry, poor boundaries
With skip connections: sharp, accurate boundaries
```

### U-Net Losses
```
Binary segmentation: Binary Cross-Entropy + Dice Loss

BCE: −[y·log(p) + (1−y)·log(1−p)]  ← standard but slow to learn boundary

Dice Loss: 1 − 2|P∩G| / (|P|+|G|)
  where P=predicted mask, G=ground truth mask
  Directly optimizes the overlap metric
  Robust to class imbalance (most pixels are background)

Combined: L = BCE + Dice  ← most common for medical segmentation
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        return self.bce(pred, target.float()) + self.dice_weight * self.dice(pred, target)
```

### U-Net Code Reference
```python
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)   # 1024 = 512 from upsample + 512 from skip
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        # Output
        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        # Bottleneck
        b  = self.bottleneck(self.pool(s4))
        # Decoder (concat skip)
        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.out(d1)

model = UNet(in_channels=3, num_classes=10)   # RGB → 10 classes
x = torch.randn(2, 3, 256, 256)
print(model(x).shape)   # → [2, 10, 256, 256]
```

---

## 4. DeepLab Series — Atrous (Dilated) Convolution

### The Resolution Problem
CNN downsamples aggressively → final feature map at 1/32 of input → poor spatial resolution for segmentation.

**Option 1:** Remove stride → feature map stays large → but receptive field shrinks (can't see large context).
**Option 2:** Atrous (dilated) convolution — expand receptive field without reducing resolution.

### Atrous Convolution
```
Regular 3×3 conv (rate=1): looks at 3×3 patch
Atrous 3×3 conv (rate=2):  looks at 5×5 patch but uses only 9 values
Atrous 3×3 conv (rate=4):  looks at 9×9 patch but uses only 9 values

Dilation rate r: inserts (r-1) zeros between kernel elements
Same parameters as regular conv, much larger effective receptive field
```

### ASPP (Atrous Spatial Pyramid Pooling)
```
DeepLabV3: apply multiple atrous convs at different rates → capture multi-scale context

Input feature map
  →  1×1 conv
  →  Atrous conv (rate=6)
  →  Atrous conv (rate=12)
  →  Atrous conv (rate=18)
  →  Global Average Pooling
  → Concatenate all → 1×1 conv → output

Each branch captures different scale context:
  Rate=6: small objects, fine details
  Rate=12: medium objects
  Rate=18: large objects, global context
```

### DeepLabV3+ (state of the art for semantic segmentation)
```
Encoder: ResNet-101 or Xception with ASPP
  → Rich multi-scale semantic features

Decoder (simple):
  1×1 conv on encoder features
  4× bilinear upsample
  + skip from encoder's low-level features (4× smaller than output)
  → concat → 3×3 conv → 4× bilinear upsample → output

Better boundaries than DeepLabV3 due to decoder skip connection
```

```python
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

model = deeplabv3_resnet101(weights='DEFAULT')
model.eval()

image = torch.randn(1, 3, 520, 520)
with torch.no_grad():
    output = model(image)['out']   # [1, 21, 520, 520] for PASCAL VOC 21 classes
    pred_mask = output.argmax(1)   # [1, 520, 520]
```

---

## 5. Mask R-CNN — Instance Segmentation

Extends Faster R-CNN with a mask prediction branch.

### Architecture
```
Image → ResNet-FPN backbone → Feature Pyramid
                                    ↓
                               RPN (proposals)
                                    ↓
                            ROI Align (pixel-accurate)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Classification    Box Regression    Mask Head
                 (class)         (box offset)    (binary mask)
                    ↓               ↓               ↓
              class scores      refined box     28×28 binary mask
                                                per class
```

### Mask Head
```
For each ROI (aligned to 14×14):
  4× Conv3×3 → ConvTranspose2d (upsample to 28×28) → Conv1×1 → sigmoid

Output: 28×28 binary mask × num_classes
At inference: take mask for predicted class only

Why 28×28? Compromise between detail and computation.
At inference: resize to actual bounding box size
```

### Mask Loss
```
Binary cross-entropy on the 28×28 mask
ONLY computed for the ground truth class of each ROI
→ No competition between classes in the mask head
→ Mask prediction decoupled from classification
```

```python
from torchvision.models.detection import maskrcnn_resnet50_fpn

model = maskrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

image = [torch.randn(3, 800, 800)]
with torch.no_grad():
    predictions = model(image)

# predictions[0]: {'boxes', 'labels', 'scores', 'masks'}
masks  = predictions[0]['masks']   # [N, 1, H, W] — soft masks (sigmoid output)
scores = predictions[0]['scores']  # [N]

# Threshold masks
binary_masks = (masks > 0.5).squeeze(1)   # [N, H, W]
```

---

## 6. SegFormer (2021) — Transformer for Segmentation

Hierarchical Transformer backbone (no positional encoding needed) + lightweight MLP decoder.

```
Architecture:
  4-stage hierarchical Transformer encoder (like Swin but simpler)
  Each stage: overlapping patch merging + self-attention
  Produces {1/4, 1/8, 1/16, 1/32} scale features

MLP Decoder:
  1×1 conv to unify channel dims
  Upsample all to 1/4 scale
  Concatenate → MLP → upsample 4× to output

Key: no complex ASPP or FPN — simple MLP decoder is sufficient
  because the hierarchical encoder captures multi-scale context via attention
```

```python
# Using Hugging Face transformers
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    'nvidia/segformer-b2-finetuned-ade-512-512'
)
```

---

## 7. SAM (Segment Anything Model, Meta 2023)

**Paradigm shift:** promptable segmentation. Given any prompt (point, box, text), segment the object.

### Architecture
```
Image Encoder: ViT-H (MAE pretrained) → image embedding (one-time, cached)
Prompt Encoder: encodes points, boxes, masks, or text → prompt embedding
Mask Decoder: lightweight transformer → 3 mask predictions (ambiguous cases)
              + confidence scores for each mask
```

### SAM Usage
```python
from segment_anything import sam_model_registry, SamPredictor

# Load model
sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h.pth')
predictor = SamPredictor(sam)

# Set image (one-time, caches embedding)
predictor.set_image(image)   # numpy H×W×3

# Segment from a point click
masks, scores, logits = predictor.predict(
    point_coords=np.array([[500, 375]]),   # x, y
    point_labels=np.array([1]),             # 1=foreground, 0=background
    multimask_output=True                   # returns 3 masks for ambiguity
)

# Segment from bounding box
masks, scores, logits = predictor.predict(
    box=np.array([100, 200, 600, 800]),    # x1, y1, x2, y2
    multimask_output=False
)

# Automatic mask generation (all objects in image)
from segment_anything import SamAutomaticMaskGenerator
generator = SamAutomaticMaskGenerator(sam)
masks = generator.generate(image)   # list of dicts with 'segmentation', 'area', 'bbox', 'score'
```

### SAM2 (2024) — Video Segmentation
```
Extends SAM to video: given a prompt in one frame, tracks the mask throughout the video.
Memory attention mechanism: stores object memory from previous frames.
Use case: video annotation, document processing (multi-page), surveillance.
```

**SAM Limitations:**
- Cannot label what the segment IS (no classification, just segmentation)
- Struggles with very thin structures (wires, hair)
- Slow for real-time applications (ViT-H encoder is large)
- SAM2 adds video support but same classification limitation

---

## 8. Evaluation Metrics

### mIoU (mean Intersection over Union)
```
For each class c:
  IoU_c = TP_c / (TP_c + FP_c + FN_c)
  = Intersection / Union at pixel level

mIoU = mean over all classes

Example (2 classes: cat, background):
  Background: 90% pixels correct → IoU=0.92
  Cat: 70% pixels correct → IoU=0.65
  mIoU = (0.92 + 0.65) / 2 = 0.785
```

```python
def compute_miou(pred, target, num_classes):
    """pred, target: [H, W] class index arrays"""
    ious = []
    for c in range(num_classes):
        pred_c   = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum()
        union        = (pred_c | target_c).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious)

# Using torchmetrics
from torchmetrics.segmentation import MeanIoU
metric = MeanIoU(num_classes=21, per_class=True)
metric.update(predictions, targets)
print(metric.compute())
```

### Dice Score (F1 for masks)
```
Dice = 2|P∩G| / (|P|+|G|)

Same as F1 score. Dice = 2×IoU / (1+IoU).
Preferred in medical imaging where mask area is small (class imbalance).
```

---

## 9. When to Use What

| Task | Model | Dataset Size | Notes |
|------|-------|-------------|-------|
| Medical binary segmentation | U-Net + Dice loss | Small (100-5K) | Standard in medical AI |
| General semantic segmentation | DeepLabV3+ or SegFormer | Medium-Large | Use pretrained COCO/ADE20K |
| Instance segmentation | Mask R-CNN | Large (>10K annotated) | Need bounding box + mask labels |
| Promptable / interactive | SAM | Any (zero-shot) | No training needed for new tasks |
| Video object segmentation | SAM2 | Any | Propagates mask through video |
| Real-time on edge | Lightweight U-Net or MobileNet-DeepLab | - | Reduce channels, quantize |
| Document zone segmentation | U-Net or YOLOv8-seg | Medium | Text zones as segments |

---

## 10. Gotchas

**mIoU is heavily affected by rare classes.**
A model that ignores all small-area rare classes (medical anomalies, traffic signs) can still achieve mIoU=0.85+. Always report per-class IoU alongside mIoU. Weight rare classes or use Dice loss.

**Upsampling artifacts at boundaries.**
Bilinear upsampling produces checkerboard artifacts for sharp boundaries. Use transposed convolution (ConvTranspose2d) or sub-pixel convolution (PixelShuffle) for cleaner boundaries.

**U-Net skip connection channel mismatch.**
Encoder output channels must match decoder input before concatenation. Debug by printing shapes at each concat. Common error: mismatched after modifying encoder depth or channel sizes.

**Semantic segmentation ignores instance boundaries.**
If two cars touch, semantic segmentation merges them. If your use case requires distinguishing instances (parking lot car counting), use Mask R-CNN or panoptic segmentation.

**SAM doesn't know what the segment IS.**
SAM outputs a mask, not a class. For document zone detection, you need to add a classification step after SAM segments regions, or use it in tandem with a classifier.

---

## 11. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| mIoU good but rare class IoU=0 | Model ignores rare class | Weighted CE loss; oversample rare class; Dice loss |
| Blurry masks, poor boundaries | Upsampling too coarse | Add skip connections; use ROI Align; increase decoder resolution |
| U-Net memory OOM | Image too large / too many channels | Reduce image size; gradient checkpointing; reduce batch size |
| Mask R-CNN masks misaligned | Using ROI Pooling instead of ROI Align | Ensure using maskrcnn_resnet50_fpn (has ROI Align) |
| Dice loss NaN | Predictions all zero + empty GT | Add epsilon to denominator; check data pipeline |
| DeepLab boundaries jagged | Low-res feature map | Use higher output stride (16 instead of 32) |
| SAM very slow | ViT-H encoder large | Cache image embeddings; use SAM-ViT-B (faster, less accurate) |

---

## 12. Interview Q&A (Senior Level)

**Q: Why does U-Net use skip connections when standard encoders work for classification?**
A: For classification, only the semantic answer matters ("this is a cat") — spatial details are discarded by GAP or flatten. For segmentation, you need to answer "which exact pixels are cat?" — spatial information must be preserved. The encoder downsamples aggressively (256→16 in feature space) to build semantic understanding, but loses precise boundary locations. Skip connections at each encoder depth pass the fine-grained spatial features directly to the decoder, which upsamples back to full resolution. The decoder then knows both "this is a cat region" (from the bottleneck) AND "here's exactly where the boundary is" (from the skip). Without skip connections, the decoder must hallucinate boundary details from the coarse feature map — result: smooth, blurry, inaccurate masks.

**Q: What is dilated convolution and when would you use it over pooling for segmentation?**
A: Dilated (atrous) convolution inserts gaps of (r-1) zeros between kernel elements, expanding the effective receptive field by a factor of r without reducing spatial resolution. A 3×3 conv with dilation=2 sees a 5×5 area but still has 9 parameters. For segmentation, pooling + upsampling loses spatial precision — the quantized pooling and interpolated upsampling create misalignment. Dilated convolution maintains full spatial resolution throughout the network while still accumulating large receptive field. DeepLab uses ASPP — multiple dilation rates in parallel — to capture multi-scale context at full resolution. Tradeoff: dilated convolution uses more memory (larger feature maps) and can have gridding artifacts (checkerboard patterns) if the stride pattern doesn't cover all pixels — mitigated by mixing different dilation rates.

**Q: How does Mask R-CNN extend Faster R-CNN, and what is the key difference in its loss function?**
A: Mask R-CNN adds a third head (mask branch) to Faster R-CNN's existing classification and box regression heads. The critical engineering choice is ROI Align instead of ROI Pooling — ROI Align uses bilinear interpolation for sub-pixel accurate feature alignment, which is critical for pixel-level mask prediction. The mask head is a small FCN (4× conv3×3) producing a 28×28 binary mask per class. Key insight in the loss: the mask loss is computed only for the ground truth class of each instance, not all classes simultaneously. This decouples mask prediction from classification — the network learns masks without competition between classes — which is more efficient and achieves better masks than predicting a single multi-class mask.

---

## 13. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| Detection backbone shared with segmentation | `02_object_detection.md` | Mask R-CNN = Faster R-CNN + mask head |
| Dice loss derivation | `../../1.deep learning/fundamentals/06_specialized_losses.md` | Dice loss for imbalanced segmentation |
| ViT as encoder (SegFormer, SAM) | `../../1.deep learning/architectures/04_transformer.md` | Transformer encoder in vision |
| Skip connections in U-Net | `../../1.deep learning/fundamentals/05_modern_components.md` | Residual/skip connection concept |
| Document zone segmentation | `02_object_detection.md` | YOLO-seg vs U-Net for document |

---

## Key Takeaway

**Task hierarchy:**
```
Semantic (label pixels by class, no instances)
  ← FCN → DeepLabV3+ → SegFormer

Instance (separate object instances, no background)
  ← Mask R-CNN

Panoptic (semantic + instance unified)
  ← Panoptic FPN, Mask2Former

Promptable (user-guided, zero-shot)
  ← SAM, SAM2
```

**Architecture choices:**
- U-Net: medical segmentation, small data, binary/few classes
- DeepLabV3+: best accuracy for semantic on natural images
- Mask R-CNN: instance segmentation, standard benchmark
- SAM: interactive, zero-shot, any domain

**Loss function choice:** BCE for balanced classes, Dice for imbalanced (most segmentation tasks), BCE+Dice combined for best results.

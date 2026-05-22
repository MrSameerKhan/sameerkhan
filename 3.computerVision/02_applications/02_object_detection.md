# 02 — Object Detection

## Quick Reference

| Model | Type | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| Faster R-CNN | Two-stage | Slow (~5 FPS) | High | Accuracy-critical, offline |
| SSD | One-stage | Fast (~59 FPS) | Medium | Real-time, embedded |
| RetinaNet | One-stage | Medium (~18 FPS) | High | Class imbalance (focal loss) |
| YOLOv5/v8 | One-stage | Very fast (~140 FPS) | High | Real-time production |
| DETR | Transformer | Medium | High | No NMS needed, elegant |

**Key metrics:** IoU threshold, mAP (mean Average Precision), FPS

---

## 1. Detection vs Classification

```
Classification: input image → one class label
  "There is a cat"

Object Detection: input image → [class label + bounding box] × N objects
  Cat at (x=10, y=20, w=100, h=80), Dog at (x=100, y=30, w=120, h=90)

Bounding box representation:
  [x_center, y_center, width, height]  = YOLO format
  [x_min, y_min, x_max, y_max]         = Pascal VOC / most libraries
  [x_min, y_min, width, height]        = COCO format
```

---

## 2. Intersection over Union (IoU)

The fundamental metric for measuring detection quality.

```
IoU = Area of Intersection / Area of Union
    = intersection_area / (area_A + area_B - intersection_area)

IoU = 0.0:  no overlap
IoU = 0.5:  standard threshold for "correct detection"
IoU = 0.75: strict threshold (COCO hard metric)
IoU = 1.0:  perfect overlap
```

```python
def compute_iou(box1, box2):
    """box format: [x_min, y_min, x_max, y_max]"""
    # Intersection
    x_min = max(box1[0], box2[0]);  y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2]);  y_max = min(box1[3], box2[3])
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    # Union
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - intersection
    return intersection / (union + 1e-6)

gt   = [10, 10, 100, 100]  # ground truth box
pred = [20, 20, 110, 110]  # predicted box
print(f"IoU: {compute_iou(gt, pred):.3f}")
```

---

## 3. Anchor Boxes

**The Problem:** The network must predict boxes of wildly different sizes and aspect ratios.

**Solution: Pre-defined Prior Shapes**

```
Anchor boxes: pre-defined boxes at each spatial location of the feature map
covering different scales and aspect ratios.

Example anchors (3 scales × 3 ratios = 9 per location):
  Scales: 32², 64², 128²  (pixel area)
  Ratios: 1:1, 1:2, 2:1  (width:height)

At each feature map location, the model predicts:
  - Is there an object here? (objectness score)
  - Which class? (classification)
  - How much to adjust the anchor? (regression: Δx, Δy, Δw, Δh)
```

### Anchor Assignment (Training)

```
For each ground truth box:
  Compute IoU with every anchor

  IoU > 0.7  = positive anchor (contains an object) → regress to GT box
  IoU < 0.3  = negative anchor (background)
  0.3-0.7    = ignore during training

Each GT box assigned to anchor with highest IoU
```

### Regression Targets

```
The model doesn't predict absolute coordinates — it predicts offsets from anchor:

tx = (GT_cx - anchor_cx) / anchor_w
ty = (GT_cy - anchor_cy) / anchor_h
tw = log(GT_w / anchor_w)
th = log(GT_h / anchor_h)

Log transform for w, h ensures predicted sizes always positive
```

---

## 4. Non-Maximum Suppression (NMS)

Detection produces hundreds of overlapping boxes for the same object. NMS removes duplicates.

```
Algorithm:
  1. Sort all predicted boxes by confidence score (descending)
  2. Take highest-scoring box → keep it
  3. Remove all other boxes with IoU ≥ threshold (e.g., 0.5) with kept box
  4. Repeat with remaining boxes

Result: one box per object (ideally)
```

```python
def nms(boxes, scores, iou_threshold=0.5):
    """boxes: [N, 4] in [x_min, y_min, x_max, y_max]; scores: [N]"""
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious    = np.array([compute_iou(boxes[i], boxes[j]) for j in order[1:]])
        order   = order[1:][ious <= iou_threshold]
    return keep

# PyTorch built-in
from torchvision.ops import nms as torch_nms
keep = torch_nms(boxes, scores, iou_threshold=0.5)
```

**Soft-NMS:** Instead of removing boxes above IoU threshold, reduce their scores. Better for partially overlapping objects (e.g., crowded people).

---

## 5. Two-Stage Detection — Faster R-CNN

### The Pipeline

```
Image → CNN Backbone → Feature Map
              ↓
       RPN (Region Proposal Network)
              ↓
  ROI coordinates    Objectness scores
              ↓
    ROI Pooling / ROI Align
              ↓
    ROI Features [N×7×7×C]
              ↓
    Fast R-CNN Head (FC layers)
         ↓              ↓
  Class probabilities  Box refinements
```

### Stage 1: Region Proposal Network (RPN)

```
Runs on feature map, generates ~2000 candidate regions ("proposals")

At each feature map location (H×W locations total):
  For each of K anchor boxes:
    Output 1: Is there an object here? (2-class softmax)
    Output 2: Box offsets (4 values: Δx, Δy, Δw, Δh)

Total RPN outputs: H×W×K×6  (2 cls + 4 reg per anchor)
After RPN: apply NMS on proposals → top 300 high-confidence regions
```

### Stage 2: Detection Head (Fast R-CNN)

```
For each proposal:
  1. ROI Align: warp proposal feature to fixed size (7×7)
  2. Flatten → FC(4096) → FC(4096)
  3. Two heads:
     - Classification: FC = softmax(num_classes + 1)  [+1 for background]
     - Box regression: FC = (num_classes × 4) offsets
```

### ROI Align vs ROI Pooling

```
ROI Pooling:  quantizes (rounds) proposal coordinates to feature map grid
  → misalignment for small objects

ROI Align: uses bilinear interpolation instead of rounding
  → pixel-accurate spatial mapping
  → critical for segmentation (Mask R-CNN)
```

### Faster R-CNN Losses

```
Total Loss = RPN_cls + RPN_reg + det_cls + det_reg

RPN_cls: binary cross-entropy (object vs background)
RPN_reg: smooth L1 loss on box offsets for positive anchors
det_cls: cross-entropy on N_class + 1 classes
det_reg: smooth L1 loss on per-class box refinements

Smooth L1 (Huber loss):
  For small x: 0.5x²   (like MSE, gentle)
  For large x: |x| - 0.5  (like L1, robust to outliers)
```

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pretrained Faster R-CNN
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
model.eval()

# Inference
image = [torch.randn(3, 800, 800)]  # list of images
with torch.no_grad():
    predictions = model(image)

# predictions[0]: {'boxes': Tensor[N,4] xyxy, 'labels': Tensor[N], 'scores': Tensor[N]}
boxes  = predictions[0]['boxes']    # [N, 4] xyxy format
labels = predictions[0]['labels']   # [N]
scores = predictions[0]['scores']   # [N]

# Filter by score
mask            = scores > 0.5
filtered_boxes  = boxes[mask]
filtered_labels = labels[mask]
```

---

## 6. Feature Pyramid Network (FPN)

Critical for detecting objects at multiple scales.

```
Problem: small objects disappear in deep CNN (feature map too small)

FPN: creates feature pyramid from backbone feature maps

Top-down pathway:
  C5 (deep, small, semantically rich)
    ↑ upsample + lateral connection
  C4
    ↑ upsample + lateral connection
  C3 (shallow, large, spatially detailed)

Result: P3, P4, P5 — each with 256 channels
  P3: detects small objects  (large spatial resolution)
  P4: detects medium objects
  P5: detects large objects

RPN runs at all pyramid levels → multi-scale proposals
```

FPN is now standard in almost all modern detectors (Faster R-CNN, RetinaNet, YOLO).

---

## 7. One-Stage Detection — YOLO

**Core Idea:** Skip region proposals. Directly predict boxes and classes from the feature map.

### YOLO Grid System (YOLOv1-v3 concept)

```
Divide image into S×S grid (e.g., 13×13)
Each cell predicts:
  B bounding boxes (each: x, y, w, h, confidence)
  C class probabilities (conditional on object present)

Output tensor: S × S × (B×5 + C)
For COCO: 13×13×(3×5 + 80) = 13×13×255

x, y: box center relative to cell (0-1)
w, h: box size relative to image
confidence: P(object) × IoU(pred, gt)
```

### YOLOv3 — Anchor Boxes at 3 Scales

```
3 detection heads at different scales (like FPN):
  13×13: large objects  (stride 32)
  26×26: medium objects (stride 16)
  52×52: small objects  (stride 8)

3 anchors per scale = 9 anchors total (clustered from COCO via k-means)
```

### YOLOv5/v8 — Modern YOLO

```
YOLOv5: CSP backbone, PANet neck, anchor-based head
  Training: mosaic augmentation, mixup, copy-paste
  Inference: 140 FPS on V100

YOLOv8 (2023): anchor-free, decoupled head
  Anchor-free: directly predict (cx, cy, w, h) without anchors
  Decoupled: separate branches for classification and regression
  Better small object detection
```

```python
from ultralytics import YOLO

# Load pretrained
model = YOLO('yolov8m.pt')   # n=nano, s=small, m=medium, l=large, x=extra-large

# Inference
results = model('image.jpg', conf=0.25, iou=0.45)
for r in results:
    boxes  = r.boxes.xyxy    # [N, 4]
    scores = r.boxes.conf    # [N]
    labels = r.boxes.cls     # [N]

# Fine-tune on custom dataset
model.train(data='custom.yaml', epochs=100, imgsz=640, batch=16,
            device='cuda', lr0=0.01, cos_lr=True)
```

```yaml
# custom.yaml for YOLOv8
path: /data/dataset
train: images/train
val: images/val
test: images/test  # optional

nc: 3  # number of classes
names: ['cat', 'dog', 'bird']
```

---

## 8. RetinaNet + Focal Loss

### The Class Imbalance Problem in Detection

```
One-stage detectors evaluate ~100K anchors per image.
Positive anchors (contain object): ~10-100
Negative anchors (background): ~99,900-99,990

99.99% of anchors are background → model learns to predict background only
→ accuracy of ~99.99% even without detecting anything useful
→ CE loss dominated by easy negatives
```

### Focal Loss (Lin et al. 2017)

```
Standard CE:   CE(p, y) = -log(p_t)

Focal Loss:    FL(p_t) = -(1 - p_t)^γ · log(p_t)

γ (gamma): focusing parameter (default γ=2)
  Easy examples (p_t > 0.5):  (1 - p_t)^γ ≈ 0  → contribution near zero
  Hard examples (p_t ≈ 0):    (1 - p_t)^γ = 1  → full contribution

α (alpha): class balance weight
  FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

```python
import torch
import torch.nn.functional as F

def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    """predictions: [N, num_classes] raw logits; targets: [N] class indices"""
    ce_loss      = F.cross_entropy(predictions, targets, reduction='none')
    p_t          = torch.exp(-ce_loss)
    focal_weight = (1 - p_t) ** gamma
    loss         = alpha * focal_weight * ce_loss
    return loss.mean()
```

**RetinaNet architecture:** ResNet-50 + FPN backbone → Classification subnet: 4× Conv3×3 → Conv → Sigmoid (K×A outputs per location) + Box regression subnet: 4× Conv3×3 → Conv → 4×A outputs. Training: focal loss for classification, smooth L1 for regression.

---

## 9. SSD (Single Shot MultiBox Detector)

```
Multi-scale feature maps from VGG backbone:
  Conv4_3:  38×38  → detect small objects
  Conv7:    19×19
  Conv8_2:  10×10
  Conv9_2:  5×5
  Conv10_2: 3×3
  Conv11_2: 1×1   → detect large objects

At each scale: predict class scores + box offsets for each anchor
No ROI, no ROI pooling → one pass through network
Default boxes (anchors): at each location, several boxes with different aspect ratios
Total predictions: 8732 boxes (before NMS)
```

**SSD vs YOLO vs Faster R-CNN:**
- Speed: SSD ≈ YOLO >> Faster R-CNN
- Accuracy: Faster R-CNN > YOLO > RetinaNet > SSD (historically)
- Modern: YOLOv8 beats SSD on both speed and accuracy

---

## 10. mAP (mean Average Precision)

The standard evaluation metric for object detection.

### Precision-Recall Curve per Class

```
For each confidence threshold t:
  Precision(t) = TP(t) / (TP(t) + FP(t))
  Recall(t)    = TP(t) / (TP(t) + FN(t))

A detection is TP if:
  IoU with any GT box ≥ threshold (e.g., 0.5) AND
  No other higher-confidence detection already matched this GT box

AP = Area under Precision-Recall curve for one class
mAP = mean AP over all classes
```

### COCO mAP Notation

```
mAP@.5:      IoU threshold = 0.5  (lenient)
mAP@.75:     IoU threshold = 0.75 (strict)
mAP@.5:.95:  average mAP over IoU thresholds from 0.5 to 0.95 step 0.05 (COCO primary)

mAP_S: mAP for small objects (area < 32²)
mAP_M: mAP for medium objects
mAP_L: mAP for large objects
```

```python
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75], box_format='xyxy')
# predictions: list of dicts [{'boxes': Tensor, 'scores': Tensor, 'labels': Tensor}]
# targets:     list of dicts [{'boxes': Tensor, 'labels': Tensor}]
metric.update(predictions, targets)
result = metric.compute()
print(f"mAP@0.5: {result['map_50']:.4f}")
print(f"mAP@0.5:0.95: {result['map']:.4f}")
```

---

## 11. DETR (DEtection TRansformer)

**Removes:** Anchor boxes, NMS, multi-scale feature maps. **Uses:** Transformer encoder-decoder + bipartite matching loss.

```
Image → CNN backbone → Flattened features → Transformer Encoder
                                                    ↓
                              N learned object queries → Transformer Decoder
                                                    ↓
                              N predictions (class + box) via FFN heads

Loss: Hungarian algorithm finds optimal matching between N predictions and GT boxes
→ No NMS needed (each query predicts exactly one object or "no object")
```

**Weakness:** Slow to train (500 epochs on COCO), struggles with small objects (no FPN in original), slow inference vs YOLO. Improved by Deformable DETR, DINO, RT-DETR.

---

## 11.5 Modern Detectors (2024-2025)

### YOLO Family Beyond v8

| Model | Year | Highlights |
|-------|------|-----------|
| YOLOv8 (Ultralytics) | 2023 | Anchor-free, decoupled head, widely deployed |
| YOLOv9 | 2024 | PGI (Programmable Gradient Information) + GELAN architecture |
| YOLOv10 (Tsinghua) | 2024 | **NMS-free training** via dual head + consistent dual assignments |
| YOLOv11 (Ultralytics) | 2024 | C3k2 block + position-sensitive head; current default in production |
| YOLOv12 | 2025 | Area attention — adds efficient attention to YOLO backbone |
| RT-DETR (Baidu) | 2024 | DETR variant fast enough for real-time (beats YOLOv8 on COCO) |
| RT-DETRv2 / v3 | 2024-25 | Distilled and refined RT-DETR variants |
| D-FINE | 2024 | Refined DETR with fine-grained distribution refinement; SOTA on COCO real-time |

**Senior interview answer:** "I'd default to YOLOv8 or v11 for production — proven training pipeline, great tooling. For research / SOTA benchmarks, RT-DETR or D-FINE — DETR-style models are now competitive in latency. The big 2024 advance was NMS-free detection (YOLOv10, DETR variants), which removes a brittle post-processing step."

### Open-Vocabulary Detection (Detect by Name)

Classical detectors are limited to a fixed class set. Open-vocabulary detectors accept arbitrary text class names at inference — no retraining needed.

| Model | Year | Idea |
|-------|------|------|
| GLIP (Microsoft, 2022) | 2022 | Aligns image regions with text phrases — like CLIP for detection |
| OWL-ViT / OWLv2 (Google) | 2022-23 | ViT + CLIP-style text encoder for open-vocab detection |
| Grounding DINO (IDEA) | 2023 | DINO + grounding head, very strong open-vocab + phrase grounding |
| YOLO-World (Tencent) | 2024 | YOLOv8-speed real-time open-vocabulary detection |
| APE (Aligning and Prompting Everything) | 2024 | Unified detection + segmentation + retrieval |

```python
# Grounding DINO — detect by text prompt
from groundingdino.util.inference import load_model, predict
model = load_model("groundingdino_swint_ogc.cfg.py", "weights.pth")
boxes, logits, phrases = predict(
    model=model, image=image,
    caption="a stop sign . a person crossing the street . a red car",
    box_threshold=0.35, text_threshold=0.25,
)
```

---

## 12. When to Use What

| Scenario | Model | Why |
|----------|-------|-----|
| Fastest inference (edge/mobile) | YOLOv8n | Nano model, highest FPS |
| Production accuracy + speed | YOLOv8m or YOLOv8l | Best accuracy/speed tradeoff |
| Highest accuracy, offline | Faster R-CNN with ResNet-101+FPN | Two-stage, exhaustive |
| Class imbalance (rare objects) | RetinaNet with focal loss | Focal loss handles imbalance |
| Custom training, quick start | YOLOv8 (ultralytics) | Best ecosystem, easy API |
| Research / no-NMS requirement | DETR / Deformable DETR | Clean end-to-end pipeline |
| Document layout detection | Faster R-CNN or YOLO (small objects) | Text regions are small |

---

## 13. Gotchas

**IoU threshold choice changes your evaluation results dramatically.** mAP@0.5 and mAP@0.5:0.95 can differ by 20-30 points. Always report which threshold. COCO standard is mAP@0.5:0.95.

**NMS threshold controls duplicate suppression aggressiveness.** NMS IoU=0.3: aggressive, removes more boxes (good for sparse objects). NMS IoU=0.6: lenient, keeps more boxes (good for dense/overlapping objects like crowds). Default 0.45 in most YOLO implementations.

**YOLO confidence threshold affects both FP and FN.** High confidence (0.7+): few detections, high precision, low recall. Low confidence (0.1): many detections, low precision, high recall. Tune for your use case.

**Two-stage detectors need many GPU hours to train from scratch.** Faster R-CNN on COCO from scratch: ~36 hours on 8 V100. Always start from COCO-pretrained weights and fine-tune on your custom data.

**Small object detection requires high input resolution.** YOLO default is 640×640. For small objects (OCR bounding boxes, sub-mm features), use 1280×1280 or tile the image.

---

## 14. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Many duplicate detections | NMS threshold too high | Reduce NMS IoU threshold |
| Missing detections of small objects | Input resolution too low | Increase img size; use FPN |
| Model detects background regions | Confidence threshold too low | Increase conf threshold |
| mAP low despite visual inspection looking good | IoU threshold too strict | Check mAP@0.5 separately |
| Training loss NaN | Large gradient from anchor with huge offset | Clip gradients; check anchor sizes match object sizes |
| Inference much slower than expected | NMS overhead or CPU inference | Use GPU; batch inference; TensorRT export |

---

## 15. Code Reference — Custom YOLO Training

```python
from ultralytics import YOLO
import yaml

# Dataset structure required:
# dataset/
#   images/train/*.jpg
#   images/val/*.jpg
#   labels/train/*.txt   (YOLO format: class cx cy w h, normalized 0-1)
#   labels/val/*.txt

# dataset.yaml
config = {
    'path': '/data/document_detection',
    'train': 'images/train',
    'val':   'images/val',
    'nc':    5,
    'names': ['title', 'paragraph', 'table', 'figure', 'header']
}
with open('dataset.yaml', 'w') as f:
    yaml.dump(config, f)

# Train YOLOv8
model   = YOLO('yolov8m.pt')    # start from COCO pretrained
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=1280,                  # larger for document detection
    batch=8,
    device='cuda',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    augment=True,
    fliplr=0.5,
    mosaic=0.3,                  # reduce mosaic for documents (keeps aspect ratio)
    project='runs/detect',
    name='document_yolov8m'
)

# Evaluate
metrics = model.val(data='dataset.yaml')
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")

# Export for deployment
model.export(format='onnx',  imgsz=12)   # ONNX
model.export(format='torchscript')        # TorchScript
```

---

## 16. Interview Q&A (Senior Level)

**Q: What's the fundamental difference between one-stage and two-stage detectors?**

A: Two-stage (Faster R-CNN): Stage 1 — RPN proposes candidate regions (objectness only, ~2000 proposals). Stage 2 — each proposal classified + box refined. Decoupling these two tasks means each is simpler and more accurate, but sequentially running 2000 region classifiers is slow. One-stage (YOLO, SSD, RetinaNet): directly predict class + box from every anchor in a single forward pass — no proposal stage. Faster but historically less accurate because the network must solve harder multi-task problem simultaneously. Modern YOLO has largely closed this accuracy gap while maintaining speed advantage. The class imbalance problem (99% background anchors) is the main challenge for one-stage detectors — solved by focal loss in RetinaNet.

**Q: Why does focal loss work for object detection?**

A: In one-stage detection, ~99.99% of anchors are background. Standard cross-entropy gives each example loss proportional to -log(p). Easy negatives have high confidence (p_t=0.99 for background) → CE = -log(0.99) = 0.01 each, but with 100K easy negatives → 1000 units of loss total swamps the 10 hard positive anchors contributing maybe 5 units. The gradient is dominated by easy negatives → model learns to predict background everywhere. Focal loss down-weights easy examples: (1-p_t)^γ: when p_t=0.99 (easy), weight = 0.0001 → contribution near zero. When p_t=0.01 (hard), weight = 1 → full contribution. The focusing parameter γ=2 effectively discards easy examples and focuses training on hard examples, enabling one-stage training with correct gradient balance.

**Q: What is ROI Align and why is it better than ROI Pooling for instance segmentation?**

A: ROI Pooling quantizes (rounds to nearest integer) the proposal coordinates onto the feature map grid. For a proposal at (33.7, 56.3), it rounds to (34, 56) — loses sub-pixel information. In feature extraction, this ~0.5 pixel misalignment per layer accumulates — object features misaligned from actual object position. For classification this is tolerable. For segmentation (where per-pixel accuracy matters), misalignment of even 1-2 pixels destroys mask quality. ROI Align uses bilinear interpolation at four sampling points within each ROI bin, computing feature values at fractional coordinates without quantization. This preserves spatial alignment — Mask R-CNN uses ROI Align specifically to get pixel-accurate masks.

---

## 17. Connections

| This file | Links to | Why |
|-----------|---------|-----|
| Backbone architectures | `../01_fundamentals/02_cnn_architectures.md` | ResNet-FPN as detection backbone |
| DETR deep dive | `06_detr_deep.md` | Deformable DETR, DINO, RT-DETR derivations |
| Focal loss math | `../../2.deep learning/01_fundamentals/06_specialized_losses.md` | Focal loss derivation |
| Transformer in DETR | `../../2.deep learning/02_architectures/04_transformer.md` | DETR uses encoder-decoder transformer |
| Anchor-free → keypoint detection | `03_segmentation.md` | Anchor-free idea extends to segmentation |
| Open-vocab + segmentation (Grounded-SAM) | `03_segmentation.md` | SAM + Grounding DINO pipeline |
| mAP metric | `../../1.machine learning/01_fundamentals/04_model_evaluation.md` | Evaluation framework |

---

## Key Takeaway

```
Detection = localization + classification
The hard part: handling thousands of candidate boxes efficiently

Two-stage (Faster R-CNN): propose → classify. Accurate but slow.
One-stage (YOLO/RetinaNet): predict all at once. Fast, competitive accuracy.
Focal loss: the key that unlocked one-stage detectors from class imbalance
FPN: the key that unlocked multi-scale detection

For production: YOLOv8 with custom training
For document layout detection: YOLOv8 with high input resolution (1280+)
and heavy augmentation for scan artifacts
```

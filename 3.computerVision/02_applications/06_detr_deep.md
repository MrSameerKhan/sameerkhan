# DETR — Object Detection with Transformers

> The architecture that killed anchor boxes and NMS. Set prediction reframes detection as a transformer task.

---

## Table of Contents

1. Objective
2. Core concept — set prediction with a transformer
3. The bipartite matching loss
4. Architecture in detail
5. Variants — Deformable DETR, DINO-DETR, RT-DETR
6. Failure modes
7. Interview questions (5)
8. Further reading

---

## 1. Objective

Before DETR (2020), object detection used:
- **Anchor boxes** — predefined box shapes/sizes at each spatial location
- **Region proposals** (Faster R-CNN) or **dense per-location predictions** (YOLO)
- **NMS** (non-max suppression) to dedupe overlapping detections

DETR (Carion et al. 2020) showed that a transformer can do detection END-TO-END:
- No anchors
- No NMS
- Single forward pass produces a set of N predictions
- Bipartite matching aligns predictions to ground truth

The result: a cleaner formulation, fewer hyperparameters, and the foundation of all modern transformer-based detection.

**Senior interview Q:** "Explain DETR. How does it eliminate NMS?"

---

## 2. Core Concept — Set Prediction with a Transformer

Detection = predict a **SET** of `(class, bounding_box)` tuples for each image.

**DETR's reframing:**
1. **CNN backbone** extracts feature map (e.g., ResNet-50 → 32×32×2048)
2. **Transformer encoder** processes the feature map (self-attention over all spatial locations)
3. **N learnable "object queries"** (e.g., N=100) attend to the encoder outputs via cross-attention
4. **Each object query outputs** one (class, bbox) prediction — independent of the others
5. Output: a set of N predictions, where some are "no object"

The set is predicted in parallel. Each query learns to specialize on a different location/scale during training.

### Why No NMS

Each object query produces ONE prediction. The queries don't communicate during inference (well, they do via self-attention, but that's part of the learning). At training time, the matching loss (next section) ensures each ground-truth object is assigned to exactly ONE query — different queries specialize to different objects. At inference, the model has learned to make non-redundant predictions.

**NMS is replaced by training-time bipartite matching.**

---

## 3. The Bipartite Matching Loss

This is the technical core.

```
Given: N predictions  {(p_i, b_i)}_{i=1..N}   where p_i = class probs, b_i = bbox
       M ground truths {(c_j, g_j)}_{j=1..M}   where c_j = class, g_j = bbox

Each ground truth must be matched to ONE prediction (M ≤ N;
unmatched predictions are labeled "no object")
```

### The Matching Cost

For each candidate (i, j) assignment:

```
cost(i, j) = -p_i(c_j)  +  λ_L1 · ||b_i - g_j||_1  +  λ_IoU · GIoU_loss(b_i, g_j)
```

### Solve Bipartite Matching via Hungarian Algorithm

Find the assignment σ: {1..M} → {1..N} that minimizes total cost. **The Hungarian algorithm solves this in O(N³) — fast for N=100.**

### The Training Loss

After matching:

```
L = Σ_{j=1..M} [ -log p_σ(j)(c_j)  +  λ_box(b_σ(j), g_j) ]  +  L_∅ if not matched = -log p_i(no_object)
```

Each ground truth is supervised by its matched prediction. Unmatched predictions are pushed to predict "no object." Predictions specialize automatically.

### Why This Works

- Each GT → one prediction (deterministic via Hungarian)
- No duplicate predictions for the same object → no NMS needed
- Each query learns to cover a specific "slot" in the prediction space

---

## 4. Architecture in Detail

```
Input image (3 × H × W)
        ↓
ResNet-50 backbone → feature map (C × H/32 × W/32), e.g., 2048 × 32 × 32
        ↓
• 1×1 conv to reduce dim
Feature map (d × 32 × 32), d=256
        ↓
• Flatten spatial + add 2D positional encoding
Feature sequence (1024 × 256)
        ↓
TRANSFORMER ENCODER (6 layers, 8 heads, self-attention)
        ↓
Encoded features (1024 × 256)
        ↓
TRANSFORMER DECODER (6 layers, 8 heads, cross-attention to encoded features)
        ↑
N=100 learnable "object queries" (100 × 256) — INPUTS to the decoder
        ↓
N output embeddings (100 × 256)
        ↓
Two MLP heads:
  - class_head: (N × num_classes + 1)   [+1 = "no object"]
  - bbox_head:  (N × 4)                 [cx, cy, w, h normalized]
```

### Training Time and Convergence

DETR famously needed **500 epochs to converge** on COCO — much slower than Faster R-CNN. Reasons: the matching loss is noisy early on (random assignments), and the model has to discover spatial structure from scratch. This is one of the main things successors fix.

---

## 5. Variants — Deformable DETR, DINO-DETR, RT-DETR

### Deformable DETR (Zhu et al. 2020)

Replace standard self-attention with **deformable attention** — each query attends to only K (e.g., 4) sampled points around its reference location, not the full feature map.

Result: **10× faster training** (50 epochs vs 500), better small-object performance, lower memory.

This is the practical DETR variant. Most production transformer detectors use Deformable DETR or its descendants.

### DINO-DETR (Zhang et al. 2022)

Adds:
- **Denoising training** — add noise to GT boxes and ask model to denoise them; stabilizes matching
- **Mixed query selection** — initialize queries from top-K backbone proposals
- **Contrastive denoising** — discriminate noisy GT from random noise

Result: SOTA on COCO around 2023.

### RT-DETR (Lv et al. 2023)

Real-time DETR. Engineering optimizations + redesigned encoder. Comparable to YOLO in speed (60+ FPS) while keeping DETR's clean formulation.

Used in some 2024-2025 production deployments — the "transformer detection that's fast enough."

### Open-vocab DETR Variants

- **OWL-ViT** — CLIP-style text encoder, can detect arbitrary categories via text prompt
- **Grounding DINO** — language-grounded object detection
- **DINO-X / Grounded-SAM** — combine grounding with segmentation

---

## 6. Failure Modes

1. **Slow training (vanilla DETR)** — 500 epochs is impractical. Fix: use Deformable DETR or RT-DETR variants.

2. **Small object detection is weak (vanilla DETR)** — fixed-size feature maps lose detail. Fix: multi-scale features (FPN-style) in Deformable DETR.

3. **Bipartite matching is unstable early in training** — predictions get matched to different GT each iteration; matching is noisy. Fix: denoising (DINO-DETR) provides stable targets.

4. **N is fixed and must exceed expected number of objects** — N=100 fails on images with >100 objects. Most settings use N=300 to be safe.

5. **Memory for global attention** — vanilla DETR's encoder has O(H²W²) attention on feature maps. High-res images blow this up. Deformable attention is the fix.

6. **No-object class imbalance** — most of the N queries predict "no object." Class weights or focal loss help.

---

## 7. Interview Questions (5)

**Q1: How does DETR eliminate the need for NMS?**

DETR uses bipartite matching during training: each ground-truth object is assigned to exactly ONE prediction via the Hungarian algorithm. Unmatched predictions are labeled "no object." Each of the N object queries learns to specialize on a distinct location/scale, so at inference the model produces non-redundant predictions. No duplicates — no NMS.

**Q2: What's the role of object queries in DETR?**

N learnable embeddings (e.g., 100 × 256) that serve as INPUTS to the transformer decoder. They cross-attend to the encoded image features and end to the encoder outputs. Each query learns to specialize on a different location/scale during training — some queries focus on small objects, others on large, some on the center of the image, etc.

**Q3: Why is vanilla DETR so slow to train, and how was it fixed?**

Due to: (1) full self-attention over the feature map is expensive; (2) bipartite matching is noisy early in training so each GT gets matched to different predictions per iteration. Deformable DETR fixes both by using sparse "deformable attention" (each query attends only to K sampled points) — 10× faster convergence.

**Q4: How does DETR compare to YOLO?**

YOLO: dense predictions at every spatial location + NMS. Fast, mature, well-optimized. DETR: end-to-end set prediction, no anchors/NMS, cleaner formulation but historically slower. RT-DETR has closed the speed gap in 2023-24. For production: YOLO if speed is critical and you have abundant tuning expertise; DETR/Deformable DETR for cleaner pipelines and better long-tail object handling.

**Q5: How does Grounding DINO extend DETR?**

Adds a text encoder (BERT-style). The detection features cross-attend to image features. Enables open-vocabulary detection — detect ANY described object, not just COCO's 80 classes. The prompt is a text query like "person riding bicycle." The text encoder (BERT-style) processes the caption; detection features cross-attend to image features. Outputs are boxes corresponding to text-described objects. Enables open-vocabulary detection — detect ANY described object.

---

## 8. Further Reading

- DETR (Carion et al. 2020) — arXiv:2005.12872 — the original
- Deformable DETR (Zhu et al. 2020) — arXiv:2010.04159
- DINO-DETR (Zhang et al. 2022) — arXiv:2203.03605
- RT-DETR (Lv et al. 2023) — arXiv:2304.08069
- Grounding DINO (Liu et al. 2023) — arXiv:2303.05499
- Hungarian algorithm — Kuhn 1955 (classical OR result, used by all DETR variants)

---

## Key Takeaway

```
DETR = CNN backbone + Transformer encoder-decoder + N object queries
     + bipartite matching loss (Hungarian algorithm)
     → no anchors, no NMS, end-to-end set prediction

Vanilla DETR: elegant but 500 epochs / slow / weak on small objects
Deformable DETR: deformable attention → 50 epochs, FPN-scale, practical
DINO-DETR: denoising training → stable matching → SOTA 2023
RT-DETR: engineering → real-time speed, YOLO competitive

Production choice:
  Speed: YOLO v8-v12 > RT-DETR > Deformable DETR > vanilla DETR
  Accuracy: DINO-DETR ≈ D-FINE > RT-DETR > YOLO (at full resolution)
  Open-vocab: Grounding DINO for zero-shot detection by text
```

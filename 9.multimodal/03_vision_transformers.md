# 03 — Vision Transformers (ViT, Swin, DINO, DeiT)

> How transformers work on images. Forward pass traced with numbers. Foundational for understanding CLIP, Donut, and LayoutLM v3.

---

## Why Vision Transformers?

```
CNN approach:
  Convolution: local receptive field, hierarchical features
  Problem: fixed local context — a pixel at top-left cannot attend directly
           attend to a pixel at bottom-right in early layers
  Inductive bias: translation equivariance baked in

Transformer approach:
  Self-attention: every patch attends to every other patch from layer 1
  Long-range dependencies captured immediately
  Less inductive bias → needs more data, but scales better with data/compute

Key empirical finding (ViT paper, 2020):
  CNNs win at small data (< 10M images)
  Transformers win at large data (> 10M images)
  At web scale (100M+): transformers dominate
```

---

## Part 1: ViT (Vision Transformer) — Full Forward Pass

### Architecture Summary

```
Model variants:
  ViT-B/16: Base, patch 16×16, 12 layers, 12 heads, d=768, 86M params
  ViT-B/32: Base, patch 32×32, 12 layers, 12 heads, d=768, 88M params
  ViT-L/16: Large, patch 16×16, 24 layers, 16 heads, d=1024, 307M params
  ViT-H/14: Huge, patch 14×14, 32 layers, 16 heads, d=1280, 632M params

We trace ViT-B/16 on a 224×224 image.
```

### Step 1: Patch Extraction and Embedding

```
Input image: 224×224×3 (RGB)
Patch size: 16×16

Number of patches: (224/16) × (224/16) = 14 × 14 = 196 patches
Each patch: 16×16×3 = 768 raw values (flattened)

Linear projection: E ∈ R^(768×768)  (patch_dim = model_dim)
  patch_i ∈ R^768  →  E · patch_i = embed_i ∈ R^768

Now we have 196 patch embeddings, each ∈ R^768.

Add learnable [CLS] token:
  cls_token ∈ R^768  (initialized randomly, learned during training)
  Prepended to patch sequence → 197 tokens total

Sequence: [CLS, patch_1, patch_2, ..., patch_196]
            1  +  196                           = 197 tokens × 768 dims
```

### Step 2: Position Embeddings

```
Transformers have no notion of order → must add position information.

ViT uses 1D learnable position embeddings:
  pos_embed ∈ R^(197×768) — one vector per position
  197 positions: 0=[CLS], 1=top-left patch, 2=next patch, ..., 196=bottom-right

Add to patch embeddings:
  input[i] = patch_embed[i] + pos_embed[i]   for i in 0..196

Why learned (not sinusoidal)?
  Images have 2D structure, not 1D. Learned pos embeddings can capture 2D
  proximity implicitly. Sinusoidal assumes 1D sequence.

Note: ViT with 2D positional embeddings slightly outperforms 1D,
      but the difference is small. Most implementations use 1D.
```

### Step 3: Transformer Blocks (12 layers)

```
Each block:
  LayerNorm + Multi-Head Self-Attention + residual add
  LayerNorm + FFN (MLP) + residual add

Multi-Head Self-Attention (12 heads, d=768, d_head=64):

  Q = x · W_Q    (197×768) · (768×768) = (197×768)
  K = x · W_K    (197×768) · (768×768) = (197×768)
  V = x · W_V    (197×768) · (768×768) = (197×768)

  Split into 12 heads: each head sees (197×64) slices

  For one head:
    A = Q_h · K_h^T / √64    (197×64) · (64×197) = (197×197)
    A = softmax(A)             attention weights, sum to 1 per row
    out_h = A · V_h            (197×197) · (197×64) = (197×64)

  Concatenate 12 heads: (197×768)
  Project: (197×768) · W_O (768×768) = (197×768)

Attention visualization insight:
  In lower layers: [CLS] attends to local patches (background, texture)
  In upper layers: [CLS] attends to semantically important patches
                   (for document: text regions, header, total amount)

FFN (position-wise MLP):
  Linear(768 → 3072) + GELU + Linear(3072 → 768)
  Applied independently to each of the 197 token positions
```

### Step 4: Classification Head

```
After 12 transformer blocks:
  Output: 197 tokens, each ∈ R^768

Take ONLY [CLS] token: cls_output ∈ R^768

Classification head:
  Linear(768, num_classes) = logits → softmax → probabilities

For ImageNet (1000 classes):
  logits ∈ R^1000 → softmax → [0.001, 0.002, 0.87, ...] → class 281 = "tabby cat"

For document classification (5 classes):
  logits ∈ R^5 → softmax → [0.73, 0.08, 0.05, 0.10, 0.02] → "invoice"
```

---

## Why Does [CLS] Token Represent the Whole Image?

```
At layer 1:  [CLS] attends to all 196 patches (its attention weights span all)
At layer 2:  [CLS] representation now contains info from all patches
             → [CLS] attends again to all patches using this richer representation
...
After 12 layers: [CLS] has iteratively aggregated information from the entire image

This is the "global aggregator" design:
  [CLS] is trained to summarize the image for the classification task.
  It's a learned aggregation, not max-pool or avg-pool.

Alternative: use avg-pool over all 196 patch tokens (sometimes used)
  Slightly different inductive bias: all patches equally weighted in final repr.
  Works similarly in practice.
```

---

## Part 2: Patch Size Trade-Off

```
Patch size 32×32: 49 patches → shorter sequence → faster, less memory
                  Less detail → lower accuracy on fine-grained tasks
                  Good for: large images, fast inference, coarse classification

Patch size 16×16: 196 patches → more detail → higher accuracy
                  Standard for most vision tasks
                  Good for: ImageNet-scale classification, detection

Patch size 14×14: 256 patches → even more detail
                  Used in ViT-H/14 (CLIP ViT-L/14) and DINOv2
                  Better for dense prediction (segmentation, detection)

Patch size 8×8:   784 patches → very fine-grained, very expensive
                  Rare — used for specialized high-resolution tasks

For documents: small patch size matters
  Text characters are 10-30 pixels tall
  32×32 patches may merge characters → loss of OCR-relevant features
  Donut uses Swin with 4×4 patches → very fine-grained
  LayoutLM v3 uses 16×16 patches → balanced
```

---

## Part 3: Swin Transformer

### Why Swin?

```
ViT problem: attention is O(n²) in sequence length
  224×224 with 4×4 patches = 3,136 tokens
  Attention: 3136² = ~10M operations per layer → very slow

Swin solution: window-based local attention within shifted windows
  Each token attends only to tokens in its W×W window (e.g., 7×7 = 49 tokens)
  Complexity: O(n) instead of O(n²) → scales to high resolution

Document AI relevance:
  Documents are large (2560×1920) → need high resolution → Swin is essential
  Donut uses Swin encoder for this reason
```

### Swin Architecture

```
Stage 1: patch 4×4 → H/4 × W/4 tokens, C=96
         + Swin blocks (window attention within 7×7 windows)

Stage 2: patch merge (2×2+1) → H/8 × W/8, C=192
         + Swin blocks

Stage 3: patch merge → H/16 × W/16, C=384
         + Swin blocks

Stage 4: patch merge → H/32 × W/32, C=768
         + Swin blocks

For 2560×1920 document (Donut):
  Stage 1: 640×480 = 307,200 tokens, C=96
  Stage 2: 320×240 = 76,800 tokens, C=192
  Stage 3: 160×120 = 19,200 tokens, C=384
  Stage 4: 80×60 = 4,800 tokens, C=768  ← final feature map
```

### Shifted Window Attention

```
Problem with plain window attention:
  Tokens at window boundaries can't attend to each other → no cross-window communication

Swin alternates two types of blocks:
  Block 1: Regular windows (partition starting from top-left)
  Block 2: Shifted windows (partition shifted by W/2, W/2)

Effect: information flows between adjacent windows over layers
  Layer 1: window A can't see window B
  Layer 2 (shifted): token at boundary of old A now shares window with boundary of old B
  → Information propagates across the image despite local attention

This gives Swin the best of both worlds:
  Efficiency of local attention (O(n))
  Global context (via shifted windows across layers)
```

---

## Part 4: DeiT — Training ViT Without Huge Data

```
Problem: ViT-B/16 needs 300M+ images to match ResNet (JFT dataset, Google-scale)
  Most organizations don't have 300M images.

DeiT (Data-efficient Image Transformers, Meta 2020):
  Trains ViT on ImageNet-1K ONLY (1.2M images) and matches CNN performance.
  Key technique: knowledge distillation from CNN teacher.

Architecture addition: distillation token
  ViT:   [CLS, patch_1, ..., patch_196]
  DeiT:  [CLS, dist_token, patch_1, ..., patch_196]  → adds 1 extra token

Training objective (two losses):
  1. Standard cross-entropy from [CLS] output (true labels)
  2. Distillation loss from [dist_token] output (match CNN teacher's predictions)

  L_total = 0.5 × L_CE(cls_output, true_label)
           + 0.5 × L_KD(dist_output, teacher_prediction)

Teacher: RegNet or EfficientNet (CNN pre-trained on ImageNet)

At inference: average predictions from [CLS] and [dist_token]
              + 1-2% accuracy boost over using [CLS] alone

Result: DeiT-B matches ResNet-50 on ImageNet-1K without extra data.
```

---

## Part 5: DINO — Self-Supervised ViT

### What DINO Does

```
DINO (Self-DIstillation with NO labels):
  Trains ViT using self-supervised learning — no labels required.

Key insight: self-supervised ViT learns explicit semantic segmentation
             even though no segmentation labels were provided.
             Attention maps naturally highlight "foreground objects."
```

### Training Mechanism

```
Two ViT networks: student and teacher (identical architecture)
  Student: updated via gradient descent
  Teacher: exponential moving average (EMA) of student
           teacher_weights = 0.996 × teacher_weights + 0.004 × student_weights
           No gradient to teacher. Teacher is more stable.

Input: single image → create two augmented views
  view_1: random crop + color jitter + blur (global, 224×224)
  view_2: different random crop + augmentations (local, 96×96)

Forward pass:
  student(view_2) → student_output ∈ R^(65536 prototypes, softmax)
  teacher(view_1) → teacher_output ∈ R^(softmax + centering)

Loss: cross-entropy(student_output, teacher_output)
  → train student to predict what teacher sees in different view

Centering: teacher output shifted by running mean to prevent collapse
  → all-one prediction (mode collapse) is prevented this way.

Result: both views of the same image → similar embeddings
        different images → different embeddings
        → semantic features emerge without labels
```

### DINO Attention Maps

```
Remarkable property: ViT self-attention heads in DINO segment objects.

For a document image:
  Head 1: attends to text regions
  Head 2: attends to table structures
  Head 3: attends to logos/images
  Head 4: attends to form fields / checkboxes

This happens without any segmentation supervision.
The model discovers that "text" and "tables" are semantically coherent groups.

Practical use: DINO attention maps as near-free segmentation
  threshold attention map of [CLS] = object mask
  Works surprisingly well for document layout analysis.

Code:
  outputs = model.get_intermediate_layers(image, n=1)
  attentions = model.get_last_selfattention(image)  # (1, heads, tokens, tokens)
  # attentions[0, :, 0, 1:] = [CLS] attention to each patch for each head
  # reshape to H×W = attention heatmap
```

### DINOv2

```
DINOv2 (Meta, 2023): DINO with improvements
  Curated training data (LVD-142M — 142M high-quality images)
  Additional objectives: iBOT (patch-level masked modeling)
  Larger models: ViT-g (1B params)

Result: DINOv2 features work out-of-the-box for:
  Depth estimation, semantic segmentation, classification
  WITHOUT any fine-tuning — just k-NN or linear probe on top

Document AI: DINOv2 features excellent for document layout analysis
  Better than ImageNet-pretrained ViT for document similarity
```

---

## Part 6: Attention in Vision — What Gets Attended

```
Layer-by-layer attention analysis (ViT-B/16, 224×224):

Layers 1-3 (early):
  [CLS] attends broadly to all patches
  Attention: diffuse, ~1/196 ≈ 0.5% per patch
  What's learned: low-level textures, edges, colors

Layers 4-7 (middle):
  Attention starts concentrating on semantically meaningful regions
  For invoice: starts focusing on text-heavy areas
  Heads specialize: some for horizontal, some for vertical patterns

Layers 8-12 (late):
  [CLS] attends sharply to the most diagnostic patches (5-15 patches dominate)
  For invoice: header area (invoice number, date), total amount area
  Attention entropy decreases: more focused, more selective

Attention head diversity:
  12 heads × 12 layers = 144 attention patterns
  Some heads: nearby patches (local texture)
  Some heads: same-row patches (reading direction)
  Some heads: [CLS] + key content patches (global understanding)
  DINO makes this specialization explicit and interpretable
```

---

## Part 7: ViT vs CNN — Decision Guide

```
Use ViT when:
  Large dataset (> 1M images) or large pre-trained model available
  Need long-range dependencies (whole-document understanding)
  Task benefits from attention interpretability
  Modality fusion (combine with text — CLIP, LayoutLM v3)
  High-resolution inputs (with Swin variant)

Use CNN when:
  Small dataset (< 100K images) without pre-training
  Need translation equivariance (object detection)
  Strict inference speed requirements (edge deployment)
  Well-understood, stable training dynamics needed

Hybrid (ConvMixer / EfficientNet):
  CNN designed with ViT design choices (large kernels, fewer norms, GeLU)
  Best of both: CNN efficiency + ViT-inspired architecture
  Often competitive with ViT at lower compute

For Document AI specifically:
  Swin = Donut encoder (high resolution, efficient)
  ViT-B/32 = CLIP image encoder (fast, 49 patches)
  ViT-L/14 = CLIP ViT-L (slower, better features)
  ResNet + LayoutLM v1/v2 image features
  ViT-B/16 + LayoutLM v3 image patches
```

---

## Part 8: Key Numbers to Memorize

```
ViT-B/16:
  Input: 224×224 → 196 patches (16×16) + 1 CLS = 197 tokens
  d_model = 768, heads = 12, d_head = 64, layers = 12, FFN = 3072
  Params: 86M

ViT-B/32:
  Input: 224×224 → 49 patches (32×32) + 1 CLS = 50 tokens
  Same model dims as ViT-B/16
  Faster: 196→49 tokens → 4× fewer attention operations

Swin-B (Donut encoder):
  Input: 2560×1920 → 640×480 initial patches (4×4)
  Stage 2: 320×240 = 76,800; Stage 3: 160×120 = 19,200
  Stage 4: 80×60 = 4,800 tokens → 768-dim final vectors
  Window size: 7×7 = 49 local tokens per window

CLIP ViT-B/32:
  Image: 224×49 = 512-dim embedding
  Text: 77 tokens = 512-dim embedding
  Pre-trained: 400M image-text pairs from internet

DINOv2 ViT-g:
  1B parameters, patch 14×14
  Features work zero-shot for depth, segmentation, classification
```

---

## Part 9: Interview Questions

**Q: How does ViT handle images, and how is it different from a CNN?**

ViT splits the image into fixed-size patches (e.g., 16×16), flattens and linearly projects each patch into a d-dimensional embedding, adds a [CLS] token and learned position embeddings, then runs a standard transformer encoder.

The key difference from CNN:
- CNN: local convolution + hierarchical features, translation equivariant, O(n) compute
- ViT: global self-attention from layer 1 → every patch can attend to every other patch. No translation equivariance (position embeddings are learned). O(n²) compute in sequence length → needs Swin for high-resolution images.

ViT wins at large scale; CNN wins on small datasets.

---

**Q: What is the role of the [CLS] token in ViT?**

[CLS] is a learnable token prepended to the patch sequence. It has no spatial meaning — its job is to aggregate information from all patches.

Through self-attention, [CLS] attends to all 196 patches at every layer. After 12 layers, [CLS] representation contains a global summary of the image. A linear classification head on [CLS] predicts the image class.

Alternative: average-pool all patch embeddings at the final layer. Both work similarly — [CLS] gives the model a dedicated "summary" slot.

---

**Q: Why does Donut use Swin instead of plain ViT?**

Document images are high resolution (2560×1920 or larger). Plain ViT with 16×16 patches would create (2560/16)×(1920/16) = 160×120 = 19,200 tokens. Self-attention on 19,200 tokens = 368M operations per layer → infeasible.

Swin uses 4×4 patches (su 640×480 = 307K tokens initially) but restricts attention to local 7×7 windows → only 49 tokens per attention computation. Shifted windows propagate information globally without quadratic cost. Final feature map: 80×60 = 4,800 tokens at 2560×1920 resolution.

Result: high-resolution document processing at practical compute cost.

---

**Q: What makes DINO attention maps useful for document layout?**

DINO trains ViT with self-supervised learning where the model must predict one augmented view from another. Without labels, the model discovers that semantically coherent regions should have similar representations.

In documents, text runs form coherent visual patterns, tables have regular structure, headers are visually distinct. DINO heads naturally specialize to these patterns: one head for text regions, one for tables, one for separators, etc.

Practical use: take [CLS]-patch attention weights, reshape to H×W grid, threshold = rough segmentation of document into layout regions. No segmentation labels needed.

---

## Key Takeaway

```
ViT: image → N patches → linear project → [CLS] + patch tokens → 12-layer transformer
     → [CLS] output → linear → class prediction
     ViT-B/16: 196 patches, 86M params. Global attention from layer 1.

Swin: hierarchical ViT with window attention (O(n) not O(n²))
      Stages reduce spatial resolution, increase channels via 4 Swin stages.
      Donut encoder: 2560×1920 → 4,800 final vectors

DeiT: ViT trained on ImageNet-1K (not 300M) using distillation token
      + CNN teacher → matches large-scale ViT with 1.2M images.

DINO: self-supervised ViT. No labels. Teacher-student with EMA training.
      Learns semantic segmentation implicitly. Attention maps highlight objects.
      DINOv2: 1B param model, zero-shot features for depth/segmentation/classification.

For Document AI:
  CLIP uses ViT-B/32 (fast, 49 patches, 512-dim)
  LayoutLM v3 uses ViT-B/16 patches (224×224, 196 patches)
  Donut uses Swin (2560×1920, window attention, 4,800 final vectors)
```

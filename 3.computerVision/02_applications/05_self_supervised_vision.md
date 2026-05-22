# Self-Supervised Vision — SimCLR, MoCo, DINO, BYOL, MAE

> Train strong vision features without labels. The foundation of every modern vision foundation model (CLIP, DINOv2, SAM all use self-supervised pretraining).

---

## Table of Contents

1. Objective
2. Core concept — pretext tasks
3. Contrastive methods (SimCLR, MoCo)
4. Non-contrastive methods (BYOL, DINO, MAE)
5. Comparison — what to use
6. Failure modes
7. Interview questions (5)
8. Further reading

---

## 1. Objective

Supervised vision pretraining (ImageNet-1k labels) was the standard until ~2020. Then self-supervised methods caught up, and on many downstream tasks they **BEAT** supervised pretraining.

**Why?** Labels are expensive and noisy. Self-supervised methods use unlimited unlabeled images. They learn more general features that transfer better.

**Senior interview Q:** "How do you train good vision features without labels?" or "What is contrastive learning?"

---

## 2. Core Concept — Pretext Tasks

Self-supervised learning constructs a "pretext task" — one that requires LEARNING good features but doesn't need labels.

**Three families:**
1. **Contrastive** — pull two views of the same image together, push different images apart (SimCLR, MoCo)
2. **Self-distillation** — student network predicts teacher network's representations (BYOL, DINO)
3. **Reconstruction** — mask part of the image, predict it (MAE, MaskFeat)

All three produce strong frozen features that transfer to classification, detection, segmentation.

---

## 3. Contrastive Methods (SimCLR, MoCo)

### SimCLR (Chen et al. 2020)

The cleanest contrastive method.

```
For a batch of N images:
  1. Apply 2 random augmentations to each image (crop, color jitter, blur)
     → 2N augmented views total

  2. Encode all 2N views:  z_i = projection_head(encoder(view_i))

  3. For each (z_i, z_j) pair from the same source image (POSITIVE):
     l_ij = -log [ exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ) ]
     where sim = cosine similarity, τ = temperature, sum over all 2N-1 negatives

  4. Total loss = average l_ij over all positive pairs
```

**The InfoNCE loss.** Pulls positive pairs together, pushes 2N-2 negatives apart per sample.

### Why Temperature τ Matters

```
Low τ (e.g., 0.05): hard distinctions, slow convergence, strong features
High τ (e.g., 0.5): soft distinctions, fast convergence, weaker features
Default: 0.07-0.1
```

### Why the Projection Head Matters

SimCLR adds a 2-layer MLP **after** the encoder, only for the contrastive loss. Final features = encoder output, NOT projection output. The projection head absorbs the contrastive-specific representation; the encoder learns general features.

### Why a Big Batch Matters

SimCLR with batch size 256: poor features. Batch size 4096+: SOTA features. More negatives in the batch → harder contrastive task → better features. **Big-batch GPU clusters are SimCLR's hidden requirement.**

### MoCo (He et al. 2019)

Addresses SimCLR's batch-size limitation. Instead of in-batch negatives, maintain a **queue of negative features** from past batches. Momentum-update the encoder used to compute queue features for stability.

Result: trains with batch 256, has effectively 65K negatives from the queue. Same quality as SimCLR with much smaller batches.

**MoCo v3 (2021)** drops the queue, adopts SimCLR-like loss, but keeps momentum encoder for ViTs.

---

## 4. Non-Contrastive Methods (BYOL, DINO, MAE)

These avoid the big-batch / negative-mining problem entirely.

### BYOL (Grill et al. 2020) — "Bootstrap Your Own Latent"

Two networks: student and teacher (EMA copy of student).

```
1. Two augmented views x_1, x_2 of the same image
2. Student computes prediction:  z_1 = predictor(projection(student(x_1)))
3. Teacher computes target:      z_2 = projection(teacher(x_2)).detach()
4. Loss = MSE(normalize(z_1), normalize(z_2))
5. Teacher = momentum-EMA(student)  — updated, not trained
6. Repeat with x_1 ↔ x_2 swapped
```

**No negatives needed.** The student tries to predict the teacher's representation of a different view. The teacher's stability + momentum + asymmetric architecture (only student has the predictor) prevents collapse.

**Counter-intuitive:** why doesn't this collapse to z_1 = z_2 = constant? The asymmetric design + batch normalization + EMA teacher are conjectured to be the keys. Active research.

### DINO (Caron et al. 2021) — Self-distillation with No Labels

Similar to BYOL but with subtle differences:
- Output is a softmax distribution, not just a vector
- Cross-entropy loss instead of MSE
- Centering + sharpening tricks prevent collapse
- **Trained on ViTs** (vs BYOL on ResNets)
- Discovered surprising property: **attention maps highlight objects without supervision**

**DINOv2 (2023)** scaled this up — became the de facto open-source vision foundation model in 2023-2025.

### MAE — Masked Autoencoder (He et al. 2021)

The vision analog of BERT's MLM.

```
1. Mask 75% of image patches randomly
2. Encode the 25% visible patches only (efficient; encoder sees less)
3. Decode all patches (visible + masked) using a small decoder
4. Loss = pixel reconstruction error on the masked patches
```

Genius part: encoder only processes 25% of patches → 4× faster pretraining. Yet representations are excellent for downstream fine-tuning.

**MAE is the default vision pretraining method as of 2024.**

---

## 5. Comparison — What to Use

| Method | Batch needs | Compute | Output quality | When |
|--------|------------|---------|---------------|------|
| SimCLR | Huge (4096+) | High | Strong | If you have a big-batch cluster |
| MoCo v3 | Moderate (256) | Moderate | Strong | General-purpose, ViT-friendly |
| BYOL | Small (256) | Moderate | Strong | When negatives are hard to define |
| DINO/DINOv2 | Moderate | High | **Strongest features** | Vision foundation models |
| MAE | Big (32k+) | **Lowest** | Strong, esp. for fine-tuning | Pretraining efficiently |
| CLIP-style | Big (32k+) | Very high | Strong + text-aligned | Multimodal applications |

**2026 production defaults:**
- Need a frozen feature extractor → **DINOv2** (already trained, just use it)
- Need to pretrain on your own data → **MAE**
- Need text-aligned features → **CLIP / SigLIP**
- Research / new domain → **MoCo v3 or DINO**

---

## 6. Failure Modes

1. **Collapse** — all images map to the same representation, loss = 0, useless features. Mitigations: large batch (SimCLR), momentum teacher (BYOL, DINO), centering (DINO).

2. **Augmentation sensitivity** — features become invariant to whatever you augment. If you flip horizontally, features won't distinguish left-right. Choose augmentations carefully — they encode INVARIANCES into the features.

3. **Domain mismatch** — pretrained on natural images, applied to X-rays — features are bad. Always pretrain on your domain if possible (DINOv2-medical exists).

4. **Linear evaluation is misleading** — feature quality is usually measured via "freeze features, train a linear classifier on labels, report accuracy." Linear-probe accuracy can saturate while fine-tune accuracy still improves.

5. **Large batch = environmental cost** — SimCLR's "use batch 4096" requires 64-128 GPUs. Methods like MoCo, BYOL, MAE explicitly try to avoid this.

---

## 7. Interview Questions (5)

**Q1: What is contrastive learning?**

Train an encoder so that two augmented views of the same image produce similar representations (positive pair), while views from different images are dissimilar (negatives). InfoNCE loss formalizes this as a softmax classification across positives vs negatives: for each anchor, predict which of the 2N-1 other representations is its augmented pair. Used in SimCLR and CLIP.

**Q2: Why does SimCLR need such large batches?**

The batch size determines how many negatives each positive pair competes against. With batch 256, you have ~510 negatives per anchor. With batch 4096, ~8190. More negatives = harder contrastive task = better features. SimCLR feature quality scales with batch size — that's why MoCo / BYOL exist as low-batch alternatives.

**Q3: Why doesn't BYOL collapse?**

This is a known puzzle. By naive analysis it should: student just predicts the teacher's output from a different view (still being researched). EMA teacher provides stability; the predictor head + asymmetry between student and teacher creates an implicit contrastive signal; and batch normalization plays an important role.

**Q4: How is MAE different from contrastive methods?**

MAE is a reconstruction-based method: mask 75% of patches, encode the rest, decode all patches, compute pixel-level reconstruction loss. No contrastive loss, no positives/negatives. Only 25% of patches are encoded — 4× cheaper than methods that process the full image. Has become the default vision pretraining method in 2024.

**Q5: When would you use DINO/DINOv2 features over a supervised ImageNet model?**

When your downstream task has limited labels, or when you need general-purpose features (not just classification — segmentation, dense prediction tasks). DINOv2 features tend to transfer better to detection/segmentation tasks. Also when your domain differs from ImageNet. Modern multimodal models (LLaVA, GPT-4V) use DINOv2 or similar as their vision encoder.

---

## 8. Further Reading

- SimCLR (Chen et al. 2020) — arXiv:2002.05709
- MoCo (He et al. 2019) — arXiv:1911.05722
- MoCo v3 (Chen et al. 2021) — arXiv:2104.02057
- BYOL (Grill et al. 2020) — arXiv:2006.07733
- DINO (Caron et al. 2021) — arXiv:2104.14294
- DINOv2 (Oquab et al. 2023) — arXiv:2304.07193
- MAE (He et al. 2021) — arXiv:2111.06377
- InfoNCE (Oord et al. 2018) — arXiv:1807.03748 — the loss function origin

---

## Key Takeaway

```
Self-supervised pretraining = learn from unlabeled data via pretext tasks

Three families:
  Contrastive (SimCLR, MoCo):  same image → close; different images → far
  Self-distillation (BYOL, DINO):  predict teacher's view, no negatives needed
  Reconstruction (MAE):  mask patches, reconstruct them

Production 2024-25:
  DINOv2 = best frozen features (zero-shot to any vision task)
  MAE    = best pretraining when you have domain-specific unlabeled data
  CLIP   = needed when text-image alignment matters

The features learned are general: one pretrained ViT serves
classification, detection, segmentation, retrieval, VQA.
```

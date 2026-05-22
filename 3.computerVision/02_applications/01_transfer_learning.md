# 01 — Transfer Learning

## Quick Reference

| Strategy | Freeze | Train | When |
|----------|--------|-------|------|
| Feature extraction | All backbone | Head only | Very small dataset (<1K images), similar domain |
| Partial fine-tuning | Early layers | Last N layers + head | Medium dataset (1K-10K), similar domain |
| Full fine-tuning | Nothing | Everything | Large dataset (>10K) or very different domain |
| Linear probing | All backbone | Linear head only | Evaluating backbone quality, SSL embeddings |

**Rule:** Start with feature extraction. Unfreeze progressively if performance plateaus.

---

## 1. Why Transfer Learning

### The Problem Without It

```
ImageNet training from scratch:
  Dataset: 1.2M images, 1000 classes
  Hardware: 8 V100 GPUs
  Time: ~1 week
  Cost: ~$1000+ cloud compute

Your task: classify 500 X-ray images + 2 classes
Training from scratch: impossible (not enough data, no compute)
```

### What Gets Transferred

```
ImageNet pre-trained CNN feature hierarchy:

Layer 1-2 (early):   edges, gradients, color blobs  = GENERIC   → transfer to anything
Layer 3-4 (middle):  textures, patterns, parts       = SEMI-GENERIC → useful for most image tasks
Layer 5+  (late):    class-specific features (dog ears, car wheels) = TASK-SPECIFIC → replace

Key insight: Early features are universal. Reuse them. Only retrain task-specific top layers.
```

### Why It Works (Domain Perspective)

```
Source domain (ImageNet): 1000 diverse object classes
Target domain (medical):  2 classes (normal/abnormal)

Despite very different objects, the underlying visual primitives (edges, textures, gradients)
appear in all natural images → early layers transfer perfectly.

Exception: very non-natural images (satellite, hyperspectral, molecular)
→ early layers still transfer, but less effectively than for natural images.
```

---

## 2. Feature Extraction

**What It Means:** Remove the classification head (final FC layer). Freeze all backbone weights. Add new head. Train only the new head.

```
Pretrained ResNet-50:
  [Conv blocks]  = Frozen, weights fixed
  [GAP]          = Frozen
  [FC 1000]      = REMOVE

New head:
  [FC 512] + [ReLU] + [Dropout 0.5] + [FC n_classes] + Softmax
```

```python
import torch
import torch.nn as nn
import torchvision.models as models

def build_feature_extractor(backbone_name='resnet50', num_classes=10, dropout=0.5):
    # Load pretrained backbone
    backbone = models.resnet50(weights='IMAGENET1K_V2')
    # Freeze ALL backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    # Replace final FC
    in_features = backbone.fc.in_features   # 2048 for ResNet-50
    backbone.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return backbone

model = build_feature_extractor(num_classes=5)
# Confirm only head is trainable
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
# Typical: Total: 25M | Trainable: 10K (0.04%)
```

### Extracting Features as Vectors (Offline)

```python
# Extract backbone features for all images once, then train a simple classifier
backbone = models.resnet50(weights='IMAGENET1K_V2')
backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove FC
backbone.eval()

features, labels = [], []
with torch.no_grad():
    for images, lbls in dataloader:
        feats = backbone(images).flatten(1)   # [B, 2048]
        features.append(feats.cpu())
        labels.append(lbls)

features = torch.cat(features)   # [N, 2048]
labels   = torch.cat(labels)

# Now train any sklearn classifier on these 2048-dim features
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(features.numpy(), labels.numpy())
```

**When to use offline extraction:** dataset fits in memory, you want to iterate fast over head architectures without re-running backbone each time.

---

## 3. Fine-Tuning

### Partial Fine-Tuning (Most Common)

Unfreeze the last N convolutional blocks + head. Keep early layers frozen (they're already optimal for low-level features).

```
ResNet-50 structure:
  layer1 (early edges)   → FREEZE
  layer2 (textures)      → FREEZE
  layer3 (parts)         → UNFREEZE (if enough data)
  layer4 (task features) → UNFREEZE
  GAP + FC (head)        → UNFREEZE always
```

```python
def build_fine_tuned(backbone_name='resnet50', num_classes=10, unfreeze_from='layer3'):
    backbone = models.resnet50(weights='IMAGENET1K_V2')
    # Start fully frozen
    for param in backbone.parameters():
        param.requires_grad = False
    # Unfreeze from specified layer onward
    unfreeze = False
    for name, module in backbone.named_children():
        if name == unfreeze_from:
            unfreeze = True
        if unfreeze:
            for param in module.parameters():
                param.requires_grad = True
    # Replace head (always trainable)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone
```

### Full Fine-Tuning

```python
def build_full_finetune(num_classes=10):
    backbone = models.resnet50(weights='IMAGENET1K_V2')
    # ALL layers trainable, but use discriminative LR
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone
```

### Discriminative Learning Rates (Critical)

Different LRs for different layers — early layers need tiny updates, head needs larger updates.

```python
import torch.optim as optim

model = build_full_finetune(num_classes=4)
# Group parameters by layer depth
param_groups = [
    {'params': model.layer1.parameters(), 'lr': 1e-5},   # early = small LR
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3},   # head = large LR
]
optimizer = optim.Adam(param_groups)
```

---

## 4. Progressive Unfreezing (Recommended Workflow)

The safest strategy for maximum performance:

```
Phase 1: Feature extraction (head only)
  - Freeze backbone, LR=1e-3, train 10-20 epochs
  - Lets head converge without disturbing backbone

Phase 2: Unfreeze last block (layer4)
  - LR=1e-4 for layer4, LR=1e-3 for head
  - Train until convergence (~10 epochs)

Phase 3: Unfreeze more layers
  - LR=1e-5 for early layers, 1e-4 for later, 1e-3 for head
  - Train to final convergence

Why: if you unfreeze all at once with high LR = "catastrophic forgetting"
— pretrained weights overwritten before head is trained, performance collapses
```

```python
def train_phase(model, dataloader, epochs, optimizer):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

# Phase 1
model = build_feature_extractor(num_classes=5)
opt1  = optim.Adam(model.fc.parameters(), lr=1e-3)
train_phase(model, train_loader, epochs=15, optimizer=opt1)

# Phase 2: unfreeze layer4
for param in model.layer4.parameters():
    param.requires_grad = True
opt2 = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),     'lr': 1e-3}
])
train_phase(model, train_loader, epochs=10, optimizer=opt2)
```

---

## 5. Data Augmentation for Transfer Learning

Small datasets — heavy augmentation essential to prevent overfitting.

```python
from torchvision import transforms

# Standard augmentation for transfer learning
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # random crop + resize
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],           # ImageNet stats
                         [0.229, 0.224, 0.225])
])

# Validation: only normalize (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

**Always use ImageNet normalization** when using ImageNet-pretrained backbones — the weights expect this exact normalization.

### Mixup and CutMix (Advanced)

```python
def mixup_batch(x, y, alpha=0.4):
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
```

---

## 6. When to Use Which Strategy

| Dataset Size | Domain Similarity | Strategy | LR |
|---|---|---|---|
| Very small (<500) | Similar (natural images) | Feature extraction only | 1e-3 (head) |
| Small (500-5K) | Similar | Partial fine-tune (layer4+head) | 1e-4/1e-3 |
| Medium (5K-50K) | Similar | Full fine-tune + discriminative LR | 1e-5/1e-4/1e-3 |
| Large (>50K) | Similar | Full fine-tune + uniform LR | 1e-4 uniform |
| Any size | Very different (satellite, medical) | Full fine-tune, progressive unfreeze | 1e-5 early, 1e-3 head |
| Small | Very different | Start from feature extraction, evaluate before unfreezing | — |

---

## 6.5 Modern Backbones — Beyond ImageNet-Supervised Pretraining

The textbook story (ImageNet-pretrained ResNet → fine-tune) is still useful, but the strongest 2024-25 baselines start from a **self-supervised vision foundation model**:

| Source backbone | Pretraining | When to use |
|---|---|---|
| DINOv2 (ViT-S/B/L/g) | Self-distillation on 142M curated images (no labels) | Default for any vision task — frozen DINOv2 + linear/MLP head usually beats fine-tuned supervised ResNet-50 |
| MAE (ViT-B/L/H) | Masked patch reconstruction | When you have unlabeled in-domain data to continue pretraining |
| CLIP / SigLIP | Image-text contrastive | Zero-shot classification, retrieval, or as feature extractor when class names matter |
| SAM image encoder | Promptable segmentation | Strong for segmentation tasks; sometimes good for classification too |
| Supervised ImageNet (ResNet-50, ConvNeXt) | Classic, still solid | Documented, fast, easy to debug |

```python
# DINOv2 in 5 lines — modern feature extractor
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval()
features = dinov2(images)   # [B, 768] CLS token
# Train a small classifier on top — often beats fine-tuned ResNet-50
```

### Parameter-Efficient Fine-Tuning (PEFT) for Vision

Full fine-tuning of large ViT/DINOv2 (300M-1B params) is expensive. LoRA and adapter-based methods from NLP transfer to vision:

| Method | Trainable params | Approach |
|---|---|---|
| LoRA | ~0.5% of model | Add low-rank update matrices to attention Q/V |
| VPT (Visual Prompt Tuning) | ~0.1% | Prepend learnable prompt tokens; freeze ViT |
| AdaptFormer | ~1% | Parallel adapter modules in transformer blocks |
| BitFit | ~0.05% | Only train bias terms |

```python
# LoRA on a vision transformer (via peft library)
from peft import LoraConfig, get_peft_model
config     = LoraConfig(r=8, target_modules=["qkv"], lora_alpha=16)
peft_model = get_peft_model(vit_model, config)
peft_model.print_trainable_parameters()   # ~0.5% trainable
```

**When PEFT wins:** large pretrained model + small target dataset (avoids catastrophic forgetting), or multi-task settings where you want one model + many lightweight adapters.

---

## 7. Domain Adaptation

When source and target domains differ significantly:

### Maximum Mean Discrepancy (MMD)

Add a loss term that minimizes the distribution shift between source and target feature spaces:
```
Total Loss = Task Loss + λ · MMD(source_features, target_features)
```

### DANN (Domain-Adversarial Neural Network)

Train a domain classifier that predicts source vs target domain. Then reverse its gradient — backbone learns domain-invariant features.

### Practical Approaches

```
1. Simple domain adaptation: fine-tune with target data
   (even unlabeled — use pseudo-labels or self-supervised objectives)

2. Self-supervised pre-training on target domain (no labels needed)
   → Then fine-tune supervised on labeled target

3. SimCLR-style pretraining on target domain
   → Model learns target domain features without labels → then fine-tune
```

---

## 8. Gotchas

**Forgetting ImageNet normalization.** If using ImageNet-pretrained weights, always normalize with ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Using wrong normalization → backbone receives unexpected inputs → bad features → poor accuracy.

**Catastrophic forgetting on small LR.** Even with LR=1e-4, training for 100 epochs on a small dataset overwrites pretrained features. Use early stopping or cosine annealing with warmup.

**BatchNorm running stats during freezing.** When backbone is frozen, BatchNorm layers in frozen blocks should be set to eval mode — they use running stats, not batch stats. Otherwise BN tracks stats for the frozen layers during training and running stats drift.

```python
def freeze_backbone_properly(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and 'fc' not in name:
            module.eval()   # use running stats, not batch stats
            for param in module.parameters():
                param.requires_grad = False
```

**Not using the right pretrained weights.** `models.resnet50(pretrained=True)` uses IMAGENET1K_V1 (old). `weights='IMAGENET1K_V2'` uses improved weights (+2-3% accuracy). Always specify the weights version explicitly.

**Unfreezing too fast (catastrophic forgetting).** Going from feature extraction directly to full fine-tune with LR=1e-3 → wipes pretrained weights. Always use discriminative LRs and reduce LR for backbone layers.

---

## 9. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Accuracy stays at random chance despite training | Wrong normalization; head not connected | Check ImageNet normalization; verify grad flow |
| Fast overfit despite small LR | Too many trainable params for dataset size | Freeze more layers; add Dropout |
| Val loss worse than training from scratch | Catastrophic forgetting | Reduce LR; freeze earlier layers; progressive unfreezing |
| BN layers behave differently train vs eval | Frozen BN still updating stats | Set frozen BN to `.eval()` mode |
| Low accuracy on very different domain | Not enough target-domain signal | More fine-tuning epochs; consider self-supervised pretraining on target |
| Slow training despite GPU | Backbone weights still stored on GPU unnecessarily | Feature extraction: compute features offline, cache to disk |

---

## 10. Code Reference — Full Pipeline

```python
import torch, torchvision
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Data
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_ds    = datasets.ImageFolder('data/train', train_transform)
val_ds      = datasets.ImageFolder('data/val',   val_transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

# Model (feature extraction)
model = models.resnet50(weights='IMAGENET1K_V2')
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(train_ds.classes.__len__() * 0 + 2048, len(train_ds.classes)))

device    = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model     = model.to(device)

# Training
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss()

best_val_acc = 0
for epoch in range(20):
    model.train()
    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Validation
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            preds   = model(imgs).argmax(1)
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
```

---

## 11. Interview Q&A (Senior Level)

**Q: When would you fine-tune vs train from scratch?**

A: Fine-tune in almost all practical cases. Train from scratch only when: (1) your domain is fundamentally different from any available pretraining source (rare: custom sensor modalities, proprietary data types), (2) you have truly massive labeled data (millions of images) where fine-tuning provides no benefit, (3) you need a fully custom architecture not available pretrained. Even in medical imaging where images look very different from ImageNet, fine-tuned models typically outperform scratch-trained ones because early-layer edge/texture detectors are universal across natural image statistics.

**Q: What is catastrophic forgetting and how do you prevent it in fine-tuning?**

A: Catastrophic forgetting: when training on a new task, model overwrites weights for old task — pretrained features destroyed. Prevention: (1) Discriminative learning rates — much lower LR for early layers (1e-5) than head (1e-3). (2) Progressive unfreezing — train head first, then gradually unfreeze from top to bottom. (3) L2-SP (L2 toward starting point) — regularize weights toward pretrained initialization rather than toward zero. (4) Elastic Weight Consolidation (EWC) — penalize changes to weights important for source task. In practice: discriminative LR + progressive unfreezing is sufficient for most fine-tuning scenarios.

**Q: Why must BatchNorm layers be set to eval mode when the backbone is frozen?**

A: When `requires_grad=False`, the BN parameters (γ, β) don't update, which is intended. But BN has two modes: train mode (computes batch statistics μ/σ from current batch, updates running stats) and eval mode (uses stored running statistics). During training with frozen backbone, BN is in train mode by default — it computes new statistics from each batch and updates running_mean/running_var. This corrupts the running statistics that were carefully accumulated during ImageNet training. These corrupted stats are then used at inference time → wrong normalization → degraded predictions. Fix: `module.eval()` on all frozen BN layers so they use the correct pretrained running stats.

---

## 12. Connections

| This file | Links to | Why |
|-----------|---------|-----|
| Backbone architectures | `../01_fundamentals/02_cnn_architectures.md` | Which backbone to choose |
| ViT / Swin / DeiT depth | `../01_fundamentals/04_vision_transformer_deep.md` | Modern attention backbones |
| Self-supervised foundation models | `05_self_supervised_vision.md` | DINOv2, MAE, I-JEPA, CLIP — primary alternative to ImageNet-supervised |
| Data augmentation basics | `../01_fundamentals/01_cnn_mechanics.md` | Augmentation for small datasets |
| Mixup / CutMix / RandAugment | `../../2.deep learning/01_fundamentals/04_generalization.md` | Modern aug recipe for ViT/ConvNeXt |
| Focal loss for imbalanced | `../../2.deep learning/01_fundamentals/06_specialized_losses.md` | Fine-tuning with class imbalance |
| Feature extraction → sklearn | `../../1.machine learning/02_algorithms/02_tree_models.md` | Use extracted features with XGBoost |
| LoRA fine-tuning (NLP origin) | `../../6.llms/` | Same PEFT pattern across modalities |
| Domain adaptation in NLP | `../../4.nlp/` | BERT fine-tuning = same concept |

---

## Key Takeaway

```
The transfer learning hierarchy:
  Feature extraction (fastest, safest)
    → Partial fine-tune (last 1-2 blocks + head)
      → Full fine-tune with discriminative LR (best accuracy)

Critical success factors:
  1. ImageNet normalization — always
  2. Frozen BN in eval mode — don't forget
  3. Discriminative LR — backbone gets 10-100× smaller LR than head
  4. Progressive unfreezing — don't unfreeze all at once

Your domain (document automation): ImageNet→document is a big domain shift.
Best strategy: ResNet-50 feature extraction → evaluate → partial fine-tune layer4+head
→ full fine-tune if dataset > 5K images.
Heavy augmentation: perspective transforms, blur, JPEG compression artifacts
(simulate real scan conditions).
```

# Semi-Supervised Learning — Deep Learning Perspective

> **Prerequisites:** `1.machine learning/02_algorithms/06_semi_supervised_learning.md` (self-training, label propagation, co-training)  
> **This file:** SSL with deep neural networks — contrastive learning, VAEs, BERT-style pre-training as SSL.

---

## Why Deep SSL Is Different

```
Classical SSL (label propagation, co-training):
  Works on fixed hand-crafted features
  Graph built in original feature space
  Scales to ~10K unlabeled samples

Deep SSL:
  Features ARE learned — the backbone learns from unlabeled data
  Can leverage millions of unlabeled samples
  The pre-training IS the semi-supervised step
```

---

## Self-Supervised Pre-training (The Modern Approach)

The dominant paradigm: pre-train on unlabeled data with a pretext task, then fine-tune on labeled data.

```
Phase 1 (pre-training, no labels):
  Define a pretext task from the data structure itself:
  - Masked token prediction (BERT)
  - Next sentence prediction
  - Image patch prediction (MAE)
  - Contrastive matching (SimCLR, CLIP)

Phase 2 (fine-tuning, small labeled set):
  Remove pretext task head
  Add task-specific head (linear layer)
  Fine-tune on L labeled examples

Why this works:
  Pre-training forces the model to learn rich representations of P(X)
  Fine-tuning adapts these representations to P(Y|X) with few labels
```

---

## SimCLR — Contrastive Self-Supervised Learning

**Core idea:** augmented views of the same image should be close in embedding space; views from different images should be far apart.

```
For each image xᵢ in a batch of N images:
  - Create two augmented views: tᵢ = aug₁(xᵢ), t'ᵢ = aug₂(xᵢ)
  - Total: 2N views per batch

Goal:
  similarity(encode(tᵢ), encode(t'ᵢ)) → HIGH  ← same image, different augment
  similarity(encode(tᵢ), encode(tⱼ))  → LOW   ← different images
```

### NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

```
For pair (tᵢ, t'ᵢ):

  sim(u, v) = u·v / (||u|| · ||v||)   ← cosine similarity

  ℓ(i, j) = -log [ exp(sim(zᵢ, zⱼ)/τ) / Σ_{k≠i} exp(sim(zᵢ, zₖ)/τ) ]

  Total loss = (1/2N) × Σᵢ [ℓ(i, i') + ℓ(i', i)]
  (symmetric: both directions averaged)

τ = temperature (0.07–0.5), N = batch size (larger = more negatives = better)
```

### Dry Run (N=4 images → 8 views)

```
Batch: [cat₁, cat₂, dog₁, dog₂] → augment → 8 views

Embeddings after projection head (2D for illustration):
  cat₁_aug1: [0.9, 0.1]    cat₁_aug2: [0.85, 0.15]   ← should be close
  cat₂_aug1: [0.8, 0.2]    cat₂_aug2: [0.75, 0.25]
  dog₁_aug1: [0.1, 0.9]    dog₁_aug2: [0.15, 0.85]
  dog₂_aug1: [0.2, 0.8]    dog₂_aug2: [0.25, 0.75]

Cosine similarities for cat₁_aug1:
  vs cat₁_aug2: cos ≈ 0.999  ← positive pair
  vs cat₂_aug1: cos ≈ 0.980  ← hard negative (same class)
  vs dog₁_aug1: cos ≈ 0.101  ← easy negative (different class)

Loss for cat₁_aug1 (τ=0.1):
  numerator:   exp(0.999/0.1) = exp(9.99) = 21,794
  denominator: exp(9.99) + exp(9.80) + exp(9.75) + exp(1.01) + ... (7 negatives)
             ≈ 21,794 + 18,034 + 17,154 + 2.7 + ...
  loss = -log(21,794 / ~64,000) ≈ -log(0.34) = 1.08

After training:
  cat₁ and cat₂ embeddings are close (same visual features)
  cats and dogs are far apart (different visual features)
  → Fine-tune on 100 labeled examples: linear probe achieves 90%+ accuracy
```

### SimCLR Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class SimCLR(nn.Module):
    def __init__(self, encoder='resnet50', projection_dim=128):
        super().__init__()
        # Encoder: ResNet without final FC layer
        resnet = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        encoder_dim = 2048  # ResNet-50 output

        # Projection head: 2-layer MLP (discarded after pre-training)
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()  # (B, 2048)
        z = self.projector(h)          # (B, 128) — used for contrastive loss
        return h, z                    # h = representation, z = projection

def nt_xent_loss(z1, z2, temperature=0.1):
    """NT-Xent loss for a batch of N image pairs."""
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)             # (2N, 128)
    z = F.normalize(z, dim=1)

    # Similarity matrix (2N × 2N)
    sim = torch.matmul(z, z.T) / temperature   # (2N, 2N)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2*N, dtype=bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(N)]).to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss

# Pre-training loop
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = SimCLR()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

for epoch in range(100):
    for x, _ in unlabeled_loader:   # labels ignored during pre-training
        x1 = augmentation(x)
        x2 = augmentation(x)
        _, z1 = model(x1)
        _, z2 = model(x2)
        loss = nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# After pre-training: linear evaluation
# Freeze encoder, train only a linear classifier on labeled data
for param in model.encoder.parameters():
    param.requires_grad = False

linear_classifier = nn.Linear(2048, n_classes)
# Fine-tune on 1-10% of labeled data
```

---

## MAE — Masked Autoencoders (for Vision)

**Paper:** "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)

```
Inspired by BERT's masked token prediction, applied to images.

Pre-training:
  1. Divide image into patches (e.g., 196 patches for 224×224)
  2. Randomly mask 75% of patches (leave 25% visible)
  3. Encoder processes only the visible patches (faster — 75% fewer tokens)
  4. Decoder reconstructs the masked patches from:
       - encoded visible patches
       - learnable mask tokens (one per masked position)
  5. Loss: MSE on masked patches only (pixel space)

Fine-tuning:
  Discard decoder
  Fine-tune encoder (ViT) on labeled data with task-specific head
```

```python
# MAE pseudo-code
class MAE(nn.Module):
    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder     = encoder     # ViT-B/16 (processes visible patches only)
        self.decoder     = decoder     # lightweight transformer (reconstruct masked)
        self.mask_ratio  = mask_ratio
        self.mask_token  = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))

    def forward(self, imgs):
        # 1. Patchify
        patches = patchify(imgs)           # (B, 196, 768)

        # 2. Random masking
        n_visible = int(196 * (1 - self.mask_ratio))  # 49 visible patches
        noise     = torch.rand(B, 196)
        ids_shuffle = noise.argsort(dim=1)
        ids_visible = ids_shuffle[:, :n_visible]
        ids_masked  = ids_shuffle[:, n_visible:]

        visible_patches = patches.gather(1, ids_visible.unsqueeze(-1).expand(-1,-1,768))

        # 3. Encode visible patches (fast — only 49 tokens)
        latent = self.encoder(visible_patches)        # (B, 49, 512)

        # 4. Decode: restore full 196-patch sequence
        mask_tokens = self.mask_token.expand(B, 147, -1)  # 147 masked positions
        # Unshuffle: interleave encoded + mask tokens at original positions
        full_tokens = unshuffle(latent, mask_tokens, ids_shuffle)
        pred = self.decoder(full_tokens)                    # (B, 196, 768)

        # 5. Loss on masked patches only
        target = patches.gather(1, ids_masked.unsqueeze(-1).expand(-1,-1,768))
        pred_masked = pred.gather(1, ids_masked.unsqueeze(-1).expand(-1,-1,768))
        loss = F.mse_loss(pred_masked, target)
        return loss
```

**Key result:** MAE with 75% masking + ViT-H achieves 87.8% ImageNet top-1 accuracy with only 1% of labeled data for fine-tuning.

---

## BERT as Semi-Supervised SSL

BERT pre-training IS semi-supervised learning for NLP:

```
Pre-training (unlabeled — all of Wikipedia + BookCorpus):
  MLM: mask 15% of tokens → predict masked tokens
  NSP: predict if sentence B follows sentence A (now known to be less useful)

Fine-tuning (labeled — task-specific small dataset):
  Add linear head, fine-tune all weights or just head

Semi-supervised framing:
  L = {labeled fine-tuning examples}  (hundreds to thousands)
  U = {Wikipedia + BookCorpus}         (billions of tokens)

Why it works: MLM forces BERT to learn:
  - Syntax (predict "dogs [MASK] in the park" → "run/play/bark")
  - Semantics (predict "[MASK] capital of France" → "Paris")
  - Context (bi-directional attention captures full context)
```

---

## Comparison: Classical vs Deep SSL

| | Self-Training | Label Propagation | SimCLR | MAE | BERT |
|--|--------------|-------------------|--------|-----|------|
| Data type | Any | Any (low-dim) | Images | Images | Text |
| Labeled data needed | 100–10K | 10–1K | 100–1K | 100–1K | 100–10K |
| Unlabeled data used | Prediction | Graph propagation | Augmentation pairs | Masked patches | Masked tokens |
| Scales to millions? | ✓ | ✗ | ✓ | ✓ | ✓ |
| State of the art (2024) | ✗ | ✗ | ✓ | ✓ | ✓ |

---

## Practical SSL Workflow (Production)

```
Step 1: Collect labeled data (start small — 500–1000 examples)
Step 2: Collect unlabeled data (production logs, web crawl, etc.)
Step 3: Choose approach based on data type:
  Images  → FixMatch or fine-tune pre-trained ViT (SimCLR/DINO)
  Text    → Fine-tune BERT/RoBERTa on small labeled set
  Tabular → Self-training with GBM

Step 4: Baseline — supervised only on labeled set
Step 5: Apply SSL method
Step 6: Compare on held-out test set
Step 7: If SSL improves: use it. If not: check unlabeled data quality.

Rule of thumb:
  < 1K labeled: SSL helps significantly (+5-15% accuracy)
  1K–10K labeled: SSL helps moderately (+1-5%)
  > 10K labeled: SSL marginal benefit; full fine-tuning usually sufficient
```

---

## Gotchas

**Projection head trick (SimCLR).** The contrastive loss is applied on the projection head output (z), NOT the encoder output (h). But fine-tuning uses h (the encoder output), not z. Discarding the projection head after pre-training consistently gives better results — the projection head compresses information needed for reconstruction but not useful for downstream tasks.

**MAE mask ratio.** 75% masking sounds extreme but works better than 50% or 25%. Higher masking forces the model to learn richer representations (can't just copy neighboring patches). Lower masking → task too easy → weak representations.

**Batch size critical for SimCLR.** NT-Xent needs many negatives. SimCLR uses batch size 4096–8192. Smaller batches (256) → fewer negatives → poor representations. Memory Bank or MoCo solve this by maintaining a queue of negatives.

---

## Interview Q&A

**Q: How is BERT pre-training a form of semi-supervised learning?**
A: BERT pre-training uses billions of unlabeled tokens (Wikipedia, BookCorpus) to learn general language representations via masked token prediction. This is the "unsupervised" phase — no task-specific labels. Fine-tuning then uses small labeled datasets (hundreds to thousands) to adapt these representations. The semi-supervised framing: L = labeled task data, U = pre-training corpus. The model leverages P(X) structure from U to build features, then P(Y|X) from L.

**Q: SimCLR vs FixMatch — when to use each?**
A: SimCLR: fully self-supervised (no labeled data at all during pre-training), needs large batch (4096+), then fine-tune with small labeled set. Best for representation learning when you have millions of unlabeled images. FixMatch: directly semi-supervised — uses both labeled and unlabeled simultaneously, works with small batch sizes, more practical for modest compute. Use SimCLR when building a general-purpose vision encoder; use FixMatch for a specific classification task with limited labels.

---

## Connections

- **Classical SSL:** `1.machine learning/02_algorithms/06_semi_supervised_learning.md`
- **BERT pre-training:** `5.transformers/02_models/01_bert_family.md`
- **CLIP/DINO contrastive:** `7.multimodal/04_clip_finetuning_end_to_end.md`, `7.multimodal/03_vision_transformers.md`
- **FixMatch code:** `1.machine learning/02_algorithms/06_semi_supervised_learning.md`

## Key Takeaway

Deep SSL = pre-train on unlabeled data → fine-tune on small labeled set. Three main paradigms: **contrastive** (SimCLR — augmented views of same image should be close, different images far), **generative** (MAE — reconstruct 75% masked patches), **masked prediction** (BERT — predict masked tokens). All force the model to learn rich representations from P(X) without labels. The key insight: high-quality features from pre-training can be adapted to downstream tasks with as few as 100 labeled examples.

# 08 — Semi-Supervised Learning (Deep Learning Perspective)

> **Prerequisites:** `../../1.machine learning/02_algorithms/06_semi_supervised_learning.md` (self-training, label propagation, co-training)
> **This file:** SSL with deep neural networks — contrastive learning, VAEs, BERT-style pre-training as SSL.

---

## Why Deep SSL Is Different

```
Classical SSL (label propagation, co-training):
  Works on fixed hand-crafted features
  Graph built in original feature space
  Scales to ~1M unlabeled samples

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
```

**Why this works:** Pre-training forces the model to learn rich representations of P(X). Fine-tuning adapts these representations to P(Y|X) with few labels.

---

## SimCLR — Contrastive Self-Supervised Learning

**Core idea:** augmented views of the same image should be close in embedding space; views from different images should be far apart.

For each image x_i in a batch of N images:
- Create two augmented views: t_i = aug_1(x_i), t'_i = aug_2(x_i)
- Total: 2N views per batch

```
Goal:
  similarity(encode(t_i), encode(t'_i)) = HIGH  ← same image, different augment
  similarity(encode(t_i), encode(t_j))  = LOW   ← different images
```

### NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)

For pair (t_i, t'_i):

```
sim(u, v) = u·v / (||u|| · ||v||)   = cosine similarity

ℓ(i, j) = −log [ exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ) ]

Total loss = (1/2N) × Σ_i [ℓ(i, i') + ℓ(i', i)]   (symmetric: both directions averaged)

τ = temperature (0.07-0.5),  N = batch size (larger = more negatives = better)
```

### Dry Run (N=4 images → 8 views)

```
Batch: [cat1, cat2, dog1, dog2] + augment → 8 views

Cosine similarities for cat1_aug1:
  vs cat1_aug2: cos = 0.999  → positive pair
  vs cat2_aug1: cos = 0.980  → hard negative (same class)
  vs dog1_aug2: cos = 0.101  → easy negative (different class)

Loss for cat1_aug1 (τ=0.1):
  numerator:    exp(0.999/0.1) = exp(9.99) ≈ 21,794
  denominator:  exp(9.99) + exp(9.80) + ... (7 negatives) ≈ 60,000
  loss = −log(21,794 / 60,000) ≈ −log(0.36) = 1.08

After training:
  cat and cat+ embeddings are close (same visual features)
  cats and dogs are far apart (different visual features)
  Fine-tune on 100 labeled examples → linear probe achieves 90%+ accuracy
```

### SimCLR Code

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms

class SimCLR(nn.Module):
    def __init__(self, encoder='resnet50', projection_dim=128):
        super().__init__()
        resnet       = models.resnet50(pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove FC
        encoder_dim  = 2048

        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim),
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()   # (B, 2048)
        z = self.projector(h)           # (B, 128) — used for contrastive loss
        return h, z                     # h = representation, z = projection

def nt_xent_loss(z1, z2, temperature=0.1):
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2N, 128)

    sim = torch.matmul(z, z.T) / temperature   # (2N, 2N)
    mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    labels = torch.cat([torch.arange(N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)

# Pre-training loop
augmentation = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model     = SimCLR()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

for epoch in range(100):
    for x, _ in unlabeled_loader:   # labels ignored during pre-training
        x1 = augmentation(x); x2 = augmentation(x)
        _, z1 = model(x1);   _, z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

# After pre-training: freeze encoder, train only linear classifier
for param in model.encoder.parameters():
    param.requires_grad = False
linear_classifier = nn.Linear(2048, n_classes)
# Fine-tune on 1-10% of labeled data
```

---

## MAE — Masked Autoencoders (for Vision)

**Paper:** "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)

### Pre-training

1. Divide image into patches (e.g., 196 patches for 224×224)
2. Randomly mask **75%** of patches (leave 25% visible)
3. Encoder processes only the visible patches (faster — 75% fewer tokens)
4. Decoder reconstructs the masked patches from:
   - encoded visible patches
   - learnable mask tokens (one per masked position)
5. Loss: MSE on masked patches only (pixel space)

### Fine-tuning

- Discard decoder
- Fine-tune encoder (ViT) on labeled data with task-specific head

**Key result:** MAE with 75% masking + ViT-H achieves **87.8% ImageNet top-1** with only 1% of labeled data.

```python
class MAE(nn.Module):
    def __init__(self, mask_ratio=0.75):
        super().__init__()
        self.encoder    = encoder   # ViT-B/16 (processes visible patches only)
        self.decoder    = decoder   # lightweight transformer (reconstruct masked)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))

    def forward(self, imgs):
        patches = patchify(imgs)   # (B, 196, 768)

        # Random masking — keep 25% visible
        n_visible   = int(196 * (1 - self.mask_ratio))   # 49
        noise       = torch.rand(imgs.size(0), 196, device=imgs.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_visible = ids_shuffle[:, :n_visible]
        ids_masked  = ids_shuffle[:, n_visible:]

        visible_patches = patches.gather(1, ids_visible.unsqueeze(-1).expand(-1, -1, 768))

        # Encode only visible patches
        latent = self.encoder(visible_patches)   # (B, 49, 512)

        # Restore full sequence: encoded + mask tokens
        mask_tokens = self.mask_token.expand(imgs.size(0), 147, -1)
        full_tokens = unshuffle(latent, mask_tokens, ids_shuffle)   # (B, 196, 768)
        pred        = self.decoder(full_tokens)

        # Loss only on masked patches
        target      = patches.gather(1, ids_masked.unsqueeze(-1).expand(-1, -1, 768))
        pred_masked = pred.gather(1, ids_masked.unsqueeze(-1).expand(-1, -1, 768))
        return F.mse_loss(pred_masked, target)
```

---

## BERT as Semi-Supervised SSL

BERT pre-training IS semi-supervised learning for NLP:

```
Pre-training (unlabeled — all of Wikipedia + BookCorpus):
  MLM: mask 15% of tokens → predict masked tokens
  NSP: predict if sentence B follows sentence A

Fine-tuning (labeled = task-specific small dataset):
  Add linear head, fine-tune all weights or just head

Semi-supervised framing:
  L = (labeled fine-tuning examples)   (hundreds to thousands)
  U = (Wikipedia + BookCorpus)         (billions of tokens)
```

**Why it works:** MLM forces BERT to learn: syntax (predict "dogs [MASK] in the park" → "run/play/bark"), semantics ("[MASK] capital of France" → "Paris"), context (bidirectional attention captures full context).

---

## 2023-2025 Vision SSL — DINO, BEiT, I-JEPA

SimCLR and MAE are the canonical contrastive and masked variants. The frontier moved on:

| Method | Year | Idea |
|--------|------|------|
| MoCo / BYOL | 2020 | Contrastive without negatives (BYOL) via momentum encoder + stop-gradient — works without SimCLR-style negatives |
| DINO (Meta) | 2021 | Self-distillation: student ViT learns to match a momentum-EMA teacher on different crops; emergent attention maps localize objects without labels |
| DINOv2 (Meta) | 2023 | Scaled DINO to 1B+ params on curated 142M images; features rival supervised ViT-L/H without any labels |
| BEiT / BEiT v2/v3 (Microsoft) | 2022-23 | "BERT for images" — predict discrete visual tokens for masked patches |
| MAE / SimMIM / CAE | 2022-23 | Variants on masked image modeling: different encoder/decoder visibility, different prediction targets |
| I-JEPA (Meta) | 2023 | Predict missing patches in **feature space** (not pixel space) — much better representations than MAE for downstream tasks |
| V-JEPA (Meta) | 2024 | I-JEPA for video; pretraining target for "world models" |

**Senior interview answer:** "MAE is the canonical masked-image SSL — fast pretraining, good reconstruction features. DINOv2 is the current go-to off-the-shelf vision foundation model for any task that doesn't need fine-tuning. I-JEPA is the most interesting 2023 result because predicting in feature space — not pixel space — sidesteps the problem that pixel-level prediction wastes capacity on details that don't matter for downstream tasks."

---

## Comparison: Classical vs Deep SSL

| | Self-Training | Label Prop | SimCLR | MAE | DINO | BERT |
|--|---|---|---|---|---|---|
| Data type | Any | Any (low-dim) | Images | Images | Images | Text |
| Labeled data needed | 100-10K | 10-1K | 100-1K | 100-1K | 100-1K | 100-10K |
| Unlabeled data used | Prediction | Graph propagation | Augmentation pairs | Masked patches | Distillation crops | Masked tokens |
| Scales to millions? | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| State of the art (2025) | ✗ | ✗ | Legacy | Strong | DINOv2 / I-JEPA | RoBERTa/DeBERTa legacy; LLM era dominates |

---

## Practical SSL Workflow (Production)

```
Step 1: Collect labeled data (start small — 500-1000 examples)
Step 2: Collect unlabeled data (production logs, web crawl, etc.)
Step 3: Choose approach based on data type:
  Images  → Fine-tune ViT (SimCLR/DINO)
  Text    → Fine-tune BERT/RoBERTa on small labeled set
  Tabular → Self-training using GBM
Step 4: Baseline — supervised only on labeled set
Step 5: Apply SSL method
Step 6: Compare on held-out test set
Step 7: If SSL improves: use it. If not: check unlabeled data quality.

Rule of thumb:
  < 1K labeled:   SSL helps significantly (+5-15% accuracy)
  1K-10K labeled: SSL helps moderately (+1-5%)
  > 1M labeled:   SSL marginal benefit; full fine-tuning usually sufficient
```

---

## Gotchas

**1. Projection head trick (SimCLR).** The contrastive loss is applied on the projection head output (z), NOT the encoder output (h). But fine-tuning uses h (the encoder output), not z. Discarding the projection head after pre-training consistently gives better results — the projection head compresses information needed for reconstruction but not useful for downstream tasks.

**2. MAE mask ratio.** 75% masking sounds extreme but works — forces the model to learn richer representations (can't just copy neighboring patches). Lower masking → task too easy → weak representations.

**3. Batch size critical for SimCLR.** NT-Xent needs many negatives. Smaller batches (256) → fewer negatives → poor representations. SimCLR uses batch size 4096-8192. Solutions: Memory Bank or MoCo maintain a queue of negatives to solve this by.

---

## Interview Q&A

**Q: How is BERT pre-training a form of semi-supervised learning?**

BERT pre-training uses billions of unlabeled tokens (Wikipedia, BookCorpus) to learn general language representations via masked token prediction. This is the "unsupervised" phase — no task-specific labels. Fine-tuning then uses small labeled datasets (hundreds to thousands) to adapt these representations to task-specific tasks. The semi-supervised framing: L = labeled fine-tuning examples, U = Wikipedia + BookCorpus. The model learns P(X) structure from U to build features, then P(Y|X) from L.

**Q: SimCLR vs FixMatch — when to use each?**

SimCLR is fully self-supervised (no labeled data at all during pre-training), needs large batch (4096+), then fine-tunes — best for representation learning when you have millions of unlabeled images. FixMatch is directly semi-supervised — uses both labeled and unlabeled simultaneously, works with small batch sizes, more practical for a specific classification task with limited labels.

---

## Connections

- Classical SSL: `../../1.machine learning/02_algorithms/06_semi_supervised_learning.md`
- BERT pre-training: `../../5.transformers/02_models/01_bert_family.md`
- CLIP/DINO contrastive: `../../9.multimodal/04_clip_fine_tuning.md`
- Vision Transformers: `../../9.multimodal/03_vision_transformers.md`

---

## Key Takeaway

```
Deep SSL = pre-train on unlabeled data → fine-tune on small labeled set

Three main paradigms:
  Contrastive (SimCLR) — augmented views of same image should be close;
                          different images far
  Generative  (MAE)    — reconstruct 75% masked patches
  Masked pred (BERT)   — predict masked tokens

All force the model to learn rich representations from P(X) so that high-quality
features from pre-training can be adapted to downstream tasks with as few as
100 labeled examples.
```

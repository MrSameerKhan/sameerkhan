# 06 — Specialized Loss Functions

## Quick Reference

| Loss | Task | When Standard CE Fails |
|------|------|----------------------|
| Focal Loss | Object detection, class imbalance | 99% easy negatives drown out hard positives |
| CTC Loss | OCR, ASR — seq-to-seq without alignment | Output shorter than input, no frame-level labels |
| Contrastive Loss | Metric learning, embeddings | Classification head not useful; need distance-based retrieval |
| Triplet Loss | Face verification, image retrieval | Pair sampling insufficient; need anchor-positive-negative structure |
| NT-Xent (SimCLR) | Self-supervised contrastive | No labels; augmented views of same image should be similar |
| Label Smoothing CE | Classification with noisy labels | Model overconfident; poor calibration |
| Dice Loss | Segmentation with class imbalance | Pixel-wise CE ignores foreground/background ratio |

---

## 1. Focal Loss

**Problem it solves:** Object detection datasets have extreme class imbalance — for every foreground object, there are thousands of background patches. Standard cross-entropy: easy negatives (background, predicted with 0.99 confidence) still contribute large cumulative loss → model optimizes these rather than learning hard cases.

### Formula
```
FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

Where:
  p_t = model probability for the true class
  (1 − p_t)^γ = modulating factor (down-weights easy examples)
  α_t = class balancing weight (optional, handles class frequency)
  γ (gamma) = focusing parameter, typically 2.0
```

**Behavior:**
- Well-classified example: p_t = 0.95 → (1-0.95)² = 0.0025 → contribution ~0
- Hard example: p_t = 0.05 → (1-0.05)² = 0.90 → contribution large

```
γ=0: Focal Loss = standard cross-entropy
γ=2: Hard examples get 100× more weight than easy examples (p_t=0.1 vs 0.9)
```

**Numeric example:**
```
Easy negative: p_t = 0.98
  CE:    −log(0.98) = 0.020
  FL(γ=2): −(1−0.98)² · log(0.98) = 0.0004 · 0.020 = 0.000008  ← near-zero

Hard positive: p_t = 0.05
  CE:    −log(0.05) = 3.0
  FL(γ=2): −(1−0.05)² · log(0.05) = 0.90 · 3.0 = 2.7  ← stays large
```

### When to Use
- Object detection (RetinaNet, YOLO)
- Any task with severe class imbalance (rare disease classification, anomaly detection)
- When model ignores rare classes despite oversampling

### Code
```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)                    # pt = probability of correct class
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()

# Or using torchvision (for detection)
from torchvision.ops import sigmoid_focal_loss
loss = sigmoid_focal_loss(inputs=logits, targets=targets.float(),
                           alpha=0.25, gamma=2.0, reduction='mean')
```

---

## 2. CTC Loss (Connectionist Temporal Classification)

**Problem it solves:** Sequence-to-sequence where:
- Input and output have different lengths (input: image columns or audio frames, output: characters)
- No frame-level alignment labels (you don't know which frame maps to which character)
- Classic in OCR, ASR, handwriting recognition

### The Alignment Problem
```
Input sequence:  [img_1, img_2, img_3, img_4, img_5, img_6, img_7]  (7 frames)
Target text:     "CAT"  (3 characters)

Which frames correspond to which characters? Unknown.
Possible alignments:
  C C A T T T T
  C C C A T T T
  C C A A A T T
  ... (many valid alignments)
```

### CTC Solution
Add a **blank token** (−) as a special output class. Sum over all valid alignments that collapse to the target:

```
"CAT" can come from: "---CCA-TT-", "C-C-A--T--", etc.

Collapse rule:
  1. Remove repeated characters
  2. Remove blank tokens
  → All produce "CAT"
```

CTC loss = −log P(target | input) = −log Σ P(alignment) over all valid alignments

Uses dynamic programming (forward-backward algorithm) to compute this sum efficiently.

### Requirements
- Input length ≥ Output length (model must output at least as many frames as target characters)
- Outputs are **conditionally independent** per timestep (limitation: no language model prior)

### When to Use
- OCR / Handwriting recognition (text line recognition without character segmentation)
- Speech-to-text (ASR) without forced alignment
- Any sequence labeling where alignment is unknown

### Code
```python
import torch
import torch.nn as nn

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

# logits: [T, N, C] — sequence length × batch × num_classes (including blank)
# targets: [N, S] or 1D concatenated targets
# input_lengths: [N] — actual sequence lengths (before padding)
# target_lengths: [N] — actual target lengths

log_probs = torch.log_softmax(logits, dim=2)  # CTC expects log probabilities
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

# Typical setup: BiLSTM or CNN → linear head → CTCLoss
# input_lengths: usually full sequence length (T) for each sample
# zero_infinity=True: ignore inf losses from impossible alignments (input < target len)
```

### CTC Decoding at Inference
```python
# Greedy: argmax per timestep, then collapse
pred = log_probs.argmax(dim=2)  # [T, N]
# Collapse: remove blanks and repeated chars
# "C-CCAATT--" → "CAT"

# Beam search: better but slower
# Libraries: torch-ctcdecode, torchaudio, or ctcdecode
```

---

## 3. Contrastive Loss (Siamese Networks)

**Problem it solves:** Learn an embedding space where similar pairs are close and dissimilar pairs are far apart. No class labels needed — just pair labels (same/different).

### Formula
```
L = y · d²  +  (1−y) · max(0, margin − d)²

Where:
  d = Euclidean distance between two embeddings
  y = 1 if same class (positive pair), 0 if different class (negative pair)
  margin = minimum distance enforced for negative pairs (e.g., 1.0)
```

**Behavior:**
- Positive pair (y=1): minimize d → pull same-class embeddings together
- Negative pair (y=0): maximize d, at least to margin → push different-class embeddings apart (but only up to margin, not infinitely)

### Limitation
Dead negatives: once d > margin, negative pairs contribute zero gradient. Needs hard negative mining.

---

## 4. Triplet Loss

**Problem it solves:** More stable than contrastive loss — instead of pairs, uses (anchor, positive, negative) triplets. Enforces relative distance: anchor-positive < anchor-negative.

### Formula
```
L = max(0,  d(anchor, positive) − d(anchor, negative) + margin)

Where:
  d = distance (Euclidean or cosine)
  margin = minimum gap between positive and negative distances (e.g., 0.2)
```

**Behavior:**
- If d(a,p) < d(a,n) − margin → loss = 0 (already correct)
- If d(a,p) > d(a,n) − margin → loss > 0 → gradients push a-p closer and a-n farther

### Hard Negative Mining (Critical)
Random triplets → most are easy (already satisfy margin) → zero gradient → model doesn't learn.

**Hard negatives**: negatives that are close to anchor in embedding space (hardest to separate).
**Semi-hard**: negatives farther than positive but within margin.

```python
# Online hard negative mining: compute all pairwise distances in batch,
# find hardest valid triplets per anchor
# Libraries: pytorch-metric-learning handles this
```

### When to Use (Contrastive vs Triplet)
| | Contrastive | Triplet |
|--|-------------|---------|
| Pairs | (anchor, positive/negative) | (anchor, positive, negative) |
| Stability | Lower (can oscillate) | Higher (relative comparison) |
| Mining | Positive/negative mining | Triplet mining (harder) |
| Use case | Signature verification, face verification | Face recognition, image retrieval |

---

## 5. NT-Xent Loss (SimCLR — Self-Supervised Contrastive)

**Problem it solves:** Learn representations without labels. Two augmented views of the same image should be similar; views from different images should be dissimilar.

### Formula
```
L = −log( exp(sim(z_i, z_j)/τ)  /  Σ_{k≠i} exp(sim(z_i, z_k)/τ) )

Where:
  z_i, z_j = embeddings of two views of same image (positive pair)
  sim = cosine similarity
  τ = temperature (controls concentration of distribution, typically 0.07–0.5)
  Denominator = all 2N−2 other samples in batch (negatives)
```

**Key insight:** Larger batch size = more negatives = better representations. SimCLR typically uses batch size 4096-8192.

### Code
```python
def nt_xent_loss(z1, z2, temperature=0.5):
    """z1, z2: [batch_size, embedding_dim], L2 normalized"""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = torch.mm(z, z.T) / temperature  # [2B, 2B]
    # Remove self-similarity
    sim.fill_diagonal_(float('-inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(batch_size) + batch_size,
                        torch.arange(batch_size)]).to(z.device)
    return F.cross_entropy(sim, labels)
```

---

## 6. Label Smoothing

**Problem it solves:** Standard CE pushes model to output probability 1.0 for the true class → overconfident, poorly calibrated. Bad for transfer learning and knowledge distillation.

### Formula
```
Smoothed target: y_smooth = (1 − ε) · y_hard + ε / K

Where:
  y_hard = one-hot label
  ε = smoothing factor (typically 0.1)
  K = number of classes

Effect:
  True class:  target = 1.0 − ε + ε/K = 0.9 + 0.1/K  (not 1.0)
  Other class: target = ε/K  (not 0.0)
```

### When to Use
- When model is overconfident (calibration ECE is high)
- Image classification training (ImageNet — transformers benefit a lot)
- Knowledge distillation (teacher soft labels already smoothed)
- Not useful if you need exact probability outputs (medical diagnosis)

```python
loss = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 7. Dice Loss (Segmentation)

**Problem it solves:** Pixel-wise CE for segmentation ignores class imbalance — in medical imaging, foreground (tumor) may be 1% of pixels. CE optimizes average accuracy → predicts all background.

### Formula
```
Dice = 2 · |A ∩ B| / (|A| + |B|)
     = 2 · Σ(p_i · g_i) / (Σp_i + Σg_i)

Dice Loss = 1 − Dice

Where:
  p_i = predicted probability for pixel i
  g_i = ground truth (0 or 1) for pixel i
```

**Why Dice?** Dice coefficient (F1 for segmentation) is recall+precision balanced — foreground accuracy matters regardless of overall pixel count.

```python
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

# Common: combine with BCE
total_loss = bce_loss + dice_loss
```

---

## 8. When to Use What — Master Table

| Scenario | Loss | Notes |
|----------|------|-------|
| Standard classification | Cross Entropy | Always start here |
| Severe class imbalance | Focal Loss (γ=2, α=0.25) | Detection; rare class problems |
| OCR / ASR (no alignment labels) | CTC Loss | Sequence-to-sequence, variable length |
| Face verification (same/different) | Contrastive Loss | Binary pair judgement |
| Face recognition (retrieval) | Triplet Loss + hard mining | 1:N search in embedding space |
| Self-supervised image representation | NT-Xent (SimCLR) | No labels, large batch needed |
| Document retrieval / embedding | Triplet or NT-Xent | Similar docs close in embedding space |
| Medical image segmentation | Dice + BCE | Small foreground regions |
| Classification with noisy labels | Label Smoothing (ε=0.1) | Better calibration, regularization |
| Knowledge distillation | KL Divergence | Soft teacher targets vs hard student outputs |

---

## 9. Gotchas

**CTC: input_lengths must be exact.**
If you pad sequences and pass max_length as input_lengths instead of actual lengths → CTC treats padding frames as real → alignment broken → loss explodes or NaN. Always pass true lengths.

**CTC: blank class must be included in num_classes.**
vocab_size = num_chars + 1 (for blank). Forgetting the blank token → wrong output dimension → CUDA error or shape mismatch.

**Triplet: random sampling = dead loss.**
~99% of random triplets are easy (already satisfy margin) → zero gradient → model doesn't train. Always use online hard or semi-hard mining.

**Focal loss: γ tuning matters.**
γ=0 is standard CE. γ=2 is RetinaNet default. Higher γ → more focus on hardest examples → can hurt if your hard examples are actually mislabeled. Tune γ on validation set.

**NT-Xent: small batch → bad negatives.**
SimCLR needs batch 4096+ for meaningful contrastive signal. On small batch (256), most negatives are too easy → representations don't learn well. Use MoCo (momentum encoder + queue of negatives) for memory-efficient contrastive learning.

**Label smoothing + distillation interaction.**
If teacher was trained with label smoothing and student also uses it → double-smoothing → too much uncertainty. Use one or the other.

---

## 10. Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| CTC loss = inf | input_length < target_length | Ensure sequence encoder output ≥ target length |
| CTC loss = NaN | Padding in input_lengths | Pass true sequence lengths, not padded lengths |
| Focal loss same as CE | γ=0 or all examples already easy | Set γ=2; check class balance in data |
| Triplet loss = 0 from epoch 1 | All random triplets are easy | Enable online hard negative mining |
| NT-Xent poor representations | Batch too small | Use MoCo queue; increase batch size |
| Dice loss not improving | Threshold for binarization wrong | Check sigmoid vs softmax output; adjust smooth |
| Model confident wrong class | No label smoothing, overfitting | Add label_smoothing=0.1 to CE |

---

## 11. Code Reference — Full CTC Pipeline (OCR)

```python
import torch
import torch.nn as nn

class CTCOCRModel(nn.Module):
    """CNN backbone → BiLSTM sequence model → CTC head"""
    def __init__(self, num_classes, hidden_size=256):
        super().__init__()
        # CNN: extract features from image columns
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        # BiLSTM: sequence modeling over CNN feature columns
        self.rnn = nn.LSTM(128 * 8, hidden_size, num_layers=2,
                           bidirectional=True, batch_first=False)
        # Projection: hidden → vocabulary (including blank=0)
        self.head = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 for blank

    def forward(self, x):
        # x: [B, 1, H, W] (grayscale text line image)
        features = self.cnn(x)                          # [B, 128, H/4, W/4]
        B, C, H, W = features.shape
        features = features.permute(3, 0, 1, 2)        # [W/4, B, C, H/4]
        features = features.reshape(W, B, C * H)       # [T, B, features]
        output, _ = self.rnn(features)                  # [T, B, hidden*2]
        logits = self.head(output)                      # [T, B, num_classes+1]
        return torch.log_softmax(logits, dim=2)

ctc = nn.CTCLoss(blank=0, zero_infinity=True)
log_probs = model(images)                               # [T, B, C]
T, B, _ = log_probs.shape
input_lengths = torch.full((B,), T, dtype=torch.long)  # all sequences full length
loss = ctc(log_probs, targets, input_lengths, target_lengths)
```

---

## 12. Interview Q&A (Senior Level)

**Q: Why does standard cross-entropy fail for object detection, and how does Focal Loss fix it?**
A: In detection, the foreground/background ratio is typically 1:1000. With CE, easy background samples (predicted with 0.99 confidence) have small individual loss but contribute massively in aggregate → the model is "rewarded" for correctly classifying backgrounds, not for detecting objects. Focal Loss adds a modulating factor (1−p_t)^γ that down-weights easy examples exponentially. A well-classified sample with p_t=0.97 gets multiplied by (0.03)²=0.0009 — near-zero contribution. Hard examples (p_t=0.1) get multiplied by (0.9)²=0.81 — full contribution. The result: training focuses on difficult, informative examples.

**Q: How does CTC handle the alignment problem? What are its limitations?**
A: CTC introduces a blank token and defines a many-to-one mapping: any sequence of frame-level predictions that collapses (by removing blanks and consecutive duplicates) to the target text is a valid alignment. CTC maximizes the sum of probabilities of all valid alignments — computed efficiently via dynamic programming (forward-backward algorithm). Limitations: (1) Assumes frame-level outputs are conditionally independent — no learned language model prior at the output. (2) Can't model token interdependencies (unlike attention-based seq2seq). Fix: combine CTC with attention decoder (best of both — faster convergence with CTC, better accuracy with attention). Used in modern ASR (Whisper-style hybrid training).

**Q: Triplet loss works poorly with random sampling. What's the right approach?**
A: With random triplets, most satisfy the margin already → zero gradient → no learning. Hard negative mining is essential. Online hard mining (OHM): within each batch, compute all pairwise distances and select the hardest valid triplets — hardest positive (farthest from anchor with same label) and hardest negative (closest to anchor with different label). Avoid fully hard negatives (can destabilize training → they may be mislabeled or genuinely ambiguous). Semi-hard negatives (farther than positive but within margin) are often better. Libraries like `pytorch-metric-learning` handle this automatically.

**Q: When would you combine multiple loss functions and how?**
A: Common combinations: (1) **CTC + Attention decoder**: λ·CTC + (1−λ)·CE — CTC provides stable gradient early, attention provides better sequence modeling. (2) **BCE + Dice** for segmentation: BCE optimizes pixel accuracy, Dice optimizes F1 for imbalanced foreground. (3) **Reconstruction + Perceptual** in VAE: MSE gives blurry outputs; add VGG feature-level MSE to sharpen. (4) **Task loss + Contrastive** in multi-task: classification head + contrastive head on same encoder. Tuning: start with equal weights, then sweep. Common mistake: terms at very different scales → one dominates. Normalize losses to same order of magnitude first.

**Q: How would you handle a document OCR pipeline with no character-level annotations?**
A: CTC is the standard answer. You have text line images and transcriptions — no character bounding boxes needed. Pipeline: (1) CNN backbone extracts visual features from image columns (sliding window over width). (2) BiLSTM or Transformer encoder models sequence over columns. (3) CTC head with vocabulary + blank token. (4) CTC loss → backprop without needing to know which column corresponds to which character. At inference: greedy CTC decode (argmax per timestep → collapse). For better accuracy: beam search with language model rescoring (character-level LM improves CTC outputs significantly).

---

## 13. Connections

| This file | Links to | Why |
|-----------|----------|-----|
| CTC Loss | `../architectures/03_rnn_lstm_gru.md` | BiLSTM is the typical CTC encoder |
| Contrastive / NT-Xent | `../architectures/04_transformer.md` | BERT pre-training uses MLM, not contrastive — contrast with SimCLR |
| Focal Loss | `../../2.computerVision/` | RetinaNet, YOLO — detection loss baseline |
| Dice Loss | Future: segmentation | UNet / DeepLab training for document layout |
| Label Smoothing | `../fundamentals/01_foundations.md` | Cross-entropy base, calibration concept |
| CTC in document pipeline | Your domain | Text line recognition, form field extraction |

---

## Key Takeaway

**Start with CE. Switch when you have a structural problem:**
- Class imbalance → Focal Loss
- Unknown alignment → CTC
- Need metric/embedding space → Triplet or NT-Xent
- Overconfidence / noisy labels → Label Smoothing
- Segmentation foreground imbalance → Dice + BCE

For your domain (document automation): **CTC** is the most critical — it's the backbone of every OCR engine. Know it cold: blank token, collapse rule, why input_length ≥ target_length, how to combine with attention decoder.

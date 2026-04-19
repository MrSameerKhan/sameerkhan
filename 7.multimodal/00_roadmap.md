# Multimodal AI — Roadmap & Navigation Guide

> **Goal:** Build fluency in models that fuse vision + language. This folder covers the core architectures, end-to-end traces, and interview-ready mental models.

---

## Folder Map

```
7.multimodal/
├── 00_roadmap.md                     ← you are here
├── 01_vision_language_models.md      ← CLIP, BLIP, LLaVA architecture overview
├── 02_document_ai.md                 ← Document understanding landscape
├── 03_vision_transformers.md         ← ViT, Swin, DeiT, DINO (with dry-run)
├── 04_clip_finetuning_end_to_end.md  ← Contrastive learning full trace
├── 05_donut_end_to_end.md            ← OCR-free doc parsing full trace
└── 06_layoutlm_end_to_end.md         ← LayoutLM v3 full trace
```

---

## Reading Order

### If you have 2 hours (interview tomorrow)

| Order | File | What you get |
|-------|------|--------------|
| 1 | `03_vision_transformers.md` §1–3 | ViT patch embeddings, 197 tokens, position embeddings |
| 2 | `04_clip_finetuning_end_to_end.md` §1–4 | Dual encoder, InfoNCE loss, N×N similarity matrix |
| 3 | `06_layoutlm_end_to_end.md` §1–3 | OCR → bbox → embeddings → predictions |
| 4 | `05_donut_end_to_end.md` §1–3 | Swin encoder, autoregressive decoder, teacher forcing |

### If you have a full day (deep prep)

Read every file top to bottom in folder order (01 → 06). Each file cross-references the others.

---

## Architecture Decision Guide

```
What type of task?
│
├── Image classification only
│   └── Use ViT (or Swin for high-res) → see 03_vision_transformers.md
│
├── Image + text alignment (zero-shot classification, retrieval)
│   └── Use CLIP → see 04_clip_finetuning_end_to_end.md
│
├── Document with structured fields (invoice, form, receipt)
│   │
│   ├── Have OCR pipeline already?
│   │   └── Use LayoutLM v3 → see 06_layoutlm_end_to_end.md
│   │
│   └── No OCR, want end-to-end?
│       └── Use Donut → see 05_donut_end_to_end.md
│
├── General visual Q&A / captioning
│   └── Use BLIP-2 or LLaVA → see 01_vision_language_models.md
│
└── Document with dense text (scientific papers, PDFs)
    └── Use Nougat → see 02_document_ai.md
```

---

## Model Cheat Sheet

| Model | Encoder | Decoder | Input | Pre-training | Params |
|-------|---------|---------|-------|--------------|--------|
| ViT-B/16 | 12-layer transformer | — | 196+1 patches | Supervised (ImageNet-21K) | 86M |
| Swin-B | 4-stage window attn | — | 307K→4.8K tokens | Supervised | 88M |
| DeiT-B | ViT + distil token | — | 197+1 tokens | KD from CNN teacher | 86M |
| DINO | ViT (self-supervised) | — | 197 tokens | Self-supervised (ImageNet) | 86M |
| CLIP (ViT-B/32) | ViT image + text transformer | — | 49+1 / 77 tokens | Contrastive (400M pairs) | 151M |
| LayoutLM v3 | Unified text+image transformer | — | Text+bbox+patches | MLM + MIM + WPA | 133M |
| Donut | Swin encoder | BART decoder | 2560×1920 raw image | SynthDoG (500K docs) | ~200M |
| BLIP-2 | Frozen ViT + Q-Former | Frozen LLM | patches + text | Stage-wise | 12B+ |
| LLaVA | CLIP ViT | LLaMA/Vicuna | 256 patches + text | Instruct tuning | 7-13B |

---

## Key Numbers to Know

### ViT-B/16
- Image: 224×224 → 14×14 = **196 patches** (16×16 each)
- Sequence: 196 + [CLS] = **197 tokens**
- Hidden dim: **d = 768**, heads = 12, d_head = 64
- Parameters: **86M**

### CLIP (ViT-B/32)
- Patches: 224/32 = 7×7 = **49 patches** + [CLS] = 50 tokens
- Embedding dim: **512** (after linear projection + L2 norm)
- Temperature: logit_scale = exp(2.66) ≈ **14.3**
- Pre-training batch: **32,768 pairs**

### LayoutLM v3
- Bbox normalized to **[0, 1000]**: x_norm = int(x / width × 1000)
- Image: 224×224 → 14×14 = **196 patches** (16×16 each)
- Parameters: **133M** (v3), 113M (v1), 200M (v2)

### Donut
- Input image: **2560×1920** (before resize), 4×4 patch extraction
- Swin stages: 307K → 77K → 19K → **4,800 vectors ∈ ℝ^768**
- Max output tokens: **512** (configurable)

### Swin-B
- Input: 224×224, **4×4 patches** → 3136 tokens
- Window size: **7×7** = 49 tokens per window
- Stages: 4 → token counts: 3136 → 784 → 196 → 49

---

## Pre-training Objectives Summary

| Model | Objective | What it learns |
|-------|-----------|----------------|
| ViT | Cross-entropy (supervised) | Visual features from labels |
| DINO | Cross-entropy (student vs teacher) | Semantic structure without labels |
| CLIP | InfoNCE contrastive | Align image ↔ text embeddings |
| LayoutLM v3 | MLM + MIM + WPA | Text, layout, image alignment |
| Donut | Token-level CE (seq2seq) | Read structured text from images |

**MLM** = Masked Language Modeling (mask 15% text tokens)  
**MIM** = Masked Image Modeling (mask 40% patches, predict dVAE tokens)  
**WPA** = Word-Patch Alignment (binary: does text token ↔ image patch?)  
**InfoNCE** = Maximize similarity of matched pairs, minimize unmatched

---

## Loss Functions

### InfoNCE (CLIP)
```
L_I = -(1/N) Σ log[ exp(s_ii/τ) / Σ_j exp(s_ij/τ) ]
L = (L_I + L_T) / 2
```
Where s_ii = diagonal (matching pairs), τ = temperature.

### Cross-Entropy (Donut, LayoutLM)
```
L = -(1/T) Σ_t log P(y_t | y_<t, x)
```
All padding positions masked with -100 before loss computation.

### DeiT Distillation
```
L_total = 0.5 × L_CE(student, labels) + 0.5 × L_KD(student, teacher_softmax)
```

---

## Common Interview Questions (Cross-File)

**Q: Why does CLIP use temperature scaling?**  
A: Controls sharpness of the softmax. Low τ → hard distribution (confident), high τ → uniform. Learnable parameter initialized to 0.07 (τ), optimized during training.

**Q: ViT vs Swin — when to use which?**  
A: ViT: global attention from layer 1, better with large pre-training data (JFT-300M). Swin: local window attention O(n) vs O(n²), hierarchical features, better for high-res images and dense prediction (detection, segmentation).

**Q: Why does Donut not need OCR?**  
A: Swin encoder reads raw pixels at 2560×1920 resolution. Decoder generates structured text autoregressively via cross-attention to visual features. No bounding boxes, no text detection step.

**Q: LayoutLM v3 vs Donut — key difference?**  
A: LayoutLM v3 requires OCR first (text + bbox input). Donut is OCR-free (raw image only). LayoutLM v3 is better when you have reliable OCR; Donut is better for noisy scans or when you want end-to-end differentiability.

**Q: What is the [CLS] token and why prepend it?**  
A: Learnable token prepended to the sequence. Has no spatial meaning — forced to attend to all other tokens across all 12 layers. Final [CLS] representation is used as the global image summary for classification. In DINO, removing [CLS] supervision causes training collapse.

**Q: What is WPA in LayoutLM v3?**  
A: Word-Patch Alignment. Binary classification: does this text token spatially align with this image patch? Forces the model to learn cross-modal spatial correspondence during pre-training.

---

## Connections to Other Folders

| Topic | This folder | Cross-reference |
|-------|-------------|-----------------|
| Transformer basics | All files use attention | `5.transformers/01_fundamentals/` |
| LoRA fine-tuning | CLIP fine-tuning strategies | `5.transformers/02_models/09_parameter_efficient_tuning.md` |
| BERT pre-training | LayoutLM v3 inherits MLM | `5.transformers/02_models/02_bert_and_variants.md` |
| Contrastive learning | CLIP InfoNCE | `5.transformers/02_models/` |
| OCR systems | LayoutLM v3 input | `7.multimodal/02_document_ai.md` |
| Seq2seq | Donut decoder | `4.nlp/03_sequence_models/` |

---

## What to Build (Code Phase)

| Project | Model | Task | File |
|---------|-------|------|------|
| Document classifier | CLIP | Zero-shot invoice vs contract | `04_clip_finetuning_end_to_end.md` |
| Invoice field extractor | LayoutLM v3 | Token classification | `06_layoutlm_end_to_end.md` |
| OCR-free form parser | Donut | Seq2seq generation | `05_donut_end_to_end.md` |
| Image feature extractor | ViT/DINO | Embedding extraction | `03_vision_transformers.md` |

---

## Key Takeaway

Multimodal AI splits into two families:

1. **Contrastive models (CLIP, DINO)** — learn by comparing pairs. No labels needed at scale. Zero-shot capable. Weakness: no generation.

2. **Generative models (Donut, LLaVA, BLIP-2)** — encoder reads image, decoder generates text. Can answer questions, fill forms, caption images. Needs task-specific fine-tuning.

**LayoutLM v3** sits in between — it classifies tokens (discriminative) but uses a unified transformer over text + layout + image patches, making it the best choice for structured document understanding when OCR is available.

The trend: **frozen large encoders + small trainable adapters**. You rarely train ViT or CLIP from scratch — you fine-tune the last layers or use LoRA (1–5% of parameters) to adapt to your domain.

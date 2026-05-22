# 3. Computer Vision

> Scope: vision-specific applications + ViT depth. Vision-language models live in `../9.multimodal/`.
> Tier 2 (Theory).

---

## Reading Order

| If you're learning... | Read in order |
|---|---|
| CNN from scratch | `01_fundamentals/01_cnn_mechanics` → `02_cnn_architectures` → `03_cnn_end_to_end` |
| ViT family | `01_fundamentals/04_vision_transformer_deep` (after the transformer file in `../5.transformers/`) |
| Production CV | `02_applications/01_transfer_learning` → `02_object_detection` → `03_segmentation` → `04_explainability` |
| Modern self-supervised | `02_applications/05_self_supervised_vision` (DINOv2, MAE, I-JEPA, CLIP, SigLIP) |
| DETR family | `02_applications/06_detr_deep` |

---

## Folder TOC

### `01_fundamentals/`

| File | Owns |
|------|------|
| `01_cnn_mechanics.md` | Conv, pooling, receptive field, dilated/depthwise/grouped convs |
| `02_cnn_architectures.md` | Evolution: LeNet → ConvNeXt → Vision Mamba / MambaVision |
| `03_cnn_end_to_end.md` | Worked example — forward + backward pass with numbers |
| `04_vision_transformer_deep.md` | SSOT: ViT / Swin / DeiT / hierarchical windowed attention |

### `02_applications/`

| File | Owns |
|------|------|
| `01_transfer_learning.md` | Feature extraction → fine-tuning → DINOv2 as backbone + LoRA for vision |
| `02_object_detection.md` | Anchors, NMS, Faster R-CNN, YOLO v8-v12, RT-DETR, D-FINE = open-vocab (Grounding DINO, YOLO-World, OWLv2) |
| `03_segmentation.md` | U-Net, DeepLab, Mask R-CNN, Mask2Former, OneFormer, SAM / SAM 2, Grounded-SAM |
| `04_explainability.md` | Grad-CAM, SHAP, Attention Rollout for ViT |
| `05_self_supervised_vision.md` | SSOT: DINOv2, MAE, I-JEPA, BEIT, CLIP, SigLIP, V-JEPA |
| `06_detr_deep.md` | SSOT: DETR family — Deformable DETR, DINO, RT-DETR |

---

## SSOT Topics Owned Here

- ViT / Swin / DeiT depth → `01_fundamentals/04_vision_transformer_deep.md`
- Self-supervised vision (DINOv2, MAE, I-JEPA, CLIP, SigLIP) → `02_applications/05_self_supervised_vision.md`
- Modern detectors (YOLOv9/v10/v11/v12, RT-DETR, D-FINE) → `02_applications/02_object_detection.md`
- Open-vocabulary detection (Grounding DINO, YOLO-World, OWLv2) → `02_applications/02_object_detection.md`
- Mask2Former / OneFormer / SAM / Grounded-SAM → `02_applications/03_segmentation.md`
- DETR family deep dive → `02_applications/06_detr_deep.md`

---

## Connections

| This folder | Links to | Why |
|---|---|---|
| CNN as universal building block | `../2.deep learning/02_architectures/02_cnn.md` | Core CNN reference lives in DL |
| Transformer architecture | `../2.deep learning/02_architectures/04_transformer.md` | Attention math, ViT backbone |
| Vision-language models (CLIP, LLaVA, Qwen-VL) | `../9.multimodal/` | Cross-modal models extend CV |
| Document AI (LayoutLM, Donut, CoPali) | `../9.multimodal/02_document_ai.md` | Document-specific vision models |
| Modern training recipe (MixUp/CutMix/RandAugment) | `../2.deep learning/01_fundamentals/04_generalization.md` | Augmentation for ViT/ConvNeXt |

---

## Practice

No dedicated CV phase in `code_practice/` (legacy CV notebooks in `../archive/3.computerVision/01_code/`). For Document AI hands-on, see `../code_practice/05_rag/` and the active resume RAG project at `../archive/projects/rag_system/`.

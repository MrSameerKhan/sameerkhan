# 11. Multimodal

Scope: vision+language fusion — VLMs, document AI, multimodal RAG.

```mermaid
mindmap
  root((9. Multimodal))
    Vision-Language Models
      CLIP · dual encoder · InfoNCE
      BLIP-2 · Q-Former · LLaVA
      Modern VLMs 2024-25
        PaliGemma · Qwen-VL · Florence-2 · Phi-3
    Document AI
      OCR → layout → LayoutLM pipeline
      Donut · end-to-end no OCR
      LayoutLM v3 · bbox normalization
      ColPali · multimodal RAG
    Vision Transformers
      ViT · patch embedding · CLS token
      Swin · window attention · dense prediction
      DINO · DINOv2 · self-supervised
    CLIP fine-tuning
      Contrastive loss · LoRA on CLIP
    Audio
      Whisper · mel spectrogram · encoder-decoder
      Audio LLMs · Gemini native
    Hallucination mitigation
      Object · attribute · relation types
      Verifier · constrained · ensemble
``` Pure-vision applications in `../3.computerVision/`; pure-NLP in `../4.nlp/`. **Tier: 2 (Theory).** Note: `00_roadmap.md` is legacy navigation. This README supersedes it. `00_roadmap.md` will be archived in Phase 8.

---

## Reading Order

| If you're learning... | Read in order |
|---|---|
| VLM fundamentals | `01_vision_language_models` → `03_vision_transformers` |
| Document AI (Sameer's domain) | `02_document_ai` (incl. ColPali / native-VLM parsing added in Phase 2.5) → `06_layoutlm_end_to_end` → `05_donut_end_to_end` |
| CLIP / contrastive VLM | `01_vision_language_models` → `04_clip_finetuning_end_to_end` |
| Modern VLMs (2024-25) | `07_modern_vlms_2024_2025` (LLaVA, Qwen-VL, Pixtral, Llama-3.2-Vision) |
| Audio + multimodal | `08_audio_multimodal` |
| VLM failure modes | `09_vlm_hallucination_mitigation` |

---

## Folder TOC

| File | Owns |
|---|---|
| `01_vision_language_models.md` | CLIP / BLIP / LLaVA architectures |
| `02_document_ai.md` | OCR, LayoutLM, Donut, ColPali / ColQwen, native-VLM parsing (Claude / GPT-4o / Qwen2.5-VL) |
| `03_vision_transformers.md` | ViT for VLM context (depth — v3 cv) |
| `04_clip_finetuning_end_to_end.md` | Worked example — contrastive learning with numbers |
| `05_donut_end_to_end.md` | Worked example — OCR-free document parsing |
| `06_layoutlm_end_to_end.md` | Worked example — LayoutLM v3 invoice extraction |
| `07_modern_vlms_2024_2025.md` | **SSOT:** LLaVA / Qwen-VL / Pixtral / Llama-3.2-Vision / Florence-2 / Phi-3-Vision / InternVL |
| `08_audio_multimodal.md` | Whisper, audio fusion, native multimodal (Gemini Live, GPT-4o Realtime) |
| `09_vlm_hallucination_mitigation.md` | Failure modes + detection patterns |

---

## SSOT Topics Owned Here

- Modern VLMs (LLaVA / Qwen-VL / Pixtral / Llama-3.2-Vision) → `07_modern_vlms_2024_2025.md`
- Document AI (LayoutLM / Donut / ColPali / native-VLM parsing) → `02_document_ai.md`
- Audio multimodal → `08_audio_multimodal.md`
- VLM hallucination mitigation → `09_vlm_hallucination_mitigation.md`

---

## Connections

- **Pure vision** (CNN, ViT, segmentation): `../3.computerVision/`
- **Self-supervised vision foundation models** (DINOv2, MAE, I-JEPA, CLIP, SigLIP): `../3.computerVision/02_applications/05_self_supervised_vision.md`
- **Transformer architecture** (ViT is a transformer): `../5.transformers/01_fundamentals/05_vision_transformers.md`
- **Generative models** (Diffusion / Stable Diffusion / DiT): `../2.deep learning/02_architectures/05_generative.md`
- **Document RAG** (visual retrieval): `../7.rag/`
- **Document processing system design**: `../11.system_design/04_document_processing_pipeline.md`

---

## Practice

No dedicated multimodal phase in `code_practice/` yet. Document AI work happens via Sameer's `../11.system_design/` and via `../code_practice/05_rag/`.

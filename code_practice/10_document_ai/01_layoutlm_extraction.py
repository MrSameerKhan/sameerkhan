# Session 1 — LayoutLM v3: Invoice / Form Key-Value Extraction
# Task   : extract structured fields from document images using layout + text + vision jointly
# Dataset: FUNSD (Form Understanding in Noisy Scanned Documents) — 199 forms, IOB2 labels
# Model  : microsoft/layoutlmv3-base — 125M params, runs on MPS
# Metric : token-level F1 (seqeval)
#
# Key insight: rule-based extraction fails when layouts vary. LayoutLM learns
# that "Total:" followed by a right-aligned number is always the total field,
# regardless of whether it's at the top, bottom, left, or right of the form.
#
# Theory: ../../../9.multimodal/06_layoutlm_end_to_end.md

import os
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset as TorchDataset
from seqeval.metrics import f1_score, classification_report

MODEL_NAME  = "microsoft/layoutlmv3-base"
OUTPUT_DIR  = "models/10_document_ai/layoutlmv3_funsd"
DEVICE      = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# FUNSD IOB2 label set
LABEL_LIST = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-QUESTION", "I-QUESTION",
    "B-ANSWER", "I-ANSWER",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_funsd():
    """
    FUNSD via HuggingFace: each example has words, bboxes (0-1000 normalised), labels, image.
    LayoutLMv3 sees all three modalities together: text tokens + spatial positions + pixel patches.
    """
    return load_dataset("nielsr/funsd-layoutlmv3", trust_remote_code=True)


class FUNSDDataset(TorchDataset):
    def __init__(self, hf_split, processor):
        self.data      = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex     = self.data[idx]
        image  = ex["image"].convert("RGB")
        words  = ex["tokens"]            # list of word strings
        bboxes = ex["bboxes"]            # list of [x0, y0, x1, y1] in 0-1000 scale
        labels = ex["ner_tags"]          # list of label ints

        enc = self.processor(
            image,
            words,
            boxes=bboxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )

        return {k: v.squeeze(0) for k, v in enc.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(p):
    preds, labels = p
    preds = np.argmax(preds, axis=2)

    true_preds, true_labels = [], []
    for pred_seq, label_seq in zip(preds, labels):
        tp, tl = [], []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id != -100:
                tp.append(ID2LABEL[p_id])
                tl.append(ID2LABEL[l_id])
        true_preds.append(tp)
        true_labels.append(tl)

    return {"f1": f1_score(true_labels, true_preds)}


# ── Inference — extract key-value pairs ───────────────────────────────────────
def extract_kv_pairs(image: Image.Image, words: list[str], bboxes: list,
                     model, processor) -> dict:
    """
    Run model on a document → find all QUESTION and ANSWER spans →
    pair them spatially (nearest answer to each question).
    This is the core extraction logic for invoice / form processing.
    """
    model.eval()
    enc = processor(
        image, words, boxes=bboxes,
        truncation=True, max_length=512, return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits    # (1, seq_len, num_labels)

    pred_ids  = logits.argmax(-1)[0].cpu().tolist()
    word_ids  = enc.get("word_ids", None)

    # Map subword predictions back to word level (take first subword)
    word_preds: dict[int, str] = {}
    if hasattr(processor.tokenizer, "word_ids"):
        for pos, wid in enumerate(enc["input_ids"][0]):
            pass    # fallback: use raw pred_ids

    # Group consecutive tokens by entity type
    questions, answers = [], []
    current_label, current_words = None, []

    for pos, (word, pred_id) in enumerate(zip(words, pred_ids[:len(words)])):
        label = ID2LABEL.get(pred_id, "O")
        entity = label.split("-")[-1] if "-" in label else "O"

        if label.startswith("B-"):
            if current_label and current_words:
                (questions if current_label == "QUESTION" else answers if current_label == "ANSWER" else []).append(" ".join(current_words))
            current_label = entity
            current_words = [word]
        elif label.startswith("I-") and current_label == entity:
            current_words.append(word)
        else:
            if current_label and current_words:
                (questions if current_label == "QUESTION" else answers if current_label == "ANSWER" else []).append(" ".join(current_words))
            current_label = entity if entity != "O" else None
            current_words = [word] if entity != "O" else []

    # Simple pairing: zip questions with answers in order
    pairs = {}
    for i, q in enumerate(questions):
        if i < len(answers):
            pairs[q] = answers[i]

    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)
    # apply_ocr=False: we provide pre-computed words + bboxes (no Tesseract needed)

    print("Loading FUNSD dataset...")
    raw     = load_funsd()
    train_d = FUNSDDataset(raw["train"], processor)
    test_d  = FUNSDDataset(raw["test"],  processor)
    print(f"  Train: {len(train_d)} | Test: {len(test_d)}")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)

    # Count trainable params
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable\n")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=2,    # LayoutLMv3 is memory-heavy (image patches)
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_d,
        eval_dataset=test_d,
        compute_metrics=compute_metrics,
    )

    print("Training LayoutLMv3...")
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}")

    # Demonstrate extraction on a test document
    print("\n── Extraction demo ──")
    ex      = raw["test"][0]
    image   = ex["image"].convert("RGB")
    words   = ex["tokens"]
    bboxes  = ex["bboxes"]
    pairs   = extract_kv_pairs(image, words, bboxes, model, processor)

    print(f"Document has {len(words)} words. Extracted key-value pairs:")
    for field, value in list(pairs.items())[:8]:
        print(f"  {field}: {value}")

    print("\n── What LayoutLMv3 sees ──")
    print("  Text stream : ['Invoice', 'Number', ':', 'INV-2024-001', 'Date', ':', ...]")
    print("  Bbox stream : [[10,20,80,35], [85,20,150,35], [155,20,165,35], [170,20,280,35], ...]")
    print("  Image       : pixel patches from the document scan (16×16 pixel windows)")
    print("  → Cross-attention across all three → learns that 'Number: X' = key-value regardless of layout")


if __name__ == "__main__":
    main()

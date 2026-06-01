# Session 2 — Donut: OCR-Free Document Understanding
# Task   : parse document images end-to-end without an OCR step
# Model  : naver-clova-ix/donut-base-finetuned-docvqa (pre-trained, no fine-tuning needed)
#          naver-clova-ix/donut-base-finetuned-cord-v2 (for structured receipt extraction)
# Key    : Donut reads pixel values directly → generates structured JSON tokens
#          No OCR errors, no OCR dependency, no bounding box alignment
#
# Theory: ../../../9.multimodal/05_donut_end_to_end.md

import re
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DonutProcessor, VisionEncoderDecoderModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ── Model loading ─────────────────────────────────────────────────────────────
def load_docvqa():
    """Donut fine-tuned on DocVQA: answers questions about document images."""
    name      = "naver-clova-ix/donut-base-finetuned-docvqa"
    processor = DonutProcessor.from_pretrained(name)
    model     = VisionEncoderDecoderModel.from_pretrained(name).to(DEVICE)
    model.eval()
    return processor, model


def load_cord():
    """Donut fine-tuned on CORD: structured receipt/invoice parsing → JSON output."""
    name      = "naver-clova-ix/donut-base-finetuned-cord-v2"
    processor = DonutProcessor.from_pretrained(name)
    model     = VisionEncoderDecoderModel.from_pretrained(name).to(DEVICE)
    model.eval()
    return processor, model


# ── Inference: DocVQA ─────────────────────────────────────────────────────────
def docvqa_answer(image: Image.Image, question: str, processor, model) -> str:
    """
    Donut DocVQA inference:
    1. Encode document image → visual features (ViT encoder)
    2. Prompt decoder with task token + question
    3. Generate answer tokens auto-regressively
    4. Decode to text
    """
    prompt    = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    dec_input = processor.tokenizer(
        prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(DEVICE)

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=dec_input,
            max_new_tokens=128,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )

    seq     = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    answer  = seq.split("<s_answer>")[-1].strip()
    # Strip any remaining tags
    answer  = re.sub(r"<[^>]+>", "", answer).strip()
    return answer


# ── Inference: Structured extraction (CORD) ───────────────────────────────────
def extract_structure(image: Image.Image, processor, model) -> dict:
    """
    Donut CORD inference: image → structured JSON without OCR.
    Output schema: {menu: [{nm, cnt, price}], sub_total: {...}, total: {...}}
    """
    task_prompt = "<s_cord-v2>"
    dec_input   = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(DEVICE)

    pixel_values = processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=dec_input,
            max_new_tokens=512,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )

    seq  = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # Convert Donut token output to dict
    data = processor.token2json(seq)
    return data


# ── Synthetic document image creator ─────────────────────────────────────────
def make_invoice_image() -> Image.Image:
    """
    Create a simple synthetic invoice image using PIL.
    In production: load real document scans or convert PDFs with PyMuPDF.
    """
    img  = Image.new("RGB", (600, 800), color="white")
    draw = ImageDraw.Draw(img)

    # Use default font (PIL built-in)
    lines = [
        ("INVOICE", 30, 40, 18),
        ("Invoice Number: INV-2024-00847", 30, 80, 12),
        ("Date: November 15, 2024", 30, 105, 12),
        ("Due Date: December 15, 2024", 30, 130, 12),
        ("", 30, 155, 12),
        ("From: DataSync Solutions Ltd.", 30, 180, 12),
        ("To: Al Rajhi Capital, Riyadh", 30, 205, 12),
        ("", 30, 230, 12),
        ("SERVICES", 30, 260, 14),
        ("API Integration Services   4 days @ $1,200/day    $4,800.00", 30, 290, 11),
        ("Data Migration Consulting  2 days @ $950/day      $1,900.00", 30, 315, 11),
        ("Training Sessions          3 hours @ $350/hr      $1,050.00", 30, 340, 11),
        ("", 30, 365, 11),
        ("Subtotal:  $7,750.00", 30, 400, 12),
        ("VAT (15%): $1,162.50", 30, 425, 12),
        ("TOTAL DUE: $8,912.50", 30, 450, 14),
        ("", 30, 490, 12),
        ("Payment Terms: Net 30", 30, 520, 12),
        ("Currency: USD", 30, 545, 12),
    ]

    for text, x, y, size in lines:
        if text:
            font = ImageFont.load_default()
            draw.text((x, y), text, fill="black", font=font)

    draw.rectangle([20, 20, 580, 780], outline="black", width=2)
    draw.line([20, 60, 580, 60], fill="black", width=1)
    draw.line([20, 380, 580, 380], fill="black", width=1)
    return img


def make_mortgage_doc_image() -> Image.Image:
    """Synthetic mortgage document for demo."""
    img  = Image.new("RGB", (600, 700), color="white")
    draw = ImageDraw.Draw(img)

    lines = [
        "MORTGAGE APPLICATION SUMMARY",
        "",
        "Applicant: Sarah Mitchell",
        "Property Address: 14 Elmwood Close, Manchester M20 4PQ",
        "Purchase Price: £385,000",
        "Loan Amount: £320,000",
        "LTV Ratio: 83.1%",
        "Fixed Rate Product: 5-year fixed at 4.61%",
        "Term: 25 years (300 months)",
        "Monthly Payment: £1,788.43",
        "",
        "Down Payment: £65,000 (16.9%)",
        "First-Time Buyer: Yes",
        "",
        "AFFORDABILITY CHECK",
        "Gross Monthly Income: £6,200",
        "Existing Monthly Debts: £450",
        "Debt-Service Ratio: 35.9% (within 45% limit)",
        "Stress Rate: 7.61% (product rate + 3%)",
        "",
        "STATUS: APPROVED",
        "Decision Date: 2024-11-15",
    ]

    y = 30
    for line in lines:
        if line:
            draw.text((30, y), line, fill="black", font=ImageFont.load_default())
        y += 28

    draw.rectangle([15, 15, 585, y + 10], outline="black", width=2)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    # ── Demo 1: DocVQA — question answering on document image ─────────────────
    print("Loading Donut DocVQA model...")
    proc_vqa, model_vqa = load_docvqa()

    invoice_img  = make_invoice_image()
    mortgage_img = make_mortgage_doc_image()

    print("\n── Document VQA (OCR-free) ──")
    vqa_tests = [
        (invoice_img,  "What is the total amount due?"),
        (invoice_img,  "What is the invoice number?"),
        (invoice_img,  "What is the VAT rate?"),
        (mortgage_img, "What is the loan amount?"),
        (mortgage_img, "What is the applicant's name?"),
        (mortgage_img, "Is the application approved?"),
    ]

    for img, question in vqa_tests:
        answer = docvqa_answer(img, question, proc_vqa, model_vqa)
        print(f"  Q: {question}")
        print(f"  A: {answer}\n")

    # ── Demo 2: Structured extraction (CORD) ──────────────────────────────────
    print("\nLoading Donut CORD model (receipt/invoice structured extraction)...")
    proc_cord, model_cord = load_cord()

    print("\n── Structured extraction (image → JSON, no OCR) ──")
    result = extract_structure(invoice_img, proc_cord, model_cord)
    print(json.dumps(result, indent=2))

    # ── Key insight summary ────────────────────────────────────────────────────
    print("\n── Why Donut vs OCR → NLP pipeline ──")
    print("""
  Traditional pipeline:
    Image → OCR (errors) → NLP model → extraction
    Problem: OCR errors cascade — "Totai" instead of "Total", garbled amounts

  Donut:
    Image → ViT encoder → causal decoder → JSON tokens
    One model, one inference call, no OCR dependency

  ViT encoder: 14×14 patch embeddings across the document image
  Decoder:     auto-regressively generates task-specific tokens
               DocVQA: <s_answer>$8,912.50</s_answer>
               CORD:   <s_menu><s_nm>API Integration</s_nm>...
    """)


if __name__ == "__main__":
    main()

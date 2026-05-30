# Session 3 — ColPali: Vision-RAG on Document Images
# Task   : retrieve relevant document pages by visual similarity (no OCR needed)
# Model  : vidore/colpali-v1.2 (PaliGemma 3B + ColBERT-style late interaction)
# Shows  : image patch embeddings, MaxSim scoring, visual RAG pipeline
#
# ColPali paper: Faysse et al. 2024 — "ColPali: Efficient Document Retrieval with Vision LMs"
# Key insight: instead of extracting text → embedding text, embed the entire page as image patches.
# Tables, charts, handwriting, complex layouts all become retrievable.
#
# ⚠️  ColPali (3B params) requires GPU with ≥8 GB VRAM. Falls back to CLIP on MPS/CPU.
#     Fallback demonstrates the same visual retrieval concept with a much smaller model.
#
# Theory: ../../../9.multimodal/02_document_ai.md

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
USE_COLPALI = torch.cuda.is_available()   # ColPali needs CUDA; CLIP fallback on MPS/CPU


# ── ColPali retrieval ─────────────────────────────────────────────────────────
def colpali_retrieve(queries: list[str], page_images: list[Image.Image], top_k: int = 3):
    """
    ColPali late interaction retrieval:
    1. Encode each page as a set of image patch embeddings (N_patches per page)
    2. Encode each query as a set of token embeddings (N_tokens)
    3. MaxSim: for each query token, find its most similar patch; sum across tokens
    4. Rank pages by MaxSim score
    """
    from colpali_engine.models import ColPali, ColPaliProcessor

    print("Loading ColPali (vidore/colpali-v1.2) — ~6 GB download first run...")
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")

    # Encode document pages — each page → (N_patches, D) embedding matrix
    batch_images = processor.process_images(page_images).to("cuda")
    with torch.no_grad():
        image_embeddings = model(**batch_images)   # (N_pages, N_patches, D)

    results = []
    for query in queries:
        # Encode query → (N_tokens, D)
        batch_query = processor.process_queries([query]).to("cuda")
        with torch.no_grad():
            query_embeddings = model(**batch_query)   # (1, N_tokens, D)

        # MaxSim scoring: for each query token, max similarity across all patches of a page
        # Final score per page = sum of per-token MaxSim
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
        top_ids = scores[0].topk(min(top_k, len(page_images))).indices.tolist()
        results.append([(i, float(scores[0][i])) for i in top_ids])

    return results


# ── CLIP fallback retrieval (MPS/CPU compatible) ──────────────────────────────
def clip_retrieve(queries: list[str], page_images: list[Image.Image], top_k: int = 3):
    """
    CLIP-based visual retrieval (concept demonstration for non-CUDA machines).
    Single embedding per page (vs ColPali's patch-level late interaction).
    Less accurate on complex layouts but shows the same architectural idea.
    """
    from transformers import CLIPProcessor, CLIPModel

    print("Loading CLIP (openai/clip-vit-base-patch32) — ~600 MB...")
    clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    clip_model.eval()

    # Encode all pages
    page_inputs = clip_proc(images=page_images, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        page_embs = clip_model.get_image_features(**page_inputs)
        page_embs = page_embs / page_embs.norm(dim=-1, keepdim=True)  # L2-norm

    results = []
    for query in queries:
        query_inputs = clip_proc(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            q_emb = clip_model.get_text_features(**query_inputs)
            q_emb = q_emb / q_emb.norm(dim=-1, keepdim=True)

        scores  = (page_embs @ q_emb.T).squeeze(-1)   # (N_pages,)
        top_ids = scores.topk(min(top_k, len(page_images))).indices.tolist()
        results.append([(i, float(scores[i])) for i in top_ids])

    return results


# ── Synthetic document page factory ──────────────────────────────────────────
def make_page(title: str, content: str, color: str = "white") -> Image.Image:
    img  = Image.new("RGB", (595, 842), color=color)   # A4 at 72dpi
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.rectangle([20, 20, 575, 822], outline="navy", width=2)
    draw.text((30, 35), title, fill="navy", font=font)
    draw.line([20, 60, 575, 60], fill="navy", width=1)

    y = 80
    for line in content.split("\n"):
        draw.text((30, y), line, fill="black", font=font)
        y += 18
    return img


# ── Synthetic document corpus ─────────────────────────────────────────────────
def build_corpus() -> tuple[list[Image.Image], list[str]]:
    """
    8 document pages simulating a mortgage application file.
    In production: convert PDF pages to images with PyMuPDF (see session 04).
    """
    pages = [
        make_page("Page 1 — Mortgage Application Cover", "Applicant: Sarah Mitchell\nDate: 15 November 2024\nProduct: 5-Year Fixed Rate Mortgage\nProperty: 14 Elmwood Close, Manchester M20 4PQ\nApplication Reference: APP-2024-08471"),
        make_page("Page 2 — Loan Details", "Loan Amount: £320,000\nProperty Value: £390,000\nLTV Ratio: 82.1%\nMonthly Payment: £1,788.43\nFixed Rate: 4.61%\nTerm: 25 years\nFirst-Time Buyer: Yes"),
        make_page("Page 3 — Income & Affordability", "Monthly Gross Income: £6,200\nEmployer: TechCorp Ltd (4 years)\nExisting Monthly Debts: £450\nDebt-Service Ratio: 35.9%\nAffordability Status: PASS\nStress Rate: 7.61%"),
        make_page("Page 4 — Property Valuation Report", "Address: 14 Elmwood Close, Manchester M20 4PQ\nValuation Date: 10 November 2024\nEstimated Market Value: £390,000\nValuation Method: Comparable Sales\nValuer: Jones & Partners RICS\nCondition: Good"),
        make_page("Page 5 — Bank Statements Summary", "Account: Barclays current account\nPeriod: August - October 2024\nAverage Monthly Credit: £6,205\nAverage Monthly Debit: £3,840\nEnd Balance Oct: £12,430\nNo irregular large withdrawals noted"),
        make_page("Page 6 — Early Repayment Charges", "ERC Schedule — 5-Year Fixed Product:\nYear 1: 5% of outstanding balance\nYear 2: 4% of outstanding balance\nYear 3: 3% of outstanding balance\nYear 4: 2% of outstanding balance\nYear 5: 1% of outstanding balance\nAfter fixed period: No ERC"),
        make_page("Page 7 — Insurance Requirements", "Buildings Insurance: Required from day of completion\nMinimum Coverage: £390,000 (rebuild value)\nLife Insurance: Recommended but not mandatory\nPayment Protection: Optional\nProvider: Customer's choice — provide certificate at completion"),
        make_page("Page 8 — Decision Notice", "APPLICATION STATUS: APPROVED\nDecision Date: 15 November 2024\nOffer Valid Until: 15 February 2025\nConditions: Satisfactory buildings insurance certificate required\nNext Step: Sign mortgage deed within 30 days\nContact: mortgage.offers@bank.co.uk"),
    ]
    titles = [
        "Cover page / application overview",
        "Loan amount, LTV, monthly payment",
        "Income, affordability, debt-service ratio",
        "Property valuation report",
        "Bank statements summary",
        "Early repayment charge schedule",
        "Insurance requirements",
        "Decision notice and approval status",
    ]
    return pages, titles


# ── Visual RAG pipeline ───────────────────────────────────────────────────────
def visual_rag(query: str, page_images, page_titles, retrieve_fn, top_k: int = 3) -> list[dict]:
    results_raw = retrieve_fn([query], page_images, top_k=top_k)[0]
    return [
        {"rank": i+1, "page": idx+1, "title": page_titles[idx], "score": score}
        for i, (idx, score) in enumerate(results_raw)
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    model_label = "ColPali (late interaction)" if USE_COLPALI else "CLIP (single-vector fallback)"
    print(f"Model:  {model_label}\n")

    print("Building document corpus...")
    pages, titles = build_corpus()
    print(f"  {len(pages)} pages in corpus\n")

    retrieve_fn = colpali_retrieve if USE_COLPALI else clip_retrieve

    queries = [
        "What is the early repayment charge in year 3?",
        "Is the mortgage application approved?",
        "What insurance is required?",
        "What is the monthly payment amount?",
    ]

    print("── Visual document retrieval ──\n")
    for query in queries:
        results = visual_rag(query, pages, titles, retrieve_fn, top_k=3)
        print(f"Query: {query}")
        for r in results:
            print(f"  #{r['rank']} Page {r['page']:2d} ({r['score']:.3f}) — {r['title']}")
        print()

    # Architecture comparison
    print("── ColPali vs CLIP vs text-RAG ──")
    print("""
  Text RAG:
    PDF → OCR → chunk text → embed text → cosine search
    Fails on: tables, handwritten notes, charts, complex layouts, scans with OCR errors

  CLIP single-vector:
    Page image → single 512D embedding → cosine search
    Better than text RAG for visual content but limited precision

  ColPali late interaction:
    Page image → 1030 patch embeddings (each patch = 14×14 pixels)
    Query text → N token embeddings
    MaxSim: query_token_j → max similarity over all patches → sum
    Precision: each query token finds its most relevant page region
    Recovers: tables (each cell is a patch), charts (visual patterns), handwriting

  MaxSim formula:
    score(query q, page p) = Σⱼ max_i sim(q_j, p_i)
    where q_j = j-th query token embedding, p_i = i-th page patch embedding
    """)


if __name__ == "__main__":
    main()

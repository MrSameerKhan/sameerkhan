# Session 7 — Sentence Transformers: Semantic Document Search
# Task   : encode documents as dense vectors, retrieve the most similar ones to a query
# Dataset: SNLI (Natural Language Inference) — pairs with entailment labels for fine-tuning
# Model  : sentence-transformers/all-MiniLM-L6-v2 (pre-trained) → fine-tune with contrastive loss
# Metric : cosine similarity, Recall@k on a held-out query set
#
# Bridge to RAG: the same bi-encoder + cosine search pattern is the retrieval stage of RAG.
# In production: replace numpy search with FAISS / pgvector for millions of documents.

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 2
LR          = 2e-5
TRAIN_SIZE  = 4000
SAVE_DIR    = "models/05_transformers/sentence_transformers"
DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# ── Mean pooling ──────────────────────────────────────────────────────────────
def mean_pool(token_embeddings, attention_mask):
    """Average token embeddings weighted by attention mask — ignores padding."""
    mask_expanded = attention_mask.unsqueeze(-1).float()           # (B, L, 1)
    summed        = (token_embeddings * mask_expanded).sum(dim=1)  # (B, H)
    counts        = mask_expanded.sum(dim=1).clamp(min=1e-9)       # (B, 1)
    return F.normalize(summed / counts, p=2, dim=1)                # (B, H) unit-normalised


def encode(texts: list[str], model, tokenizer) -> torch.Tensor:
    model.eval()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
    return mean_pool(out.last_hidden_state, enc["attention_mask"])  # (N, 384)


# ── Dataset — MultipleNegativesRankingLoss style ──────────────────────────────
class NLIPairDataset(Dataset):
    """Use SNLI entailment pairs as (anchor, positive) for contrastive training."""

    def __init__(self, hf_split, tokenizer):
        # Keep only entailment pairs (label=0)
        self.data      = [ex for ex in hf_split if ex["label"] == 0 and ex["hypothesis"].strip()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor   = self.data[idx]["premise"]
        positive = self.data[idx]["hypothesis"]

        enc_a = self.tokenizer(anchor,   max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
        enc_p = self.tokenizer(positive, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "anchor_ids":      enc_a["input_ids"].squeeze(0),
            "anchor_mask":     enc_a["attention_mask"].squeeze(0),
            "positive_ids":    enc_p["input_ids"].squeeze(0),
            "positive_mask":   enc_p["attention_mask"].squeeze(0),
        }


# ── MultipleNegativesRankingLoss ──────────────────────────────────────────────
def mnrl_loss(anchor_emb: torch.Tensor, positive_emb: torch.Tensor) -> torch.Tensor:
    """
    In-batch negatives: every other (anchor, positive) pair in the batch is a negative.
    scores[i][j] = cosine_sim(anchor_i, positive_j)
    Positive = diagonal; negatives = off-diagonal.
    Cross-entropy over rows → pushes anchor closer to its positive, away from all others.
    """
    scores = anchor_emb @ positive_emb.T      # (B, B) cosine sims (both already L2-normalised)
    labels = torch.arange(scores.size(0), device=DEVICE)
    return F.cross_entropy(scores * 20.0, labels)   # temperature scaling (20 = 1/0.05)


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()

        out_a = model(input_ids=batch["anchor_ids"].to(DEVICE),   attention_mask=batch["anchor_mask"].to(DEVICE))
        out_p = model(input_ids=batch["positive_ids"].to(DEVICE), attention_mask=batch["positive_mask"].to(DEVICE))

        emb_a = mean_pool(out_a.last_hidden_state, batch["anchor_mask"].to(DEVICE))
        emb_p = mean_pool(out_p.last_hidden_state, batch["positive_mask"].to(DEVICE))

        loss = mnrl_loss(emb_a, emb_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# ── Semantic search ───────────────────────────────────────────────────────────
def semantic_search(query: str, corpus: list[str], corpus_emb: torch.Tensor, model, tokenizer, top_k: int = 3):
    query_emb = encode([query], model, tokenizer)                      # (1, 384)
    scores    = (query_emb @ corpus_emb.T).squeeze(0)                 # (N,)
    top_idx   = scores.topk(top_k).indices.tolist()
    return [(corpus[i], scores[i].item()) for i in top_idx]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    # Fine-tuning on NLI entailment pairs
    raw       = load_dataset("snli")
    train_raw = raw["train"].select(range(TRAIN_SIZE))

    train_loader = DataLoader(
        NLIPairDataset(train_raw, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch}/{EPOCHS} | loss: {loss:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\nSaved to {SAVE_DIR}")

    # Semantic search demo — simulated financial document corpus
    print("\n── Semantic Search ──")
    corpus = [
        "The maximum loan-to-value ratio is 90% for standard residential mortgages.",
        "Early repayment charges of 2% apply within the initial fixed-rate period.",
        "Interest rates are reviewed quarterly based on the Bank of England base rate.",
        "First-time buyers are eligible for a 95% LTV mortgage under the Help to Buy scheme.",
        "All applications require proof of income for the past three months.",
        "The fixed-rate period lasts five years before reverting to the variable rate.",
        "Documents must be submitted within 30 days of the initial application date.",
        "Joint applications require both applicants to provide identification documents.",
    ]

    corpus_emb = encode(corpus, model, tokenizer)   # (8, 384)

    queries = [
        "What is the LTV limit for first-time buyers?",
        "How much do I pay if I repay the loan early?",
        "What income documents do I need to provide?",
    ]
    for query in queries:
        results = semantic_search(query, corpus, corpus_emb, model, tokenizer, top_k=2)
        print(f"\n  Query: {query}")
        for doc, score in results:
            print(f"    [{score:.3f}] {doc}")


if __name__ == "__main__":
    main()

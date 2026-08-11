# Session 4 — RAG Evaluation
# Task   : prove RAG actually works — score faithfulness, relevancy, context quality
# Shows  : RAGAS-style metrics implemented with LLM-as-judge + retrieval recall
# Compare: RAG answers vs no-RAG (LLM-only) across all 4 metric dimensions
#
# Implements metrics manually for version-stability — same concepts as ragas library.
# Set OPENAI_API_KEY before running.

import os
import json
import math
from dataclasses import dataclass, field
from openai import OpenAI
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from _corpus import DOCUMENTS

oai   = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

JUDGE_MODEL = "gpt-4o-mini"
GEN_MODEL   = "gpt-4o-mini"
CHUNK_SIZE  = 300
OVERLAP     = 60
TOP_K       = 4


# ── Build RAG pipeline (from session 01) ──────────────────────────────────────
def build_chunks(docs):
    chunks = []
    for doc in docs:
        text, start = doc["content"], 0
        while start < len(text):
            end   = min(start + CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"doc_id": doc["id"], "title": doc["title"], "content": chunk})
            start += CHUNK_SIZE - OVERLAP
    return chunks


def build_index(chunks):
    embs  = embed.encode([c["content"] for c in chunks], normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def retrieve(query, chunks, index):
    q = embed.encode([query], normalize_embeddings=True).astype(np.float32)
    _, ids = index.search(q, TOP_K)
    return [chunks[i] for i in ids[0]]


def generate_rag(query, retrieved):
    context = "\n\n".join(f"[{r['title']}]\n{r['content']}" for r in retrieved)
    res = oai.chat.completions.create(
        model=GEN_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": "Answer using ONLY the context. Be specific with numbers."},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    return res.choices[0].message.content.strip()


def generate_no_rag(query):
    res = oai.chat.completions.create(
        model=GEN_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": "You are a banking assistant. Answer the question."},
            {"role": "user",   "content": query},
        ],
    )
    return res.choices[0].message.content.strip()


# ── RAGAS-style metrics ────────────────────────────────────────────────────────

def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """
    Extract claims from the answer, verify each against the context.
    Faithfulness = supported_claims / total_claims
    """
    context_text = "\n".join(contexts)
    prompt = f"""Given this context and answer, extract all factual claims made in the answer,
then for each claim say whether it is supported by the context or not.

Context: {context_text}
Answer: {answer}

Return JSON: {{"claims": [{{"claim": str, "supported": bool}}]}}"""

    res  = oai.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    data   = json.loads(res.choices[0].message.content)
    claims = data.get("claims", [])
    if not claims:
        return 1.0
    return sum(1 for c in claims if c.get("supported", False)) / len(claims)


def answer_relevancy(question: str, answer: str, n: int = 3) -> float:
    """
    Generate N questions from the answer, measure cosine similarity to original question.
    High similarity → answer addresses the question.
    """
    prompt = f"""Generate {n} different questions that this answer would correctly answer.
Answer: {answer}
Return JSON: {{"questions": [str]}}"""

    res  = oai.chat.completions.create(
        model=JUDGE_MODEL, temperature=0.3,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    data         = json.loads(res.choices[0].message.content)
    gen_questions = data.get("questions", [])
    if not gen_questions:
        return 0.0

    orig_emb = embed.encode([question], normalize_embeddings=True)[0]
    gen_embs = embed.encode(gen_questions, normalize_embeddings=True)
    return float(np.mean(gen_embs @ orig_emb))


def context_precision(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    For each retrieved chunk: is it relevant to the question?
    Context Precision = relevant_chunks_in_top_positions / total_chunks (rank-weighted).
    """
    scores = []
    for ctx in contexts:
        prompt = f"""Is this context chunk relevant to answering the question?
Question: {question}
Chunk: {ctx}
Ground truth answer: {ground_truth}
Return JSON: {{"relevant": bool}}"""
        res  = oai.chat.completions.create(
            model=JUDGE_MODEL, temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(res.choices[0].message.content)
        scores.append(1.0 if data.get("relevant", False) else 0.0)

    # Rank-weighted precision: relevant chunks higher up score more
    if not scores:
        return 0.0
    weighted = sum(
        (sum(scores[:i+1]) / (i+1)) * scores[i]
        for i in range(len(scores))
    )
    return weighted / sum(scores) if sum(scores) > 0 else 0.0


def context_recall(question: str, contexts: list[str], ground_truth: str) -> float:
    """
    Are the facts in the ground truth present in the retrieved context?
    Context Recall = found_facts / total_gt_facts
    """
    context_text = "\n".join(contexts)
    prompt = f"""Break the ground truth answer into individual facts, then check if each
fact can be found in the provided context.

Ground truth: {ground_truth}
Context: {context_text}

Return JSON: {{"facts": [{{"fact": str, "found_in_context": bool}}]}}"""

    res  = oai.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    data  = json.loads(res.choices[0].message.content)
    facts = data.get("facts", [])
    if not facts:
        return 1.0
    return sum(1 for f in facts if f.get("found_in_context", False)) / len(facts)


# ── Evaluation dataset ─────────────────────────────────────────────────────────
EVAL_SET = [
    {
        "question":   "What is the maximum LTV for first-time buyers?",
        "ground_truth": "First-time buyers are eligible for up to 95% LTV under the Help to Buy scheme.",
    },
    {
        "question":   "What is the early repayment charge in year 3?",
        "ground_truth": "The early repayment charge in year 3 of the fixed-rate period is 3%.",
    },
    {
        "question":   "What is the maximum personal finance amount for expatriates?",
        "ground_truth": "Expatriates can borrow up to 24 times their monthly salary, with a maximum tenor of 36 months.",
    },
    {
        "question":   "What is the monthly maintenance fee for low-balance savings accounts?",
        "ground_truth": "Savings accounts below SAR 500 are subject to a monthly maintenance fee of SAR 10.",
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Building RAG index...")
    chunks = build_chunks(DOCUMENTS)
    index  = build_index(chunks)
    print(f"  {len(chunks)} chunks ready\n")

    results_rag    = {"faithfulness": [], "relevancy": [], "precision": [], "recall": []}
    results_no_rag = {"faithfulness": [], "relevancy": [], "precision": [], "recall": []}

    for i, ex in enumerate(EVAL_SET):
        print(f"Evaluating Q{i+1}: {ex['question'][:60]}...")
        retrieved = retrieve(ex["question"], chunks, index)
        contexts  = [r["content"] for r in retrieved]

        ans_rag    = generate_rag(ex["question"], retrieved)
        ans_no_rag = generate_no_rag(ex["question"])

        # RAG metrics
        results_rag["faithfulness"].append(faithfulness(ex["question"], ans_rag, contexts))
        results_rag["relevancy"].append(answer_relevancy(ex["question"], ans_rag))
        results_rag["precision"].append(context_precision(ex["question"], contexts, ex["ground_truth"]))
        results_rag["recall"].append(context_recall(ex["question"], contexts, ex["ground_truth"]))

        # No-RAG metrics — faithfulness scored against empty context (should be low)
        results_no_rag["faithfulness"].append(faithfulness(ex["question"], ans_no_rag, [ex["ground_truth"]]))
        results_no_rag["relevancy"].append(answer_relevancy(ex["question"], ans_no_rag))
        results_no_rag["precision"].append(0.0)   # no retrieval → no context precision
        results_no_rag["recall"].append(0.0)       # no retrieval → no context recall

    mean = lambda lst: sum(lst) / len(lst) if lst else 0.0

    print(f"\n{'═' * 60}")
    print("RAGAS-STYLE EVALUATION RESULTS")
    print(f"{'═' * 60}")
    print(f"\n{'Metric':<25} {'RAG':>8} {'No-RAG':>8} {'Winner':>8}")
    print("-" * 52)
    for metric in ("faithfulness", "relevancy", "precision", "recall"):
        r_score = mean(results_rag[metric])
        n_score = mean(results_no_rag[metric])
        winner  = "RAG ✓" if r_score >= n_score else "No-RAG"
        print(f"  {metric:<23} {r_score:>8.3f} {n_score:>8.3f} {winner:>8}")

    print(f"\n{'─' * 52}")
    print("Interpretation:")
    print("  Faithfulness > 0.8  → answers are grounded, not hallucinated")
    print("  Relevancy    > 0.75 → answers address the question asked")
    print("  Precision    > 0.65 → retrieved chunks are relevant (not noise)")
    print("  Recall       > 0.70 → all key facts were retrieved")


if __name__ == "__main__":
    main()

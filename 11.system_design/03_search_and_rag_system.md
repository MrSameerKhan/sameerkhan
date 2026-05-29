# 03 Search & RAG System Design

## Problem Statement

Design a semantic search system for a document corpus (e.g., enterprise knowledge base, legal docs, internal wiki). Scale: 10M documents, 10K QPS, latency < 200ms.

---

## Architecture

```
              OFFLINE (document ingestion)
  Documents + Parse + Chunk + Embed + Vector DB + BM25 Index
  Metadata extraction + Metadata store (Postgres)

                         ↓ (indexed docs available)

               ONLINE (search pipeline)
  Query
    + [Query Processing]
    + [Hybrid Retrieval]
    + [Reranking]
    + [Generation (for RAG)]
    + Response
```

---

## Indexing Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber
from pathlib import Path
import json

class DocumentIndexer:
    def __init__(self, vectorstore, bm25_index, embedding_model):
        self.vectorstore = vectorstore
        self.bm25 = bm25_index
        self.encoder = embedding_model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def index_document(self, doc_path: str, metadata: dict = None):
        """Full indexing pipeline for one document."""
        # 1. Extract text
        text, layout_metadata = self.extract_text(doc_path)

        # 2. Chunk
        chunks = self.splitter.create_documents(
            [text],
            metadatas=[{
                "source": doc_path,
                "page": layout_metadata.get("page"),
                "section": layout_metadata.get("section"),
                "doc_type": metadata.get("type", "unknown"),
                "date": metadata.get("date"),
                **metadata
            }]
        )

        # 3. Embed
        texts = [c.page_content for c in chunks]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True, batch_size=64)

        # 4. Store in vector DB (dense) + BM25 index (sparse)
        self.vectorstore.add_documents(chunks)
        self.bm25.add_documents([texts, (c.metadata for c in chunks)])

        return len(chunks)

    def extract_text(self, doc_path: str) -> tuple[str, dict]:
        path = Path(doc_path)
        if path.suffix == ".pdf":
            with pdfplumber.open(path) as pdf:
                pages = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if pages:
                        pages.append(text)
                return "\n\n".join(pages), {"total_pages": len(pdf.pages)}
        else:
            return path.read_text(), {}

# Batch indexing (parallel processing for large corpus)
from concurrent.futures import ThreadPoolExecutor

def batch_index(doc_paths: list, indexer: DocumentIndexer, workers: int = 8):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(indexer.index_document, path): path
                   for path in doc_paths}
        results = {}
        for future in futures:
            try:
                results[futures[future]] = future.result()
            except Exception as e:
                results[futures[future]] = f"Error: {e}"
    return results
```

---

## Hybrid Retrieval

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, vectorstore, bm25_corpus, embedding_model,
                 dense_weight=0.5, sparse_weight=0.5, k=50):
        self.vectorstore = vectorstore
        self.bm25 = BM25Okapi(bm25_corpus)
        self.encoder = embedding_model
        self.dense_w = dense_weight
        self.sparse_w = sparse_weight
        self.k = k

    def retrieve(self, query: str, filters: dict = None) -> list:
        # 1. Dense retrieval (semantic)
        query_emb = self.encoder.encode([query], normalize_embeddings=True)
        dense_results = self.vectorstore.similarity_search_with_score(
            query,
            k=self.k,
            filter=filters,   # metadata filtering (e.g., doc_type="contract")
        )
        dense_ids = {doc.metadata['chunk_id']: rank
                     for rank, (doc, score) in enumerate(dense_results)}

        # 2. Sparse retrieval (BM25)
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        sparse_top_k = np.argsort(bm25_scores)[::-1][:self.k]
        sparse_ids = {idx: rank for rank, idx in enumerate(sparse_top_k)}

        # 3. Reciprocal Rank Fusion (RRF)
        all_ids = set(dense_ids.keys()) | set(sparse_ids.keys())
        rrf_k = 60   # standard RRF constant

        rrf_scores = {}
        for chunk_id in all_ids:
            score = 0
            if chunk_id in dense_ids:
                score += self.dense_w / (rrf_k + dense_ids[chunk_id])
            if chunk_id in sparse_ids:
                score += self.sparse_w / (rrf_k + sparse_ids[chunk_id])
            rrf_scores[chunk_id] = score

        # Return top-k sorted by RRF score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:self.k]]
```

---

## Reranking

```python
from sentence_transformers import CrossEncoder

class SearchReranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[str], top_n: int = 5) -> list[int]:
        pairs = [(query, doc) for doc in candidates]
        scores = self.cross_encoder.predict(pairs)
        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices[:top_n].tolist()
```

---

## Query Processing

```python
class QueryProcessor:
    """Enhance the query before retrieval."""

    def __init__(self, llm):
        self.llm = llm

    def expand_query(self, query: str) -> list[str]:
        """Generate alternative phrasings of the query."""
        prompt = f"""Generate 3 alternative phrasings of this search query.
Return as JSON list. Keep the same meaning.

Query: {query}

Alternatives:"""
        alternatives = json.loads(self.llm(prompt))
        return [query] + alternatives[:2]   # original + 2 alternatives

    def extract_filters(self, query: str) -> dict:
        """Extract metadata filters from natural language query."""
        prompt = f"""Extract search filters from this query.
Return JSON with keys: doc_type, date_range, author.
Use null if not specified.

Query: {query}

Filters:"""
        return json.loads(self.llm(prompt))

    def decompose(self, complex_query: str) -> list[str]:
        """Decompose complex query into simpler sub-queries."""
        if len(complex_query.split()) < 15:
            return [complex_query]   # simple query, no decomposition

        prompt = f"""Break this complex question into 2-3 simpler sub-questions.
JSON list format.

Question: {complex_query}

Sub-questions:"""
        return json.loads(self.llm(prompt))
```

---

## RAG Generation

```python
from anthropic import Anthropic

client = Anthropic()

class RAGPipeline:
    def __init__(self, retriever, reranker, query_processor):
        self.retriever = retriever
        self.reranker = reranker
        self.query_processor = query_processor

    def answer(self, query: str) -> dict:
        # 1. Query processing
        filters = self.query_processor.extract_filters(query)
        sub_queries = self.query_processor.decompose(query)

        # 2. Retrieve for each sub-query
        all_candidates = []
        for sq in sub_queries:
            candidates = self.retriever.retrieve(sq, filters=filters)
            all_candidates.extend(candidates)

        # Deduplicate
        seen = set()
        unique_candidates = [c for c in all_candidates
                             if c.chunk_id not in seen and not seen.add(c.chunk_id)]

        # 3. Rerank
        top_chunks = self.reranker.rerank(query, unique_candidates, top_n=5)

        # 4. Format context
        context = self._format_context(top_chunks)

        # 5. Generate answer
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1000,
            system="""Answer questions using ONLY the provided context.
If not in context, say 'I don't have information about this.'
Always cite sources as [Source: filename, page X].""",
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }]
        )

        return {
            "answer": response.content[0].text,
            "sources": [c.metadata for c in top_chunks],
            "chunks_retrieved": len(unique_candidates),
        }

    def _format_context(self, chunks: list) -> str:
        parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "")
            parts.append(f"[{i+1}] Source: {source}, Page: {page}\n{chunk.page_content}")
        return "\n\n---\n\n".join(parts)
```

---

## Scalability

```
At 10K QPS with 10M documents:

Bottleneck analysis:
  - Embedding query: ~5ms (GPU, batched)
  - ANN search (FAISS): ~2ms for 10M vectors
  - BM25 search: ~10ms
  - Reranking (5 docs × cross-encoder): ~30ms
  - LLM generation: ~500ms (dominates for RAG)
  - Total: ~550ms for RAG, ~50ms for pure search

Optimization:
  1. Pre-compute: embed all documents offline (refresh daily for new docs)
  2. Caching: cache results for frequent queries (Redis, TTL=1hr)
  3. Tiered serving:
     - Search-only (no LLM): < 50ms
     - RAG (with LLM): < 500ms, async response or streaming
  4. ANN index sharding: partition by document category or date range
  5. Horizontal scaling: multiple retriever replicas behind load balancer
  6. LLM batching: vLLM for batched generation
```

---

## Evaluation

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Offline: RAGAS framework
results = evaluate(
    eval_dataset,   # {"question", "answer", "contexts", "ground_truth"}
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# Additional retrieval metrics
def recall_at_k(relevant_docs: set, retrieved_docs: list) -> float:
    """Fraction of relevant docs in top-k retrieved."""
    return len(relevant_docs & set(retrieved_docs)) / len(relevant_docs)

def mrr(relevant_docs: list, retrieved_docs: list) -> float:
    """Mean Reciprocal Rank"""
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1 / (i + 1)
    return 0.0
```

---

## Interview Q&A

**Q: How would you handle a query that spans multiple documents?**
A: Query decomposition + retrieve and rerank separately for each sub-query. In generation conditioned on retrieved context, explicitly instruct the LLM to synthesize information from multiple sources. Return citations for each source document. For conflicting information across documents — instruct the LLM to surface the conflict rather than arbitrarily choosing one.

**Q: How do you prevent the RAG system from hallucinating?**
A: Layered defense: (1) Retrieval quality — if relevant context isn't retrieved, generation will hallucinate; improve retrieval with hybrid search + reranking; (2) Prompt — "Answer ONLY using the provided information"; (3) Faithfulness scoring — use an NLI model to verify each claim in the answer is supported by retrieved context; (4) Citation enforcement — require the model to cite specific chunks; (5) Confidence thresholding — if top retrieved chunk similarity is below threshold, return "I don't have enough information" rather than generating from weak context.

---

## Connections

- RAG theory: `../7.rag/01_rag.md` — chunking, retrieval patterns, Self-RAG/CRAG
- RAG pipeline: `../7.rag/02_rag_pipeline.md` — dense vs BM25 vs hybrid, reranking decisions
- Indirect prompt injection: `../7.rag/03_indirect_prompt_injection.md` — top threat for RAG systems
- System Design Framework: `../10.system_design/01_ml_system_design_framework.md`

---

## Key Takeaway

Search system: hybrid retrieval (dense + sparse) → reranking (cross-encoder) → result. RAG adds: generation conditioned on retrieved context. Key components: chunking strategy, embedding model, ANN index, BM25 index, RRF fusion, cross-encoder reranking, LLM generation. The fundamental challenge: retrieval quality is the ceiling — generation can't produce what's not retrieved. Optimize retrieval first; tune generation second.

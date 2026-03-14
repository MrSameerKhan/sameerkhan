# Document Processing Pipeline Design

## Problem Statement
Design an end-to-end document processing pipeline for a company that receives 100K documents/day (invoices, contracts, receipts) and needs to extract structured data, classify document types, and route for processing.
Latency: < 30 seconds per document. Accuracy: > 95% field-level precision.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION                                       │
│  Email attachment / API upload / S3 bucket watch                   │
│  → Message queue (SQS/Kafka)                                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     PREPROCESSING                                    │
│  Format conversion (PDF → image) → Deskew → Denoise → Quality check│
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     CLASSIFICATION                                   │
│  Document type classifier → invoice / contract / receipt / other   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     EXTRACTION                                       │
│  OCR → Layout analysis → Field extraction → Validation             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     POST-PROCESSING                                  │
│  Schema validation → Business rules → Confidence scoring           │
│  → Human review queue (low confidence) → Output to downstream      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Document Classification

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
import torch

class DocumentClassifier:
    """Two-stage: fast text classifier → fallback to vision classifier."""

    def __init__(self):
        # Fast path: text-based classification (if OCR text available)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            "./doc-classifier-deberta"
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained("./doc-classifier-deberta")

        # Slow path: vision-based (for scanned docs, handwriting)
        # or use GPT-4V for zero-shot if training data insufficient
        self.classes = ["invoice", "purchase_order", "receipt", "contract",
                        "delivery_note", "bank_statement", "id_document", "other"]

    def classify(self, text: str = None, image: Image.Image = None) -> dict:
        if text and len(text.strip()) > 50:
            return self._text_classify(text[:512])
        elif image:
            return self._vision_classify(image)
        else:
            return {"class": "other", "confidence": 0.0}

    def _text_classify(self, text: str) -> dict:
        inputs = self.text_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.text_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        predicted_idx = probs.argmax().item()
        return {
            "class": self.classes[predicted_idx],
            "confidence": probs[predicted_idx].item(),
            "all_probs": {cls: prob.item() for cls, prob in zip(self.classes, probs)},
        }
```

---

## Extraction Engine

```python
from pydantic import BaseModel, validator
import re
from datetime import datetime
from typing import Optional

# ─── Schema definitions ───────────────────────────────────────────────────
class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    date: Optional[str] = None          # YYYY-MM-DD
    due_date: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    total_amount: Optional[float] = None
    tax_amount: Optional[float] = None
    currency: str = "USD"
    line_items: list[dict] = []
    payment_terms: Optional[str] = None

    @validator('date', 'due_date', pre=True)
    def normalize_date(cls, v):
        if v is None:
            return None
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%d-%m-%Y',
                    '%B %d, %Y', '%b %d, %Y', '%d %B %Y']:
            try:
                return datetime.strptime(str(v).strip(), fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        return None

    @validator('total_amount', 'tax_amount', pre=True)
    def normalize_amount(cls, v):
        if v is None:
            return None
        cleaned = re.sub(r'[^\d.,]', '', str(v)).replace(',', '')
        try:
            return float(cleaned)
        except ValueError:
            return None

SCHEMAS = {
    "invoice": InvoiceData,
    "receipt": ReceiptData,
    "contract": ContractData,
    # ...
}

# ─── Extraction strategies ────────────────────────────────────────────────
class ExtractionEngine:
    def __init__(self, ocr_engine, llm_client):
        self.ocr = ocr_engine
        self.llm = llm_client

    def extract(self, image: Image.Image, doc_type: str) -> dict:
        schema = SCHEMAS.get(doc_type)
        if schema is None:
            return {"error": f"No schema for doc_type={doc_type}"}

        # Strategy 1: Try rule-based extraction first (fast)
        ocr_text = self.ocr.extract(image)
        rule_result = self.rule_based_extract(ocr_text, doc_type)

        # Strategy 2: If rules fail on key fields, use LLM
        missing_fields = [f for f in schema.__fields__
                         if getattr(rule_result, f, None) is None
                         and schema.__fields__[f].required]

        if missing_fields:
            llm_result = self.llm_extract(image, ocr_text, doc_type, missing_fields)
            # Merge: prefer LLM for missing fields
            for field in missing_fields:
                setattr(rule_result, field, getattr(llm_result, field, None))

        return rule_result.dict()

    def rule_based_extract(self, text: str, doc_type: str) -> BaseModel:
        """Fast regex-based extraction."""
        schema = SCHEMAS[doc_type]

        # Invoice-specific patterns
        patterns = {
            "invoice_number": r"(?:invoice|inv|bill)\s*(?:no|number|#|num)?[:\s]*([A-Z0-9\-/]+)",
            "date": r"(?:invoice\s+date|date)[:\s]*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+ \d{1,2},?\s*\d{4})",
            "total_amount": r"(?:total|amount due|grand total)[:\s]*\$?([\d,]+\.?\d*)",
            "vendor_name": r"^([A-Z][A-Za-z\s&,\.]+(?:Inc|LLC|Ltd|Corp|Co)\.?)",
        }

        extracted = {}
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted[field] = match.group(1).strip()

        return schema(**extracted)

    def llm_extract(self, image, ocr_text: str, doc_type: str, missing_fields: list):
        """LLM extraction for fields rules missed."""
        schema = SCHEMAS[doc_type]
        schema_json = {f: str(schema.__fields__[f].outer_type_) for f in missing_fields}

        prompt = f"""Extract the following fields from this {doc_type}.
Return JSON with ONLY these fields: {json.dumps(missing_fields)}

Document text:
{ocr_text[:2000]}

Required schema: {json.dumps(schema_json)}
JSON:"""

        response = self.llm.complete(prompt)
        partial = json.loads(response)
        return schema(**partial)
```

---

## Confidence Scoring & Human Review Queue

```python
class ConfidenceScorer:
    """Score extraction confidence; route low-confidence to human review."""

    THRESHOLDS = {
        "auto_approve": 0.90,     # fully automated
        "human_review": 0.70,     # flag for review but continue processing
        "reject": 0.50,           # reject and send for manual reprocessing
    }

    def score(self, extracted_data: dict, doc_type: str, ocr_confidence: float) -> dict:
        field_scores = {}

        for field, value in extracted_data.items():
            if value is None:
                field_scores[field] = 0.0
            else:
                score = self._score_field(field, value, doc_type)
                field_scores[field] = score

        # Overall confidence: weighted average (weight important fields more)
        important_fields = {"invoice_number": 0.3, "date": 0.2, "total_amount": 0.3,
                           "vendor_name": 0.2}
        overall = sum(
            field_scores.get(f, 0) * w
            for f, w in important_fields.items()
        )

        # Penalize for low OCR confidence
        overall *= (0.5 + 0.5 * ocr_confidence)

        return {
            "overall_confidence": overall,
            "field_scores": field_scores,
            "decision": self._route(overall),
            "review_reason": self._review_reason(field_scores),
        }

    def _score_field(self, field: str, value, doc_type: str) -> float:
        validators = {
            "date": lambda v: 1.0 if re.match(r'\d{4}-\d{2}-\d{2}', str(v)) else 0.5,
            "total_amount": lambda v: 1.0 if isinstance(v, float) and v > 0 else 0.3,
            "invoice_number": lambda v: 1.0 if re.match(r'[A-Z0-9\-/]{3,20}', str(v)) else 0.6,
        }
        return validators.get(field, lambda v: 0.8)(value)

    def _route(self, confidence: float) -> str:
        if confidence >= self.THRESHOLDS["auto_approve"]:
            return "auto_approve"
        elif confidence >= self.THRESHOLDS["human_review"]:
            return "human_review"
        else:
            return "reject"

class HumanReviewQueue:
    """Manage documents requiring human review."""

    def queue_for_review(self, doc_id: str, extracted_data: dict, confidence_report: dict):
        review_item = {
            "doc_id": doc_id,
            "extracted_data": extracted_data,
            "confidence_report": confidence_report,
            "review_fields": [
                f for f, s in confidence_report["field_scores"].items()
                if s < 0.8
            ],
            "queued_at": datetime.utcnow().isoformat(),
            "priority": "high" if confidence_report["overall_confidence"] < 0.6 else "normal",
        }
        # Push to review queue (SQS, Redis, or DB)
        self.sqs.send_message(
            QueueUrl=self.review_queue_url,
            MessageBody=json.dumps(review_item),
        )
```

---

## Async Processing with Message Queue

```python
import boto3
import json
from concurrent.futures import ThreadPoolExecutor
import logging

class DocumentProcessor:
    def __init__(self, pipeline, queue_url, output_bucket):
        self.pipeline = pipeline
        self.sqs = boto3.client('sqs')
        self.s3 = boto3.client('s3')
        self.queue_url = queue_url
        self.output_bucket = output_bucket

    def process_message(self, message: dict):
        doc_id = message["doc_id"]
        doc_path = message["s3_path"]

        try:
            # Download document
            image = download_from_s3(doc_path)

            # Full pipeline
            result = self.pipeline.process(image, doc_id)

            # Store result
            self.s3.put_object(
                Bucket=self.output_bucket,
                Key=f"results/{doc_id}.json",
                Body=json.dumps(result),
            )

            # Notify downstream
            self.notify_downstream(doc_id, result)

            return True

        except Exception as e:
            logging.error(f"Failed to process {doc_id}: {e}")
            self.handle_failure(doc_id, str(e))
            return False

    def run(self, num_workers: int = 8):
        """Consume from SQS queue continuously."""
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            while True:
                response = self.sqs.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=20,  # long polling
                )

                messages = response.get("Messages", [])
                if not messages:
                    continue

                futures = []
                for msg in messages:
                    body = json.loads(msg["Body"])
                    future = executor.submit(self.process_message, body)
                    futures.append((future, msg["ReceiptHandle"]))

                for future, receipt_handle in futures:
                    if future.result():  # success
                        self.sqs.delete_message(
                            QueueUrl=self.queue_url,
                            ReceiptHandle=receipt_handle,
                        )
                    # On failure: message returns to queue after visibility timeout
```

---

## Metrics & SLAs

```
Throughput: 100K docs/day = ~1.2 docs/second average
  Peak: 10× = 12 docs/second
  Workers needed: 12 docs/s × 15s/doc = 180 concurrent workers (scale with k8s)

Latency SLA:
  P50: < 5 seconds
  P95: < 30 seconds
  P99: < 60 seconds (or human review queue)

Quality SLAs:
  Field-level precision: > 95%
  Field-level recall: > 90%
  Auto-approval rate: > 80% (human review for remainder)

Monitoring:
  - Processing queue depth (should not grow unboundedly)
  - Per-doc-type precision/recall (drift detection)
  - OCR confidence distribution
  - Human review rate (increase → model degraded)
  - Error rate by doc type and source
```

---

## Interview Q&A

**Q: How would you handle a 10× spike in document volume?**
A: The queue (SQS/Kafka) absorbs spikes naturally — documents sit in queue rather than overwhelming processing. Auto-scaling: Kubernetes HPA scales processing pods based on queue depth (SQS approximate message count as custom metric). For sustained high volume: pre-warm GPU instances, increase pod replicas pre-emptively for known high-traffic periods. Fast-path optimization: simple documents (clean PDFs with text layer) skip OCR and go directly to text extraction — route by document quality score.

**Q: How do you measure and improve extraction accuracy?**
A: Ground truth collection: sample 1% of documents for human annotation (stratified by doc type, confidence score, vendor). Measure field-level precision and recall per document type. Low precision = model hallucinates values. Low recall = model misses fields. Improvement strategies: (1) Add missed patterns to rule-based extractor; (2) Fine-tune extraction model on new failure examples (active learning — prioritize low-confidence extractions for annotation); (3) Pre/post-process for specific vendors (vendor-specific templates); (4) Ensemble: use multiple extractors and take majority vote for critical fields.

---

## Connections
- **Document AI (6.multimodal/02):** The models used in this pipeline (LayoutLM, Donut, OCR)
- **RAG System (8.system_design/03):** Extracted structured data can be stored and retrieved via RAG
- **MLOps (7.mlops/):** CI/CD, monitoring, and retraining apply directly to this pipeline
- **NLP Applications (3.nlp/applications/):** NER and IE are the core extraction techniques

## Key Takeaway
Document processing pipeline = Ingest → Classify → Extract → Validate → Route. Key design decisions: (1) message queue for async processing and spike handling, (2) multi-strategy extraction (rules fast, LLM for gaps), (3) confidence scoring with human review fallback, (4) per-field schema validation with Pydantic. The 95% precision target requires the human review loop — never trust pure automation for high-stakes document fields. Monitor human review rate as the primary quality signal.

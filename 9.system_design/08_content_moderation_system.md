# Content Moderation System Design

## Problem Statement
Design an automated content moderation system for a social platform.
Scale: 10M posts/day, <500ms moderation decision, 99.9% uptime.
Requirement: catch harmful content before it reaches users; minimize false positives.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    CONTENT INGESTION                              │
│  Post submitted → Kafka queue → Moderation pipeline             │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              AUTOMATED MODERATION PIPELINE                        │
│                                                                   │
│  [Fast pre-filter]      < 10ms  — blocklist, known hash match   │
│       ↓ pass                                                     │
│  [ML classifiers]       < 100ms — text + image models           │
│       ↓ score                                                    │
│  [Policy engine]        < 10ms  — apply thresholds + rules      │
│       ↓ decision                                                 │
│  AUTO_APPROVE / AUTO_REMOVE / HUMAN_REVIEW                      │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│              HUMAN REVIEW QUEUE (async)                          │
│  Borderline cases → prioritized review queue → moderator        │
│  Decisions feed back → model retraining                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stage 1 — Fast Pre-Filter (<10ms)

Catch obvious violations without running expensive ML models.

```python
import hashlib
import re
from typing import Optional

class PreFilter:
    def __init__(self, blocklist_path: str, hash_db_path: str):
        # Keyword blocklist (exact + regex)
        with open(blocklist_path) as f:
            self.blocklist = set(line.strip().lower() for line in f)

        # PhotoDNA / Microsoft hash database of known CSAM
        self.known_hashes = load_hash_db(hash_db_path)

    def check_text(self, text: str) -> Optional[str]:
        """Returns violation type or None."""
        text_lower = text.lower()

        # 1. Exact keyword match
        words = set(text_lower.split())
        if words & self.blocklist:
            matched = words & self.blocklist
            return f"BLOCKLIST:{','.join(matched)}"

        # 2. URL blocklist (known malware/phishing domains)
        urls = re.findall(r'https?://\S+', text)
        for url in urls:
            domain = extract_domain(url)
            if domain in self.blocked_domains:
                return f"BLOCKED_URL:{domain}"

        return None

    def check_image(self, image_hash: str) -> Optional[str]:
        """Perceptual hash match against known violation database."""
        if image_hash in self.known_hashes:
            return f"KNOWN_VIOLATION:{self.known_hashes[image_hash]}"
        return None

    def check(self, post: dict) -> Optional[str]:
        if "text" in post:
            result = self.check_text(post["text"])
            if result:
                return result
        if "image_hash" in post:
            result = self.check_image(post["image_hash"])
            if result:
                return result
        return None
```

---

## Stage 2 — ML Classifiers

### Text Classification

```python
from transformers import pipeline
import torch

class TextModerationClassifier:
    """
    Multi-label classifier: one score per violation category.
    Categories: hate_speech, harassment, spam, misinformation,
                adult_content, violence, self_harm
    """
    def __init__(self):
        # Fine-tuned RoBERTa on moderation dataset
        self.classifier = pipeline(
            "text-classification",
            model="facebook/roberta-hate-speech-dynabench-r4-target",
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,  # return all labels
        )

        # Custom fine-tuned multi-label model for all categories
        self.multi_classifier = load_finetuned_moderation_model()

    def score(self, text: str) -> dict:
        if len(text) > 512:
            text = text[:512]  # truncate to model limit

        results = self.multi_classifier(text)
        # Returns: {"hate_speech": 0.02, "harassment": 0.91, "spam": 0.05, ...}
        return results

class ImageModerationClassifier:
    """
    Scores: nudity, violence, gore, drugs, weapons
    Uses: NSFW detector + custom fine-tuned ViT
    """
    def __init__(self):
        self.nudity_model   = load_model("nudity_detector_vit_b16")
        self.violence_model = load_model("violence_detector_vit_b16")

    def score(self, image_bytes: bytes) -> dict:
        image = preprocess_image(image_bytes)
        return {
            "nudity":   float(self.nudity_model(image)),
            "violence": float(self.violence_model(image)),
            "gore":     float(self.gore_model(image)),
        }
```

### LLM-Based Moderation (High-Accuracy, Slower)

```python
from anthropic import Anthropic

client = Anthropic()

MODERATION_PROMPT = """You are a content moderation assistant. Analyze the following post and determine if it violates platform policies.

Categories to check:
- hate_speech: attacks based on race, religion, gender, sexual orientation, disability
- harassment: targeted abuse of a specific individual
- misinformation: false health/safety claims (NOT opinion or satire)
- spam: unsolicited commercial content, scam links
- violence: explicit threats or glorification of violence
- adult_content: explicit sexual content

Post: {text}

Respond in JSON:
{{"violation": true/false, "category": "category_name or null", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

def llm_moderate(text: str) -> dict:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # cheap + fast
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": MODERATION_PROMPT.format(text=text[:1000])
        }]
    )
    import json
    return json.loads(response.content[0].text)

# Use LLM for: borderline cases, complex context (sarcasm, satire), appeals
```

---

## Stage 3 — Policy Engine

Convert raw scores into decisions based on configurable thresholds.

```python
from dataclasses import dataclass
from enum import Enum

class Decision(Enum):
    AUTO_APPROVE  = "auto_approve"
    AUTO_REMOVE   = "auto_remove"
    HUMAN_REVIEW  = "human_review"

@dataclass
class PolicyThresholds:
    # Format: (auto_remove_threshold, human_review_threshold)
    # auto_remove if score > auto_remove_threshold
    # human_review if score > human_review_threshold
    # auto_approve if score <= human_review_threshold

    hate_speech:   tuple = (0.85, 0.50)
    harassment:    tuple = (0.85, 0.50)
    spam:          tuple = (0.95, 0.70)   # higher threshold (more false positives)
    misinformation:tuple = (0.90, 0.60)
    adult_content: tuple = (0.90, 0.60)
    violence:      tuple = (0.80, 0.45)
    self_harm:     tuple = (0.70, 0.35)   # lower threshold (safety critical)

class PolicyEngine:
    def __init__(self):
        self.thresholds = PolicyThresholds()

    def decide(self, scores: dict, context: dict) -> tuple[Decision, str]:
        """
        Returns (decision, reason).
        context: user history, account age, verification status, etc.
        """
        # Context adjustments
        if context.get("verified_organization"):
            # Give benefit of doubt to verified orgs (news, health authorities)
            score_multiplier = 0.8
        elif context.get("new_account_days", 0) < 7:
            # New accounts: stricter (common spam/abuse vector)
            score_multiplier = 1.2
        else:
            score_multiplier = 1.0

        # Check each category
        for category, score in scores.items():
            adjusted = min(1.0, score * score_multiplier)
            thresholds = getattr(self.thresholds, category, (0.9, 0.6))
            auto_remove_t, review_t = thresholds

            if adjusted >= auto_remove_t:
                return Decision.AUTO_REMOVE, f"{category}:{adjusted:.2f}"
            if adjusted >= review_t:
                return Decision.HUMAN_REVIEW, f"{category}:{adjusted:.2f}"

        return Decision.AUTO_APPROVE, "all_clear"
```

### Dry Run — Policy Decision

```
Post: "You should go kill yourself, you idiot"
Author: new account (3 days old) → multiplier = 1.2

ML scores:
  harassment:  0.94 → adjusted = 0.94 × 1.2 = 1.0 → capped at 1.0
  self_harm:   0.71 → adjusted = 0.71 × 1.2 = 0.85
  hate_speech: 0.12 → adjusted = 0.14

Policy check:
  harassment: 1.0 >= 0.85 (auto_remove_threshold) → AUTO_REMOVE ✓

Decision: AUTO_REMOVE, "harassment:1.00"
Action: post removed immediately, author notified, strike added to account
```

---

## Human Review Queue

```python
from dataclasses import dataclass
from datetime import datetime
import heapq

@dataclass
class ReviewItem:
    post_id:       str
    priority:      float   # higher = review sooner
    scores:        dict
    reason:        str
    submitted_at:  datetime
    context:       dict

    def __lt__(self, other):
        return self.priority > other.priority  # max heap

class ReviewQueue:
    def __init__(self):
        self.queue = []  # priority queue (max heap by priority)

    def enqueue(self, item: ReviewItem):
        heapq.heappush(self.queue, item)

    def compute_priority(self, scores: dict, context: dict) -> float:
        """Higher priority = review sooner."""
        max_score  = max(scores.values())
        category   = max(scores, key=scores.get)

        # Base priority from score
        priority = max_score * 10

        # Boost for high-risk categories
        if category in ("self_harm", "violence"):
            priority *= 2.0

        # Boost for viral content (high impressions = high harm potential)
        impressions = context.get("impressions_so_far", 0)
        if impressions > 10_000:
            priority *= 1.5

        # Boost for appeals (user-reported errors)
        if context.get("is_appeal"):
            priority *= 1.2

        return priority

    def dequeue(self) -> ReviewItem:
        return heapq.heappop(self.queue)
```

---

## Appeals & Feedback Loop

```python
def handle_appeal(post_id: str, appeal_reason: str, moderator_decision: str):
    """
    User appeals auto-removal decision.
    Moderator reviews → outcome fed back for model improvement.
    """
    original = get_moderation_record(post_id)

    if moderator_decision == "overturn":
        # False positive: model was wrong
        # 1. Restore post
        restore_post(post_id)

        # 2. Log training example: this post should NOT have been removed
        log_training_example(
            post_id=post_id,
            text=original["text"],
            label="clean",
            was_prediction=original["scores"],
            notes=appeal_reason,
        )

        # 3. If same content type keeps getting overturned: adjust threshold
        update_threshold_if_needed(original["category"])

    elif moderator_decision == "uphold":
        # True positive: model was correct, user is trying to evade
        log_training_example(post_id, original["text"], label="violation")
```

---

## Enforcement Actions

```python
class EnforcementEngine:
    def enforce(self, decision: Decision, post: dict, author_id: str):
        if decision == Decision.AUTO_APPROVE:
            publish_post(post)
            return

        if decision == Decision.AUTO_REMOVE:
            remove_post(post["id"])
            notify_author(author_id, reason=post["violation_reason"])
            add_strike(author_id)

            # Escalating penalties
            strikes = get_strike_count(author_id)
            if strikes >= 3:
                restrict_account(author_id, days=7)
            if strikes >= 5:
                suspend_account(author_id)
            if strikes >= 10:
                ban_account(author_id)

        elif decision == Decision.HUMAN_REVIEW:
            # Content withheld pending review (not shown publicly)
            withhold_post(post["id"])
            enqueue_for_review(post, author_id)
```

---

## Evaluation

```
Precision = TP / (TP + FP)   — how often a flagged post is actually violating
Recall    = TP / (TP + FN)   — how often we catch actual violations

Trade-off:
  High recall  → catch more violations, but more false positives
  High precision → fewer false positives, but miss more violations

Target operating points:
  Hate speech: precision=0.95, recall=0.80  (high precision — avoid censorship)
  CSAM/CSEM:   precision=0.99, recall=0.99  (both critical — zero tolerance)
  Spam:        precision=0.85, recall=0.90  (tolerate some false positives)
  Self-harm:   precision=0.85, recall=0.95  (safety → higher recall)

Evaluation metrics:
  AUC-ROC:     ranking quality across all thresholds
  F1 per class: balance precision/recall at operating threshold
  Human review rate: what % of posts go to human review (cost driver)
  Overturn rate: how often humans reverse ML decision (model quality signal)
  Time to action: median time from post submission to enforcement
```

---

## Interview Q&A

**Q: How do you handle the precision-recall tradeoff in content moderation?**
A: Different categories have different tradeoffs. For CSAM (child sexual abuse material) and imminent violence threats: maximize recall — a false positive (removing clean content) is far less costly than a false negative (leaving harmful content up). For political speech or satire: maximize precision — false positives cause censorship backlash and trust damage. In practice: use low thresholds (higher recall) + human review buffer for borderline cases. The review queue prioritizes by harm potential × engagement reach so the most dangerous content gets reviewed first.

**Q: How do you prevent the system from over-indexing on false positives for marginalized communities?**
A: Historical problem: moderation models trained on biased data over-flag content from Black, LGBTQ+, or Arabic users because that content appeared in training data labeled as "hate speech" when it was actually legitimate expression. Mitigations: (1) evaluate model performance disaggregated by demographic group — alert if FPR significantly higher for any group; (2) diversify training data labelers — use annotators from affected communities; (3) context window — consider author profile and thread context, not just isolated post text; (4) appeals process — users can appeal, overturn rates per demographic tracked as a fairness metric.

**Q: What happens when a new type of harmful content emerges (e.g., new slang for hate speech)?**
A: The pre-filter blocklist needs immediate update (manual, quick). The ML classifier won't catch it until retrained. Defense-in-depth: (1) user reports — users flagging content is a fast signal; build a report classifier that identifies clusters of similar-looking flagged content; (2) semantic search — new harmful phrases often cluster in embedding space near known violations; run daily nearest-neighbor search from new flagged content; (3) near-weekly model retraining with new labeled examples from human moderators; (4) expert review team — small team monitors emerging trends to update blocklists in hours.

---

## Connections
- **ML system design framework:** `9.system_design/01_ml_system_design_framework.md`
- **Text classification:** `4.nlp/04_applications/01_text_classification.md`
- **Model monitoring (overturn rate as drift signal):** `8.mlops/09_monitoring_end_to_end.md`
- **LLM-as-judge pattern:** `6.llms/09_agents_end_to_end.md`

## Key Takeaway
Content moderation = 3-stage pipeline: fast pre-filter (blocklist + hash match, <10ms) → ML classifiers (text + image models, <100ms) → policy engine (thresholds + context adjustments → AUTO_APPROVE / AUTO_REMOVE / HUMAN_REVIEW). Key design choices: different precision/recall tradeoffs per category (CSAM: maximize recall; political speech: maximize precision); context matters (new accounts get stricter thresholds); human review queue prioritized by harm × reach; feedback loop from moderator decisions → model retraining.

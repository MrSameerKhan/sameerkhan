# 07 Feed Ranking System Design

## Problem Statement

Design a feed ranking system for a social platform. Scale: 10M posts/day, +500ms moderation decision, 99.9% uptime. Requirement: catch harmful content before it reaches users; minimize false positives.

---

## Architecture

```
              OFFLINE (hours/daily)
  User logs + Feature engineering + Train ranking model
  User embeddings + ANN Index (FAISS/ScaNN)
  Post embeddings pre-computed, refresh every few hours

                         ↓

               ONLINE (real-time feed request)
  User request
    ↓
  [Candidate Generation] — ~1000 posts from multiple sources
    ↓
  [Ranking] — LightGBM on 224-dim features → score all candidates
    ↓
  [Reranking + Policy] — diversity caps, freshness boost, ads injection
    ↓
  Return top-25
```

---

## Stage 1 — Candidate Generation

Multiple candidate sources (union all):

```
1. Social graph candidates (connections/follows):
   Posts from people user follows → ~300 candidates
   Weighted by: relationship strength, recency

2. Interest-based candidates (collaborative filtering):
   Users with similar engagement history × their liked posts → ~300 candidates
   Two-tower model: user_emb · post_emb + ANN search

3. Trending content:
   Posts with high engagement velocity in last 2 hours → ~200 candidates
   engagement_rate = (likes + comments×2 + shares×3) / impressions

4. Sponsored content:
   Ads matching user profile → ~100 candidates
   Billed per impression, must be included in final feed

5. Exploration (new creators, new topics):
   ~100 candidates outside user's usual interest graph
   Prevents filter bubble, surfaces new content

Total: ~1000 candidates → pass to ranking
```

```python
import redis
import json
from typing import List

class CandidateGenerator:
    def __init__(self, redis_client, faiss_index, post_store):
        self.redis = redis_client
        self.index = faiss_index   # ANN index of post embeddings
        self.posts = post_store

    def generate(self, user_id: int, n_candidates: int = 1000) -> list[dict]:
        candidates = {}

        # 1. Social graph candidates (from Redis pre-computed following list)
        following = self.redis.smembers(f"following:{user_id}")
        for author_id in list(following)[:50]:   # top 50 followed accounts
            recent_posts = self.redis.zrevrange(
                f"author_posts:{author_id}", 0, 5   # last 6 posts
            )
            for post_id in recent_posts:
                candidates[post_id] = {"source": "social", "author": author_id}

        # 2. Interest-based (ANN on user embedding)
        user_emb = self.get_user_embedding(user_id)
        _, post_indices = self.index.search(user_emb, 300)
        for idx in post_indices[0]:
            post_id = self.index.to_post_id(idx)
            candidates[post_id] = {"source": "interest"}

        # 3. Trending
        trending = self.redis.zrevrange("trending:global", 0, 199)
        for post_id in trending:
            candidates[post_id] = {"source": "trending"}

        # 4. Deduplicate, exclude already seen
        seen = self.redis.smembers(f"seen:{user_id}")
        candidates = {k: v for k, v in candidates.items() if k not in seen}

        return list(candidates.items())[:n_candidates]

    def get_user_embedding(self, user_id: int):
        emb = self.redis.smembers(f"user_emb:{user_id}")
        if emb:
            return json.loads(emb)
        # Fallback: compute from user history
        return self.compute_user_embedding(user_id)
```

---

## Stage 2 — Ranking Model

Score all 1000 candidates with a rich model.

### Feature Engineering

```
User features (pre-computed, cached):
  - Engagement rates by content type (video, image, text)
  - Active hours (when does this user typically engage?)
  - Topic affinity vector (interested in: tech=0.8, sports=0.2, ...)
  - Network density (how many mutual connections with author?)

Post features:
  - Age in hours (freshness)
  - Author follower count, past engagement rate
  - Content type (video, image, link, text)
  - Initial engagement velocity (likes per minute in first hour)
  - Embedding similarity to user's interests

User × Post interaction features (real-time):
  - Author relationship strength (close friend, acquaintance, stranger)
  - Has user engaged with this author recently?
  - Is this in user's top-5 interest topics?
  - Time since user's last session
```

### GBDT Ranker

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

# Training data: user_id, post_id, features, label (1=engaged, 0=not)
# Collected from production logs, 30 days of interactions

def build_ranking_features(user_id: int, post_ids: list[str]) -> np.ndarray:
    """Build feature matrix for ranking model."""
    user_feats = get_user_features(user_id)   # [128,]
    rows = []
    for post_id in post_ids:
        post_feats    = get_post_features(post_id)            # [64,]
        interact_feats = get_interaction_features(user_id, post_id)   # [32,]
        row = np.concatenate([user_feats, post_feats, interact_feats])
        rows.append(row)
    return np.array(rows)   # [n_posts, 224]

# Training
params = {
    "objective":    "lambdarank",   # LTR objective
    "metric":       "ndcg",
    "ndcg_eval_at": [5, 10],
    "num_leaves":   63,
    "learning_rate": 0.05,
    "n_estimators":  500,
    "min_data_in_leaf": 50,
}

model = lgb.LGBMRanker(**params)
model.fit(
    X_train, y_train,
    group=group_train,    # number of posts per query (user)
    eval_set=[(X_val, y_val)],
    eval_group=eval_group,
    callbacks=[lgb.early_stopping(50)],
)

# Inference: score 1000 candidates
def rank_candidates(user_id: int, candidates: list[str]) -> list[tuple]:
    features = build_ranking_features(user_id, candidates)
    scores   = model.predict(features)
    ranked   = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return ranked[:100]   # top-100 → reranking stage
```

---

## Stage 3 — Reranking + Policy

Apply business rules and diversity constraints to top-100.

```python
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class FeedItem:
    post_id:   str
    score:     float
    author_id: str
    topic:     str
    post_type: str   # "video", "image", "text", "link"
    age_hours: float
    is_ad:     bool = False

def rerank_feed(items: list[FeedItem], user_id: int,
                target_size: int = 25) -> list[FeedItem]:
    final_feed    = []
    author_counts = {}
    topic_counts  = {}
    video_count   = 0
    ad_positions  = [3, 8, 15]   # ads always at these positions

    # Inject ads first (guaranteed positions)
    ads     = [i for i in items if i.is_ad]
    organic = [i for i in items if not i.is_ad]

    organic_ptr = 0
    ad_ptr      = 0

    for position in range(target_size):
        if position in ad_positions and ad_ptr < len(ads):
            final_feed.append(ads[ad_ptr])
            ad_ptr += 1
            continue

        # Find next organic item passing all constraints
        while organic_ptr < len(organic):
            item = organic[organic_ptr]
            organic_ptr += 1

            # 1. Diversity: max 3 posts per author
            if author_counts.get(item.author_id, 0) >= 3:
                continue

            # 2. Topic diversity: max 6 posts per topic
            if topic_counts.get(item.topic, 0) >= 6:
                continue

            # 3. Content mix: max 40% videos
            if item.post_type == "video" and video_count >= target_size * 0.4:
                continue

            # 4. Freshness penalty: posts >48 hours old score reduced
            if item.age_hours > 48:
                item.score *= 0.5   # demote stale content

            # 5. Passed all filters → add to feed
            final_feed.append(item)
            author_counts[item.author_id]  = author_counts.get(item.author_id, 0) + 1
            topic_counts[item.topic]       = topic_counts.get(item.topic, 0) + 1
            if item.post_type == "video":
                video_count += 1
            break

    return final_feed
```

---

## Freshness Boost

```python
def apply_freshness_boost(score: float, age_hours: float,
                           engagement_velocity: float) -> float:
    """
    Boost new content with high early engagement.
    """
    age_hours           = age_hours              # how old the post is
    engagement_velocity = (likes + comments) / hours_since_post   # per hour

    # Time decay: exponential half-life of 6 hours
    time_decay = 0.5 ** (age_hours / 6.0)   # score halves every 6 hours

    # Velocity boost: high early engagement = potential viral content
    # Normalized: typical velocity ~10 eng/hr; viral ~500 eng/hr
    velocity_boost = min(2.0, 1.0 + engagement_velocity / 100.0)

    final_score = score * time_decay * velocity_boost
    return final_score

# Dry run:
# Post A: 24 hrs old, velocity=5  → 0.85 × 0.0625 × 1.05 = 0.056
# Post B: 1 hr old,  velocity=50  → 0.70 × 0.891  × 1.50 = 0.935  ← boosted
# Post C: 2 hrs old, velocity=5   → 0.90 × 0.794  × 1.05 = 0.750
# Ranking: B > C > A  (freshness + engagement velocity trumps base score)
```

---

## Online A/B Testing

```python
def get_ranking_variant(user_id: int) -> str:
    """Stable hash-based assignment to experiment arm."""
    import hashlib
    bucket = int(hashlib.md5(f"feed_rank_v2_{user_id}".encode()).hexdigest(), 16) % 100
    if bucket < 50:
        return "control"    # LightGBM v1
    else:
        return "treatment"  # LightGBM v2 (new features)

# Metrics to track per variant:
# - CTR (clicks / impressions)
# - Long clicks (>30s dwell time) + quality signal
# - Session length
# - Return rate (come back within 24h)
# - Negative signals (hide, unfollow, report)
```

---

## Cold Start — New Users

```
New user (0 interactions):
  Step 1: Onboarding — show topic selector (10 topics)
  Step 2: Use selected topics + retrieve trending in those topics
  Step 3: First 50 interactions + build initial user embedding
  Step 4: Standard collaborative filtering kicks in

New user heuristics:
  - Weight trending content heavily (70%) vs personalized (30%)
  - Surface content from high-quality creators (follower count + engagement rate)
  - Diversity boost: show 1 post per topic initially
  - Explore aggressively: 30% exploration budget (vs 10% for established users)
```

---

## Key Metrics

```
Engagement:
  CTR:             clicks / impressions (primary — target >4%)
  Long-click rate: dwell >30s / clicks  (quality engagement)
  Share rate:      shares / impressions (viral indicator)

Session:
  Session length:      minutes per session
  Posts per session:   engagement depth
  Return rate:         users returning next day (D1 retention)

Ecosystem health:
  Content diversity:   fraction of unique authors in top-100 shown
  Creator coverage:    fraction of creators who get at least 100 impressions
  Filter bubble score: topic entropy of user's feed (higher = more diverse)

Guardrails:
  Spam/abuse rate:  reported posts per 1000 impressions (< 0.1%)
  Ad CTR:           click-through on sponsored content (monetization)
  Complaint rate:   hide/unfollow/mute actions (< 2%)
```

---

## System Constraints & Scaling

```
Latency budget: 100ms P95
  Candidate generation: 15ms  (Redis lookups + ANN search)
  Feature computation:  20ms  (feature store reads)
  Ranking inference:    10ms  (LightGBM on 1000 posts)
  Reranking:             5ms  (rule-based, fast)
  Network + serialization: 50ms
  Total:               ~100ms ✓

Throughput:
  100K QPS × 1000 candidates = 100M rankings/second
  LightGBM: ~1M predictions/second per CPU core
  Need: 100 CPU cores + 10 ranking servers with 10 cores each

Caching:
  User embeddings: Redis, refresh every 30 minutes
  Post features:   Redis, TTL=1 hour (post features change as engagement grows)
  Trending posts:  Redis sorted set, refresh every 5 minutes
  Seen posts:      Redis set per user, TTL=7 days
```

---

## Interview Q&A

**Q: How do you balance freshness vs relevance in a feed?**
A: Two mechanisms: (1) Explicit freshness feature — include post age (hours) as a ranking feature; model learns to down-rank stale content automatically; (2) engagement velocity boost — posts with high early engagement (likes per minute in first hour) get a temporary score multiplier. The multiplier decays with time. This surfaces potentially viral content while it's still fresh, but doesn't permanently disadvantage slightly older high-quality content. Finally, the reranking stage applies a hard filter: posts older than 48 hours get a 50% score penalty regardless of ranking model output.

**Q: How do you prevent filter bubbles?**
A: Four mechanisms: (1) exploration candidates — 10% of candidate pool comes from topics/creators outside user's feed history; (2) topic diversity constraints in reranking — max 4 posts per topic per feed load; (3) serendipity metric — track topic entropy of each user's feed, alert if entropy drops below threshold; (4) periodic "break the bubble" sessions — every 10th feed load, boost exploration budget to 30%. Monitor creator coverage (are small creators getting any exposure?) as a health metric.

**Q: Walk me through the ranking pipeline for a single user request.**
A: (1) Candidate generation (~15ms): pull posts from Redis for followed accounts, run ANN search on user embedding in post embedding space; add trending and exploration candidates — total ~1000 posts. (2) Ranking (~30ms): compute 224-dim features for each (user, post) pair from feature store, run LightGBM ranker → scores for all 1000, take top-100. (3) Reranking (~5ms): apply diversity constraints (max 3 per author, max 4 per topic), inject ads at fixed positions, apply freshness boost, return top-25. Total <100ms p95.

---

## Connections

- Recommendation system (retrieval + ranking): `11.system_design/02_recommendation_system.md`
- A/B testing dry run: `11.system_design/02_recommendation_system.md` §A/B Testing
- Feature stores: `10.mlops/04_pipelines_and_infra.md`
- Two-tower retrieval: `11.system_design/02_recommendation_system.md` §Retrieval Stage

---

## Key Takeaway

Feed ranking = 3-stage funnel: candidate generation (~1000 posts from social graph + ANN + trending) → ranking (LightGBM LTR with 224 user+post features) → reranking (diversity caps; freshness boost → top-25). Key tensions: freshness vs relevance (solve with velocity boost + time decay), personalization vs diversity (solve with exploration budget + topic caps), engagement vs quality (solve with long-click as primary metric, not raw CTR).

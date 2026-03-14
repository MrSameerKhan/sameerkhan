# Monitoring & Drift Detection

## Quick Reference
| Drift Type | What Changes | Detection Method |
|-----------|-------------|-----------------|
| Data drift | Input distribution shifts | KL divergence, PSI, KS test |
| Concept drift | P(Y\|X) changes | Monitor prediction accuracy |
| Label drift | Output distribution changes | Track prediction distribution |
| Feature drift | Specific features shift | Per-feature statistics |
| Model degradation | Performance degrades | Ground truth labels + metrics |

**Core principle:** A model that performs well in development will degrade in production — when and by how much is unknowable without monitoring.

---

## Core Concepts

### What to Monitor

```
Layer 1 — Infrastructure metrics (always alert immediately)
  ├── Latency P50, P95, P99
  ├── Throughput (requests/second)
  ├── Error rate (5xx responses)
  ├── GPU memory utilization
  └── CPU/memory usage

Layer 2 — Data quality (alert within hours)
  ├── Missing value rates per feature
  ├── Feature distribution statistics (mean, std, percentiles)
  ├── Out-of-vocabulary token rate (NLP)
  ├── Input length distribution
  └── Data schema violations

Layer 3 — Model behavior (alert within days)
  ├── Prediction distribution (class probabilities)
  ├── Confidence/calibration (mean prediction confidence)
  ├── Prediction entropy
  └── Feature importance shifts (if using SHAP)

Layer 4 — Business metrics (alert within weeks)
  ├── Precision/recall with delayed ground truth
  ├── Business KPIs (approval rate, extraction accuracy)
  └── User feedback signals (thumbs up/down, corrections)
```

---

### Data Drift Detection

**Population Stability Index (PSI):**
```
PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

PSI < 0.1:  no significant drift (green)
PSI 0.1-0.2: moderate drift (yellow — investigate)
PSI > 0.2:  significant drift (red — retrain)
```

```python
import numpy as np
from scipy import stats
import pandas as pd

def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index for continuous features."""
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Avoid division by zero
    expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts / len(actual))

    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

def ks_test_drift(reference: np.ndarray, current: np.ndarray, threshold: float = 0.05):
    """Kolmogorov-Smirnov test for distribution shift."""
    statistic, p_value = stats.ks_2samp(reference, current)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < threshold,
    }

def chi_squared_drift(reference: np.ndarray, current: np.ndarray, threshold: float = 0.05):
    """Chi-squared test for categorical features."""
    # Get all categories
    all_cats = set(reference) | set(current)

    ref_counts = {c: np.sum(reference == c) for c in all_cats}
    cur_counts = {c: np.sum(current == c) for c in all_cats}

    # Expected frequencies (from reference, scaled to current size)
    ref_total = len(reference)
    cur_total = len(current)

    expected = np.array([ref_counts.get(c, 0) / ref_total * cur_total for c in all_cats])
    observed = np.array([cur_counts.get(c, 0) for c in all_cats])

    statistic, p_value = stats.chisquare(observed, f_exp=expected)
    return {"statistic": statistic, "p_value": p_value, "drift_detected": p_value < threshold}
```

---

### Evidently AI (Production Drift Monitoring)

```python
import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset,
    TextOverviewPreset,
)
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset, NoTargetPerformanceTestPreset

# Reference data: training/validation distribution
reference = pd.read_parquet("reference_data.parquet")

# Current production data (last 24h)
current = pd.read_parquet("production_data_24h.parquet")

column_mapping = ColumnMapping(
    target="label",
    prediction="prediction",
    text_features=["input_text"],
    numerical_features=["text_length", "confidence"],
    categorical_features=["doc_type"],
)

# ─── Drift Report (HTML) ──────────────────────────────────────────────────
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    ClassificationPreset(),  # only if ground truth available
])
report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
report.save_html("drift_report.html")

# ─── Test Suite (pass/fail assertions) ───────────────────────────────────
tests = TestSuite(tests=[
    DataStabilityTestPreset(),
    NoTargetPerformanceTestPreset(),
    TestColumnDrift(column_name="text_length"),
    TestShareOfDriftedColumns(lt=0.3),           # <30% of columns drifted
    TestValueRange(column_name="confidence", left=0.0, right=1.0),
])
tests.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
test_results = tests.as_dict()
all_passed = test_results["summary"]["all_passed"]
```

---

### Concept Drift

```
Data drift: input X distribution changes (same model may still be correct)
Concept drift: P(Y|X) changes (correct answer changes for same input)

Example:
  Document classification model trained on pre-2024 documents
  New regulation in 2024 changes what "compliant" means
  → Same document text, different correct label → concept drift

Detection: requires ground truth labels (delayed)
  Monitor: prediction accuracy over rolling window
  Compare: current accuracy vs baseline (validation set accuracy)
  Alert: when drop exceeds threshold (e.g., >5% relative drop)

Types:
  Sudden drift: abrupt change (e.g., new product launch changes queries)
  Gradual drift: slow change over months (e.g., language evolution)
  Seasonal drift: periodic patterns (e.g., COVID queries spike seasonally)
  Recurring drift: pattern returns (e.g., tax season queries)
```

---

### Prediction Monitoring Without Labels

```python
# When ground truth is delayed or expensive, monitor proxy signals

class PredictionMonitor:
    def __init__(self, reference_predictions: np.ndarray, window_size: int = 1000):
        self.reference_dist = reference_predictions
        self.window = []
        self.window_size = window_size
        self.alerts = []

    def add_prediction(self, prediction: float, metadata: dict = None):
        self.window.append(prediction)
        if len(self.window) > self.window_size:
            self.window.pop(0)

        if len(self.window) >= self.window_size:
            self._check_drift()

    def _check_drift(self):
        current = np.array(self.window)

        # 1. Prediction distribution drift (KS test)
        ks_result = ks_test_drift(self.reference_dist, current)
        if ks_result["drift_detected"]:
            self.alerts.append({"type": "prediction_drift", "p_value": ks_result["p_value"]})

        # 2. Confidence calibration monitoring
        mean_confidence = current.max(axis=-1).mean() if len(current.shape) > 1 else current.mean()
        if mean_confidence < 0.6:  # threshold from calibration analysis
            self.alerts.append({"type": "low_confidence", "mean_conf": mean_confidence})

        # 3. Prediction entropy (for classification)
        if len(current.shape) > 1:
            entropy = -np.sum(current * np.log(current + 1e-8), axis=-1).mean()
            if entropy > self.reference_entropy_threshold:
                self.alerts.append({"type": "high_entropy", "entropy": entropy})
```

---

### Retraining Strategy

```
When to retrain:
  1. Scheduled: retrain every N days regardless (simple, predictable)
  2. Triggered: retrain when drift metric exceeds threshold (reactive)
  3. Continuous: online learning — update model weights on each batch (complex)

Retraining data strategies:
  1. Full retrain: use all historical + new data
     ✓ Best for slowly changing distributions
     ✗ Expensive for large datasets

  2. Sliding window: use only last K months of data
     ✓ Adapts quickly to distribution changes
     ✗ Forgets older but still valid patterns

  3. Weighted retrain: weight recent data more
     ✓ Balance between stability and adaptability
     ✗ Need to tune time-decay weighting

  4. Active learning: selectively label high-uncertainty predictions
     ✓ Most label-efficient
     ✗ Requires human labeling infrastructure
```

---

### Monitoring Stack

```python
# ─── Prometheus + Grafana (industry standard) ─────────────────────────────
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total requests', ['endpoint', 'status'])
REQUEST_LATENCY = Histogram('ml_request_latency_seconds', 'Request latency',
                             buckets=[.01, .025, .05, .1, .25, .5, 1, 2.5])
PREDICTION_CONFIDENCE = Histogram('ml_prediction_confidence', 'Prediction confidence',
                                    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
DRIFT_PSI = Gauge('ml_feature_psi', 'PSI for feature drift', ['feature_name'])

start_http_server(8001)  # Prometheus scrapes this

# Instrument predictions
@app.post("/predict")
async def predict(request: PredictRequest):
    start = time.time()
    try:
        result = run_model(request)
        REQUEST_COUNT.labels(endpoint="predict", status="success").inc()
        PREDICTION_CONFIDENCE.observe(result["confidence"])
        return result
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="predict", status="error").inc()
        raise
    finally:
        REQUEST_LATENCY.observe(time.time() - start)

# Daily drift computation job
def compute_and_report_drift():
    for feature in tracked_features:
        psi_score = psi(reference[feature], production_last_24h[feature])
        DRIFT_PSI.labels(feature_name=feature).set(psi_score)
        if psi_score > 0.2:
            send_alert(f"Feature {feature} PSI={psi_score:.3f} exceeds threshold")
```

---

## Gotchas

**Distribution shift ≠ performance degradation:** A feature distribution can shift significantly while model performance stays the same (if the shift is in an unimportant feature). Don't over-alert on drift — correlate drift with performance signals before taking action.

**Delayed ground truth:** For many tasks (fraud detection, document classification with human review), ground truth arrives days to weeks later. Use prediction monitoring as an early warning system; confirm with labeled data when available.

**Reference data selection:** Monitor relative to your training distribution, not your most recent production window. Using recent production as reference masks gradual drift.

**Alert fatigue:** Too many alerts → engineers ignore them. Tier alerts: critical (immediate action), warning (investigate this week), info (log only). Set thresholds based on actual model sensitivity analysis.

---

## Interview Q&A

**Q: What is data drift and how do you detect it in production?**
A: Data drift is when the statistical distribution of input features changes after deployment — the model was trained on one distribution but is being served a different one. Detection: (1) PSI (Population Stability Index) for continuous features — bins the distribution and compares proportions; PSI >0.2 signals significant drift, (2) KS test — compares CDFs of reference vs current; p-value threshold for statistical significance, (3) Chi-squared test for categorical features. In practice: compute these daily on a rolling window, set up automated alerts, and route to a dashboard (Grafana + Prometheus or Evidently).

**Q: What is the difference between data drift and concept drift?**
A: Data drift: the input distribution P(X) changes — different types of queries, different document formats, different user demographics. The model might still be correct if the drift is in irrelevant features. Concept drift: the relationship P(Y|X) changes — the correct answer for the same input changes. Example: a sentiment model trained pre-COVID; "getting sick" had negative sentiment but "getting sick of lockdowns" has different connotations post-COVID. Concept drift requires retraining; data drift might not. Detecting concept drift requires ground truth labels, which is why delayed label collection is critical.

---

## Connections
- **Model Evaluation (ML/fundamentals/04):** Same metrics used in monitoring as in evaluation
- **MLOps Serving (7.mlops/02):** Latency/throughput monitoring from the serving layer
- **MLOps Pipelines (7.mlops/04):** Automated retraining triggered by monitoring alerts

## Key Takeaway
Monitor in four layers: infrastructure (latency, errors), data quality (missing values, schema), model behavior (prediction distribution, confidence), and business metrics (accuracy with labels). PSI > 0.2 = retrain signal. Use Evidently for drift reports, Prometheus+Grafana for real-time dashboards. The hardest part: concept drift requires delayed ground truth — invest in label collection pipelines early.

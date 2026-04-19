# Model Registry — End-to-End

> **Workflow:** Train → Log → Register → Stage → Validate → Deploy → Monitor

---

## What is a Model Registry?

```
Problem: team trains 50 experiments/week. How do you know:
  - Which model is in production right now?
  - What data/code produced it?
  - Who approved it for deployment?
  - How do you roll it back if it breaks?

Model Registry = versioned, auditable catalog of models.

Registry stores:
  - Model artifact (weights file)
  - Metadata: metrics, parameters, git commit, training data hash
  - Stage: None → Staging → Production → Archived
  - Transition log: who promoted it, when, why
```

---

## MLflow — Full Workflow

### Step 1: Train and Log Experiment

```python
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score
import numpy as np
import pandas as pd

# Set tracking server (local or remote)
mlflow.set_tracking_uri("http://mlflow-server:5000")  # or "sqlite:///mlflow.db" locally
mlflow.set_experiment("invoice-classifier-v2")

# --- Training run ---
with mlflow.start_run(run_name="gbm-depth5-lr0.05") as run:
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

    # Log hyperparameters
    params = {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "min_samples_leaf": 20,
    }
    mlflow.log_params(params)

    # Log environment
    mlflow.log_param("python_version", "3.11")
    mlflow.log_param("training_data", "invoices_2024_v3.parquet")
    mlflow.log_param("git_commit", "bf8905b")

    # Train
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred       = model.predict(X_val)

    metrics = {
        "val_auc":      roc_auc_score(y_val, y_pred_proba),
        "val_f1":       f1_score(y_val, y_pred),
        "val_accuracy": (y_pred == y_val).mean(),
    }
    mlflow.log_metrics(metrics)
    # Output: val_auc=0.923, val_f1=0.871, val_accuracy=0.912

    # Log model artifact
    signature = mlflow.models.infer_signature(X_train, y_pred_proba)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:3],
        registered_model_name="invoice-classifier",  # auto-register
    )

    # Log additional artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("feature_importance.json")
    mlflow.log_dict({"threshold": 0.42, "cutoff_date": "2024-01-01"}, "config.json")

print(f"Model registered as 'invoice-classifier' version 1")
```

### Step 2: Compare Runs and Pick Best

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs in experiment, sort by AUC
runs = mlflow.search_runs(
    experiment_names=["invoice-classifier-v2"],
    filter_string="metrics.val_auc > 0.90",
    order_by=["metrics.val_auc DESC"],
)

print(runs[["run_id", "params.max_depth", "params.learning_rate",
            "metrics.val_auc", "metrics.val_f1"]].head(5))

# Output:
# run_id              depth  lr    val_auc  val_f1
# bf8905b...          5      0.05  0.923    0.871   ← best
# d3a295a...          4      0.05  0.915    0.862
# fafce0a...          5      0.10  0.908    0.854
# ...

best_run_id = runs.iloc[0]["run_id"]
```

### Step 3: Register and Stage Model

```python
client = MlflowClient()

# Get all versions of the model
versions = client.search_model_versions("name='invoice-classifier'")
for v in versions:
    print(f"Version {v.version}: stage={v.current_stage}, run={v.run_id[:8]}")
# Version 3: stage=Production,  run=abc123
# Version 4: stage=Staging,     run=def456
# Version 5: stage=None,        run=bf8905b  ← just registered

# Promote version 5 to Staging
client.transition_model_version_stage(
    name="invoice-classifier",
    version=5,
    stage="Staging",
    archive_existing_versions=False,  # keep current Staging for comparison
)

# Add description / tags
client.update_model_version(
    name="invoice-classifier",
    version=5,
    description="GBM depth=5, lr=0.05. AUC 0.923 (+0.8% vs v3). Trained on 2024 data."
)
client.set_model_version_tag("invoice-classifier", 5, "validated_by", "sameer")
client.set_model_version_tag("invoice-classifier", 5, "dataset_version", "v3")
```

### Step 4: Staging Validation (CI Gate)

```python
# validation_gate.py — runs in CI pipeline before production promotion
import mlflow.sklearn
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import sys

REQUIRED_AUC      = 0.91   # must beat this threshold
REQUIRED_LATENCY  = 50     # ms p95

def validate_staging_model(model_name: str, version: int):
    # 1. Load model from registry
    model_uri = f"models:/{model_name}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    # 2. Load holdout test set (never seen during training)
    X_test, y_test = load_holdout_data("test_holdout_2024q4.parquet")

    # 3. Functional metrics
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    f1  = f1_score(y_test, y_pred_proba > 0.42)

    print(f"AUC: {auc:.4f} (required: >{REQUIRED_AUC})")
    print(f"F1:  {f1:.4f}")

    if auc < REQUIRED_AUC:
        print(f"FAILED: AUC {auc:.4f} < {REQUIRED_AUC}")
        return False

    # 4. Latency test
    import time
    batch = X_test[:100]
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        model.predict_proba(batch)
        times.append((time.perf_counter() - t0) * 1000)

    p95_latency = np.percentile(times, 95)
    print(f"Latency p95: {p95_latency:.1f}ms (required: <{REQUIRED_LATENCY}ms)")

    if p95_latency > REQUIRED_LATENCY:
        print(f"FAILED: Latency {p95_latency:.1f}ms > {REQUIRED_LATENCY}ms")
        return False

    # 5. Schema validation — does model accept expected input shape?
    try:
        model.predict_proba(X_test[:1])
    except Exception as e:
        print(f"FAILED: Schema error: {e}")
        return False

    print("PASSED all gates ✓")
    return True

if __name__ == "__main__":
    passed = validate_staging_model("invoice-classifier", version=5)
    sys.exit(0 if passed else 1)

# In CI (GitHub Actions / Jenkins):
# python validation_gate.py && mlflow models promote ...
```

**Dry run output:**
```
AUC: 0.9231 (required: >0.91)     ✓
F1:  0.8714
Latency p95: 12.3ms (<50ms)       ✓
Schema: OK                         ✓
PASSED all gates ✓
```

### Step 5: Promote to Production

```python
client = MlflowClient()

# Promote Staging → Production (archives current Production automatically)
client.transition_model_version_stage(
    name="invoice-classifier",
    version=5,
    stage="Production",
    archive_existing_versions=True,  # version 3 → Archived
)

# Log promotion event
client.set_model_version_tag("invoice-classifier", 5, "promoted_at", "2024-04-19")
client.set_model_version_tag("invoice-classifier", 5, "promoted_by", "sameer")
client.set_model_version_tag("invoice-classifier", 5, "promotion_reason",
                              "AUC +0.8%, passed CI gate")

print("Version 5 is now Production.")
```

### Step 6: Load Production Model in Serving

```python
import mlflow.sklearn

# Always load by stage, not version number — decouples serving code from version
model = mlflow.sklearn.load_model("models:/invoice-classifier/Production")

# Or load specific version for canary
model_v5 = mlflow.sklearn.load_model("models:/invoice-classifier/5")

# FastAPI serving
from fastapi import FastAPI
import numpy as np

app = FastAPI()
_model = None

@app.on_event("startup")
def load_model():
    global _model
    _model = mlflow.sklearn.load_model("models:/invoice-classifier/Production")

@app.post("/predict")
def predict(features: list[float]):
    x = np.array(features).reshape(1, -1)
    prob = _model.predict_proba(x)[0][1]
    return {"probability": float(prob), "label": int(prob > 0.42)}
```

### Step 7: Rollback

```python
# Production model degraded — roll back to previous version
client = MlflowClient()

# Version 3 was the previous production model (now Archived)
client.transition_model_version_stage(
    name="invoice-classifier",
    version=3,           # previous good version
    stage="Production",
    archive_existing_versions=True,  # archives v5
)

print("Rolled back to version 3.")
# Serving layer picks it up automatically (loads by stage "Production")
```

---

## Complete State Machine

```
            register
None ──────────────────► Staging
                             │
                    CI gate passes?
                    ┌────────┴────────┐
                   Yes               No
                    │                 │
                    ▼                 ▼
               Production        Archived (failed)
                    │
            performance degrades?
                    │
                    ▼
               Archived
                    │
              rollback needed?
                    │
                    ▼
               Production (previous version)
```

---

## Pytorch Model Logging

```python
import mlflow.pytorch
import torch
import torch.nn as nn

class InvoiceClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

with mlflow.start_run():
    mlflow.log_params({"hidden_dim": 256, "dropout": 0.2, "lr": 1e-3})

    # ... training loop ...

    mlflow.log_metrics({"val_auc": 0.931, "val_f1": 0.882})

    # Log PyTorch model
    mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path="model",
        registered_model_name="invoice-classifier-torch",
        # Optional: wrap with custom inference logic
        conda_env={"dependencies": ["torch==2.2.0", "numpy"]},
    )
```

---

## Key Numbers

| Metric | Typical Value |
|--------|--------------|
| Staging validation AUC threshold | current prod − 0.005 (no regression) |
| Latency p95 budget | <50ms (online), <500ms (batch) |
| Time to promote staging → production | 1–4 hours (CI + manual review) |
| Registry artifact size | 1MB (sklearn) to 14GB (LLM) |
| MLflow UI port | 5000 (default) |

---

## Gotchas

**Never load by version number in serving.** `models:/name/Production` decouples serving from registry ops. If you hardcode version 5, rollback to version 3 doesn't take effect automatically.

**Artifact path matters.** `mlflow.sklearn.log_model(..., artifact_path="model")` sets the path within the run. Loading: `models:/name/Production` resolves to the artifact. Inconsistent paths across runs → confusion when loading.

**Schema drift.** Model trained on 128 features; production sends 130 features. Log `input_example` and `signature` when logging — MLflow will validate at serving time. Add a schema check in your CI gate.

**Staging ≠ shadow mode.** Staging just means "ready for validation." Shadow mode = running production traffic through new model without serving its predictions (to collect live performance data). Implement shadow mode separately in the serving layer.

---

## Interview Q&A

**Q: What is a model registry and why do you need one?**
A: A model registry is a versioned catalog of ML model artifacts with associated metadata (metrics, parameters, training data, git commit). You need it because: (1) reproducibility — know exactly what code/data produced the production model; (2) governance — who approved it, when, why; (3) rollback — immediately restore the previous version if production degrades; (4) team coordination — multiple engineers can train models without overwriting each other. Without a registry, you end up with model.pkl files on someone's laptop and no way to audit or roll back.

**Q: What checks should a staging validation gate include?**
A: (1) Functional metrics on holdout test set — must meet or beat a threshold (e.g., AUC > 0.91); (2) regression check — new model must not be worse than current production by more than a margin; (3) latency test — p95 inference time under budget; (4) schema validation — accepts expected input shape and types; (5) edge case tests — known hard examples should pass; (6) data drift check — training distribution should match expected serving distribution.

**Q: How do you roll back a model in production?**
A: In MLflow, transition the previous version (now Archived) back to Production stage. The serving layer loads by `models:/name/Production`, so it picks up the rollback automatically — no code change needed. The key is that serving code must reference stage, not version number. With a 60-second model reload interval, rollback takes effect within 1 minute.

---

## Connections

- **Experiment tracking:** `8.mlops/01_experiment_tracking.md` — MLflow runs, metrics, artifacts
- **Serving and inference:** `8.mlops/02_serving_and_inference.md` — loading model, FastAPI, vLLM
- **Monitoring:** `8.mlops/09_monitoring_end_to_end.md` — detecting when production model degrades
- **CI/CD pipelines:** `8.mlops/04_pipelines_and_infra.md` — how validation gate integrates with GitHub Actions

## Key Takeaway

Model registry = train → log (metrics + artifact) → register → stage (Staging) → validate (CI gate: AUC + latency + schema) → promote (Production) → monitor → rollback if needed. Always load by stage (`Production`) not version number — that's what enables zero-code-change rollbacks. MLflow is the standard open-source tool; SageMaker Model Registry and Vertex AI Model Registry are cloud equivalents.

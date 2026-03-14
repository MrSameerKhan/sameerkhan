# ML Pipelines & Infrastructure

## Quick Reference
| Tool | Category | Best For |
|------|----------|----------|
| Airflow | Orchestration | Complex DAG dependencies, mature ecosystem |
| Prefect | Orchestration | Python-native, simpler than Airflow |
| Kubeflow Pipelines | ML-specific orchestration | Kubernetes-native ML workflows |
| Feature Store (Feast) | Feature management | Consistent train/serve features |
| GitHub Actions / Jenkins | CI/CD | Automated testing + deployment |
| Docker + Kubernetes | Container orchestration | Scalable model serving |
| SageMaker / Vertex AI | Managed ML platform | Cloud-native end-to-end |

---

## Core Concepts

### ML Pipeline Components

```
                         ┌──────────────────────────────────────────┐
                         │         DATA PIPELINE                     │
                         │  Raw data → Clean → Feature engineering  │
                         │  → Feature store → Train/serve split     │
                         └──────────────────────────────────────────┘
                                              ↓
                         ┌──────────────────────────────────────────┐
                         │         TRAINING PIPELINE                 │
                         │  Data load → Train → Evaluate → Register │
                         └──────────────────────────────────────────┘
                                              ↓
                         ┌──────────────────────────────────────────┐
                         │         DEPLOYMENT PIPELINE               │
                         │  Test → Stage → Canary → Production      │
                         └──────────────────────────────────────────┘
                                              ↓
                         ┌──────────────────────────────────────────┐
                         │         MONITORING PIPELINE               │
                         │  Metrics collection → Drift → Retrain    │
                         └──────────────────────────────────────────┘
```

---

### Training Pipeline with Prefect

```python
from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
import mlflow
import pandas as pd

@task(retries=3, retry_delay_seconds=60)
def load_data(data_path: str) -> pd.DataFrame:
    """Load training data with automatic retry on failure."""
    df = pd.read_parquet(data_path)
    assert len(df) > 0, "Empty dataset"
    return df

@task
def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Data quality checks before training."""
    # Check schema
    required_cols = ['text', 'label', 'doc_type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check label distribution
    label_dist = df['label'].value_counts(normalize=True)
    if label_dist.max() > 0.95:
        raise ValueError(f"Severely imbalanced labels: {label_dist.to_dict()}")

    # Drop corrupted rows
    df = df.dropna(subset=['text', 'label'])
    return df

@task
def train_model(df: pd.DataFrame, config: dict):
    """Train model and log to MLflow."""
    with mlflow.start_run():
        mlflow.log_params(config)
        model, metrics = run_training(df, config)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "model",
                                  registered_model_name="invoice-extractor")
        return metrics["val_f1"], mlflow.active_run().info.run_id

@task
def evaluate_and_gate(val_f1: float, run_id: str, threshold: float = 0.85):
    """Quality gate: only promote models above threshold."""
    if val_f1 < threshold:
        raise ValueError(f"Model quality gate failed: val_f1={val_f1:.3f} < {threshold}")

    # Promote to staging
    client = mlflow.MlflowClient()
    model_version = client.get_run(run_id).data.tags.get("mlflow.log-model.history")
    client.transition_model_version_stage(
        name="invoice-extractor",
        version=model_version,
        stage="Staging",
    )
    return run_id

@task
def run_integration_tests(run_id: str):
    """Run integration tests against the staged model."""
    model = mlflow.pytorch.load_model(f"models:/invoice-extractor/Staging")
    test_cases = load_test_cases("tests/integration/invoice_test_cases.json")

    results = []
    for case in test_cases:
        prediction = model.predict(case["input"])
        results.append(prediction == case["expected"])

    pass_rate = sum(results) / len(results)
    if pass_rate < 0.95:
        raise ValueError(f"Integration tests failed: pass_rate={pass_rate:.2%}")

@flow(name="invoice-extractor-training", log_prints=True)
def training_pipeline(
    data_path: str = "s3://data-lake/invoice/train_v3.parquet",
    config: dict = None,
):
    config = config or {
        "model_name": "microsoft/deberta-v3-base",
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 10,
    }

    df = load_data(data_path)
    df = validate_data(df)
    val_f1, run_id = train_model(df, config)
    run_id = evaluate_and_gate(val_f1, run_id)
    run_integration_tests(run_id)
    print(f"Training complete! Model staged. val_f1={val_f1:.3f}")

# Schedule: retrain every Sunday at 2 AM
deployment = Deployment.build_from_flow(
    flow=training_pipeline,
    name="weekly-retrain",
    schedule=CronSchedule(cron="0 2 * * 0"),
)
deployment.apply()
```

---

### Feature Store (Feast)

```python
# Feature stores solve the train-serve skew problem:
# "I computed feature X differently in training vs production"

from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import String, Float32, Int64
from datetime import timedelta

# Define entities and feature views
document_entity = Entity(
    name="document_id",
    description="Document identifier"
)

document_features = FeatureView(
    name="document_stats",
    entities=[document_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="text_length", dtype=Int64),
        Field(name="word_count", dtype=Int64),
        Field(name="avg_confidence", dtype=Float32),
        Field(name="doc_type", dtype=String),
    ],
    source=FileSource(
        path="s3://feature-store/document_stats/",
        timestamp_field="event_timestamp",
    ),
)

# Training: get historical features (point-in-time correct)
store = FeatureStore(repo_path="./feature_repo")

entity_df = pd.DataFrame({
    "document_id": ["doc_001", "doc_002", "doc_003"],
    "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    "label": [0, 1, 0],
})

# Point-in-time join: gets feature values AS OF the event_timestamp
# Prevents future data leakage in training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["document_stats:text_length", "document_stats:avg_confidence"],
).to_df()

# Serving: get online features (from fast cache, e.g., Redis)
online_features = store.get_online_features(
    features=["document_stats:text_length", "document_stats:avg_confidence"],
    entity_rows=[{"document_id": "doc_new"}],
).to_dict()
```

---

### Docker + Kubernetes

```dockerfile
# Dockerfile for ML model serving
FROM python:3.11-slim as base

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY models/ ./models/
COPY src/ ./src/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: invoice-extractor
  labels:
    app: invoice-extractor
    version: "1.2.3"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: invoice-extractor
  template:
    metadata:
      labels:
        app: invoice-extractor
    spec:
      containers:
      - name: invoice-extractor
        image: gcr.io/my-project/invoice-extractor:1.2.3
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
            nvidia.com/gpu: 1      # GPU request
          limits:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/models/invoice-extractor-v3"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: invoice-extractor-svc
spec:
  selector:
    app: invoice-extractor
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: invoice-extractor-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: invoice-extractor
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

### CI/CD for ML (GitHub Actions)

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths: ['src/**', 'configs/**', 'tests/**']
  schedule:
    - cron: '0 2 * * 0'  # Weekly retrain on Sunday 2 AM

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Lint
        run: ruff check src/ && mypy src/
      - name: Unit tests
        run: pytest tests/unit/ -v --cov=src
      - name: Integration tests
        run: pytest tests/integration/ -v -m "not slow"

  train-and-evaluate:
    needs: test
    runs-on: [self-hosted, gpu]  # requires GPU runner
    steps:
      - uses: actions/checkout@v4
      - name: Train model
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_URI }}
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_KEY }}
        run: |
          python -m src.train \
            --config configs/train_config.yaml \
            --output-dir /tmp/model
      - name: Evaluate and gate
        run: |
          python -m src.evaluate \
            --model-path /tmp/model \
            --test-data s3://data/test_set.parquet \
            --min-f1 0.85
      - name: Upload model artifact
        if: success()
        run: |
          aws s3 cp /tmp/model/ s3://models/invoice-extractor/${{ github.sha }}/ --recursive

  deploy-staging:
    needs: train-and-evaluate
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/invoice-extractor \
            invoice-extractor=gcr.io/my-project/invoice-extractor:${{ github.sha }} \
            --namespace staging
      - name: Run smoke tests
        run: python tests/smoke/test_staging.py

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production  # requires manual approval
    steps:
      - name: Canary deployment (10% traffic)
        run: |
          kubectl set image deployment/invoice-extractor-canary \
            invoice-extractor=gcr.io/my-project/invoice-extractor:${{ github.sha }}
      - name: Monitor canary for 30 minutes
        run: python scripts/monitor_canary.py --duration 1800 --error-threshold 0.01
      - name: Full rollout
        run: |
          kubectl set image deployment/invoice-extractor \
            invoice-extractor=gcr.io/my-project/invoice-extractor:${{ github.sha }}
```

---

### Distributed Training

```python
# PyTorch Distributed Data Parallel (DDP) — standard for multi-GPU training
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def train_ddp(rank: int, world_size: int, model, dataset, config):
    """Called per GPU process."""
    # Setup process group
    dist.init_process_group(
        backend="nccl",  # NCCL for GPU communication
        rank=rank,
        world_size=world_size,
    )

    # Move model to this process's GPU
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    # Distributed sampler: each GPU gets non-overlapping subset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=config.batch_per_gpu, sampler=sampler)

    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        sampler.set_epoch(epoch)  # ensures shuffling differs per epoch
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Log only from rank 0 (one process)
        if rank == 0:
            print(f"Epoch {epoch}: loss={loss.item():.4f}")

    dist.destroy_process_group()

# Launch
world_size = torch.cuda.device_count()
mp.spawn(train_ddp, args=(world_size, model, dataset, config), nprocs=world_size)

# HuggingFace Accelerate (simpler DDP)
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, loader, scheduler = accelerator.prepare(
    model, optimizer, loader, scheduler
)

for batch in loader:
    with accelerator.accumulate(model):  # handles gradient accumulation across GPUs
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## Gotchas

**Train-serve skew:** The most common production ML bug. Feature computation in training differs from serving (different library versions, different time windows, different null handling). Feature store solves this. Always log features actually used at inference time for debugging.

**Data pipeline failures are silent:** A data pipeline that fails midway often produces a partial output file that looks valid but isn't. Always add row count checks and data hash verification after pipeline steps.

**GPU memory in Kubernetes:** GPUs are not multiplexed by default — one container gets the whole GPU. For smaller models, use NVIDIA MPS (Multi-Process Service) to share GPUs, or time-sharing with NVIDIA GPU Operator. Always specify `nvidia.com/gpu: 1` in resource limits or your pod might not be scheduled on a GPU node.

**Canary rollouts for ML:** Traffic splitting for ML is more complex than web services — you need to route the same user/document consistently to the same model version (session affinity) to avoid inconsistent predictions in multi-step workflows.

---

## Interview Q&A

**Q: Explain the training-serving skew problem and how to prevent it.**
A: Training-serving skew occurs when features are computed differently during training vs production inference. Example: training uses `fillna(df['salary'].median())` but serving uses `fillna(0)` for missing salary. The model learns on the training distribution but is served a different one. Prevention: (1) Feature store with shared computation logic — the same code runs for historical (training) and online (serving) features; (2) Log features actually sent to model at inference time and periodically compare to training distribution; (3) Shadow mode testing — run new model alongside old one with same inputs, compare outputs; (4) End-to-end integration tests that exercise the full inference path.

**Q: What is a feature store and when do you need one?**
A: A feature store is a centralized system for computing, storing, and serving ML features. It solves two problems: (1) Consistency — same feature computation logic for training and serving (no skew), (2) Reuse — feature team computes "customer purchase frequency last 30 days" once; multiple models use it. You need one when: multiple models use the same features, training-serving skew is a recurring bug, or you need point-in-time correctness in training (preventing leakage from using future data). Simple deployments often don't need a full feature store — a shared preprocessing library with versioning can suffice.

**Q: How would you set up a CI/CD pipeline for ML?**
A: Key stages: (1) Test — unit tests for data processing logic, model loading, API endpoints; (2) Train — trigger training on code changes or schedule; (3) Evaluate + quality gate — automated evaluation on held-out test set; reject if below threshold (e.g., F1 < 0.85); (4) Deploy to staging — run integration tests and smoke tests; (5) Canary — route 10% of traffic to new model, monitor error rate and key metrics for 30 minutes; (6) Full rollout — if canary passes, update all pods. Critical: automated rollback trigger if error rate spikes, and blue-green deployment to avoid downtime. The hardest part: the quality gate threshold — too strict blocks all deployments, too lenient lets bad models through.

---

## Connections
- **Experiment Tracking (7.mlops/01):** Training pipeline outputs go to MLflow/W&B
- **Serving (7.mlops/02):** CI/CD deploys the inference stack
- **Monitoring (7.mlops/03):** Monitoring triggers retraining pipeline
- **System Design (8.system_design):** ML pipelines are components in larger system designs

## Key Takeaway
MLOps = automating the ML lifecycle: data → train → evaluate → deploy → monitor → retrain. Prefect/Airflow for orchestration. Feature store for train-serve consistency. Docker+K8s for reproducible deployment. GitHub Actions for CI/CD with quality gates. The critical insight: automation must include quality gates — automated deployment without automated evaluation gates is dangerous. Monitor in production → trigger retraining → close the loop.

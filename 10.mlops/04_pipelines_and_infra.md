# 04 — ML Pipelines & Infrastructure

> MLOps = automating the ML lifecycle: data → train → evaluate → deploy → monitor → retrain.

---

## Quick Reference

| Tool | Purpose | Key Feature |
|---|---|---|
| Airflow | Workflow orchestration | DAGs, rich ecosystem |
| Prefect | Modern orchestration | Python-native, easier local dev |
| Kubeflow Pipelines | K8s-native ML pipelines | GPU scheduling, artifact tracking |
| Feature Store (Feast) | Train-serve consistency | Point-in-time correct features |
| GitHub Actions + Jenkins | CI/CD | Automated test + deploy |
| Docker + Kubernetes | Containerized serving | Autoscaling, GPU support |
| SageMaker + Vertex AI | Managed MLOps | End-to-end cloud ML platform |

---

## Core Concepts

### ML Pipeline Components

```
DATA → TRAINING → DEPLOYMENT → MONITORING

Step 1 — Data:
  ├── Ingest (S3, databases, APIs)
  ├── Validate (schema, distribution checks)
  └── Feature engineering (feature store)

Step 2 — Training:
  ├── Experiment tracking (MLflow/W&B)
  ├── Hyperparameter tuning
  └── Model evaluation + quality gate

Step 3 — Deployment:
  ├── Model registry (staging → production)
  ├── Serving (FastAPI, vLLM, Triton)
  └── A/B testing / canary rollout

Step 4 — Monitoring:
  ├── Infrastructure (latency, errors)
  ├── Data drift (PSI, KS test)
  └── Trigger retraining when needed
```

---

## Training Pipeline with Prefect

```python
from prefect import task, flow
from prefect.schedules import CronSchedule
from prefect.deployments import Deployment
import mlflow

@task(retries=3, retry_delay_seconds=60)
def load_data(data_path: str):
    import pandas as pd
    df = pd.read_parquet(data_path)
    return df

@task
def validate_data(df):
    # Schema checks
    required_cols = ["text", "label", "doc_type"]
    assert all(c in df.columns for c in required_cols), "Missing required columns"

    # Label distribution check
    label_counts = df["label"].value_counts(normalize=True)
    assert label_counts.max() < 0.9, "Label imbalance too high"

    # Drop corrupted rows
    df = df.dropna(subset=required_cols)
    return df

@task(log_prints=True)
def train_model(df, config: dict):
    with mlflow.start_run():
        mlflow.log_params(config)

        model = train(df, config)

        val_metrics = evaluate(model, val_data)
        mlflow.log_metrics(val_metrics)

        # Register model
        mlflow.pytorch.log_model(
            model, "model",
            registered_model_name="invoice-extractor"
        )

        return val_metrics

@task
def evaluate_and_gate(metrics: dict, min_f1: float = 0.85):
    if metrics["val_f1"] < min_f1:
        raise ValueError(f"Model F1 {metrics['val_f1']:.3f} below threshold {min_f1}")
    return True

@task
def promote_to_staging(model_name: str, version: int):
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
    )

@flow(name="training-pipeline")
def training_pipeline(data_path: str, config: dict):
    df = load_data(data_path)
    df = validate_data(df)
    metrics = train_model(df, config)
    evaluate_and_gate(metrics)
    promote_to_staging("invoice-extractor", version=1)

# Schedule
deployment = Deployment.build_from_flow(
    flow=training_pipeline,
    name="weekly-retrain",
    schedule=CronSchedule(cron="0 2 * * 0"),  # Sunday 2am
    parameters={"data_path": "s3://bucket/data/", "config": {...}},
)
deployment.apply()
```

---

## Feature Store (Feast)

```python
from feast import FeatureStore, Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String
from datetime import timedelta

# Define feature store
store = FeatureStore(repo_path="feature_repo/")

# Entity = what we're computing features for
document = Entity(name="document_id", join_keys=["document_id"])

# Feature view = group of related features
doc_features_source = FileSource(
    path="s3://bucket/features/doc_features.parquet",
    timestamp_field="event_timestamp",
)

doc_feature_view = FeatureView(
    name="document_features",
    entities=[document],
    ttl=timedelta(days=30),
    schema=[
        Field(name="text_length", dtype=Int64),
        Field(name="page_count", dtype=Int64),
        Field(name="doc_type", dtype=String),
        Field(name="confidence_score", dtype=Float64),
    ],
    online=True,
    source=doc_features_source,
)

# — Point-in-time correct training data ——————
# Training: join features as of the time each label was generated
# (prevents leakage from future feature values)
entity_df = pd.DataFrame({
    "document_id": ["doc1", "doc2", "doc3"],
    "event_timestamp": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    "label": [0, 1, 0],
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["document_features:text_length", "document_features:doc_type"],
).to_df()

# — Online features for serving ——————————————
# Real-time: fast lookup from Redis/DynamoDB
online_features = store.get_online_features(
    features=["document_features:text_length", "document_features:doc_type"],
    entity_rows=[{"document_id": "doc_new"}],
).to_dict()
```

---

## Docker + Kubernetes

```dockerfile
# Dockerfile — reproducible serving environment
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and source
COPY models/ ./models/
COPY src/ ./src/

# Security: run as non-root
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: invoice-extractor
  namespace: ml-serving
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
      - name: api
        image: gcr.io/my-project/invoice-extractor:${GIT_SHA}
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
        env:
        - name: MODEL_PATH
          value: "/app/models/model.onnx"
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
          initialDelaySeconds: 10
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

## CI/CD for ML with GitHub Actions

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main]
    paths:
      - "src/**"
      - "configs/**"
      - "tests/**"
  schedule:
    - cron: "0 2 * * 0"  # Weekly retrain

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/unit/ -v
      - name: Run data processing tests
        run: pytest tests/data/ -v

  train-and-evaluate:
    needs: test
    runs-on: self-hosted  # GPU runner
    steps:
      - name: Train model
        run: |
          python train.py \
            --config configs/train_config.yaml \
            --mlflow-uri ${{ secrets.MLFLOW_URI }}
      - name: Evaluate and gate
        run: |
          python evaluate.py \
            --min-f1 0.85 \
            --mlflow-uri ${{ secrets.MLFLOW_URI }}
      - name: Upload model artifact to S3
        run: |
          aws s3 cp models/ s3://bucket/models/${{ github.sha }}/ --recursive

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

## Distributed Training

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

    # DistributedSampler: each GPU gets non-overlapping subset
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=config.batch_size_per_gpu, sampler=sampler)

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

**Train-serve skew:** The most common production ML bug. Feature computation in training differs from serving (different library versions, different time windows, different null handling). Feature store solves this. Always log feature values at inference time and add data verification after pipeline steps.

**GPU memory in Kubernetes:** GPUs are not multiplexed by default — one container gets the whole GPU. For smaller models, use MPS (Multi-Process Service) to share GPUs, or time-sharing with NVIDIA GPU Operator. Always specify `nvidia.com/gpu: 1` in resource limits or the pod might not be scheduled on a GPU node.

**Canary rollouts for ML:** Traffic splitting for ML is more complex than web services — you need to route the same user/document consistently to the same model version (session affinity) to avoid inconsistent predictions in multi-step workflows.

---

## Interview Q&A

**Q: Explain the training-serving skew problem and how to prevent it.**

Training-serving skew occurs when features are computed differently during training vs production inference. Example: training uses `fillna(df['salary'].median())` but serving uses `fillna(0)` for missing salary. The model learns on the training distribution but is being served a different one. Prevention: (1) Feature store with shared computation logic — the same code runs for historical (training) and online (serving) features; (2) Log feature values actually sent to model at inference time and periodically compare to training distribution; (3) Shadow mode testing — run new model alongside old model with same inputs, compare outputs; (4) End-to-end integration tests that exercise the full inference path.

**Q: What is a feature store and when do you need one?**

A feature store is a centralized system for computing, storing, and serving ML features. It solves two problems: (1) Consistency — same feature computation logic for training and serving (no skew); (2) Reuse — feature team computes "customer purchase frequency last 30 days" once; multiple models use it. You need one when: multiple models use the same features; training-serving is a recurring bug; or you need point-in-time correctness in training (preventing leakage from future feature data). Simple deployments often don't need a full feature store — a shared preprocessing library with versioning can suffice.

**Q: How would you set up a CI/CD pipeline for ML?**

Key stages: (1) Test — unit tests for data processing logic, model loading, API endpoints; (2) Train — trigger training on code changes or schedule; (3) Evaluate + quality gate — automated evaluation on held-out test set; reject if below threshold (e.g., F1 < 0.85); (4) Deploy to staging — run integration tests and smoke tests; (5) Canary — route 10% of traffic to new model, monitor error rate and key metrics for 30 minutes; (6) Full rollout — if canary passes, update all pods. Critical: automated rollback trigger if error rate spikes, and blue-green deployment to avoid downtime. The hardest part: the quality gate threshold — too strict blocks all deployments, too lenient lets bad models through.

---

## Connections

- Experiment Tracking (`8.mlops/01`): Training pipeline outputs go to MLflow/W&B
- Serving (`8.mlops/02`): CI/CD deploys the inference stack
- Monitoring (`8.mlops/03`): Monitoring triggers retraining pipeline
- System Design (`10.system_design`): ML pipelines are components in larger system designs

## Key Takeaway

```
MLOps = automating the ML lifecycle: data → train → evaluate → deploy → monitor → retrain.
Prefect/Airflow for orchestration. Feature store for train-serve consistency.
Docker+K8s for reproducible deployment. GitHub Actions for CI/CD with quality gates.
The critical insight: automation must include quality gates — automated deployment
without automated evaluation gates is dangerous. Monitor in production →
trigger retraining → close the loop.
```

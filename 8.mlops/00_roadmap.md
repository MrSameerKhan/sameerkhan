# MLOps — Roadmap & Navigation Guide

---

## Folder Map

```
8.mlops/
├── 00_roadmap.md                          ← you are here
├── 01_experiment_tracking.md              ← MLflow runs, metrics, artifacts
├── 02_serving_and_inference.md            ← FastAPI, vLLM, batching, latency
├── 03_monitoring_and_drift.md             ← KS test, PSI, concept drift theory
├── 04_pipelines_and_infra.md              ← Airflow, Kubeflow, CI/CD
├── 05_on_premise_ml_deployment.md         ← on-prem GPU clusters, Kubernetes
├── 06_multi_gpu_multi_server_training.md  ← DDP, FSDP, tensor/pipeline parallel
├── 07_aws_for_ml.md                       ← SageMaker, S3, ECR, Lambda
├── 08_model_registry_end_to_end.md        ← train → register → stage → deploy
├── 09_monitoring_end_to_end.md            ← drift detection → alert → retrain
└── 10_serving_optimization_end_to_end.md  ← PyTorch → ONNX → INT8 → benchmark
```

---

## Reading Order

### Interview prep (1 hour)
| Order | File | Key concept |
|-------|------|-------------|
| 1 | `08_model_registry_end_to_end.md` | Full lifecycle: train → deploy → rollback |
| 2 | `09_monitoring_end_to_end.md` | KS test, PSI, retrain triggers |
| 3 | `10_serving_optimization_end_to_end.md` | ONNX INT8, vLLM, PagedAttention |

### MLOps system design (2 hours)
Read 01 → 04 for reference theory, then 08 → 10 for end-to-end traces.

---

## MLOps Lifecycle

```
Data → [Experiment Tracking] → [Model Registry] → [Serving] → [Monitoring] → retrain
         (01)                    (08)               (02,10)      (03,09)
                                   ↑
                            [Pipelines / CI-CD]
                                  (04)
```

---

## Key Numbers

| Metric | Typical value |
|--------|--------------|
| Staging validation AUC threshold | current prod − 0.005 |
| Drift alert: PSI warning | > 0.10 |
| Drift alert: PSI critical / retrain | > 0.20 |
| ONNX RT speedup vs PyTorch CPU | 3–4× |
| INT8 quantization speedup | 5–7× |
| INT8 accuracy drop (typical) | < 0.1% AUC |
| vLLM throughput gain vs HF | 3–5× |
| PagedAttention memory savings | 3–4× more concurrent requests |
| Replay buffer size (DQN) | 100K–1M transitions |

---

## Common Interview Topics

1. **Model registry** — stages (None → Staging → Production → Archived), rollback
2. **Data drift vs concept drift** — how to detect each, KS test + PSI
3. **ONNX export** — why it's faster, dynamic axes, opset version
4. **Quantization** — INT8 dynamic vs static, calibration, accuracy tradeoff
5. **vLLM / PagedAttention** — why LLM serving is memory-bound, pages vs contiguous
6. **Continuous batching** — why it improves GPU utilization vs static batching
7. **Retraining triggers** — performance drop, PSI threshold, schedule, data volume
8. **CI/CD for ML** — staging gate (AUC + latency + schema), promote vs reject

---

## Connections

| Topic | Cross-reference |
|-------|----------------|
| RL for RLHF training | `2.deep learning/02_architectures/07_reinforcement_learning.md` |
| Serving optimization (vLLM details) | `6.llms/` + `8.mlops/02_serving_and_inference.md` |
| A/B testing for model rollout | `9.system_design/02_recommendation_system.md` |
| Model monitoring in production | `9.system_design/` (system-level view) |

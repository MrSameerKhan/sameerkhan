# 8. MLOps

> The engineering layer between research and production. How to ship ML models that actually work at scale — reliably, cheaply, and observably.

---

## Reading Order

| File | What it covers |
|------|----------------|
| `01_experiment_tracking.md` | MLflow runs, metrics, artifacts, experiment comparison |
| `02_serving_and_inference.md` | FastAPI serving, vLLM, batch vs online inference |
| `03_feature_store.md` | Feast feature views, point-in-time joins, online/offline stores |
| `04_pipelines_and_infra.md` | Prefect flows, Kubernetes, CI/CD for ML, GitHub Actions |
| `05_on_premise_ml_deployment.md` | Full on-prem stack: NVIDIA/CUDA, FastAPI, Nginx, systemd, blue-green |
| `06_multi_gpu_multi_server_training.md` | DDP, FSDP, DeepSpeed ZeRO, multi-node torchrun |
| `07_aws_for_ml.md` | S3, EC2 GPU, SageMaker, Lambda, ECS, Step Functions, CloudWatch |
| `08_model_registry_end_to_end.md` | MLflow registry: train → stage → validate → promote → rollback |
| `09_monitoring_end_to_end.md` | Data/concept/prediction drift, PSI, KS test, alerting, retraining triggers |
| `10_serving_optimization.md` | ONNX export, INT8 quantization, ORT, vLLM/PagedAttention, AWQ |
| `11_llm_observability.md` | LangSmith, LangFuse, Phoenix, Helicone; traces, cost, PII |
| `12_llm_cost_tracking.md` | Token cost logging, model routing, caching, budget controls |
| `13_production_rag_ops.md` | Incremental indexing, semantic cache, embedding versioning, vector DB drift |

---

## Code Files

| File | Run? |
|------|------|
| `code_practice/04_llms/17_vllm_serve/` | run |
| `code_practice/04_llms/11_fastapi_serve/` | docs only |
| `code_practice/04_llms/16_observability/` | run |
| `code_practice/04_llms/13_eval_harness/` | run |
| `code_practice/05_rag/10_production_rag/` | docs only |

---

## MIT Topics Missing Here

- Kubernetes internals (covered lightly — operators, CRDs)
- Terraform / infrastructure-as-code for ML infra
- Full MLOps platform comparisons (Vertex AI Pipelines vs SageMaker Pipelines vs Azure ML)

---

## Connections

- Classical ML algorithms: `../1.machine_learning/`
- Deep learning training (DDP/FSDP basics): `../1.machine_learning/01_fundamentals/02_training_loops.md`
- LLM serving (vLLM internals): `../6.llms/05_vllm_internals.md`
- RAG patterns (the upstream of 13_production_rag_ops): `../7.rag/`
- Multi-tenant RAG system design: `../11.system_design/10_multi_tenant_rag.md`
- Distribution shift theory (the math behind drift): `../1.machine_learning/01_fundamentals/04_generalization.md`
- Agent evaluation (production agent metrics): `../8.agents/09_agent_evaluation.md`

## Practice

- vLLM serving → `../code_practice/04_llms/17_vllm_serve/` (run)
- FastAPI serving → `../code_practice/04_llms/11_fastapi_serve/` (docs only)
- Observability → `../code_practice/04_llms/16_observability/` (run)
- Eval harness → `../code_practice/04_llms/13_eval_harness/` (run)
- Production RAG → `../code_practice/05_rag/10_production_rag/` (docs only)

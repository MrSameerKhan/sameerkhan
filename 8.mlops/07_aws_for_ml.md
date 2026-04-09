# AWS for ML — Services, Architecture, and Interview Prep

## Quick Reference — ML-Relevant AWS Services

```
Data Storage:
  S3          → store datasets, model artifacts, logs (cheap, durable)
  EFS         → shared file system across EC2 instances (NFS-like, for training)
  FSx Lustre  → high-performance file system for large-scale training jobs

Compute:
  EC2         → virtual machines (GPU instances for training/inference)
  Lambda      → serverless functions (light inference, preprocessing triggers)
  ECS/EKS     → container orchestration (deploy inference containers at scale)
  Batch       → managed batch compute (large offline inference jobs)

ML Platform:
  SageMaker   → end-to-end managed ML (training, tuning, deployment, monitoring)

Orchestration:
  Step Functions → ML pipelines / workflows (chain train → eval → deploy)
  EventBridge    → event-driven triggers (new S3 file → start pipeline)
  Airflow (MWAA) → managed Airflow for complex DAG pipelines

Monitoring:
  CloudWatch  → logs, metrics, alarms for any AWS service
  CloudTrail  → audit log of all API calls (who did what, when)

Networking:
  VPC         → private network (your EC2/ECS lives inside a VPC)
  IAM         → permissions (who/what can access what)
  ECR         → Docker image registry (push images, pull in ECS/SageMaker)
```

---

## 1. S3 — Object Storage for ML

S3 is the backbone of every ML workflow on AWS. Everything lands in S3: raw data, processed datasets, model checkpoints, evaluation results, logs.

```python
import boto3
from pathlib import Path

s3 = boto3.client("s3", region_name="ap-south-1")  # Mumbai region

# ─── Upload dataset to S3 ─────────────────────────────────────────────────
def upload_dataset(local_path: str, bucket: str, prefix: str):
    for file in Path(local_path).rglob("*"):
        if file.is_file():
            key = f"{prefix}/{file.relative_to(local_path)}"
            s3.upload_file(str(file), bucket, key)
            print(f"Uploaded: {key}")

upload_dataset("./data/train", "my-ml-bucket", "datasets/sentiment/v2/train")

# ─── Download model checkpoint ────────────────────────────────────────────
s3.download_file(
    "my-ml-bucket",
    "models/bert-sentiment/v3/pytorch_model.bin",
    "/opt/models/bert-sentiment/pytorch_model.bin"
)

# ─── List files with prefix ───────────────────────────────────────────────
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket="my-ml-bucket", Prefix="datasets/sentiment/"):
    for obj in page.get("Contents", []):
        print(f"{obj['Key']}  {obj['Size']/(1024**2):.1f}MB  {obj['LastModified']}")

# ─── Presigned URL (give client temporary access to a file) ───────────────
url = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": "my-ml-bucket", "Key": "models/bert-v3/model.tar.gz"},
    ExpiresIn=3600,   # URL expires in 1 hour
)
print(url)  # share with client — no credentials needed
```

**S3 bucket structure for ML:**
```
my-ml-bucket/
├── datasets/
│   ├── raw/                     ← original data, never modified
│   │   └── 2024-01-15/
│   ├── processed/               ← cleaned, tokenized, formatted
│   │   └── sentiment-v2/
│   └── splits/                  ← train/val/test splits
│       └── sentiment-v2/train/
├── models/
│   ├── bert-sentiment/
│   │   ├── v1/                  ← model artifacts per version
│   │   └── v3/
│   └── onnx/
├── experiments/
│   └── run-20240115-143022/     ← metrics, configs, artifacts per run
├── inference-logs/
│   └── 2024-01/
└── checkpoints/
    └── training-run-20240115/   ← mid-training checkpoints
```

---

## 2. EC2 — GPU Instances for Training and Inference

### 2.1 GPU Instance Types for ML

```
Training instances (large GPU memory, NVLink):
  p3.2xlarge    1 × V100 16GB    $3.06/hr    single model training
  p3.8xlarge    4 × V100 16GB    $12.24/hr   multi-GPU, NVLink
  p3.16xlarge   8 × V100 16GB    $24.48/hr   multi-GPU, NVLink
  p4d.24xlarge  8 × A100 40GB    $32.77/hr   large model training
  p4de.24xlarge 8 × A100 80GB    $40.96/hr   very large models

Inference instances (lower cost, sufficient GPU):
  g4dn.xlarge   1 × T4 16GB      $0.526/hr   cost-effective inference
  g4dn.12xlarge 4 × T4 16GB      $3.912/hr   multi-GPU inference
  g5.xlarge     1 × A10G 24GB    $1.006/hr   best price/perf for inference
  inf1.xlarge   AWS Inferentia    $0.228/hr   AWS custom chip, ~3× cheaper

Cost optimization:
  Spot instances: up to 90% cheaper than On-Demand
  Use for: training (can checkpoint and resume if interrupted)
  Don't use for: production inference (can be terminated mid-request)
  On-Demand: use for inference (guaranteed availability)
```

### 2.2 Launch and Configure a Training Instance

```bash
# ─── Launch via AWS CLI ───────────────────────────────────────────────────
aws ec2 run-instances \
    --image-id ami-0d70546e43a941d70 \    # Deep Learning AMI (Ubuntu 22.04)
    --instance-type p3.2xlarge \
    --key-name my-ssh-key \
    --security-group-ids sg-0abc123 \
    --subnet-id subnet-0abc123 \
    --iam-instance-profile Name=ml-training-role \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=training-bert-v3}]'

# ─── SSH into instance ────────────────────────────────────────────────────
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=training-bert-v3" \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text
# 52.34.xx.xx

ssh -i ~/.ssh/my-ssh-key.pem ubuntu@52.34.xx.xx

# ─── Mount S3 bucket as local filesystem (s3fs) ───────────────────────────
sudo apt install -y s3fs
echo "AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY" \
    > ~/.passwd-s3fs && chmod 600 ~/.passwd-s3fs

mkdir -p /data
s3fs my-ml-bucket /data \
    -o passwd_file=~/.passwd-s3fs \
    -o allow_other \
    -o endpoint=ap-south-1

# Now /data/datasets/, /data/models/ etc. are accessible as local directories

# ─── Stop instance when done (saves money) ───────────────────────────────
aws ec2 stop-instances --instance-ids i-0abc123456
# Stopped instances: no compute charges, still pay for EBS storage
```

---

## 3. SageMaker — Managed ML Platform

SageMaker handles: training jobs, hyperparameter tuning, model registry, endpoints.
Use when: you want AWS to manage the compute, you don't want to SSH into servers.

### 3.1 Training Job

```python
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker import get_execution_role

role = get_execution_role()   # IAM role SageMaker uses to access S3, EC2
session = sagemaker.Session()

# ─── Define training job ──────────────────────────────────────────────────
estimator = PyTorch(
    entry_point="train.py",          # your training script
    source_dir="./src",              # directory with train.py + dependencies
    role=role,
    instance_count=1,
    instance_type="ml.p3.2xlarge",   # SageMaker instance type (ml. prefix)
    framework_version="2.1",
    py_version="py310",
    hyperparameters={
        "model_name": "bert-base-uncased",
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 16,
    },
    output_path="s3://my-ml-bucket/sagemaker-output/",
    checkpoint_s3_uri="s3://my-ml-bucket/checkpoints/",   # save checkpoints to S3
)

# ─── Data channels: S3 paths mapped to local /opt/ml/input/data/<channel> ─
data_channels = {
    "train": "s3://my-ml-bucket/datasets/sentiment-v2/train/",
    "validation": "s3://my-ml-bucket/datasets/sentiment-v2/val/",
}

# ─── Launch training job ──────────────────────────────────────────────────
estimator.fit(data_channels, wait=True)   # wait=True blocks until done
# Logs stream to your terminal. Output model saved to S3 output_path.
```

**Your training script (train.py) reads data from environment:**
```python
# In train.py — SageMaker injects these paths automatically
import os

TRAIN_DATA_DIR = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
VAL_DATA_DIR   = os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
MODEL_DIR      = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")    # save model here
OUTPUT_DIR     = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output")

# ... training code ...

# Save model (SageMaker compresses /opt/ml/model → model.tar.gz → uploads to S3)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
```

### 3.2 Hyperparameter Tuning

```python
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="validation:f1",
    objective_type="Maximize",
    hyperparameter_ranges={
        "learning_rate": ContinuousParameter(1e-5, 5e-4),
        "batch_size": IntegerParameter(8, 32),
        "epochs": IntegerParameter(2, 5),
    },
    max_jobs=20,                  # run 20 training jobs total
    max_parallel_jobs=4,          # run 4 at a time
    strategy="Bayesian",          # Bayesian optimization (smarter than random)
)

tuner.fit(data_channels)

# Get best hyperparameters
best_job = tuner.best_training_job()
print(tuner.best_estimator().hyperparameters())
```

### 3.3 Real-Time Inference Endpoint

```python
# Deploy trained model to a managed endpoint
predictor = estimator.deploy(
    initial_instance_count=2,          # 2 instances for redundancy
    instance_type="ml.g5.xlarge",      # GPU inference instance
    endpoint_name="bert-sentiment-v3",
)

# Call the endpoint
import json

response = predictor.predict(
    data=json.dumps({"texts": ["This movie was great!"]}),
    initial_args={"ContentType": "application/json"}
)
print(json.loads(response))

# Auto-scaling (scale based on invocations per instance)
import boto3
autoscaling = boto3.client("application-autoscaling")

autoscaling.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/bert-sentiment-v3/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)

autoscaling.put_scaling_policy(
    PolicyName="InvocationsPerInstance",
    ServiceNamespace="sagemaker",
    ResourceId=f"endpoint/bert-sentiment-v3/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 1000,            # scale when > 1000 invocations/instance/min
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 300,
        "ScaleOutCooldown": 60,
    }
)
```

---

## 4. Lambda — Serverless Inference / Triggers

Use Lambda for: lightweight inference (ONNX < 250MB), preprocessing triggers, event-driven pipelines.
Don't use for: heavy GPU inference (Lambda has no GPU), long-running jobs (max 15 min).

```python
# lambda_handler.py — deployed as a Lambda function
import json
import boto3
import onnxruntime as ort
import numpy as np
import os

# Model loaded ONCE when Lambda container starts (warm start optimization)
MODEL_BUCKET = os.environ["MODEL_BUCKET"]
MODEL_KEY = os.environ["MODEL_KEY"]

# Download model to /tmp (Lambda's writable directory, 512MB–10GB)
s3 = boto3.client("s3")
s3.download_file(MODEL_BUCKET, MODEL_KEY, "/tmp/model.onnx")

# Load ONNX model
sess = ort.InferenceSession(
    "/tmp/model.onnx",
    providers=["CPUExecutionProvider"]
)

def lambda_handler(event, context):
    """Called for every request."""
    body = json.loads(event.get("body", "{}"))
    texts = body.get("texts", [])

    if not texts:
        return {"statusCode": 400, "body": json.dumps({"error": "texts required"})}

    # Tokenize + infer (using pre-tokenized input for simplicity)
    inputs = tokenize(texts)   # your tokenization function
    logits = sess.run(["logits"], {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    })[0]

    probs = softmax(logits)
    labels = ["negative", "positive"]
    predictions = [
        {"text": t, "label": labels[np.argmax(p)], "confidence": float(np.max(p))}
        for t, p in zip(texts, probs)
    ]

    return {
        "statusCode": 200,
        "body": json.dumps({"predictions": predictions}),
        "headers": {"Content-Type": "application/json"},
    }
```

```bash
# Deploy Lambda via AWS CLI
# Package dependencies
pip install onnxruntime transformers -t ./lambda_package/
cp lambda_handler.py ./lambda_package/

cd lambda_package && zip -r ../lambda.zip . && cd ..

# Create Lambda function
aws lambda create-function \
    --function-name ml-inference \
    --runtime python3.11 \
    --role arn:aws:iam::123456789:role/lambda-ml-role \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb://lambda.zip \
    --timeout 30 \
    --memory-size 3008 \
    --environment Variables="{MODEL_BUCKET=my-ml-bucket,MODEL_KEY=models/onnx/model.onnx}"

# Add API Gateway trigger (HTTP endpoint → Lambda)
aws lambda create-function-url-config \
    --function-name ml-inference \
    --auth-type AWS_IAM
```

---

## 5. ECS + ECR — Container-Based Inference at Scale

Use ECS when: you have a Docker-based inference server (FastAPI + PyTorch) and want AWS to manage the cluster.

```bash
# ─── Push Docker image to ECR ─────────────────────────────────────────────
AWS_ACCOUNT=123456789012
REGION=ap-south-1

# Create ECR repository
aws ecr create-repository --repository-name ml-inference --region $REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com

# Build and push
docker build -t ml-inference .
docker tag ml-inference:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ml-inference:v3
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ml-inference:v3
```

```json
// ECS Task Definition (defines one "container unit")
{
  "family": "ml-inference-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "ml-inference",
      "image": "123456789012.dkr.ecr.ap-south-1.amazonaws.com/ml-inference:v3",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "MODEL_PATH", "value": "/models/bert-v3"},
        {"name": "DEVICE", "value": "cpu"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/ml-inference",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

```bash
# Create ECS Service with auto-scaling
aws ecs create-service \
    --cluster ml-cluster \
    --service-name ml-inference-service \
    --task-definition ml-inference-task:3 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-abc],securityGroups=[sg-abc],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=ml-inference,containerPort=8000"
```

---

## 6. IAM — Permissions (Critical to Understand)

IAM controls who/what can access which AWS resource. You will be asked about this.

```
Key concepts:
  User:   a human identity (developer, admin)
  Role:   an identity assumed by a service (EC2 assumes role to access S3)
  Policy: a JSON document that says "allow/deny action on resource"
  Group:  collection of users with shared permissions

For ML workloads — typical roles:
  training-role:     EC2/SageMaker → read S3 (datasets), write S3 (checkpoints),
                     write CloudWatch (logs)
  inference-role:    ECS/Lambda → read S3 (model), write CloudWatch (logs)
  pipeline-role:     Step Functions → invoke SageMaker, Lambda, read/write S3
  developer-role:    human → launch EC2, manage SageMaker, read S3 (not write prod)
```

```json
// Example: training IAM policy
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::my-ml-bucket/datasets/*",
        "arn:aws:s3:::my-ml-bucket"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::my-ml-bucket/checkpoints/*"
    },
    {
      "Effect": "Allow",
      "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
      "Resource": "arn:aws:logs:*:*:log-group:/ml-training/*"
    }
  ]
}
```

**Principle of least privilege:** grant only the permissions needed. Training job should NOT have permission to delete S3 objects or launch EC2 instances.

---

## 7. Step Functions — ML Pipeline Orchestration

Use Step Functions to chain: preprocess → train → evaluate → if good → deploy → notify.

```json
// State machine definition (simplified)
{
  "Comment": "ML training and deployment pipeline",
  "StartAt": "PreprocessData",
  "States": {
    "PreprocessData": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:function:preprocess-data",
      "Next": "TrainModel"
    },
    "TrainModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
      "Parameters": {
        "TrainingJobName.$": "$.job_name",
        "AlgorithmSpecification": {"TrainingInputMode": "File", "...": "..."},
        "InputDataConfig": [{"ChannelName": "train", "DataSource": {"S3DataSource": {"S3Uri.$": "$.train_data_uri"}}}],
        "OutputDataConfig": {"S3OutputPath.$": "$.output_path"},
        "ResourceConfig": {"InstanceType": "ml.p3.2xlarge", "InstanceCount": 1, "VolumeSizeInGB": 50}
      },
      "Next": "EvaluateModel"
    },
    "EvaluateModel": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:function:evaluate-model",
      "Next": "CheckAccuracy"
    },
    "CheckAccuracy": {
      "Type": "Choice",
      "Choices": [
        {
          "Variable": "$.f1_score",
          "NumericGreaterThanEquals": 0.85,
          "Next": "DeployModel"
        }
      ],
      "Default": "NotifyFailure"
    },
    "DeployModel": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sagemaker:createEndpoint.sync",
      "Next": "NotifySuccess"
    },
    "NotifySuccess": {"Type": "Task", "Resource": "arn:aws:lambda:...:function:notify", "End": true},
    "NotifyFailure": {"Type": "Task", "Resource": "arn:aws:lambda:...:function:notify", "End": true}
  }
}
```

---

## 8. CloudWatch — Monitoring and Alerting

```python
import boto3

cloudwatch = boto3.client("cloudwatch", region_name="ap-south-1")

# ─── Push custom metrics ──────────────────────────────────────────────────
def push_inference_metrics(latency_ms: float, batch_size: int):
    cloudwatch.put_metric_data(
        Namespace="MLInference",
        MetricData=[
            {
                "MetricName": "InferenceLatencyMs",
                "Value": latency_ms,
                "Unit": "Milliseconds",
                "Dimensions": [{"Name": "ModelVersion", "Value": "v3"}],
            },
            {
                "MetricName": "BatchSize",
                "Value": batch_size,
                "Unit": "Count",
            },
        ],
    )

# ─── Create alarm ────────────────────────────────────────────────────────
cloudwatch.put_metric_alarm(
    AlarmName="HighInferenceLatency",
    MetricName="InferenceLatencyMs",
    Namespace="MLInference",
    Statistic="p95",
    Period=300,              # 5-minute window
    EvaluationPeriods=2,     # alarm if 2 consecutive periods breach threshold
    Threshold=500.0,
    ComparisonOperator="GreaterThanThreshold",
    AlarmActions=["arn:aws:sns:ap-south-1:123456789:ml-alerts"],  # SNS → email/Slack
    TreatMissingData="notBreaching",
)
```

---

## 9. Typical ML Architecture on AWS

```
┌────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│  Raw data → S3 (raw/) → Lambda (trigger on upload) →               │
│  Step Functions (preprocess pipeline) → S3 (processed/)            │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                       TRAINING LAYER                                │
│  SageMaker Training Job (ml.p3.2xlarge)                            │
│  Input: S3 (processed/) → Output: S3 (models/v3/)                 │
│  Checkpoints: S3 (checkpoints/)                                    │
│  Logs: CloudWatch                                                  │
│  Experiments: SageMaker Experiments (or MLflow on EC2)             │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                      EVALUATION LAYER                               │
│  Lambda evaluator: load model from S3, run eval dataset            │
│  If F1 > 0.85 → Step Functions triggers deployment                 │
│  Model registered in SageMaker Model Registry                      │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                  │
│  Option A: SageMaker Endpoint (managed, auto-scaling)              │
│  Option B: ECS Fargate (Docker container, more control)            │
│  Option C: Lambda (serverless, small ONNX model)                   │
│  ALB (Application Load Balancer) in front of all options           │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│                     MONITORING LAYER                                │
│  CloudWatch: latency, error rate, invocation count                 │
│  SageMaker Model Monitor: data drift, prediction drift             │
│  CloudTrail: who called what API (audit)                           │
│  SNS → email/Slack alerts on CloudWatch alarms                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## 10. Cost Optimization

```
Training:
  Use Spot instances: up to 90% cheaper than On-Demand
  Always checkpoint to S3 every N steps — Spot can be interrupted
  Use SageMaker Managed Spot Training (handles interruption/resume automatically)
  
  aws sagemaker create-training-job ... \
      --enable-managed-spot-training \
      --stopping-condition MaxRuntimeInSeconds=86400,MaxWaitTimeInSeconds=86400 \
      --checkpoint-config S3Uri=s3://my-bucket/checkpoints/

Inference:
  Right-size instances: profile your model on g4dn.xlarge vs g5.xlarge vs inf1
  Batch transform for offline inference (not real-time endpoint)
  Lambda for low-traffic endpoints (pay per request, not per hour)
  Graviton instances (ARM): 20-40% cheaper for CPU-bound inference

Storage:
  S3 lifecycle policies: move old data to S3-IA (Infrequent Access) or Glacier
  Delete unused checkpoints automatically after 30 days
  
  aws s3api put-bucket-lifecycle-configuration \
      --bucket my-ml-bucket \
      --lifecycle-configuration '{
          "Rules": [{
              "ID": "expire-old-checkpoints",
              "Filter": {"Prefix": "checkpoints/"},
              "Status": "Enabled",
              "Expiration": {"Days": 30}
          }]
      }'
```

---

## 11. Interview Q&A

**Q: How would you set up a model training pipeline on AWS?**
A: S3 for data storage (raw → processed → train/val/test splits). SageMaker Training Job for managed training — specifies instance type, entry point script, S3 input/output paths. The training script reads from `/opt/ml/input/data/` and saves the model to `/opt/ml/model/` — SageMaker handles S3 upload automatically. For orchestration: Step Functions chains preprocess Lambda → SageMaker Training Job → evaluate Lambda → conditional deploy. CloudWatch for logs and alarms. MLflow or SageMaker Experiments for experiment tracking.

**Q: When would you use SageMaker vs ECS for inference?**
A: SageMaker Endpoint: when you want fully managed serving, auto-scaling, A/B testing between model variants, and don't need to customize the container deeply. Best for standard PyTorch/TensorFlow models. ECS: when you need full control over the container (custom inference server, specific dependencies, non-standard batching logic), when you're already using ECS for other services, or when you want to avoid SageMaker's overhead costs. Lambda: for very lightweight models (ONNX < 250MB), event-driven inference, or low-traffic endpoints where paying per invocation is cheaper than a running endpoint.

**Q: How do you control costs for ML workloads on AWS?**
A: Training: Spot instances (up to 90% cheaper) with checkpointing to S3 every N steps — SageMaker Managed Spot Training handles interruption and resume automatically. Inference: right-size instances (profile on smallest GPU first), use Batch Transform for offline workloads instead of a running endpoint, Lambda for bursty/low-traffic endpoints. Storage: S3 lifecycle policies to move old data to Infrequent Access after 30 days and expire checkpoints after 30 days. Always tag resources with project/team for cost attribution.

**Q: What is IAM and why does it matter for ML systems?**
A: IAM controls who and what can access AWS resources. For ML: each component (EC2 training instance, Lambda inference, Step Functions pipeline) assumes an IAM Role with only the permissions it needs. The training role gets: S3 read on datasets, S3 write on checkpoints, CloudWatch write for logs — nothing else. This principle of least privilege prevents accidental or malicious data access. In practice, a misconfigured IAM policy is one of the most common causes of S3 data breaches. For interviews: always mention that the inference role should NOT have write access to the training data bucket.

**Q: How would you set up event-driven ML inference on AWS?**
A: New file uploaded to S3 → S3 Event Notification → Lambda trigger → Lambda calls SageMaker Endpoint (or runs ONNX inference inline) → result written to S3 or pushed to SQS → downstream service reads from SQS. For high throughput: S3 event → SQS queue → ECS workers pull from queue → batch inference → write results to S3. SQS decouples ingestion from processing — if inference is slow, requests queue up instead of being lost.

---

## Key Takeaway

AWS ML stack = S3 (data + model storage) + EC2/SageMaker (compute) + ECR + ECS/Lambda (serving) + Step Functions (pipelines) + CloudWatch (monitoring) + IAM (permissions). For JPMorgan context: data in S3, training on SageMaker or EC2 GPU instances (Spot for cost savings), model artifacts back to S3, inference on SageMaker Endpoint or ECS behind an ALB, pipelines in Step Functions, monitoring in CloudWatch with SNS alerts. IAM roles for every component — least privilege. Cost optimization: Spot for training + right-sized instances for inference + S3 lifecycle policies.

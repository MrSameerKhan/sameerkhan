# 05 — On-Premise ML Model Deployment — End to End

> A remote client device sends a request → load balancer → inference server → ML model → response back to client. Every step from bare metal to production-ready serving.

---

## 0. The Full Picture

```
CLIENT DEVICE
(mobile app / laptop / another server)
POST https://api.company.com/predict
{"text": "cat sat on mat"}
        |
        | HTTPS (port 443)
        ↓
LOAD BALANCER (Nginx)
- TLS termination (HTTPS → HTTP internally)
- Route to healthy inference server
- Rate limiting
- Port 443 externally + Port 8000 internally
        |
   _____|_____
  ↓           ↓
Inference #1    Inference #2
FastAPI+Uvicorn FastAPI+Uvicorn
GPU: NVIDIA A100 GPU: NVIDIA A100
Port 8000       Port 8000
        |
        ↓
MODEL LAYER
PyTorch / ONNX / TensorRT model loaded in GPU memory
Preprocessing → Tokenizer → Forward pass → Decode output
        |
        ↓
MONITORING (Prometheus + Grafana)
Latency P50/P95/P99 · Requests/s · GPU util · Error rate
```

---

## 1. Hardware Prerequisites

```
Minimum for a small NLP model (BERT-base, 110M params):
  CPU:  8 cores (Intel Xeon or AMD EPYC)
  RAM:  32 GB
  GPU:  NVIDIA GPU with ≥ 8 GB VRAM (RTX 3080, A10)
  Disk: 100 GB SSD (OS + model weights + logs)
  NIC:  1 Gbps Ethernet (10 Gbps for high-throughput)

For large models (70B LLM):
  CPU:  16+ cores
  RAM:  64 GB
  GPU:  NVIDIA A100 80 GB (or 2× A10 24 GB with tensor parallel)
  Disk: 500 GB NVMe SSD
  NIC:  10 Gbps Ethernet

Check GPU presence:
  lspci | grep -i nvidia
  nvidia-smi   (should show GPU model, memory, driver version)
```

---

## 2. Step-by-Step Setup

### Step 1: OS Prerequisites

```bash
# Tested on: Ubuntu 22.04 LTS (recommended for CUDA compatibility)
# Run as root or with sudo

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential build tools
sudo apt install -y \
  build-essential curl wget git unzip \
  software-properties-common ca-certificates \
  gnupg lsb-release htop nvtop \
  net-tools ufw

# Check OS version
lsb_release -a
# Ubuntu 22.04.3 LTS
```

### Step 2: NVIDIA Driver + CUDA

```bash
# — Install NVIDIA Driver ——————————————————————
# Ubuntu shortcut (installs recommended driver automatically):
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# OR manually specify version:
sudo apt install -y nvidia-driver-535

# Reboot required after driver install
sudo reboot

# Verify driver
nvidia-smi
# Output:
# | NVIDIA-SMI 535.xx   Driver Version: 535.xx   CUDA Version: 12.2 |
# | GPU 0   NVIDIA A100 80GB Off  | 00000000:01:00.0 Off |   0 |

# — Install CUDA Toolkit ——————————————————————
# Match CUDA version to your driver version (nvidia-smi shows compatible CUDA)
# Requires CUDA 525+
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

# Verify CUDA
nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.1, V12.1.66

# — Install cuDNN (required for PyTorch neural nets) ———
sudo apt install -y libcudnn8 libcudnn8-dev
```

### Step 3: Python Environment

```bash
# — Install Python 3.11 ——————————————————————
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Verify
python3.11 --version
# Python 3.11.7

# Create isolated virtual environment
# One env per model/project — no dependency conflicts
python3.11 -m venv /opt/ml_serving/venv

# Activate
source /opt/ml_serving/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# — Install ML dependencies ——————————————————
# PyTorch with CUDA (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HuggingFace
pip install transformers accelerate sentencepiece

# API server
pip install fastapi uvicorn[standard] pydantic

# Monitoring
pip install prometheus-client prometheus-fastapi-instrumentator

# Utilities
pip install numpy scipy python-dotenv requests

# Freeze requirements
pip freeze > /opt/ml_serving/requirements.txt

# Verify GPU in Python
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# CUDA: True, GPU: NVIDIA A100 80GB PCIe
```

### Step 4: Docker (Optional but Recommended)

Docker isolates the entire runtime — same image runs on dev, staging, prod.

```bash
# — Install Docker ————————————————————————————
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o \
  /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
# Docker version 24.0.5

# — Install NVIDIA Container Toolkit (GPU access inside containers) ———
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o \
  /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] \
  https://#g" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test: run nvidia-smi inside a container
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Step 5: Model Preparation

```
Directory structure:
mkdir -p /opt/ml_serving/{models,logs,config,code}
# /opt/ml_serving/
#   ├── models/      ← model weights
#   ├── logs/        ← application logs
#   ├── config/      ← env files, nginx config
#   └── code/        ← inference server code
```

```python
# save_model.py — run once to download and optimize the model
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "bert-base-uncased"
SAVE_PATH = "/opt/ml_serving/models/bert-sentiment"

# Download
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Put in eval mode (disables dropout)
model.eval()

# Save locally (so inference server doesn't need internet)
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"Model saved to {SAVE_PATH}")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
# Model size: 109.5M parameters
```

Optional: Export to ONNX for faster inference:

```python
# export_onnx.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/opt/ml_serving/models/bert-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("/opt/ml_serving/models/bert-sentiment")
model.eval()

# Create dummy input for tracing
dummy = tokenizer("dummy input", return_tensors="pt", padding="max_length", max_length=128)

# Export
torch.onnx.export(
    model,
    args=(dummy["input_ids"], dummy["attention_mask"]),
    f="/opt/ml_serving/models/bert-sentiment/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=16,
)
print("ONNX model exported")

# Verify ONNX
import onnxruntime as ort
sess = ort.InferenceSession(
    "/opt/ml_serving/models/bert-sentiment/model.onnx",
    providers=["CUDAExecutionProvider"]
)
print(ort.InferenceSession.run(sess, None, {
    "input_ids": dummy["input_ids"].numpy(),
    "attention_mask": dummy["attention_mask"].numpy(),
}))
```

### Step 6: Inference Server (FastAPI)

```python
# /opt/ml_serving/code/server.py
import time
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from prometheus_fastapi_instrumentator import Instrumentator
import logging

# — Logging setup ————————————————————————————
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("/opt/ml_serving/logs/server.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("inference_server")

# — Config —————————————————————————————————
MODEL_PATH = "/opt/ml_serving/models/bert-sentiment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
LABELS = ["negative", "positive"]

# — Load model at startup ————————————————————
logger.info(f"Loading model from {MODEL_PATH} on {DEVICE}...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()  # disable dropout
logger.info(f"Model loaded in {time.time() - start:.2f}s on {DEVICE}")

# — FastAPI App ————————————————————————————
app = FastAPI(title="ML Inference Server", version="1.0.0")

# CORS: allow calls from other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to specific domains in prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Prometheus metrics auto-instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# — Request / Response schemas ————————————
class PredictRequest(BaseModel):
    texts: list[str]   # a batch of texts
    class Config:
        json_schema_extra = {
            "example": {"texts": ["This movie was great!", "Terrible experience."]}
        }

class PredictionResult(BaseModel):
    text: str
    label: str
    confidence: float
    inference_time_ms: float

class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    total_time_ms: float
    device: str

# — Inference function ———————————————————
@torch.inference_mode()  # equivalent to no_grad() but faster
def predict_batch(texts: list[str]) -> list[dict]:
    """Run model inference on a batch of texts."""
    t0 = time.time()

    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    # Move tensors to GPU
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    # Forward pass
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Softmax → probabilities
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    inference_ms = (time.time() - t0) * 1000

    return [
        {
            "text": text,
            "label": LABELS[int(probs[i].argmax())],
            "confidence": float(probs[i].max()),
            "inference_time_ms": round(inference_ms / len(texts), 2),
        }
        for i, text in enumerate(texts, probs)
    ]

# — API Endpoints ————————————————————————
@app.get("/health")
def health():
    """Health check — load balancer pings this."""
    return {
        "status": "healthy",
        "device": DEVICE,
        "gpu_memory_used_mb": (
            torch.cuda.memory_allocated() // (1024 ** 2) if DEVICE == "cuda" else 0
        ),
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Main inference endpoint."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list is empty")

    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.texts)} exceeds max {MAX_BATCH_SIZE}"
        )

    t0 = time.time()
    logger.info(f"Predicting batch of {len(request.texts)} texts")

    try:
        results = predict_batch(request.texts)
    except RuntimeError as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    total_ms = round((time.time() - t0) * 1000, 2)
    logger.info(f"Batch done in {total_ms}ms")

    return PredictResponse(
        predictions=results,
        total_time_ms=total_ms,
        device=DEVICE,
    )

@app.get("/model-info")
def model_info():
    """Model metadata."""
    return {
        "model_path": MODEL_PATH,
        "labels": LABELS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_batch_size": MAX_BATCH_SIZE,
        "device": DEVICE,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
```

```bash
# Start the server:
source /opt/ml_serving/venv/bin/activate
uvicorn server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \        # 1 worker per GPU (GPU memory not shareable across workers)
  --log-level info

# With gunicorn (production — process manager + uvicorn workers)
pip install gunicorn
gunicorn server:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind 0.0.0.0:8000 \
  --worker-connections 1000 \
  --timeout 60 \
  --access-logfile /opt/ml_serving/logs/access.log \
  --error-logfile /opt/ml_serving/logs/error.log \
  --daemon   # run in background
```

### Step 7: systemd Service (Auto-start on Boot)

```bash
sudo tee /etc/systemd/system/ml-inference.service << 'EOF'
[Unit]
Description=ML Inference Server
After=network.target
Wants=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/ml_serving/code
Environment="PATH=/opt/ml_serving/venv/bin:/usr/local/cuda-12.1/bin:/usr/bin:/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"
ExecStart=/opt/ml_serving/venv/bin/gunicorn server:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind 0.0.0.0:8000 \
  --worker-connections 1000 \
  --timeout 60 \
  --access-logfile /opt/ml_serving/logs/access.log \
  --error-logfile /opt/ml_serving/logs/error.log
Restart=always
RestartSec=5
StandardOutput=append:/opt/ml_serving/logs/service.log
StandardError=append:/opt/ml_serving/logs/service.log

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable ml-inference   # auto-start on boot
sudo systemctl start ml-inference

# Check status
sudo systemctl status ml-inference

# View logs
sudo journalctl -u ml-inference -f
```

### Step 8: Load Balancer (Nginx)

Nginx handles: TLS termination, reverse proxy, load balancing, rate limiting.

```bash
# Install nginx
sudo apt install -y nginx

# SSL certificate (self-signed for internal use)
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/server.key \
  -out /etc/nginx/ssl/server.crt \
  -subj "/C=IN/ST=Maharashtra/L=Mumbai/O=Company/CN=api.company.com"

# For production: use Let's Encrypt (certbot) for real certificates
# sudo apt install certbot python3-certbot-nginx
# sudo certbot --nginx -d api.company.com
```

```nginx
# /etc/nginx/sites-available/ml-api
upstream inference_servers {
    # Round-robin load balancing (default)
    server 127.0.0.1:8000 weight=1;  # inference server #1
    server 127.0.0.1:8001 weight=1;  # inference server #2 (different port if same machine)
    # For multiple machines:
    # server 192.168.1.10:8000 weight=1;
    # server 192.168.1.11:8000 weight=1;
    keepalive 32;  # persistent connections to upstream
}

# Rate limiting zone: 100 req/s per IP
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;

server {
    listen 443 ssl http2;
    server_name api.company.com;

    # SSL
    ssl_certificate     /etc/nginx/ssl/server.crt;
    ssl_certificate_key /etc/nginx/ssl/server.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!kRSA;

    # Request size limit (prevent huge payloads)
    client_max_body_size 10m;

    # Timeouts (match your model inference time)
    proxy_connect_timeout 5s;
    proxy_read_timeout   60s;
    proxy_send_timeout   60s;

    location /predict {
        # Rate limiting: burst of 20, then strict 100 req/s
        limit_req zone=api_limit burst=20 nodelay;
        limit_req_status 429;

        proxy_pass http://inference_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Connection "";  # keepalive
    }

    location /health {
        # Health checks: no rate limiting
        proxy_pass http://inference_servers;
        proxy_set_header Host $host;
    }

    location /metrics {
        # Restrict metrics to internal network only
        allow 10.0.0.0/8;
        allow 192.168.0.0/16;
        deny all;
        proxy_pass http://inference_servers;
    }
}

# Redirect HTTP → HTTPS
server {
    listen 80;
    server_name api.company.com;
    return 301 https://$host$request_uri;
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/ml-api /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # remove default page

# Test config
sudo nginx -t
# nginx: configuration file /etc/nginx/nginx.conf test is successful

# Start nginx
sudo systemctl enable nginx
sudo systemctl start nginx
sudo systemctl status nginx
```

### Step 9: Firewall

```bash
# UFW (Uncomplicated Firewall)
sudo ufw enable

# Allow SSH (CRITICAL — do this first or you'll be locked out)
sudo ufw allow 22/tcp

# Allow HTTPS (client traffic)
sudo ufw allow 443/tcp

# Allow HTTP (nginx redirects to HTTPS)
sudo ufw allow 80/tcp

# Block direct access to inference server port (only nginx should reach it)
# By default, ufw blocks all incoming unless explicitly allowed
# Port 8000 NOT added → only nginx on localhost can reach it

# Check rules
sudo ufw status verbose
```

### Step 10: Monitoring (Prometheus + Grafana)

```bash
# — Prometheus ——————————————————————————————
wget https://github.com/prometheus/prometheus/releases/download/v2.48.0/prometheus-2.48.0.linux-amd64.tar.gz
tar xvf prometheus-2.48.0.linux-amd64.tar.gz
sudo mv prometheus-2.48.0.linux-amd64 /opt/prometheus

# /opt/prometheus.yml
cat > /opt/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-inference'
    static_configs:
      - targets: ['localhost:8000']   # FastAPI /metrics endpoint
    metrics_path: '/metrics'

  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9400']   # nvidia_gpu_exporter
EOF

# Start Prometheus
/opt/prometheus/prometheus --config.file=/opt/prometheus/prometheus.yml &

# — Grafana ——————————————————————————————
sudo apt install -y grafana
sudo systemctl enable grafana-server
sudo systemctl start grafana-server
# Access: http://server_ip:3000 (admin/admin)
# Add Prometheus as data source → import FastAPI dashboard (ID: 14282)

# — NVIDIA GPU Exporter ——————————————————
# Exports GPU metrics (utilization, memory, temperature) to Prometheus
wget https://github.com/utkuozdemir/nvidia_gpu_exporter/releases/download/v1.2.0/nvidia_gpu_exporter_1.2.0_linux_amd64.tar.gz
tar xvf nvidia_gpu_exporter_1.2.0_linux_amd64.tar.gz
./nvidia_gpu_exporter --web.listen-address ":9400" &
```

Key metrics in Grafana:

```
http_requests_total{endpoint="/predict"}           → total requests
http_request_duration_seconds{endpoint="/predict"} → latency histogram (P50/P95/P99)
http_request_duration_seconds_total                → error rate
nvidia_gpu_memory_used_bytes                       → GPU memory usage
nvidia_gpu_utilization_rate                        → GPU utilization %
nvidia_gpu_temperature_celsius                     → GPU temperature (alert > 85°C)
```

---

## 3. Complete Request Flow — Step by Step

```
Client sends:
  POST https://api.company.com/predict
  Headers: Content-Type: application/json
  Body: {"texts": ["cat sat on mat"]}

Step 1 — Network:
  Client DNS resolves api.company.com → 192.168.1.100 (server IP)
  TCP connection on port 443
  TLS handshake (client verifies server certificate)

Step 2 — Nginx receives request:
  TLS termination: HTTPS → plain HTTP
  Reads upstream pool: [127.0.0.1:8000, 127.0.0.1:8001]
  Round-robin: route to 127.0.0.1:8000 (Server #1's turn)
  Check rate limit: user's IP has 43 requests in last second → under limit, allow
  Forward to FastAPI: POST http://127.0.0.1:8000/predict

Step 3 — FastAPI receives request:
  Uvicorn receives HTTP request
  Pydantic validates: {"texts": ["cat sat on mat"]} ✓
  Calls predict_batch(["cat sat on mat"])

Step 4 — Preprocessing:
  tokenizer("cat sat on mat") →
    input_ids: tensor([[101, 2611, 2938, 2006, 13523, 102]])
    attention_mask: tensor([[1, 1, 1, 1, 1, 1]])
  .to("cuda") → tensors moved to GPU memory

Step 5 — Model Inference:
  model forward pass on GPU [3ms]
  → encoder layers (BERT: 12 layers, 12 heads)
  → [CLS] hidden state → linear head
  → logits: tensor([-0.83, 1.24]) (negative, positive)

Step 6 — Postprocessing:
  argmax([-0.83, 1.24]) = 1 → "positive"
  confidence = 0.86

Step 7 — Response built:
  {
    "predictions": [{"text": "cat sat on mat", "label": "positive", "confidence": 0.86,
                     "inference_time_ms": 5.1}],
    "total_time_ms": 5.1,
    "device": "cuda"
  }

Step 8 — Response sent back:
  FastAPI → Uvicorn → HTTP response → Nginx
  Nginx adds headers (X-Request-ID, etc.) → HTTPS response → Client

Total latency: ~5-20ms (depends on model size, batch size, GPU)
```

---

## 4. Docker Deployment (Recommended for Reproducibility)

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt update && apt install -y python3.11 python3.11-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model
COPY code/ ./code/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "code.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  inference-1:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]   # GPU 0
              capabilities: [gpu]
    volumes:
      - /opt/ml_serving/logs:/app/logs
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  inference-2:
    build: .
    ports:
      - "8001:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]   # GPU 1 (second GPU)
              capabilities: [gpu]
    volumes:
      - /opt/ml_serving/logs:/app/logs
    restart: always

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
```

```bash
# Start everything
docker compose up -d

# Check logs
docker compose logs -f inference-1

# Scale up (add another replica)
docker compose up -d --scale inference-1=3
```

---

## 5. Verification Checklist

```bash
# — From a remote client machine ————————————

# Health check
curl -k https://api.company.com/health
# {"status": "healthy", "device": "cuda", "gpu_memory_used_mb": 420}

# Prediction
curl -k -X POST https://api.company.com/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["cat sat on mat", "terrible product!"]}'
# {"predictions": [{"text": "cat sat on mat", "label": "positive", "confidence": 0.86, ...}, ...]}

# Model info
curl -k https://api.company.com/model-info

# Load test (install: pip install locust)
locust -f locustfile.py --host https://api.company.com

# — On the server ————————————————————————

# Check services running
sudo systemctl status ml-inference nginx

# Check GPU utilization during load
watch -n 1 nvidia-smi

# Check logs
tail -f /opt/ml_serving/logs/server.log
tail -f /opt/ml_serving/logs/access.log

# Check Prometheus metrics
curl http://localhost:8000/metrics | grep http_request

# Check firewall
sudo ufw status
```

---

## 6. Common Issues and Fixes

```
Problem: CUDA out of memory
  Reason: batch too large or GPU too small
  Fix: reduce MAX_BATCH_SIZE, use model quantization (fp16/int8)
  Check: watch nvidia-smi during inference

Problem: High latency (>500ms for small model)
  Reason: tokenization on CPU, small batch size, no GPU
  Fix: ensure DEVICE == "cuda", use ONNX Runtime with CUDAExecutionProvider
  Check: add timing logs around each step

Problem: Port 8000 not reachable from client
  Reason: firewall blocking direct access (good — only nginx should expose it)
  Fix: client should use port 443 (nginx), not 8000
  Check: sudo ufw status

Problem: Nginx 502 Bad Gateway
  Reason: inference server not running or wrong port
  Fix: check systemctl status ml-inference, check upstream port in nginx config
  Check: curl http://localhost:8000/health

Problem: Model loads on CPU despite GPU available
  Reason: CUDA not in PATH, or torch installed without CUDA
  Fix: python -c "import torch; print(torch.cuda.is_available())"
  If False: reinstall PyTorch with correct CUDA wheel

Problem: Server crashes after hours of use
  Reason: GPU memory leak (tensors accumulating)
  Fix: ensure @torch.inference_mode() decorator, call torch.cuda.empty_cache() periodically
```

---

## 7. Production Gaps — What Most Guides Skip

### 7.1 Network Setup (Before Everything Else)

```bash
# — Set static IP (so server IP never changes) ——
# Edit netplan config
sudo nano /etc/netplan/00-installer-config.yaml
```

```yaml
network:
  version: 2
  ethernets:
    eth3:   # your NIC name (check with: ip link show)
      dhcp4: false
      addresses:
        - 192.168.1.100/24   # static IP
      routes:
        - to: default
          via: 192.168.1.1   # gateway
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

```bash
sudo netplan apply

# Set hostname
sudo hostnamectl set-hostname ml-inference-01

# Add to /etc/hosts (all servers know each other)
sudo tee -a /etc/hosts << 'EOF'
192.168.1.100  ml-inference-01
192.168.1.101  ml-inference-02
192.168.1.110  ml-loadbalancer
EOF

# Sync time (critical for TLS certificates and log timestamps)
sudo apt install -y chrony
sudo systemctl enable chrony --now
chronyc tracking   # verify time is synced
```

### 7.2 Security Hardening

```bash
# — SSH: key-based auth only, disable password login ——

# On your LOCAL machine: generate SSH key pair
ssh-keygen -t ed25519 -C "sameer@company.com" -f ~/.ssh/ml_serving_key

# Copy public key to server
ssh-copy-id -i ~/.ssh/ml_serving_key.pub ubuntu@192.168.1.100

# On SERVER: disable password authentication
sudo nano /etc/ssh/sshd_config
# Set these values:
#   PasswordAuthentication no
#   PubkeyAuthentication yes
#   PermitRootLogin no    # never log in as root directly
#   MaxAuthTries 3        # lock after 3 wrong attempts
#   AllowTcpForwarding no

sudo systemctl restart sshd

# Verify: from local machine
ssh -i ~/.ssh/ml_serving_key ubuntu@192.168.1.100  # should work
ssh ubuntu@192.168.1.100                            # should fail (password auth disabled)

# — fail2ban: auto-ban IPs with too many failed SSH attempts ——
sudo apt install -y fail2ban

sudo tee /etc/fail2ban/jail.local << 'EOF'
[sshd]
enabled = true
port = 22
maxretry = 3      # ban after 3 failures
bantime = 3600    # ban for 1 hour
findtime = 600    # within 10 minutes

[nginx-limit-req]
enabled = true
port = http,https
logpath = /var/log/nginx/error.log
maxretry = 10
bantime = 600
EOF

sudo systemctl enable fail2ban --now
sudo fail2ban-client status sshd   # check banned IPs
```

### 7.3 API Authentication (API Keys on FastAPI)

Never expose the inference endpoint without authentication.

```python
# auth.py — add to your FastAPI app
import hashlib, hmac, time
from fastapi import Header, HTTPException, Depends
from functools import lru_cache
import os

# In production: store in environment variable or secret manager, never in code
VALID_API_KEYS = set(os.environ.get("API_KEYS", "").split(","))

def verify_api_key(x_api_key: str = Header(..., description="API key")):
    """Validate API key on every request."""
    if not x_api_key or x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key

# In server.py — protect the /predict endpoint
@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
async def predict(request: PredictRequest):
    ...

# Client sends:
# curl -H "x-api-key: sk-your-key-here" -X POST https://api.company.com/predict ...

# Generate secure random API keys:
# python3 -c "import secrets; print('sk-' + secrets.token_urlsafe(32))"

# Set in environment (never hardcode):
# echo 'API_KEYS=sk-AbCdEFGHIjkUMnOpQrStUvWxYz012345678901234567' >> /opt/ml_serving/.env
```

### 7.4 Secrets Management (.env File)

```bash
# /opt/ml_serving/.env — never commit to git
cat /opt/ml_serving/.env
```

```
MODEL_PATH=/opt/ml_serving/models/bert-sentiment
DEVICE=cuda
MAX_BATCH_SIZE=32
MAX_SEQ_LENGTH=128
API_KEYS=sk-AbCdEFGHIjkUMnOpQrStUvWxYz012345678901234567
LOG_LEVEL=INFO
PROMETHEUS_PORT=8001
```

```python
# Load in server.py
from dotenv import import load_dotenv
load_dotenv("/opt/ml_serving/.env")
import os

MODEL_PATH = os.environ["MODEL_PATH"]
DEVICE = os.environ.get("DEVICE", "cpu")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "8"))
API_KEYS = set(os.environ.get("API_KEYS", "").split(","))
```

```bash
# Permissions: only owner can read .env
chmod 600 /opt/ml_serving/.env

# Add to .gitignore
echo ".env" >> /opt/ml_serving/.gitignore
```

### 7.5 Model Warm-Up

The first inference is always slow — CUDA compiles kernels, allocates memory, JIT compiles ops. Warm up the model during startup so users don't see the slow first request.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # — Startup ——————————————————
    logger.info("Warming up model...")
    dummy_texts = ["warmup text"] * MAX_BATCH_SIZE

    for i in range(3):   # 3 warm-up passes
        t0 = time.time()
        _ = predict_batch(dummy_texts)
        logger.info(f"Warm-up pass {i+1}: {(time.time()-t0)*1000:.1f}ms")

    # After 3 passes, CUDA is fully warmed up
    logger.info("Model warm-up complete. Ready to serve.")
    yield

    # — Shutdown ——————————————————
    logger.info("Shutting down. Draining in-flight requests...")
    # FastAPI/uvicorn handles this automatically with SIGTERM

app = FastAPI(title="ML Inference Server", lifespan=lifespan)

# Warm-up output in logs:
# Warm-up pass 1: 312ms   → slow (CUDA kernel compilation)
# Warm-up pass 2: 18.7ms  → fast (compiled + cached)
# Warm-up pass 3: 17.2ms  → stable
# Model warm-up complete. Ready to serve.
```

### 7.6 Dynamic Batching (High Throughput)

Instead of processing each request as it arrives, collect requests over a short window (e.g., 20ms) and process them as one batch. Dramatically improves GPU utilization.

```python
# dynamic_batcher.py
import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

@dataclass
class PendingRequest:
    texts: list[str]
    future: asyncio.Future
    arrival_time: float = field(default_factory=time.time)

class DynamicBatcher:
    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 20.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000  # convert to seconds
        self.queue: deque[PendingRequest] = deque()
        self.lock = asyncio.Lock()
        self._processing = False

    async def add_request(self, texts: list[str]) -> list[dict]:
        """Add texts to batch queue. Returns when result is ready."""
        future = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append(PendingRequest(texts=texts, future=future))
            if not self._processing:
                self._processing = True
                asyncio.create_task(self._process_loop())
        return await future

    async def _process_loop(self):
        """Process pending requests in batches."""
        while self.queue:
            await asyncio.sleep(self.max_wait_ms)  # wait to collect requests

            async with self.lock:
                # Collect up to max_batch_size requests
                batch_requests = []
                all_texts = []
                while self.queue and len(all_texts) < self.max_batch_size:
                    req = self.queue.popleft()
                    batch_requests.append(req)
                    all_texts.extend(req.texts)

            if not all_texts:
                continue

            # Process the full batch
            results = predict_batch(all_texts)

            # Distribute results back to individual futures
            idx = 0
            for req in batch_requests:
                n = len(req.texts)
                req.future.set_result(results[idx:idx + n])
                idx += n

        async with self.lock:
            self._processing = False

# Usage in endpoint:
batcher = DynamicBatcher(max_batch_size=32, max_wait_ms=20)

@app.post("/predict")
async def predict(request: PredictRequest):
    results = await batcher.add_request(request.texts)
    return PredictResponse(predictions=results, ...)
```

```
Why dynamic batching matters:

Without batching:
  100 requests arrive in 1 second, each single text
  100 × (model forward pass) = 100 × 15ms = 1500ms GPU time
  GPU utilization: ~30% (mostly waiting between requests)

With dynamic batching (20ms window):
  20ms window collects ~2 requests on avg
  50 batches × (model forward pass) = 50 × 10ms = 500ms GPU time
  GPU utilization: ~70-80%
  Latency per request: +20ms wait + total ~35ms (vs 15ms) — acceptable tradeoff
```

### 7.7 Graceful Shutdown

When you restart the server (to deploy a new model), in-flight requests should complete before the process exits.

```bash
# FastAPI + uvicorn handles graceful shutdown automatically with SIGTERM
# But you need to configure the timeout correctly

# In gunicorn command:
gunicorn server:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --graceful-timeout 30 \  # wait 30s for in-flight requests to finish
  --timeout 60             # will worker if no response in 60s

# Manual graceful restart sequence:
# Step 1: Tell nginx to stop sending new traffic to server-1
#   nginx upstream: mark server as down
# Step 2: Wait for in-flight requests to drain (watch access log goes quiet)
tail -f /opt/ml_serving/logs/access.log

# Step 3: Restart service
sudo systemctl restart ml-inference

# Step 4: Verify healthy
curl http://localhost:8000/health

# Step 5: Re-add to nginx upstream
sudo nginx -s reload
```

### 7.8 Blue-Green Deployment (Zero Downtime Model Update)

When you have a new model version, deploy it without any downtime.

```
Current state:
  nginx → upstream: [server-1:8000 (model v1), server-2:8000 (model v1)]

Blue-green steps:
  Step 1: Start new model v2 on server-1:8001 (different port)
  Step 2: Nginx health check passes on 8001
  Step 3: nginx reload → upstream: [server-1:8001 (v2), server-2:8000 (v1)]
  Step 4: Monitor metrics — no error rate increase
  Step 5: Deploy v2 to server-2:8001
  Step 6: nginx reload → upstream: [server-1:8001 (v2), server-2:8001 (v2)]
  Step 7: Shut down old v1 processes
```

```bash
# Step 1: Start v2 on new port (don't kill v1 yet)
MODEL_PATH=/opt/ml_serving/models/bert-v2 \
  uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1 &

# Step 2: Health check
curl http://localhost:8001/health
# {"status": "healthy", "device": "cuda"}

# Step 3: Update nginx upstream to include v2
# /etc/nginx/sites-available/ml-api
# upstream inference_servers {
#   server 127.0.0.1:8001 weight=1;  # v2 (new)
#   server 127.0.0.1:8000 weight=1;  # v1 (old, still running)
# }
sudo nginx -t && sudo nginx -s reload

# Step 4: Watch for errors (5 minutes)
watch -n 5 "curl -s http://localhost:9090/api/v1/query?query=http_requests_total"

# Step 5: All good → remove v1 from nginx, kill old process
# Update nginx to only use port 8001, reload, then kill port 8000 process
```

### 7.9 Model Quantization for Production

Quantize the model to reduce memory and increase throughput.

```python
# FP16 (half precision) — 2× memory reduction, same accuracy for most models
model = model.half()   # convert all weights to float16
model.to("cuda")

# Or load in fp16 directly:
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,   # load in fp16
    device_map="cuda",
)

# INT8 quantization (4× memory reduction) using bitsandbytes
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="cuda",
)

# ONNX Runtime quantization (best for CPU serving)
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input="/opt/ml_serving/models/bert/model.onnx",
    model_output="/opt/ml_serving/models/bert/model_int8.onnx",
    weight_type=QuantType.QInt8,
)
```

```
Memory and speed comparison for BERT-base:

| Precision | Model size | GPU memory | Throughput | Accuracy loss |
|-----------|-----------|------------|------------|---------------|
| FP32      | 440 MB    | 1.4 GB     | 1× baseline| 0%            |
| FP16      | 220 MB    | 0.7 GB     | 1.8×       | <0.1%         |
| INT8      | 110 MB    | 0.4 GB     | 2.5×       | <0.5%         |
| INT4      | 55 MB     | 0.2 GB     | 3×         | -1-2%         |
```

### 7.10 Log Rotation (Prevent Disk Full)

Without log rotation, inference logs fill the disk within days.

```bash
sudo tee /etc/logrotate.d/ml-inference << 'EOF'
/opt/ml_serving/logs/*.log {
    daily
    rotate 14         # keep 14 days of logs
    compress          # gzip rotated logs
    delaycompress     # compress previous, not current
    missingok         # don't error if log file missing
    notifempty        # don't rotate empty logs
    sharedscripts
    postrotate
        # Signal gunicorn to reopen log files after rotation
        kill -USR1 $(cat /var/run/ml-inference.pid) 2>/dev/null || true
    endscript
}
EOF

# Test rotation
sudo logrotate -d /etc/logrotate.d/ml-inference  # dry run
sudo logrotate -f /etc/logrotate.d/ml-inference  # force rotate now

# Check disk usage
df -h /opt/ml_serving/logs/
du -sh /opt/ml_serving/logs/*
```

### 7.11 Disk Setup (Production)

```bash
# — Mount a dedicated NVMe SSD for models and logs ——
# (separate from OS disk — model I/O doesn't impact OS)

# List disks
lsblk
# NAME    MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
# sda       8:0    0   50G  0  disk /          ← OS disk
# nvme0n1 259:0    0  500G  0  disk             ← data disk (unmounted)

# Format with ext4
sudo mkfs.ext4 /dev/nvme0n1

# Create mount point
sudo mkdir -p /opt/ml_serving

# Mount
sudo mount /dev/nvme0n1 /opt/ml_serving

# Auto-mount on boot (get UUID first)
sudo blkid /dev/nvme0n1
# /dev/nvme0n1: UUID="a1b2c3d4-..."  TYPE="ext4"
echo "UUID=a1b2c3d4-... /opt/ml_serving ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab

# Verify
sudo mount -a  # re-mount all entries in fstab (tests it's valid)
df -h | grep ml_serving
# /dev/nvme0n1  500G  20G  480G   4%  /opt/ml_serving

# — Disk alert cron ——————————————————————
# Alert when disk is 80% full
cat << 'EOF' > /etc/cron.d/disk-alert
*/15 * * * * root df -h /opt/ml_serving | awk 'NR==2{gsub(/%/,""); if($5>80) print "ALERT: ML serving disk " $5 "% full"}' | mail -s "Disk Alert" admin@company.com 2>/dev/null
EOF
```

### 7.12 Backup Strategy

```bash
# — Model weights backup (models are irreplaceable) ——

# Option 1: rsync to backup server (run daily via cron)
# /etc/cron.d/model-backup
# 0 2 * * * ubuntu rsync -av2 --delete \
#   /opt/ml_serving/models/ \
#   backup-server:/backups/ml-models/$(date +%Y-%m-%d)/ \
#   >> /opt/ml_serving/logs/backup.log 2>&1

# Option 2: tar + compress + copy to NFS share
daily_backup() {
    DATE=$(date +%Y-%m-%d)
    tar czf /backups/models-${DATE}.tar.gz /opt/ml_serving/models/
    # Delete backups older than 30 days
    find /backups/ -name "models-*.tar.gz" -mtime +30 -delete
}

# Option 3: checksum verification (detect corruption)
md5sum /opt/ml_serving/models/bert-sentiment/pytorch_model.bin \
  > /opt/ml_serving/models/bert-sentiment/checksums.md5
# Before every deployment: verify checksums match
md5sum -c /opt/ml_serving/models/bert-sentiment/checksums.md5
```

### 7.13 Load Testing

Before going live, stress test to find the breaking point.

```python
# locustfile.py
from locust import HttpUser, task, between
import json, random

class InferenceUser(HttpUser):
    wait_time = between(0.1, 0.5)  # random wait between requests

    SAMPLE_TEXTS = [
        "This product is amazing and I love it!",
        "Terrible experience. Would not recommend.",
        "Average quality, nothing special.",
        "Best purchase I've made this year.",
        "Broken on arrival. Very disappointed.",
    ]

    @task(10)
    def predict_single(self):
        """Single text prediction."""
        self.client.post(
            "/predict",
            json={"texts": [random.choice(self.SAMPLE_TEXTS)]},
            headers={"x-api-key": "sk-test-key"},
        )

    @task(3)
    def predict_batch(self):
        """Batch prediction (more realistic)."""
        batch = random.sample(self.SAMPLE_TEXTS, k=min(4, len(self.SAMPLE_TEXTS)))
        self.client.post(
            "/predict",
            json={"texts": batch},
            headers={"x-api-key": "sk-test-key"},
        )

    @task(1)
    def health_check(self):
        self.client.get("/health")
```

```bash
pip install locust

# Run load test: 50 concurrent users, ramp up over 30s
locust -f locustfile.py \
  --host https://api.company.com \
  --users 50 \
  --spawn-rate 5 \
  --run-time 120s \
  --headless \
  --csv results/load_test_50u

# Results:
# Name           Reqs  Fails  Median  P95   P99  RPS
# POST/predict   4020  0      12ms    28ms  45ms  48.2
# GET /health    480   0      2ms     4ms   4ms   4.0

# Increase until you see failures or P95 > SLA
# Common finding: GPU saturates at ~200 QPS for BERT-Base batch=1
```

### 7.14 Complete Production Checklist

```
Infrastructure:
  ☑ Static IP assigned
  ☑ Hostname set in /etc/hosts on all servers
  ☑ NTP synced (chrony)
  ☑ NVMe SSD mounted for models/logs
  ☑ Disk alert at 80% full

Security:
  ☑ SSH key-only auth (password auth disabled)
  ☑ Root login disabled
  ☑ fail2ban running
  ☑ UFW firewall (22, 80, 443 only)
  ☑ .env file permissions 600
  ☑ API key auth on /predict endpoint
  ☑ No secrets in code or git

Server:
  ☑ CUDA driver verified (nvidia-smi)
  ☑ Python venv isolated
  ☑ Model warm-up on startup
  ☑ @torch.inference_mode() on forward pass
  ☑ Dynamic batching configured
  ☑ Graceful shutdown (graceful-timeout 30s)
  ☑ systemd service (auto-start, auto-restart)
  ☑ logrotate configured

Load Balancer:
  ☑ TLS certificate (Let's Encrypt, not self-signed in prod)
  ☑ HTTP → HTTPS redirect
  ☑ Rate limiting (100 req/s per IP)
  ☑ /health endpoint for health checks
  ☑ /metrics blocked from public

Reliability:
  ☑ Blue-green deployment procedure documented
  ☑ Model checksum verification
  ☑ Daily model backup (rsync or tar to backup server)
  ☑ Load test completed (know your breaking point)

Monitoring:
  ☑ Prometheus scraping /metrics
  ☑ Grafana dashboard (latency P50/P95/P99, GPU util, error rate)
  ☑ GPU temp alert (> 85°C)
  ☑ Error rate alert (> 1%)
  ☑ Disk full alert (> 80%)
```

---

## Key Takeaway

```
On-premise ML deployment = hardware (GPU + drivers + CUDA) → Python env →
model (save locally, optional ONNX export) → FastAPI inference server
(Uvicorn/Gunicorn) → systemd (auto-start) → Nginx (TLS + load balancing +
rate limiting) → firewall (block direct GPU server access) → monitoring
(Prometheus + Grafana).

The request flow: client → HTTPS → Nginx → HTTP → FastAPI → tokenize →
GPU forward pass → JSON → Nginx → HTTPS → client. Docker wraps everything
for reproducibility.

Critical details: one worker per GPU; @torch.inference_mode for speed;
health endpoint for load balancer; GPU memory monitoring to catch OOM before
it crashes production.

Production extras that most guides skip: static IP + NTP + hostname setup,
SSH key-only + fail2ban, model warm-up (3 passes), dynamic batching (20ms
window, fp16 → 2× speed), blue-green deployment (zero downtime), dedicated
NVMe disk, backup with checksums, load testing to find breaking point.
```

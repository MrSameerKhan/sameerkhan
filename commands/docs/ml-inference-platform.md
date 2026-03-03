ML Inference Platform — Notes (Steps 1 to 14)

Owner: Sameer Khan
Platform: Windows laptop + Docker Desktop (Linux containers)
Outcome: Local training + artifact packaging + local SageMaker simulation + Jenkins CI + ECR push working.

Step 1 — Initialize Repository Structure
What we did

Created a clean repo on C: drive (important for Docker Desktop file sharing).

Recommended structure (what we used):

ml-inference-platform/
├── training/
├── inference/
├── jenkins/
├── artifacts/                 # generated, not committed
├── .gitignore
└── .dockerignore

Git ignore decisions

In .gitignore (or at least not committed):

.venv/

artifacts/

__pycache__/

.pytest_cache/

.DS_Store

In .dockerignore:

.venv/

artifacts/ (do NOT bake artifacts into image)

.git/

__pycache__/

Step 2 — Setup Python Virtual Environment (Windows)
What we did

Created and activated venv:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip


(If needed once)

Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Step 3 — Train Model & Generate Artifacts
Goal

Generate versioned artifacts locally.

We implemented

training/train.py trains a simple text classifier using:

TfidfVectorizer (scikit-learn)

small Keras dense network (TensorFlow)

Output artifacts (per version)

Running:

python training\train.py --model_version v3.0.0 --out_dir artifacts


Creates:

artifacts/versioned/v3.0.0/
├── keras_model/          # Keras SavedModel-style folder (keras save)
├── preprocess.joblib     # TF-IDF vectorizer
├── label_map.json        # label_to_id + id_to_label
└── metadata.json         # version, hashes, metrics, etc.

Why versioned artifacts

supports reproducibility

allows rollback

clean separation of model versions

Step 4 — Smoke Test Model Locally
Goal

Confirm artifacts load and a prediction flow works before Docker/SageMaker.

What we did

Created and ran training/smoke_test.py that:

loads preprocess.joblib

loads keras_model

loads label_map.json

runs a sample prediction on new text

Command:

python training\smoke_test.py --artifact_dir artifacts\versioned\v3.0.0


Expected result:

prints predicted label + confidence

proves artifacts are valid

Step 5 — Package Artifacts into saved_model.tar.gz
Goal

Create SageMaker-compatible model artifact archive:

SageMaker will download from S3

extract into /opt/ml/model

What we did

Created training/package_artifacts.py to produce:
artifacts/versioned/v3.0.0/saved_model.tar.gz

Inside tar:

saved_model/
  saved_model.pb
  variables/
preprocess.joblib
label_map.json
metadata.json


Command:

python training\package_artifacts.py --artifact_dir artifacts\versioned\v3.0.0


Verification (on Git Bash / WSL, because Windows PowerShell tar differs):

tar -tf artifacts/versioned/v3.0.0/saved_model.tar.gz

Step 6 — Create Inference Runtime (app + serve)
Goal

Implement SageMaker BYOC contract:

container starts with: docker run <image> serve

health check: GET /ping

inference: POST /invocations

What we created

inference/app.py

reads artifacts from MODEL_DIR env var with default /opt/ml/model

loads preprocess.joblib, label_map.json, metadata.json

exposes:

/ping (200 only if artifacts loaded)

/invocations (POST; dummy response confirming loaded artifacts)

Key configuration:

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")


inference/serve

ENTRYPOINT script invoked by SageMaker

starts gunicorn on port 8080

IMPORTANT: use sh, not bash (slim images often lack bash)

Example serve (minimal):

#!/usr/bin/env sh
set -e
exec gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 60 app:app

Step 7 — Define Inference Dependencies
Goal

Keep inference runtime lightweight.

inference/requirements.txt (runtime only):

flask
gunicorn
numpy
joblib
scikit-learn


Note:

We intentionally did NOT add training-only packages here.

TensorFlow is added later when real inference is wired into container.

Step 8 — Create Inference Dockerfile
Goal

Build a Linux container that satisfies SageMaker runtime expectations.

inference/Dockerfile:

base: python:3.9-slim

copy requirements + pip install

copy inference files

fix CRLF line endings for serve (Windows → Linux)

chmod +x serve

expose 8080

ENTRYPOINT /opt/program/serve

Key lines:

RUN sed -i 's/\r$//' serve
RUN chmod +x serve
ENTRYPOINT ["/opt/program/serve"]

Step 9 — Build Inference Docker Image (Local)

Built from repo root:

docker build -t ml-infer:docker -f inference\Dockerfile .


Confirmed image exists:

docker images

Step 10 — Run Container with Artifacts Mounted (Simulate SageMaker)
Goal

Simulate SageMaker behavior locally:

image contains code + runtime

model artifacts mounted to /opt/ml/model

Run:

docker run --rm -p 8080:8080 `
  -v "C:\Users\samee\Desktop\ml-inference-platform\artifacts\versioned\v3.0.0:/opt/ml/model" `
  ml-infer:docker serve


Important: path must be real and on C: for Docker Desktop file sharing.

Step 11 — Validate /ping Endpoint

From Git Bash / curl:

curl http://127.0.0.1:8080/ping


Expected:

HTTP 200

status ok

Step 12 — Validate /invocations Endpoint

From Git Bash:

curl -X POST http://127.0.0.1:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"text":"refund my money"}'


Expected JSON:

echoes request

returns model_version

returns available_labels

This confirmed:

container starts correctly via serve

artifacts are loaded from /opt/ml/model

inference endpoint reachable on port 8080

Step 13 — Jenkins in Docker + Docker Access Working
Goal

Jenkins pipeline must be able to run:

docker build

docker push

What we had initially

Jenkins running in Docker, socket mounted:

/var/run/docker.sock mounted ✅
But Docker CLI missing inside Jenkins ❌

Fix 1 — Build custom Jenkins image with Docker CLI

Created: jenkins/Dockerfile.jenkins

FROM jenkins/jenkins:lts
USER root
RUN apt-get update && apt-get install -y docker.io && rm -rf /var/lib/apt/lists/*
USER jenkins


Built:

docker build -t jenkins-docker:lts -f jenkins\Dockerfile.jenkins .

Fix 2 — Jenkins volume permission issue

Jenkins failed with:

cannot touch /var/jenkins_home/... Permission denied


Solution: recreate volume (fresh Jenkins home):

docker rm -f jenkins
docker volume rm jenkins_home

Run Jenkins
docker run -d --name jenkins `
  -p 8081:8080 -p 50000:50000 `
  -v jenkins_home:/var/jenkins_home `
  -v /var/run/docker.sock:/var/run/docker.sock `
  jenkins-docker:lts

Fix 3 — Docker socket permission denied

Error:

permission denied ... /var/run/docker.sock


We inspected inside Jenkins container:

docker exec -it jenkins bash -lc "id && ls -l /var/run/docker.sock"


On Docker Desktop, socket was root:root (gid 0), so we:

added jenkins user to group 0

restarted container

Verification command (worked):

docker exec -it jenkins bash -lc "docker ps"


Result:

Jenkins can run Docker commands

Step 13 complete

Step 14 — Jenkins Pipeline: Build & Push Image to ECR
14.1 Unlock Jenkins

Got initial admin password:

docker exec -it jenkins bash -lc "cat /var/jenkins_home/secrets/initialAdminPassword"


Installed suggested plugins, created admin user.

14.2 Add AWS creds to Jenkins (Credentials)

In Jenkins UI:

Manage Jenkins → Credentials → global → Add

Kind: Username with password

ID: aws-cli

username = AWS_ACCESS_KEY_ID

password = AWS_SECRET_ACCESS_KEY

14.3 Verify AWS creds in pipeline

Created a minimal pipeline stage to run:

aws sts get-caller-identity


Confirmed success:
Account: 570617927874

14.4 Full Jenkinsfile for ECR push

Pipeline stages:

checkout

build docker image using inference/Dockerfile

ensure ECR repo exists

login to ECR

tag image with b<buildnum>-<gitsha>

push to ECR

Final pushed image:
570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-infer:b2-91db72b

This completes Step 14.

Result After Step 14

✅ We can train + package model locally
✅ We can run inference container locally in SageMaker-like mode
✅ Jenkins can build Docker images
✅ Jenkins can authenticate to AWS
✅ Jenkins can push images to ECR with versioned tags



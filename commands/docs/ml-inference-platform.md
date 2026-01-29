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


ML Inference Platform — Notes (Steps 14 to 17)

Owner: Sameer Khan
Platform: Windows laptop + Docker Desktop (Linux containers) + Jenkins (Dockerized) + AWS
Outcome: End-to-end CI/CD from code → ECR → SageMaker endpoint → invocation working.

Step 14 — Jenkins Pipeline (Extended): CI + CD Preparation
Goal

Extend Jenkins from only pushing to ECR → to full CI/CD, capable of:

building inference images

pushing to ECR

deploying to SageMaker

invoking endpoint for smoke testing

Step 14.5 — Jenkins Runtime Constraints (Important Discovery)
Problem

Jenkins is running inside Docker, so:

Jenkins workspace exists inside Jenkins container

Docker engine runs on host

Any docker run -v $PWD:/work fails because host cannot see Jenkins-internal paths

This caused errors like:

python: can't open file '/work/sagemaker/deploy.py'

Correct Solution

Use Docker’s volume sharing between containers:

--volumes-from jenkins


This allows:

Python containers

skopeo containers

to see the exact same Jenkins workspace.

This is the correct production pattern when Jenkins itself is containerized.

Step 14.6 — Python Not Available in Jenkins Container
Problem

Jenkins container did not have Python installed:

python: not found

Correct Design Decision

Do not install Python into Jenkins.

Instead:

run deploy & invoke scripts inside throwaway Python containers

mount Jenkins workspace into those containers

Example pattern:

docker run --rm \
  --volumes-from jenkins \
  -w "$WORKSPACE" \
  python:3.10-slim bash -lc "python script.py"


This keeps Jenkins:

minimal

reproducible

production-grade

Step 15 — Deploy to SageMaker via Jenkins (BYOC)
Goal

Create / update SageMaker endpoint automatically from Jenkins using:

custom inference image (BYOC)

model artifacts from S3

Step 15.1 — OCI Image Manifest Issue (Critical Learning)
Problem

Docker Buildx (docker buildx build --push) produces OCI image index by default:

application/vnd.oci.image.index.v1+json


SageMaker does NOT support OCI manifests.

Error observed:

Unsupported manifest media type application/vnd.oci.image.index.v1+json


This happened even though:

image was linux/amd64

image worked locally

image ran fine in Docker

Step 15.2 — Correct Fix: Registry-Level Conversion (Not Rebuild)
Why rebuilding was wrong

Rebuilding on Linux does not guarantee Docker v2 manifest

Buildx still pushes OCI by default

Waste of time and compute

Correct fix (used successfully)

Convert the already-pushed image in ECR to Docker v2 schema2 using skopeo:

application/vnd.docker.distribution.manifest.v2+json


This conversion:

happens at registry level

does not rebuild layers

is fast

is deterministic

Step 15.3 — Jenkins Stage: OCI → Docker v2 Conversion

We added a Jenkins stage using skopeo inside a container:

Key details:

skopeo image has ENTRYPOINT = skopeo

must override entrypoint to /bin/sh

must pass AWS ECR token explicitly

Correct pattern:

docker run --rm \
  --entrypoint /bin/sh \
  quay.io/skopeo/stable:latest -c '
    skopeo copy --format v2s2 \
      --src-creds "AWS:$PASS" \
      --dest-creds "AWS:$PASS" \
      docker://<oci-image> \
      docker://<docker-v2-image>
  '


Result:

new tag *-v2

SageMaker-compatible image

Step 15.4 — Manifest Verification (Mandatory)

Before deploying, Jenkins verifies:

aws ecr batch-get-image \
  --repository-name ml-infer \
  --image-ids imageTag=<tag>-v2 \
  --query 'images[].imageManifestMediaType'


Expected output:

application/vnd.docker.distribution.manifest.v2+json


Only this image is passed to SageMaker.

Step 15.5 — SageMaker Deployment Script (deploy.py)

Jenkins calls sagemaker/deploy.py which:

Creates model (if not exists)

Creates endpoint config (if not exists)

Updates endpoint if exists

Waits until endpoint is InService

Design decisions:

Stable endpoint name (textcls-endpoint-dev)

Immutable model & config names

Supports rollback by reusing older config

Deployment executed from Jenkins using:

docker run --rm \
  --volumes-from jenkins \
  -w "$WORKSPACE" \
  python:3.10-slim bash -lc "python sagemaker/deploy.py ..."

Step 16 — Invoke SageMaker Endpoint (Smoke Test)
Goal

Ensure deployment is actually usable, not just “green”.

Jenkins runs a smoke test immediately after deploy:

python sagemaker/invoke.py \
  --endpoint-name textcls-endpoint-dev \
  --text "refund my money"


Expected response:

HTTP 200

JSON containing:

model_version

available_labels

echoed input

This ensures:

container started successfully

artifacts loaded from /opt/ml/model

endpoint networking works

IAM permissions are correct

If this step fails → pipeline fails.

This is true CD, not just CI.

Step 17 — Versioning, Rollback, and Deployment Strategy
17.1 — Image Versioning (Immutable)

Each Jenkins build produces:

b<BUILD_NUMBER>-<git_sha>-amd64
b<BUILD_NUMBER>-<git_sha>-amd64-v2


Rules:

never overwrite tags

never use latest for deployments

always deploy by exact tag

17.2 — SageMaker Resource Naming Strategy
Resource	Strategy
Model	textcls-model-<image_tag>
EndpointConfig	textcls-config-<image_tag>
Endpoint	textcls-endpoint-dev (stable)

This allows:

safe updates

instant rollback

auditability

17.3 — Rollback Procedure (Zero Rebuild)

Rollback requires no Docker build.

Steps:

Identify previous good endpoint config

Run:

aws sagemaker update-endpoint \
  --endpoint-name textcls-endpoint-dev \
  --endpoint-config-name <old-config>


Traffic switches automatically.

17.4 — Cost Awareness

InService endpoints incur cost per hour

Failed endpoints do not serve traffic but should be deleted

Best practice:

keep dev endpoint only when actively testing

delete after session

Final Outcome (After Step 17)

✅ Local training and versioned artifacts
✅ SageMaker-compatible artifact packaging
✅ Local SageMaker simulation using Docker
✅ Jenkins running in Docker with Docker access
✅ Jenkins CI building inference images
✅ Jenkins CD deploying to SageMaker
✅ Automatic OCI → Docker v2 conversion
✅ Endpoint smoke-tested via Jenkins
✅ Rollback strategy defined and tested

Summary (Why this is production-grade)

This platform matches real enterprise ML systems:

Code → Jenkins → ECR → SageMaker → Endpoint → Smoke Test


No shortcuts, no hacks, no manual console steps.
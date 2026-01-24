End-to-End ML API CI/CD (Mac M1 → AWS ECR → Windows “Prod”)

# End-to-End ML API CI/CD (Mac M1 → AWS ECR → Windows “Prod”)  
**Scope:** From zero → working deployment on Windows (**up to Step 5.9**).  
**Not included:** GitHub webhook auto-trigger (we’ll do that next).

---

## 0) Target Architecture

- **MacBook (M1)**: development + Docker build + (later) Jenkins CI build
- **AWS ECR (ap-south-1)**: Docker image registry
- **Windows laptop**: “prod server” that **pulls image from ECR** and runs container

Flow:

Mac (build linux/amd64) → ECR (push) → Windows (pull + run)


Why `linux/amd64`?
- Mac M1 = ARM64
- Windows laptop typically = AMD64  
So we build **linux/amd64** to avoid runtime errors on Windows.

---

## 1) Create a Clean GitHub Repo

### 1.1 Create repo on GitHub
- Repo name: `ml-api`
- Add: README
- Add: `.gitignore` (Python)

### 1.2 Clone to Mac
```bash
cd ~/Documents
git clone https://github.com/<YOUR_GITHUB_USERNAME>/ml-api.git
cd ml-api


2) Create FastAPI Inference Service + Dockerfile (Mac)

2.1 Create folder structure

cd ~/Documents/ml-api
mkdir -p app tests
touch app/main.py requirements.txt Dockerfile .dockerignore

2.2 Add API code: app/main.py
2.3 Add dependencies: requirements.txt
2.4 Add Dockerfile
2.5 Add .dockerignore
2.6 Ensure Docker Desktop is running (Mac)
2.7 Build & run locally (Mac)
Build: docker build -t ml-api:local .
Run: docker run --rm -p 8080:8080 ml-api:local
Test (new terminal):
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"x":10}'
2.8 Commit & push to GitHub
git add .
git commit -m "Add FastAPI inference service with Dockerfile"
git push origin main


3) Configure AWS + Create ECR + Push Image from Mac (M1)
3.1 AWS CLI check (Mac)
aws --version
3.2 Configure AWS credentials (Mac)
aws configure
AWS Access Key ID
AWS Secret Access Key
Default region: ap-south-1
Output: json
Verify: aws sts get-caller-identity
3.3 Create ECR repository (one time)
aws ecr create-repository --repository-name ml-model-api --region ap-south-1
3.4 Login Docker to ECR (Mac)
aws ecr get-login-password --region ap-south-1 \
| docker login --username AWS --password-stdin 570617927874.dkr.ecr.ap-south-1.amazonaws.com
3.5 Build Windows-compatible image on M1 (linux/amd64) Create buildx builder (safe to run multiple times):
docker buildx create --use --name mlbuilder || true
Build amd64 + load locally: docker buildx build --platform linux/amd64 -t ml-model-api:latest --load .
Verify image exists:docker images | grep ml-model-api
3.6 Tag image for ECR 
docker tag ml-model-api:latest \
570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-model-api:latest
3.7 Push to ECR docker push 570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-model-api:latest
3.8 Verify image exists in ECR aws ecr describe-images --repository-name ml-model-api --region ap-south-1


4) Windows “Prod” Pull & Run from ECR
4.1 Install tools (Windows)
Install Docker Desktop
Install AWS CLI (MSI from AWS site)
docker --version
aws --version

4.2 Configure AWS (Windows)
aws configure
Use same credentials as Mac.
Region: ap-south-1
Output: json
Verify: aws sts get-caller-identity

4.3 Login Docker to ECR (Windows)
aws ecr get-login-password --region ap-south-1 `
| docker login --username AWS --password-stdin 570617927874.dkr.ecr.ap-south-1.amazonaws.com
4.4 Pull image 
docker pull 570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-model-api:latest
4.5 Run container
If an old container exists:
docker rm -f ml-model-api
Run: docker run -d --name ml-model-api -p 8080:8080 `
570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-model-api:latest
4.6 Test (Windows)
curl http://localhost:8080/health
curl -Method POST http://localhost:8080/predict -ContentType "application/json" -Body '{"x":10}'


5) Add Jenkins CI (Docker-based on Mac) → Build & Push to ECR
Outcome: Jenkins will build (linux/amd64) and push images to ECR.
5.1 Run Jenkins container (Mac)
docker run -d --name jenkins \
  -p 8081:8080 -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  --user root \
  jenkins/jenkins:lts
Open : http://localhost:8081
5.2 Unlock Jenkins (Mac)
Get admin password:
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
5.3 Ensure Jenkins can run Docker (inside Jenkins container)
docker exec -it jenkins bash
docker version
If docker: command not found, install Docker CLI in the container:
apt-get update && apt-get install -y docker.io
docker version
exit
5.4 Install AWS CLI inside Jenkins container
docker exec -it jenkins bash
apt-get update
apt-get install -y awscli
aws --version
exit
5.5 Add AWS credentials in Jenkins (UI)
Jenkins UI:
Manage Jenkins → Credentials → (global) → Add Credentials
Kind: Username with password
Username = AWS_ACCESS_KEY_ID
Password = AWS_SECRET_ACCESS_KEY
ID = aws-creds
Create
5.6 Add Jenkinsfile to repo (Mac)
Create file Jenkinsfile in repo root:
Commit & push:
git add Jenkinsfile
git commit -m "Add Jenkins pipeline to build amd64 and push to ECR"
git push origin main
5.7 Create Jenkins Pipeline job (UI)
Jenkins UI:
New Item → Name: ml-api-ci
Type: Pipeline
Pipeline section:
Definition: Pipeline script from SCM
SCM: Git
Repo URL: https://github.com/<YOUR_GITHUB_USERNAME>/ml-api.git
Branch: */main
Save → Build Now
5.8 Verify ECR has Jenkins tags (Mac)
aws ecr describe-images --repository-name ml-model-api --region ap-south-1
You should see:
latest
numeric build tags like 1, 2, etc.

5.9 Deploy latest Jenkins-built image to Windows (manual deploy)
On Windows PowerShell:
docker pull 570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-model-api:latest
Remove existing container: docker rm -f ml-model-api
Run new container: docker run -d --name ml-model-api -p 8080:8080 `
570617927874.dkr.ecr.ap-south-1.amazonaws.com/ml-model-api:latest
Verify: curl http://localhost:8080/health
curl -Method POST http://localhost:8080/predict -ContentType "application/json" -Body '{"x":10}'
Expected:

{"status":"ok"}

{"prediction":21}
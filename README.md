# Cloud Computing (CS3081) Mini Project: Distilled Command Parser

**Author:** Biram Praveen · **Roll:** 230101014 · **Repository:** https://github.com/PRAVEEN1404B/cloud-assignment

A 70M-parameter command parser distilled from the `phi3.5` teacher model, deployed as a containerised microservice on a local k3d Kubernetes cluster, featuring SeaweedFS object storage, Horizontal Pod Autoscaling (HPA), Ingress routing, stateless Network ACLs, automated CI/CD with a quality gate, and empirical performance analysis.

Given a natural language input:
```
set an alarm for seven thirty am tomorrow
```

The service returns:
```json
{"intent": "alarm_set", "date": "tomorrow", "time": "seven thirty am", "place": null}
```

---

## Repository Layout

```
cluster/k3d.yaml              k3d cluster configuration
infra/                        namespaces, SeaweedFS, pipeline config, worker Jobs
prompts/teacher_prompt.txt    Task 1 teacher prompt (temperature 0, seed 42)
worker/                       sharded teacher worker + clean.py + Dockerfile
training/                     config.yaml, train.py, evaluate.py, data helpers
service/                      FastAPI service, strict validation, Dockerfile
k8s/                          Deployment, Service, Ingress, HPA, Secret
network/                      subnets.md, nacl.json, NACL explanation
scripts/                      bring-up, publishing, evaluation, measurement, kaggle
tests/                        24 unit tests used by CI
results/                      measured outputs and evaluation evidence
report/report.md              report source (export to report/report.pdf)
kaggle_train_notebook.ipynb   one-click Kaggle notebook for fast GPU training
AI_USE.md                     AI usage and validation disclosure
make_base_asset.py            generates bad-model asset for quality gate demo
```

---

## Quick Start: Build, Deploy, Test, and Tear Down

### 1. Run Unit Tests & NACL Check
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v   # 24 tests
python3 tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check  # 6/6
```

### 2. Fast GPU Training on Kaggle (Recommended)
Instead of waiting 8–12 hours for CPU labelling and training, run on Kaggle GPU in **~12–15 minutes**:
1. Open [kaggle.com](https://www.kaggle.com) $\to$ **New Notebook** $\to$ Enable **GPU T4 x2** & **Internet On**.
2. Import `kaggle_train_notebook.ipynb` (or follow [KAGGLE_TRAINING_GUIDE.md](KAGGLE_TRAINING_GUIDE.md)).
3. Run the pipeline:
   ```bash
   !pip install -q transformers torch accelerate pyyaml boto3
   !python scripts/kaggle_pipeline.py
   ```
4. Download `cloud_assignment_artifacts.zip` and extract into this repository root:
   ```bash
   unzip -o cloud_assignment_artifacts.zip
   ```

### 3. Local Training on CPU (Alternative)
```bash
# Smoke test (subset=50)
python3 training/train.py --subset 50

# Full runs
python3 training/train.py --version v1
python3 training/train.py --version v2

# Evaluate
python3 training/evaluate.py --model models/v1 --out results/student_v1_predictions.jsonl
python3 tools/score.py results/student_v1_predictions.jsonl data/validation.jsonl --out results/student_v1_validation.json
```

### 4. Deploy Infrastructure (k3d Cluster & SeaweedFS)
```bash
make up
```
This single command:
- Creates the k3d cluster with port forwarding (`8080:80` for Ingress, `8333:8333` for S3).
- Applies `infra/namespaces.yaml`, `infra/seaweedfs.yaml`, and `k8s/secret.yaml`.
- Initialises `datasets` and `models` buckets in SeaweedFS.
- Builds and imports `cloud-parser:dev` and `cloud-parser-worker:dev` images into k3d.
- Deploys the service in namespace `app`.

### 5. Local Docker Container Verification
```bash
# Build image
docker build -t cloud-parser:dev service

# Run with 1 CPU and 1 GiB limit
docker run -d --name parser-test --cpus=1 --memory=1g -p 8000:8000 \
  -e MODEL_URI=/models/v1 -e MODEL_VERSION=v1 -v "$PWD/models:/models:ro" cloud-parser:dev

# Test endpoints
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/readyz
curl -si -X POST http://localhost:8000/parse -H 'Content-Type: application/json' \
  -d '{"text": "set an alarm for seven thirty am tomorrow"}'
```

### 6. Cluster Load Testing & Performance Metrics
```bash
# Run Locust load test through Ingress
make loadtest

# Sample pod CPU & memory metrics
make metrics

# Compute latency, RPS, cost, and SLO budget
make analyse
```

### 7. Tear Down
```bash
make down
```

---

## CI/CD Workflow & Quality Gate
GitHub Actions (`.github/workflows/ci.yml`) runs on every commit:
1. **Linting** (`ruff check`)
2. **Unit Tests** (24 tests via `pytest`)
3. **Model Evaluation** (downloads release asset from GitHub Release `final-models`)
4. **Quality Gate** (`python tools/score.py ... --min-validity 95`)
5. **Secret Scan** (`gitleaks`)

To test the quality gate failure, run the workflow with input `bad_model=true` to test against the untrained base Pythia-70M model.

---

## Licence & Citation
Commands sourced from the SLURP dataset (Bastianelli et al., EMNLP 2020), licensed CC BY 4.0. Teacher responses generated using `phi3.5` (Ollama/Hugging Face, temperature 0, seed 42). Student model is `EleutherAI/pythia-70m` (Apache 2.0).

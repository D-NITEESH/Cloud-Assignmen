# Manual steps

Everything in this repository that can run without a Docker daemon, phi3.5 or a
cluster has already been built and tested: `pytest` passes (24 tests),
`nacl_sim.py --check` reports 6/6, and `clean.py` was exercised on fixtures.
The remaining work needs your machine, your Ollama, and your GitHub account.
Do the phases in order. Save every command's output where indicated; the report
must quote those files.

Time estimate: phase 1 is the long pole (4-12 hours of CPU labelling). Start it
first and do phases 4-6 while it runs.

---

## Phase 0: environment (30-45 min)

```powershell
# Windows: start Docker Desktop and enable the WSL2 backend
wsl --install -d Ubuntu
```

Inside Ubuntu:

```bash
sudo apt update && sudo apt install -y make curl git python3-venv
curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
k3d version && docker version && kubectl version --client
```

Move the repo into the WSL filesystem for build speed (bind mounts from `D:`
are 5-10x slower):

```bash
mkdir -p ~/work && cp -r /mnt/d/assignment/cloud_assignment ~/work/
cd ~/work/cloud_assignment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

Pull the teacher and bind Ollama to all interfaces so Pods can reach it:

```powershell
# Windows PowerShell
setx OLLAMA_HOST 0.0.0.0
# quit Ollama from the tray, then start it again from a new terminal:
ollama serve
ollama pull phi3.5
```

Check from inside the cluster (after `make up`, phase 2):

```bash
kubectl -n pipeline run curl --rm -it --restart=Never --image=curlimages/curl -- \
  curl -s http://host.docker.internal:11434/api/tags
```

If that fails, use the in-cluster teacher instead (needs ~4 GB RAM):

```bash
kubectl apply -f infra/ollama.yaml
kubectl -n pipeline rollout status deploy/ollama
kubectl -n pipeline exec deploy/ollama -- ollama pull phi3.5
kubectl -n pipeline set env deploy/ollama OLLAMA_NUM_PARALLEL=2
# then point workers at it:
kubectl -n pipeline create configmap pipeline-config \
  --from-literal=OLLAMA_URL=http://ollama.pipeline.svc.cluster.local:11434 \
  --from-literal=MODEL=phi3.5 --from-literal=PROMPT_FILE=/app/prompts/teacher_prompt.txt \
  --from-literal=S3_ENDPOINT=http://seaweedfs.data.svc.cluster.local:8333 \
  --from-literal=CHECKPOINT=20 --dry-run=client -o yaml | kubectl apply -f -
```

## Phase 1: cluster, storage, buckets (10 min)

```bash
make up 2>&1 | tee results/task1_up.log
```

Expect a warning that the parser Pods are not Ready: no model exists yet. That
is correct and is the readiness evidence you need.

Capture Task 1 evidence:

```bash
kubectl get nodes -o wide > results/task1_nodes.txt
kubectl -n data get pods,svc,pvc > results/task1_seaweedfs.txt
kubectl -n data get pods -o jsonpath='{.items[0].metadata.name}'   # seaweedfs pod
python3 scripts/init_buckets.py --endpoint http://localhost:8333   # idempotent; prints both buckets
```

Verify the pool object exists and has 3,000 lines:

```bash
kubectl -n data exec deploy/seaweedfs -- sh -c \
  'ls -l /data && echo ok' | tee results/task1_storage.txt
```

Readiness during load (Task 4 mark): delete the parser Pods and watch the
Service having no endpoints while `/readyz` is 503:

```bash
kubectl -n app delete pods -l app=command-parser
kubectl -n app get endpoints command-parser -w > results/task4_no_endpoints.txt &
curl -s -o /dev/null -w 'readyz=%{http_code}\n' http://localhost:8080/readyz \
  | tee results/task4_readyz_during_load.txt
```

## Phase 2: teacher labelling (start now, 4-12 h)

```bash
make workers
kubectl -n pipeline get jobs -w          # four Jobs, SHARD_INDEX 0..3
```

Sanity-check one shard early:

```bash
kubectl -n data exec deploy/seaweedfs -- ls /data 2>/dev/null || true
# From the host, inspect the object through S3:
python3 - <<'PY'
import boto3
s3 = boto3.client("s3", endpoint_url="http://localhost:8333",
                  aws_access_key_id="local-access-key",
                  aws_secret_access_key="local-secret-key", region_name="us-east-1")
for n in (0, 1, 2, 3):
    try:
        body = s3.get_object(Bucket="datasets", Key=f"teacher/shard-{n}.jsonl")["Body"].read()
        print(n, len(body.splitlines()), body.splitlines()[:1])
    except Exception as exc:
        print(n, "not written yet", exc)
PY
```

Wait until all four Jobs report `Complete`:

```bash
kubectl -n pipeline wait --for=condition=Complete job --all --timeout=24h
kubectl -n pipeline get jobs > results/task1_jobs.txt
```

If a Job fails, fix the cause and re-run it; the worker resumes from its object.
If you must change `SHARD_COUNT`, delete the old shard objects first, because a
different count means a different partition.

Then clean:

```bash
make clean 2>&1 | tee results/task1_clean.log
cat results/clean_counts.json | tee results/clean_counts.txt
```

Check the printed line `coverage: pool=3000 ... missing=0 repeated=0`. Copy the
counts table into the report. `train.sha256` must equal the checksum printed
here and the file uploaded to `datasets/v1/`.

## Phase 3: train and evaluate (30-60 min)

Smoke test first (rubric checks that this runs end to end):

```bash
make train SUBSET=50 2>&1 | tee results/train_smoke.log
cat training/loss.csv
```

Then the real runs:

```bash
make train    2>&1 | tee results/train_v1.log
make train-v2 2>&1 | tee results/train_v2.log
cp training/loss.csv results/loss_v1.csv
cp training/loss_v2.csv results/loss_v2.csv
```

Evaluate all three. `make teacher-eval` sends 100 commands to phi3.5 with the
same prompt and settings (about 20-40 minutes on CPU):

```bash
make teacher-eval 2>&1 | tee results/teacher_eval.log
make eval-base    2>&1 | tee results/pythia_base_eval.log
make eval-v1      2>&1 | tee results/student_v1_eval.log
make eval-v2      2>&1 | tee results/student_v2_eval.log
```

Publish v1 and v2 to SeaweedFS and build release tarballs:

```bash
make publish-v1 2>&1 | tee results/publish_v1.log
make publish-v2 2>&1 | tee results/publish_v2.log
kubectl -n app get pods -w       # they retry the download and become Ready
```

Verify the download happened in the Pod logs:

```bash
kubectl -n app logs -l app=command-parser --tail=50 > results/task3_model_download.txt
```

## Phase 4: Task 3 container checks (30 min)

Image size (must be <= 2 GB) and no weights inside:

```bash
docker build -t cloud-parser:dev service
docker image inspect cloud-parser:dev --format '{{.Size}}' | tee results/task3_image_size.txt
docker run --rm --entrypoint sh cloud-parser:dev -c 'ls -l /app; du -sh /app' \
  | tee results/task3_image_contents.txt
```

Run with exactly 1 CPU and 1 GiB, model mounted (no weights in the image):

```bash
docker run -d --name parser-test --cpus=1 --memory=1g -p 8000:8000 \
  -e MODEL_URI=/models/v1 -e MODEL_VERSION=v1 -e S3_ENDPOINT=http://localhost:8333 \
  -e MAX_NEW_TOKENS=96 -v "$PWD/models:/models:ro" cloud-parser:dev

# readiness goes 503 -> 200
for i in $(seq 1 30); do curl -s -o /dev/null -w '%{http_code}\n' localhost:8000/readyz; sleep 2; done \
  | tee results/task3_readiness.txt

# 50 sequential requests
python3 - <<'PY' | tee results/task3_50_requests.txt
import json, time, urllib.request
texts = [json.loads(l)["text"] for l in open("data/validation.jsonl")][:50]
start = time.time()
codes = {}
for text in texts:
    req = urllib.request.Request("http://localhost:8000/parse",
                                 json.dumps({"text": text}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            codes[r.status] = codes.get(r.status, 0) + 1
    except Exception as exc:
        codes[str(exc)] = codes.get(str(exc), 0) + 1
print("codes:", codes, "seconds:", round(time.time() - start, 1))
PY

# peak memory from the cgroup (v2) and a live sample
docker exec parser-test cat /sys/fs/cgroup/memory.peak | tee results/task3_peak_memory.txt
docker stats --no-stream parser-test | tee -a results/task3_peak_memory.txt
```

Headers and fault injection:

```bash
curl -si -X POST localhost:8000/parse -H 'content-type: application/json' \
  -d '{"text":"what is the weather in london"}' | tee results/task3_headers.txt

docker run --rm -p 8001:8000 -e FAULT_INJECT=1 -e MODEL_URI=/models/v1 \
  -e MODEL_VERSION=v1 -v "$PWD/models:/models:ro" cloud-parser:dev &
sleep 20
curl -si -X POST localhost:8001/parse -H 'content-type: application/json' \
  -d '{"text":"hello"}' | tee results/task3_fault_inject.txt   # 500 {"error": "invalid_model_output"}
docker stop parser-test
```

## Phase 5: Task 4 cluster behaviour (1-2 h)

Basics through the Ingress:

```bash
kubectl -n app get deploy,svc,ingress,hpa > results/task4_objects.txt
curl -si -X POST localhost:8080/parse -H 'content-type: application/json' \
  -d '{"text":"set my alarm for six am tomorrow"}' | tee results/task4_ingress.txt
kubectl -n app describe pod -l app=command-parser | grep -A2 -E 'Limits|Requests' \
  > results/task4_resources.txt
```

HPA 1 -> 2 -> 1 (start with one replica so the scale-up is visible):

```bash
kubectl -n app scale deploy/command-parser --replicas=1
kubectl -n app get hpa -w > results/task4_hpa.txt &
locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 3m \
  --host http://localhost:8080 --csv results/loadtest_hpa
# watch: deployment goes 1 -> 2 under load, back to 1 after the 60s stabilisation
```

Rolling update + rollback, while a longer load runs:

```bash
locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 6m \
  --host http://localhost:8080 --csv results/loadtest_rollout & LOADPID=$!

kubectl -n app set env deployment/command-parser MODEL_VERSION=v2 MODEL_URI=s3://models/v2
kubectl -n app rollout status deployment/command-parser --timeout=10m \
  | tee results/task4_rollout.txt

kubectl -n app rollout undo deployment/command-parser
kubectl -n app rollout status deployment/command-parser --timeout=10m \
  | tee results/task4_rollback.txt

curl -si -X POST localhost:8080/parse -H 'content-type: application/json' \
  -d '{"text":"what is the weather in london"}' | tee results/task4_rollback_header.txt
wait $LOADPID
```

Failed-request fraction for the rollout comes from
`results/loadtest_rollout_stats.csv` (failures / requests, must be <= 0.5%).

Recovery, while a load runs:

```bash
locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 3m \
  --host http://localhost:8080 --csv results/loadtest_recovery &
POD=$(kubectl -n app get pods -l app=command-parser -o jsonpath='{.items[0].metadata.name}')
kubectl -n app delete pod "$POD"
kubectl -n app get pods -w > results/task4_recovery_pods.txt &
# after the run, the replica count must be back to 2
```

Cold start three times:

```bash
make coldstart 2>&1 | tee results/task4_coldstart_run.txt
```

Load balancing with exactly two Ready replicas:

```bash
kubectl -n app scale deploy/command-parser --replicas=2
kubectl -n app get pods -l app=command-parser
make loadtest
cat results/loadtest_headers.json | tee results/task4_headers.txt
```

Each replica should have served at least 30% of the successful requests.

## Phase 6: Task 5 (already correct, just re-check)

```bash
python3 tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check \
  | tee results/nacl_check.txt     # 6 / 6
```

The subnet table is in `network/subnets.md`. Nothing else to do.

## Phase 7: GitHub and CI (45 min)

```bash
git init && git add -A && git commit -m "Complete cloud assignment"
gh repo create cloud-assignment --public --source=. --push
```

Create the model release (weights never committed):

```bash
# upload the good assets and the deliberately bad one
ls release/
gh release create final-models release/pythia70m-command-parser-v1.tar.gz \
  release/pythia70m-command-parser-v2.tar.gz \
  --title "Distilled pythia-70m command parser" \
  --notes "v1 and v2 weights, tokenizer and per-file SHA-256 checksums"
make bad-model        # writes release/pythia70m-base-v1.tar.gz
gh release upload final-models release/pythia70m-base-v1.tar.gz
```

Run CI:

```bash
gh workflow run ci.yml
gh run list --workflow ci.yml --limit 3
```

Save the green run URL. Then demonstrate the gate:

```bash
gh workflow run ci.yml -f bad_model=true
gh run list --workflow ci.yml --limit 1 --json databaseId,conclusion,url
```

Save the failing run URL in the report. The gate step fails with
`QUALITY GATE FAILED: json_validity ... < 95.0`.

Secret scan must report nothing: check the `secret-scan` job in the green run.

## Phase 8: Task 7 numbers (20 min)

Run the standard load test while sampling resources:

```bash
bash scripts/watch_metrics.sh 150 & METRICS=$!
make loadtest
wait $METRICS
make analyse 2>&1 | tee results/analyse.txt
```

`results/summary.json` now contains p50/p95, RPS, error rate, cost per 1,000
requests and the SLO error budget. Confirm the error rate is <= 1% and p95 is
<= 5,000 ms; if not, investigate before writing the report.

## Phase 9: report and submission (2-3 h)

1. Fill every `⟨FILL ...⟩` placeholder in `report/report.md` from `results/`.
   Export to `report/report.pdf`, at most 8 pages excluding references and
   appendices. Keep "measured" and "assumed" visually distinct.
2. Write the three-page handwritten reflection, scan it, name it exactly
   `<roll_number>.pdf`. Include your name, roll number, GitHub link, learnings,
   challenges, solutions, findings and improvements.
3. Update `README.md` with your GitHub repo URL.
4. Tag the final commit:

   ```bash
   git add -A && git commit -m "Final submission"
   git tag -a final -m "Final submission"
   git push origin main --tags
   ```

5. Zip everything as `<roll_number>.zip` containing the report, the scanned
   reflection, the source tree, and the README with the repo link.

No commits after the deadline are considered, and only the `final` tag is
assessed.

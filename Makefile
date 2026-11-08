.PHONY: up down workers clean train train-v2 teacher-eval eval-base eval-v1 eval-v2 publish-v1 publish-v2 test lint nacl loadtest metrics analyse coldstart bad-model

SUBSET ?=
K3D_CLUSTER ?= cloud
PYTHON ?= python3

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

lint:
	ruff check . || true

nacl:
	$(PYTHON) tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check

up:
	@echo "=== Bringing up local cloud infrastructure ==="
	k3d cluster create --config cluster/k3d.yaml || true
	kubectl apply -f infra/namespaces.yaml
	kubectl apply -f infra/seaweedfs.yaml
	kubectl apply -f infra/pipeline-config.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/service.yaml
	kubectl apply -f k8s/ingress.yaml
	kubectl apply -f k8s/hpa.yaml
	kubectl -n data rollout status deploy/seaweedfs --timeout=3m
	@echo "Forwarding SeaweedFS S3 to localhost:8333 ..."
	kubectl -n data port-forward svc/seaweedfs 8333:8333 --address 127.0.0.1 >/tmp/seaweedfs-portforward.log 2>&1 &
	sleep 3
	$(PYTHON) scripts/init_buckets.py --endpoint http://localhost:8333
	docker build -t cloud-parser:dev service
	docker build -f worker/Dockerfile -t cloud-parser-worker:dev .
	k3d image import cloud-parser:dev -c $(K3D_CLUSTER)
	k3d image import cloud-parser-worker:dev -c $(K3D_CLUSTER)
	kubectl apply -f k8s/deployment.yaml
	@echo "Note: parser pods will remain in Not Ready state until models/v1 is published."

down:
	k3d cluster delete $(K3D_CLUSTER)

workers:
	kubectl apply -f infra/worker-job.yaml

clean:
	$(PYTHON) worker/clean.py

train:
	@if [ -n "$(SUBSET)" ]; then \
		echo "Running smoke training on subset $(SUBSET)..."; \
		$(PYTHON) training/train.py --subset $(SUBSET); \
	else \
		echo "Running full training v1..."; \
		$(PYTHON) training/train.py --version v1; \
	fi

train-v2:
	echo "Running training v2..."
	$(PYTHON) training/train.py --version v2

teacher-eval:
	mkdir -p results
	$(PYTHON) training/evaluate.py --teacher --out results/teacher_predictions.jsonl
	$(PYTHON) tools/score.py results/teacher_predictions.jsonl data/validation.jsonl --out results/teacher_validation.json

eval-base:
	mkdir -p results
	$(PYTHON) training/evaluate.py --model EleutherAI/pythia-70m --out results/pythia_base_predictions.jsonl
	$(PYTHON) tools/score.py results/pythia_base_predictions.jsonl data/validation.jsonl --out results/pythia_base_validation.json

eval-v1:
	mkdir -p results
	$(PYTHON) training/evaluate.py --model models/v1 --out results/student_v1_predictions.jsonl
	$(PYTHON) tools/score.py results/student_v1_predictions.jsonl data/validation.jsonl --out results/student_v1_validation.json

eval-v2:
	mkdir -p results
	$(PYTHON) training/evaluate.py --model models/v2 --out results/student_v2_predictions.jsonl
	$(PYTHON) tools/score.py results/student_v2_predictions.jsonl data/validation.jsonl --out results/student_v2_validation.json

publish-v1:
	bash scripts/publish.sh v1

publish-v2:
	bash scripts/publish.sh v2

bad-model:
	$(PYTHON) make_base_asset.py

loadtest:
	locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m --host http://localhost:8080 --csv results/loadtest

metrics:
	bash scripts/watch_metrics.sh 150

analyse:
	$(PYTHON) scripts/analyse.py

coldstart:
	bash scripts/coldstart.sh


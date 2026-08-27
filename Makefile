# FinPulse — local development and cluster targets.
#
# Local stack (Airflow + MinIO + Postgres) runs under docker compose.
# The Kubernetes path runs on kind; Chameleon (Terraform/kvm) is the
# production target and uses the same Helm charts and Argo templates.

SHELL := /bin/bash
CLUSTER ?= finpulse
REGISTRY ?= localhost:5001
IMAGE ?= $(REGISTRY)/finpulse-app
TAG ?= dev

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- local (no Kubernetes) ------------------------------------------------

.PHONY: up
up: ## Start Airflow + MinIO + Postgres via docker compose
	docker compose up -d

.PHONY: down
down: ## Stop the compose stack
	docker compose down

.PHONY: train
train: ## Train the LSTM and write model + scaler
	cd train && python lstm_train_pytorch.py

.PHONY: ablation
ablation: ## Run the seeded sentiment ablation
	cd train && python ablation.py --seeds 5

.PHONY: panel
panel: ## Build the multi-ticker training panel (needs the Kaggle tweet corpus)
	python -m pipeline.build_panel --out train/panel.csv --market-dir train/market

.PHONY: train-panel
train-panel: ## Train across the panel's tickers
	cd train && DATA_CSV_PATH=panel.csv python lstm_train_pytorch.py

.PHONY: serve-torch
serve-torch: ## Serve locally with the PyTorch backend (:8000)
	BACKEND=torch MODEL_PATH=train/lstm_model.pth SCALER_PATH=train/scaler.pkl \
	  python -m serving.app

.PHONY: serve-onnx
serve-onnx: ## Serve locally with the ONNX backend (:8000)
	BACKEND=onnx ONNX_MODEL_PATH=App/src/lstm_model.onnx SCALER_PATH=train/scaler.pkl \
	  python -m serving.app

# --- preflight -------------------------------------------------------------

.PHONY: preflight
preflight: ## Check that the cluster toolchain is installed and running
	@missing=0; daemon_down=0; \
	for t in docker kind kubectl helm; do \
	  if command -v $$t >/dev/null 2>&1; then echo "  ok      $$t"; \
	  else echo "  MISSING $$t"; missing=1; fi; done; \
	if command -v docker >/dev/null 2>&1 && ! docker info >/dev/null 2>&1; then \
	  echo "  DOWN    docker daemon"; daemon_down=1; fi; \
	if [ $$missing -ne 0 ]; then \
	  echo ""; echo "Install the missing tools:  brew install kind kubectl helm"; fi; \
	if [ $$daemon_down -ne 0 ]; then \
	  echo ""; echo "Start the Docker daemon:"; \
	  echo "    colima start --cpus 4 --memory 8      # note --cpus, not --cpu"; \
	  echo "    (or launch Docker Desktop)"; fi; \
	if [ $$missing -ne 0 ] || [ $$daemon_down -ne 0 ]; then \
	  echo ""; echo "See docs/RUNBOOK.md"; exit 1; fi; \
	echo "  toolchain ready"

# --- container ------------------------------------------------------------

.PHONY: image
image: preflight ## Build the serving image (ONNX only, ~1 min)
	docker build -t $(IMAGE):$(TAG) --build-arg MODEL_VERSION=$(TAG) .

.PHONY: image-torch
image-torch: preflight ## Build with the torch backend too (~2GB, slow on ARM)
	docker build -t $(IMAGE):$(TAG) --build-arg MODEL_VERSION=$(TAG) \
	  --build-arg INCLUDE_TORCH=true .

.PHONY: image-push
image-push: image ## Push to the local kind registry
	docker push $(IMAGE):$(TAG)

# --- kind cluster ---------------------------------------------------------

.PHONY: cluster-up
cluster-up: preflight ## Create the kind cluster with ArgoCD, Argo Workflows, MLflow, MinIO
	./scripts/kind-up.sh

.PHONY: cluster-down
cluster-down: ## Delete the kind cluster (keeps the registry)
	./scripts/kind-down.sh

.PHONY: workflows
workflows: preflight ## Install the Argo WorkflowTemplates
	kubectl apply -n argo -f workflows/

.PHONY: deploy-staging
deploy-staging: preflight ## Deploy the staging chart to the cluster
	kubectl create namespace finpulse-staging --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install finpulse-staging k8s/staging \
	  --namespace finpulse-staging -f kind/values-kind.yaml \
	  --set image.repository=$(IMAGE) --set image.tag=$(TAG) --wait

.PHONY: smoke
smoke: preflight ## Verify the deployed service returns a real prediction
	./scripts/smoke-test.sh

# --- validation (no cluster required) -------------------------------------

.PHONY: test
test: ## Run the unit and integration tests (no cluster required)
	python -m pytest tests/ -q

.PHONY: lint
lint: ## Static-validate charts, workflows and Python
	@echo "==> helm template"
	@for c in k8s/platform k8s/staging k8s/canary k8s/production; do \
	  helm template "$$c" >/dev/null && echo "    ok  $$c"; done
	@echo "==> python compile"
	@git ls-files '*.py' | grep -v kubespray | xargs -n1 python3 -m py_compile && echo "    ok  all python"
	@echo "==> contract single-sourced"
	@test $$(grep -rn "^FEATURE_COLS = \[" --include="*.py" . | grep -v '\.git' | wc -l) -eq 1 \
	  && echo "    ok  one FEATURE_COLS definition" \
	  || { echo "    FAIL: FEATURE_COLS defined more than once"; exit 1; }
	@echo "==> sequences cut per ticker"
	@! git ls-files 'train/*.py' 'pipeline/*.py' | xargs grep -ln "range(len(data) - seq_length)" \
	  && echo "    ok  no panel-wide sliding window" \
	  || { echo "    FAIL: a sliding window ignores ticker boundaries"; exit 1; }

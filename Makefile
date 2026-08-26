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

.PHONY: serve-torch
serve-torch: ## Serve locally with the PyTorch backend (:8000)
	BACKEND=torch MODEL_PATH=train/lstm_model.pth SCALER_PATH=train/scaler.pkl \
	  python -m serving.app

.PHONY: serve-onnx
serve-onnx: ## Serve locally with the ONNX backend (:8000)
	BACKEND=onnx ONNX_MODEL_PATH=App/src/lstm_model.onnx SCALER_PATH=train/scaler.pkl \
	  python -m serving.app

# --- container ------------------------------------------------------------

.PHONY: image
image: ## Build the serving image
	docker build -t $(IMAGE):$(TAG) --build-arg MODEL_VERSION=$(TAG) .

.PHONY: image-push
image-push: image ## Push to the local kind registry
	docker push $(IMAGE):$(TAG)

# --- kind cluster ---------------------------------------------------------

.PHONY: cluster-up
cluster-up: ## Create the kind cluster with ArgoCD, Argo Workflows, MLflow, MinIO
	./scripts/kind-up.sh

.PHONY: cluster-down
cluster-down: ## Delete the kind cluster (keeps the registry)
	./scripts/kind-down.sh

.PHONY: workflows
workflows: ## Install the Argo WorkflowTemplates
	kubectl apply -n argo -f workflows/

.PHONY: deploy-staging
deploy-staging: ## Deploy the staging chart to the cluster
	kubectl create namespace finpulse-staging --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install finpulse-staging k8s/staging \
	  --namespace finpulse-staging -f kind/values-kind.yaml \
	  --set image.repository=$(IMAGE) --set image.tag=$(TAG) --wait

.PHONY: smoke
smoke: ## Verify the deployed service returns a real prediction
	./scripts/smoke-test.sh

# --- validation (no cluster required) -------------------------------------

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

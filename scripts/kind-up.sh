#!/usr/bin/env bash
# Bring up the local FinPulse cluster: kind + a local registry + platform
# services. Idempotent - safe to re-run.
set -euo pipefail

CLUSTER="${CLUSTER:-finpulse}"
REG_NAME="${REG_NAME:-kind-registry}"
REG_PORT="${REG_PORT:-5001}"
ARGOCD_VERSION="${ARGOCD_VERSION:-v2.13.2}"
ARGO_WF_VERSION="${ARGO_WF_VERSION:-v3.6.5}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not found. See docs/local-cluster.md"; exit 1; }; }
need kind; need kubectl; need docker; need helm

# kind plus ArgoCD, Argo Workflows, MLflow, MinIO and Postgres needs real
# memory. Under ~6 GiB the API server gets starved and kubectl times out
# mid-apply, which looks like a manifest problem but is not.
# A full VM disk surfaces as "input/output error" writing containerd's
# meta.db, which reads like corruption rather than a capacity problem.
if command -v colima >/dev/null 2>&1; then
  DISK_PCT=$(colima ssh -- df --output=pcent /var/lib 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
  if [ -n "$DISK_PCT" ] && [ "$DISK_PCT" -ge 85 ] 2>/dev/null; then
    echo "WARNING: the Docker VM disk is ${DISK_PCT}% full."
    echo "         docker system prune -af --volumes"
    echo "         or: colima delete && colima start --cpus 4 --memory 8 --disk 60"
    echo ""
  fi
fi

MEM_BYTES=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)
MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))
if [ "$MEM_GB" -gt 0 ] && [ "$MEM_GB" -lt 6 ]; then
  echo "WARNING: the Docker VM reports ${MEM_GB} GiB. 8 GiB is the practical floor."
  echo "         colima: colima stop && colima start --cpus 4 --memory 8"
  echo "         Close memory-heavy apps first - host swap will starve the VM too."
  echo ""
fi

# --- local registry -------------------------------------------------------
# Argo builds images with Kaniko and pushes them somewhere the cluster can
# pull from; a registry container on the kind network plays that role.
if [ "$(docker inspect -f '{{.State.Running}}' "$REG_NAME" 2>/dev/null || true)" != "true" ]; then
  echo "==> starting local registry on :$REG_PORT"
  docker run -d --restart=always -p "127.0.0.1:$REG_PORT:5000" --name "$REG_NAME" registry:2
else
  echo "==> registry $REG_NAME already running"
fi

# --- cluster --------------------------------------------------------------
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "==> kind cluster '$CLUSTER' already exists"
else
  echo "==> creating kind cluster '$CLUSTER'"
  kind create cluster --name "$CLUSTER" --config "$ROOT/kind/kind-config.yaml"
fi

# Join the registry to the kind network so nodes can resolve it by name.
if ! docker network inspect kind | grep -q "\"$REG_NAME\""; then
  docker network connect kind "$REG_NAME" 2>/dev/null || true
fi

# The API server accepts connections before it is ready to serve heavy
# requests. Applying immediately is what produces "the server was unable to
# return a response in the time allotted".
echo "==> waiting for the control plane"
kubectl wait --for=condition=ready node --all --timeout=180s

# Advertise the registry per the kind local-registry convention.
kubectl apply -f - <<REGHOSTING
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REG_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
REGHOSTING

# --- platform services ----------------------------------------------------
echo "==> installing ArgoCD"
kubectl create namespace argocd --dry-run=client -o yaml | kubectl apply -f -
# --server-side avoids the client-side merge, which GETs every object first
# and is what times out on a memory-constrained cluster. ARGOCD_VERSION is
# pinned rather than "stable" so a run is reproducible.
kubectl apply -n argocd --server-side --force-conflicts --request-timeout=180s \
  -f "https://raw.githubusercontent.com/argoproj/argo-cd/${ARGOCD_VERSION}/manifests/install.yaml"
kubectl wait --for=condition=available deployment/argocd-server -n argocd --timeout=300s
kubectl patch svc argocd-server -n argocd -p \
  '{"spec":{"type":"NodePort","ports":[{"port":80,"targetPort":8080,"nodePort":30080,"name":"http"}]}}'

echo "==> installing Argo Workflows"
kubectl create namespace argo --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argo --server-side --force-conflicts --request-timeout=180s \
  -f "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_WF_VERSION}/quick-start-minimal.yaml"
kubectl patch svc argo-server -n argo -p \
  '{"spec":{"type":"NodePort","ports":[{"port":2746,"targetPort":2746,"nodePort":30081,"name":"web"}]}}'

echo "==> installing platform chart (MLflow + MinIO + Postgres)"
kubectl create namespace finpulse-platform --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic minio-credentials -n finpulse-platform \
  --from-literal=accesskey="${MINIO_ROOT_USER:-minioadmin}" \
  --from-literal=secretkey="${MINIO_ROOT_PASSWORD:?set MINIO_ROOT_PASSWORD}" \
  --dry-run=client -o yaml | kubectl apply -f -
helm upgrade --install finpulse-platform "$ROOT/k8s/platform" \
  --namespace finpulse-platform -f "$ROOT/kind/values-kind.yaml" --wait --timeout 10m

echo "==> waiting for platform pods"
kubectl wait --for=condition=ready pod --all -n finpulse-platform --timeout=300s || true

cat <<DONE

Cluster '$CLUSTER' is up.

  ArgoCD          http://localhost:8080
    password      kubectl -n argocd get secret argocd-initial-admin-secret \\
                    -o jsonpath="{.data.password}" | base64 -d
  Argo Workflows  http://localhost:2746
  MLflow          http://localhost:5000
  MinIO console   http://localhost:9001
  Registry        localhost:$REG_PORT

Next:  make deploy-staging      then  make smoke
Down:  make cluster-down
DONE

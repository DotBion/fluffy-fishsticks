#!/usr/bin/env bash
# Tear down the local cluster. The registry is kept by default so cached
# layers survive; pass --all to remove it too.
set -euo pipefail

CLUSTER="${CLUSTER:-finpulse}"
REG_NAME="${REG_NAME:-kind-registry}"

kind delete cluster --name "$CLUSTER" || true

if [ "${1:-}" = "--all" ]; then
  docker rm -f "$REG_NAME" 2>/dev/null || true
  echo "removed registry $REG_NAME"
else
  echo "registry $REG_NAME left running (pass --all to remove)"
fi

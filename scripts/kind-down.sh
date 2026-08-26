#!/usr/bin/env bash
# Tear down the local cluster. The registry is kept by default so cached
# layers survive; pass --all to remove it too.
#
# kind delete can fail outright when the Docker VM is unhealthy: writes to
# containerd's meta.db return "input/output error". That reads like disk
# corruption but is more often the host paging heavily enough that writes to
# the VM's virtual disk time out - it has been observed on a disk only 7%
# full. Fall back to removing the node containers directly.
set -uo pipefail

CLUSTER="${CLUSTER:-finpulse}"
REG_NAME="${REG_NAME:-kind-registry}"

echo "==> deleting cluster '$CLUSTER'"
if ! kind delete cluster --name "$CLUSTER"; then
  echo "==> kind delete failed; removing node containers directly"
  NODES=$(docker ps -a --filter "name=^${CLUSTER}-" --format '{{.Names}}' 2>/dev/null || true)
  for n in $NODES; do
    docker rm -f -v "$n" >/dev/null 2>&1 && echo "    removed $n" || echo "    STUCK   $n"
  done

  if docker ps -a --filter "name=^${CLUSTER}-" --format '{{.Names}}' | grep -q .; then
    cat <<'STUCK'

Containers could not be removed. Check, in this order:

  1. Host disk. The VM's disk is a sparse file on the host; when the host
     fills, writes inside the VM fail with I/O errors even though the guest
     reports free space:
         df -h /
         du -sh ~/.colima ~/Library/Caches 2>/dev/null

  2. VM disk, which is usually NOT the problem:
         colima ssh -- df -h /var/lib
         docker system df        # a NEGATIVE reclaimable size means the
                                 # containerd metadata is inconsistent
         docker system prune -af --volumes

  3. Rebuild the VM - last resort, destroys cached images - and only if the
     errors persist on a disk with free space:
         colima delete && colima start --cpus 4 --memory 8 --disk 60
STUCK
    exit 1
  fi
fi

if [ "${1:-}" = "--all" ]; then
  docker rm -f "$REG_NAME" >/dev/null 2>&1 && echo "removed registry $REG_NAME"
else
  echo "registry $REG_NAME left running (pass --all to remove)"
fi

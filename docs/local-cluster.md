# Local cluster (kind)

Chameleon (`Terraform/kvm/`) is the production target. This runs the same
Helm charts and Argo templates on a laptop so the CD path can actually be
exercised rather than only statically validated.

## Prerequisites

| tool | install |
| --- | --- |
| Docker | Docker Desktop, or `brew install colima docker` |
| kind | `brew install kind` |
| kubectl | `brew install kubectl` |
| helm | `brew install helm` |

    brew install kind kubectl helm

Check everything at once:

    make preflight

It reports each missing tool and whether the Docker daemon is actually
running — a Docker CLI on PATH does not mean the daemon is up. With colima:

    colima start --cpus 4 --memory 8

8 GB is not excessive: kind plus ArgoCD, Argo Workflows, MLflow, MinIO and
Postgres will not fit comfortably in less.

**Host memory matters as much as VM memory.** If macOS is already swapping
heavily (check Activity Monitor -> Memory -> Swap Used), the colima VM gets
starved no matter what you allocated, and the symptom is confusing:

    Error from server (Timeout): error when retrieving current configuration of:
    ... the server was unable to return a response in the time allotted

That is the Kubernetes API server too slow to answer, not a manifest problem.

A different failure mode looks like corruption but is usually capacity:

    Error response from daemon: cannot remove container "finpulse-control-plane":
    write /var/lib/containerd/io.containerd.metadata.v1.bolt/meta.db: input/output error

This looks like disk corruption and is usually the *host* disk being full.
The VM's disk is a sparse file on the host; when the host fills, that file
cannot grow, so writes inside the VM fail while the guest's own `df` still
reports free space - observed here with the guest at 7% used. The same
condition later surfaced as `no space left on device` writing
`~/.colima/default/colima.yaml`, which named the real problem directly.

Check `df -h /` on the Mac before investigating anything inside the VM.

Check `docker system df` too - a *negative* reclaimable size means containerd's
metadata accounting is inconsistent. Only if the disk is genuinely near full
does `docker system prune -af --volumes` help, and rebuilding the VM with
`colima delete` is a last resort that costs you every cached image.
Close browsers and editors before bringing the cluster up. `kind-up.sh` warns
when the Docker VM reports under 6 GiB, but it cannot see host swap.

## A note for macOS

macOS filesystems are case-insensitive but git is not. This repo keeps a
single lowercase `scripts/` directory for exactly that reason - an earlier
`Scripts/` (capital S) meant files extracted into `scripts/` were committed
under the other spelling and broke on Linux while appearing fine locally.

Do not paste shell comments onto a command line in zsh: `INTERACTIVE_COMMENTS`
is off by default in interactive zsh, so `cmd --flag 4  # note` passes the
comment as arguments.

## Bring it up

    export MINIO_ROOT_PASSWORD=some-dev-password
    make cluster-up

That creates a 3-node cluster (1 control-plane, 2 workers — mirroring the
Chameleon topology), starts a local registry, and installs ArgoCD, Argo
Workflows, MLflow, MinIO and Postgres.

| service | URL |
| --- | --- |
| ArgoCD | http://localhost:8080 |
| Argo Workflows | http://localhost:2746 |
| MLflow | http://localhost:5000 |
| MinIO console | http://localhost:9001 |
| finpulse-app (staging) | http://localhost:8000 |

ArgoCD admin password:

    kubectl -n argocd get secret argocd-initial-admin-secret \
      -o jsonpath="{.data.password}" | base64 -d

## Deploy and verify

    make image-push          # build and push to the local registry
    make deploy-staging      # helm install with the kind overlay
    make smoke               # asserts a real prediction comes back

`make smoke` is the check that matters: it port-forwards the service, posts a
real 10-day window, and fails unless the prediction lands in a plausible price
range and `/metrics` is exposed. Applying manifests successfully is not the
same as the service working.

## Run the CD workflows

    make workflows                                   # install WorkflowTemplates
    argo submit -n argo --from workflowtemplate/train-model \
      -p endpoint-ip=<training-service-host>

## Tear down

    make cluster-down          # keeps the registry and its cached layers
    ./scripts/kind-down.sh --all   # removes the registry too

## How this differs from Chameleon

| | kind | Chameleon |
| --- | --- | --- |
| service exposure | NodePort, mapped to localhost | ClusterIP + `externalIPs` on the floating IP |
| registry | `localhost:5001` container | in-cluster `registry.kube-system.svc` |
| provisioning | `kind create cluster` | Terraform + Kubespray |
| values | `kind/values-kind.yaml` | chart defaults |

The charts carry both paths behind `serviceType`, so neither environment
needs a forked manifest.

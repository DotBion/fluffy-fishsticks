# Local cluster (kind)

Chameleon (`Terraform/kvm/`) is the production target. This runs the same
Helm charts and Argo templates on a laptop so the CD path can actually be
exercised rather than only statically validated.

## Prerequisites

| tool | install |
| --- | --- |
| Docker | https://docs.docker.com/get-docker/ |
| kind | `go install sigs.k8s.io/kind@v0.23.0` or `brew install kind` |
| kubectl | `brew install kubectl` |
| helm | `brew install helm` |

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

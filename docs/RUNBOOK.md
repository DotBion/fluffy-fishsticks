# Runbook

Three ways to run this project, in increasing order of cost. Pick the
cheapest one that answers your question.

| | what it proves | needs | time |
| --- | --- | --- | --- |
| 1. Python only | the model trains and serves correctly | python + pip | ~2 min |
| 2. Docker Compose | the data pipeline runs end to end | + docker | ~10 min |
| 3. kind cluster | the CD pipeline actually deploys | + kind, kubectl, helm, 8 GB | ~20 min |

## Why the kind cluster exists

The README describes train -> register -> build -> deploy through
staging/canary/production via ArgoCD and Argo Workflows. That was written for
a Chameleon Cloud cluster that no longer exists, so none of it has ever run.
The Helm charts and Argo templates are valid YAML that has never been
executed.

kind is a local stand-in for Chameleon: same charts, same workflows, running
somewhere real. It exists to convert "config that looks correct" into "config
that demonstrably works". Chameleon (`Terraform/kvm/`) remains the production
target; `kind/values-kind.yaml` is the only difference between them.

---

## 1. Python only

    python -m venv .venv && source .venv/bin/activate
    pip install -r train/requirements.txt

    make train        # writes train/lstm_model.pth and train/scaler.pkl
    make ablation     # 5 seeds per arm, prints mean +/- std

    make serve-torch  # :8000, PyTorch backend
    # or
    make serve-onnx   # :8000, ONNX backend

Check it:

    curl -s localhost:8000/health | python -m json.tool

    python - <<'PY'
    import csv, json, urllib.request
    rows = sorted(csv.DictReader(open("train/data_2018.csv")), key=lambda r: r["date"])
    cols = ["open","high","low","close","volume","daily_avg_sentiment_score"]
    body = json.dumps({"input": [[[float(r[c]) for c in cols] for r in rows[-10:]]]}).encode()
    req = urllib.request.Request("http://localhost:8000/predict", body,
                                 {"Content-Type": "application/json"})
    print(json.load(urllib.request.urlopen(req)))
    PY

Expect a prediction near $160 against a last actual close of $157.74.

## 2. Docker Compose

    cp .env.example .env        # fill in the keys you have
    make up                     # Airflow + MinIO + Postgres

- Airflow http://localhost:8080
- MinIO   http://localhost:9001

Trigger `stock_sentiment_etl` from the Airflow UI. It fetches OHLCV, scores
tweet sentiment with the finance-augmented VADER lexicon, joins them, and
uploads the training set to MinIO. Training then reads from MinIO when
`MINIO_BUCKET`, `MINIO_TRAINING_OBJECT` and `MINIO_ENDPOINT` are set.

    make down

## 3. kind cluster

    brew install kind kubectl helm
    colima start --cpus 4 --memory 8      # or Docker Desktop
    make preflight                        # expect "toolchain ready"

    export MINIO_ROOT_PASSWORD=some-dev-password
    make cluster-up                       # 8-15 min on first run

| service | URL |
| --- | --- |
| ArgoCD | http://localhost:8080 |
| Argo Workflows | http://localhost:2746 |
| MLflow | http://localhost:5000 |
| MinIO console | http://localhost:9001 |
| finpulse-app | http://localhost:8000 |

Deploy and verify:

    make image-push       # ONNX-only image, ~1 min
    make deploy-staging
    make smoke            # fails unless a real prediction comes back

`make smoke` is the only step that distinguishes "manifests applied" from
"the service works". It posts a real 10-day window and asserts the prediction
lands in a plausible price range and that /metrics is exposed.

Teardown:

    make cluster-down            # keeps the registry cache
    ./scripts/kind-down.sh --all # removes it too

---

## Troubleshooting

**`make: *** No rule to make target`** - you are not in the repo root.

**`Error from server (Timeout) ... retrieving current configuration`** - the
API server is starved, not a manifest problem. Free host memory and restart
the VM. Check Activity Monitor -> Memory -> Swap Used; heavy swapping starves
the VM whatever you allocated.

**`input/output error` writing `meta.db`**, or **`no space left on device`
writing `~/.colima/default/colima.yaml`** - the *host* disk is full, not the
VM's.

The VM's disk is a sparse file on the host. When the host fills, that file
cannot grow, so writes inside the VM fail with I/O errors while the guest's
own `df` still reports plenty free - it has free blocks it cannot
materialise. Observed here: guest at 7% used, host full. A *negative*
reclaimable size in `docker system df` is the same failure showing up in
containerd's accounting.

Check the host first, not the guest:

    df -h /
    du -sh ~/.colima ~/Library/Caches ~/Downloads 2>/dev/null | sort -h

`~/.colima` is usually the largest item and never shrinks on its own.
`colima delete` reclaims all of it; the VM and its cached images are
re-created on the next `colima start`.

**pip install takes 20+ minutes** - you are building the torch image. `make
image` is ONNX-only (~380 MB of deps); `make image-torch` is the ~5.7 GB one
and is rarely needed, since `make serve-torch` runs that backend directly.

**A cluster step fails on a low-memory machine** - drop to a single node by
deleting the `- role: worker` entry from `kind/kind-config.yaml`, then
`make cluster-down && make cluster-up`.

## What has and has not been verified

Verified: both backends serve identical predictions to six decimal places;
the ONNX-only dependency set serves correctly with torch not installed;
training runs end to end; MLflow logs and registers FinPulseLSTM; every chart
renders and parses under both serviceType branches; all five DAG callables
resolve.

Not verified: the image build and a live cluster deploy. Those need a machine
with a working container runtime and registry access, which is exactly what
step 3 is for.

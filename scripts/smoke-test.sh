#!/usr/bin/env bash
# Verify the deployed service actually serves a prediction in the cluster.
# This is the check that distinguishes "manifests applied" from "it works".
set -euo pipefail

NS="${NS:-finpulse-staging}"
SVC="${SVC:-finpulse-app}"
PORT="${PORT:-8000}"

echo "==> waiting for $SVC in $NS"
kubectl wait --for=condition=available deployment/"$SVC" -n "$NS" --timeout=180s

kubectl port-forward -n "$NS" "svc/$SVC" 18000:"$PORT" >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
for _ in $(seq 30); do curl -sf http://localhost:18000/health >/dev/null 2>&1 && break; sleep 1; done

echo "==> /health"
curl -sf http://localhost:18000/health | tee /dev/stderr | grep -q '"status": *"ok"' \
  || { echo "FAIL: health not ok"; exit 1; }

echo "==> /predict with a real 10-day window"
WINDOW=$(python3 - <<'PY'
import csv, json
rows = sorted(csv.DictReader(open("train/data_2018.csv")), key=lambda r: r["date"])
cols = ["open","high","low","close","volume","daily_avg_sentiment_score"]
print(json.dumps({"input": [[[float(r[c]) for c in cols] for r in rows[-10:]]]}))
PY
)
RESP=$(curl -sf -X POST http://localhost:18000/predict -H 'Content-Type: application/json' -d "$WINDOW")
echo "$RESP"
python3 -c "
import json,sys
p = json.loads('''$RESP''')['predictions'][0]
assert 50 < p < 500, f'prediction \${p} outside a plausible price range'
print(f'==> PASS: predicted \${p:.2f}')
"

echo "==> /metrics"
curl -sf http://localhost:18000/metrics | grep -q lstm_predictions_total \
  || { echo "FAIL: metrics missing"; exit 1; }
echo "==> PASS: metrics exposed"

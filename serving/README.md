# serving/

One inference service, two backends.

    BACKEND=torch python -m serving.app     # local debugging, port 8000
    BACKEND=onnx  python -m serving.app     # deployment path

| file | role |
| --- | --- |
| `contract.py` | **single source of truth** for `FEATURE_COLS`, `SEQ_LENGTH`, scaler load/apply and window validation |
| `backends/torch_backend.py` | loads the state_dict training produced; also defines `LSTMModel` |
| `backends/onnx_backend.py` | ONNX Runtime with graph optimizations; CUDA when available |
| `app.py` | the Flask app, one metrics implementation shared by both backends |

`train/models.py` re-exports the contract, so training and serving cannot
drift: change the feature order or window in `contract.py` and both follow.

## Endpoints

- `POST /predict` — raw, unscaled `(batch, 10, 6)` window; returns a price
- `GET /health` — status, active backend, feature order, sequence length
- `GET /metrics` — Prometheus: request outcome, latency, predicted-price
  distribution, all labelled by backend

## Ports

| service | port |
| --- | --- |
| LSTM serving (this) | 8000 |
| FinBERT | 8001 |
| training trigger | 9090 |
| orchestrator | 8081 |

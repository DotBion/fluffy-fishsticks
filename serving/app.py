"""LSTM inference service.

One server, one metrics implementation, either backend:

    BACKEND=torch python -m serving.app     # local debugging
    BACKEND=onnx  python -m serving.app     # deployment

Accepts RAW (unscaled) OHLCV + sentiment windows and returns a predicted
next-day close in price units. Scaling and inverse scaling happen here using
the scaler persisted at training time, so callers never handle normalized
values.
"""

import os

import numpy as np
from flask import Flask, Response, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .backends import load_backend
from .contract import (
    FEATURE_COLS,
    SEQ_LENGTH,
    inverse_close,
    load_scaler,
    scale_window,
    validate_window,
)

app = Flask(__name__)

def _metric(cls, name, *args, **kwargs):
    """Create a metric, reusing it if already registered.

    The default registry is process-global, so re-importing this module (as a
    parametrized test across backends does) would otherwise raise
    DuplicateTimeseries.
    """
    try:
        return cls(name, *args, **kwargs)
    except ValueError:
        from prometheus_client import REGISTRY

        existing = REGISTRY._names_to_collectors.get(name)
        if existing is None:
            raise
        return existing


PREDICTIONS = _metric(Counter, "lstm_predictions_total", "Prediction requests", ["outcome", "backend"])
LATENCY = _metric(Histogram, "lstm_inference_seconds", "End-to-end inference latency", ["backend"])
PREDICTED_CLOSE = _metric(
    Histogram,
    "lstm_predicted_close_dollars",
    "Distribution of predicted close prices",
    buckets=(50, 100, 150, 175, 200, 225, 250, 300, 500),
)

backend = load_backend()
scaler = load_scaler()


@app.route("/")
def home():
    return f"FinPulse LSTM inference server ({backend.name} backend) is running!"


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "features": FEATURE_COLS,
        "seq_length": SEQ_LENGTH,
        **backend.describe(),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if "input" not in payload:
            PREDICTIONS.labels(outcome="bad_request", backend=backend.name).inc()
            return jsonify({"error": "Missing 'input' key in JSON payload"}), 400

        data = np.array(payload["input"], dtype=np.float32)
        error = validate_window(data)
        if error:
            PREDICTIONS.labels(outcome="bad_request", backend=backend.name).inc()
            return jsonify({"error": error}), 400

        with LATENCY.labels(backend=backend.name).time():
            scaled = scale_window(scaler, data)
            preds = inverse_close(scaler, backend.predict_scaled(scaled))

        for p in preds:
            PREDICTED_CLOSE.observe(float(p))
        PREDICTIONS.labels(outcome="success", backend=backend.name).inc()

        return jsonify({"predictions": preds.tolist(), "backend": backend.name}), 200

    except Exception as e:
        PREDICTIONS.labels(outcome="error", backend=backend.name).inc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Production: gunicorn -w 2 -b 0.0.0.0:8000 serving.app:app
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

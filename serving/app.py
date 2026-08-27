"""LSTM inference service.

One server, one metrics implementation, either backend:

    BACKEND=torch python -m serving.app     # local debugging
    BACKEND=onnx  python -m serving.app     # deployment

Accepts RAW (unscaled) OHLCV + sentiment windows and returns a predicted
next-day close in price units. Scaling and inverse scaling happen here using
the scaler persisted at training time, so callers never handle normalized
values.

A multi-ticker model carries one scaler per symbol, so the request may name
a `ticker`. Single-ticker models ignore the field, which keeps every caller
written before per-ticker scaling working unchanged.
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


PREDICTIONS = _metric(Counter, "lstm_predictions_total", "Prediction requests", ["outcome", "backend", "ticker"])
LATENCY = _metric(Histogram, "lstm_inference_seconds", "End-to-end inference latency", ["backend"])
# Bucket edges span the panel: MSFT traded near $95 in 2018 and AMZN near
# $1,500, so the single-ticker AAPL range would put every Amazon prediction
# in the overflow bucket.
PREDICTED_CLOSE = _metric(
    Histogram,
    "lstm_predicted_close_dollars",
    "Distribution of predicted close prices",
    ["ticker"],
    buckets=(50, 100, 150, 200, 300, 500, 800, 1200, 1800, 3000),
)

backend = load_backend()
scalers = load_scaler()


def _label(ticker):
    """Collapse the ticker to a bounded label.

    The value comes straight off the request body, so labelling with it
    directly would let any caller mint a new Prometheus time series per
    request. Only tickers the loaded model actually covers get their own
    series.
    """
    if not ticker:
        return "default"
    return ticker if ticker in scalers.tickers else "unknown"


def _count(outcome, ticker):
    """Increment the request counter."""
    PREDICTIONS.labels(
        outcome=outcome, backend=backend.name, ticker=_label(ticker)
    ).inc()


@app.route("/")
def home():
    return f"FinPulse LSTM inference server ({backend.name} backend) is running!"


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "features": FEATURE_COLS,
        "seq_length": SEQ_LENGTH,
        "tickers": scalers.tickers,
        "multi_ticker": scalers.is_multi_ticker,
        **backend.describe(),
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    ticker = None
    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            _count("bad_request", None)
            return jsonify({"error": "Body must be a JSON object"}), 400

        ticker = (payload.get("ticker") or "").upper() or None

        if "input" not in payload:
            _count("bad_request", ticker)
            return jsonify({"error": "Missing 'input' key in JSON payload"}), 400

        data = np.array(payload["input"], dtype=np.float32)
        error = validate_window(data)
        if error:
            _count("bad_request", ticker)
            return jsonify({"error": error}), 400

        # An unknown or missing ticker is the caller's mistake, not a server
        # fault: answer 400 with the list of tickers this model covers rather
        # than 500 with a stack trace.
        try:
            scaler = scalers.for_ticker(ticker)
        except KeyError as e:
            _count("bad_request", ticker)
            return jsonify({"error": str(e.args[0])}), 400

        with LATENCY.labels(backend=backend.name).time():
            scaled = scale_window(scaler, data)
            preds = inverse_close(scaler, backend.predict_scaled(scaled))

        for p in preds:
            PREDICTED_CLOSE.labels(ticker=_label(ticker)).observe(float(p))
        _count("success", ticker)

        return jsonify({
            "predictions": preds.tolist(),
            "backend": backend.name,
            "ticker": ticker,
        }), 200

    except Exception as e:
        _count("error", ticker)
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Production: gunicorn -w 2 -b 0.0.0.0:8000 serving.app:app
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

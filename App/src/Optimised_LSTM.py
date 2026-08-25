"""ONNX Runtime inference server for the LSTM, with Prometheus metrics.

Serves the quantized ONNX export produced by Onnx_export.py. Accepts RAW
(unscaled) windows and returns a predicted next-day close in price units,
matching the contract of train/inference_server_lstm.py.
"""

import os
import time

import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, request
from joblib import load
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

app = Flask(__name__)

MODEL_PATH = os.getenv("ONNX_MODEL_PATH", "lstm_model.onnx")
SCALER_PATH = os.getenv("SCALER_PATH", "../../train/scaler.pkl")

FEATURE_COLS = ["open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]
INPUT_SIZE = len(FEATURE_COLS)
TARGET_IDX = FEATURE_COLS.index("close")
SEQ_LENGTH = 10

# --- Prometheus metrics ---
PREDICTIONS = Counter("lstm_predictions_total", "Prediction requests", ["outcome"])
LATENCY = Histogram("lstm_inference_seconds", "End-to-end inference latency")
PREDICTED_CLOSE = Histogram(
    "lstm_predicted_close_dollars",
    "Distribution of predicted close prices",
    buckets=(50, 100, 150, 175, 200, 225, 250, 300, 500),
)

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# CUDAExecutionProvider is silently skipped when onnxruntime has no GPU build.
providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
             if p in ort.get_available_providers()]
ort_session = ort.InferenceSession(MODEL_PATH, sess_options, providers=providers)
INPUT_NAME = ort_session.get_inputs()[0].name

if not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        f"Scaler not found at {SCALER_PATH}. Run 'python train/lstm_train_pytorch.py' first — "
        "predictions cannot be mapped back to prices without it."
    )
scaler = load(SCALER_PATH)


def _inverse_close(scaled_close):
    padded = np.zeros((len(scaled_close), INPUT_SIZE))
    padded[:, TARGET_IDX] = scaled_close
    return scaler.inverse_transform(padded)[:, TARGET_IDX]


@app.route("/")
def home():
    return "ONNX-Optimized LSTM Server is running!"


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_PATH,
        "providers": ort_session.get_providers(),
        "features": FEATURE_COLS,
        "seq_length": SEQ_LENGTH,
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if "input" not in payload:
            PREDICTIONS.labels(outcome="bad_request").inc()
            return jsonify({"error": "Missing 'input' key in JSON payload"}), 400

        data = np.array(payload["input"], dtype=np.float32)
        if data.ndim != 3 or data.shape[2] != INPUT_SIZE:
            PREDICTIONS.labels(outcome="bad_request").inc()
            return jsonify({
                "error": f"Expected shape (batch, seq_len, {INPUT_SIZE}) "
                         f"with features {FEATURE_COLS}, got {list(data.shape)}"
            }), 400

        with LATENCY.time():
            batch, seq_len, n_features = data.shape
            scaled = scaler.transform(data.reshape(-1, n_features)).reshape(batch, seq_len, n_features)
            outputs = ort_session.run(None, {INPUT_NAME: scaled.astype(np.float32)})
            preds = _inverse_close(np.atleast_1d(outputs[0]).astype(np.float32).ravel())

        for p in preds:
            PREDICTED_CLOSE.observe(float(p))
        PREDICTIONS.labels(outcome="success").inc()

        return jsonify({"predictions": preds.tolist()}), 200

    except Exception as e:
        PREDICTIONS.labels(outcome="error").inc()
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    # Production: gunicorn -w 4 -b 0.0.0.0:9090 Optimised_LSTM:app
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "9090")))

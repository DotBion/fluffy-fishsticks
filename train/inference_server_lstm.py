"""Flask inference server for the sentiment-aware LSTM.

Accepts RAW (unscaled) OHLCV + sentiment windows and returns a predicted
next-day close in the same units as the training data. Scaling and inverse
scaling are applied here using the scaler persisted at training time.
"""

import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from joblib import load

from models import DEFAULTS, FEATURE_COLS, TARGET_IDX, LSTMModel

app = Flask(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "lstm_model.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

INPUT_SIZE = len(FEATURE_COLS)
SEQ_LENGTH = DEFAULTS["seq_length"]

model = LSTMModel(
    input_size=INPUT_SIZE,
    hidden_dim=DEFAULTS["hidden_dim"],
    num_layers=DEFAULTS["num_layers"],
    dropout=DEFAULTS["dropout"],
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

if not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        f"Scaler not found at {SCALER_PATH}. Run 'python lstm_train_pytorch.py' first — "
        "predictions cannot be mapped back to prices without it."
    )
scaler = load(SCALER_PATH)


def _inverse_close(scaled_close):
    """Map scaled close values back to price units via the fitted scaler."""
    padded = np.zeros((len(scaled_close), INPUT_SIZE))
    padded[:, TARGET_IDX] = scaled_close
    return scaler.inverse_transform(padded)[:, TARGET_IDX]


@app.route("/")
def home():
    return "PyTorch LSTM Inference Server is running!"


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": FEATURE_COLS, "seq_length": SEQ_LENGTH}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        if "input" not in data:
            return jsonify({"error": "Missing 'input' key in JSON payload"}), 400

        input_data = np.array(data["input"], dtype=np.float32)

        if input_data.ndim != 3 or input_data.shape[2] != INPUT_SIZE:
            return jsonify({
                "error": f"Input must be shape (batch, seq_len, {INPUT_SIZE}) "
                         f"with features {FEATURE_COLS}, got {list(input_data.shape)}"
            }), 400

        if input_data.shape[1] != SEQ_LENGTH:
            return jsonify({
                "error": f"Expected seq_len={SEQ_LENGTH}, got {input_data.shape[1]}"
            }), 400

        # Scale with the training-time scaler, predict, then invert.
        batch, seq_len, n_features = input_data.shape
        scaled = scaler.transform(input_data.reshape(-1, n_features)).reshape(batch, seq_len, n_features)

        with torch.no_grad():
            scaled_pred = model(torch.tensor(scaled, dtype=torch.float32)).numpy()

        scaled_pred = np.atleast_1d(scaled_pred)
        predictions = _inverse_close(scaled_pred)

        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))

"""The model's input/output contract — the single source of truth.

Feature order, window length and scaler handling were previously duplicated
across train/models.py, App/src/Optimised_LSTM.py and App/src/App.py. Changing
the window in one place left the others silently disagreeing. Everything that
serves or trains the LSTM imports these definitions from here.
"""

import os

import numpy as np

# Column order is part of the contract: the fitted scaler and the saved
# weights both depend on this exact sequence.
FEATURE_COLS = ["open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]
TARGET_COL = "close"
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)
INPUT_SIZE = len(FEATURE_COLS)

SEQ_LENGTH = 10

MODEL_DEFAULTS = {
    "seq_length": SEQ_LENGTH,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
}


def load_scaler(path=None):
    """Load the MinMaxScaler fitted at training time.

    Serving without it returns values in normalized [0, 1] space that cannot
    be mapped to prices, so a missing scaler is a hard failure rather than a
    silent degradation.
    """
    from joblib import load

    path = path or os.getenv("SCALER_PATH", "scaler.pkl")
    if not os.path.exists(path):
        raise RuntimeError(
            f"Scaler not found at {path}. Run 'python train/lstm_train_pytorch.py' "
            "first — predictions cannot be mapped back to prices without it."
        )
    return load(path)


def validate_window(data):
    """Check a raw (batch, seq_len, n_features) request payload.

    Returns an error string, or None when the payload is acceptable.
    """
    if data.ndim != 3 or data.shape[2] != INPUT_SIZE:
        return (f"Input must be shape (batch, seq_len, {INPUT_SIZE}) "
                f"with features {FEATURE_COLS}, got {list(data.shape)}")
    if data.shape[1] != SEQ_LENGTH:
        return f"Expected seq_len={SEQ_LENGTH}, got {data.shape[1]}"
    return None


def scale_window(scaler, data):
    """Scale a raw window with the training-time scaler."""
    batch, seq_len, n_features = data.shape
    flat = scaler.transform(data.reshape(-1, n_features))
    return flat.reshape(batch, seq_len, n_features).astype(np.float32)


def inverse_close(scaler, scaled_close):
    """Map scaled close predictions back into price units."""
    scaled_close = np.atleast_1d(scaled_close).ravel()
    padded = np.zeros((len(scaled_close), INPUT_SIZE))
    padded[:, TARGET_IDX] = scaled_close
    return scaler.inverse_transform(padded)[:, TARGET_IDX]

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


class ScalerBundle:
    """The fitted scalers that ship alongside the weights, keyed by ticker.

    A multi-ticker model cannot share one scaler: AMZN traded near $1,500 in
    2018 while MSFT traded near $95, and a shared min-max would compress the
    cheaper symbol into a sliver of the range. So the artifact holds one
    scaler per ticker.

    A single-ticker model still writes a bare scaler, and the images already
    built carry one, so `wrap` accepts either shape and files them under
    DEFAULT_KEY. That keeps a request with no `ticker` field working exactly
    as it did before per-ticker scaling existed.
    """

    DEFAULT_KEY = "__default__"

    def __init__(self, scalers):
        if not scalers:
            raise ValueError("ScalerBundle needs at least one scaler.")
        self._scalers = dict(scalers)

    @classmethod
    def wrap(cls, obj):
        """Build a bundle from whatever was unpickled."""
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(obj)
        return cls({cls.DEFAULT_KEY: obj})

    @property
    def tickers(self):
        """The tickers this model was trained on, excluding the default slot."""
        return sorted(k for k in self._scalers if k != self.DEFAULT_KEY)

    @property
    def is_multi_ticker(self):
        return bool(self.tickers)

    def for_ticker(self, ticker=None):
        """Pick the scaler for one symbol.

        With no ticker, a single-ticker model uses its only scaler and a
        multi-ticker model refuses: silently normalising an AMZN window with
        the AAPL scaler would return a confident, meaningless price.
        """
        if ticker:
            key = ticker.upper()
            if key in self._scalers:
                return self._scalers[key]
            if not self.is_multi_ticker:
                # A bare scaler carries no record of what it was fitted on,
                # so it cannot contradict the caller. Erroring here would
                # break every already-deployed single-ticker image the moment
                # a caller started sending the field.
                return self._scalers[self.DEFAULT_KEY]
            raise KeyError(
                f"No scaler for {key}. This model was trained on: "
                f"{', '.join(self.tickers)}"
            )
        if self.DEFAULT_KEY in self._scalers:
            return self._scalers[self.DEFAULT_KEY]
        if len(self._scalers) == 1:
            return next(iter(self._scalers.values()))
        raise KeyError(
            "This model covers several tickers, so a 'ticker' field is "
            f"required. Known tickers: {', '.join(self.tickers)}"
        )

    def to_payload(self):
        """What gets pickled to scaler.pkl."""
        return dict(self._scalers)


def load_scaler(path=None):
    """Load the scaler bundle fitted at training time.

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
    return ScalerBundle.wrap(load(path))


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

"""Train the sentiment-aware LSTM on daily OHLCV + sentiment features.

Run directly:  python lstm_train_pytorch.py
Importing this module has no side effects.

Handles both dataset shapes. A frame with no `ticker` column - the committed
data_2018.csv - trains exactly as before and writes a single scaler. A panel
carrying a `ticker` column trains across several symbols at once, cutting
sequences per ticker and fitting one scaler per ticker.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from models import DEFAULTS, FEATURE_COLS, LSTMModel  # noqa: E402
from pipeline.dataset import (  # noqa: E402
    TICKER_COL,
    as_panel,
    describe_panel,
    fit_scalers,
    make_sequences,
    split_panel,
)
from serving.contract import TARGET_COL  # noqa: E402

config = {
    "epochs": 20,
    "batch_size": 16,
    "lr": 1e-3,
    "seq_length": DEFAULTS["seq_length"],
    "hidden_dim": DEFAULTS["hidden_dim"],
    "num_layers": DEFAULTS["num_layers"],
    "dropout": DEFAULTS["dropout"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

MODEL_PATH = os.getenv("MODEL_PATH", "lstm_model.pth")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

# Date boundaries suit a multi-year panel; a fraction is the only thing that
# can split a single year. Unset by default so the committed 2018 data works.
TRAIN_END = os.getenv("TRAIN_END") or None
VAL_END = os.getenv("VAL_END") or None
VAL_FRACTION = float(os.getenv("VAL_FRACTION", "0.2"))


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_panel(data_path, feature_cols=FEATURE_COLS, tickers=None):
    """Read the training CSV as a panel, optionally restricted to some tickers."""
    df = pd.read_csv(data_path)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{data_path} is missing required columns: {missing}")

    panel = as_panel(df)
    if tickers:
        wanted = {t.upper() for t in tickers}
        unknown = wanted - set(panel[TICKER_COL])
        if unknown:
            raise ValueError(f"{data_path} has no rows for: {sorted(unknown)}")
        panel = panel[panel[TICKER_COL].isin(wanted)].reset_index(drop=True)
    return panel


def _to_price(scaler, values, feature_cols):
    """Map scaled close values back into dollars for one ticker."""
    values = np.atleast_1d(values).ravel()
    idx = list(feature_cols).index(TARGET_COL)
    padded = np.zeros((len(values), len(feature_cols)))
    padded[:, idx] = values
    return scaler.inverse_transform(padded)[:, idx]


def evaluate(model, scalers, X, y, meta, feature_cols=FEATURE_COLS):
    """Scaled MSE overall, plus MAE and RMSE in dollars per ticker.

    Price-unit errors have to be computed per ticker: each symbol has its own
    scaler, so there is no single inverse transform for the whole batch.
    """
    model.eval()
    with torch.no_grad():
        pred = np.atleast_1d(
            model(torch.tensor(X, dtype=torch.float32).to(config["device"])).cpu().numpy()
        )

    metrics = {"val_mse": float(np.mean((pred - y) ** 2))}
    abs_err, sq_err = [], []
    for ticker in sorted(set(meta[TICKER_COL])):
        rows = np.flatnonzero(meta[TICKER_COL].values == ticker)
        p = _to_price(scalers[ticker], pred[rows], feature_cols)
        t = _to_price(scalers[ticker], y[rows], feature_cols)
        metrics[f"val_mae_price_{ticker}"] = float(np.mean(np.abs(p - t)))
        metrics[f"val_rmse_price_{ticker}"] = float(np.sqrt(np.mean((p - t) ** 2)))
        abs_err.append(np.abs(p - t))
        sq_err.append((p - t) ** 2)

    metrics["val_mae_price"] = float(np.mean(np.concatenate(abs_err)))
    metrics["val_rmse_price"] = float(np.sqrt(np.mean(np.concatenate(sq_err))))
    metrics["n_val"] = float(len(y))
    return metrics


def train(data_path=None, feature_cols=FEATURE_COLS, seed=None, verbose=True,
          tickers=None, train_end=None, val_end=None, val_frac=None):
    """Train one model. Returns (model, scalers, metrics).

    `scalers` maps ticker -> fitted MinMaxScaler. serving.contract.ScalerBundle
    consumes that mapping directly, so what is served is what was fitted.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    feature_cols = list(feature_cols)
    data_path = data_path or os.getenv("DATA_CSV_PATH", "data_2018.csv")
    panel = load_panel(data_path, feature_cols, tickers)

    train_panel, val_panel, _ = split_panel(
        panel,
        train_end=train_end if train_end is not None else TRAIN_END,
        val_end=val_end if val_end is not None else VAL_END,
        val_frac=VAL_FRACTION if val_frac is None else val_frac,
    )
    if val_panel.empty:
        raise ValueError(
            "The split left no validation rows. For a single-year dataset drop "
            "TRAIN_END/VAL_END and let VAL_FRACTION split it."
        )

    # Fitted on the training rows only. Fitting on the whole file - what this
    # script used to do - leaks the held-out period's price range into the
    # normalisation and flatters the validation loss.
    scalers = fit_scalers(train_panel, feature_cols)

    X_train, y_train, _ = make_sequences(train_panel, scalers, config["seq_length"], feature_cols)
    X_val, y_val, meta_val = make_sequences(val_panel, scalers, config["seq_length"], feature_cols)

    if verbose:
        print(describe_panel(panel).to_string())
        print(f"\nsequences: {len(X_train)} train / {len(X_val)} val "
              f"across {len(scalers)} ticker(s), {len(feature_cols)} features")

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=config["batch_size"], shuffle=False)

    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(config["device"])

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_loss = val_loss = float("nan")
    for epoch in range(config["epochs"]):
        model.train()
        running = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(config["device"]), y_batch.to(config["device"])
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * X_batch.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(config["device"]), y_batch.to(config["device"])
                running += criterion(model(X_batch), y_batch).item() * X_batch.size(0)
        val_loss = running / len(val_loader.dataset)

        if verbose:
            print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

    metrics = {"train_mse": train_loss}
    metrics.update(evaluate(model, scalers, X_val, y_val, meta_val, feature_cols))
    return model, scalers, metrics


if __name__ == "__main__":
    from pipeline.data_source import resolve_training_csv

    data_path, source = resolve_training_csv(local_fallback="data_2018.csv")
    print(f"Training data source: {source}\n")

    tickers = [t for t in os.getenv("TICKERS", "").replace(",", " ").split() if t]
    model, scalers, metrics = train(data_path=data_path, tickers=tickers or None)

    torch.save(model.state_dict(), MODEL_PATH)
    # The scalers are part of the model artifact: without them, served
    # predictions come back in normalized [0, 1] space and cannot be mapped
    # to prices. A dict is written even for one ticker so the serving side
    # has a single shape to reason about.
    dump(scalers, SCALER_PATH)
    print(f"\nSaved model   -> {MODEL_PATH}")
    print(f"Saved scalers -> {SCALER_PATH} ({', '.join(sorted(scalers))})")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<28} {v:.6f}")

    # --- MLflow tracking (optional: skipped cleanly if unavailable) ---
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow
            import mlflow.pytorch

            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "finpulse-lstm"))
            with mlflow.start_run():
                mlflow.log_params({k: v for k, v in config.items()})
                mlflow.log_param("features", ",".join(FEATURE_COLS))
                mlflow.log_param("tickers", ",".join(sorted(scalers)))
                mlflow.log_param("data_source", source)
                mlflow.log_metrics(metrics)
                # Log the raw state_dict and scalers as run artifacts. The
                # registered pyfunc model serializes to data/model.pt2 (a
                # traced graph), which the inference server cannot consume -
                # it calls load_state_dict. The build pipeline pulls these two
                # files, so what gets served is exactly what was trained.
                mlflow.log_artifact(SCALER_PATH)
                mlflow.log_artifact(MODEL_PATH)
                # MLflow >=3 traces the graph for serialization, so it needs a
                # representative input; without one log_model raises and the run
                # is marked FAILED.
                example = torch.randn(
                    1, config["seq_length"], len(FEATURE_COLS), dtype=torch.float32
                )
                mlflow.pytorch.log_model(
                    model.cpu(),
                    name="model",
                    input_example=example.numpy(),
                    registered_model_name=os.getenv("MLFLOW_MODEL_NAME", "FinPulseLSTM"),
                )
                print(f"Logged run to {os.environ['MLFLOW_TRACKING_URI']}")
        except Exception as e:
            print(f"[warn] MLflow logging failed: {e}")
    else:
        print("[info] MLFLOW_TRACKING_URI unset — skipping experiment tracking.")

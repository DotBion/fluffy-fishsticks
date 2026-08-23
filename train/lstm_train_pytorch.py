"""Train the sentiment-aware LSTM on daily OHLCV + sentiment features.

Run directly:  python lstm_train_pytorch.py
Importing this module has no side effects.
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from models import DEFAULTS, FEATURE_COLS, TARGET_IDX, LSTMModel

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


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(data, seq_length):
    """Sliding windows of `seq_length` rows predicting the next row's close."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][TARGET_IDX])
    return np.array(X), np.array(y)


def load_data(data_path, feature_cols=FEATURE_COLS):
    df = pd.read_csv(data_path).sort_values(by="date")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    return df, data_scaled, scaler


def train(data_path=None, feature_cols=FEATURE_COLS, seed=None, verbose=True):
    """Train one model. Returns (model, scaler, metrics)."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    data_path = data_path or os.getenv("DATA_CSV_PATH", "data_2018.csv")
    _, data_scaled, scaler = load_data(data_path, feature_cols)

    X, y = create_sequences(data_scaled, config["seq_length"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

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
            print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    return model, scaler, {"train_mse": train_loss, "val_mse": val_loss}


if __name__ == "__main__":
    model, scaler, metrics = train()
    torch.save(model.state_dict(), MODEL_PATH)
    # The scaler is part of the model artifact: without it, served predictions
    # come back in normalized [0, 1] space and cannot be mapped to prices.
    dump(scaler, SCALER_PATH)
    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved scaler -> {SCALER_PATH}")
    print(f"Final metrics: {metrics}")

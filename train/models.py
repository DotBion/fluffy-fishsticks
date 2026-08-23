"""Model definitions and feature contract shared by training and serving.

Kept import-side-effect free on purpose: importing this module must never
train, read data, or touch the filesystem.
"""

import torch
import torch.nn as nn

# Column order is part of the model contract. Training, serving and the
# saved scaler all depend on this exact sequence.
FEATURE_COLS = ["open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]
TARGET_COL = "close"
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)

DEFAULTS = {
    "seq_length": 10,
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
}


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)

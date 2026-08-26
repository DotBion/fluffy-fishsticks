"""PyTorch backend — loads the state_dict the training run produced.

This is the path that stays trivially in sync with training, which makes it
the better choice for local debugging.
"""

import os

import torch
import torch.nn as nn

from ..contract import INPUT_SIZE, MODEL_DEFAULTS


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        # squeeze(-1) not squeeze(): a batch of one would otherwise collapse
        # to a 0-d tensor, which is exactly the single-request serving case.
        return out.squeeze(-1)


class TorchBackend:
    name = "torch"

    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("MODEL_PATH", "lstm_model.pth")
        self.model = LSTMModel(
            input_size=INPUT_SIZE,
            hidden_dim=MODEL_DEFAULTS["hidden_dim"],
            num_layers=MODEL_DEFAULTS["num_layers"],
            dropout=MODEL_DEFAULTS["dropout"],
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        self.model.eval()

    def predict_scaled(self, scaled):
        with torch.no_grad():
            return self.model(torch.tensor(scaled, dtype=torch.float32)).numpy()

    def describe(self):
        return {"backend": self.name, "model": self.model_path}

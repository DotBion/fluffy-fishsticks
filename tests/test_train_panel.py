"""The whole loop on a multi-ticker panel: CSV -> train -> artifacts -> serve.

Each piece has its own unit tests; this one exists because the pieces have
to agree on the shape of `scaler.pkl`, and that agreement is what breaks
silently. Training writes a ticker -> scaler mapping; serving has to read it
back and pick the right one.
"""

import importlib
import os
import sys

import pytest
import torch
from joblib import dump

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "train"))

from serving.contract import SEQ_LENGTH, ScalerBundle, load_scaler  # noqa: E402


@pytest.fixture
def panel_csv(panel, tmp_path):
    path = tmp_path / "panel.csv"
    panel.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def trained(panel_csv):
    """Two epochs is enough to prove the plumbing; this is not a quality test."""
    import lstm_train_pytorch as trainer

    original = trainer.config["epochs"]
    trainer.config["epochs"] = 2
    try:
        yield trainer.train(data_path=panel_csv, verbose=False, val_frac=0.25)
    finally:
        trainer.config["epochs"] = original


def test_training_returns_one_scaler_per_ticker(trained):
    _, scalers, _ = trained
    assert sorted(scalers) == ["AAPL", "AMZN", "MSFT"]


def test_metrics_are_reported_per_ticker_and_overall(trained):
    _, _, metrics = trained
    for ticker in ("AAPL", "AMZN", "MSFT"):
        assert f"val_mae_price_{ticker}" in metrics
        assert metrics[f"val_mae_price_{ticker}"] > 0
    assert {"train_mse", "val_mse", "val_mae_price", "val_rmse_price"} <= set(metrics)
    assert all(isinstance(v, float) for v in metrics.values())


def test_price_errors_track_each_tickers_own_scale(trained):
    """AMZN trades near $1,500 and MSFT near $95, so their dollar errors differ.

    A single shared scaler would collapse them toward one number.
    """
    _, _, metrics = trained
    assert metrics["val_mae_price_AMZN"] > metrics["val_mae_price_MSFT"]


def test_the_artifacts_train_writes_are_the_artifacts_serving_reads(trained, tmp_path):
    model, scalers, _ = trained
    weights = tmp_path / "lstm_model.pth"
    scaler_path = tmp_path / "scaler.pkl"
    torch.save(model.state_dict(), weights)
    dump(scalers, scaler_path)

    bundle = load_scaler(str(scaler_path))
    assert isinstance(bundle, ScalerBundle)
    assert bundle.tickers == ["AAPL", "AMZN", "MSFT"]

    os.environ["BACKEND"] = "torch"
    os.environ["MODEL_PATH"] = str(weights)
    os.environ["SCALER_PATH"] = str(scaler_path)
    import serving.app

    importlib.reload(serving.app)
    client = serving.app.app.test_client()

    assert client.get("/health").get_json()["tickers"] == ["AAPL", "AMZN", "MSFT"]

    def predict(ticker, price):
        row = [price, price * 1.01, price * 0.99, price, 3.0e7, 0.25]
        r = client.post("/predict", json={
            "input": [[row] * SEQ_LENGTH], "ticker": ticker,
        })
        assert r.status_code == 200, r.get_json()
        return r.get_json()["predictions"][0]

    # Each prediction should land near its own ticker's price level rather
    # than somewhere between the three.
    assert 50 < predict("MSFT", 95.0) < 300
    assert 800 < predict("AMZN", 1500.0) < 3000


def test_a_ticker_the_model_never_saw_is_refused(trained, tmp_path):
    _, scalers, _ = trained
    path = tmp_path / "scaler.pkl"
    dump(scalers, path)
    with pytest.raises(KeyError, match="No scaler for NVDA"):
        load_scaler(str(path)).for_ticker("NVDA")


def test_restricting_to_a_subset_trains_on_that_subset(panel_csv):
    import lstm_train_pytorch as trainer

    original = trainer.config["epochs"]
    trainer.config["epochs"] = 1
    try:
        _, scalers, _ = trainer.train(
            data_path=panel_csv, verbose=False, val_frac=0.25, tickers=["AAPL", "MSFT"]
        )
    finally:
        trainer.config["epochs"] = original
    assert sorted(scalers) == ["AAPL", "MSFT"]


def test_an_unknown_ticker_in_the_request_is_rejected_at_load(panel_csv):
    import lstm_train_pytorch as trainer

    with pytest.raises(ValueError, match="no rows for"):
        trainer.load_panel(panel_csv, tickers=["NVDA"])


def test_a_split_that_leaves_no_validation_rows_fails_loudly(panel_csv):
    import lstm_train_pytorch as trainer

    with pytest.raises(ValueError, match="no validation rows"):
        trainer.train(data_path=panel_csv, verbose=False,
                      train_end="2030-01-01", val_end="2031-01-01")

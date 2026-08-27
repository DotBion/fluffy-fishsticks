"""End-to-end checks against the real Flask app and the committed weights.

The app imports its backend and scaler at module scope, so each test module
that needs a different artifact has to set the environment before importing
and reload afterwards.
"""

import importlib
import json
import os

import numpy as np
import pytest
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from serving.contract import INPUT_SIZE, SEQ_LENGTH

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO, "train", "lstm_model.pth")
SCALER = os.path.join(REPO, "train", "scaler.pkl")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(WEIGHTS) and os.path.exists(SCALER)),
    reason="committed model artifacts not present",
)


def _window(price=170.0):
    """A plausible raw request window: ten days of OHLCV plus sentiment.

    The intraday range is a fraction of the price rather than a fixed number
    of dollars, so two windows at different price levels are the same shape.
    """
    row = [price, price * 1.012, price * 0.988, price, 3.0e7, 0.25]
    return [[row for _ in range(SEQ_LENGTH)]]


def _client(scaler_path):
    """A test client bound to a specific scaler artifact."""
    os.environ["BACKEND"] = "torch"
    os.environ["MODEL_PATH"] = WEIGHTS
    os.environ["SCALER_PATH"] = scaler_path
    import serving.app

    importlib.reload(serving.app)
    serving.app.app.config["TESTING"] = True
    return serving.app.app.test_client()


@pytest.fixture
def legacy_client():
    """The single-ticker artifact the current image ships."""
    return _client(SCALER)


@pytest.fixture
def panel_client(tmp_path):
    """A multi-ticker artifact, built by refitting the shipped ranges."""
    base = MinMaxScaler().fit(
        np.array([[0.0] * INPUT_SIZE, [200.0, 200.0, 200.0, 200.0, 1e8, 1.0]])
    )
    rich = MinMaxScaler().fit(
        np.array([[0.0] * INPUT_SIZE, [2000.0, 2000.0, 2000.0, 2000.0, 1e8, 1.0]])
    )
    path = tmp_path / "panel_scaler.pkl"
    dump({"AAPL": base, "AMZN": rich}, path)
    return _client(str(path))


def test_health_reports_the_contract(legacy_client):
    body = legacy_client.get("/health").get_json()
    assert body["status"] == "ok"
    assert body["seq_length"] == SEQ_LENGTH
    assert body["multi_ticker"] is False
    assert body["tickers"] == []


def test_an_unlabelled_request_still_predicts(legacy_client):
    """The behaviour every existing caller and the smoke test depend on."""
    r = legacy_client.post("/predict", json={"input": _window()})
    assert r.status_code == 200
    body = r.get_json()
    assert len(body["predictions"]) == 1
    assert 50 < body["predictions"][0] < 500
    assert body["ticker"] is None


def test_the_wrong_window_length_is_a_400(legacy_client):
    short = [[[170.0] * INPUT_SIZE for _ in range(SEQ_LENGTH - 1)]]
    r = legacy_client.post("/predict", json={"input": short})
    assert r.status_code == 400
    assert "seq_len" in r.get_json()["error"]


def test_a_missing_input_key_is_a_400(legacy_client):
    r = legacy_client.post("/predict", json={"ticker": "AAPL"})
    assert r.status_code == 400


def test_a_non_object_body_is_a_400(legacy_client):
    r = legacy_client.post("/predict", data=json.dumps([1, 2, 3]),
                           content_type="application/json")
    assert r.status_code == 400


def test_health_lists_the_panel_tickers(panel_client):
    body = panel_client.get("/health").get_json()
    assert body["multi_ticker"] is True
    assert body["tickers"] == ["AAPL", "AMZN"]


def test_the_ticker_selects_the_scaler(panel_client):
    """The whole point of per-ticker scaling, stated as a ratio.

    Both fixtures have a minimum of zero and ranges of 200 and 2000, so a
    $170 AAPL window and a $1,700 AMZN window normalise to the same point.
    Identical scaled input means identical scaled output, and the inverse
    transform then has to put the two predictions exactly ten times apart.
    A shared scaler could not do that.
    """
    cheap = panel_client.post("/predict", json={"input": _window(170.0), "ticker": "AAPL"})
    rich = panel_client.post("/predict", json={"input": _window(1700.0), "ticker": "AMZN"})
    assert cheap.status_code == rich.status_code == 200
    a = cheap.get_json()["predictions"][0]
    b = rich.get_json()["predictions"][0]
    assert b == pytest.approx(a * 10, rel=1e-4), f"AMZN {b} vs AAPL {a}"
    assert rich.get_json()["ticker"] == "AMZN"


def test_the_same_window_under_two_scalers_gives_two_answers(panel_client):
    """A shared scaler would return the identical number for both."""
    a = panel_client.post("/predict", json={"input": _window(), "ticker": "AAPL"})
    b = panel_client.post("/predict", json={"input": _window(), "ticker": "AMZN"})
    assert a.get_json()["predictions"][0] != b.get_json()["predictions"][0]


def test_a_multi_ticker_model_rejects_an_unlabelled_request(panel_client):
    r = panel_client.post("/predict", json={"input": _window()})
    assert r.status_code == 400
    assert "ticker" in r.get_json()["error"]


def test_an_unknown_ticker_is_a_400_naming_the_known_ones(panel_client):
    r = panel_client.post("/predict", json={"input": _window(), "ticker": "NVDA"})
    assert r.status_code == 400
    assert "AAPL, AMZN" in r.get_json()["error"]


def test_an_unknown_ticker_does_not_mint_a_metric_series(panel_client):
    """Label values come off the request body, so they have to be bounded."""
    panel_client.post("/predict", json={"input": _window(), "ticker": "NOTATICKER"})
    text = panel_client.get("/metrics").get_data(as_text=True)
    assert 'ticker="NOTATICKER"' not in text
    assert 'ticker="unknown"' in text


def test_a_legacy_artifact_accepts_a_ticker_it_cannot_check(legacy_client):
    r = legacy_client.post("/predict", json={"input": _window(), "ticker": "AAPL"})
    assert r.status_code == 200
    assert r.get_json()["ticker"] == "AAPL"


def test_metrics_are_exposed_and_labelled(panel_client):
    panel_client.post("/predict", json={"input": _window(), "ticker": "AAPL"})
    text = panel_client.get("/metrics").get_data(as_text=True)
    assert "lstm_predictions_total" in text
    assert 'ticker="AAPL"' in text
    assert "lstm_inference_seconds" in text

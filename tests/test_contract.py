"""The serving contract, including what the already-built images rely on.

The ONNX image in the registry carries a bare pickled MinMaxScaler and its
callers send no ticker. Per-ticker scaling must not break either.
"""

import numpy as np
import pytest
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from serving.contract import (
    FEATURE_COLS,
    INPUT_SIZE,
    SEQ_LENGTH,
    TARGET_IDX,
    ScalerBundle,
    inverse_close,
    load_scaler,
    scale_window,
    validate_window,
)


def _fitted(low, high):
    scaler = MinMaxScaler()
    scaler.fit(np.array([[low] * INPUT_SIZE, [high] * INPUT_SIZE]))
    return scaler


def test_target_index_points_at_close():
    assert FEATURE_COLS[TARGET_IDX] == "close"


def test_a_bare_scaler_still_serves_unlabelled_requests():
    bundle = ScalerBundle.wrap(_fitted(0, 200))
    assert bundle.tickers == []
    assert bundle.is_multi_ticker is False
    assert bundle.for_ticker() is bundle.for_ticker(None)


def test_a_bare_scaler_tolerates_a_ticker_it_cannot_verify():
    """The compatibility path for images already in the registry.

    A bare scaler has no record of what it was fitted on, so it cannot
    contradict the caller — and refusing would break every deployed
    single-ticker service as soon as a caller started sending the field.
    """
    bundle = ScalerBundle.wrap(_fitted(0, 200))
    assert bundle.for_ticker("AAPL") is bundle.for_ticker()
    assert bundle.for_ticker("ANYTHING") is bundle.for_ticker()


def test_a_dict_of_scalers_is_keyed_by_ticker():
    bundle = ScalerBundle.wrap({"AAPL": _fitted(0, 200), "AMZN": _fitted(0, 2000)})
    assert bundle.tickers == ["AAPL", "AMZN"]
    assert bundle.is_multi_ticker is True
    assert bundle.for_ticker("aapl") is bundle.for_ticker("AAPL")


def test_multi_ticker_refuses_an_unlabelled_request():
    """Guessing would return a confident, wrong price rather than an error."""
    bundle = ScalerBundle.wrap({"AAPL": _fitted(0, 200), "AMZN": _fitted(0, 2000)})
    with pytest.raises(KeyError, match="'ticker' field is"):
        bundle.for_ticker()


def test_unknown_ticker_names_what_the_model_covers():
    bundle = ScalerBundle.wrap({"AAPL": _fitted(0, 200), "AMZN": _fitted(0, 2000)})
    with pytest.raises(KeyError, match="AAPL, AMZN"):
        bundle.for_ticker("NVDA")


def test_a_single_named_ticker_needs_no_label():
    bundle = ScalerBundle.wrap({"AAPL": _fitted(0, 200)})
    assert bundle.for_ticker() is bundle.for_ticker("AAPL")


def test_a_named_single_ticker_model_still_refuses_a_different_symbol():
    """Unlike a bare scaler, this one knows what it was fitted on."""
    bundle = ScalerBundle.wrap({"AAPL": _fitted(0, 200)})
    with pytest.raises(KeyError, match="No scaler for NVDA"):
        bundle.for_ticker("NVDA")


def test_wrapping_a_bundle_is_idempotent():
    bundle = ScalerBundle.wrap(_fitted(0, 200))
    assert ScalerBundle.wrap(bundle) is bundle


def test_an_empty_bundle_is_rejected():
    with pytest.raises(ValueError):
        ScalerBundle({})


def test_load_scaler_reads_both_artifact_shapes(tmp_path):
    legacy = tmp_path / "legacy.pkl"
    dump(_fitted(0, 200), legacy)
    assert load_scaler(str(legacy)).tickers == []

    panel = tmp_path / "panel.pkl"
    dump({"AAPL": _fitted(0, 200), "MSFT": _fitted(0, 120)}, panel)
    assert load_scaler(str(panel)).tickers == ["AAPL", "MSFT"]


def test_a_missing_scaler_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError, match="Scaler not found"):
        load_scaler(str(tmp_path / "absent.pkl"))


def test_round_trip_recovers_the_close_price():
    scaler = _fitted(0, 500)
    window = np.full((1, SEQ_LENGTH, INPUT_SIZE), 123.0, dtype=np.float32)
    scaled = scale_window(scaler, window)
    recovered = inverse_close(scaler, scaled[0, -1, TARGET_IDX])
    assert recovered[0] == pytest.approx(123.0, rel=1e-5)


def test_validate_window_rejects_the_wrong_shape():
    assert validate_window(np.zeros((1, SEQ_LENGTH, INPUT_SIZE))) is None
    assert "seq_len" in validate_window(np.zeros((1, SEQ_LENGTH + 1, INPUT_SIZE)))
    assert "shape" in validate_window(np.zeros((1, SEQ_LENGTH, INPUT_SIZE - 1)))
    assert "shape" in validate_window(np.zeros((SEQ_LENGTH, INPUT_SIZE)))

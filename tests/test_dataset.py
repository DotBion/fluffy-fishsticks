"""The panel builder's correctness properties.

These target the two defects that a single-ticker dataset cannot expose:
windows spanning a ticker boundary, and a scaler fitted across the split.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.dataset import (
    DEFAULT_TICKERS,
    TICKER_COL,
    as_panel,
    build_panel,
    chronological_split,
    describe_panel,
    fit_scalers,
    make_sequences,
)
from serving.contract import FEATURE_COLS, SEQ_LENGTH, TARGET_IDX


def test_default_panel_excludes_the_duplicate_alphabet_listing():
    assert "GOOGL" in DEFAULT_TICKERS
    assert "GOOG" not in DEFAULT_TICKERS


def test_build_panel_rejects_a_repeated_trading_day():
    frame = pd.DataFrame({
        "date": ["2018-01-02", "2018-01-02"],
        **{c: [1.0, 1.0] for c in FEATURE_COLS},
    })
    with pytest.raises(ValueError, match="duplicate"):
        build_panel({"AAPL": frame})


def test_legacy_frame_without_a_ticker_column_still_loads(single_ticker_frame):
    panel = as_panel(single_ticker_frame, default_ticker="AAPL")
    assert set(panel[TICKER_COL]) == {"AAPL"}
    assert len(panel) == len(single_ticker_frame)


def test_split_boundaries_do_not_overlap(panel):
    train, val, test = chronological_split(panel, "2018-12-31", "2019-06-30")
    assert train["date"].max() <= pd.Timestamp("2018-12-31")
    assert val["date"].min() > pd.Timestamp("2018-12-31")
    assert val["date"].max() <= pd.Timestamp("2019-06-30")
    if not test.empty:
        assert test["date"].min() > pd.Timestamp("2019-06-30")
    assert len(train) + len(val) + len(test) == len(panel)


def test_split_applies_to_every_ticker(panel):
    train, val, _ = chronological_split(panel, "2018-12-31", "2019-06-30")
    assert set(train[TICKER_COL]) == set(panel[TICKER_COL])
    assert set(val[TICKER_COL]) == set(panel[TICKER_COL])


def test_scalers_are_fitted_per_ticker_on_training_rows_only(panel):
    train, val, _ = chronological_split(panel, "2018-12-31", "2019-06-30")
    scalers = fit_scalers(train)

    assert set(scalers) == set(train[TICKER_COL].unique())

    for ticker, scaler in scalers.items():
        rows = train[train[TICKER_COL] == ticker][FEATURE_COLS].values
        scaled = scaler.transform(rows)
        # Fitted on exactly these rows, so they span [0, 1] and nothing more.
        assert scaled.min() == pytest.approx(0.0, abs=1e-9)
        assert scaled.max() == pytest.approx(1.0, abs=1e-9)

    # The held-out period was never seen by the scaler. If it had been, the
    # validation rows would also be confined to [0, 1].
    escaped = False
    for ticker, scaler in scalers.items():
        rows = val[val[TICKER_COL] == ticker][FEATURE_COLS].values
        scaled = scaler.transform(rows)
        escaped |= bool(scaled.min() < 0 or scaled.max() > 1)
    assert escaped, "validation rows all landed inside the training range"


def test_a_ticker_too_short_to_window_is_rejected(panel):
    short = panel.groupby(TICKER_COL, group_keys=False).head(SEQ_LENGTH)
    with pytest.raises(ValueError, match="too few"):
        fit_scalers(short)


def test_sequences_never_span_two_tickers(panel):
    """The defect this module exists to prevent.

    Each window is reconstructed from the raw panel by matching its scaled
    values back to one ticker's rows; a window built across a boundary could
    not be matched to any single ticker.
    """
    train, _, _ = chronological_split(panel, "2019-12-31", "2020-12-31")
    scalers = fit_scalers(train)
    X, y, meta = make_sequences(train, scalers)

    for ticker, group in train.groupby(TICKER_COL):
        rows = scalers[ticker].transform(group.sort_values("date")[FEATURE_COLS].values)
        idx = np.flatnonzero(meta[TICKER_COL].values == ticker)
        for offset, position in enumerate(idx):
            expected = rows[offset:offset + SEQ_LENGTH]
            np.testing.assert_allclose(X[position], expected, rtol=1e-6)
            assert y[position] == pytest.approx(rows[offset + SEQ_LENGTH][TARGET_IDX], rel=1e-6)


def test_window_count_is_per_ticker_not_per_panel(panel):
    train, _, _ = chronological_split(panel, "2019-12-31", "2020-12-31")
    scalers = fit_scalers(train)
    X, y, meta = make_sequences(train, scalers)

    counts = train[TICKER_COL].value_counts()
    expected = sum(n - SEQ_LENGTH for n in counts)
    # A panel-wide sliding window would yield len(train) - SEQ_LENGTH, which
    # is larger by SEQ_LENGTH for every ticker boundary crossed.
    assert len(X) == expected
    assert len(X) == len(y) == len(meta)
    assert len(X) < len(train) - SEQ_LENGTH


def test_sequence_shape_matches_the_serving_contract(panel):
    train, _, _ = chronological_split(panel, "2019-12-31", "2020-12-31")
    X, _, _ = make_sequences(train, fit_scalers(train))
    assert X.shape[1:] == (SEQ_LENGTH, len(FEATURE_COLS))
    assert X.dtype == np.float32


def test_target_date_follows_the_window(panel):
    train, _, _ = chronological_split(panel, "2019-12-31", "2020-12-31")
    _, _, meta = make_sequences(train, fit_scalers(train))
    for ticker, group in meta.groupby(TICKER_COL):
        dates = pd.to_datetime(group["target_date"])
        assert dates.is_monotonic_increasing
        source = train[train[TICKER_COL] == ticker].sort_values("date")["date"]
        # The first target is the day after the first full window.
        assert dates.iloc[0] == source.iloc[SEQ_LENGTH]


def test_validation_rows_scale_with_the_training_scaler(panel):
    train, val, _ = chronological_split(panel, "2018-12-31", "2019-06-30")
    scalers = fit_scalers(train)
    X_val, _, meta = make_sequences(val, scalers)
    assert len(X_val) > 0
    assert set(meta[TICKER_COL]) <= set(scalers)


def test_a_ticker_absent_from_training_is_a_hard_error(panel):
    train, val, _ = chronological_split(panel, "2018-12-31", "2019-06-30")
    scalers = fit_scalers(train)
    scalers.pop("MSFT")
    with pytest.raises(KeyError, match="No scaler for MSFT"):
        make_sequences(val, scalers)


def test_describe_panel_reports_one_row_per_ticker(panel):
    summary = describe_panel(panel)
    assert list(summary.columns) == ["rows", "first", "last"]
    assert len(summary) == panel[TICKER_COL].nunique()


# --- fractional splitting -------------------------------------------------
# Date boundaries cannot split a single year, which is exactly the shape of
# the committed data_2018.csv.

def test_fractional_split_keeps_every_ticker_in_every_split(panel):
    from pipeline.dataset import fractional_split

    train, val, test = fractional_split(panel, val_frac=0.2, test_frac=0.1)
    for part in (train, val, test):
        assert set(part[TICKER_COL]) == set(panel[TICKER_COL])
    assert len(train) + len(val) + len(test) == len(panel)


def test_fractional_split_stays_chronological_within_each_ticker(panel):
    from pipeline.dataset import fractional_split

    train, val, test = fractional_split(panel, val_frac=0.2, test_frac=0.1)
    for ticker in panel[TICKER_COL].unique():
        t = train[train[TICKER_COL] == ticker]["date"]
        v = val[val[TICKER_COL] == ticker]["date"]
        s = test[test[TICKER_COL] == ticker]["date"]
        assert t.max() < v.min(), f"{ticker}: validation starts before training ends"
        assert v.max() < s.min(), f"{ticker}: test starts before validation ends"


def test_fractional_split_honours_the_requested_proportions(panel):
    from pipeline.dataset import fractional_split

    train, val, _ = fractional_split(panel, val_frac=0.2)
    assert len(val) == pytest.approx(len(panel) * 0.2, rel=0.02)
    assert len(train) == len(panel) - len(val)


def test_a_single_year_frame_still_yields_a_validation_split(single_ticker_frame):
    """What date boundaries get wrong and fractions get right."""
    from pipeline.dataset import chronological_split, fractional_split

    p = as_panel(single_ticker_frame)
    _, val_by_date, _ = chronological_split(p, "2018-12-31", "2019-12-31")
    assert val_by_date.empty

    _, val_by_fraction, _ = fractional_split(p, val_frac=0.2)
    assert len(val_by_fraction) > SEQ_LENGTH


def test_split_panel_picks_the_right_strategy(panel, single_ticker_frame):
    from pipeline.dataset import split_panel

    _, by_date, _ = split_panel(panel, train_end="2018-12-31", val_end="2019-06-30")
    assert by_date["date"].min() > pd.Timestamp("2018-12-31")

    _, by_fraction, _ = split_panel(as_panel(single_ticker_frame), val_frac=0.2)
    assert not by_fraction.empty


def test_impossible_fractions_are_rejected(panel):
    from pipeline.dataset import fractional_split

    with pytest.raises(ValueError, match="Invalid split fractions"):
        fractional_split(panel, val_frac=0.8, test_frac=0.3)

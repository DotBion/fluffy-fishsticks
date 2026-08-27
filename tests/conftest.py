"""Shared fixtures. Tests import the repo's packages from the repo root."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.contract import FEATURE_COLS  # noqa: E402


def _series(ticker, start, periods, base_price, seed):
    """A synthetic OHLCV+sentiment series with a ticker-specific price level."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    close = base_price + np.cumsum(rng.normal(0, base_price * 0.01, periods))
    return pd.DataFrame({
        "date": dates,
        "open": close + rng.normal(0, 0.5, periods),
        "high": close + abs(rng.normal(1, 0.5, periods)),
        "low": close - abs(rng.normal(1, 0.5, periods)),
        "close": close,
        "volume": rng.integers(1e6, 5e7, periods).astype(float),
        "daily_avg_sentiment_score": rng.uniform(-1, 1, periods),
        "ticker": ticker,
    })[["date"] + FEATURE_COLS + ["ticker"]]


@pytest.fixture
def panel():
    """Three tickers at very different price levels, 2018 through 2019.

    The price gap is the point: a single shared scaler would compress the
    cheap ticker into a sliver of [0, 1].
    """
    return pd.concat([
        _series("AAPL", "2018-01-01", 400, 170.0, seed=1),
        _series("AMZN", "2018-01-01", 400, 1500.0, seed=2),
        _series("MSFT", "2018-01-01", 400, 95.0, seed=3),
    ], ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


@pytest.fixture
def single_ticker_frame():
    """A legacy frame with no ticker column, like the committed data_2018.csv."""
    return _series("AAPL", "2018-01-01", 251, 170.0, seed=4).drop(columns=["ticker"])

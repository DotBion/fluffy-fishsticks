"""Multi-ticker panel construction for the LSTM.

The original pipeline trained on one ticker for one year: 251 rows of AAPL
2018, of which 49 ended up in validation. Every number the project reports
rests on those 49 points. This module builds the same feature contract over
several tickers and several years, and fixes two defects that only become
visible once there is more than one ticker:

1. Sequence windows must never span two companies. A frame sorted by date
   interleaves tickers, so the naive sliding window produced windows made of
   four days of AAPL followed by six of AMZN. Sorting by (ticker, date)
   moves the problem rather than solving it: the window straddling the last
   AAPL row and the first AMZN row is still nonsense. Windows are therefore
   cut per ticker and concatenated.

2. The scaler must be fitted on the training window alone. The previous code
   called ``scaler.fit_transform`` on the whole file and split afterwards,
   so the minimum and maximum of the held-out period were baked into the
   normalisation of the training period. That is leakage, and it flatters
   the validation loss.

Prices differ by an order of magnitude across the panel (AMZN traded near
$1,500 in 2018 while AAPL traded near $170), so each ticker gets its own
scaler and the model learns the shape of a series rather than its level.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from serving.contract import FEATURE_COLS, SEQ_LENGTH, TARGET_COL

# The tweet corpus - not the market data - is what bounds the panel. Market
# history is available for any listed symbol, but the Kaggle tweet dataset
# covers exactly these six symbols over 2015-2020, and a row without
# sentiment cannot be used by a model whose sixth feature is sentiment.
ALL_TICKERS = ("AAPL", "AMZN", "GOOG", "GOOGL", "MSFT", "TSLA")

# GOOG and GOOGL are Alphabet class C and class A: two listings of one
# company, tracking each other within a percent or so, and the corpus tags
# most Alphabet tweets under both. Keeping both would let one company
# contribute twice to the reported test error, so the default panel keeps
# the voting class only. Pass ALL_TICKERS explicitly to include both.
DEFAULT_TICKERS = ("AAPL", "AMZN", "GOOGL", "MSFT", "TSLA")

TICKER_COL = "ticker"

# Chronological, not random. Adjacent trading days are highly correlated, so
# a shuffled split lets the model interpolate between neighbours it has
# already seen and reports an error far below what it would achieve on
# genuinely future data.
DEFAULT_SPLIT = {"train_end": "2018-12-31", "val_end": "2019-12-31"}


def _require_columns(df, columns, what):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing required columns: {missing}")


def build_panel(frames):
    """Stack per-ticker frames into one long panel.

    `frames` maps ticker -> DataFrame carrying the feature columns and a
    date column. The result is sorted by (ticker, date) and carries a
    `ticker` column; nothing downstream may assume the rows of one ticker
    are contiguous beyond that sort.
    """
    if not frames:
        raise ValueError("No per-ticker frames supplied.")

    parts = []
    for ticker, frame in frames.items():
        _require_columns(frame, ["date"] + list(FEATURE_COLS), f"frame for {ticker}")
        part = frame.loc[:, ["date"] + list(FEATURE_COLS)].copy()
        part[TICKER_COL] = ticker
        part["date"] = pd.to_datetime(part["date"])
        parts.append(part)

    panel = pd.concat(parts, ignore_index=True)
    duplicates = panel.duplicated(subset=[TICKER_COL, "date"]).sum()
    if duplicates:
        raise ValueError(
            f"Panel has {duplicates} duplicate (ticker, date) rows; "
            "the same trading day appears twice for one symbol."
        )
    return panel.sort_values([TICKER_COL, "date"]).reset_index(drop=True)


def as_panel(df, default_ticker="AAPL"):
    """Accept either a panel or a legacy single-ticker frame.

    data_2018.csv predates the ticker column. Rather than rewrite it, a
    frame without one is treated as a single-ticker panel, which keeps the
    committed dataset trainable by the new code path unchanged.
    """
    df = df.copy()
    if TICKER_COL not in df.columns:
        df[TICKER_COL] = default_ticker
    _require_columns(df, ["date", TICKER_COL] + list(FEATURE_COLS), "training frame")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values([TICKER_COL, "date"]).reset_index(drop=True)


def chronological_split(panel, train_end=None, val_end=None):
    """Split by calendar date, applying the same boundaries to every ticker.

    Returns (train, val, test). The test frame is empty when the panel ends
    before `val_end`, which is the normal case for the committed 2018 data.
    """
    train_end = pd.Timestamp(train_end or DEFAULT_SPLIT["train_end"])
    val_end = pd.Timestamp(val_end or DEFAULT_SPLIT["val_end"])
    if val_end < train_end:
        raise ValueError(f"val_end {val_end.date()} precedes train_end {train_end.date()}")

    dates = panel["date"]
    return (
        panel[dates <= train_end].reset_index(drop=True),
        panel[(dates > train_end) & (dates <= val_end)].reset_index(drop=True),
        panel[dates > val_end].reset_index(drop=True),
    )


def fractional_split(panel, val_frac=0.2, test_frac=0.0):
    """Split each ticker's own series chronologically by fraction.

    Date boundaries are the better tool for a multi-year panel, but they
    cannot split a single year: `train_end="2018-12-31"` puts all of
    data_2018.csv in training and leaves nothing to validate on. Splitting
    per ticker by position keeps every symbol represented in every split
    regardless of how long its history is, and is still chronological - the
    validation rows always follow the training rows in time.
    """
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError(f"Invalid split fractions: val={val_frac}, test={test_frac}")

    trains, vals, tests = [], [], []
    for _, group in panel.groupby(TICKER_COL, sort=True):
        group = group.sort_values("date")
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        trains.append(group.iloc[:n_train])
        vals.append(group.iloc[n_train:n_train + n_val])
        tests.append(group.iloc[n_train + n_val:])

    def _concat(parts):
        joined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return joined.reset_index(drop=True)

    return _concat(trains), _concat(vals), _concat(tests)


def split_panel(panel, train_end=None, val_end=None, val_frac=0.2, test_frac=0.0):
    """Split by date when boundaries are given, else by per-ticker fraction."""
    if train_end or val_end:
        return chronological_split(panel, train_end, val_end)
    return fractional_split(panel, val_frac=val_frac, test_frac=test_frac)


def fit_scalers(train_panel, feature_cols=FEATURE_COLS):
    """One MinMaxScaler per ticker, fitted on the training rows only.

    Fitting on the full series would leak the held-out period's price range
    into training. A ticker with fewer rows than a single window is rejected
    here rather than producing a scaler nothing can use.
    """
    scalers = {}
    for ticker, group in train_panel.groupby(TICKER_COL, sort=True):
        if len(group) <= SEQ_LENGTH:
            raise ValueError(
                f"{ticker} has {len(group)} training rows, too few for a "
                f"{SEQ_LENGTH}-day window plus a target."
            )
        scaler = MinMaxScaler()
        scaler.fit(group[list(feature_cols)].values)
        scalers[ticker] = scaler
    if not scalers:
        raise ValueError("Training split is empty — check the split boundaries.")
    return scalers


def make_sequences(panel, scalers, seq_length=SEQ_LENGTH, feature_cols=FEATURE_COLS):
    """Cut sliding windows per ticker and scale each with its own scaler.

    Returns (X, y, meta) where X is (n, seq_length, n_features), y is the
    scaled next-day close, and meta is a frame of the ticker and target date
    behind each row so errors can be attributed per symbol.

    Values outside the training range scale outside [0, 1]. That is correct
    and deliberately not clipped: a test period that trades above anything
    seen in training is information the metrics should reflect, not hide.
    """
    feature_cols = list(feature_cols)
    if TARGET_COL not in feature_cols:
        raise ValueError(f"feature_cols must contain the target {TARGET_COL!r}")
    # Derived from the column list actually in use, not from the contract's
    # index: the ablation trains a five-feature arm where the contract's
    # TARGET_IDX would happen to be right by coincidence and wrong in general.
    target_idx = feature_cols.index(TARGET_COL)
    windows, targets, meta_ticker, meta_date = [], [], [], []

    for ticker, group in panel.groupby(TICKER_COL, sort=True):
        scaler = scalers.get(ticker)
        if scaler is None:
            raise KeyError(
                f"No scaler for {ticker}; it appears in this split but not in "
                "the training split. Fit scalers on the training panel first."
            )
        group = group.sort_values("date")
        if len(group) <= seq_length:
            # Not an error here: a short validation slice for one ticker just
            # contributes no windows, while the rest of the panel still does.
            continue

        scaled = scaler.transform(group[feature_cols].values)
        dates = group["date"].values
        for i in range(len(scaled) - seq_length):
            windows.append(scaled[i:i + seq_length])
            targets.append(scaled[i + seq_length][target_idx])
            meta_ticker.append(ticker)
            meta_date.append(dates[i + seq_length])

    if not windows:
        raise ValueError(
            f"No sequences produced: every ticker has {seq_length} or fewer "
            "rows in this split."
        )

    meta = pd.DataFrame({TICKER_COL: meta_ticker, "target_date": meta_date})
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        meta,
    )


def describe_panel(panel):
    """One row per ticker: row count and date range. Used in training logs."""
    summary = panel.groupby(TICKER_COL)["date"].agg(["count", "min", "max"])
    summary.columns = ["rows", "first", "last"]
    return summary

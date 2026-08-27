"""Seeded ablation: does daily tweet sentiment improve next-day close prediction?

Trains two arms with identical architecture, split and window:
  baseline  = OHLCV only          (5 features)
  sentiment = OHLCV + sentiment   (6 features)

Repeats each arm over N seeds and reports mean +/- std, so the comparison
does not rest on a single run. A persistence baseline - "tomorrow closes
where today closed" - is reported alongside, because a price model that
cannot beat persistence has learned nothing worth deploying.

    python ablation.py --seeds 5
    python ablation.py --seeds 5 --data panel.csv --tickers AAPL,MSFT
"""

import argparse
import statistics
import sys

import numpy as np

from lstm_train_pytorch import config, load_panel, train
from models import FEATURE_COLS
from pipeline.dataset import TICKER_COL, split_panel

BASELINE_COLS = ["open", "high", "low", "close", "volume"]
SENTIMENT_COLS = FEATURE_COLS

REPORTED = [
    ("val_mse", "val MSE (scaled)", "{:.6f}"),
    ("val_mae_price", "MAE ($)", "{:.2f}"),
    ("val_rmse_price", "RMSE ($)", "{:.2f}"),
]


def persistence(data_path, tickers, val_frac):
    """Error of predicting the previous close, on the same validation rows.

    Reported in raw dollars so it is directly comparable to the model's MAE
    and RMSE, which are inverse-transformed back into price units.
    """
    panel = load_panel(data_path, SENTIMENT_COLS, tickers)
    _, val_panel, _ = split_panel(panel, val_frac=val_frac)

    abs_err, sq_err = [], []
    for _, group in val_panel.groupby(TICKER_COL):
        close = group.sort_values("date")["close"].values
        diff = close[1:] - close[:-1]
        abs_err.append(np.abs(diff))
        sq_err.append(diff ** 2)

    abs_err = np.concatenate(abs_err)
    sq_err = np.concatenate(sq_err)
    return {
        "val_mae_price": float(np.mean(abs_err)),
        "val_rmse_price": float(np.sqrt(np.mean(sq_err))),
        "n_val": float(len(abs_err)),
    }


def run_arm(name, feature_cols, data_path, seeds, tickers, val_frac):
    print(f"\n=== {name}: {len(feature_cols)} features, {len(seeds)} seeds ===")
    runs = []
    for seed in seeds:
        _, _, m = train(
            data_path=data_path, feature_cols=feature_cols, seed=seed,
            verbose=False, tickers=tickers, val_frac=val_frac,
        )
        runs.append(m)
        print(f"  seed {seed}: val_mse={m['val_mse']:.6f}  "
              f"MAE=${m['val_mae_price']:.2f}  RMSE=${m['val_rmse_price']:.2f}")
    return runs


def agg(runs, key):
    vals = [r[key] for r in runs]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", default="data_2018.csv")
    ap.add_argument("--tickers", default="", help="comma-separated subset of the panel")
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    seeds = list(range(args.seeds))
    tickers = [t for t in args.tickers.replace(",", " ").split() if t] or None

    naive = persistence(args.data, tickers, args.val_frac)
    baseline = run_arm("BASELINE (no sentiment)", BASELINE_COLS, args.data, seeds, tickers, args.val_frac)
    sentiment = run_arm("WITH SENTIMENT", SENTIMENT_COLS, args.data, seeds, tickers, args.val_frac)

    n_val = int(baseline[0]["n_val"])
    print(f"\n{'='*74}\nRESULTS over {len(seeds)} seeds (mean +/- std), "
          f"n_val={n_val}, epochs={config['epochs']}\n{'='*74}")
    print(f"{'metric':<18}{'baseline':>22}{'with sentiment':>22}{'delta':>12}")
    for key, label, fmt in REPORTED:
        b_m, b_s = agg(baseline, key)
        s_m, s_s = agg(sentiment, key)
        delta = (s_m - b_m) / b_m * 100 if b_m else 0.0
        print(f"{label:<18}{fmt.format(b_m)+' +/- '+fmt.format(b_s):>22}"
              f"{fmt.format(s_m)+' +/- '+fmt.format(s_s):>22}{delta:>11.1f}%")

    print(f"\n{'persistence':<18}{'MAE $'+format(naive['val_mae_price'], '.2f'):>22}"
          f"{'RMSE $'+format(naive['val_rmse_price'], '.2f'):>22}")

    b_m, b_s = agg(baseline, "val_mse")
    s_m, s_s = agg(sentiment, "val_mse")
    print(f"\nNegative delta = sentiment helps. Spread overlap check: "
          f"baseline [{b_m-b_s:.6f}, {b_m+b_s:.6f}] vs sentiment [{s_m-s_s:.6f}, {s_m+s_s:.6f}]")
    if abs(s_m - b_m) < max(b_s, s_s):
        print("WARNING: difference is smaller than one std — not a reliable effect at this sample size.")

    s_mae, _ = agg(sentiment, "val_mae_price")
    if s_mae > naive["val_mae_price"]:
        print(f"WARNING: the model's MAE (${s_mae:.2f}) is worse than predicting "
              f"yesterday's close (${naive['val_mae_price']:.2f}). On this much data "
              "that is the expected outcome, and it is the number to improve.")


if __name__ == "__main__":
    sys.exit(main())

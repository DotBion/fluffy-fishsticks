"""Seeded ablation: does daily tweet sentiment improve next-day close prediction?

Trains two arms with identical architecture, split and window:
  baseline  = OHLCV only          (5 features)
  sentiment = OHLCV + sentiment   (6 features)

Repeats each arm over N seeds and reports mean +/- std, so the comparison
does not rest on a single run.

    python ablation.py --seeds 5
"""

import argparse
import statistics

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from lstm_train_pytorch import config, create_sequences, load_data, train
from models import FEATURE_COLS

BASELINE_COLS = ["open", "high", "low", "close", "volume"]
SENTIMENT_COLS = FEATURE_COLS


def evaluate(model, scaler, data_path, feature_cols):
    """Return scaled MSE plus MAE/RMSE in price units on the held-out split."""
    _, data_scaled, _ = load_data(data_path, feature_cols)
    target_idx = feature_cols.index("close")

    X, y = create_sequences(data_scaled, config["seq_length"])
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model.eval()
    with torch.no_grad():
        pred = np.atleast_1d(model(torch.tensor(X_val, dtype=torch.float32)).numpy())

    def to_price(vals):
        padded = np.zeros((len(vals), len(feature_cols)))
        padded[:, target_idx] = vals
        return scaler.inverse_transform(padded)[:, target_idx]

    pred_price, true_price = to_price(pred), to_price(y_val)
    return {
        "val_mse_scaled": float(np.mean((pred - y_val) ** 2)),
        "mae_price": float(np.mean(np.abs(pred_price - true_price))),
        "rmse_price": float(np.sqrt(np.mean((pred_price - true_price) ** 2))),
        "n_val": len(y_val),
    }


def run_arm(name, feature_cols, data_path, seeds):
    print(f"\n=== {name}: {len(feature_cols)} features, {len(seeds)} seeds ===")
    runs = []
    for seed in seeds:
        model, scaler, _ = train(data_path=data_path, feature_cols=feature_cols, seed=seed, verbose=False)
        m = evaluate(model, scaler, data_path, feature_cols)
        runs.append(m)
        print(f"  seed {seed}: val_mse={m['val_mse_scaled']:.6f}  MAE=${m['mae_price']:.2f}  RMSE=${m['rmse_price']:.2f}")
    return runs


def agg(runs, key):
    vals = [r[key] for r in runs]
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", default="data_2018.csv")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    baseline = run_arm("BASELINE (no sentiment)", BASELINE_COLS, args.data, seeds)
    sentiment = run_arm("WITH SENTIMENT", SENTIMENT_COLS, args.data, seeds)

    print(f"\n{'='*66}\nRESULTS over {len(seeds)} seeds (mean +/- std), n_val={baseline[0]['n_val']}\n{'='*66}")
    print(f"{'metric':<18}{'baseline':>24}{'with sentiment':>24}{'delta':>8}")
    for key, label, fmt in [
        ("val_mse_scaled", "val MSE (scaled)", "{:.6f}"),
        ("mae_price", "MAE ($)", "{:.2f}"),
        ("rmse_price", "RMSE ($)", "{:.2f}"),
    ]:
        b_m, b_s = agg(baseline, key)
        s_m, s_s = agg(sentiment, key)
        delta = (s_m - b_m) / b_m * 100 if b_m else 0.0
        print(f"{label:<18}{fmt.format(b_m)+' +/- '+fmt.format(b_s):>24}"
              f"{fmt.format(s_m)+' +/- '+fmt.format(s_s):>24}{delta:>7.1f}%")

    b_m, b_s = agg(baseline, "val_mse_scaled")
    s_m, s_s = agg(sentiment, "val_mse_scaled")
    print(f"\nNegative delta = sentiment helps. Spread overlap check: "
          f"baseline [{b_m-b_s:.6f}, {b_m+b_s:.6f}] vs sentiment [{s_m-s_s:.6f}, {s_m+s_s:.6f}]")
    if abs(s_m - b_m) < max(b_s, s_s):
        print("WARNING: difference is smaller than one std — not a reliable effect at this sample size.")


if __name__ == "__main__":
    main()

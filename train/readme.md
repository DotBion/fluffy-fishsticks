# Training and inference

## Feature contract

The model consumes a fixed, ordered feature vector — defined once in `models.py`
as `FEATURE_COLS` and shared by training, evaluation and serving:

    ["open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]

Target is next-day `close`. Lookback window is 10 days.

## Train

    python lstm_train_pytorch.py

Writes two artifacts, both required to serve:

| file             | why it matters                                                     |
| ---------------- | ------------------------------------------------------------------ |
| `lstm_model.pth` | model weights                                                       |
| `scaler.pkl`     | the fitted `MinMaxScaler` — without it predictions stay in [0, 1] and cannot be mapped back to prices |

Override paths with `DATA_CSV_PATH`, `MODEL_PATH`, `SCALER_PATH`.

## Serve

    python inference_server_lstm.py        # port 5000, override with PORT

`POST /predict` takes **raw, unscaled** values. Scaling and inverse scaling
happen server-side, so the response is a price in the same units as the
training data.

    {"input": [[[170.16, 172.30, 169.26, 172.26, 25048048.0, 0.3256], ... x10 ]]}
    -> {"predictions": [172.41]}

`GET /health` reports the expected feature order and sequence length.

## Ablation

    python ablation.py --seeds 5

Trains OHLCV-only and OHLCV+sentiment arms over N seeds with identical
architecture, split and window, and reports mean +/- std for scaled MSE and
price-space MAE/RMSE. It warns when the gap between arms is smaller than one
standard deviation — at this dataset size (251 rows of AAPL 2018, ~48 held-out
points) that check matters.

### Result (5 seeds, PyTorch path, AAPL 2018, n_val=49)

| metric           | baseline (OHLCV) | with sentiment    | delta |
| ---------------- | ---------------- | ----------------- | ----- |
| val MSE (scaled) | 0.010863 ± 0.002182 | 0.010263 ± 0.003122 | -5.5% |
| MAE ($)          | 7.27 ± 0.82      | 7.04 ± 1.20       | -3.2% |
| RMSE ($)         | 8.85 ± 0.91      | 8.56 ± 1.31       | -3.3% |

The mean effect points the right way and its size (-5.5%) closely matches the
single-run Keras result in `lstm_aapl_2018_SentimentEffect_v1.ipynb` on the
`dev-nc3610` branch (0.00951 -> 0.00900, -5.4%).

**But the effect is not statistically established.** The gap between arms is
smaller than one standard deviation across seeds, and the ranges overlap
heavily. Individual seeds contradict the mean — the worst sentiment run
(0.014581) is worse than four of the five baseline runs. Do not describe this
as "sentiment improves prediction by 5%". The defensible claim is that the
direction is consistent while the magnitude is within run-to-run noise at this
dataset size.

Reducing the uncertainty needs more data, not more seeds: 251 rows of one
ticker for one year yields ~49 held-out points. Extending to multiple tickers
and several years is the meaningful next step.

## Data source

Training resolves its dataset in this order:

1. **MinIO** — when `MINIO_BUCKET`, `MINIO_TRAINING_OBJECT` and `MINIO_ENDPOINT`
   are set, it downloads the object the Airflow DAG (`stock_sentiment_etl`)
   produced. This is the wired-up path: the pipeline's output is what trains.
2. **Local CSV** — `DATA_CSV_PATH`, defaulting to `data_2018.csv`, so a fresh
   clone can train with no infrastructure running.

The resolved source is printed at startup and recorded as the `data_source`
param on the MLflow run.

## Experiment tracking

Set `MLFLOW_TRACKING_URI` to log params, metrics, the scaler artifact, and the
model, registered as `FinPulseLSTM` (override with `MLFLOW_MODEL_NAME`):

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python lstm_train_pytorch.py

Unset, training runs normally and skips tracking. Note MLflow 3.x rejects
filesystem stores (`file://./mlruns`) by default — use SQLite or the Postgres
backend the compose stack provides.

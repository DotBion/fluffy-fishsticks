# Training and inference

## Feature contract

The model consumes a fixed, ordered feature vector — defined once in
`serving/contract.py` as `FEATURE_COLS` and imported by training, evaluation
and serving so the three cannot drift:

    ["open", "high", "low", "close", "volume", "daily_avg_sentiment_score"]

Target is next-day `close`. Lookback window is 10 days.

## Datasets

Two shapes are accepted, and the difference is one column.

| file | shape | rows |
| --- | --- | --- |
| `data_2018.csv` | no `ticker` column — treated as a single-ticker panel | 251 (AAPL, 2018) |
| a panel built by `pipeline/build_panel.py` | carries `ticker` | 5 tickers × up to 6 years |

Build a panel with:

    python -m pipeline.build_panel --out train/panel.csv \
        --tweet-csv /path/Tweet.csv --company-tweet-csv /path/Company_Tweet.csv

The tweet corpus bounds what is possible, not the market data. Kaggle's
"Tweet Sentiment's Impact on Stock Returns" covers AAPL, AMZN, GOOG, GOOGL,
MSFT and TSLA over 2015–2020; a row with no sentiment cannot feed a model
whose sixth feature is sentiment. The corpus is ~3M rows and is not committed
here. `train/tweets_2018_limited.csv` is a 10,000-row slice of it, used by
the tests.

GOOG and GOOGL are two listings of Alphabet. `DEFAULT_TICKERS` keeps GOOGL
only, so one company does not contribute twice to the reported error.

## Train

    python lstm_train_pytorch.py                       # data_2018.csv
    DATA_CSV_PATH=panel.csv python lstm_train_pytorch.py   # the panel
    TICKERS=AAPL,MSFT DATA_CSV_PATH=panel.csv python lstm_train_pytorch.py

Writes two artifacts, both required to serve:

| file | why it matters |
| --- | --- |
| `lstm_model.pth` | model weights |
| `scaler.pkl` | the fitted scalers — without them predictions stay in [0, 1] and cannot be mapped back to prices |

`scaler.pkl` holds a `{ticker: MinMaxScaler}` mapping. One scaler per ticker
is not an optimisation: AMZN traded near \$1,500 in 2018 and MSFT near \$95,
so a shared min–max would compress the cheaper symbol into a sliver of the
range and the model would spend its capacity on price level rather than
shape. Serving reads the mapping through `ScalerBundle`, which also accepts
the bare scaler that older artifacts contain.

Override paths with `DATA_CSV_PATH`, `MODEL_PATH`, `SCALER_PATH`.

### Splitting

Chronological, never shuffled — adjacent trading days are highly correlated,
so a random split lets the model interpolate between neighbours it has
already seen.

    VAL_FRACTION=0.2                      # default: last 20% of each ticker
    TRAIN_END=2018-12-31 VAL_END=2019-12-31   # explicit dates, for a multi-year panel

Date boundaries are clearer for a panel but cannot split a single year, which
is why the fraction is the default.

Scalers are fitted on the **training rows only**. Fitting on the whole file
— which this script used to do — leaks the held-out period's price range into
the normalisation and flatters the validation loss.

## Serve

    BACKEND=torch MODEL_PATH=train/lstm_model.pth SCALER_PATH=train/scaler.pkl \
      python -m serving.app

`POST /predict` takes **raw, unscaled** values. Scaling and inverse scaling
happen server-side, so the response is a price in the same units as the
training data.

    {"input": [[[170.16, 172.30, 169.26, 172.26, 25048048.0, 0.3256], ... x10 ]],
     "ticker": "AAPL"}
    -> {"predictions": [172.41], "backend": "torch", "ticker": "AAPL"}

`ticker` is optional for a single-ticker model and required for a
multi-ticker one — guessing which scaler to use would return a confident,
meaningless price. `GET /health` reports the feature order, the window
length, and the tickers the loaded model covers.

## Ablation

    python ablation.py --seeds 5
    python ablation.py --seeds 5 --data panel.csv --tickers AAPL,MSFT

Trains OHLCV-only and OHLCV+sentiment arms over N seeds with identical
architecture, split and window, and reports mean ± std for scaled MSE and
price-space MAE/RMSE, alongside a persistence baseline.

### Result (5 seeds, PyTorch path, AAPL 2018, n_val=40)

| metric | baseline (OHLCV) | with sentiment | delta |
| --- | --- | --- | --- |
| val MSE (scaled) | 0.015831 ± 0.002598 | 0.014758 ± 0.003097 | −6.8% |
| MAE (\$) | 8.15 ± 0.73 | 7.85 ± 0.95 | −3.7% |
| RMSE (\$) | 9.65 ± 0.81 | 9.30 ± 1.00 | −3.6% |
| **persistence** | **MAE \$4.02** | **RMSE \$5.06** | |

Two things to read off that table, in order of importance.

**The model loses to persistence.** Predicting that tomorrow closes where
today closed gives MAE \$4.02; the model gives \$7.85, roughly twice the
error. On 191 training sequences that is the expected outcome, and it is the
number any future claim about this model has to beat. Nothing here supports
a claim that the model predicts prices usefully.

**The sentiment effect is not statistically established.** The mean points
the right way, but the gap between arms is smaller than one standard
deviation across seeds and the ranges overlap heavily. The defensible claim
is that the direction is consistent while the magnitude is within
run-to-run noise at this dataset size — not "sentiment improves prediction
by 7%".

These numbers are higher than the ones this file used to report
(0.010863 / 0.010263, n_val=49). That earlier run fitted the scaler on the
whole file before splitting, so the validation period's price range was
baked into the normalisation. The current numbers are the leak-free ones.
`ablation_results.txt` carries the full output.

## A caveat on the committed artifacts

`data_2018.csv`'s sentiment column, and therefore `lstm_model.pth`,
`scaler.pkl` and `App/src/lstm_model.onnx`, were produced before a bug in the
finance lexicon was found: 30 of its 68 terms were stored capitalised
("Bullish", "Plunge", "Disaster"), and VADER lower-cases every token before
looking it up, so those entries never matched anything. The lexicon is fixed
in `pipeline/sentiment.py` and `make test` now asserts no key can be
capitalised, but regenerating the CSV needs the full tweet corpus. Until it
is regenerated, the committed sentiment feature reflects 38 active terms, not
68.

## Data source

Training resolves its dataset in this order:

1. **MinIO** — when `MINIO_BUCKET`, `MINIO_TRAINING_OBJECT` and
   `MINIO_ENDPOINT` are set, it downloads the object the Airflow DAG
   (`stock_sentiment_etl`) produced. This is the wired-up path: the
   pipeline's output is what trains.
2. **Local CSV** — `DATA_CSV_PATH`, defaulting to `data_2018.csv`, so a fresh
   clone can train with no infrastructure running.

The resolved source is printed at startup and recorded as the `data_source`
param on the MLflow run.

## Experiment tracking

Set `MLFLOW_TRACKING_URI` to log params, metrics, the scaler artifact, and the
model, registered as `FinPulseLSTM` (override with `MLFLOW_MODEL_NAME`):

    MLFLOW_TRACKING_URI=sqlite:///mlflow.db python lstm_train_pytorch.py

Per-ticker validation errors are logged as separate metrics
(`val_mae_price_AAPL`, `val_rmse_price_AMZN`, …) so a panel run shows which
symbol carries the error rather than one averaged number.

Unset, training runs normally and skips tracking. Note MLflow 3.x rejects
filesystem stores (`file://./mlruns`) by default — use SQLite or the Postgres
backend the compose stack provides.

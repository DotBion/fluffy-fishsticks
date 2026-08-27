"""Daily ETL: market data + tweet sentiment -> multi-ticker training panel in MinIO.

Produces the OHLCV+sentiment CSV that train/lstm_train_pytorch.py consumes,
so the pipeline and the model are actually connected.

The DAG runs over a list of tickers rather than one. Alpha Vantage's free
tier allows 25 requests a day, and `outputsize=full` returns the whole
history in a single call, so a five-ticker panel costs five requests per run
regardless of how many years it covers.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from minio import Minio

sys.path.insert(0, "/opt/airflow")
from pipeline.dataset import DEFAULT_TICKERS, build_panel, describe_panel  # noqa: E402
from pipeline.market import fetch_panel  # noqa: E402
from pipeline.sentiment import join_market_and_sentiment, score_panel  # noqa: E402

default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "stock_sentiment_etl",
    default_args=default_args,
    description="Fetch OHLCV, score tweet sentiment, join into a panel, upload to MinIO",
    schedule_interval="@daily",
    catchup=False,
)

TICKERS = [t for t in os.environ.get("TICKERS", ",".join(DEFAULT_TICKERS))
           .replace(",", " ").split() if t]
TEMP_DIR = "/tmp/alpha_vantage"

# Window is configurable rather than hardcoded to 2018. The previous version
# ran @daily but always re-filtered to 2018-01-01..2018-12-31, so every run
# rewrote the identical object forever. The defaults are the tweet corpus's
# extent — outside it the sentiment join returns nothing.
WINDOW_START = os.environ.get("WINDOW_START", "2015-01-01")
WINDOW_END = os.environ.get("WINDOW_END", "2020-12-31")

TWEET_CSV = os.environ.get("TWEET_CSV", "/mnt/block/kaggle_datasets/Tweet.csv")
COMPANY_TWEET_CSV = os.environ.get("COMPANY_TWEET_CSV", "/mnt/block/kaggle_datasets/Company_Tweet.csv")

BUCKET = os.environ.get("MINIO_BUCKET", "stock-data")
TRAINING_OBJECT = os.environ.get("MINIO_TRAINING_OBJECT", "panel/training_data.csv")

PANEL_PATH = f"{TEMP_DIR}/training_panel.csv"


def _market_path(symbol):
    return f"{TEMP_DIR}/{symbol}_market.csv"


def _sentiment_path(symbol):
    return f"{TEMP_DIR}/{symbol}_sentiment.csv"


def _minio_client():
    return Minio(
        os.environ.get("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def extract_market_data():
    """Fetch and window OHLCV for every ticker.

    Extraction and column renaming are one task rather than two:
    pipeline.market returns the contract's column names directly, so a
    separate transform step would only copy a file.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    frames = fetch_panel(TICKERS, WINDOW_START, WINDOW_END)
    for symbol, frame in frames.items():
        frame.to_csv(_market_path(symbol), index=False)


def score_daily_sentiment():
    """Score tweets with finance-augmented VADER and average per ticker per day."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    scored = score_panel(TWEET_CSV, COMPANY_TWEET_CSV, TICKERS, WINDOW_START, WINDOW_END)
    for symbol, frame in scored.items():
        frame.to_csv(_sentiment_path(symbol), index=False)


def build_training_set():
    """Join each ticker's market and sentiment, then stack into one panel."""
    joined = {}
    for symbol in TICKERS:
        market_path, sentiment_path = _market_path(symbol), _sentiment_path(symbol)
        if not (os.path.exists(market_path) and os.path.exists(sentiment_path)):
            print(f"[warn] skipping {symbol}: missing market or sentiment output")
            continue
        merged = join_market_and_sentiment(
            pd.read_csv(market_path), pd.read_csv(sentiment_path)
        )
        if merged.empty:
            print(f"[warn] skipping {symbol}: market and sentiment share no dates")
            continue
        joined[symbol] = merged

    if not joined:
        raise ValueError(
            "Every ticker lost its market/sentiment join — check the date window "
            f"({WINDOW_START} .. {WINDOW_END}) against the tweet corpus's extent."
        )

    panel = build_panel(joined)
    panel.to_csv(PANEL_PATH, index=False)
    print(describe_panel(panel).to_string())
    print(f"Training panel: {len(panel)} rows across {len(joined)} tickers")


def upload_to_minio():
    client = _minio_client()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    client.fput_object(BUCKET, TRAINING_OBJECT, PANEL_PATH)
    print(f"Uploaded {TRAINING_OBJECT} to {BUCKET}")

    # The per-ticker intermediates go up too: they are what makes a bad
    # sentiment score or a gap in the market data traceable to one symbol.
    for symbol in TICKERS:
        for local, obj in ((_market_path(symbol), f"{symbol}/market_window.csv"),
                           (_sentiment_path(symbol), f"{symbol}/daily_sentiment.csv")):
            if os.path.exists(local):
                client.fput_object(BUCKET, obj, local)


extract_task = PythonOperator(task_id="extract_market_data", python_callable=extract_market_data, dag=dag)
sentiment_task = PythonOperator(task_id="score_daily_sentiment", python_callable=score_daily_sentiment, dag=dag)
join_task = PythonOperator(task_id="build_training_set", python_callable=build_training_set, dag=dag)
upload_task = PythonOperator(task_id="upload_to_minio", python_callable=upload_to_minio, dag=dag)

# Market extraction and sentiment scoring are independent; both feed the join.
extract_task >> join_task
sentiment_task >> join_task
join_task >> upload_task

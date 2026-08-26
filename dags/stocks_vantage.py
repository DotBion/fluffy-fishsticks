"""Daily ETL: market data + tweet sentiment -> training set in MinIO.

Produces the OHLCV+sentiment CSV that train/lstm_train_pytorch.py consumes,
so the pipeline and the model are actually connected.
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from alpha_vantage.timeseries import TimeSeries
from minio import Minio

sys.path.insert(0, "/opt/airflow")
from pipeline.sentiment import (  # noqa: E402
    daily_average,
    join_market_and_sentiment,
    load_tweets,
    score_tweets,
)

default_args = {
    "owner": "airflow",
    "start_date": datetime(2023, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "stock_sentiment_etl",
    default_args=default_args,
    description="Fetch OHLCV, score tweet sentiment, join, and upload to MinIO",
    schedule_interval="@daily",
    catchup=False,
)

API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
SYMBOL = os.environ.get("TICKER", "AAPL")
TEMP_DIR = "/tmp/alpha_vantage"

# Window is configurable rather than hardcoded to 2018. The previous version
# ran @daily but always re-filtered to 2018-01-01..2018-12-31, so every run
# rewrote the identical object forever.
WINDOW_START = os.environ.get("WINDOW_START", "2018-01-01")
WINDOW_END = os.environ.get("WINDOW_END", "2018-12-31")

TWEET_CSV = os.environ.get("TWEET_CSV", "/mnt/block/kaggle_datasets/Tweet.csv")
COMPANY_TWEET_CSV = os.environ.get("COMPANY_TWEET_CSV", "/mnt/block/kaggle_datasets/Company_Tweet.csv")

BUCKET = os.environ.get("MINIO_BUCKET", "stock-data")
TRAINING_OBJECT = f"{SYMBOL}/training_data.csv"

RAW_PATH = f"{TEMP_DIR}/{SYMBOL}_daily.csv"
MARKET_PATH = f"{TEMP_DIR}/{SYMBOL}_market_window.csv"
SENTIMENT_PATH = f"{TEMP_DIR}/{SYMBOL}_daily_sentiment.csv"
TRAINING_PATH = f"{TEMP_DIR}/{SYMBOL}_training_data.csv"


def _minio_client():
    return Minio(
        os.environ.get("MINIO_ENDPOINT", "minio:9000"),
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def extract_stock_data():
    if not API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set in the Airflow environment.")
    os.makedirs(TEMP_DIR, exist_ok=True)
    ts = TimeSeries(key=API_KEY, output_format="pandas")
    data, _ = ts.get_daily(symbol=SYMBOL, outputsize="full")
    data.index = pd.to_datetime(data.index)
    data.to_csv(RAW_PATH)


def transform_market_data():
    df = pd.read_csv(RAW_PATH, index_col=0, parse_dates=True)
    df = df[(df.index >= WINDOW_START) & (df.index <= WINDOW_END)]
    if df.empty:
        raise ValueError(f"No market rows between {WINDOW_START} and {WINDOW_END}")
    df = df.rename(columns={
        "1. open": "open", "2. high": "high", "3. low": "low",
        "4. close": "close", "5. volume": "volume",
    })
    df.index.name = "date"
    df.reset_index().to_csv(MARKET_PATH, index=False)


def score_daily_sentiment():
    """Score tweets with finance-augmented VADER and average per day."""
    tweets = load_tweets(TWEET_CSV, COMPANY_TWEET_CSV)
    scored = score_tweets(tweets, SYMBOL, WINDOW_START, WINDOW_END)
    daily_average(scored).to_csv(SENTIMENT_PATH, index=False)


def build_training_set():
    market = pd.read_csv(MARKET_PATH)
    sentiment = pd.read_csv(SENTIMENT_PATH)
    merged = join_market_and_sentiment(market, sentiment)
    if merged.empty:
        raise ValueError("Market/sentiment join produced no rows — check the date window.")
    merged.to_csv(TRAINING_PATH, index=False)
    print(f"Training set: {len(merged)} rows, {merged.date.min()} .. {merged.date.max()}")


def upload_to_minio():
    client = _minio_client()
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)
    for local, obj in ((TRAINING_PATH, TRAINING_OBJECT),
                       (SENTIMENT_PATH, f"{SYMBOL}/daily_sentiment.csv"),
                       (MARKET_PATH, f"{SYMBOL}/market_window.csv")):
        client.fput_object(BUCKET, obj, local)
        print(f"Uploaded {obj} to {BUCKET}")


extract_task = PythonOperator(task_id="extract_stock_data", python_callable=extract_stock_data, dag=dag)
transform_task = PythonOperator(task_id="transform_market_data", python_callable=transform_market_data, dag=dag)
sentiment_task = PythonOperator(task_id="score_daily_sentiment", python_callable=score_daily_sentiment, dag=dag)
join_task = PythonOperator(task_id="build_training_set", python_callable=build_training_set, dag=dag)
upload_task = PythonOperator(task_id="upload_to_minio", python_callable=upload_to_minio, dag=dag)

# Market extraction and sentiment scoring are independent; both feed the join.
extract_task >> transform_task >> join_task
sentiment_task >> join_task
join_task >> upload_task

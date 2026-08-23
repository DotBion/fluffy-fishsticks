from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from minio import Minio
import pandas as pd
import os

# ------------------- DAG CONFIG ------------------- #

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}
dag = DAG(
    'stock_etl_alpha_vantage_minio_only',
    default_args=default_args,
    description='Fetch stock data and upload to MinIO without host storage',
    schedule_interval='@daily',
    catchup=False
)

API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
SYMBOL = 'AAPL'
TEMP_DIR = '/tmp/alpha_vantage'  # temp space inside the container
FILENAME = f"{SYMBOL}_daily.csv"
TRANSFORMED_FILENAME = f"{SYMBOL}_daily_2018.csv"

# ------------------- TASK FUNCTIONS ------------------- #

def extract_stock_data():
    if not API_KEY:
        raise RuntimeError("ALPHA_VANTAGE_API_KEY is not set in the Airflow environment.")
    os.makedirs(TEMP_DIR, exist_ok=True)
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=SYMBOL, outputsize='full')
    data.index = pd.to_datetime(data.index)
    data.to_csv(f"{TEMP_DIR}/{FILENAME}")

def transform_data():
    df = pd.read_csv(f"{TEMP_DIR}/{FILENAME}", index_col=0, parse_dates=True)
    df_2018 = df[(df.index >= '2018-01-01') & (df.index <= '2018-12-31')]
    df_2018 = df_2018.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    })
    df_2018.to_csv(f"{TEMP_DIR}/{TRANSFORMED_FILENAME}")

def load_to_minio():
    client = Minio(
        "minio:9000",  # or "localhost:9000" if accessed outside Docker
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=False
    )

    bucket = "stock-data"
    object_name = f"{SYMBOL}/{TRANSFORMED_FILENAME}"
    file_path = f"{TEMP_DIR}/{TRANSFORMED_FILENAME}"

    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    client.fput_object(bucket, object_name, file_path)
    print(f"Uploaded {object_name} to MinIO bucket {bucket}")

# ------------------- TASK DEFINITIONS ------------------- #

extract_task = PythonOperator(
    task_id='extract_stock_data',
    python_callable=extract_stock_data,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

upload_task = PythonOperator(
    task_id='upload_to_minio',
    python_callable=load_to_minio,
    dag=dag
)

# ------------------- TASK DEPENDENCIES ------------------- #

extract_task >> transform_task >> upload_task

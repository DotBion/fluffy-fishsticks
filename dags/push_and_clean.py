from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from minio import Minio
import os

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    dag_id='etl_kaggle_to_minio',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='ETL DAG to push Kaggle datasets from /mnt/block to MinIO'
)

MINIO_CLIENT = Minio(
    "minio:9000",  # use "localhost:9000" if running outside Docker
    access_key=os.environ.get("MINIO_ROOT_USER", ""),
    secret_key=os.environ.get("MINIO_ROOT_PASSWORD", ""),
    secure=False
)

def upload_directory_to_minio(local_dir, minio_prefix):
    bucket = "kaggle-data"
    if not MINIO_CLIENT.bucket_exists(bucket):
        MINIO_CLIENT.make_bucket(bucket)

    for fname in os.listdir(local_dir):
        local_path = os.path.join(local_dir, fname)
        if os.path.isfile(local_path):
            object_name = f"{minio_prefix}/{fname}"
            MINIO_CLIENT.fput_object(bucket, object_name, local_path)
            print(f"Uploaded {object_name}")

def upload_financial_tweets():
    upload_directory_to_minio(
        "/mnt/block/kaggle_datasets/financial_tweets", "financial_tweets"
    )

def upload_top_company_tweets():
    upload_directory_to_minio(
        "/mnt/block/kaggle_datasets/top_companies", "top_companies"
    )

upload_financial = PythonOperator(
    task_id='upload_financial_tweets_to_minio',
    python_callable=upload_financial_tweets,
    dag=dag
)

upload_companies = PythonOperator(
    task_id='upload_top_companies_to_minio',
    python_callable=upload_top_company_tweets,
    dag=dag
)

upload_financial >> upload_companies

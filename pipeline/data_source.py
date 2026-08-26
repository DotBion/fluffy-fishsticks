"""Resolve the training dataset from MinIO/S3 when available, else local disk.

Lets the DAG's output feed training without forcing a MinIO dependency on
anyone running the training script standalone.
"""

import os
import tempfile

import pandas as pd


def resolve_training_csv(local_fallback="data_2018.csv"):
    """Return (path, source_description).

    Prefers the object written by the Airflow DAG; falls back to the committed
    CSV so a fresh clone can train with no infrastructure running.
    """
    bucket = os.environ.get("MINIO_BUCKET")
    obj = os.environ.get("MINIO_TRAINING_OBJECT")
    endpoint = os.environ.get("MINIO_ENDPOINT")

    if bucket and obj and endpoint:
        try:
            from minio import Minio

            client = Minio(
                endpoint,
                access_key=os.environ["MINIO_ROOT_USER"],
                secret_key=os.environ["MINIO_ROOT_PASSWORD"],
                secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
            )
            dest = os.path.join(tempfile.gettempdir(), os.path.basename(obj))
            client.fget_object(bucket, obj, dest)
            return dest, f"minio://{bucket}/{obj}"
        except Exception as e:
            print(f"[warn] MinIO fetch failed ({e}); falling back to {local_fallback}")

    path = os.environ.get("DATA_CSV_PATH", local_fallback)
    return path, f"file://{os.path.abspath(path)}"


def load_frame(path):
    return pd.read_csv(path).sort_values(by="date")

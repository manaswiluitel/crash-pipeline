

# cleaner/minio_io.py

from minio import Minio

import os



MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")

MINIO_USER = os.getenv("MINIO_USER")

MINIO_PASS = os.getenv("MINIO_PASS")

MINIO_SECURE = False



def minio_client():

    return Minio(MINIO_ENDPOINT, access_key=MINIO_USER, secret_key=MINIO_PASS, secure=MINIO_SECURE)



def fetch_silver_csv(bucket: str, corr_id: str) -> str:

    cli = minio_client()

    key = f"crash/corr={corr_id}/merged.csv"

    local_path = f"/tmp/{corr_id}_merged.csv"

    cli.fget_object(bucket, key, local_path)

    return local_path


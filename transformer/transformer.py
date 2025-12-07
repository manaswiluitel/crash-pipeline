# transformer/transformer.py
import os
import io
import json
import gzip
import socket
import logging
import time
import random
import traceback
from typing import List, Dict, Any
from pathlib import Path
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server

import pika
from minio import Minio
from minio.error import S3Error
import polars as pl

# ---------------------------------
# Prometheus Metrics
# ---------------------------------

# 1. Service uptime (implicit via process_start_time_seconds - auto-tracked by prometheus_client)
service_info = Info('transformer_service', 'Service information and uptime tracking')
service_info.info({'version': '1.0', 'service': 'transformer'})

# 2. Run/Request count
jobs_processed_total = Counter(
    'transformer_jobs_processed_total',
    'Total number of transformation jobs processed',
    ['status']  # success, failure, invalid
)

# 3. Error count (failures)
errors_total = Counter(
    'transformer_errors_total',
    'Total errors encountered by type',
    ['error_type']  # minio, rabbitmq, parsing, merging, writing
)

operations_total = Counter(
    'transformer_operations_total',
    'Total operations by type and outcome',
    ['operation', 'status']  # operation=load/merge/write/publish, status=success/failure
)

# 4. Latency of each run
job_duration_seconds = Histogram(
    'transformer_job_duration_seconds',
    'Duration of transformation job processing',
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1200]  # 1s to 20min
)

# 5. Rows processed
rows_input_total = Counter(
    'transformer_rows_input_total',
    'Total number of input rows by dataset',
    ['dataset', 'corr_id']  # crashes, vehicles, people
)

rows_output_total = Counter(
    'transformer_rows_output_total',
    'Total number of output rows after merging',
    ['corr_id']
)

rows_processed_gauge = Gauge(
    'transformer_rows_in_last_job',
    'Number of rows in the most recent job',
    ['dataset']  # crashes, vehicles, people, merged
)

columns_processed_gauge = Gauge(
    'transformer_columns_in_last_job',
    'Number of columns in the most recent merged output'
)

# 6. Duration of each major function
function_duration_seconds = Histogram(
    'transformer_function_duration_seconds',
    'Duration of major transformation functions',
    ['function'],  # load_dataset, merge, make_csv_safe, write_csv, publish_clean
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60, 120]
)

# 7. Success vs failure counters
stage_operations_total = Counter(
    'transformer_stage_operations_total',
    'Operations by transformation stage and outcome',
    ['stage', 'status']  # stage=load/standardize/aggregate/merge/write, status=success/failure
)

# Additional useful metrics
current_jobs_gauge = Gauge(
    'transformer_current_jobs',
    'Number of jobs currently being processed'
)

queue_messages_processed_total = Counter(
    'transformer_queue_messages_processed_total',
    'Total number of queue messages processed'
)

minio_operations_total = Counter(
    'transformer_minio_operations_total',
    'MinIO operations',
    ['operation', 'status']  # operation=list/read/write/bucket_check, status=success/failure
)

rabbitmq_operations_total = Counter(
    'transformer_rabbitmq_operations_total',
    'RabbitMQ operations',
    ['operation', 'status']  # operation=publish/consume, status=success/failure
)

objects_processed_total = Counter(
    'transformer_objects_processed_total',
    'Total MinIO objects processed',
    ['dataset']  # crashes, vehicles, people
)

merge_join_rows = Gauge(
    'transformer_merge_join_rows',
    'Rows involved in merge operations',
    ['join_type']  # left_crashes, right_vehicles, right_people
)

data_quality_metrics = Gauge(
    'transformer_data_quality',
    'Data quality metrics from last job',
    ['metric']  # empty_datasets, null_percentage, duplicate_keys
)

# ---------------------------------
# Logging
# ---------------------------------
logging.basicConfig(level=logging.INFO, format="[transformer] %(message)s")
logging.getLogger("pika").setLevel(logging.WARNING)

# ---------------------------------
# Env / Config
# ---------------------------------
RABBIT_URL       = os.getenv("RABBITMQ_URL")
TRANSFORM_QUEUE  = os.getenv("TRANSFORM_QUEUE", "transform")
CLEAN_QUEUE      = os.getenv("CLEAN_QUEUE", "clean")
MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS     = os.getenv("MINIO_USER")
MINIO_SECRET     = os.getenv("MINIO_PASS")
MINIO_SECURE     = False
RAW_BUCKET       = os.getenv("RAW_BUCKET")
XFORM_BUCKET_ENV = os.getenv("XFORM_BUCKET")
PREFIX           = "crash"
METRICS_PORT     = int(os.getenv("METRICS_PORT", "8000"))

# ---------------------------------
# MinIO client
# ---------------------------------
def minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    )

# ---------------------------------
# Object helpers
# ---------------------------------
def list_objects_recursive(cli: Minio, bucket: str, prefix: str) -> List[str]:
    """List objects with metrics tracking."""
    with function_duration_seconds.labels(function='list_objects').time():
        try:
            out = []
            for obj in cli.list_objects(bucket, prefix=prefix, recursive=True):
                if getattr(obj, "is_dir", False):
                    continue
                out.append(obj.object_name)
            minio_operations_total.labels(operation='list', status='success').inc()
            return out
        except Exception as e:
            minio_operations_total.labels(operation='list', status='failure').inc()
            errors_total.labels(error_type='minio').inc()
            raise

def read_json_gz_array(cli: Minio, bucket: str, key: str) -> List[Dict[str, Any]]:
    """Read JSON/JSONL gzipped array with metrics tracking."""
    with function_duration_seconds.labels(function='read_object').time():
        resp = None
        data = b""
        try:
            resp = cli.get_object(bucket, key)
            data = resp.read()
            minio_operations_total.labels(operation='read', status='success').inc()
        except Exception as e:
            minio_operations_total.labels(operation='read', status='failure').inc()
            errors_total.labels(error_type='minio').inc()
            raise
        finally:
            try:
                if resp is not None:
                    resp.close()
                    resp.release_conn()
            except Exception:
                pass

        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            try:
                payload = gzip.decompress(data)
            except OSError:
                payload = data
        else:
            payload = data

        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")

        try:
            arr = json.loads(text)
        except json.JSONDecodeError:
            errors_total.labels(error_type='parsing').inc()
            return []

        if isinstance(arr, list):
            return arr
        if isinstance(arr, dict) and isinstance(arr.get("data"), list):
            return arr["data"]
        return []

def write_csv(cli: Minio, bucket: str, key: str, df: pl.DataFrame) -> None:
    """Write CSV to MinIO with metrics tracking."""
    with function_duration_seconds.labels(function='write_csv').time():
        try:
            buf = io.BytesIO()
            df.write_csv(buf)
            data = buf.getvalue()
            cli.put_object(
                bucket,
                key,
                data=io.BytesIO(data),
                length=len(data),
                content_type="text/csv; charset=utf-8",
            )
            minio_operations_total.labels(operation='write', status='success').inc()
            operations_total.labels(operation='write', status='success').inc()
            stage_operations_total.labels(stage='write_csv', status='success').inc()
        except Exception as e:
            minio_operations_total.labels(operation='write', status='failure').inc()
            operations_total.labels(operation='write', status='failure').inc()
            stage_operations_total.labels(stage='write_csv', status='failure').inc()
            errors_total.labels(error_type='minio').inc()
            raise

# ---------------------------------
# Load & merge
# ---------------------------------
def _keys_for_corr(cli: Minio, bucket: str, prefix: str, dataset_alias: str, corr: str) -> List[str]:
    """Get keys for a specific correlation ID."""
    base = f"{prefix}/{dataset_alias}/"
    keys = list_objects_recursive(cli, bucket, base)
    needle = f"/corr={corr}/"
    filtered = [k for k in keys if (k.endswith(".json.gz") or k.endswith(".json")) and needle in k]
    objects_processed_total.labels(dataset=dataset_alias).inc(len(filtered))
    return filtered

def load_dataset(cli: Minio, raw_bucket: str, prefix: str, dataset_alias: str, corr: str) -> pl.DataFrame:
    """Load dataset with metrics tracking."""
    with function_duration_seconds.labels(function=f'load_dataset_{dataset_alias}').time():
        try:
            keys = _keys_for_corr(cli, raw_bucket, prefix, dataset_alias, corr)
            rows_all: List[Dict[str, Any]] = []
            
            for k in keys:
                rows = read_json_gz_array(cli, raw_bucket, k)
                if rows:
                    rows_all.extend(rows)
            
            df = pl.DataFrame(rows_all) if rows_all else pl.DataFrame()
            
            # Track metrics
            row_count = len(df) if not df.is_empty() else 0
            rows_input_total.labels(dataset=dataset_alias, corr_id=corr).inc(row_count)
            rows_processed_gauge.labels(dataset=dataset_alias).set(row_count)
            
            if row_count == 0:
                data_quality_metrics.labels(metric=f'empty_{dataset_alias}').set(1)
            else:
                data_quality_metrics.labels(metric=f'empty_{dataset_alias}').set(0)
            
            operations_total.labels(operation='load', status='success').inc()
            stage_operations_total.labels(stage=f'load_{dataset_alias}', status='success').inc()
            
            logging.info(f"Loaded {dataset_alias}: {row_count} rows from {len(keys)} objects")
            return df
            
        except Exception as e:
            operations_total.labels(operation='load', status='failure').inc()
            stage_operations_total.labels(stage=f'load_{dataset_alias}', status='failure').inc()
            errors_total.labels(error_type='minio').inc()
            raise

def basic_standardize(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize dataframe with metrics tracking."""
    with function_duration_seconds.labels(function='standardize').time():
        try:
            if df.is_empty():
                return df
            
            initial_rows = len(df)
            df = df.rename({c: c.strip().lower() for c in df.columns})
            df = df.unique(maintain_order=True)
            
            duplicates = initial_rows - len(df)
            if duplicates > 0:
                data_quality_metrics.labels(metric='duplicate_rows_removed').set(duplicates)
            
            stage_operations_total.labels(stage='standardize', status='success').inc()
            return df
            
        except Exception as e:
            stage_operations_total.labels(stage='standardize', status='failure').inc()
            errors_total.labels(error_type='merging').inc()
            raise

def aggregate_many_to_one(df: pl.DataFrame, id_col: str, prefix: str) -> pl.DataFrame:
    """Aggregate many-to-one relationships with metrics tracking."""
    with function_duration_seconds.labels(function=f'aggregate_{prefix}').time():
        try:
            if df.is_empty():
                return df
            
            keep_fields = [c for c in df.columns if c != id_col]
            aggs = [pl.len().alias(f"{prefix}_count")]
            
            for c in keep_fields:
                aggs.append(
                    pl.col(c).drop_nulls().cast(pl.Utf8).unique().sort().implode().alias(f"{prefix}_{c}_list")
                )
            
            result = df.group_by(id_col, maintain_order=True).agg(aggs)
            stage_operations_total.labels(stage=f'aggregate_{prefix}', status='success').inc()
            return result
            
        except Exception as e:
            stage_operations_total.labels(stage=f'aggregate_{prefix}', status='failure').inc()
            errors_total.labels(error_type='merging').inc()
            raise

def merge_crash_vehicles_people(
    crashes: pl.DataFrame,
    vehicles: pl.DataFrame,
    people: pl.DataFrame,
    id_col: str
) -> pl.DataFrame:
    """Merge datasets with comprehensive metrics tracking."""
    with function_duration_seconds.labels(function='merge_datasets').time():
        try:
            crashes = basic_standardize(crashes)
            vehicles = basic_standardize(vehicles)
            people = basic_standardize(people)

            id_lower = id_col.lower()

            def _ensure_id(df: pl.DataFrame) -> pl.DataFrame:
                if df.is_empty() or id_lower in df.columns:
                    return df
                for c in df.columns:
                    if c.lower() == id_lower:
                        return df.rename({c: id_lower})
                return df

            crashes = _ensure_id(crashes)
            vehicles = _ensure_id(vehicles)
            people = _ensure_id(people)

            if not crashes.is_empty() and id_lower not in crashes.columns:
                return crashes

            # Track merge join sizes
            merge_join_rows.labels(join_type='left_crashes').set(len(crashes) if not crashes.is_empty() else 0)
            merge_join_rows.labels(join_type='right_vehicles').set(len(vehicles) if not vehicles.is_empty() else 0)
            merge_join_rows.labels(join_type='right_people').set(len(people) if not people.is_empty() else 0)

            veh_agg = aggregate_many_to_one(vehicles, id_lower, prefix="veh") if (not vehicles.is_empty() and id_lower in vehicles.columns) else pl.DataFrame()
            ppl_agg = aggregate_many_to_one(people, id_lower, prefix="ppl") if (not people.is_empty() and id_lower in people.columns) else pl.DataFrame()

            out = crashes
            if not veh_agg.is_empty():
                out = out.join(veh_agg, on=id_lower, how="left")
                stage_operations_total.labels(stage='join_vehicles', status='success').inc()
                
            if not ppl_agg.is_empty():
                out = out.join(ppl_agg, on=id_lower, how="left")
                stage_operations_total.labels(stage='join_people', status='success').inc()

            out = out.unique(subset=[id_lower], keep="first", maintain_order=True)
            
            operations_total.labels(operation='merge', status='success').inc()
            stage_operations_total.labels(stage='merge_all', status='success').inc()
            
            return out
            
        except Exception as e:
            operations_total.labels(operation='merge', status='failure').inc()
            stage_operations_total.labels(stage='merge_all', status='failure').inc()
            errors_total.labels(error_type='merging').inc()
            raise

# ---------------------------------
# CSV safety
# ---------------------------------
def make_csv_safe(df: pl.DataFrame) -> pl.DataFrame:
    """Make DataFrame CSV-safe with metrics tracking."""
    with function_duration_seconds.labels(function='make_csv_safe').time():
        try:
            if df.is_empty():
                return df

            def _jsonable(x):
                if x is None or isinstance(x, (str, int, float, bool)):
                    return x
                if isinstance(x, bytes):
                    try:
                        return x.decode("utf-8")
                    except Exception:
                        return x.hex()
                if isinstance(x, (list, tuple, set)):
                    return [_jsonable(v) for v in list(x)]
                if isinstance(x, dict):
                    return {k: _jsonable(v) for k, v in x.items()}
                if hasattr(x, "to_list"):
                    try:
                        return [_jsonable(v) for v in x.to_list()]
                    except Exception:
                        pass
                if hasattr(x, "to_dict"):
                    try:
                        return {k: _jsonable(v) for k, v in x.to_dict().items()}
                    except Exception:
                        pass
                return str(x)

            fixes, drop_cols = [], []
            for name, dtype in df.schema.items():
                if isinstance(dtype, (pl.List, pl.Struct)) or dtype.__class__.__name__ == "Array":
                    fixes.append(
                        pl.col(name).map_elements(
                            lambda x: json.dumps(_jsonable(x), ensure_ascii=False),
                            return_dtype=pl.String
                        ).alias(f"{name}_json")
                    )
                    drop_cols.append(name)

            if not fixes:
                return df
            
            out = df.with_columns(fixes)
            result = out.drop(drop_cols) if drop_cols else out
            
            stage_operations_total.labels(stage='make_csv_safe', status='success').inc()
            return result
            
        except Exception as e:
            stage_operations_total.labels(stage='make_csv_safe', status='failure').inc()
            errors_total.labels(error_type='parsing').inc()
            raise

# ---------------------------------
# Rabbit publish (minimal clean message)
# ---------------------------------
def publish_clean_job(amqp_url: str, corr: str) -> None:
    """Publish clean job with metrics tracking."""
    with function_duration_seconds.labels(function='publish_clean').time():
        params = pika.URLParameters(amqp_url)
        conn = None
        try:
            conn = pika.BlockingConnection(params)
            ch = conn.channel()
            ch.queue_declare(queue=CLEAN_QUEUE, durable=True)
            ch.basic_publish(
                exchange="",
                routing_key=CLEAN_QUEUE,
                body=json.dumps({"type": "clean", "corr_id": corr}).encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,
                    content_type="application/json",
                    correlation_id=corr,
                    type="clean",
                ),
            )
            rabbitmq_operations_total.labels(operation='publish', status='success').inc()
            operations_total.labels(operation='publish', status='success').inc()
            stage_operations_total.labels(stage='publish_clean', status='success').inc()
            
        except Exception as e:
            rabbitmq_operations_total.labels(operation='publish', status='failure').inc()
            operations_total.labels(operation='publish', status='failure').inc()
            stage_operations_total.labels(stage='publish_clean', status='failure').inc()
            errors_total.labels(error_type='rabbitmq').inc()
            raise
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

def setup_logging(corrid=None):
    """Setup logging to both console and file"""
    handlers = [logging.StreamHandler()]
    if corrid:
        log_dir = Path("minio-data") / "raw-data" / "_runs" / f"corr={corrid}"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "transformer.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('[transformer] %(message)s'))
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        format="[transformer] %(message)s",
        handlers=handlers,
        force=True
    )

# ---------------------------------
# Transform runner (writes CSV, then publishes clean job)
# ---------------------------------
def run_transform_job(msg: dict):
    """Run transformation job with comprehensive metrics tracking."""
    job_start = time.time()
    current_jobs_gauge.inc()
    
    corr = msg.get("corr_id")
    
    try:
        setup_logging(corr)
        raw_bucket = msg.get("raw_bucket", RAW_BUCKET)
        out_bucket = msg.get("xform_bucket") or msg.get("clean_bucket") or XFORM_BUCKET_ENV
        prefix = PREFIX
        
        if not corr or not out_bucket:
            raise ValueError("run_transform_job: missing corr_id or (xform_bucket|clean_bucket|XFORM_BUCKET)")

        cli = minio_client()

        # Ensure target bucket exists
        with function_duration_seconds.labels(function='ensure_bucket').time():
            try:
                if not cli.bucket_exists(out_bucket):
                    cli.make_bucket(out_bucket)
                minio_operations_total.labels(operation='bucket_check', status='success').inc()
            except S3Error as e:
                if e.code not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
                    minio_operations_total.labels(operation='bucket_check', status='failure').inc()
                    raise

        # Load raw pages (partitioned by year; filter by corr)
        crashes_df = load_dataset(cli, raw_bucket, prefix, "crashes", corr)
        vehicles_df = load_dataset(cli, raw_bucket, prefix, "vehicles", corr)
        people_df = load_dataset(cli, raw_bucket, prefix, "people", corr)

        merged = merge_crash_vehicles_people(
            crashes=crashes_df,
            vehicles=vehicles_df,
            people=people_df,
            id_col="crash_record_id",
        )

        # Track output metrics
        rows_output_total.labels(corr_id=corr).inc(merged.height)
        rows_processed_gauge.labels(dataset='merged').set(merged.height)
        columns_processed_gauge.set(merged.width)

        out_key = f"{prefix}/corr={corr}/merged.csv"
        write_csv(cli, out_bucket, out_key, make_csv_safe(merged))
        logging.info(f"Wrote s3://{out_bucket}/{out_key} (rows={merged.height}, cols={merged.width})")

        # publish tiny clean job
        publish_clean_job(RABBIT_URL, corr)
        logging.info(f"Published clean job to '{CLEAN_QUEUE}' corr={corr}")

        # Record successful job
        job_duration = time.time() - job_start
        job_duration_seconds.observe(job_duration)
        jobs_processed_total.labels(status='success').inc()
        logging.info(f"Transform job completed in {job_duration:.2f}s")

    except ValueError as e:
        logging.error(f"Invalid job configuration: {e}")
        jobs_processed_total.labels(status='invalid').inc()
        raise
        
    except Exception as e:
        logging.error(f"Transform job failed: {e}")
        jobs_processed_total.labels(status='failure').inc()
        raise
        
    finally:
        current_jobs_gauge.dec()

# ---------------------------------
# RabbitMQ consumer
# ---------------------------------
def wait_for_port(host: str, port: int, tries: int = 60, delay: float = 1.0):
    for _ in range(tries):
        try:
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except OSError:
            time.sleep(delay)
    return False

def start_consumer():
    from pika.exceptions import AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError

    # Start Prometheus metrics server
    start_http_server(METRICS_PORT)
    logging.info(f"📊 Metrics server started on port {METRICS_PORT}")

    params = pika.URLParameters(RABBIT_URL)

    host = params.host or "rabbitmq"
    port = params.port or 5672
    if not wait_for_port(host, port, tries=60, delay=1.0):
        raise SystemExit(f"[transformer] RabbitMQ not reachable at {host}:{port} after waiting.")

    max_tries = 60
    base_delay = 1.5
    conn = None

    for i in range(1, max_tries + 1):
        try:
            conn = pika.BlockingConnection(params)
            break
        except (AMQPConnectionError, ProbableAccessDeniedError, ProbableAuthenticationError) as e:
            if i == 1:
                logging.info(f"Waiting for RabbitMQ @ {RABBIT_URL} …")
            if i % 10 == 0:
                logging.info(f"Still waiting (attempt {i}/{max_tries}): {e.__class__.__name__}")
            time.sleep(base_delay + random.random())

    if conn is None or not conn.is_open:
        raise SystemExit("[transformer] Could not connect to RabbitMQ after multiple attempts.")

    ch = conn.channel()
    ch.queue_declare(queue=TRANSFORM_QUEUE, durable=True)
    ch.basic_qos(prefetch_count=1)

    def on_msg(chx, method, props, body):
        queue_messages_processed_total.inc()
        
        try:
            msg = json.loads(body.decode("utf-8"))
            mtype = msg.get("type", "")
            
            if mtype not in ("transform", "clean"):
                logging.info(f"ignoring message type={mtype!r}")
                jobs_processed_total.labels(status='invalid').inc()
                chx.basic_ack(delivery_tag=method.delivery_tag)
                return

            logging.info(f"Received transform job (type={mtype}) corr={msg.get('corr_id')}")
            rabbitmq_operations_total.labels(operation='consume', status='success').inc()
            
            run_transform_job(msg)
            chx.basic_ack(delivery_tag=method.delivery_tag)
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            errors_total.labels(error_type='parsing').inc()
            jobs_processed_total.labels(status='failure').inc()
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            
        except Exception:
            traceback.print_exc()
            jobs_processed_total.labels(status='failure').inc()
            chx.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    logging.info(f"Up. Waiting for jobs on queue '{TRANSFORM_QUEUE}'")
    ch.basic_consume(queue=TRANSFORM_QUEUE, on_message_callback=on_msg)
    
    try:
        ch.start_consuming()
    except KeyboardInterrupt:
        try: ch.stop_consuming()
        except Exception: pass
        try: conn.close()
        except Exception: pass

if __name__ == "__main__":
    start_consumer()
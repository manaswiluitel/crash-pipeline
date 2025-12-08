# -*- coding: utf-8 -*-
import os
import io
import time
import socket
import json
import ast
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server

import numpy as np
import pandas as pd
import pika

from minio import Minio
from minio.error import S3Error

# -----------------------------
# Prometheus Metrics
# -----------------------------

# 1. Service uptime (implicit via process_start_time_seconds - auto-tracked by prometheus_client)
service_info = Info('cleaner_service', 'Service information and uptime tracking')
service_info.info({'version': '1.0', 'service': 'cleaner'})

# 2. Run/Request count
jobs_processed_total = Counter(
    'cleaner_jobs_processed_total',
    'Total number of cleaning jobs processed',
    ['status']  # success, failure, invalid
)

# 3. Error count (failures)
errors_total = Counter(
    'cleaner_errors_total',
    'Total errors encountered by type',
    ['error_type']  # minio, parsing, cleaning, database, network
)

operations_total = Counter(
    'cleaner_operations_total',
    'Total operations by type and outcome',
    ['operation', 'status']  # operation=fetch/clean/write/upsert, status=success/failure
)

# 4. Latency of each run
job_duration_seconds = Histogram(
    'cleaner_job_duration_seconds',
    'Duration of cleaning job processing',
    buckets=[1, 5, 10, 30, 60, 120, 300, 600]  # 1s to 10min
)

# 5. Rows processed
rows_input_total = Counter(
    'cleaner_rows_input_total',
    'Total number of input rows received',
    ['corr_id']
)

rows_output_total = Counter(
    'cleaner_rows_output_total',
    'Total number of output rows after cleaning',
    ['corr_id']
)

rows_dropped_total = Counter(
    'cleaner_rows_dropped_total',
    'Total number of rows dropped during cleaning',
    ['reason']  # duplicates, outliers, invalid_location, invalid_age
)

rows_processed_gauge = Gauge(
    'cleaner_rows_in_last_job',
    'Number of rows processed in the most recent job',
    ['stage']  # input, output
)

# 6. Duration of each major function
function_duration_seconds = Histogram(
    'cleaner_function_duration_seconds',
    'Duration of major cleaning functions',
    ['function'],  # fetch_data, clean_nulls, clean_outliers, clean_duplicates, etc.
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 60]
)

# 7. Success vs failure counters
stage_operations_total = Counter(
    'cleaner_stage_operations_total',
    'Operations by cleaning stage and outcome',
    ['stage', 'status']  # stage=fetch/boolean_clean/time_features/location_filter/etc, status=success/failure
)

# Additional useful metrics
current_jobs_gauge = Gauge(
    'cleaner_current_jobs',
    'Number of jobs currently being processed'
)

queue_messages_processed_total = Counter(
    'cleaner_queue_messages_processed_total',
    'Total number of queue messages processed'
)

data_quality_metrics = Gauge(
    'cleaner_data_quality',
    'Data quality metrics from last job',
    ['metric']  # null_percentage, duplicate_count, outlier_count
)

upsert_operations = Counter(
    'cleaner_upsert_operations_total',
    'Database upsert operations',
    ['operation']  # inserted, updated, unchanged
)

minio_operations_total = Counter(
    'cleaner_minio_operations_total',
    'MinIO operations',
    ['operation', 'status']  # operation=fetch, status=success/failure
)

# -----------------------------
# Env / Config
# -----------------------------
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
CLEAN_QUEUE  = os.getenv("CLEAN_QUEUE", "clean")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS   = os.getenv("MINIO_USER", "admin")
MINIO_SECRET   = os.getenv("MINIO_PASS", "admin123")
MINIO_SECURE   = (os.getenv("MINIO_SSL", "false").lower() == "true")

# The transformer writes to XFORM_BUCKET at crash/corr=<corr>/merged.csv
XFORM_BUCKET = os.getenv("XFORM_BUCKET", "transform-data")
PREFIX       = "crash"

GOLD_CSV_OUT = "/data/gold/cleaned_data.csv"
GOLD_DB_PATH = "/data/gold/gold.duckdb"
GOLD_TABLE   = "gold.crash_data"
GOLD_PK      = "crash_record_id"

METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))

# -----------------------------
# RabbitMQ preflight
# -----------------------------
def wait_for_rabbitmq(host: str, port: int, retries=30, delay=2):
    for _ in range(retries):
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(delay)
    return False

# -----------------------------
# MinIO client + fetch helper
# -----------------------------
def minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS,
        secret_key=MINIO_SECRET,
        secure=MINIO_SECURE,
    )

def fetch_merged_csv(cli: Minio, bucket: str, corr: str) -> pd.DataFrame:
    """Fetch merged CSV from MinIO with metrics tracking."""
    with function_duration_seconds.labels(function='fetch_data').time():
        key = f"{PREFIX}/corr={corr}/merged.csv"
        resp = None
        try:
            resp = cli.get_object(bucket, key)
            data = resp.read()  # bytes
            minio_operations_total.labels(operation='fetch', status='success').inc()
            operations_total.labels(operation='fetch', status='success').inc()
        except Exception as e:
            minio_operations_total.labels(operation='fetch', status='failure').inc()
            operations_total.labels(operation='fetch', status='failure').inc()
            errors_total.labels(error_type='minio').inc()
            raise
        finally:
            try:
                if resp is not None:
                    resp.close()
                    resp.release_conn()
            except Exception:
                pass
        
        if not data:
            raise FileNotFoundError(f"Empty object for s3://{bucket}/{key}")
        
        df = pd.read_csv(io.BytesIO(data))
        rows_input_total.labels(corr_id=corr).inc(len(df))
        rows_processed_gauge.labels(stage='input').set(len(df))
        return df

# -----------------------------
# Your cleaning pipeline (with metrics)
# -----------------------------
def clean_dataframe(data: pd.DataFrame, corr: str = "unknown") -> pd.DataFrame:
    """Clean dataframe with comprehensive metrics tracking."""
    
    initial_rows = len(data)
    
    with function_duration_seconds.labels(function='clean_dataframe_total').time():
        # --- Drop columns ---
        with function_duration_seconds.labels(function='drop_columns').time():
            drop_cols_1 = [
                'report_type', 'statements_taken_i', 'date_police_notified',
                'location_json', 'street_name'
            ]
            data = data.copy()
            data = data.drop([c for c in drop_cols_1 if c in data.columns], axis=1, errors='ignore')
            stage_operations_total.labels(stage='drop_columns', status='success').inc()

        # --- Standardize boolean columns ---
        with function_duration_seconds.labels(function='clean_booleans').time():
            try:
                bool_cols = [col for col in data.columns if col.endswith('_i')]
                mapping = {
                    'Y': 1, 'y': 1, 'yes': 1, 'true': 1, 1: 1,
                    'N': 0, 'n': 0, 'no': 0, 'false': 0, 0: 0
                }
                for col in bool_cols:
                    data[col] = data[col].map(mapping)
                    data[col] = data[col].astype(float)
                stage_operations_total.labels(stage='clean_booleans', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='clean_booleans', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Time features ---
        with function_duration_seconds.labels(function='time_features').time():
            try:
                if 'crash_date' in data.columns:
                    data['crash_date'] = pd.to_datetime(data['crash_date'])
                    data['year'] = data['crash_date'].dt.year
                    if 'crash_month' in data.columns:
                        data['month'] = data['crash_month']
                    if 'crash_hour' in data.columns:
                        data['hour'] = data['crash_hour']
                    data['day'] = data['crash_date'].dt.day
                    if 'crash_day_of_week' in data.columns and 'crash_hour' in data.columns:
                        data['is_weekend'] = ((data['crash_day_of_week'] == 1) | (data['crash_day_of_week'] == 7)).astype(bool)
                        data['hour_bin'] = pd.cut(data['crash_hour'], bins=[0, 6, 12, 18, 24],
                                                 labels=['night', 'morning', 'afternoon', 'evening'])
                stage_operations_total.labels(stage='time_features', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='time_features', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Location filters & bins ---
        with function_duration_seconds.labels(function='location_filter').time():
            try:
                if 'latitude' in data.columns and 'longitude' in data.columns:
                    before_filter = len(data)
                    data = data[~((data['latitude'] < 41.5) | (data['latitude'] > 42))].copy()
                    data = data[~((data['longitude'] < -88) | (data['longitude'] > -87.4))].copy()
                    dropped = before_filter - len(data)
                    if dropped > 0:
                        rows_dropped_total.labels(reason='invalid_location').inc(dropped)
                    
                    data['lat_bin'] = data['latitude'].round(2)
                    data['lng_bin'] = data['longitude'].round(2)
                    data['grid_id'] = data['lat_bin'].astype(str) + '_' + data['lng_bin'].astype(str)
                stage_operations_total.labels(stage='location_filter', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='location_filter', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Vehicles and People counts ---
        with function_duration_seconds.labels(function='count_features').time():
            try:
                def safe_literal_eval(x):
                    try:
                        return ast.literal_eval(x)
                    except (ValueError, SyntaxError, TypeError):
                        return []

                if 'veh_unit_type_list_json' in data.columns:
                    data['veh_count'] = data['veh_unit_type_list_json'].apply(
                        lambda x: len(safe_literal_eval(x)) if isinstance(x, str) else 0
                    )
                else:
                    data['veh_count'] = 0

                if 'ppl_age_list_json' in data.columns:
                    data['ppl_count'] = data['ppl_age_list_json'].apply(
                        lambda x: len(safe_literal_eval(x)) if isinstance(x, str) else 0
                    )
                    data['ppl_age_mean'] = data['ppl_age_list_json'].apply(
                        lambda x: (lambda lst: sum(int(a) for a in lst)/len(lst) if lst else 0)([int(a) for a in safe_literal_eval(x)])
                    )
                    data['ppl_age_min'] = data['ppl_age_list_json'].apply(
                        lambda x: (lambda lst: min(int(a) for a in lst) if lst else 0)([int(a) for a in safe_literal_eval(x)])
                    )
                    data['ppl_age_max'] = data['ppl_age_list_json'].apply(
                        lambda x: (lambda lst: max(int(a) for a in lst) if lst else 0)([int(a) for a in safe_literal_eval(x)])
                    )
                    # Remove absurd ages
                    before_age_filter = len(data)
                    data = data[~((data['ppl_age_min'] < 0) | (data['ppl_age_max'] > 110))].copy()
                    dropped = before_age_filter - len(data)
                    if dropped > 0:
                        rows_dropped_total.labels(reason='invalid_age').inc(dropped)
                
                stage_operations_total.labels(stage='count_features', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='count_features', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Contributory cause grouping ---
        with function_duration_seconds.labels(function='group_causes').time():
            try:
                def group_contributory_cause(cause):
                    cause = str(cause).upper()
                    if 'SPEED' in cause:
                        return 'Speeding'
                    elif 'UNDER THE INFLUENCE' in cause or 'IMPAIRMENT' in cause:
                        return 'DUI/Impairment'
                    elif 'DISTRACTION' in cause or 'INATTENTION' in cause:
                        return 'Distraction/Inattention'
                    elif 'FAILING TO YIELD' in cause:
                        return 'Failure-to-Yield'
                    elif 'FOLLOWING TOO CLOSELY' in cause:
                        return 'Following Too Closely'
                    elif any(w in cause for w in ['WEATHER', 'RAIN', 'SNOW', 'ICE']):
                        return 'Weather-related'
                    elif 'LIGHTING' in cause or 'VISION' in cause:
                        return 'Lighting/Visibility'
                    else:
                        return 'Other'

                if 'prim_contributory_cause' in data.columns:
                    data['prim_contributory_cause_group'] = data['prim_contributory_cause'].apply(group_contributory_cause)
                
                stage_operations_total.labels(stage='group_causes', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='group_causes', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Handle missing values ---
        with function_duration_seconds.labels(function='clean_nulls').time():
            try:
                null_count_before = data.isnull().sum().sum()
                
                boolean_cols = [c for c in ['intersection_related_i','hit_and_run_i','crash_date_est_i',
                                           'private_property_i','work_zone_i'] if c in data.columns]
                if boolean_cols:
                    data[boolean_cols] = data[boolean_cols].fillna(0)

                data = data.replace(['None', 'none', 'NONE', '', ' ', 'nan', 'NaN', 'NAN'], np.nan)
                data = data.replace([None], np.nan)

                # Clean categorical
                categorical_cols = data.select_dtypes(include='object').columns.tolist()
                exclude_cols = [col for col in categorical_cols if data[col].astype(str).str.contains(r'[\[,]', na=False).any()]
                categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
                unknown_values = ['UNK', 'UNKNOWN', 'N/A', '—', '']
                for col in categorical_cols:
                    data[col] = data[col].astype(str).str.strip().str.upper().replace(unknown_values, np.nan)

                mode_impute_cols = [c for c in ['device_condition', 'weather_condition', 'traffic_control_device'] if c in data.columns]
                for col in mode_impute_cols:
                    mode_val = data[col].mode(dropna=True)
                    if not mode_val.empty:
                        data[col] = data[col].fillna(mode_val[0])

                for col in data.columns:
                    if data[col].isnull().any():
                        if str(data[col].dtype) in ['int64', 'float64']:
                            data[col] = data[col].fillna(data[col].median())
                        else:
                            mv = data[col].mode()
                            if not mv.empty:
                                data[col] = data[col].fillna(mv[0])
                
                null_count_after = data.isnull().sum().sum()
                null_percentage = (null_count_after / (len(data) * len(data.columns))) * 100 if len(data) > 0 else 0
                data_quality_metrics.labels(metric='null_percentage').set(null_percentage)
                
                stage_operations_total.labels(stage='clean_nulls', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='clean_nulls', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Outliers / clipping ---
        with function_duration_seconds.labels(function='clean_outliers').time():
            try:
                outlier_count = 0
                
                if 'ppl_count' in data.columns:
                    outlier_count += (data['ppl_count'] > 10).sum()
                    data['ppl_count'] = data['ppl_count'].clip(upper=10)
                    
                if 'veh_count' in data.columns:
                    outlier_count += (data['veh_count'] > 5).sum()
                    data['veh_count'] = data['veh_count'].clip(upper=5)
                    
                if 'injuries_total' in data.columns:
                    outlier_count += (data['injuries_total'] > 20).sum()
                    data['injuries_total'] = data['injuries_total'].clip(upper=20)

                def clip_ages_in_list(age_list_str):
                    try:
                        age_list = ast.literal_eval(age_list_str)
                        clipped_age_list = [str(min(int(age), 100)) for age in age_list]
                        return str(clipped_age_list)
                    except (ValueError, SyntaxError, TypeError):
                        return age_list_str
                        
                if 'ppl_age_list_json' in data.columns:
                    data['ppl_age_list_json'] = data['ppl_age_list_json'].apply(clip_ages_in_list)
                    
                if 'ppl_age_max' in data.columns:
                    outlier_count += (data['ppl_age_max'] > 100).sum()
                    data['ppl_age_max'] = data['ppl_age_max'].clip(upper=100)
                
                rows_dropped_total.labels(reason='outliers').inc(outlier_count)
                data_quality_metrics.labels(metric='outlier_count').set(outlier_count)
                
                stage_operations_total.labels(stage='clean_outliers', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='clean_outliers', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Target (KSI) ---
        with function_duration_seconds.labels(function='create_target').time():
            try:
                for c in ['injuries_fatal','injuries_incapacitating','injuries_non_incapacitating','injuries_reported_not_evident']:
                    if c not in data.columns:
                        data[c] = 0
                        
                data['ksi_flag'] = (
                    (data['injuries_fatal'] > 0) |
                    (data['injuries_incapacitating'] > 0) |
                    (data['injuries_non_incapacitating'] > 0) |
                    (data['injuries_reported_not_evident'] > 0)
                ).astype(int)

                # Drop leakage + other fields if present
                leak_cols = [
                    'injuries_fatal','injuries_incapacitating','injuries_non_incapacitating',
                    'injuries_reported_not_evident','injuries_total','photos_taken_i',
                    'injuries_unknown','injuries_no_indication',
                    'most_severe_injury','ppl_injury_classification_list_json','work_zone_type','dooring_i'
                ]
                data = data.drop([c for c in leak_cols if c in data.columns], axis=1, errors='ignore')
                
                stage_operations_total.labels(stage='create_target', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='create_target', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

        # --- Grain & dedupe ---
        with function_duration_seconds.labels(function='deduplicate').time():
            try:
                if 'crash_date' in data.columns:
                    data = data.sort_values(by='crash_date', ascending=False)
                    
                before_dedup = len(data)
                subset_cols = [c for c in ['beat_of_occurrence','crash_day_of_week','crash_hour',
                                          'latitude','longitude','street_no','street_direction'] if c in data.columns]
                if subset_cols:
                    data = data.drop_duplicates(subset=subset_cols, keep='first')
                    
                duplicates_dropped = before_dedup - len(data)
                if duplicates_dropped > 0:
                    rows_dropped_total.labels(reason='duplicates').inc(duplicates_dropped)
                    data_quality_metrics.labels(metric='duplicate_count').set(duplicates_dropped)
                
                stage_operations_total.labels(stage='deduplicate', status='success').inc()
            except Exception as e:
                stage_operations_total.labels(stage='deduplicate', status='failure').inc()
                errors_total.labels(error_type='cleaning').inc()
                raise

    # Final metrics
    final_rows = len(data)
    rows_output_total.labels(corr_id=corr).inc(final_rows)
    rows_processed_gauge.labels(stage='output').set(final_rows)
    
    total_dropped = initial_rows - final_rows
    print(f"📊 Cleaning summary: {initial_rows} → {final_rows} rows ({total_dropped} dropped)")
    
    return data

# -----------------------------
# DuckDB upsert
# -----------------------------
from .duckdb_writer import ensure_schema_and_table, upsert_dataframe
import duckdb

def upsert_to_gold(df: pd.DataFrame):
    """Upsert to DuckDB with metrics tracking."""
    with function_duration_seconds.labels(function='upsert_database').time():
        con = duckdb.connect(GOLD_DB_PATH)
        try:
            ensure_schema_and_table(con, GOLD_TABLE, df, pk=GOLD_PK)
            summary = upsert_dataframe(con, GOLD_TABLE, df, key=GOLD_PK)
            
            # Track upsert operations
            if 'inserted' in summary:
                upsert_operations.labels(operation='inserted').inc(summary['inserted'])
            if 'updated' in summary:
                upsert_operations.labels(operation='updated').inc(summary['updated'])
            if 'unchanged' in summary:
                upsert_operations.labels(operation='unchanged').inc(summary['unchanged'])
            
            operations_total.labels(operation='upsert', status='success').inc()
            stage_operations_total.labels(stage='database_upsert', status='success').inc()
            
        except Exception as e:
            operations_total.labels(operation='upsert', status='failure').inc()
            stage_operations_total.labels(stage='database_upsert', status='failure').inc()
            errors_total.labels(error_type='database').inc()
            raise
        finally:
            con.close()
        return summary

# -----------------------------
# Message handler
# -----------------------------
def handle_message(ch, method, properties, body):
    """Handle RabbitMQ message with comprehensive metrics."""
    job_start = time.time()
    current_jobs_gauge.inc()
    queue_messages_processed_total.inc()
    
    try:
        msg = json.loads(body.decode("utf-8"))
        corr = msg.get("corr_id")
        mtype = msg.get("type")
        
        if mtype != "clean" or not corr:
            print(f"⚠️  Skipping message: {msg}")
            jobs_processed_total.labels(status='invalid').inc()
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        print(f"🧹 Cleaner: received clean job corr={corr}")

        # Fetch merged.csv from MinIO
        cli = minio_client()
        df_raw = fetch_merged_csv(cli, XFORM_BUCKET, corr)

        # Run cleaning
        df_clean = clean_dataframe(df_raw, corr=corr)

        # Persist CSV
        with function_duration_seconds.labels(function='write_csv').time():
            try:
                os.makedirs(os.path.dirname(GOLD_CSV_OUT), exist_ok=True)
                df_clean.to_csv(GOLD_CSV_OUT, index=False)
                print(f"✅ Cleaned data saved to {GOLD_CSV_OUT}")
                operations_total.labels(operation='write', status='success').inc()
            except Exception as e:
                operations_total.labels(operation='write', status='failure').inc()
                errors_total.labels(error_type='filesystem').inc()
                raise

        # Upsert to DuckDB
        summary = upsert_to_gold(df_clean)
        print("✅ Upsert summary:", summary)

        # Record successful job
        job_duration = time.time() - job_start
        job_duration_seconds.observe(job_duration)
        jobs_processed_total.labels(status='success').inc()
        
        # ACK on success
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"✅ Clean job done corr={corr} in {job_duration:.2f}s")

    except (S3Error, FileNotFoundError) as e:
        print(f"❌ MinIO error: {e}")
        errors_total.labels(error_type='minio').inc()
        jobs_processed_total.labels(status='failure').inc()
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        errors_total.labels(error_type='parsing').inc()
        jobs_processed_total.labels(status='failure').inc()
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
    except Exception as e:
        # Log and drop (consistent with transformer's drop-on-fail pattern)
        print(f"❌ Cleaner error: {e}")
        errors_total.labels(error_type='unknown').inc()
        jobs_processed_total.labels(status='failure').inc()
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        
    finally:
        current_jobs_gauge.dec()

# -----------------------------
# Main: connect & consume
# -----------------------------
def main():
    # Start Prometheus metrics server
    start_http_server(METRICS_PORT)
    print(f"📊 Metrics server started on port {METRICS_PORT}")
    
    # TCP reachability check
    host = "rabbitmq"
    port = 5672
    if not wait_for_rabbitmq(host, port):
        raise SystemExit(f"RabbitMQ not reachable at {host}:{port}")

    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=CLEAN_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=1)

    print(f"✅ Connected to RabbitMQ and consuming '{CLEAN_QUEUE}'")
    channel.basic_consume(queue=CLEAN_QUEUE, on_message_callback=handle_message)

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        try: channel.stop_consuming()
        except Exception: pass
        try: connection.close()
        except Exception: pass
        print("👋 Cleaner shut down.")

if __name__ == "__main__":
    main()

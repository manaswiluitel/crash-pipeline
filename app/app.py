import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime

import base64
import streamlit as st
import requests
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import boto3
from botocore.client import Config
import pika
from dotenv import load_dotenv
from flask import Flask
import subprocess
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, generate_latest
from prometheus_client.core import CollectorRegistry
from joblib import load
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, accuracy_score, precision_recall_curve)

# Load environment variables
load_dotenv()

# ---------------------------------
# Prometheus Metrics Setup
# ---------------------------------
registry = CollectorRegistry()

# 1. App uptime (implicit via process_start_time_seconds - auto-tracked)
app_info = Info('streamlit_app', 'Streamlit ML UI information and uptime tracking', registry=registry)
app_info.info({'version': '1.0', 'service': 'ml_ui'})

# 2. Model quality metrics
model_accuracy = Gauge('streamlit_model_accuracy', 'Current model accuracy score', registry=registry)
model_precision = Gauge('streamlit_model_precision', 'Current model precision at threshold', registry=registry)
model_recall = Gauge('streamlit_model_recall', 'Current model recall at threshold', registry=registry)
model_f1_score = Gauge('streamlit_model_f1_score', 'Current model F1 score', registry=registry)
model_roc_auc = Gauge('streamlit_model_roc_auc', 'Current model ROC-AUC score', registry=registry)
model_pr_auc = Gauge('streamlit_model_pr_auc', 'Current model PR-AUC score', registry=registry)

# 3. Prediction latency
prediction_duration_seconds = Histogram(
    'streamlit_prediction_duration_seconds',
    'Duration of model prediction operations',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)
# 4. Additional useful metrics

# Predictions counter
predictions_total = Counter(
    'streamlit_predictions_total',
    'Total number of predictions made',
    ['outcome'],  # positive, negative
    registry=registry
)

# Data loading metrics
data_load_duration_seconds = Histogram(
    'streamlit_data_load_duration_seconds',
    'Duration of data loading operations',
    ['source'],  # gold_db, csv_upload
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60],
    registry=registry
)

data_rows_loaded = Counter(
    'streamlit_data_rows_loaded_total',
    'Total number of data rows loaded',
    ['source'],
    registry=registry
)

# Page view tracking
page_views_total = Counter(
    'streamlit_page_views_total',
    'Total page views by page name',
    ['page'],
    registry=registry
)

# Job submissions
jobs_submitted_total = Counter(
    'streamlit_jobs_submitted_total',
    'Total ETL jobs submitted',
    ['mode', 'status'],  # streaming/backfill, success/failure
    registry=registry
)

# Health check status
service_health_status = Gauge(
    'streamlit_service_health_status',
    'Health status of dependent services (1=healthy, 0=unhealthy)',
    ['service'],  # MinIO, RabbitMQ, Extractor, Transformer, Cleaner
    registry=registry
)
# Gold database metrics
gold_db_rows = Gauge(
    'streamlit_gold_db_total_rows',
    'Total rows in gold database',
    registry=registry
)

gold_db_latest_date = Gauge(
    'streamlit_gold_db_latest_timestamp',
    'Latest crash date in gold database (Unix timestamp)',
    registry=registry
)

# Model loading status
model_load_status = Gauge(
    'streamlit_model_load_status',
    'Model loading status (1=success, 0=failure)',
    registry=registry
)

# Current active users/sessions (approximate)
active_sessions = Gauge(
    'streamlit_active_sessions',
    'Approximate number of active sessions',
    registry=registry
)
# Error tracking
errors_total = Counter(
    'streamlit_errors_total',
    'Total errors encountered',
    ['error_type'],  # model_load, prediction, data_load, rabbitmq, minio
    registry=registry
)

# Confusion matrix metrics (live)
confusion_matrix_values = Gauge(
    'streamlit_confusion_matrix',
    'Confusion matrix values from live predictions',
    ['actual', 'predicted'],
    registry=registry
)

# ---------------------------------
# Start Prometheus metrics server
# ---------------------------------
def start_prometheus_server():
    """Start Prometheus metrics HTTP server"""
    metrics_port = int(os.getenv("STREAMLIT_METRICS_PORT", "8000"))
    start_http_server(metrics_port, registry=registry)
    print(f"📊 Streamlit metrics server started on port {metrics_port}")

# Start metrics server in background
metrics_thread = threading.Thread(target=start_prometheus_server, daemon=True)
metrics_thread.start()

# Initialize static metrics from notebook results
STATIC_METRICS = {
    "primary_metric": "average_precision",
    "primary_score": 0.738,
    "precision@t": 0.733,
    "recall@t": 0.534,
    "f1@t": 0.618,
    "roc_auc": 0.818,
    "pr_auc": 0.738,
    "threshold": 0.53
}

# Set initial model quality metrics from static benchmark
model_precision.set(STATIC_METRICS["precision@t"])
model_recall.set(STATIC_METRICS["recall@t"])
model_f1_score.set(STATIC_METRICS["f1@t"])
model_roc_auc.set(STATIC_METRICS["roc_auc"])
model_pr_auc.set(STATIC_METRICS["pr_auc"])

# Start health endpoint server
app_health = Flask(__name__)

@app_health.route("/health")
def health():
    return {"status": "running", "service": "app"}, 200

@app_health.route("/metrics")
def metrics():
    """Expose Prometheus metrics via Flask endpoint"""
    return generate_latest(registry), 200, {'Content-Type': 'text/plain; charset=utf-8'}

def start_app_health_server():
    """Start the health endpoint server in a background thread"""
    health_port = int(os.getenv("APP_HEALTH_PORT", "8501"))
    def run_server():
        app_health.run(host="0.0.0.0", port=health_port, debug=False)
    
    health_thread = threading.Thread(target=run_server, daemon=True)
    health_thread.start()
    print(f"App health endpoint running on port {health_port}")

# Start health server
start_app_health_server()

# Track active session (simplified)
active_sessions.set(1)

st.set_page_config(page_title="ML Pipeline Dashboard", layout="wide")


LABELS = [
	{
		"emoji": "🚗",
		"name": "Crash Type",
		"label_col": "crash_type",
		"type": "binary",
		"positive": "INJURY AND / OR TOW DUE TO CRASH",
		"pipeline": "We built a model to predict crash_type towing using context like road surface conditions, road defects, damage, and lighting condition",
		"features": [
			("road_surface_cond", "Wet/Icy surfaces lead to more crashes"),
			("lighting_condition", "Poor lighting increases crash risk"),
			("posted_speed_limit", "Higher speeds correlate with severe crashes"),
			("road_defects", "Road defects contribute to crash severity"),
			("damage", "Extent of vehicle damage cost indicates crash severity"),
		],
		"sources": {"crashes": ["road_surface_cond","lighting_condition","posted_speed_limit","road_defects"]},
		"class_imbalance": {"positives": "24%", "negatives": "76%", "ratio": "~6:25", "handling": "none"},
		"grain": "crash",
		"window": "rolling 180 days",
		"filters": "city-limits only",
		"leakage": "removed injuries_fatal, injuries_incapacitating, injuries_non_incapacitating, injuries_reported_not_evident, injuries_total, most_severe_injury, and ppl_injury_classification_list_json",
		"gold_table": "gold.crash_data",
	},
]


def get_api_base():
	# allow override via env var
	return st.sidebar.text_input("Backend API base URL", value=os.environ.get("API_BASE", "http://localhost:8000"))


def render_label_card(label):
	with st.container():
		cols = st.columns([0.1, 0.9])
		with cols[0]:
			st.markdown(f"<div style='font-size:36px'>{label['emoji']}</div>", unsafe_allow_html=True)
		with cols[1]:
			st.subheader(f"{label['name']}")
			st.markdown(f"**Label predicted:** {label['label_col']} • **Type:** {label['type']} • **Positive:** {label['positive']}")
			st.markdown(f"**Pipeline:** {label['pipeline']}")
			st.markdown("**Key features (why they help):**")
			for feat, why in label["features"]:
				st.markdown(f"- **{feat}** — {why}")
			st.markdown("**Source columns (subset):**")
			for src, cols in label["sources"].items():
				st.markdown(f"- {src}: {', '.join(cols)}")
			st.markdown("**Class imbalance:**")
			ci = label["class_imbalance"]
			st.markdown(f"Positives: {ci['positives']} | Negatives: {ci['negatives']} | Ratio: {ci['ratio']}\n\nHandling: {ci['handling']}")
			st.markdown(f"**Data grain & filters:**\nOne row = {label['grain']}  •  Window: {label['window']}  •  Filters: {label['filters']}")
			st.markdown(f"**Leakage/caveats:** {label['leakage']}")
			st.markdown(f"**Gold table:** {label['gold_table']}")

def fetch_health(api_base, timeout=2):
    """Check health of MinIO and RabbitMQ services using boto3 with metrics tracking"""
    health = {}
    
    # Determine if we're running in Docker or locally
    # Use service names in Docker, localhost when running locally
    is_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
    minio_host = 'minio' if is_docker else 'localhost'
    rabbitmq_host = 'rabbitmq' if is_docker else 'localhost'
    extractor_host = 'extractor' if is_docker else 'localhost'
    transformer_host = 'transformer' if is_docker else 'localhost'
    cleaner_host = 'cleaner' if is_docker else 'localhost'
    
    # MinIO health check using boto3
    try:
        # MinIO connection details from .env
        # Use default port 9000 in Docker, or from env if set
        minio_port = '9000' if is_docker else os.getenv('MINIO_API_PORT', '9000')
        endpoint_url = f"http://{minio_host}:{minio_port}"
        access_key = os.getenv('MINIO_USER', 'admin')
        secret_key = os.getenv('MINIO_PASS', 'admin123')
        
        # Initialize MinIO client
        s3_client = boto3.client('s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'  # Default region
        )
        
        # Test MinIO by listing buckets
        s3_client.list_buckets()
        health["MinIO"] = {"status": "running"}
        service_health_status.labels(service='MinIO').set(1)
    except Exception as e:
        health["MinIO"] = {"status": "error", "details": str(e)}
        service_health_status.labels(service='MinIO').set(0)
        errors_total.labels(error_type='minio').inc()

    # RabbitMQ health check
    try:
        # RabbitMQ connection details from .env
        # Use default port 5672 in Docker, or from env if set
        rabbitmq_port = 5672 if is_docker else int(os.getenv('RABBIT_PORT', '5672'))
        credentials = pika.PlainCredentials(
            os.getenv('RABBIT_USER', 'guest'),
            os.getenv('RABBIT_PASS', 'guest')
        )
        parameters = pika.ConnectionParameters(
            host=rabbitmq_host,
            port=rabbitmq_port,
            credentials=credentials,
            connection_attempts=1,
            socket_timeout=timeout
        )
        connection = pika.BlockingConnection(parameters)
        connection.close()
        health["RabbitMQ"] = {"status": "running"}
        service_health_status.labels(service='RabbitMQ').set(1)
    except Exception as e:
        health["RabbitMQ"] = {"status": "error", "details": str(e)}
        service_health_status.labels(service='RabbitMQ').set(0)
        errors_total.labels(error_type='rabbitmq').inc()
    
    # Extractor health check
    try:
        extractor_port = int(os.getenv('EXTRACTOR_HEALTH_PORT', '8001'))
        resp = requests.get(f"http://{extractor_host}:{extractor_port}/health", timeout=timeout)
        resp.raise_for_status()
        health["Extractor"] = resp.json()
        service_health_status.labels(service='Extractor').set(1)
    except Exception as e:
        health["Extractor"] = {"status": "error", "details": str(e)}
        service_health_status.labels(service='Extractor').set(0)
    
    # Transformer health check
    try:
        transformer_port = int(os.getenv('TRANSFORMER_HEALTH_PORT', '8002'))
        resp = requests.get(f"http://{transformer_host}:{transformer_port}/health", timeout=timeout)
        resp.raise_for_status()
        health["Transformer"] = resp.json()
        service_health_status.labels(service='Transformer').set(1)
    except Exception as e:
        health["Transformer"] = {"status": "error", "details": str(e)}
        service_health_status.labels(service='Transformer').set(0)
    
    # Cleaner health check
    try:
        cleaner_port = int(os.getenv('CLEANER_HEALTH_PORT', '8003'))
        resp = requests.get(f"http://{cleaner_host}:{cleaner_port}/health", timeout=timeout)
        resp.raise_for_status()
        health["Cleaner"] = resp.json()
        service_health_status.labels(service='Cleaner').set(1)
    except Exception as e:
        health["Cleaner"] = {"status": "error", "details": str(e)}
        service_health_status.labels(service='Cleaner').set(0)
    
    return health

def render_health_section(api_base):
    st.markdown("### Container Health")
    health = fetch_health(api_base)

    services = ["MinIO", "RabbitMQ", "Extractor", "Transformer", "Cleaner"]
    cols = st.columns(len(services))

    for c, s in zip(cols, services):
        with c:
            # Get service status
            status = health.get(s, {})
            is_running = isinstance(status, dict) and status.get("status") == "running"
            
            # Determine display properties
            if is_running:
                color = "#d4f7d4"  # Light green
                emoji = "🟢"
                text = "Running"
            else:
                color = "#ffcccc"  # Light red
                emoji = "🔴"
                error_msg = status.get("details", "Not responding") if isinstance(status, dict) else "Not responding"
                text = f"Error: {error_msg}" if "error" in str(status.get("status", "")).lower() else "Not responding"

            # Render status card
            st.markdown(f"""
                <div style='background:{color};padding:12px;border-radius:8px;text-align:center'>
                    <div style='font-size:22px'>{emoji} {s}</div>
                    <div style='font-size:14px;margin-top:6px'><b>{text}</b></div>
            """, unsafe_allow_html=True)
            
            # Show additional metrics if available
            if is_running:
                metrics = [
                    ("Queue Size", status.get("queue_size", "N/A")),
                    ("Processed Items", status.get("processed_items", "N/A")),
                    ("Last Processed", status.get("last_processed", "N/A"))
                ]
                for label, value in metrics:
                    if value != "N/A":
                        st.markdown(f"""
                            <div style='font-size:12px;margin-top:4px'>{label}: {value}</div>
                        """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

    if isinstance(health, dict) and "error" in health:
        st.error(f"Health API error: {health['error']}")


def render_home():
    page_views_total.labels(page='home').inc()
    st.title("ML Labels & Pipeline Dashboard")
    st.markdown("A simple landing page that orients users to the available ML labels/pipelines and shows container health at a glance.")

    st.markdown("---")
    st.markdown("### Label Overview")
    for label in LABELS:
        render_label_card(label)
        st.markdown("---")

    api_base = get_api_base()
    render_health_section(api_base)


# MinIO helpers using boto3
def get_minio_client():
    """Get MinIO client"""
    is_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
    minio_host = 'minio' if is_docker else 'localhost'
    minio_port = '9000' if is_docker else os.getenv('MINIO_API_PORT', '9000')
    endpoint_url = f"http://{minio_host}:{minio_port}"
    access_key = os.getenv('MINIO_USER', 'admin')
    secret_key = os.getenv('MINIO_PASS', 'admin123')
    
    return boto3.client('s3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def list_buckets(api_base):
    """Get list of MinIO buckets"""
    try:
        s3_client = get_minio_client()
        response = s3_client.list_buckets()
        return [bucket['Name'] for bucket in response.get('Buckets', [])]
    except Exception as e:
        st.error(f"Failed to list buckets: {e}")
        return ["raw-data", "transform-data", "cleaned-data"]  # fallback

def list_objects_minio(bucket, prefix=""):
    """List objects in MinIO bucket with prefix"""
    try:
        # Normalize prefix to avoid invalid characters
        prefix = normalize_prefix(prefix) if prefix else ""
        s3_client = get_minio_client()
        objects = []
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                objects.extend([obj['Key'] for obj in page['Contents']])
        
        return objects
    except Exception as e:
        st.error(f"Failed to list objects: {e}")
        return []

def delete_objects_minio(bucket, prefix="", delete_bucket=False):
    """Delete objects in MinIO bucket (optionally delete bucket)"""
    try:
        s3_client = get_minio_client()
        
        # List all objects with the prefix
        objects_to_delete = list_objects_minio(bucket, prefix)
        
        if not objects_to_delete:
            return {"success": True, "message": "No objects found to delete", "deleted": 0}
        
        # Delete objects
        for obj_key in objects_to_delete:
            s3_client.delete_object(Bucket=bucket, Key=obj_key)
        
        # Optionally delete bucket if empty
        if delete_bucket:
            try:
                s3_client.delete_bucket(Bucket=bucket)
                return {"success": True, "message": f"Deleted bucket '{bucket}'", "deleted": len(objects_to_delete)}
            except Exception as e:
                return {"success": True, "message": f"Deleted {len(objects_to_delete)} objects, but couldn't delete bucket: {e}", "deleted": len(objects_to_delete)}
        
        return {"success": True, "message": f"Deleted {len(objects_to_delete)} objects with prefix '{prefix}'", "deleted": len(objects_to_delete)}
    except Exception as e:
        st.error(f"Failed to delete objects: {e}")
        return {"error": str(e)}

# Legacy API helpers for compatibility
def list_objects(api_base, bucket, prefix=""):
    """List objects in MinIO bucket with prefix"""
    return list_objects_minio(bucket, prefix)

def delete_objects(api_base, bucket, prefix="", delete_bucket=False):
    """Delete objects in MinIO bucket (optionally delete bucket)"""
    return delete_objects_minio(bucket, prefix, delete_bucket)


# MinIO helpers for listing corr IDs
def list_corrs_minio(bucket, prefix=""):
    """Return corr directories from MinIO (names like corr=...)
    If none found, returns an empty list."""
    try:
        # Normalize prefix to avoid invalid characters
        prefix = normalize_prefix(prefix) if prefix else ""
        objects = list_objects_minio(bucket, prefix)
        corrs = set()
        for obj_key in objects:
            # Extract corr=... from path
            parts = obj_key.split('/')
            for part in parts:
                if part.startswith('corr='):
                    corrs.add(part)
        return sorted(list(corrs))
    except Exception:
        return []

def list_years_minio(bucket, prefix=""):
    """Return year directories from MinIO (names like year=...)"""
    try:
        # Normalize prefix to avoid invalid characters
        prefix = normalize_prefix(prefix) if prefix else ""
        objects = list_objects_minio(bucket, prefix)
        years = set()
        for obj_key in objects:
            # Extract year=... from path
            parts = obj_key.split('/')
            for part in parts:
                if part.startswith('year='):
                    years.add(part.replace('year=', ''))
        return sorted(list(years))
    except Exception:
        return []

def normalize_prefix(prefix):
    """Normalize prefix by removing leading/trailing slashes and preventing double slashes"""
    if not prefix:
        return ""
    # Remove leading and trailing slashes
    prefix = prefix.strip('/')
    # Replace multiple consecutive slashes with single slash
    while '//' in prefix:
        prefix = prefix.replace('//', '/')
    return prefix

def preview_corr_contents_minio(bucket, prefix):
    """List immediate files under a prefix in MinIO"""
    try:
        # Normalize prefix to avoid invalid characters
        prefix = normalize_prefix(prefix)
        if not prefix:
            # If prefix is empty, list from bucket root
            objects = list_objects_minio(bucket, "")
        else:
            # Ensure prefix ends with / for proper filtering
            if not prefix.endswith('/'):
                prefix = prefix + '/'
            objects = list_objects_minio(bucket, prefix)
        # Get just the immediate children (one level deep)
        items = set()
        for obj_key in objects:
            # Remove the prefix and get first part
            relative_key = obj_key[len(prefix):] if obj_key.startswith(prefix) else obj_key
            parts = relative_key.strip('/').split('/')
            if parts and parts[0]:
                items.add(parts[0])
        return sorted(list(items))
    except Exception as e:
        st.error(f"Failed to preview contents: {e}")
        return []

# Local helpers that read from the bundled minio-data folder for previewing (fallback)
def list_corrs_local(bucket, prefix=""):
    """Return corr directories under minio-data/<bucket>/<prefix> (names like corr=...)
    If none found, returns an empty list."""
    try:
        # Try MinIO first
        return list_corrs_minio(bucket, prefix)
    except Exception:
        # Fallback to local
        base = Path("minio-data") / bucket
        target = base / Path(prefix)
        try:
            if not target.exists() or not target.is_dir():
                return []
            corrs = [p.name for p in target.iterdir() if p.is_dir() and p.name.startswith("corr=")]
            return sorted(corrs)
        except Exception:
            return []


def preview_corr_contents_local(bucket, prefix, corr):
    """List immediate children under minio-data/<bucket>/<prefix>/<corr>"""
    try:
        # Try MinIO first
        # Normalize prefix and corr to avoid double slashes
        prefix = normalize_prefix(prefix) if prefix else ""
        corr = normalize_prefix(corr) if corr else ""
        
        # Build full prefix properly
        if prefix and corr:
            full_prefix = f"{prefix}/{corr}"
        elif prefix:
            full_prefix = prefix
        elif corr:
            full_prefix = corr
        else:
            full_prefix = ""
        
        return preview_corr_contents_minio(bucket, full_prefix)
    except Exception as e:
        # Fallback to local
        try:
            p = Path("minio-data") / bucket
            if prefix:
                p = p / Path(prefix)
            if corr:
                p = p / corr
            if not p.exists() or not p.is_dir():
                return []
            items = [child.name for child in p.iterdir()]
            return sorted(items)
        except Exception:
            return []

# DuckDB helpers
def get_gold_db_info():
    """Get info about gold.duckdb tables and rows"""
    db_path = "data/gold/gold.duckdb"
    try:
        if not Path(db_path).exists():
            return {
                "exists": False,
                "path": db_path,
                "tables": [],
                "total_rows": 0
            }
            
        con = duckdb.connect(db_path)
        # We know there's only one table: gold.gold.crash_data
        table_name = '"gold"."gold"."crash_data"'
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
        return {
            "exists": True,
            "path": db_path,
            "tables": [{"name": table_name, "rows": count}],
            "total_rows": count
        }
    except Exception as e:
        st.error(f"Failed to get DB info: {e}")
        return {
            "exists": False,
            "path": db_path,
            "error": str(e)
        }

def wipe_gold_db():
    """Delete and recreate gold.duckdb"""
    db_path = "data/gold/gold.duckdb"
    try:
        if Path(db_path).exists():
            Path(db_path).unlink()
        
        # Create empty DB
        con = duckdb.connect(db_path)
        con.close()
        return {"success": True}
    except Exception as e:
        return {"error": str(e)}

def preview_gold_table(table, columns=None, limit=50):
    """Preview rows from a gold table with optional column selection"""
    try:
        con = duckdb.connect("data/gold/gold.duckdb")
        col_str = "*" if not columns else ", ".join(columns)
        df = con.execute(f"SELECT {col_str} FROM {table} LIMIT {limit}").df()
        return df
    except Exception as e:
        st.error(f"Failed to preview table: {e}")
        return None

def get_gold_table_schema(table):
    """Get column names and types for a gold table"""
    try:
        con = duckdb.connect("data/gold/gold.duckdb")
        schema = con.execute(f"DESCRIBE {table}").df()
        return schema
    except Exception as e:
        st.error(f"Failed to get schema: {e}")
        return None

def render_data_management():
    page_views_total.labels(page='data_management').inc()
    st.title("Data Management")
    st.markdown("Centralized admin for storage and warehouse housekeeping.")

    # Get API base from sidebar
    api_base = get_api_base()

    # Create tabs for different sections
    minio_tab, gold_tab, peek_tab = st.tabs([
        "MinIO Browser & Delete",
        "Gold Admin (DuckDB)",
        "Quick Peek (Gold)"
    ])

    # A) MinIO Browser & Delete
    with minio_tab:
        st.markdown("### MinIO Browser & Delete")
        st.markdown("Manage objects in object storage. Remove either a specific folder (prefix) or an entire bucket.")

        # Get available buckets
        buckets = list_buckets(api_base)

        # 1) Delete by Folder section
        st.subheader("1) Delete by Folder (Prefix)")
        
        col1, col2 = st.columns(2)
        with col1:
            bucket = st.selectbox("Bucket", buckets, key="prefix_bucket")
            
            # Show appropriate prefix options based on bucket
            prefix_options = []
            if bucket == "raw-data":
                prefix_options = [
                    "crash/crashes/",
                    "crash/people/",
                    "crash/vehicles/",
                    "_runs/",
                    "_markers/crash/crashes/"
                ]
            elif bucket == "transform-data":
                prefix_options = ["crash/"]
            elif bucket == "cleaned-data":
                prefix_options = []  # Empty for cleaned-data bucket
                
            # Allow custom prefix or selection from common ones
            use_custom = st.checkbox("Use custom prefix", key="prefix_select")
            if use_custom:
                prefix = st.text_input("Custom Prefix", help="e.g., crash/corr=YYYY-MM-DD-HH-MM-SS/")
            else:
                if prefix_options:
                    prefix = st.selectbox("Common Prefixes", prefix_options, key=f"prefix_select_{bucket}")
                else:
                    st.info("No common prefixes for this bucket")
                    prefix = ""
            
            confirm_prefix = st.checkbox("I confirm folder deletion", key="confirm_prefix")

        selected_year = None
        selected_corr = None
        # Normalize prefix to avoid trailing slashes
        prefix = normalize_prefix(prefix) if prefix else ""
        corr_base_path = prefix
        raw_year_prefixes = ["crash/crashes", "crash/people", "crash/vehicles"]
        if bucket == "raw-data" and any(prefix.startswith(p) for p in raw_year_prefixes) and prefix:
            # Try to get years from MinIO
            try:
                years = list_years_minio(bucket, prefix)
                if years:
                    selected_year = st.selectbox("Year", sorted(years), key=f"year_select_{bucket}_{prefix}")
                    # Build corr_base_path without double slashes
                    corr_base_path = f"{prefix}/year={selected_year}" if prefix else f"year={selected_year}"
                    corr_base_path = normalize_prefix(corr_base_path)
            except Exception:
                # Fallback to local
                prefix_path = Path("minio-data") / bucket / Path(prefix)
                if prefix_path.exists():
                    years = [p.name.replace("year=","") for p in prefix_path.iterdir() if p.is_dir() and p.name.startswith("year=")]
                    if years:
                        selected_year = st.selectbox("Year", sorted(years), key=f"year_select_{bucket}_{prefix}")
                        corr_base_path = f"{prefix}/year={selected_year}" if prefix else f"year={selected_year}"
                        corr_base_path = normalize_prefix(corr_base_path)
            # Show corr dropdown after year if year selected
            if selected_year:
                corr_list = list_corrs_local(bucket, corr_base_path)
                if corr_list:
                    selected_corr = st.selectbox("Corr ID (select run)", corr_list, key=f"corr_select_{bucket}_{prefix}_{selected_year}")
        elif bucket == "transform-data" and prefix:
            corr_list = list_corrs_local(bucket, corr_base_path)
            if corr_list:
                selected_corr = st.selectbox("Corr ID (select run)", corr_list, key=f"corr_select_{bucket}_{prefix}")

        # Preview button and results
        if st.button("Preview", key="preview_folder_btn", disabled=not prefix):
            # For raw-data with year and corr selected
            if bucket == "raw-data" and selected_year and selected_corr:
                items = preview_corr_contents_local(bucket, corr_base_path, selected_corr)
                if items:
                    st.markdown(f"**Contents under {selected_corr}:**")
                    for it in items:
                        st.text(f"• {it}")
                else:
                    st.info("No items found under this corr")
            # For raw-data with year selected but no corr (preview year folder)
            elif bucket == "raw-data" and selected_year and not selected_corr:
                items = preview_corr_contents_minio(bucket, corr_base_path)
                if items:
                    st.markdown(f"**Contents under year={selected_year}:**")
                    for it in items:
                        st.text(f"• {it}")
                else:
                    st.info("No items found under this year")
            # For transform-data with corr selected
            elif bucket == "transform-data" and selected_corr:
                items = preview_corr_contents_local(bucket, corr_base_path, selected_corr)
                if items:
                    st.markdown(f"**Contents under {selected_corr}:**")
                    for it in items:
                        st.text(f"• {it}")
                else:
                    st.info("No items found under this corr")
            # Otherwise, show folder contents for prefix
            else:
                items = preview_corr_contents_minio(bucket, prefix)
                if items:
                    st.markdown(f"**Contents under {prefix or 'bucket root'}:**")
                    for it in items:
                        st.text(f"• {it}")
                else:
                    st.info("No items found with this prefix")

        # Delete button
        delete_prefix = None
        if selected_corr:
            delete_prefix = f"{corr_base_path}/{selected_corr}" if corr_base_path else selected_corr
            delete_prefix = normalize_prefix(delete_prefix)
        elif selected_year:
            delete_prefix = corr_base_path
        else:
            delete_prefix = prefix
        
        if st.button("Delete Folder/Corr", key="delete_folder_btn", disabled=not (selected_corr and confirm_prefix)):
            if delete_prefix:
                result = delete_objects_minio(bucket, delete_prefix)
                if "error" in result:
                    st.error(f"Failed to delete: {result['error']}")
                else:
                    st.success(result.get('message', 'Deleted successfully'))
                    st.rerun()
            else:
                st.error("Please select a Corr ID to delete")

        # 2) Delete by Bucket section
        st.markdown("---")
        st.subheader("2) Delete by Bucket")
        
        col1, col2 = st.columns(2)
        with col1:
            bucket_to_delete = st.selectbox("Bucket", buckets, key="delete_bucket")
            confirm_bucket = st.checkbox("I confirm bucket deletion", key="confirm_bucket")

        # Delete bucket button
        if st.button("Delete Bucket",
                key="delete_bucket_btn",
                disabled=not confirm_bucket):
            result = delete_objects_minio(bucket_to_delete, "", delete_bucket=True)
            if "error" in result:
                st.error(f"Failed to delete bucket: {result['error']}")
            else:
                st.success(result.get('message', 'Bucket deleted successfully'))
                st.rerun()

    # B) Gold Admin
    with gold_tab:
        st.markdown("### Gold Admin (DuckDB)")
        st.markdown("Reset or inspect the analytical warehouse stored in gold.duckdb.")

        # Get current DB info
        db_info = get_gold_db_info()

        # Status card
        st.markdown("#### Status")
        if db_info["exists"]:
            st.success(f"Database: {db_info['path']}")
            st.markdown(f"Total rows: {db_info['total_rows']:,}")
            for table in db_info["tables"]:
                st.markdown(f"• {table['name']}: {table['rows']:,} rows")
        else:
            if "error" in db_info:
                st.error(f"Error: {db_info['error']}")
            else:
                st.warning(f"Database does not exist: {db_info['path']}")

        # Wipe controls
        st.markdown("---")
        confirm_wipe = st.checkbox("I confirm database wipe", key="confirm_wipe")
        if st.button("Wipe Gold DB", key="wipe_db_btn", disabled=not confirm_wipe):
            result = wipe_gold_db()
            if "error" in result:
                st.error(f"Failed to wipe DB: {result['error']}")
            else:
                st.success("Database wiped successfully")
                st.rerun()

    # C) Quick Peek
    with peek_tab:
        st.markdown("### Quick Peek (Gold)")
        st.markdown("Confirm that cleaned data looks sane by viewing columns and sample rows.")

        # Get DB info
        db_info = get_gold_db_info()
        if not db_info["exists"]:
            st.warning("Gold database does not exist")
            return
            
        # We know the table name
        table = '"gold"."gold"."crash_data"'
        
        # Get schema for column selection
        schema = get_gold_table_schema(table)
        if schema is None:
            return

        # Column selection
        all_columns = schema["column_name"].tolist()
        default_columns = ["crash_date", "damage", "device_condition", "crash_type","crash_day_of_week", 
                         "lighting_condition", "posted_speed_limit", "road_defect"]
        columns = st.multiselect("Columns", all_columns, default=default_columns,
                               help="If empty, will use default columns")
        
        if not columns:  # If no selection, use defaults
            columns = default_columns

        # Row limit slider
        limit = st.slider("Rows (limit)", 10, 200, 50)

        # Preview button and results
        if st.button("Preview", key="preview_data_btn"):
            df = preview_gold_table(table, columns, limit)
            if df is not None:
                st.dataframe(df)
                
                # Show schema for selected columns
                st.markdown("#### Column Types")
                selected_schema = schema[schema["column_name"].isin(columns)]
                st.dataframe(selected_schema)


# Cache the schema data for 5 minutes to avoid repeated API calls
@st.cache_data(ttl=300)
def fetch_schema(api_base):
    """Fetch the schema for vehicles and people datasets from Chicago API"""
    vehicles = []
    people = []
    
    # Fetch from Chicago Socrata API metadata endpoints
    try:
        # Vehicles API: https://data.cityofchicago.org/api/v3/views/68nd-jvt3/query.json
        # People API: https://data.cityofchicago.org/api/v3/views/u6pd-qa9d/query.json
        # Use metadata endpoint to get columns: /api/views/{view_id}
        vehicles_url = "https://data.cityofchicago.org/api/views/68nd-jvt3.json"
        people_url = "https://data.cityofchicago.org/api/views/u6pd-qa9d.json"
        
        # Fetch vehicles columns
        try:
            resp = requests.get(vehicles_url, timeout=10)
            resp.raise_for_status()
            vehicles_data = resp.json()
            # Extract column names from columns field
            if "columns" in vehicles_data:
                vehicles = [col.get("fieldName", "") for col in vehicles_data["columns"] if col.get("fieldName")]
        except Exception as e:
            st.warning(f"Failed to fetch vehicles schema from API: {e}")
            # Fallback: try querying with limit=1 to get column names
            try:
                query_url = "https://data.cityofchicago.org/resource/68nd-jvt3.json?$limit=1"
                resp = requests.get(query_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    vehicles = list(data[0].keys())
            except:
                pass
        
        # Fetch people columns
        try:
            resp = requests.get(people_url, timeout=10)
            resp.raise_for_status()
            people_data = resp.json()
            # Extract column names from columns field
            if "columns" in people_data:
                people = [col.get("fieldName", "") for col in people_data["columns"] if col.get("fieldName")]
        except Exception as e:
            st.warning(f"Failed to fetch people schema from API: {e}")
            # Fallback: try querying with limit=1 to get column names
            try:
                query_url = "https://data.cityofchicago.org/resource/u6pd-qa9d.json?$limit=1"
                resp = requests.get(query_url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    people = list(data[0].keys())
            except:
                pass
        
        return {"vehicles": vehicles, "people": people}
    except Exception as e:
        st.error(f"Failed to fetch schema from Chicago API: {e}")
        return {"vehicles": [], "people": []}

def generate_corrid():
    """Generate a correlation ID for the request"""
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    return f"corr={timestamp}"

def render_enrichment_controls(schema, prefix=""):
    """Render the common enrichment controls section.

    prefix: a short string used to namespace Streamlit widget keys (e.g. 'stream' or 'backfill')
    """
    if prefix:
        key_prefix = f"{prefix}_"
    else:
        key_prefix = ""

    st.markdown("### Enrichment Columns")

    # Vehicles section
    st.markdown("#### Vehicles Dataset")
    include_vehicles = st.checkbox("Include Vehicles", key=f"{key_prefix}include_vehicles")
    if include_vehicles:
        select_all_vehicles = st.checkbox("Select all vehicle columns", key=f"{key_prefix}select_all_vehicles")
        vehicle_columns = schema.get("vehicles", [])
        if select_all_vehicles:
            selected_vehicle_cols = vehicle_columns
        else:
            selected_vehicle_cols = st.multiselect(
                "Vehicle columns to fetch",
                options=vehicle_columns,
                default=[],
                key=f"{key_prefix}vehicle_cols"
            )
    else:
        selected_vehicle_cols = []

    # People section
    st.markdown("#### People Dataset")
    include_people = st.checkbox("Include People", key=f"{key_prefix}include_people")
    if include_people:
        select_all_people = st.checkbox("Select all people columns", key=f"{key_prefix}select_all_people")
        people_columns = schema.get("people", [])
        if select_all_people:
            selected_people_cols = people_columns
        else:
            selected_people_cols = st.multiselect(
                "People columns to fetch",
                options=people_columns,
                default=[],
                key=f"{key_prefix}people_cols"
            )
    else:
        selected_people_cols = []

    return selected_vehicle_cols, selected_people_cols

def preview_request(mode, corrid, window, vehicle_cols, people_cols):
    """Generate and display the preview JSON matching streaming.json/backfill.json structure"""
    # Base structure matching the JSON files
    request = {
        "mode": mode,
        "source": "crash",
        "join_key": "crash_record_id",
        "primary": {
            "id": "85ca-t3if",
            "alias": "crashes",
            "order": "crash_date, crash_record_id",
            "page_size": 2000
        },
        "enrich": [],
        "batching": {
            "id_batch_size": 50,
            "max_workers": {
                "vehicles": 4,
                "people": 4
            }
        },
        "storage": {
            "bucket": "raw-data",
            "prefix": "crash",
            "compress": True
        }
    }
    
    # Add mode-specific fields
    if mode == "streaming":
        request["primary"]["where_by"] = {"since_days": window.get("since_days", 7)}
    elif mode == "backfill":
        request["date_range"] = {
            "field": "crash_date",
            "start": window.get("start", ""),
            "end": window.get("end", "")
        }
    
    # Add enrichment datasets
    if vehicle_cols:
        # Ensure crash_record_id is included (it's the join key)
        cols_list = vehicle_cols if isinstance(vehicle_cols, list) else [c.strip() for c in vehicle_cols.split(",") if c.strip()]
        if "crash_record_id" not in cols_list:
            cols_list.insert(0, "crash_record_id")
        request["enrich"].append({
            "id": "68nd-jvt3",
            "alias": "vehicles",
            "select": ",".join(cols_list)
        })
    
    if people_cols:
        # Ensure crash_record_id is included (it's the join key)
        cols_list = people_cols if isinstance(people_cols, list) else [c.strip() for c in people_cols.split(",") if c.strip()]
        if "crash_record_id" not in cols_list:
            cols_list.insert(0, "crash_record_id")
        request["enrich"].append({
            "id": "u6pd-qa9d",
            "alias": "people",
            "select": ",".join(cols_list)
        })
    
    return request

import base64
import requests

import base64
import requests

def publish_to_rabbitmq(api_base, request):
    """Publish the request to RabbitMQ using the Management API"""
    mode = request.get("mode", "unknown")

    try:
        # Generate correlation ID
        corr_id = request.get("corr_id")
        if not corr_id:
            corrid = request.get("corrid", generate_corrid())
            corr_id = corrid.replace("corr=", "") if corrid.startswith("corr=") else corrid
        
        job_request = request.copy()
        job_request["corr_id"] = corr_id
        
        # Write tracking files
        run_dir = Path("minio-data") / "raw-data" / "_runs" / f"corr={corr_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        job_path = run_dir / "job.json"
        job_path.write_text(json.dumps(job_request, indent=2))
        
        manifest = {
            "corr": corr_id,
            "mode": request.get("mode", "unknown"),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "queued"
        }
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        # PUBLISH TO RABBITMQ VIA HTTP API
        is_docker = os.getenv('DOCKER_ENV', 'false').lower() == 'true'
        rabbitmq_host = 'rabbitmq' if is_docker else 'localhost'
        rabbitmq_mgmt_port = int(os.getenv('RABBITMQ_MGMT_PORT', '15672'))
        
        rabbit_user = os.getenv('RABBIT_USER', 'guest')
        rabbit_pass = os.getenv('RABBIT_PASS', 'guest')
        
        # Base64 encode the payload
        payload_json = json.dumps(job_request)
        payload_base64 = base64.b64encode(payload_json.encode('utf-8')).decode('utf-8')
        
        # Build API request
        url = f"http://{rabbitmq_host}:{rabbitmq_mgmt_port}/api/exchanges/%2F/amq.default/publish"
        
        publish_payload = {
            "properties": {
                "content_type": "application/json",
                "delivery_mode": 2
            },
            "routing_key": "extract",
            "payload": payload_base64,
            "payload_encoding": "base64"
        }
        
        # Make HTTP POST to RabbitMQ Management API
        response = requests.post(
            url,
            auth=(rabbit_user, rabbit_pass),
            headers={'content-type': 'application/json'},
            json=publish_payload,
            timeout=10
        )
        
        response.raise_for_status()
        result = response.json()
        jobs_submitted_total.labels(mode=mode, status='success').inc()
        
        if result.get('routed'):
            return {
                "success": True, 
                "message": f"✅ Job published to RabbitMQ!\n\nCorrelation ID: {corr_id}\n\nThe extractor will now:\n1. Fetch data from Chicago API\n2. Write to raw-data bucket\n3. Trigger transformer\n4. Transform data will appear in transform-data bucket"
            }
        else:
            return {
                "success": False, 
                "message": f"⚠️ Published but not routed. Check if extractor service is running."
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False, 
            "message": f"❌ Failed to publish to RabbitMQ: {e}"
        }
    except Exception as e:
        jobs_submitted_total.labels(mode=mode, status='failure').inc()
        errors_total.labels(error_type='rabbitmq').inc()
        return {
            "success": False,
            "message": f"❌ Failed: {e}"
        }
    
def render_data_fetcher():
    page_views_total.labels(page='data_fetcher').inc()
    st.title("Data Fetcher")
    st.markdown("Fetch crash data with enrichments from vehicles and people datasets.")

    # Get API base from sidebar
    api_base = get_api_base()
    
    # Generate correlation ID
    corrid = generate_corrid()
    
    # Fetch schema for enrichment options
    schema = fetch_schema(api_base)
    
    # Create tabs for Streaming and Backfill
    streaming_tab, backfill_tab = st.tabs(["Streaming", "Backfill"])
    
    # i. Streaming Tab
    with streaming_tab:
        st.markdown("### Streaming Mode")
        st.markdown("Fetch recent crash data (last N days)")
        
        # Header info
        st.markdown(f"**Mode:** streaming")
        st.markdown(f"**Correlation ID:** {corrid}")
        
        # Window control
        since_days = st.number_input(
            "Since Days",
            min_value=1,
            max_value=365,
            value=30,
            help="Fetch data from this many days ago until now"
        )
        
        # Enrichment controls
        vehicle_cols, people_cols = render_enrichment_controls(schema, prefix="stream")
        
        # Preview and Actions
        with st.expander("Preview Request JSON"):
            window = {"since_days": since_days}
            request = preview_request("streaming", corrid, window, vehicle_cols, people_cols)
            st.json(request)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Publish to RabbitMQ", key="stream_publish"):
                result = publish_to_rabbitmq(api_base, request)
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
        
        with col2:
            if st.button("Reset Form", key="stream_reset"):
                st.rerun()
    
    # ii. Backfill Tab
    with backfill_tab:
        st.markdown("### Backfill Mode")
        st.markdown("Fetch historical crash data for a specific time range")
        
        # Header info
        st.markdown(f"**Mode:** backfill")
        st.markdown(f"**Correlation ID:** {corrid}")
        
        # Date/Time controls
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", key="backfill_start_date")
            start_time = st.time_input("Start Time", key="backfill_start_time")
        with col2:
            end_date = st.date_input("End Date", key="backfill_end_date")
            end_time = st.time_input("End Time", key="backfill_end_time")
        
        # Combine date and time
        start_dt = datetime.combine(start_date, start_time)
        end_dt = datetime.combine(end_date, end_time)
        
        # Enrichment controls
        vehicle_cols, people_cols = render_enrichment_controls(schema, prefix="backfill")
        
        # Preview and Actions
        with st.expander("Preview Request JSON"):
            window = {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            }
            request = preview_request("backfill", corrid, window, vehicle_cols, people_cols)
            st.json(request)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Publish to RabbitMQ", key="backfill_publish"):
                result = publish_to_rabbitmq(api_base, request)
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
        
        with col2:
            if st.button("Reset Form", key="backfill_reset"):
                st.rerun()


# --- Scheduler helpers ---
def get_schedules():
    path = Path("minio-data/raw-data/_schedules/schedules.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []

def save_schedules(schedules):
    path = Path("minio-data/raw-data/_schedules/schedules.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schedules, indent=2))

def render_scheduler():
    page_views_total.labels(page='scheduler').inc()
    st.title("Scheduler")
    st.markdown("Automate the pipeline to run regularly.")

    # Frequency selection
    freq = st.selectbox("Select Frequency", ["Daily", "Weekly", "Custom cron"], key="sched_freq")
    if freq == "Custom cron":
        cron_str = st.text_input("Cron string (advanced)", value="0 9 * * *", key="sched_cron")
    else:
        cron_str = "0 9 * * *" if freq == "Daily" else "0 9 * * 0"
    run_time = st.time_input("Run start time", value=None, key="sched_time")
    config_type = st.selectbox("Config Type", ["streaming"], key="sched_config_type")

    # Create schedule button
    if st.button("Create Schedule", key="sched_create"):
        # Load streaming.json as config
        try:
            config = json.loads(Path("streaming.json").read_text())
        except Exception as e:
            st.error(f"Failed to load streaming.json: {e}")
            config = {}
        schedules = get_schedules()
        new_sched = {
            "cron": cron_str,
            "run_time": run_time.strftime("%H:%M"),
            "config_type": config_type,
            "config": config,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_run": None
        }
        schedules.append(new_sched)
        save_schedules(schedules)
        st.success("Schedule created!")
        st.rerun()

    # Active schedules table
    st.markdown("### Active Schedules")
    schedules = get_schedules()
    if not schedules:
        st.info("No active schedules.")
    else:
        for i, sched in enumerate(schedules):
            cols = st.columns([2,2,2,2,1])
            cols[0].markdown(f"**Cron:** {sched['cron']}")
            cols[1].markdown(f"**Time:** {sched['run_time']}")
            cols[2].markdown(f"**Type:** {sched['config_type']}")
            cols[3].markdown(f"**Last Run:** {sched['last_run'] or '-'}")
            if cols[4].button("🗑️", key=f"delete_sched_{i}"):
                schedules.pop(i)
                save_schedules(schedules)
                st.rerun()


# --- EDA helpers ---
def load_gold_df():
    db_path = "data/gold/gold.duckdb"
    table = '"gold"."gold"."crash_data"'
    try:
        con = duckdb.connect(db_path)
        df = con.execute(f"SELECT * FROM {table}").df()
        return df
    except Exception as e:
        st.error(f"Failed to load gold.duckdb: {e}")
        return None

def render_eda():
    page_views_total.labels(page='eda').inc()
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Summary statistics and visualizations for cleaned crash data.")

    df = load_gold_df()
    if df is None or df.empty:
        st.warning("No data loaded from gold.duckdb.")
        return

    st.header("Summary Statistics")
    st.markdown(f"**Row count:** {len(df):,}")
    st.markdown("**Missing values:**")
    st.dataframe(df.isnull().sum().to_frame("missing_count"))

    # Numeric stats
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        st.markdown("**Numeric columns:**")
        st.dataframe(df[num_cols].agg(["min", "max", "mean"]).T)
    # Top categories
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique() < 20]
    if cat_cols:
        st.markdown("**Top categories (categorical columns):**")
        for c in cat_cols:
            st.write(f"{c}: {df[c].value_counts().head(5).to_dict()}")

    st.header("Visualizations")
    # Histogram: posted_speed_limit by Crash Type (top types)
    if "posted_speed_limit" in df.columns and "crash_type" in df.columns:
        st.subheader("Histogram: posted_speed_limit by Crash Type")
        top_types = df["crash_type"].value_counts().head(3).index.tolist()
        fig = px.histogram(df[df["crash_type"].isin(top_types)], x="posted_speed_limit", color="crash_type", barmode="overlay", nbins=20, title="Speed Limit Distribution for Top Crash Types")
        st.plotly_chart(fig)

    # Bar chart: weather_condition by Crash Type
    if "weather_condition" in df.columns and "crash_type" in df.columns:
        st.subheader("Bar Chart: Weather Condition by Crash Type")
        top_types = df["crash_type"].value_counts().head(3).index.tolist()
        fig = px.bar(df[df["crash_type"].isin(top_types)], x="weather_condition", color="crash_type", barmode="group", title="Crash Type Counts by Weather Condition")
        st.plotly_chart(fig)

    # Line chart: crash_hour by Crash Type
    if "crash_hour" in df.columns and "crash_type" in df.columns:
        st.subheader("Line Chart: Crash Type by Hour")
        top_types = df["crash_type"].value_counts().head(3).index.tolist()
        ct_hour = df[df["crash_type"].isin(top_types)].groupby(["crash_hour","crash_type"]).size().reset_index(name="count")
        fig = px.line(ct_hour, x="crash_hour", y="count", color="crash_type", title="Crash Type Counts by Hour")
        st.plotly_chart(fig)

    # Pie chart: crash_day_of_week by Crash Type
    if "crash_day_of_week" in df.columns and "crash_type" in df.columns:
        st.subheader("Pie Chart: Crash Type by Day of Week")
        top_types = df["crash_type"].value_counts().head(3).index.tolist()
        for t in top_types:
            day_counts = df[df["crash_type"]==t]["crash_day_of_week"].value_counts().to_frame(name='count').reset_index()
            day_counts.columns = ['day', 'count']
            fig = px.pie(day_counts, values='count', names='day', title=f"Crash Type: {t} by Day of Week")
            st.plotly_chart(fig)

    # Heatmap: hour x day-of-week by Crash Type
    if "crash_hour" in df.columns and "crash_day_of_week" in df.columns and "crash_type" in df.columns:
        st.subheader("Heatmap: Crash Type by Hour and Day of Week")
        heatmap = df.groupby(["crash_hour","crash_day_of_week","crash_type"]).size().reset_index(name="count")
        top_types = df["crash_type"].value_counts().head(3).index.tolist()
        for t in top_types:
            sub = heatmap[heatmap["crash_type"]==t]
            fig = px.density_heatmap(sub, x="crash_hour", y="crash_day_of_week", z="count", title=f"Heatmap for Crash Type: {t}")
            st.plotly_chart(fig)

    # Additional Insights
    st.header("Additional Insights")

    # 1. Box Plot: Speed Limit Distribution by Crash Type
    if "posted_speed_limit" in df.columns and "crash_type" in df.columns:
        st.subheader("1. Speed Limit Distribution by Crash Type")
        top_types = df["crash_type"].value_counts().head(5).index
        fig = px.box(df[df["crash_type"].isin(top_types)], 
                    x="crash_type", y="posted_speed_limit", 
                    title="Speed Limit Distribution by Crash Type")
        st.plotly_chart(fig)

    # 2. Stacked Bar: Lighting Condition vs Crash Type
    if "lighting_condition" in df.columns and "crash_type" in df.columns:
        st.subheader("2. Crash Types by Lighting Condition")
        light_crash = df.groupby(["lighting_condition", "crash_type"]).size().reset_index(name="count")
        fig = px.bar(light_crash, x="lighting_condition", y="count", 
                    color="crash_type", title="Crash Types by Lighting Condition", 
                    barmode="stack")
        st.plotly_chart(fig)

    # 3. Horizontal Bar Chart: Crash Type Counts
    if "crash_type" in df.columns:
        st.subheader("3. Crash Type Distribution")
        try:
            crash_type_counts = df["crash_type"].value_counts().head(10).reset_index()
            crash_type_counts.columns = ["crash_type", "count"]
            fig = px.bar(crash_type_counts, 
                        x="count", 
                        y="crash_type", 
                        orientation='h',
                        title="Top 10 Crash Types by Count",
                        labels={"count": "Number of Crashes", "crash_type": "Crash Type"})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating crash type bar chart: {e}")

    # 4. Violin Plot: Speed Distribution by Top Crash Types
    if "posted_speed_limit" in df.columns and "crash_type" in df.columns:
        st.subheader("4. Speed Distribution by Top Crash Types (Violin)")
        top_types = df["crash_type"].value_counts().head(5).index
        fig = px.violin(df[df["crash_type"].isin(top_types)], 
                       x="crash_type", y="posted_speed_limit",
                       box=True, points="all",
                       title="Speed Distribution by Crash Type")
        st.plotly_chart(fig)

    # 5. Scatter Plot Matrix: Numeric Variables by Crash Type
    if all(col in df.columns for col in ["posted_speed_limit", "crash_hour", "crash_type"]):
        st.subheader("5. Multi-Variable Correlation by Crash Type")
        # Select numeric columns and top crash types
        numeric_cols = ["posted_speed_limit", "crash_hour"]
        top_types = df["crash_type"].value_counts().head(3).index
        scatter_df = df[df["crash_type"].isin(top_types)]
        fig = px.scatter_matrix(scatter_df,
                              dimensions=numeric_cols,
                              color="crash_type",
                              title="Correlation Matrix by Crash Type")
        st.plotly_chart(fig)

    # 6. Stacked Bar: Crash Type by Day of Week
    if "crash_type" in df.columns and "crash_day_of_week" in df.columns:
        st.subheader("6. Crash Types by Day of Week")
        try:
            day_crash = df.groupby(["crash_day_of_week", "crash_type"]).size().reset_index(name="count")
            top_types = df["crash_type"].value_counts().head(5).index
            day_crash_filtered = day_crash[day_crash["crash_type"].isin(top_types)]
            fig = px.bar(day_crash_filtered, 
                        x="crash_day_of_week", 
                        y="count", 
                        color="crash_type",
                        title="Top 5 Crash Types by Day of Week",
                        barmode="stack",
                        labels={"crash_day_of_week": "Day of Week", "count": "Number of Crashes"})
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating stacked bar chart: {e}")

    # 7. Grouped Violin: Characteristics by Crash Type
    if "crash_hour" in df.columns and "crash_type" in df.columns and "posted_speed_limit" in df.columns:
        st.subheader("7. Crash Characteristics Distribution")
        top_types = df["crash_type"].value_counts().head(3).index
        
        # Create subplots for different characteristics
        fig = make_subplots(rows=1, cols=2, 
                           subplot_titles=("Hour of Day", "Speed Limit"))
        
        colors = px.colors.qualitative.Set3[:len(top_types)]
        for i, crash_type in enumerate(top_types):
            subset = df[df["crash_type"] == crash_type]
            
            # Hour distribution
            fig.add_trace(go.Violin(x=[crash_type]*len(subset),
                                  y=subset["crash_hour"],
                                  name=crash_type,
                                  side='positive',
                                  line_color=colors[i],
                                  showlegend=False), row=1, col=1)
                                  
            # Speed limit distribution
            fig.add_trace(go.Violin(x=[crash_type]*len(subset),
                                  y=subset["posted_speed_limit"],
                                  name=crash_type,
                                  side='positive',
                                  line_color=colors[i],
                                  showlegend=True), row=1, col=2)
        
        fig.update_layout(title="Distribution of Crash Characteristics by Type",
                         height=500,
                         showlegend=True)
        st.plotly_chart(fig)

    # 8. Donut Chart: Crash Type Percentage Distribution
    if "crash_type" in df.columns:
        st.subheader("8. Crash Type Percentage Distribution")
        try:
            crash_type_counts = df["crash_type"].value_counts().head(8).reset_index()
            crash_type_counts.columns = ["crash_type", "count"]
            fig = px.pie(crash_type_counts, 
                        values="count", 
                        names="crash_type",
                        hole=0.4,
                        title="Top 8 Crash Types by Percentage",
                        labels={"count": "Count", "crash_type": "Crash Type"})
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error creating donut chart: {e}")

    # 9. Ridgeline Plot: Speed Distribution Overlap
    if "posted_speed_limit" in df.columns and "crash_type" in df.columns:
        st.subheader("9. Ridgeline Plot: Speed Distribution Overlap")
        top_types = df["crash_type"].value_counts().head(4).index
        fig = go.Figure()
        for i, crash_type in enumerate(top_types):
            subset = df[df["crash_type"] == crash_type]["posted_speed_limit"]
            kde = gaussian_kde(subset.dropna())
            x_range = np.linspace(subset.min(), subset.max(), 100)
            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range) + i*0.4,
                                   name=crash_type, fill='tonexty'))
        fig.update_layout(title="Speed Distribution Overlap by Crash Type",
                         showlegend=True)
        st.plotly_chart(fig)

    # 10. Time Series Decomposition
    if "crash_hour" in df.columns and "crash_type" in df.columns:
        st.subheader("10. Hourly Pattern Decomposition")
        top_type = df["crash_type"].value_counts().index[0]
        hourly_counts = df[df["crash_type"] == top_type].groupby("crash_hour").size()
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("24-Hour Pattern", "Deviation from Mean"))
        
        fig.add_trace(go.Scatter(x=hourly_counts.index, y=hourly_counts.values,
                                mode='lines+markers', name='Hourly Count'), 
                     row=1, col=1)
        
        deviation = hourly_counts - hourly_counts.mean()
        fig.add_trace(go.Bar(x=deviation.index, y=deviation.values,
                            name='Deviation'), 
                     row=2, col=1)
        
        fig.update_layout(height=600, title=f"Hourly Pattern Analysis for {top_type}")
        st.plotly_chart(fig)

def get_corrids_from_minio():
    """Get all correlation IDs from raw-data/_runs in MinIO"""
    try:
        s3_client = get_minio_client()
        corrids = set()
        
        # List objects in _runs folder
        paginator = s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket='raw-data', Prefix='_runs/'):
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                # Extract corr= from path like _runs/corr=2025-11-06-23-27-44/...
                parts = obj['Key'].split('/')
                for part in parts:
                    if part.startswith('corr='):
                        corrids.add(part.replace('corr=', ''))
        
        return sorted(list(corrids))
    except Exception as e:
        print(f"Error getting corrids from MinIO: {e}")
        return []

def get_manifest_for_corrid(corrid):
    """Get manifest.json for a specific corrid from _runs folder"""
    try:
        manifest_path = Path("minio-data") / "raw-data" / "_runs" / f"corr={corrid}" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"Error loading manifest for {corrid}: {e}")
        return None

def get_run_history():
    """Get ETL run history from MinIO _runs manifests"""
    corrids = get_corrids_from_minio()
    if not corrids:
        return pd.DataFrame()
    
    runs = []
    for corrid in corrids:
        manifest = get_manifest_from_minio(corrid)
        if manifest:
            runs.append({
                "corrid": corrid,
                "mode": manifest.get("mode", "unknown"),
                "where": manifest.get("where", ""),
                "started_at": manifest.get("started_at", ""),
                "ended_at": manifest.get("finished_at", ""),
            })
        else:
            runs.append({
                "corrid": corrid,
                "mode": "unknown",
                "started_at": "",
                "ended_at": ""
            })
    
    return pd.DataFrame(runs)

def get_latest_run_details():
    """Get details of the most recent run from MinIO"""
    corrids = get_corrids_from_minio()
    if not corrids:
        return None
    
    # Get latest corrid (sorted, so last one is newest)
    latest_corrid = corrids[-1]
    manifest = get_manifest_from_minio(latest_corrid)
    
    if manifest:
        return {
            "corrid": latest_corrid,
            "mode": manifest.get("mode", "unknown"),
            "where": manifest.get("where", ""),
            "started_at": manifest.get("started_at", ""),
            "ended_at": manifest.get("finished_at", "")
        }
    
    return {
        "corrid": latest_corrid,
        "mode": "unknown",
        "started_at": "",
        "ended_at": ""
    }

def get_gold_snapshot():
    """Get current state of gold tables"""
    db_path = "data/gold/gold.duckdb"
    if not Path(db_path).exists():
        return pd.DataFrame()
        
    try:
        con = duckdb.connect(db_path)
        table_name = '"gold"."gold"."crash_data"'
        # Get table info
        snapshot = []
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        max_date = con.execute(f"SELECT MAX(crash_date) FROM {table_name}").fetchone()[0]
        snapshot.append({
            "table": table_name,
            "row_count": count,
            "latest_date": max_date
        })
        con.close()
        return pd.DataFrame(snapshot)
    except Exception as e:
        print(f"Error in get_gold_snapshot: {str(e)}")
        return pd.DataFrame()
    
def get_manifest_from_minio(corrid):
    """Get manifest.json from MinIO for a specific corrid"""
    try:
        s3_client = get_minio_client()
        manifest_key = f"_runs/corr={corrid}/manifest.json"
        
        response = s3_client.get_object(Bucket='raw-data', Key=manifest_key)
        manifest_data = response['Body'].read().decode('utf-8')
        return json.loads(manifest_data)
    except Exception as e:
        print(f"Error loading manifest from MinIO for {corrid}: {e}")
        # Fallback to local filesystem
        try:
            manifest_path = Path("minio-data") / "raw-data" / "_runs" / f"corr={corrid}" / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    return json.load(f)
        except Exception as e2:
            print(f"Error loading manifest locally for {corrid}: {e2}")
        return None
    
def get_gold_row_count():
    """Get total row count from gold.duckdb"""
    db_path = "data/gold/gold.duckdb"
    if not Path(db_path).exists():
        return 0
    
    try:
        count = 0        
        con = duckdb.connect(db_path)
        table_name = '"gold"."gold"."crash_data"'
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        con.close()
        gold_db_rows.set(count)
        return count
    except Exception as e:
        errors_total.labels(error_type='data_load').inc()
        print(f"Error getting gold row count: {e}")
        return 0

def get_latest_crash_date():
    """Get max crash_date from gold.duckdb"""
    db_path = "data/gold/gold.duckdb"
    if not Path(db_path).exists():
        return None
    
    try:
        con = duckdb.connect(db_path)
        table_name = '"gold"."gold"."crash_data"'
        max_date = con.execute(f"SELECT MAX(crash_date) FROM {table_name}").fetchone()[0]
        con.close()
        if max_date:
            timestamp = max_date.timestamp() if hasattr(max_date, 'timestamp') else 0
            gold_db_latest_date.set(timestamp)
        return max_date
    except Exception as e:
        errors_total.labels(error_type='data_load').inc() 
        print(f"Error getting latest crash date: {e}")
        return None

def get_cleaner_upsert_log(corrid):
    """Get upsert log from cleaner for rows processed"""
    try:
        # Check local file system for cleaner logs
        log_path = Path("minio-data") / "raw-data" / "_runs" / f"corr={corrid}" / "cleaner_upsert.log"
        if log_path.exists():
            with open(log_path) as f:
                return f.read()
        return None
    except Exception as e:
        print(f"Error loading cleaner log for {corrid}: {e}")
        return None

def render_reports():
    page_views_total.labels(page='reports').inc()
    st.title("Reports")
    st.markdown("Summarized metrics and health of the ETL process over time.")

    # Get data
    corrids = get_corrids_from_minio()
    latest_run = get_latest_run_details()
    gold_row_count = get_gold_row_count()
    latest_crash_date = get_latest_crash_date()

    # Summary Cards
    st.header("📊 Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_corrids = len(corrids)
        st.metric("Total Runs (Corrids)", total_corrids)
            
    with col2:
        st.metric("Gold Row Count", f"{gold_row_count:,}")
        
    with col3:
        if latest_crash_date:
            st.metric("Latest Data Date Fetched", latest_crash_date.strftime("%Y-%m-%d") if hasattr(latest_crash_date, 'strftime') else str(latest_crash_date))
        else:
            st.metric("Latest Data Date Fetched", "N/A")
            
    with col4:
        if latest_run and latest_run.get("ended_at"):
            st.metric("Last Run Timestamp", latest_run["ended_at"])
        else:
            st.metric("Last Run Timestamp", "N/A")

    # Latest Corr ID with copy
    st.markdown("---")
    if corrids:
        latest_corrid = corrids[-1]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Latest Corr ID:** `{latest_corrid}`")
        with col2:
            if st.button("📋 Copy Corr ID", key="copy_corrid"):
                st.code(latest_corrid, language=None)
                st.success("Corr ID shown above - use Ctrl+C to copy")

    # Latest Run Summary
    st.header("🔍 Latest Run Summary")
    if latest_run:
        st.markdown(f"### Run: `{latest_run['corrid']}`")
        
        # Config from manifest
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Configuration (from manifest.json):**")
            manifest = get_manifest_from_minio(latest_run['corrid'])
            if manifest:
                config_data = {
                    "mode": manifest.get("mode", "unknown"),
                    "where": manifest.get("where", ""),
                }
                st.json(config_data)
            else:
                st.info("Manifest not available")
            
        with col2:
            st.markdown("**Timing (from manifest.json):**")
            if manifest:
                st.markdown(f"- **Started:** {manifest.get('started_at', 'N/A')}")
                st.markdown(f"- **Finished:** {manifest.get('finished_at', manifest.get('ended_at', 'N/A'))}")
                
                # Calculate duration
                if manifest.get("started_at") and manifest.get("finished_at"):
                    try:
                        start = datetime.fromisoformat(manifest["started_at"].replace("Z", "+00:00"))
                        end = datetime.fromisoformat(manifest["finished_at"].replace("Z", "+00:00"))
                        duration = end - start
                        st.markdown(f"- **Duration:** {duration}")
                    except:
                        pass
        
        # Rows processed
        st.markdown("**Rows Processed:**")
        st.markdown(f"Total rows in Gold DB: **{gold_row_count:,}**")
        
        # Request JSON Preview
        with st.expander("📄 Request JSON Preview"):
            job_path = Path("minio-data") / "raw-data" / "_runs" / f"corr={latest_run['corrid']}" / "job.json"
            if job_path.exists():
                with open(job_path) as f:
                    request_json = json.load(f)
                st.json(request_json)
                st.download_button(
                    "Download Request JSON",
                    data=json.dumps(request_json, indent=2),
                    file_name=f"request_{latest_run['corrid']}.json",
                    mime="application/json"
                )
            else:
                st.info("Request JSON not available")
    else:
        st.info("No runs found")

    # Download Reports
    st.header("📥 Download Reports")
    
    runs_df = get_run_history()
    gold_snapshot = get_gold_snapshot()
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Run History**")
        if not runs_df.empty:
            csv = runs_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"run_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
            pdf_content = "RUN HISTORY REPORT\n" + "="*50 + "\n\n"
            pdf_content += runs_df.to_string(index=False)
            st.download_button(
                label="Download PDF (Text)",
                data=pdf_content,
                file_name=f"run_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No run history available")

    with col2:
        st.markdown("**Gold Snapshot**")
        if not gold_snapshot.empty:
            csv = gold_snapshot.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"gold_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            pdf_content = "GOLD SNAPSHOT REPORT\n" + "="*50 + "\n\n"
            pdf_content += gold_snapshot.to_string(index=False)
            st.download_button(
                label="Download PDF (Text)",
                data=pdf_content,
                file_name=f"gold_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No gold snapshot available")
            
    with col3:
        st.markdown("**Summary Metrics**")
        summary_data = {
            "Total Runs": len(corrids),
            "Gold Row Count": gold_row_count,
            "Latest Data Date": str(latest_crash_date) if latest_crash_date else "N/A",
            "Last Run": latest_run.get("ended_at", "N/A") if latest_run else "N/A"
        }
        summary_df = pd.DataFrame(list(summary_data.items()), columns=["Metric", "Value"])
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"summary_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ===== Fixed paths (aligns with your project layout) =====
MODEL_ARTIFACT_PATH = "artifacts/model.pkl"
THRESHOLD_PATH = "artifacts/threshold.txt"
LABELS_PATH = "artifacts/labels.json"
FEATURE_NAMES_PATH = "artifacts/feature_names.json"  # optional
DUCKDB_PATH = "data/gold/gold.duckdb"
GOLD_TABLE = '"gold"."gold"."crash_data"'

# ===== Static benchmark metrics from your notebook (Section 13 output) =====
STATIC_METRICS = {
    "primary_metric": "average_precision",
    "primary_score": 0.738,
    "precision@t": 0.733,
    "recall@t": 0.534,
    "f1@t": 0.618,
    "roc_auc": 0.818,
    "pr_auc": 0.738,
    "threshold": 0.53
}

# ===== Cached helpers =====
@st.cache_resource(show_spinner=False)
def load_model_artifact():
    """Load model with metrics tracking"""
    try:
        clf = load("artifacts/model.pkl")
        model_load_status.set(1)
        return clf, None
    except Exception as e:
        model_load_status.set(0)
        errors_total.labels(error_type='model_load').inc()
        return None, f"Failed to load model from artifacts/model.pkl: {e}"
    
@st.cache_resource(show_spinner=False)
def load_threshold():
    try:
        with open(THRESHOLD_PATH, "r") as f:
            t = float(f.read().strip())
        return t, None
    except Exception as e:
        return None, f"Failed to load threshold: {e}"

@st.cache_resource(show_spinner=False)
def load_labels():
    try:
        with open(LABELS_PATH, "r") as f:
            data = json.load(f)
        classes = data.get("classes", [])
        positive_class = data.get("positive_class", "INJURY AND / OR TOW DUE TO CRASH")
        return classes, positive_class, None
    except Exception as e:
        return None, None, f"Failed to load labels: {e}"

@st.cache_resource(show_spinner=False)
def connect_duckdb_model():
    """Connect to DuckDB with metrics tracking"""
    try:
        con = duckdb.connect("data/gold/gold.duckdb")
        return con, None
    except Exception as e:
        errors_total.labels(error_type='data_load').inc()
        return None, f"Failed to connect to DuckDB: {e}"

@st.cache_data(show_spinner=False)
def query_gold(con, start_date=None, end_date=None, max_rows=5000):
    """Query gold table with metrics tracking"""
    start_time = time.time()
    
    try:
        where = []
        if start_date:
            where.append(f"crash_date >= TIMESTAMP '{start_date} 00:00:00'")
        if end_date:
            where.append(f"crash_date <= TIMESTAMP '{end_date} 23:59:59'")
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        
        sql = f"""
            SELECT * FROM "gold"."gold"."crash_data"
            {where_sql}
            ORDER BY crash_date
            LIMIT {int(max_rows)}
        """
        
        df = con.execute(sql).df()
        
        # Track metrics
        duration = time.time() - start_time
        data_load_duration_seconds.labels(source='gold_db').observe(duration)
        data_rows_loaded.labels(source='gold_db').inc(len(df))
        
        return df
        
    except Exception as e:
        errors_total.labels(error_type='data_load').inc()
        raise

# ===== Schema helpers =====
def get_expected_raw_columns_from_pipeline(model):
    # CalibratedClassifierCV wraps the pipeline; get underlying estimator
    base = getattr(model, "base_estimator", None) or getattr(model, "estimator", None) or model
    preprocess = None
    if hasattr(base, "named_steps"):
        preprocess = base.named_steps.get("preprocess", None)
    if preprocess is None:
        return None, "Pipeline has no 'preprocess' step; cannot derive expected raw columns."
    try:
        num_cols = preprocess.transformers_[0][2]
        cat_cols = preprocess.transformers_[1][2]
        return list(num_cols) + list(cat_cols), None
    except Exception as e:
        return None, f"Failed to extract raw feature columns: {e}"

def validate_schema(df, expected_raw_cols):
    missing = [c for c in expected_raw_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_raw_cols]
    return missing, extra

# ===== Metrics helper =====
def compute_live_metrics(y_true, proba_pos, threshold, positive_class):
    """Compute metrics with Prometheus tracking"""
    
    y_true_bin = (y_true == positive_class).astype(int).to_numpy()
    y_pred_bin = (proba_pos >= threshold).astype(int)
    
    # Compute metrics
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    roc_auc = roc_auc_score(y_true_bin, proba_pos)
    pr_auc = average_precision_score(y_true_bin, proba_pos)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    
    # Update Prometheus metrics
    model_accuracy.set(accuracy)
    model_precision.set(precision)
    model_recall.set(recall)
    model_f1_score.set(f1)
    model_roc_auc.set(roc_auc)
    model_pr_auc.set(pr_auc)
    
    # Track confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    confusion_matrix_values.labels(actual='negative', predicted='negative').set(cm[0, 0])
    confusion_matrix_values.labels(actual='negative', predicted='positive').set(cm[0, 1])
    confusion_matrix_values.labels(actual='positive', predicted='negative').set(cm[1, 0])
    confusion_matrix_values.labels(actual='positive', predicted='positive').set(cm[1, 1])
    
    # Count predictions
    positive_preds = y_pred_bin.sum()
    negative_preds = len(y_pred_bin) - positive_preds
    predictions_total.labels(outcome='positive').inc(positive_preds)
    predictions_total.labels(outcome='negative').inc(negative_preds)
    
    metrics = {
        "accuracy": accuracy,
        "precision@t": precision,
        "recall@t": recall,
        "f1@t": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    
    return metrics, cm

# ===== UI: Model Tab =====
def render_model_tab():
    """Render model tab with comprehensive metrics tracking"""
    page_views_total.labels(page='model').inc()
    
    st.title("Model")

    # Section 1 – Model Summary
    st.subheader("Model Summary")
    clf, err = load_model_artifact()
    if err:
        st.error(err)
        return

    threshold, terr = load_threshold()
    if terr:
        st.error(terr)
        return

    classes, positive_class, lerr = load_labels()
    if lerr:
        st.error(lerr)
        return

    outer_class_name = clf.__class__.__name__
    base = getattr(clf, "base_estimator", None) or getattr(clf, "estimator", None) or clf
    inner_class_name = base.__class__.__name__

    st.write(f"- Outer model: {outer_class_name}")
    st.write(f"- Underlying estimator: {inner_class_name}")
    st.write(f"- Decision threshold: {threshold:.2f}")
    st.write(f"- Positive class: {positive_class}")

    expected_raw_cols, colerr = get_expected_raw_columns_from_pipeline(clf)
    if colerr:
        st.warning(colerr)
        expected_raw_cols = []
    if expected_raw_cols:
        with st.expander("Expected input columns (raw features)"):
            st.write(f"{len(expected_raw_cols)} columns")
            st.text(", ".join(expected_raw_cols))

    st.info("The model expects raw, correctly named columns. One‑hot encoding and numeric imputation are handled inside the pipeline. Do not manually encode data before calling the model.")

    # Section 2 – Data Selection
    st.subheader("Data Selection")
    mode = st.radio("Choose data source", ["Gold table (filtered)", "Test CSV (upload)"], horizontal=True)

    data_df = None
    source_desc = ""
    if mode == "Gold table (filtered)":
        con, cerr = connect_duckdb_model()
        if cerr:
            st.error(cerr)
            return
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start date", value=None)
        with col2:
            end_date = st.date_input("End date", value=None)
        with col3:
            max_rows = st.number_input("Max rows", min_value=1, max_value=20000, value=5000, step=100)
        if st.button("Load gold data"):
            load_start = time.time()
            try:
                data_df = query_gold(con, start_date=start_date, end_date=end_date, max_rows=max_rows)
                load_duration = time.time() - load_start
                data_load_duration_seconds.labels(source='gold_db').observe(load_duration)
                data_rows_loaded.labels(source='gold_db').inc(len(data_df))
                source_desc = f"Gold table (rows={len(data_df)})"
            except Exception as e:
                errors_total.labels(error_type='data_load').inc()
                st.error(f"Failed to load data: {e}")
    else:
        uploaded = st.file_uploader("Upload test CSV", type=["csv"])
        if uploaded:
            load_start = time.time()
            try:
                data_df = pd.read_csv(uploaded)
                load_duration = time.time() - load_start
                data_load_duration_seconds.labels(source='csv_upload').observe(load_duration)
                data_rows_loaded.labels(source='csv_upload').inc(len(data_df))
                source_desc = f"Uploaded CSV (rows={len(data_df)})"
            except Exception as e:
                errors_total.labels(error_type='data_load').inc()
                st.error(f"Failed to read CSV: {e}")

    if data_df is not None:
        st.write(f"Loaded: {source_desc}")
        st.dataframe(data_df.head())

        # Schema validation
        if expected_raw_cols:
            missing, extra = validate_schema(data_df, expected_raw_cols)
            if missing:
                st.error(f"Missing required feature columns: {missing}")
                st.stop()
            elif extra:
                st.warning(f"Extra columns present (ignored by pipeline): {extra[:10]}")

        # Section 3 – Prediction & Metrics
        st.subheader("Prediction & Metrics")
        
        # Time the prediction
        pred_start = time.time()
        try:
            proba = clf.predict_proba(data_df)
            pred_duration = time.time() - pred_start
            prediction_duration_seconds.observe(pred_duration)
            
        except Exception as e:
            errors_total.labels(error_type='prediction').inc()
            st.error(f"Prediction failed. Ensure columns are raw and match training schema. Details: {e}")
            st.stop()

        class_list = list(getattr(clf, "classes_", []))
        if not class_list or positive_class not in class_list:
            st.error("Model classes_ missing or positive_class not found.")
            st.stop()
        pos_idx = class_list.index(positive_class)
        proba_pos = proba[:, pos_idx]
        y_pred = (proba_pos >= threshold).astype(int)

        # Static (reference) metrics from notebook
        st.markdown("Reference (held-out TEST from notebook):")
        sm = STATIC_METRICS
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Primary (PR-AUC)", f"{sm['primary_score']:.3f}")
        colB.metric("ROC-AUC", f"{sm['roc_auc']:.3f}")
        colC.metric("Precision@t", f"{sm['precision@t']:.3f}")
        colD.metric("Recall@t", f"{sm['recall@t']:.3f}")
        st.caption(f"Threshold t={sm['threshold']:.2f}")

        # Live metrics (only if ground truth present)
        if "crash_type" in data_df.columns:
            live_metrics, cm = compute_live_metrics(data_df["crash_type"], proba_pos, threshold, positive_class)
            
            # Display metrics
            lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns(5)
            lcol1.metric("Live Accuracy", f"{live_metrics.get('accuracy', 0):.3f}")
            lcol2.metric("Live PR-AUC", f"{live_metrics['pr_auc']:.3f}")
            lcol3.metric("Live ROC-AUC", f"{live_metrics['roc_auc']:.3f}")
            lcol4.metric("Live Precision@t", f"{live_metrics['precision@t']:.3f}")
            lcol5.metric("Live Recall@t", f"{live_metrics['recall@t']:.3f}")
            
            st.write(f"Prediction latency: {pred_duration:.3f}s for {len(data_df)} rows")

            st.write("Confusion Matrix (t = {:.2f})".format(threshold))
            cm_df = pd.DataFrame(cm, index=["Actual Neg", "Actual Pos"], columns=["Pred Neg", "Pred Pos"])
            st.dataframe(cm_df)

            prec_curve, rec_curve, thr_curve = precision_recall_curve(
                (data_df["crash_type"] == positive_class).astype(int), proba_pos
            )
            st.line_chart(pd.DataFrame({"Recall": rec_curve, "Precision": prec_curve}))
        else:
            st.info("Ground truth (crash_type) not present. Showing predictions only — metrics unavailable.")

        # Scored output preview
        out_preview = data_df.copy()
        out_preview["proba_pos"] = proba_pos
        out_preview["pred_label"] = np.where(y_pred == 1, positive_class, "NEGATIVE")
        st.write("Scored output preview:")
        st.dataframe(out_preview.head(20))
    else:
        st.info("Select a data mode and load data to run predictions and view metrics.")

def main():
	pages = {
		"Home": render_home,
		"Data Management": render_data_management,
		"Data Fetcher": render_data_fetcher,
		"Scheduler": render_scheduler,
		"EDA": render_eda,
		"Reports": render_reports,
        "Model": render_model_tab, # new tab
	}

	st.sidebar.title("Navigation")
	choice = st.sidebar.radio("Go to", list(pages.keys()))

	pages[choice]()


if __name__ == "__main__":
	main()


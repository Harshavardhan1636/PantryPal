"""
Prometheus Metrics Middleware for PantryPal

Collects comprehensive metrics for:
- Application performance (latency, errors, requests)
- Database connection pool
- ML prediction performance
- Model drift detection
- Business KPIs
- Worker queue health
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST
)
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ============================================================================
# Application Metrics
# ============================================================================

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

http_errors_total = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'error_type']
)

# Database metrics
db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Current database connection pool size'
)

db_connection_pool_available = Gauge(
    'db_connection_pool_available',
    'Available connections in pool'
)

db_connection_pool_in_use = Gauge(
    'db_connection_pool_in_use',
    'Connections currently in use'
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
)

db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['query_type', 'status']
)

# ============================================================================
# ML Prediction Metrics
# ============================================================================

ml_prediction_requests_total = Counter(
    'ml_prediction_requests_total',
    'Total ML prediction requests',
    ['model_name', 'prediction_type']
)

ml_prediction_duration_seconds = Histogram(
    'ml_prediction_duration_seconds',
    'ML prediction latency in seconds',
    ['model_name', 'prediction_type'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0)
)

ml_prediction_errors_total = Counter(
    'ml_prediction_errors_total',
    'Total ML prediction errors',
    ['model_name', 'error_type']
)

ml_model_inference_batch_size = Histogram(
    'ml_model_inference_batch_size',
    'Batch size for ML inference',
    ['model_name'],
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000)
)

# Model drift metrics
ml_model_drift_score = Gauge(
    'ml_model_drift_score',
    'Model drift score (PSI - Population Stability Index)',
    ['model_name', 'feature_name']
)

ml_model_prediction_distribution = Histogram(
    'ml_model_prediction_distribution',
    'Distribution of model predictions',
    ['model_name'],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

ml_model_version_info = Info(
    'ml_model_version',
    'Current ML model version and metadata'
)

# A/B test metrics
ml_ab_test_assignments_total = Counter(
    'ml_ab_test_assignments_total',
    'Total A/B test assignments',
    ['experiment_name', 'variant']
)

ml_ab_test_conversions_total = Counter(
    'ml_ab_test_conversions_total',
    'Total A/B test conversions',
    ['experiment_name', 'variant']
)

# ============================================================================
# Business Metrics
# ============================================================================

business_active_households_weekly = Gauge(
    'business_active_households_weekly',
    'Weekly active households (last 7 days)'
)

business_new_households_daily = Counter(
    'business_new_households_daily',
    'New households registered today'
)

business_waste_events_total = Counter(
    'business_waste_events_total',
    'Total waste events',
    ['waste_reason']
)

business_waste_events_per_household = Gauge(
    'business_waste_events_per_household',
    'Average waste events per household (weekly)'
)

business_money_saved_estimate_usd = Gauge(
    'business_money_saved_estimate_usd',
    'Estimated money saved by households (cumulative)',
    ['household_id']
)

business_feature_opt_in_rate = Gauge(
    'business_feature_opt_in_rate',
    'Feature opt-in rate percentage',
    ['feature_name']
)

business_pantry_items_total = Gauge(
    'business_pantry_items_total',
    'Total pantry items across all households'
)

business_receipts_processed_total = Counter(
    'business_receipts_processed_total',
    'Total receipts processed',
    ['processing_method']
)

# ============================================================================
# Worker Queue Metrics
# ============================================================================

worker_queue_length = Gauge(
    'worker_queue_length',
    'Current queue length',
    ['queue_name']
)

worker_tasks_total = Counter(
    'worker_tasks_total',
    'Total worker tasks',
    ['queue_name', 'task_name', 'status']
)

worker_task_duration_seconds = Histogram(
    'worker_task_duration_seconds',
    'Worker task duration in seconds',
    ['queue_name', 'task_name'],
    buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

worker_task_retries_total = Counter(
    'worker_task_retries_total',
    'Total task retries',
    ['queue_name', 'task_name']
)

worker_queue_age_seconds = Gauge(
    'worker_queue_age_seconds',
    'Oldest message age in queue (seconds)',
    ['queue_name']
)

# ============================================================================
# Middleware
# ============================================================================

class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect Prometheus metrics for all HTTP requests.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()
        
        # Extract request metadata
        method = request.method
        endpoint = request.url.path
        
        # Request size
        request_size = int(request.headers.get("content-length", 0))
        http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_size)
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error_type = None
            
        except Exception as e:
            # Record error
            status_code = 500
            error_type = type(e).__name__
            http_errors_total.labels(
                method=method,
                endpoint=endpoint,
                error_type=error_type
            ).inc()
            raise
        
        finally:
            # Record metrics
            duration = time.time() - start_time
            
            # Request count
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            # Latency
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
        
        # Response size
        response_size = int(response.headers.get("content-length", 0))
        http_response_size_bytes.labels(method=method, endpoint=endpoint).observe(response_size)
        
        return response


# ============================================================================
# Database Metrics Collection
# ============================================================================

def update_db_connection_pool_metrics(engine) -> None:
    """Update database connection pool metrics."""
    pool = engine.pool
    
    db_connection_pool_size.set(pool.size())
    db_connection_pool_available.set(pool.size() - pool.checkedout())
    db_connection_pool_in_use.set(pool.checkedout())


def record_db_query(query_type: str, duration: float, success: bool) -> None:
    """Record database query metrics."""
    db_queries_total.labels(
        query_type=query_type,
        status="success" if success else "failure"
    ).inc()
    
    db_query_duration_seconds.labels(query_type=query_type).observe(duration)


# ============================================================================
# ML Prediction Metrics Collection
# ============================================================================

def record_ml_prediction(
    model_name: str,
    prediction_type: str,
    duration: float,
    batch_size: int = 1,
    success: bool = True,
    error_type: str = None
) -> None:
    """Record ML prediction metrics."""
    
    ml_prediction_requests_total.labels(
        model_name=model_name,
        prediction_type=prediction_type
    ).inc()
    
    ml_prediction_duration_seconds.labels(
        model_name=model_name,
        prediction_type=prediction_type
    ).observe(duration)
    
    ml_model_inference_batch_size.labels(model_name=model_name).observe(batch_size)
    
    if not success and error_type:
        ml_prediction_errors_total.labels(
            model_name=model_name,
            error_type=error_type
        ).inc()


def update_model_drift_metrics(model_name: str, feature_name: str, psi_score: float) -> None:
    """Update model drift metrics (Population Stability Index)."""
    ml_model_drift_score.labels(
        model_name=model_name,
        feature_name=feature_name
    ).set(psi_score)


def record_model_prediction_value(model_name: str, prediction_value: float) -> None:
    """Record individual prediction value for distribution tracking."""
    ml_model_prediction_distribution.labels(model_name=model_name).observe(prediction_value)


def set_model_version_info(model_name: str, version: str, training_date: str, metrics: dict) -> None:
    """Set model version information."""
    ml_model_version_info.info({
        'model_name': model_name,
        'version': version,
        'training_date': training_date,
        'auc': str(metrics.get('auc', 0)),
        'precision': str(metrics.get('precision', 0)),
        'recall': str(metrics.get('recall', 0))
    })


# ============================================================================
# Business Metrics Collection
# ============================================================================

def update_business_metrics(
    active_households_weekly: int,
    waste_events_per_household: float,
    pantry_items_total: int
) -> None:
    """Update business KPI metrics."""
    
    business_active_households_weekly.set(active_households_weekly)
    business_waste_events_per_household.set(waste_events_per_household)
    business_pantry_items_total.set(pantry_items_total)


def record_household_registration() -> None:
    """Record new household registration."""
    business_new_households_daily.inc()


def record_waste_event(waste_reason: str) -> None:
    """Record waste event with reason."""
    business_waste_events_total.labels(waste_reason=waste_reason).inc()


def update_money_saved_estimate(household_id: int, amount_usd: float) -> None:
    """Update money saved estimate for household."""
    business_money_saved_estimate_usd.labels(household_id=str(household_id)).set(amount_usd)


def update_feature_opt_in_rate(feature_name: str, opt_in_percentage: float) -> None:
    """Update feature opt-in rate."""
    business_feature_opt_in_rate.labels(feature_name=feature_name).set(opt_in_percentage)


def record_receipt_processed(processing_method: str) -> None:
    """Record receipt processing."""
    business_receipts_processed_total.labels(processing_method=processing_method).inc()


# ============================================================================
# Worker Queue Metrics Collection
# ============================================================================

def update_worker_queue_metrics(queue_name: str, length: int, oldest_age_seconds: float) -> None:
    """Update worker queue metrics."""
    worker_queue_length.labels(queue_name=queue_name).set(length)
    worker_queue_age_seconds.labels(queue_name=queue_name).set(oldest_age_seconds)


def record_worker_task(
    queue_name: str,
    task_name: str,
    duration: float,
    status: str,  # success, failure, timeout
    retries: int = 0
) -> None:
    """Record worker task execution."""
    
    worker_tasks_total.labels(
        queue_name=queue_name,
        task_name=task_name,
        status=status
    ).inc()
    
    worker_task_duration_seconds.labels(
        queue_name=queue_name,
        task_name=task_name
    ).observe(duration)
    
    if retries > 0:
        worker_task_retries_total.labels(
            queue_name=queue_name,
            task_name=task_name
        ).inc(retries)


# ============================================================================
# Metrics Endpoint
# ============================================================================

def setup_metrics_endpoint(app: FastAPI) -> None:
    """
    Add Prometheus metrics endpoint to FastAPI app.
    """
    
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    # Add middleware
    app.add_middleware(PrometheusMetricsMiddleware)
    
    logger.info("✅ Prometheus metrics endpoint configured at /metrics")

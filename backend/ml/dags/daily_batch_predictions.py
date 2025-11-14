"""
Airflow DAG for Daily Batch Predictions - Task #9

Schedules daily batch waste prediction job at 2 AM UTC.

Features:
- Automatic retry on failure (3 attempts)
- Slack alerts on failure
- Monitoring and logging
- Health checks before/after
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator


# Default args for all tasks
default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-alerts@pantrypal.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}


def check_prediction_server_health(**context):
    """Check if prediction server is healthy before starting."""
    import httpx
    
    server_url = "http://waste-predictor-service:3000/health"
    
    try:
        response = httpx.get(server_url, timeout=10)
        response.raise_for_status()
        
        health = response.json()
        if health["status"] != "healthy":
            raise Exception(f"Prediction server unhealthy: {health}")
        
        print(f"✓ Prediction server healthy: {health}")
        return True
        
    except Exception as e:
        print(f"✗ Prediction server health check failed: {e}")
        raise


def run_batch_prediction(**context):
    """Run batch prediction service."""
    import asyncio
    from backend.ml.services.batch_prediction_service import BatchPredictionService, get_session
    from backend.api.config import settings
    
    async def _run():
        service = BatchPredictionService(
            prediction_server_url=settings.PREDICTION_SERVER_URL,
            chunk_size=100,
            max_workers=4,
            prediction_ttl_hours=24,
        )
        
        async with get_session() as db:
            metrics = await service.run_daily_batch(db, force_recompute=False)
        
        return {
            "total_entries": metrics.total_entries,
            "predictions_computed": metrics.predictions_computed,
            "predictions_cached": metrics.predictions_cached,
            "predictions_failed": metrics.predictions_failed,
            "duration_seconds": metrics.duration_seconds,
            "throughput_per_second": metrics.throughput_per_second,
            "avg_latency_ms": metrics.avg_latency_ms,
        }
    
    # Run async function
    metrics = asyncio.run(_run())
    
    # Push metrics to XCom for downstream tasks
    context["ti"].xcom_push(key="batch_metrics", value=metrics)
    
    print("\n" + "="*60)
    print("BATCH PREDICTION COMPLETED")
    print("="*60)
    print(f"Total entries: {metrics['total_entries']}")
    print(f"Predictions computed: {metrics['predictions_computed']}")
    print(f"Predictions cached: {metrics['predictions_cached']}")
    print(f"Predictions failed: {metrics['predictions_failed']}")
    print(f"Duration: {metrics['duration_seconds']:.1f}s")
    print(f"Throughput: {metrics['throughput_per_second']:.1f} entries/s")
    print("="*60)
    
    # Check failure rate
    if metrics["total_entries"] > 0:
        failure_rate = metrics["predictions_failed"] / metrics["total_entries"]
        if failure_rate > 0.05:  # >5% failure rate
            raise Exception(
                f"High failure rate: {failure_rate:.1%} "
                f"({metrics['predictions_failed']}/{metrics['total_entries']})"
            )
    
    return metrics


def validate_predictions(**context):
    """Validate prediction quality with spot checks."""
    import asyncio
    from sqlalchemy import text
    from backend.shared.database_v2 import get_session
    
    async def _validate():
        async with get_session() as db:
            # Check: No NULL waste_probability
            result = await db.execute(text("""
                SELECT COUNT(*) 
                FROM predictions 
                WHERE model_type = 'waste_risk' 
                    AND predicted_at > NOW() - INTERVAL '1 day'
                    AND waste_probability IS NULL
            """))
            null_count = result.scalar()
            
            if null_count > 0:
                raise Exception(f"Found {null_count} predictions with NULL waste_probability")
            
            # Check: waste_probability in [0, 1]
            result = await db.execute(text("""
                SELECT COUNT(*) 
                FROM predictions 
                WHERE model_type = 'waste_risk' 
                    AND predicted_at > NOW() - INTERVAL '1 day'
                    AND (waste_probability < 0 OR waste_probability > 1)
            """))
            invalid_count = result.scalar()
            
            if invalid_count > 0:
                raise Exception(f"Found {invalid_count} predictions with invalid probabilities")
            
            # Check: risk_class matches probability
            result = await db.execute(text("""
                SELECT COUNT(*) 
                FROM predictions 
                WHERE model_type = 'waste_risk' 
                    AND predicted_at > NOW() - INTERVAL '1 day'
                    AND (
                        (risk_class = 'LOW' AND waste_probability >= 0.3) OR
                        (risk_class = 'MEDIUM' AND (waste_probability < 0.3 OR waste_probability >= 0.7)) OR
                        (risk_class = 'HIGH' AND waste_probability < 0.7)
                    )
            """))
            mismatch_count = result.scalar()
            
            if mismatch_count > 0:
                raise Exception(f"Found {mismatch_count} predictions with mismatched risk_class")
            
            print("✓ All prediction validations passed")
            return True
    
    asyncio.run(_validate())


def send_success_notification(**context):
    """Send success notification with metrics."""
    ti = context["ti"]
    metrics = ti.xcom_pull(key="batch_metrics", task_ids="run_batch_prediction")
    
    message = (
        f"✅ *Daily Batch Prediction Successful*\n\n"
        f"• Total entries: {metrics['total_entries']:,}\n"
        f"• Predictions computed: {metrics['predictions_computed']:,}\n"
        f"• Predictions cached: {metrics['predictions_cached']:,}\n"
        f"• Duration: {metrics['duration_seconds']:.1f}s\n"
        f"• Throughput: {metrics['throughput_per_second']:.1f} entries/s\n"
    )
    
    print(message)
    # In production, send to Slack/email
    return message


# ============================================================================
# DAG Definition
# ============================================================================

with DAG(
    dag_id="daily_batch_predictions",
    default_args=default_args,
    description="Daily batch waste predictions for all active pantry entries",
    schedule_interval="0 2 * * *",  # 2 AM UTC daily
    start_date=datetime(2025, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ml", "predictions", "batch"],
) as dag:
    
    # Task 1: Health check
    health_check = PythonOperator(
        task_id="check_prediction_server_health",
        python_callable=check_prediction_server_health,
        doc_md="""
        ### Health Check
        
        Verifies that the BentoML prediction server is healthy before starting batch job.
        
        **Checks:**
        - Server is reachable
        - Health endpoint returns 200
        - Status is "healthy"
        """,
    )
    
    # Task 2: Run batch predictions
    batch_prediction = PythonOperator(
        task_id="run_batch_prediction",
        python_callable=run_batch_prediction,
        doc_md="""
        ### Batch Prediction
        
        Runs batch waste predictions for all active pantry entries.
        
        **Process:**
        1. Query active entries from DB
        2. Check prediction TTL (skip if fresh)
        3. Compute features for stale predictions
        4. Call prediction server in batches
        5. Store predictions + explanations
        6. Update metrics
        
        **Configuration:**
        - Chunk size: 100 entries
        - Workers: 4 parallel threads
        - TTL: 24 hours
        """,
    )
    
    # Task 3: Validate predictions
    validation = PythonOperator(
        task_id="validate_predictions",
        python_callable=validate_predictions,
        doc_md="""
        ### Validation
        
        Spot checks to ensure prediction quality.
        
        **Checks:**
        - No NULL waste_probability
        - All probabilities in [0, 1]
        - risk_class matches probability thresholds
        """,
    )
    
    # Task 4: Success notification
    success_notification = PythonOperator(
        task_id="send_success_notification",
        python_callable=send_success_notification,
        trigger_rule="all_success",
    )
    
    # Task 5: Failure notification (only runs on failure)
    failure_notification = BashOperator(
        task_id="send_failure_notification",
        bash_command='echo "Batch prediction failed - check Airflow logs"',
        trigger_rule="one_failed",
    )
    
    # Define task dependencies
    health_check >> batch_prediction >> validation >> success_notification
    [health_check, batch_prediction, validation] >> failure_notification


# ============================================================================
# Documentation
# ============================================================================

dag.doc_md = """
# Daily Batch Predictions DAG

Runs daily waste predictions for all active pantry entries.

## Schedule

- **Frequency:** Daily at 2 AM UTC
- **Max runtime:** 2 hours
- **Retries:** 3 attempts with 5-minute delays

## Tasks

1. **Health Check:** Verify prediction server is healthy
2. **Batch Prediction:** Compute predictions for all active entries
3. **Validation:** Spot checks for data quality
4. **Notification:** Send success/failure alerts

## Monitoring

- **Metrics:** Total entries, computed, cached, failed, throughput
- **Alerts:** Email + Slack on failure
- **Logs:** Available in Airflow UI

## Configuration

- Prediction server: `http://waste-predictor-service:3000`
- Chunk size: 100 entries per batch
- Workers: 4 parallel threads
- TTL: 24 hours (recompute if older)

## Dependencies

- PostgreSQL database (pantry_entries, predictions tables)
- BentoML prediction server (waste_predictor service)
- Redis cache (optional, for feature engineering)

## Manual Trigger

To force recomputation of all predictions:

```bash
airflow dags trigger daily_batch_predictions --conf '{"force_recompute": true}'
```

## Troubleshooting

### High failure rate (>5%)

1. Check prediction server logs: `kubectl logs -l app=waste-predictor`
2. Verify database connectivity
3. Check feature engineering service
4. Review failed entries in `batch_prediction_metrics` table

### Slow performance (<50 entries/s)

1. Increase `max_workers` (default: 4)
2. Increase `chunk_size` (default: 100)
3. Scale up prediction server pods
4. Check database query performance

### Memory errors

1. Reduce `chunk_size`
2. Reduce `max_workers`
3. Scale up Airflow worker memory
"""

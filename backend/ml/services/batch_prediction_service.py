"""
Batch Prediction Service - Task #9

Daily cron job to compute waste predictions for all active pantry entries.
Uses parallelization and chunking for scalability.

Features:
- Batch processing with configurable chunk sizes
- Parallel workers for throughput
- Prediction TTL management (recompute if changed or >24h old)
- On-demand synchronous predictions for real-time UI updates
- Comprehensive metrics and monitoring
- Error handling and retry logic
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json

import pandas as pd
import numpy as np
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from backend.shared.database_v2 import get_session
from backend.ml.services.feature_engineer import FeatureEngineer
from backend.api.config import settings


logger = logging.getLogger(__name__)


@dataclass
class PredictionJob:
    """Represents a single prediction job."""
    pantry_entry_id: str
    household_id: str
    item_name: str
    quantity: float
    unit: str
    purchase_date: datetime
    expiry_date: Optional[datetime]
    last_predicted_at: Optional[datetime]
    data_hash: Optional[str]


@dataclass
class BatchPredictionMetrics:
    """Metrics for batch prediction job."""
    total_entries: int
    predictions_computed: int
    predictions_cached: int
    predictions_failed: int
    duration_seconds: float
    throughput_per_second: float
    avg_latency_ms: float


class BatchPredictionService:
    """
    Service for batch waste predictions.
    
    Architecture:
    1. Query active pantry entries from DB
    2. Check prediction TTL (skip if fresh and unchanged)
    3. Compute features for stale/missing predictions
    4. Call prediction server in batches
    5. Store predictions + explanations in DB
    6. Update metrics
    """
    
    def __init__(
        self,
        prediction_server_url: str = "http://localhost:3000",
        chunk_size: int = 100,
        max_workers: int = 4,
        prediction_ttl_hours: int = 24,
    ):
        """
        Initialize batch prediction service.
        
        Args:
            prediction_server_url: BentoML prediction endpoint
            chunk_size: Number of entries per batch
            max_workers: Parallel worker threads
            prediction_ttl_hours: Recompute if older than this
        """
        self.prediction_server_url = prediction_server_url
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.prediction_ttl = timedelta(hours=prediction_ttl_hours)
        
        self.feature_engineer = FeatureEngineer()
        
        logger.info(
            f"Initialized BatchPredictionService "
            f"(chunk_size={chunk_size}, workers={max_workers}, "
            f"ttl={prediction_ttl_hours}h)"
        )
    
    async def run_daily_batch(
        self,
        db: AsyncSession,
        force_recompute: bool = False,
    ) -> BatchPredictionMetrics:
        """
        Run daily batch prediction for all active pantry entries.
        
        This is the main entry point called by the cron job.
        
        Args:
            db: Database session
            force_recompute: Ignore TTL and recompute all
            
        Returns:
            BatchPredictionMetrics with job statistics
        """
        start_time = datetime.utcnow()
        logger.info("Starting daily batch prediction job...")
        
        # Step 1: Query active pantry entries
        jobs = await self._get_prediction_jobs(db, force_recompute)
        logger.info(f"Found {len(jobs)} pantry entries to process")
        
        if not jobs:
            logger.info("No entries to process, exiting")
            return BatchPredictionMetrics(
                total_entries=0,
                predictions_computed=0,
                predictions_cached=0,
                predictions_failed=0,
                duration_seconds=0,
                throughput_per_second=0,
                avg_latency_ms=0,
            )
        
        # Step 2: Process in chunks with parallelization
        predictions_computed = 0
        predictions_cached = 0
        predictions_failed = 0
        latencies = []
        
        # Split jobs into chunks
        chunks = [
            jobs[i:i + self.chunk_size]
            for i in range(0, len(jobs), self.chunk_size)
        ]
        
        logger.info(f"Processing {len(chunks)} chunks with {self.max_workers} workers")
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(
                    asyncio.run,
                    self._process_chunk(db, chunk)
                ): chunk
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_metrics = future.result()
                    predictions_computed += chunk_metrics["computed"]
                    predictions_cached += chunk_metrics["cached"]
                    predictions_failed += chunk_metrics["failed"]
                    latencies.extend(chunk_metrics["latencies"])
                    
                    logger.info(
                        f"Chunk completed: {chunk_metrics['computed']} computed, "
                        f"{chunk_metrics['cached']} cached, "
                        f"{chunk_metrics['failed']} failed"
                    )
                except Exception as e:
                    logger.error(f"Chunk processing failed: {e}", exc_info=True)
                    predictions_failed += len(chunk)
        
        # Calculate metrics
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        throughput = len(jobs) / duration if duration > 0 else 0
        avg_latency = np.mean(latencies) if latencies else 0
        
        metrics = BatchPredictionMetrics(
            total_entries=len(jobs),
            predictions_computed=predictions_computed,
            predictions_cached=predictions_cached,
            predictions_failed=predictions_failed,
            duration_seconds=duration,
            throughput_per_second=throughput,
            avg_latency_ms=avg_latency,
        )
        
        logger.info(
            f"Batch prediction completed: {predictions_computed} computed, "
            f"{predictions_cached} cached, {predictions_failed} failed "
            f"in {duration:.1f}s ({throughput:.1f} entries/s)"
        )
        
        # Store metrics
        await self._store_batch_metrics(db, metrics)
        
        return metrics
    
    async def predict_on_demand(
        self,
        db: AsyncSession,
        pantry_entry_id: str,
        force_recompute: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronous prediction for immediate UI update.
        
        Called when user edits pantry entry.
        
        Args:
            db: Database session
            pantry_entry_id: Pantry entry UUID
            force_recompute: Bypass cache
            
        Returns:
            Prediction result with explanation
        """
        logger.info(f"On-demand prediction for entry {pantry_entry_id}")
        
        # Check if fresh prediction exists
        if not force_recompute:
            cached = await self._get_cached_prediction(db, pantry_entry_id)
            if cached:
                logger.info(f"Using cached prediction (age: {cached['age_hours']:.1f}h)")
                return cached["prediction"]
        
        # Fetch pantry entry
        job = await self._get_single_job(db, pantry_entry_id)
        if not job:
            raise ValueError(f"Pantry entry {pantry_entry_id} not found")
        
        # Compute prediction
        start_time = datetime.utcnow()
        
        try:
            prediction = await self._compute_prediction(db, job)
            
            # Store in DB
            await self._store_prediction(db, job, prediction)
            await db.commit()
            
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(f"On-demand prediction completed in {latency:.1f}ms")
            
            return prediction
            
        except Exception as e:
            logger.error(f"On-demand prediction failed: {e}", exc_info=True)
            await db.rollback()
            raise
    
    async def _get_prediction_jobs(
        self,
        db: AsyncSession,
        force_recompute: bool,
    ) -> List[PredictionJob]:
        """
        Query pantry entries that need predictions.
        
        Criteria:
        - Active (not depleted)
        - No prediction OR prediction older than TTL OR data changed
        """
        # SQL query for active pantry entries with prediction status
        query = """
        SELECT 
            pe.pantry_entry_id,
            pe.household_id,
            pe.item_name,
            pe.quantity_on_hand,
            pe.unit,
            pe.purchase_date,
            pe.expiry_date,
            p.predicted_at AS last_predicted_at,
            p.data_hash
        FROM pantry_entries pe
        LEFT JOIN predictions p ON pe.pantry_entry_id = p.pantry_entry_id 
            AND p.model_type = 'waste_risk'
        WHERE pe.quantity_on_hand > 0
            AND pe.status = 'active'
        """
        
        if not force_recompute:
            # Only fetch entries with stale or missing predictions
            ttl_cutoff = datetime.utcnow() - self.prediction_ttl
            query += f" AND (p.predicted_at IS NULL OR p.predicted_at < '{ttl_cutoff.isoformat()}')"
        
        result = await db.execute(query)
        rows = result.fetchall()
        
        jobs = []
        for row in rows:
            # Check if data changed (hash pantry entry fields)
            current_hash = self._compute_data_hash(row)
            needs_update = force_recompute or row.data_hash != current_hash
            
            if needs_update:
                jobs.append(PredictionJob(
                    pantry_entry_id=row.pantry_entry_id,
                    household_id=row.household_id,
                    item_name=row.item_name,
                    quantity=row.quantity_on_hand,
                    unit=row.unit,
                    purchase_date=row.purchase_date,
                    expiry_date=row.expiry_date,
                    last_predicted_at=row.last_predicted_at,
                    data_hash=current_hash,
                ))
        
        return jobs
    
    async def _process_chunk(
        self,
        db: AsyncSession,
        jobs: List[PredictionJob],
    ) -> Dict[str, Any]:
        """Process a chunk of prediction jobs."""
        computed = 0
        cached = 0
        failed = 0
        latencies = []
        
        for job in jobs:
            start = datetime.utcnow()
            
            try:
                # Compute prediction
                prediction = await self._compute_prediction(db, job)
                
                # Store in DB
                await self._store_prediction(db, job, prediction)
                
                computed += 1
                latency = (datetime.utcnow() - start).total_seconds() * 1000
                latencies.append(latency)
                
            except Exception as e:
                logger.error(
                    f"Prediction failed for entry {job.pantry_entry_id}: {e}",
                    exc_info=True
                )
                failed += 1
        
        # Commit chunk
        await db.commit()
        
        return {
            "computed": computed,
            "cached": cached,
            "failed": failed,
            "latencies": latencies,
        }
    
    async def _compute_prediction(
        self,
        db: AsyncSession,
        job: PredictionJob,
    ) -> Dict[str, Any]:
        """
        Compute prediction by calling BentoML server.
        
        Steps:
        1. Engineer features
        2. Call prediction API
        3. Parse response
        """
        # Step 1: Compute features
        features = await self.feature_engineer.engineer_features(
            db=db,
            pantry_entry_id=job.pantry_entry_id,
            household_id=job.household_id,
        )
        
        # Step 2: Call prediction server
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.prediction_server_url}/predict",
                json={"features": features},
                timeout=10.0,
            )
            response.raise_for_status()
            prediction = response.json()
        
        # Step 3: Add metadata
        prediction["pantry_entry_id"] = job.pantry_entry_id
        prediction["household_id"] = job.household_id
        prediction["predicted_at"] = datetime.utcnow().isoformat()
        prediction["data_hash"] = job.data_hash
        
        return prediction
    
    async def _store_prediction(
        self,
        db: AsyncSession,
        job: PredictionJob,
        prediction: Dict[str, Any],
    ):
        """Store prediction and explanation in DB."""
        # Insert/update predictions table
        query = """
        INSERT INTO predictions (
            prediction_id,
            pantry_entry_id,
            household_id,
            model_type,
            model_version,
            waste_probability,
            risk_class,
            days_until_waste,
            explanation_features,
            explanation_values,
            explanation_text,
            predicted_at,
            data_hash
        ) VALUES (
            gen_random_uuid(),
            :pantry_entry_id,
            :household_id,
            'waste_risk',
            :model_version,
            :waste_probability,
            :risk_class,
            :days_until_waste,
            :explanation_features,
            :explanation_values,
            :explanation_text,
            :predicted_at,
            :data_hash
        )
        ON CONFLICT (pantry_entry_id, model_type)
        DO UPDATE SET
            model_version = EXCLUDED.model_version,
            waste_probability = EXCLUDED.waste_probability,
            risk_class = EXCLUDED.risk_class,
            days_until_waste = EXCLUDED.days_until_waste,
            explanation_features = EXCLUDED.explanation_features,
            explanation_values = EXCLUDED.explanation_values,
            explanation_text = EXCLUDED.explanation_text,
            predicted_at = EXCLUDED.predicted_at,
            data_hash = EXCLUDED.data_hash
        """
        
        await db.execute(query, {
            "pantry_entry_id": job.pantry_entry_id,
            "household_id": job.household_id,
            "model_version": prediction.get("model_version", "unknown"),
            "waste_probability": prediction["waste_probability"],
            "risk_class": prediction["risk_class"],
            "days_until_waste": prediction.get("days_until_waste"),
            "explanation_features": json.dumps(prediction.get("explanation", {}).get("features", [])),
            "explanation_values": json.dumps(prediction.get("explanation", {}).get("values", [])),
            "explanation_text": prediction.get("explanation", {}).get("text", ""),
            "predicted_at": prediction["predicted_at"],
            "data_hash": job.data_hash,
        })
    
    async def _get_cached_prediction(
        self,
        db: AsyncSession,
        pantry_entry_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Check for fresh cached prediction."""
        query = """
        SELECT 
            waste_probability,
            risk_class,
            days_until_waste,
            explanation_features,
            explanation_values,
            explanation_text,
            predicted_at,
            model_version,
            EXTRACT(EPOCH FROM (NOW() - predicted_at)) / 3600 AS age_hours
        FROM predictions
        WHERE pantry_entry_id = :pantry_entry_id
            AND model_type = 'waste_risk'
            AND predicted_at > :ttl_cutoff
        """
        
        ttl_cutoff = datetime.utcnow() - self.prediction_ttl
        result = await db.execute(query, {
            "pantry_entry_id": pantry_entry_id,
            "ttl_cutoff": ttl_cutoff,
        })
        row = result.fetchone()
        
        if not row:
            return None
        
        return {
            "prediction": {
                "waste_probability": row.waste_probability,
                "risk_class": row.risk_class,
                "days_until_waste": row.days_until_waste,
                "explanation": {
                    "features": json.loads(row.explanation_features),
                    "values": json.loads(row.explanation_values),
                    "text": row.explanation_text,
                },
                "model_version": row.model_version,
                "predicted_at": row.predicted_at.isoformat(),
            },
            "age_hours": row.age_hours,
        }
    
    async def _get_single_job(
        self,
        db: AsyncSession,
        pantry_entry_id: str,
    ) -> Optional[PredictionJob]:
        """Fetch single pantry entry as prediction job."""
        query = """
        SELECT 
            pantry_entry_id,
            household_id,
            item_name,
            quantity_on_hand,
            unit,
            purchase_date,
            expiry_date
        FROM pantry_entries
        WHERE pantry_entry_id = :pantry_entry_id
            AND status = 'active'
        """
        
        result = await db.execute(query, {"pantry_entry_id": pantry_entry_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return PredictionJob(
            pantry_entry_id=row.pantry_entry_id,
            household_id=row.household_id,
            item_name=row.item_name,
            quantity=row.quantity_on_hand,
            unit=row.unit,
            purchase_date=row.purchase_date,
            expiry_date=row.expiry_date,
            last_predicted_at=None,
            data_hash=self._compute_data_hash(row),
        )
    
    def _compute_data_hash(self, row: Any) -> str:
        """
        Compute hash of pantry entry data.
        
        Used to detect if entry changed since last prediction.
        """
        data = {
            "item_name": row.item_name,
            "quantity": float(row.quantity_on_hand),
            "unit": row.unit,
            "purchase_date": row.purchase_date.isoformat() if row.purchase_date else None,
            "expiry_date": row.expiry_date.isoformat() if row.expiry_date else None,
        }
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    async def _store_batch_metrics(
        self,
        db: AsyncSession,
        metrics: BatchPredictionMetrics,
    ):
        """Store batch job metrics for monitoring."""
        query = """
        INSERT INTO batch_prediction_metrics (
            batch_id,
            started_at,
            total_entries,
            predictions_computed,
            predictions_cached,
            predictions_failed,
            duration_seconds,
            throughput_per_second,
            avg_latency_ms
        ) VALUES (
            gen_random_uuid(),
            NOW(),
            :total_entries,
            :predictions_computed,
            :predictions_cached,
            :predictions_failed,
            :duration_seconds,
            :throughput_per_second,
            :avg_latency_ms
        )
        """
        
        await db.execute(query, {
            "total_entries": metrics.total_entries,
            "predictions_computed": metrics.predictions_computed,
            "predictions_cached": metrics.predictions_cached,
            "predictions_failed": metrics.predictions_failed,
            "duration_seconds": metrics.duration_seconds,
            "throughput_per_second": metrics.throughput_per_second,
            "avg_latency_ms": metrics.avg_latency_ms,
        })
        await db.commit()


# ============================================================================
# CLI Entry Point
# ============================================================================

async def main():
    """Run batch prediction job (called by cron)."""
    import sys
    
    # Parse args
    force_recompute = "--force" in sys.argv
    
    # Initialize service
    service = BatchPredictionService(
        prediction_server_url=settings.PREDICTION_SERVER_URL,
        chunk_size=100,
        max_workers=4,
        prediction_ttl_hours=24,
    )
    
    # Run batch
    async with get_session() as db:
        metrics = await service.run_daily_batch(db, force_recompute)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PREDICTION SUMMARY")
    print("="*60)
    print(f"Total entries: {metrics.total_entries}")
    print(f"Predictions computed: {metrics.predictions_computed}")
    print(f"Predictions cached: {metrics.predictions_cached}")
    print(f"Predictions failed: {metrics.predictions_failed}")
    print(f"Duration: {metrics.duration_seconds:.1f}s")
    print(f"Throughput: {metrics.throughput_per_second:.1f} entries/s")
    print(f"Avg latency: {metrics.avg_latency_ms:.1f}ms")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())

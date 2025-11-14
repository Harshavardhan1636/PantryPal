"""
Feedback Loop Service - Task #10

Collects user feedback on predictions and uses it to improve model quality.

Feedback types:
- "used": User consumed the item (not wasted)
- "wasted": User threw away the item
- "inaccurate": Prediction was wrong

Features:
- Store feedback with prediction metadata
- Label training data based on feedback
- Trigger model retraining when threshold reached
- Track feedback-prediction agreement
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd


logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """User feedback types."""
    USED = "used"  # Item consumed (not wasted)
    WASTED = "wasted"  # Item thrown away
    INACCURATE = "inaccurate"  # Prediction was wrong
    HELPFUL = "helpful"  # Prediction was helpful
    NOT_HELPFUL = "not_helpful"  # Prediction was not helpful


@dataclass
class FeedbackMetrics:
    """Metrics for feedback collection."""
    total_feedback: int
    feedback_by_type: Dict[str, int]
    agreement_rate: float  # % of feedback matching prediction
    high_risk_precision: float  # % of HIGH predictions that were wasted
    low_risk_recall: float  # % of not-wasted items predicted as LOW
    avg_days_since_prediction: float


class FeedbackCollector:
    """
    Service for collecting and processing user feedback.
    
    Workflow:
    1. User provides feedback on prediction via UI
    2. Store feedback in DB with prediction context
    3. Compute agreement metrics
    4. Label training data for retraining
    5. Trigger retraining when threshold reached
    """
    
    def __init__(
        self,
        retraining_threshold: int = 1000,  # Retrain after N new labels
    ):
        """
        Initialize feedback collector.
        
        Args:
            retraining_threshold: Number of feedback items before triggering retrain
        """
        self.retraining_threshold = retraining_threshold
        logger.info(f"Initialized FeedbackCollector (threshold={retraining_threshold})")
    
    async def record_feedback(
        self,
        db: AsyncSession,
        pantry_entry_id: str,
        household_id: str,
        feedback_type: FeedbackType,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record user feedback on a prediction.
        
        Args:
            db: Database session
            pantry_entry_id: Pantry entry UUID
            household_id: Household UUID
            feedback_type: Type of feedback
            notes: Optional user notes
            
        Returns:
            Feedback record with metadata
        """
        logger.info(
            f"Recording feedback: {feedback_type} for entry {pantry_entry_id}"
        )
        
        # Fetch latest prediction for this entry
        prediction = await self._get_latest_prediction(db, pantry_entry_id)
        
        if not prediction:
            raise ValueError(f"No prediction found for entry {pantry_entry_id}")
        
        # Compute agreement (did feedback match prediction?)
        agreement = self._compute_agreement(
            feedback_type=feedback_type,
            predicted_risk=prediction["risk_class"],
            predicted_probability=prediction["waste_probability"],
        )
        
        # Insert feedback record
        query = """
        INSERT INTO prediction_feedback (
            feedback_id,
            pantry_entry_id,
            household_id,
            prediction_id,
            feedback_type,
            predicted_risk_class,
            predicted_probability,
            agreement,
            notes,
            created_at
        ) VALUES (
            gen_random_uuid(),
            :pantry_entry_id,
            :household_id,
            :prediction_id,
            :feedback_type,
            :predicted_risk_class,
            :predicted_probability,
            :agreement,
            :notes,
            NOW()
        )
        RETURNING feedback_id
        """
        
        result = await db.execute(query, {
            "pantry_entry_id": pantry_entry_id,
            "household_id": household_id,
            "prediction_id": prediction["prediction_id"],
            "feedback_type": feedback_type.value,
            "predicted_risk_class": prediction["risk_class"],
            "predicted_probability": prediction["waste_probability"],
            "agreement": agreement,
            "notes": notes,
        })
        
        feedback_id = result.scalar()
        await db.commit()
        
        logger.info(f"Feedback recorded: {feedback_id} (agreement={agreement})")
        
        # Label training data
        await self._label_training_data(db, pantry_entry_id, feedback_type)
        
        # Check if retraining needed
        await self._check_retraining_needed(db)
        
        return {
            "feedback_id": feedback_id,
            "pantry_entry_id": pantry_entry_id,
            "feedback_type": feedback_type.value,
            "agreement": agreement,
            "prediction": prediction,
        }
    
    async def get_feedback_metrics(
        self,
        db: AsyncSession,
        household_id: Optional[str] = None,
        days: int = 30,
    ) -> FeedbackMetrics:
        """
        Compute feedback metrics for monitoring.
        
        Args:
            db: Database session
            household_id: Optional household filter
            days: Number of days to look back
            
        Returns:
            FeedbackMetrics with statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Base query
        where_clause = "WHERE f.created_at >= :cutoff"
        params = {"cutoff": cutoff}
        
        if household_id:
            where_clause += " AND f.household_id = :household_id"
            params["household_id"] = household_id
        
        # Total feedback count
        query = f"""
        SELECT COUNT(*) 
        FROM prediction_feedback f
        {where_clause}
        """
        result = await db.execute(query, params)
        total_feedback = result.scalar()
        
        # Feedback by type
        query = f"""
        SELECT feedback_type, COUNT(*) as count
        FROM prediction_feedback f
        {where_clause}
        GROUP BY feedback_type
        """
        result = await db.execute(query, params)
        feedback_by_type = {row.feedback_type: row.count for row in result}
        
        # Agreement rate (% feedback matching prediction)
        query = f"""
        SELECT AVG(CASE WHEN agreement THEN 1.0 ELSE 0.0 END) as agreement_rate
        FROM prediction_feedback f
        {where_clause}
        """
        result = await db.execute(query, params)
        agreement_rate = result.scalar() or 0.0
        
        # High risk precision (% HIGH predictions that were actually wasted)
        query = f"""
        SELECT 
            COUNT(CASE WHEN feedback_type = 'wasted' THEN 1 END)::float / 
            NULLIF(COUNT(*), 0) as precision
        FROM prediction_feedback f
        {where_clause}
            AND predicted_risk_class = 'HIGH'
            AND feedback_type IN ('used', 'wasted')
        """
        result = await db.execute(query, params)
        high_risk_precision = result.scalar() or 0.0
        
        # Low risk recall (% not-wasted items predicted as LOW)
        query = f"""
        SELECT 
            COUNT(CASE WHEN predicted_risk_class = 'LOW' THEN 1 END)::float / 
            NULLIF(COUNT(*), 0) as recall
        FROM prediction_feedback f
        {where_clause}
            AND feedback_type = 'used'
        """
        result = await db.execute(query, params)
        low_risk_recall = result.scalar() or 0.0
        
        # Average days between prediction and feedback
        query = f"""
        SELECT AVG(EXTRACT(EPOCH FROM (f.created_at - p.predicted_at)) / 86400) as avg_days
        FROM prediction_feedback f
        JOIN predictions p ON f.prediction_id = p.prediction_id
        {where_clause}
        """
        result = await db.execute(query, params)
        avg_days_since_prediction = result.scalar() or 0.0
        
        return FeedbackMetrics(
            total_feedback=total_feedback,
            feedback_by_type=feedback_by_type,
            agreement_rate=agreement_rate,
            high_risk_precision=high_risk_precision,
            low_risk_recall=low_risk_recall,
            avg_days_since_prediction=avg_days_since_prediction,
        )
    
    async def export_labeled_data(
        self,
        db: AsyncSession,
        output_path: str,
        min_days_ago: int = 7,  # Only export feedback older than N days
    ) -> int:
        """
        Export feedback-labeled data for model retraining.
        
        Args:
            db: Database session
            output_path: Path to save CSV
            min_days_ago: Only include feedback older than N days
            
        Returns:
            Number of labeled examples exported
        """
        cutoff = datetime.utcnow() - timedelta(days=min_days_ago)
        
        # Query feedback with features
        query = """
        SELECT 
            f.feedback_id,
            f.pantry_entry_id,
            f.household_id,
            f.feedback_type,
            f.predicted_probability,
            f.created_at as feedback_date,
            p.predicted_at,
            pe.item_name,
            pe.category,
            pe.quantity_on_hand,
            pe.purchase_date,
            pe.expiry_date,
            -- Add features here (join with feature tables)
            CASE 
                WHEN f.feedback_type = 'wasted' THEN 1
                WHEN f.feedback_type = 'used' THEN 0
                ELSE NULL
            END as ground_truth_label
        FROM prediction_feedback f
        JOIN predictions p ON f.prediction_id = p.prediction_id
        JOIN pantry_entries pe ON f.pantry_entry_id = pe.pantry_entry_id
        WHERE f.created_at < :cutoff
            AND f.feedback_type IN ('used', 'wasted')
        ORDER BY f.created_at DESC
        """
        
        result = await db.execute(query, {"cutoff": cutoff})
        rows = result.fetchall()
        
        if not rows:
            logger.info("No labeled data to export")
            return 0
        
        # Convert to DataFrame
        df = pd.DataFrame([dict(row) for row in rows])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(df)} labeled examples to {output_path}")
        
        return len(df)
    
    async def _get_latest_prediction(
        self,
        db: AsyncSession,
        pantry_entry_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch latest prediction for pantry entry."""
        query = """
        SELECT 
            prediction_id,
            waste_probability,
            risk_class,
            days_until_waste,
            predicted_at
        FROM predictions
        WHERE pantry_entry_id = :pantry_entry_id
            AND model_type = 'waste_risk'
        ORDER BY predicted_at DESC
        LIMIT 1
        """
        
        result = await db.execute(query, {"pantry_entry_id": pantry_entry_id})
        row = result.fetchone()
        
        if not row:
            return None
        
        return {
            "prediction_id": row.prediction_id,
            "waste_probability": row.waste_probability,
            "risk_class": row.risk_class,
            "days_until_waste": row.days_until_waste,
            "predicted_at": row.predicted_at,
        }
    
    def _compute_agreement(
        self,
        feedback_type: FeedbackType,
        predicted_risk: str,
        predicted_probability: float,
    ) -> bool:
        """
        Compute if feedback agrees with prediction.
        
        Rules:
        - "wasted" agrees with HIGH risk
        - "used" agrees with LOW risk
        - "inaccurate" always disagrees
        """
        if feedback_type == FeedbackType.INACCURATE:
            return False
        
        if feedback_type == FeedbackType.WASTED:
            return predicted_risk == "HIGH" or predicted_probability >= 0.5
        
        if feedback_type == FeedbackType.USED:
            return predicted_risk == "LOW" or predicted_probability < 0.3
        
        return True  # HELPFUL/NOT_HELPFUL always agree
    
    async def _label_training_data(
        self,
        db: AsyncSession,
        pantry_entry_id: str,
        feedback_type: FeedbackType,
    ):
        """
        Update training data labels based on feedback.
        
        This marks the pantry entry for inclusion in next training set.
        """
        if feedback_type not in [FeedbackType.USED, FeedbackType.WASTED]:
            return  # Only label binary outcomes
        
        label = 1 if feedback_type == FeedbackType.WASTED else 0
        
        query = """
        INSERT INTO training_labels (
            pantry_entry_id,
            ground_truth_label,
            labeled_at,
            label_source
        ) VALUES (
            :pantry_entry_id,
            :label,
            NOW(),
            'user_feedback'
        )
        ON CONFLICT (pantry_entry_id) 
        DO UPDATE SET
            ground_truth_label = EXCLUDED.ground_truth_label,
            labeled_at = EXCLUDED.labeled_at
        """
        
        await db.execute(query, {
            "pantry_entry_id": pantry_entry_id,
            "label": label,
        })
        await db.commit()
        
        logger.info(
            f"Labeled pantry_entry {pantry_entry_id} as "
            f"{'wasted' if label == 1 else 'not wasted'}"
        )
    
    async def _check_retraining_needed(self, db: AsyncSession):
        """
        Check if enough new labels collected to trigger retraining.
        
        Triggers retraining job if threshold exceeded.
        """
        # Count new labels since last training
        query = """
        SELECT COUNT(*) 
        FROM training_labels
        WHERE labeled_at > (
            SELECT MAX(trained_at) 
            FROM model_training_runs 
            WHERE model_type = 'waste_risk'
        )
        """
        
        result = await db.execute(query)
        new_labels = result.scalar()
        
        if new_labels >= self.retraining_threshold:
            logger.info(
                f"Retraining threshold reached: {new_labels} new labels "
                f"(threshold: {self.retraining_threshold})"
            )
            
            # Trigger retraining (in production, publish to queue)
            await self._trigger_retraining(db)
    
    async def _trigger_retraining(self, db: AsyncSession):
        """
        Trigger model retraining job.
        
        In production:
        - Publish message to training queue
        - Trigger Airflow DAG
        - Send notification to ML team
        """
        logger.info("Triggering model retraining...")
        
        # Insert retraining job
        query = """
        INSERT INTO model_training_jobs (
            job_id,
            model_type,
            trigger_reason,
            status,
            created_at
        ) VALUES (
            gen_random_uuid(),
            'waste_risk',
            'feedback_threshold',
            'pending',
            NOW()
        )
        """
        
        await db.execute(query)
        await db.commit()
        
        logger.info("Model retraining job created")
        
        # TODO: In production, also:
        # - Publish to Pub/Sub: training-jobs topic
        # - Trigger Airflow DAG: model_retraining_waste_risk
        # - Send Slack notification to #ml-alerts


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    from backend.shared.database_v2 import get_session
    
    async def demo():
        collector = FeedbackCollector(retraining_threshold=100)
        
        async with get_session() as db:
            # Record feedback
            feedback = await collector.record_feedback(
                db=db,
                pantry_entry_id="123e4567-e89b-12d3-a456-426614174000",
                household_id="456e4567-e89b-12d3-a456-426614174001",
                feedback_type=FeedbackType.WASTED,
                notes="Item expired before I could use it",
            )
            
            print("Feedback recorded:")
            print(f"  ID: {feedback['feedback_id']}")
            print(f"  Type: {feedback['feedback_type']}")
            print(f"  Agreement: {feedback['agreement']}")
            print()
            
            # Get metrics
            metrics = await collector.get_feedback_metrics(db, days=30)
            
            print("Feedback metrics (last 30 days):")
            print(f"  Total feedback: {metrics.total_feedback}")
            print(f"  Agreement rate: {metrics.agreement_rate:.1%}")
            print(f"  High-risk precision: {metrics.high_risk_precision:.1%}")
            print(f"  Low-risk recall: {metrics.low_risk_recall:.1%}")
            print(f"  Feedback by type:")
            for ftype, count in metrics.feedback_by_type.items():
                print(f"    {ftype}: {count}")
    
    asyncio.run(demo())

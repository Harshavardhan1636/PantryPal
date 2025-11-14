"""
Data Labeling UI - Admin Panel

Admin interface for reviewing and correcting low-confidence predictions.
Supports:
- Review pending corrections
- Accept/reject predictions
- Manual correction entry
- Label quality tracking
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from ..database import get_db
from ..auth.dependencies import get_current_admin_user
from ..models import (
    PantryItem,
    WastePrediction,
    UserCorrection,
    LabelQualityMetric,
    User
)


router = APIRouter(prefix="/api/v1/admin/labeling", tags=["admin", "labeling"])


# ============================================================================
# Request/Response Models
# ============================================================================

class PendingLabelResponse(BaseModel):
    """Low-confidence prediction needing review."""
    id: int
    pantry_item_id: int
    pantry_item_name: str
    current_prediction: dict
    confidence_score: float
    created_at: datetime
    household_id: int
    household_name: str
    
    class Config:
        from_attributes = True


class CorrectionRequest(BaseModel):
    """Admin correction for prediction."""
    prediction_id: int
    corrected_waste_risk_score: Optional[float] = Field(None, ge=0, le=1)
    corrected_days_until_waste: Optional[int] = Field(None, ge=0)
    corrected_waste_reason: Optional[str] = None
    correction_notes: Optional[str] = None
    action: str = Field(..., pattern="^(accept|reject|correct)$")


class LabelQualityResponse(BaseModel):
    """Label quality metrics."""
    total_predictions: int
    low_confidence_count: int
    pending_review_count: int
    corrected_count: int
    accepted_count: int
    rejected_count: int
    avg_confidence: float
    correction_rate: float
    
    class Config:
        from_attributes = True


class WasteReasonStats(BaseModel):
    """Waste reason distribution."""
    spoilage: int
    overcooked: int
    portion: int
    packaging: int
    other: int
    unknown: int


# ============================================================================
# Get Pending Labels
# ============================================================================

@router.get("/pending", response_model=List[PendingLabelResponse])
async def get_pending_labels(
    confidence_threshold: float = Query(0.7, ge=0, le=1),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("confidence_asc", pattern="^(confidence_asc|confidence_desc|date_asc|date_desc)$"),
    household_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Get predictions with low confidence needing review.
    
    Args:
        confidence_threshold: Show predictions below this threshold (default 0.7)
        limit: Max results to return
        offset: Pagination offset
        sort_by: Sort order (confidence_asc, confidence_desc, date_asc, date_desc)
        household_id: Filter by specific household
    
    Returns:
        List of pending labels needing review
    """
    
    # Query for low-confidence predictions without corrections
    query = db.query(
        WastePrediction,
        PantryItem,
    ).join(
        PantryItem,
        PantryItem.id == WastePrediction.pantry_item_id
    ).outerjoin(
        UserCorrection,
        UserCorrection.prediction_id == WastePrediction.id
    ).filter(
        and_(
            WastePrediction.confidence_score < confidence_threshold,
            UserCorrection.id.is_(None)  # Not yet reviewed
        )
    )
    
    # Filter by household if specified
    if household_id:
        query = query.filter(PantryItem.household_id == household_id)
    
    # Sort
    if sort_by == "confidence_asc":
        query = query.order_by(WastePrediction.confidence_score.asc())
    elif sort_by == "confidence_desc":
        query = query.order_by(WastePrediction.confidence_score.desc())
    elif sort_by == "date_asc":
        query = query.order_by(WastePrediction.created_at.asc())
    else:  # date_desc
        query = query.order_by(WastePrediction.created_at.desc())
    
    # Paginate
    results = query.limit(limit).offset(offset).all()
    
    # Format response
    pending_labels = []
    for prediction, pantry_item in results:
        pending_labels.append(PendingLabelResponse(
            id=prediction.id,
            pantry_item_id=pantry_item.id,
            pantry_item_name=pantry_item.name,
            current_prediction={
                "waste_risk_score": prediction.waste_risk_score,
                "days_until_waste": prediction.days_until_waste,
                "waste_reason": prediction.waste_reason,
                "model_version": prediction.model_version,
            },
            confidence_score=prediction.confidence_score,
            created_at=prediction.created_at,
            household_id=pantry_item.household_id,
            household_name=pantry_item.household.name,
        ))
    
    return pending_labels


# ============================================================================
# Submit Correction
# ============================================================================

@router.post("/corrections")
async def submit_correction(
    correction: CorrectionRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Submit admin correction for a prediction.
    
    Actions:
    - accept: Mark prediction as correct (ground truth)
    - reject: Mark prediction as incorrect
    - correct: Provide corrected values
    
    Args:
        correction: Correction details
    
    Returns:
        Success message
    """
    
    # Get prediction
    prediction = db.query(WastePrediction).filter(
        WastePrediction.id == correction.prediction_id
    ).first()
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Check if already corrected
    existing_correction = db.query(UserCorrection).filter(
        UserCorrection.prediction_id == correction.prediction_id
    ).first()
    
    if existing_correction:
        raise HTTPException(status_code=400, detail="Prediction already corrected")
    
    # Create correction record
    user_correction = UserCorrection(
        prediction_id=correction.prediction_id,
        user_id=current_admin.id,
        action=correction.action,
        corrected_waste_risk_score=correction.corrected_waste_risk_score,
        corrected_days_until_waste=correction.corrected_days_until_waste,
        corrected_waste_reason=correction.corrected_waste_reason,
        correction_notes=correction.correction_notes,
        original_prediction={
            "waste_risk_score": prediction.waste_risk_score,
            "days_until_waste": prediction.days_until_waste,
            "waste_reason": prediction.waste_reason,
            "confidence_score": prediction.confidence_score,
        },
        created_at=datetime.utcnow(),
    )
    
    db.add(user_correction)
    
    # If accepted, mark as ground truth
    if correction.action == "accept":
        prediction.is_ground_truth = True
        prediction.ground_truth_verified_at = datetime.utcnow()
        prediction.ground_truth_verified_by = current_admin.id
    
    # If corrected, update prediction with corrected values
    if correction.action == "correct":
        if correction.corrected_waste_risk_score is not None:
            prediction.waste_risk_score = correction.corrected_waste_risk_score
        
        if correction.corrected_days_until_waste is not None:
            prediction.days_until_waste = correction.corrected_days_until_waste
        
        if correction.corrected_waste_reason is not None:
            prediction.waste_reason = correction.corrected_waste_reason
        
        # Mark as ground truth after correction
        prediction.is_ground_truth = True
        prediction.ground_truth_verified_at = datetime.utcnow()
        prediction.ground_truth_verified_by = current_admin.id
    
    db.commit()
    
    return {
        "message": "Correction submitted successfully",
        "correction_id": user_correction.id,
        "action": correction.action,
    }


# ============================================================================
# Bulk Actions
# ============================================================================

@router.post("/corrections/bulk-accept")
async def bulk_accept_predictions(
    prediction_ids: List[int],
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Bulk accept multiple predictions as correct.
    
    Args:
        prediction_ids: List of prediction IDs to accept
    
    Returns:
        Count of accepted predictions
    """
    
    accepted_count = 0
    
    for pred_id in prediction_ids:
        prediction = db.query(WastePrediction).filter(
            WastePrediction.id == pred_id
        ).first()
        
        if not prediction:
            continue
        
        # Check if already corrected
        existing = db.query(UserCorrection).filter(
            UserCorrection.prediction_id == pred_id
        ).first()
        
        if existing:
            continue
        
        # Create correction
        correction = UserCorrection(
            prediction_id=pred_id,
            user_id=current_admin.id,
            action="accept",
            original_prediction={
                "waste_risk_score": prediction.waste_risk_score,
                "days_until_waste": prediction.days_until_waste,
                "waste_reason": prediction.waste_reason,
            },
            created_at=datetime.utcnow(),
        )
        
        db.add(correction)
        
        # Mark as ground truth
        prediction.is_ground_truth = True
        prediction.ground_truth_verified_at = datetime.utcnow()
        prediction.ground_truth_verified_by = current_admin.id
        
        accepted_count += 1
    
    db.commit()
    
    return {
        "message": f"Accepted {accepted_count} predictions",
        "accepted_count": accepted_count,
    }


# ============================================================================
# Label Quality Metrics
# ============================================================================

@router.get("/quality-metrics", response_model=LabelQualityResponse)
async def get_label_quality_metrics(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Get label quality metrics for the last N days.
    
    Args:
        days: Number of days to analyze
    
    Returns:
        Label quality statistics
    """
    
    since = datetime.utcnow() - timedelta(days=days)
    
    # Total predictions
    total_predictions = db.query(WastePrediction).filter(
        WastePrediction.created_at >= since
    ).count()
    
    # Low confidence predictions
    low_confidence_count = db.query(WastePrediction).filter(
        and_(
            WastePrediction.created_at >= since,
            WastePrediction.confidence_score < 0.7
        )
    ).count()
    
    # Pending review (low confidence without corrections)
    pending_review_count = db.query(WastePrediction).outerjoin(
        UserCorrection,
        UserCorrection.prediction_id == WastePrediction.id
    ).filter(
        and_(
            WastePrediction.created_at >= since,
            WastePrediction.confidence_score < 0.7,
            UserCorrection.id.is_(None)
        )
    ).count()
    
    # Corrected predictions
    corrected_count = db.query(UserCorrection).filter(
        and_(
            UserCorrection.created_at >= since,
            UserCorrection.action == "correct"
        )
    ).count()
    
    # Accepted predictions
    accepted_count = db.query(UserCorrection).filter(
        and_(
            UserCorrection.created_at >= since,
            UserCorrection.action == "accept"
        )
    ).count()
    
    # Rejected predictions
    rejected_count = db.query(UserCorrection).filter(
        and_(
            UserCorrection.created_at >= since,
            UserCorrection.action == "reject"
        )
    ).count()
    
    # Average confidence
    from sqlalchemy import func
    avg_confidence_result = db.query(
        func.avg(WastePrediction.confidence_score)
    ).filter(
        WastePrediction.created_at >= since
    ).scalar()
    
    avg_confidence = float(avg_confidence_result) if avg_confidence_result else 0.0
    
    # Correction rate
    correction_rate = (corrected_count + rejected_count) / total_predictions if total_predictions > 0 else 0.0
    
    return LabelQualityResponse(
        total_predictions=total_predictions,
        low_confidence_count=low_confidence_count,
        pending_review_count=pending_review_count,
        corrected_count=corrected_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        avg_confidence=avg_confidence,
        correction_rate=correction_rate,
    )


# ============================================================================
# Waste Reason Statistics
# ============================================================================

@router.get("/waste-reasons", response_model=WasteReasonStats)
async def get_waste_reason_statistics(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user)
):
    """
    Get distribution of waste reasons.
    
    Args:
        days: Number of days to analyze
    
    Returns:
        Waste reason statistics
    """
    
    since = datetime.utcnow() - timedelta(days=days)
    
    # Query waste reasons from ground truth predictions
    from sqlalchemy import func
    
    waste_reasons = db.query(
        WastePrediction.waste_reason,
        func.count(WastePrediction.id).label('count')
    ).filter(
        and_(
            WastePrediction.created_at >= since,
            WastePrediction.is_ground_truth == True
        )
    ).group_by(WastePrediction.waste_reason).all()
    
    # Build stats
    stats = WasteReasonStats(
        spoilage=0,
        overcooked=0,
        portion=0,
        packaging=0,
        other=0,
        unknown=0,
    )
    
    for reason, count in waste_reasons:
        if reason == "spoilage":
            stats.spoilage = count
        elif reason == "overcooked":
            stats.overcooked = count
        elif reason == "portion":
            stats.portion = count
        elif reason == "packaging":
            stats.packaging = count
        elif reason == "other":
            stats.other = count
        else:
            stats.unknown = count
    
    return stats

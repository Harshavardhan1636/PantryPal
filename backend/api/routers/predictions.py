"""
Predictions Router
Handles ML-powered waste risk predictions and recommendations.
"""

import logging
from uuid import UUID
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.dependencies import get_current_active_user, pagination_params
from backend.api.schemas import PredictionResponse, PredictionListResponse
from backend.api.config import get_settings
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import (
    User,
    Prediction,
    PantryEntry,
    HouseholdUser,
    RiskClass,
)

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{household_id}",
    response_model=PredictionListResponse,
    summary="Get waste risk predictions",
    description="Get ML-powered waste risk predictions for household with recommended actions",
)
async def get_predictions(
    household_id: UUID,
    top: int = Query(10, ge=1, le=100, description="Number of top risk items to return"),
    risk_class: RiskClass | None = Query(None, description="Filter by risk class"),
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> PredictionListResponse:
    """Get waste risk predictions for household."""
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == household_id,
            HouseholdUser.user_id == current_user.id,
            HouseholdUser.deleted_at.is_(None),
        )
    )
    membership = result.scalar_one_or_none()
    
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Household not found or access denied",
        )
    
    # Build query for latest predictions
    query = (
        select(Prediction)
        .join(PantryEntry)
        .where(
            Prediction.household_id == household_id,
            Prediction.deleted_at.is_(None),
            PantryEntry.deleted_at.is_(None),
            PantryEntry.quantity > 0,  # Only active items
        )
    )
    
    if risk_class:
        query = query.where(Prediction.risk_class == risk_class)
    
    # Order by risk score descending (highest risk first)
    query = query.order_by(Prediction.risk_score.desc()).limit(top)
    
    result = await session.execute(query)
    predictions = result.scalars().all()
    
    # Get counts by risk class
    count_query = (
        select(
            Prediction.risk_class,
            func.count(Prediction.id).label("count"),
        )
        .join(PantryEntry)
        .where(
            Prediction.household_id == household_id,
            Prediction.deleted_at.is_(None),
            PantryEntry.deleted_at.is_(None),
            PantryEntry.quantity > 0,
        )
        .group_by(Prediction.risk_class)
    )
    
    result = await session.execute(count_query)
    counts = {row.risk_class: row.count for row in result}
    
    return PredictionListResponse(
        predictions=list(predictions),
        total=len(predictions),
        high_risk_count=counts.get(RiskClass.HIGH, 0),
        medium_risk_count=counts.get(RiskClass.MEDIUM, 0),
        low_risk_count=counts.get(RiskClass.LOW, 0),
    )


@router.post(
    "/generate/{household_id}",
    response_model=PredictionListResponse,
    summary="Generate new predictions",
    description="Trigger ML model to generate fresh waste risk predictions",
)
async def generate_predictions(
    household_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> PredictionListResponse:
    """Generate fresh ML predictions for household."""
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == household_id,
            HouseholdUser.user_id == current_user.id,
            HouseholdUser.deleted_at.is_(None),
        )
    )
    membership = result.scalar_one_or_none()
    
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Household not found or access denied",
        )
    
    if not settings.FEATURE_ML_PREDICTIONS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML predictions are currently disabled",
        )
    
    # Get active pantry entries
    result = await session.execute(
        select(PantryEntry)
        .where(
            PantryEntry.household_id == household_id,
            PantryEntry.deleted_at.is_(None),
            PantryEntry.quantity > 0,
        )
    )
    
    pantry_entries = result.scalars().all()
    
    if not pantry_entries:
        return PredictionListResponse(
            predictions=[],
            total=0,
            high_risk_count=0,
            medium_risk_count=0,
            low_risk_count=0,
        )
    
    # Call ML prediction service (implement actual service call)
    from backend.api.services.ml_service import generate_waste_predictions
    
    try:
        predictions = await generate_waste_predictions(session, list(pantry_entries))
        
        # Save predictions to database
        for pred in predictions:
            session.add(pred)
        
        await session.commit()
        
        # Get counts
        counts = {
            RiskClass.HIGH: sum(1 for p in predictions if p.risk_class == RiskClass.HIGH),
            RiskClass.MEDIUM: sum(1 for p in predictions if p.risk_class == RiskClass.MEDIUM),
            RiskClass.LOW: sum(1 for p in predictions if p.risk_class == RiskClass.LOW),
        }
        
        logger.info(
            f"Generated {len(predictions)} predictions for household {household_id}"
        )
        
        return PredictionListResponse(
            predictions=list(predictions),
            total=len(predictions),
            high_risk_count=counts[RiskClass.HIGH],
            medium_risk_count=counts[RiskClass.MEDIUM],
            low_risk_count=counts[RiskClass.LOW],
        )
        
    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate predictions. Please try again later.",
        )


@router.get(
    "/item/{pantry_entry_id}",
    response_model=PredictionResponse,
    summary="Get item prediction",
    description="Get waste risk prediction for specific pantry item",
)
async def get_item_prediction(
    pantry_entry_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> Prediction:
    """Get prediction for specific pantry item."""
    # Get latest prediction for item
    result = await session.execute(
        select(Prediction)
        .where(
            Prediction.pantry_entry_id == pantry_entry_id,
            Prediction.deleted_at.is_(None),
        )
        .order_by(Prediction.created_at.desc())
        .limit(1)
    )
    
    prediction = result.scalar_one_or_none()
    
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No prediction found for this item",
        )
    
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == prediction.household_id,
            HouseholdUser.user_id == current_user.id,
            HouseholdUser.deleted_at.is_(None),
        )
    )
    membership = result.scalar_one_or_none()
    
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    
    return prediction

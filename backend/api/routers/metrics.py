"""
Metrics Router
Handles household analytics and metrics reporting.
"""

import logging
from uuid import UUID
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_active_user
from backend.api.schemas import HouseholdMetricsResponse
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import (
    User,
    HouseholdUser,
    WasteEvent,
    ConsumptionLog,
    PantryEntry,
    Prediction,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{household_id}/metrics",
    response_model=HouseholdMetricsResponse,
    summary="Get household metrics",
    description="Get comprehensive household metrics including waste, savings, and engagement",
)
async def get_household_metrics(
    household_id: UUID,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> HouseholdMetricsResponse:
    """Get household metrics."""
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
    
    # Define date range
    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=days)
    
    # === Waste Statistics ===
    waste_query = select(
        func.count(WasteEvent.id).label("count"),
        func.sum(WasteEvent.quantity).label("total_quantity"),
        func.sum(WasteEvent.estimated_cost).label("total_cost"),
        func.sum(WasteEvent.carbon_footprint_kg).label("total_carbon"),
    ).where(
        WasteEvent.household_id == household_id,
        WasteEvent.wasted_at >= period_start,
        WasteEvent.wasted_at <= period_end,
        WasteEvent.deleted_at.is_(None),
    )
    
    result = await session.execute(waste_query)
    waste_stats = result.one()
    
    waste_estimate = {
        "count": waste_stats.count or 0,
        "total_weight_kg": round(float(waste_stats.total_quantity or 0), 2),
        "total_cost": round(float(waste_stats.total_cost or 0), 2),
        "carbon_footprint_kg": round(float(waste_stats.total_carbon or 0), 3),
    }
    
    # === Top Wasted Items ===
    top_wasted_query = (
        select(
            PantryEntry.name,
            func.count(WasteEvent.id).label("waste_count"),
            func.sum(WasteEvent.quantity).label("total_wasted"),
        )
        .join(WasteEvent)
        .where(
            WasteEvent.household_id == household_id,
            WasteEvent.wasted_at >= period_start,
            WasteEvent.wasted_at <= period_end,
            WasteEvent.deleted_at.is_(None),
        )
        .group_by(PantryEntry.name)
        .order_by(func.count(WasteEvent.id).desc())
        .limit(10)
    )
    
    result = await session.execute(top_wasted_query)
    top_wasted_items = [
        {
            "name": row.name,
            "waste_count": row.waste_count,
            "total_wasted": float(row.total_wasted),
        }
        for row in result
    ]
    
    # === Waste Reasons Breakdown ===
    reasons_query = (
        select(
            WasteEvent.reason,
            func.count(WasteEvent.id).label("count"),
        )
        .where(
            WasteEvent.household_id == household_id,
            WasteEvent.wasted_at >= period_start,
            WasteEvent.wasted_at <= period_end,
            WasteEvent.deleted_at.is_(None),
        )
        .group_by(WasteEvent.reason)
    )
    
    result = await session.execute(reasons_query)
    waste_reasons_breakdown = {row.reason.value: row.count for row in result}
    
    # === Cost Saved (estimated from avoided waste) ===
    # Assume 20% of pantry value is typically wasted, calculate savings
    pantry_value_query = select(
        func.sum(PantryEntry.purchase_price)
    ).where(
        PantryEntry.household_id == household_id,
        PantryEntry.purchase_date >= period_start,
        PantryEntry.deleted_at.is_(None),
    )
    
    result = await session.execute(pantry_value_query)
    total_pantry_value = result.scalar() or 0
    
    # Calculate saved amount (pantry value - wasted cost)
    potential_waste = total_pantry_value * 0.2  # 20% typical waste rate
    actual_waste = waste_estimate["total_cost"]
    cost_saved = max(0, potential_waste - actual_waste)
    
    # === Engagement Metrics ===
    # Active users
    active_users_query = select(
        func.count(func.distinct(HouseholdUser.user_id))
    ).where(
        HouseholdUser.household_id == household_id,
        HouseholdUser.deleted_at.is_(None),
    )
    
    result = await session.execute(active_users_query)
    active_users_count = result.scalar() or 0
    
    # Pantry entries
    pantry_count_query = select(
        func.count(PantryEntry.id)
    ).where(
        PantryEntry.household_id == household_id,
        PantryEntry.deleted_at.is_(None),
        PantryEntry.quantity > 0,
    )
    
    result = await session.execute(pantry_count_query)
    pantry_entries_count = result.scalar() or 0
    
    # Consumption logs
    consumption_count_query = select(
        func.count(ConsumptionLog.id)
    ).where(
        ConsumptionLog.household_id == household_id,
        ConsumptionLog.consumed_at >= period_start,
        ConsumptionLog.deleted_at.is_(None),
    )
    
    result = await session.execute(consumption_count_query)
    consumption_logs_count = result.scalar() or 0
    
    engagement = {
        "active_users": active_users_count,
        "pantry_entries": pantry_entries_count,
        "consumption_logs": consumption_logs_count,
        "waste_events": waste_estimate["count"],
    }
    
    # === Prediction Accuracy (if available) ===
    # TODO: Implement prediction accuracy calculation
    # Compare predicted vs actual waste
    prediction_accuracy = None
    
    return HouseholdMetricsResponse(
        household_id=household_id,
        period_start=period_start,
        period_end=period_end,
        waste_estimate=waste_estimate,
        cost_saved=round(cost_saved, 2),
        engagement=engagement,
        top_wasted_items=top_wasted_items,
        waste_reasons_breakdown=waste_reasons_breakdown,
        prediction_accuracy=prediction_accuracy,
    )

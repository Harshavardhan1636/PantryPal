"""
Consumption Router
Handles food consumption logging with recipe tracking.
"""

import logging
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_active_user, pagination_params
from backend.api.schemas import (
    ConsumptionLogCreate,
    ConsumptionLogResponse,
    MessageResponse,
)
from backend.shared.database_v2 import get_session, execute_raw_sql
from backend.shared.models_v2 import (
    User,
    ConsumptionLog,
    PantryEntry,
    HouseholdUser,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=ConsumptionLogResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log consumption",
    description="Log food consumption and update pantry quantity",
)
async def log_consumption(
    request: ConsumptionLogCreate,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> ConsumptionLog:
    """Log food consumption."""
    # Get pantry entry
    result = await session.execute(
        select(PantryEntry).where(
            PantryEntry.id == request.pantry_entry_id,
            PantryEntry.deleted_at.is_(None),
        )
    )
    
    pantry_entry = result.scalar_one_or_none()
    
    if not pantry_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pantry entry not found",
        )
    
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == pantry_entry.household_id,
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
    
    # Validate quantity
    if request.quantity > pantry_entry.quantity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient quantity. Available: {pantry_entry.quantity} {pantry_entry.unit}",
        )
    
    # Create consumption log
    consumption_log = ConsumptionLog(
        household_id=pantry_entry.household_id,
        pantry_entry_id=request.pantry_entry_id,
        quantity=request.quantity,
        unit=request.unit,
        consumed_at=datetime.utcnow(),
        recipe_id=request.recipe_id,
        notes=request.notes,
    )
    
    session.add(consumption_log)
    
    # Update pantry quantity using stored function
    try:
        await execute_raw_sql(
            session,
            "SELECT consume_pantry_quantity(:entry_id, :consumed)",
            {"entry_id": str(request.pantry_entry_id), "consumed": request.quantity},
        )
    except Exception as e:
        logger.error(f"Failed to update pantry quantity: {e}")
        # Fallback to direct update
        pantry_entry.quantity -= request.quantity
    
    await session.commit()
    await session.refresh(consumption_log)
    
    logger.info(
        f"Consumption logged: {consumption_log.id} - "
        f"{request.quantity} {request.unit} from pantry entry {request.pantry_entry_id}"
    )
    
    return consumption_log


@router.get(
    "/{household_id}",
    response_model=list[ConsumptionLogResponse],
    summary="List consumption logs",
    description="Get consumption history for household",
)
async def list_consumption_logs(
    household_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> list[ConsumptionLog]:
    """List consumption logs for household."""
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
    
    # Get consumption logs
    result = await session.execute(
        select(ConsumptionLog)
        .where(
            ConsumptionLog.household_id == household_id,
            ConsumptionLog.deleted_at.is_(None),
        )
        .offset(pagination["skip"])
        .limit(pagination["limit"])
        .order_by(ConsumptionLog.consumed_at.desc())
    )
    
    logs = result.scalars().all()
    
    return list(logs)

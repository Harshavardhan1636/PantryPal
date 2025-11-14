"""
Waste Router
Handles food waste logging with photo upload and cost/carbon tracking.
"""

import logging
from uuid import UUID
from datetime import datetime
import os

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_active_user, pagination_params
from backend.api.schemas import WasteEventCreate, WasteEventResponse, MessageResponse
from backend.api.config import get_settings
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import (
    User,
    WasteEvent,
    PantryEntry,
    HouseholdUser,
    WasteReason,
)

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


def estimate_waste_cost(
    pantry_entry: PantryEntry,
    quantity: float,
) -> float:
    """
    Estimate cost of wasted food.
    
    Args:
        pantry_entry: Pantry entry being wasted
        quantity: Quantity wasted
        
    Returns:
        Estimated cost in local currency
    """
    if not pantry_entry.purchase_price:
        return 0.0
    
    # Calculate proportional cost
    cost = (quantity / pantry_entry.quantity) * pantry_entry.purchase_price
    return round(cost, 2)


def estimate_carbon_footprint(
    pantry_entry: PantryEntry,
    quantity: float,
) -> float:
    """
    Estimate carbon footprint of wasted food.
    Uses average values from Open Food Facts / EPA data.
    
    Args:
        pantry_entry: Pantry entry being wasted
        quantity: Quantity wasted
        
    Returns:
        Estimated carbon footprint in kg CO2e
    """
    # Average carbon intensity by category (kg CO2e per kg of food)
    carbon_factors = {
        "meat": 27.0,
        "dairy": 5.0,
        "vegetables": 2.0,
        "fruits": 1.1,
        "grains": 1.6,
        "default": 3.0,
    }
    
    # Get category from canonical item or default
    category = "default"
    if pantry_entry.canonical_item and pantry_entry.canonical_item.category:
        category = pantry_entry.canonical_item.category.lower()
    
    factor = carbon_factors.get(category, carbon_factors["default"])
    
    # Estimate weight (assuming unit is kg or converting)
    weight_kg = quantity if pantry_entry.unit in ["kg", "l"] else quantity * 0.5
    
    carbon_kg = weight_kg * factor
    return round(carbon_kg, 3)


@router.post(
    "",
    response_model=WasteEventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Log waste event",
    description="Log food waste with optional photo and automatic cost/carbon calculation",
)
async def log_waste(
    pantry_entry_id: UUID = Form(...),
    quantity: float = Form(..., gt=0),
    unit: str = Form(...),
    reason: WasteReason = Form(...),
    notes: str | None = Form(None),
    photo: UploadFile | None = File(None),
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> WasteEvent:
    """Log food waste event."""
    # Get pantry entry
    result = await session.execute(
        select(PantryEntry).where(
            PantryEntry.id == pantry_entry_id,
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
    
    # Handle photo upload if provided
    photo_url = None
    if photo:
        # Validate file type
        if photo.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_IMAGE_TYPES)}",
            )
        
        # Save photo
        upload_dir = f"{settings.UPLOAD_DIR}/{pantry_entry.household_id}/waste"
        os.makedirs(upload_dir, exist_ok=True)
        
        photo_path = f"{upload_dir}/{datetime.utcnow().isoformat()}_{photo.filename}"
        
        with open(photo_path, "wb") as f:
            f.write(await photo.read())
        
        photo_url = photo_path  # In production, use CDN URL
    
    # Estimate cost and carbon footprint
    estimated_cost = estimate_waste_cost(pantry_entry, quantity)
    carbon_footprint = estimate_carbon_footprint(pantry_entry, quantity)
    
    # Create waste event
    waste_event = WasteEvent(
        household_id=pantry_entry.household_id,
        pantry_entry_id=pantry_entry_id,
        quantity=quantity,
        unit=unit,
        reason=reason,
        wasted_at=datetime.utcnow(),
        estimated_cost=estimated_cost,
        carbon_footprint_kg=carbon_footprint,
        photo_url=photo_url,
        notes=notes,
    )
    
    session.add(waste_event)
    
    # Update pantry quantity
    pantry_entry.quantity = max(0, pantry_entry.quantity - quantity)
    
    await session.commit()
    await session.refresh(waste_event)
    
    logger.info(
        f"Waste event logged: {waste_event.id} - "
        f"{quantity} {unit} from pantry entry {pantry_entry_id} "
        f"(cost: ${estimated_cost}, carbon: {carbon_footprint} kg CO2e)"
    )
    
    return waste_event


@router.get(
    "/{household_id}",
    response_model=list[WasteEventResponse],
    summary="List waste events",
    description="Get waste history for household",
)
async def list_waste_events(
    household_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> list[WasteEvent]:
    """List waste events for household."""
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
    
    # Get waste events
    result = await session.execute(
        select(WasteEvent)
        .where(
            WasteEvent.household_id == household_id,
            WasteEvent.deleted_at.is_(None),
        )
        .offset(pagination["skip"])
        .limit(pagination["limit"])
        .order_by(WasteEvent.wasted_at.desc())
    )
    
    events = result.scalars().all()
    
    return list(events)

"""
Pantry Router
Handles pantry inventory management with barcode lookup and item cataloging.
"""

import logging
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.dependencies import (
    get_current_active_user,
    require_household_access,
    pagination_params,
)
from backend.api.schemas import (
    PantryEntryCreate,
    PantryEntryUpdate,
    PantryEntryResponse,
    MessageResponse,
)
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import (
    User,
    PantryEntry,
    ItemCatalog,
    HouseholdUser,
    UserRole,
)

logger = logging.getLogger(__name__)

router = APIRouter()


async def resolve_canonical_item(
    session: AsyncSession,
    barcode: str | None,
    name: str,
) -> UUID | None:
    """
    Resolve pantry item to canonical item from catalog.
    
    Args:
        session: Database session
        barcode: Product barcode (if available)
        name: Item name
        
    Returns:
        UUID of canonical item or None if not found
    """
    if barcode:
        # Try exact barcode match first
        result = await session.execute(
            select(ItemCatalog).where(ItemCatalog.barcode == barcode)
        )
        item = result.scalar_one_or_none()
        if item:
            return item.id
    
    # Try fuzzy name search using trigram similarity
    result = await session.execute(
        select(ItemCatalog)
        .where(ItemCatalog.search_vector.op("@@")(func.plainto_tsquery("english", name)))
        .limit(1)
    )
    item = result.scalar_one_or_none()
    
    if item:
        return item.id
    
    return None


@router.post(
    "",
    response_model=PantryEntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add pantry item",
    description="Add a new item to pantry inventory with barcode resolution",
)
async def add_pantry_item(
    request: PantryEntryCreate,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> PantryEntry:
    """Add item to pantry."""
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == request.household_id,
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
    
    # Resolve canonical item from catalog
    canonical_item_id = await resolve_canonical_item(
        session,
        request.barcode,
        request.name,
    )
    
    # Create pantry entry
    pantry_entry = PantryEntry(
        household_id=request.household_id,
        canonical_item_id=canonical_item_id,
        name=request.name,
        barcode=request.barcode,
        quantity=request.quantity,
        unit=request.unit,
        purchase_date=request.purchase_date,
        expiry_date=request.expiry_date,
        opened_at=request.opened_at,
        storage_location=request.storage_location,
        purchase_price=request.purchase_price,
        notes=request.notes,
    )
    
    session.add(pantry_entry)
    await session.commit()
    await session.refresh(pantry_entry)
    
    # Load relationships
    await session.refresh(pantry_entry, ["canonical_item"])
    
    logger.info(f"Pantry entry created: {pantry_entry.id} in household {request.household_id}")
    
    return pantry_entry


@router.get(
    "/{household_id}",
    response_model=list[PantryEntryResponse],
    summary="List pantry items",
    description="Get all pantry entries for a household with latest predictions",
)
async def list_pantry_items(
    household_id: UUID,
    storage_location: str | None = Query(None, description="Filter by storage location"),
    include_depleted: bool = Query(False, description="Include items with zero quantity"),
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> list[PantryEntry]:
    """List pantry items for household."""
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
    
    # Build query
    query = (
        select(PantryEntry)
        .options(
            selectinload(PantryEntry.canonical_item),
            selectinload(PantryEntry.predictions),
        )
        .where(
            PantryEntry.household_id == household_id,
            PantryEntry.deleted_at.is_(None),
        )
    )
    
    if not include_depleted:
        query = query.where(PantryEntry.quantity > 0)
    
    if storage_location:
        query = query.where(PantryEntry.storage_location == storage_location)
    
    query = query.offset(pagination["skip"]).limit(pagination["limit"])
    query = query.order_by(PantryEntry.expiry_date.asc().nullslast())
    
    result = await session.execute(query)
    entries = result.scalars().all()
    
    return list(entries)


@router.get(
    "/item/{entry_id}",
    response_model=PantryEntryResponse,
    summary="Get pantry item",
    description="Get details of a specific pantry entry",
)
async def get_pantry_item(
    entry_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> PantryEntry:
    """Get pantry item details."""
    result = await session.execute(
        select(PantryEntry)
        .options(
            selectinload(PantryEntry.canonical_item),
            selectinload(PantryEntry.predictions),
        )
        .where(
            PantryEntry.id == entry_id,
            PantryEntry.deleted_at.is_(None),
        )
    )
    
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pantry entry not found",
        )
    
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == entry.household_id,
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
    
    return entry


@router.patch(
    "/item/{entry_id}",
    response_model=PantryEntryResponse,
    summary="Update pantry item",
    description="Update pantry entry details",
)
async def update_pantry_item(
    entry_id: UUID,
    request: PantryEntryUpdate,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> PantryEntry:
    """Update pantry item."""
    result = await session.execute(
        select(PantryEntry).where(
            PantryEntry.id == entry_id,
            PantryEntry.deleted_at.is_(None),
        )
    )
    
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pantry entry not found",
        )
    
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == entry.household_id,
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
    
    # Update fields
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(entry, field, value)
    
    await session.commit()
    await session.refresh(entry, ["canonical_item"])
    
    logger.info(f"Pantry entry updated: {entry_id}")
    
    return entry


@router.delete(
    "/item/{entry_id}",
    response_model=MessageResponse,
    summary="Delete pantry item",
    description="Soft delete pantry entry",
)
async def delete_pantry_item(
    entry_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Delete pantry item."""
    result = await session.execute(
        select(PantryEntry).where(
            PantryEntry.id == entry_id,
            PantryEntry.deleted_at.is_(None),
        )
    )
    
    entry = result.scalar_one_or_none()
    
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pantry entry not found",
        )
    
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == entry.household_id,
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
    
    # Soft delete
    entry.deleted_at = datetime.utcnow()
    
    await session.commit()
    
    logger.info(f"Pantry entry deleted: {entry_id}")
    
    return MessageResponse(
        message="Pantry item deleted successfully",
        success=True,
    )

"""
Households Router
Handles household creation, management, and membership.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.dependencies import (
    get_current_active_user,
    require_household_access,
    pagination_params,
)
from backend.api.schemas import (
    HouseholdCreate,
    HouseholdUpdate,
    HouseholdResponse,
    HouseholdUserResponse,
    MessageResponse,
)
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import User, Household, HouseholdUser, UserRole

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "",
    response_model=list[HouseholdResponse],
    summary="List user's households",
    description="Get all households the current user is a member of",
)
async def list_households(
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> list[Household]:
    """List all households for current user."""
    result = await session.execute(
        select(Household)
        .join(HouseholdUser)
        .where(
            HouseholdUser.user_id == current_user.id,
            HouseholdUser.deleted_at.is_(None),
            Household.deleted_at.is_(None),
        )
        .offset(pagination["skip"])
        .limit(pagination["limit"])
        .order_by(Household.created_at.desc())
    )
    
    households = result.scalars().all()
    return list(households)


@router.post(
    "",
    response_model=HouseholdResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create household",
    description="Create a new household and add current user as owner",
)
async def create_household(
    request: HouseholdCreate,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> Household:
    """Create a new household."""
    # Create household
    household = Household(
        name=request.name,
        timezone=request.timezone,
        members_count=request.members_count,
        currency=request.currency,
        settings=request.settings,
    )
    
    session.add(household)
    await session.flush()  # Get household ID
    
    # Add creator as owner
    membership = HouseholdUser(
        household_id=household.id,
        user_id=current_user.id,
        role=UserRole.OWNER,
    )
    
    session.add(membership)
    await session.commit()
    await session.refresh(household)
    
    logger.info(f"Household created: {household.id} by user {current_user.id}")
    
    return household


@router.get(
    "/{household_id}",
    response_model=HouseholdResponse,
    summary="Get household details",
    description="Get details of a specific household",
)
async def get_household(
    household_id: UUID,
    membership: HouseholdUser = Depends(
        lambda h=None: require_household_access(h, min_role=UserRole.VIEWER)
    ),
    session: AsyncSession = Depends(get_session),
) -> Household:
    """Get household details."""
    result = await session.execute(
        select(Household).where(
            Household.id == household_id,
            Household.deleted_at.is_(None),
        )
    )
    
    household = result.scalar_one_or_none()
    
    if not household:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Household not found",
        )
    
    return household


@router.patch(
    "/{household_id}",
    response_model=HouseholdResponse,
    summary="Update household",
    description="Update household details (requires ADMIN role)",
)
async def update_household(
    household_id: UUID,
    request: HouseholdUpdate,
    membership: HouseholdUser = Depends(
        lambda h=None: require_household_access(h, min_role=UserRole.ADMIN)
    ),
    session: AsyncSession = Depends(get_session),
) -> Household:
    """Update household details."""
    result = await session.execute(
        select(Household).where(
            Household.id == household_id,
            Household.deleted_at.is_(None),
        )
    )
    
    household = result.scalar_one_or_none()
    
    if not household:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Household not found",
        )
    
    # Update fields
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(household, field, value)
    
    await session.commit()
    await session.refresh(household)
    
    logger.info(f"Household updated: {household_id}")
    
    return household


@router.delete(
    "/{household_id}",
    response_model=MessageResponse,
    summary="Delete household",
    description="Soft delete household (requires OWNER role)",
)
async def delete_household(
    household_id: UUID,
    membership: HouseholdUser = Depends(
        lambda h=None: require_household_access(h, min_role=UserRole.OWNER)
    ),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Soft delete household."""
    result = await session.execute(
        select(Household).where(
            Household.id == household_id,
            Household.deleted_at.is_(None),
        )
    )
    
    household = result.scalar_one_or_none()
    
    if not household:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Household not found",
        )
    
    # Soft delete
    from datetime import datetime
    household.deleted_at = datetime.utcnow()
    
    await session.commit()
    
    logger.info(f"Household deleted: {household_id}")
    
    return MessageResponse(
        message="Household deleted successfully",
        success=True,
    )


@router.get(
    "/{household_id}/members",
    response_model=list[HouseholdUserResponse],
    summary="List household members",
    description="Get all members of a household",
)
async def list_household_members(
    household_id: UUID,
    membership: HouseholdUser = Depends(
        lambda h=None: require_household_access(h, min_role=UserRole.VIEWER)
    ),
    session: AsyncSession = Depends(get_session),
) -> list[HouseholdUser]:
    """List all members of a household."""
    result = await session.execute(
        select(HouseholdUser)
        .options(selectinload(HouseholdUser.user))
        .where(
            HouseholdUser.household_id == household_id,
            HouseholdUser.deleted_at.is_(None),
        )
        .order_by(HouseholdUser.joined_at)
    )
    
    members = result.scalars().all()
    return list(members)


@router.post(
    "/{household_id}/members",
    response_model=HouseholdUserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add household member",
    description="Add a user to household (requires ADMIN role)",
)
async def add_household_member(
    household_id: UUID,
    user_email: str,
    role: UserRole = UserRole.MEMBER,
    membership: HouseholdUser = Depends(
        lambda h=None: require_household_access(h, min_role=UserRole.ADMIN)
    ),
    session: AsyncSession = Depends(get_session),
) -> HouseholdUser:
    """Add a member to household."""
    # Find user by email
    result = await session.execute(
        select(User).where(
            User.email == user_email,
            User.deleted_at.is_(None),
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Check if already a member
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == household_id,
            HouseholdUser.user_id == user.id,
            HouseholdUser.deleted_at.is_(None),
        )
    )
    existing_membership = result.scalar_one_or_none()
    
    if existing_membership:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a member of this household",
        )
    
    # Add member
    new_membership = HouseholdUser(
        household_id=household_id,
        user_id=user.id,
        role=role,
    )
    
    session.add(new_membership)
    await session.commit()
    await session.refresh(new_membership)
    
    logger.info(f"Member added to household {household_id}: {user.id}")
    
    return new_membership

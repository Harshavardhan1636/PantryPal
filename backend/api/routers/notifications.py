"""
Notifications Router
Handles notification scheduling and delivery management.
"""

import logging
from uuid import UUID
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_current_active_user, pagination_params
from backend.api.schemas import (
    NotificationScheduleCreate,
    NotificationResponse,
    MessageResponse,
)
from backend.api.config import get_settings
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import User, Notification, HouseholdUser

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/schedule",
    response_model=NotificationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Schedule notification",
    description="Schedule a notification for a household",
)
async def schedule_notification(
    request: NotificationScheduleCreate,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> Notification:
    """Schedule a notification."""
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
    
    if not settings.FEATURE_NOTIFICATIONS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Notifications are currently disabled",
        )
    
    # Generate notification message
    message = f"PantryPal: {request.notification_type.value}"
    
    # Create notification
    notification = Notification(
        household_id=request.household_id,
        notification_type=request.notification_type,
        message=message,
        payload=request.payload,
        scheduled_at=request.scheduled_at,
    )
    
    session.add(notification)
    await session.commit()
    await session.refresh(notification)
    
    logger.info(
        f"Notification scheduled: {notification.id} for household {request.household_id}"
    )
    
    # If scheduled for immediate delivery, trigger send
    if request.scheduled_at <= datetime.utcnow():
        # TODO: Implement actual notification sending via Twilio/SendGrid/FCM
        from backend.api.services.notification_service import send_notification
        try:
            await send_notification(notification, request.channels)
            notification.sent_at = datetime.utcnow()
            await session.commit()
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    return notification


@router.get(
    "/{household_id}",
    response_model=List[NotificationResponse],
    summary="List notifications",
    description="Get notifications for household",
)
async def list_notifications(
    household_id: UUID,
    unread_only: bool = False,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> List[Notification]:
    """List notifications for household."""
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
    
    query = select(Notification).where(
        Notification.household_id == household_id,
        Notification.deleted_at.is_(None),
    )
    
    if unread_only:
        query = query.where(Notification.read_at.is_(None))
    
    query = (
        query.offset(pagination["skip"])
        .limit(pagination["limit"])
        .order_by(Notification.created_at.desc())
    )
    
    result = await session.execute(query)
    notifications = result.scalars().all()
    
    return list(notifications)


@router.patch(
    "/{notification_id}/read",
    response_model=MessageResponse,
    summary="Mark as read",
    description="Mark notification as read",
)
async def mark_notification_read(
    notification_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> MessageResponse:
    """Mark notification as read."""
    result = await session.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.deleted_at.is_(None),
        )
    )
    
    notification = result.scalar_one_or_none()
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found",
        )
    
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == notification.household_id,
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
    
    notification.read_at = datetime.utcnow()
    await session.commit()
    
    return MessageResponse(
        message="Notification marked as read",
        success=True,
    )

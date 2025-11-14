"""
Purchases Router
Handles purchase recording and receipt OCR processing.
"""

import logging
from uuid import UUID
from typing import List

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
from backend.api.schemas import PurchaseResponse
from backend.api.config import get_settings
from backend.shared.database_v2 import get_session
from backend.shared.models_v2 import User, Purchase, HouseholdUser, PantryEntry
from backend.api.services.ocr_service import process_receipt_image

settings = get_settings()
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "",
    response_model=PurchaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload receipt",
    description="Upload receipt image for OCR processing and automatic pantry entry creation",
)
async def upload_receipt(
    household_id: UUID = Form(...),
    receipt_image: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> Purchase:
    """Upload and process receipt with OCR."""
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
    
    # Validate file type
    if receipt_image.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed: {', '.join(settings.ALLOWED_IMAGE_TYPES)}",
        )
    
    # Validate file size
    file_content = await receipt_image.read()
    file_size_mb = len(file_content) / (1024 * 1024)
    
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )
    
    # Save file and get URL (implement storage service - S3, GCS, etc.)
    from datetime import datetime
    import os
    
    upload_dir = f"{settings.UPLOAD_DIR}/{household_id}/receipts"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = f"{upload_dir}/{datetime.utcnow().isoformat()}_{receipt_image.filename}"
    
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    receipt_image_url = file_path  # In production, use CDN URL
    
    # Process OCR if enabled
    ocr_result = None
    extracted_items = []
    
    if settings.FEATURE_OCR_ENABLED:
        try:
            ocr_result = await process_receipt_image(file_content)
            extracted_items = ocr_result.get("items", [])
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            # Continue without OCR - user can manually add items
    
    # Create purchase record
    purchase = Purchase(
        household_id=household_id,
        purchased_at=datetime.utcnow(),
        total_amount=ocr_result.get("total_amount") if ocr_result else None,
        currency=membership.household.currency,
        receipt_image_url=receipt_image_url,
        receipt_ocr_result=ocr_result,
    )
    
    session.add(purchase)
    await session.flush()  # Get purchase ID
    
    # Create pantry entries from OCR results
    for item_data in extracted_items:
        pantry_entry = PantryEntry(
            household_id=household_id,
            name=item_data.get("name", "Unknown Item"),
            barcode=item_data.get("barcode"),
            quantity=item_data.get("quantity", 1.0),
            unit=item_data.get("unit", "pcs"),
            purchase_date=purchase.purchased_at,
            purchase_price=item_data.get("price"),
            storage_location="pantry",  # Default, user can update
            notes=f"Auto-imported from receipt on {purchase.purchased_at.date()}",
        )
        
        session.add(pantry_entry)
    
    await session.commit()
    await session.refresh(purchase)
    
    logger.info(
        f"Receipt processed: {purchase.id} with {len(extracted_items)} items "
        f"for household {household_id}"
    )
    
    return purchase


@router.get(
    "/{household_id}",
    response_model=List[PurchaseResponse],
    summary="List purchases",
    description="Get purchase history for household",
)
async def list_purchases(
    household_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
    pagination: dict = Depends(pagination_params),
) -> List[Purchase]:
    """List purchases for household."""
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
    
    # Get purchases
    result = await session.execute(
        select(Purchase)
        .where(
            Purchase.household_id == household_id,
            Purchase.deleted_at.is_(None),
        )
        .offset(pagination["skip"])
        .limit(pagination["limit"])
        .order_by(Purchase.purchased_at.desc())
    )
    
    purchases = result.scalars().all()
    
    return list(purchases)


@router.get(
    "/purchase/{purchase_id}",
    response_model=PurchaseResponse,
    summary="Get purchase details",
    description="Get details of a specific purchase",
)
async def get_purchase(
    purchase_id: UUID,
    current_user: User = Depends(get_current_active_user),
    session: AsyncSession = Depends(get_session),
) -> Purchase:
    """Get purchase details."""
    result = await session.execute(
        select(Purchase).where(
            Purchase.id == purchase_id,
            Purchase.deleted_at.is_(None),
        )
    )
    
    purchase = result.scalar_one_or_none()
    
    if not purchase:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Purchase not found",
        )
    
    # Verify household access
    result = await session.execute(
        select(HouseholdUser).where(
            HouseholdUser.household_id == purchase.household_id,
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
    
    return purchase

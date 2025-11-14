"""FastAPI inventory microservice for PantryPal."""
import os
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database import get_db, init_db, close_db
from inventory_service.models import (
    ItemsCatalog, InventoryBatch, ConsumptionLog, WasteEvent
)


# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    await init_db()
    yield
    await close_db()


# FastAPI app
app = FastAPI(
    title="PantryPal Inventory Service",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ItemsCatalogSearch(BaseModel):
    """Search catalog items."""
    query: str = Field(min_length=1)
    limit: int = Field(default=20, le=100)


class ItemsCatalogResponse(BaseModel):
    """Catalog item response."""
    id: str
    name: str
    brand: Optional[str]
    category: Optional[str]
    barcode: Optional[str]
    avg_shelf_life_days: Optional[int]
    typical_storage_type: Optional[str]
    image_url: Optional[str]


class InventoryBatchCreate(BaseModel):
    """Create inventory batch."""
    household_id: str
    item_id: Optional[str] = None
    custom_item_name: Optional[str] = None
    quantity: Decimal = Field(gt=0)
    unit: str = Field(min_length=1, max_length=20)
    purchase_date: date
    purchase_price: Optional[Decimal] = None
    purchase_store: Optional[str] = None
    expiry_date: Optional[date] = None
    storage_type: str = Field(min_length=1)
    packaging_type: Optional[str] = None
    location: Optional[str] = None
    notes: Optional[str] = None


class InventoryBatchUpdate(BaseModel):
    """Update inventory batch."""
    quantity: Optional[Decimal] = Field(default=None, gt=0)
    opened_date: Optional[date] = None
    expiry_date: Optional[date] = None
    storage_type: Optional[str] = None
    location: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class InventoryBatchResponse(BaseModel):
    """Inventory batch response."""
    id: str
    household_id: str
    item_name: str
    quantity: Decimal
    unit: str
    purchase_date: date
    expiry_date: Optional[date]
    storage_type: str
    status: str
    risk_score: Optional[Decimal] = None
    days_until_expiry: Optional[int] = None


class ConsumeRequest(BaseModel):
    """Consume from batch."""
    batch_id: str
    quantity: Decimal = Field(gt=0)
    meal_type: Optional[str] = None
    notes: Optional[str] = None


class WasteRequest(BaseModel):
    """Report waste."""
    batch_id: str
    quantity: Decimal = Field(gt=0)
    reason: str
    reason_details: Optional[str] = None
    photo_url: Optional[str] = None


# Mock user dependency (replace with actual auth)
async def get_current_user_id() -> UUID:
    """Get current user ID from auth token."""
    # TODO: Implement actual JWT validation
    return UUID("00000000-0000-0000-0000-000000000001")


# API Endpoints
@app.get("/catalog/search", response_model=list[ItemsCatalogResponse])
async def search_catalog(
    query: str = Query(min_length=1),
    limit: int = Query(default=20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Search item catalog by name or barcode."""
    result = await db.execute(
        select(ItemsCatalog)
        .where(
            ItemsCatalog.name.ilike(f"%{query}%") | 
            ItemsCatalog.barcode.ilike(f"%{query}%")
        )
        .limit(limit)
    )
    items = result.scalars().all()
    
    return [
        {
            "id": str(item.id),
            "name": item.name,
            "brand": item.brand,
            "category": item.category,
            "barcode": item.barcode,
            "avg_shelf_life_days": item.avg_shelf_life_days,
            "typical_storage_type": item.typical_storage_type,
            "image_url": item.image_url,
        }
        for item in items
    ]


@app.get("/catalog/barcode/{barcode}", response_model=ItemsCatalogResponse)
async def get_by_barcode(
    barcode: str,
    db: AsyncSession = Depends(get_db),
):
    """Get catalog item by barcode."""
    result = await db.execute(
        select(ItemsCatalog).where(ItemsCatalog.barcode == barcode)
    )
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {
        "id": str(item.id),
        "name": item.name,
        "brand": item.brand,
        "category": item.category,
        "barcode": item.barcode,
        "avg_shelf_life_days": item.avg_shelf_life_days,
        "typical_storage_type": item.typical_storage_type,
        "image_url": item.image_url,
    }


@app.post("/batches", response_model=dict, status_code=201)
async def create_batch(
    batch_data: InventoryBatchCreate,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Create new inventory batch."""
    batch = InventoryBatch(
        household_id=UUID(batch_data.household_id),
        item_id=UUID(batch_data.item_id) if batch_data.item_id else None,
        custom_item_name=batch_data.custom_item_name,
        quantity=batch_data.quantity,
        unit=batch_data.unit,
        purchase_date=batch_data.purchase_date,
        purchase_price=batch_data.purchase_price,
        purchase_store=batch_data.purchase_store,
        expiry_date=batch_data.expiry_date,
        storage_type=batch_data.storage_type,
        packaging_type=batch_data.packaging_type,
        location=batch_data.location,
        notes=batch_data.notes,
        created_by=user_id,
        status="active",
    )
    
    db.add(batch)
    await db.commit()
    await db.refresh(batch)
    
    return {"id": str(batch.id), "status": "created"}


@app.get("/batches/{household_id}", response_model=list[InventoryBatchResponse])
async def get_household_inventory(
    household_id: str,
    status: Optional[str] = Query(default="active"),
    db: AsyncSession = Depends(get_db),
):
    """Get all inventory batches for a household."""
    query = select(InventoryBatch, ItemsCatalog).outerjoin(
        ItemsCatalog, InventoryBatch.item_id == ItemsCatalog.id
    ).where(
        InventoryBatch.household_id == UUID(household_id)
    )
    
    if status:
        query = query.where(InventoryBatch.status == status)
    
    result = await db.execute(query)
    batches = result.all()
    
    return [
        {
            "id": str(batch.id),
            "household_id": str(batch.household_id),
            "item_name": item.name if item else batch.custom_item_name,
            "quantity": batch.quantity,
            "unit": batch.unit,
            "purchase_date": batch.purchase_date,
            "expiry_date": batch.expiry_date,
            "storage_type": batch.storage_type,
            "status": batch.status,
            "days_until_expiry": (
                (batch.expiry_date - date.today()).days
                if batch.expiry_date else None
            ),
        }
        for batch, item in batches
    ]


@app.patch("/batches/{batch_id}")
async def update_batch(
    batch_id: str,
    update_data: InventoryBatchUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update inventory batch."""
    result = await db.execute(
        select(InventoryBatch).where(InventoryBatch.id == UUID(batch_id))
    )
    batch = result.scalar_one_or_none()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(batch, field, value)
    
    await db.commit()
    return {"status": "updated"}


@app.post("/consume")
async def consume_batch(
    consume_data: ConsumeRequest,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Record consumption from batch."""
    # Get batch
    result = await db.execute(
        select(InventoryBatch).where(InventoryBatch.id == UUID(consume_data.batch_id))
    )
    batch = result.scalar_one_or_none()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Check sufficient quantity
    if batch.quantity < consume_data.quantity:
        raise HTTPException(status_code=400, detail="Insufficient quantity")
    
    # Create consumption log
    log = ConsumptionLog(
        batch_id=batch.id,
        household_id=batch.household_id,
        quantity=consume_data.quantity,
        unit=batch.unit,
        meal_type=consume_data.meal_type,
        notes=consume_data.notes,
        user_id=user_id,
    )
    db.add(log)
    
    # Update batch quantity
    batch.quantity -= consume_data.quantity
    if batch.quantity == 0:
        batch.status = "consumed"
    
    await db.commit()
    return {"status": "consumed", "remaining_quantity": float(batch.quantity)}


@app.post("/waste")
async def report_waste(
    waste_data: WasteRequest,
    user_id: UUID = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Report waste event."""
    # Get batch
    result = await db.execute(
        select(InventoryBatch).where(InventoryBatch.id == UUID(waste_data.batch_id))
    )
    batch = result.scalar_one_or_none()
    
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    # Estimate cost
    cost_estimate = None
    if batch.purchase_price:
        cost_estimate = (batch.purchase_price / batch.quantity) * waste_data.quantity
    
    # Create waste event
    event = WasteEvent(
        batch_id=batch.id,
        household_id=batch.household_id,
        quantity=waste_data.quantity,
        unit=batch.unit,
        reason=waste_data.reason,
        reason_details=waste_data.reason_details,
        photo_url=waste_data.photo_url,
        cost_estimate=cost_estimate,
        user_id=user_id,
    )
    db.add(event)
    
    # Update batch
    batch.quantity -= waste_data.quantity
    if batch.quantity <= 0:
        batch.status = "wasted"
    
    await db.commit()
    return {"status": "recorded", "cost_estimate": float(cost_estimate) if cost_estimate else None}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "inventory"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

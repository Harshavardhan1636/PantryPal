"""Inventory service database models."""
from datetime import date, datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Numeric, Date, ForeignKey, Text, JSON, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column

from shared.database import Base, TimestampMixin


class ItemsCatalog(Base, TimestampMixin):
    """Canonical item catalog (synced from Open Food Facts)."""
    
    __tablename__ = "items_catalog"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(500), nullable=False)
    brand: Mapped[Optional[str]] = mapped_column(String(255))
    category: Mapped[Optional[str]] = mapped_column(String(100))
    barcode: Mapped[Optional[str]] = mapped_column(String(50))
    avg_shelf_life_days: Mapped[Optional[int]]
    typical_storage_type: Mapped[Optional[str]] = mapped_column(String(50))
    nutritional_data: Mapped[dict] = mapped_column(JSON, default=dict)
    allergens: Mapped[list] = mapped_column(JSON, default=list)
    image_url: Mapped[Optional[str]] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(50), default="openfoodfacts")
    external_id: Mapped[Optional[str]] = mapped_column(String(255))
    metadata: Mapped[dict] = mapped_column(JSON, default=dict)


class RegionalVariant(Base, TimestampMixin):
    """Regional variants of catalog items."""
    
    __tablename__ = "regional_variants"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    item_id: Mapped[UUID] = mapped_column(ForeignKey("items_catalog.id", ondelete="CASCADE"))
    region: Mapped[str] = mapped_column(String(10))
    local_name: Mapped[str] = mapped_column(String(500))
    barcode_variants: Mapped[list] = mapped_column(JSON, default=list)
    availability: Mapped[dict] = mapped_column(JSON, default=dict)


class InventoryBatch(Base, TimestampMixin):
    """Individual inventory batch (purchase instance)."""
    
    __tablename__ = "inventory_batches"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    household_id: Mapped[UUID] = mapped_column(ForeignKey("households.id", ondelete="CASCADE"))
    item_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("items_catalog.id"))
    custom_item_name: Mapped[Optional[str]] = mapped_column(String(500))
    quantity: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    purchase_date: Mapped[date] = mapped_column(Date, nullable=False)
    purchase_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    purchase_store: Mapped[Optional[str]] = mapped_column(String(255))
    opened_date: Mapped[Optional[date]] = mapped_column(Date)
    expiry_date: Mapped[Optional[date]] = mapped_column(Date)
    predicted_expiry_date: Mapped[Optional[date]] = mapped_column(Date)
    storage_type: Mapped[str] = mapped_column(String(50), nullable=False)
    packaging_type: Mapped[Optional[str]] = mapped_column(String(100))
    location: Mapped[Optional[str]] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(50), default="active")
    notes: Mapped[Optional[str]] = mapped_column(Text)
    created_by: Mapped[Optional[UUID]] = mapped_column(ForeignKey("users.id"))
    consumed_at: Mapped[Optional[datetime]]
    
    __table_args__ = (
        CheckConstraint("quantity > 0", name="positive_quantity"),
    )


class ConsumptionLog(Base):
    """Time-series consumption tracking."""
    
    __tablename__ = "consumption_logs"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    batch_id: Mapped[UUID] = mapped_column(ForeignKey("inventory_batches.id", ondelete="CASCADE"))
    household_id: Mapped[UUID] = mapped_column(ForeignKey("households.id", ondelete="CASCADE"))
    quantity: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    consumed_at: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)
    recipe_id: Mapped[Optional[UUID]]
    meal_type: Mapped[Optional[str]] = mapped_column(String(50))
    user_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("users.id"))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    
    __table_args__ = (
        CheckConstraint("quantity > 0", name="positive_consumed_quantity"),
    )


class WasteEvent(Base):
    """Time-series waste tracking."""
    
    __tablename__ = "waste_events"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    batch_id: Mapped[UUID] = mapped_column(ForeignKey("inventory_batches.id", ondelete="CASCADE"))
    household_id: Mapped[UUID] = mapped_column(ForeignKey("households.id", ondelete="CASCADE"))
    quantity: Mapped[Decimal] = mapped_column(Numeric(10, 3), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(String(100))
    reason_details: Mapped[Optional[str]] = mapped_column(Text)
    photo_url: Mapped[Optional[str]] = mapped_column(Text)
    cost_estimate: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    wasted_at: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)
    user_id: Mapped[Optional[UUID]] = mapped_column(ForeignKey("users.id"))
    verified: Mapped[bool] = mapped_column(default=False)
    
    __table_args__ = (
        CheckConstraint("quantity > 0", name="positive_waste_quantity"),
    )

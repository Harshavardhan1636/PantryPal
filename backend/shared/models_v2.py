"""
Production-ready SQLAlchemy models for PantryPal v2.0
Senior SDE-3 Level Implementation

Features:
- UUID primary keys across all tables
- Comprehensive relationships with lazy loading optimization
- Type hints for IDE support
- Mixins for common patterns (timestamps, soft delete)
- Validation via SQLAlchemy validators
- Efficient indexing strategy
- Support for jsonb fields
- Full-text search integration
- Vector embeddings support (pgvector)
"""

import enum
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID

from sqlalchemy import (
    Boolean, Integer, BigInteger, Numeric, Text, TIMESTAMP,
    ForeignKey, CheckConstraint, UniqueConstraint, Index,
    func, text, Enum as SQLEnum, ARRAY
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
    validates
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB, TSVECTOR, INET
from sqlalchemy.ext.hybrid import hybrid_property


# ============================================================================
# BASE & MIXINS
# ============================================================================

class Base(DeclarativeBase):
    """Base class for all models"""
    type_annotation_map = {
        dict: JSONB,
        list: JSONB,
    }


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )


class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True
    )
    
    @hybrid_property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None
    
    def soft_delete(self) -> None:
        """Mark record as deleted"""
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self) -> None:
        """Restore soft-deleted record"""
        self.deleted_at = None


# ============================================================================
# ENUMS
# ============================================================================

class UserRole(str, enum.Enum):
    """User roles in household"""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class StorageType(str, enum.Enum):
    """Food storage locations"""
    FRIDGE = "fridge"
    FREEZER = "freezer"
    PANTRY = "pantry"
    COUNTER = "counter"
    OTHER = "other"


class WasteReason(str, enum.Enum):
    """Reasons for food waste"""
    EXPIRED = "expired"
    SPOILED = "spoiled"
    OVERCOOKED = "overcooked"
    DISLIKED = "disliked"
    EXCESS = "excess"
    OTHER = "other"


class NotificationType(str, enum.Enum):
    """Notification categories"""
    EXPIRY_WARNING = "expiry_warning"
    WASTE_ALERT = "waste_alert"
    RECIPE_SUGGESTION = "recipe_suggestion"
    SHOPPING_REMINDER = "shopping_reminder"
    SYSTEM = "system"


class RiskClass(str, enum.Enum):
    """Waste risk classifications"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CurrencyCode(str, enum.Enum):
    """Supported currency codes"""
    INR = "INR"
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    AUD = "AUD"
    CAD = "CAD"
    SGD = "SGD"


# ============================================================================
# USER MANAGEMENT
# ============================================================================

class User(Base, TimestampMixin, SoftDeleteMixin):
    """User account model"""
    __tablename__ = "users"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()")
    )
    
    # Authentication
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    password_hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    avatar_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # External auth
    email_verified: Mapped[bool] = mapped_column(Boolean, server_default="false")
    phone_verified: Mapped[bool] = mapped_column(Boolean, server_default="false")
    auth_provider: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    external_auth_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Preferences
    timezone: Mapped[str] = mapped_column(Text, server_default="'Asia/Kolkata'", nullable=False)
    locale: Mapped[str] = mapped_column(Text, server_default="'en-IN'")
    notification_preferences: Mapped[dict] = mapped_column(
        JSONB,
        server_default=text('\'{"email": true, "push": true, "sms": false, "quiet_hours_start": "22:00", "quiet_hours_end": "08:00"}\'')
    )
    
    # Status
    last_login: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    last_active_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, server_default="false")
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true")
    deactivated_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    # Metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    # Relationships
    households: Mapped[List["HouseholdUser"]] = relationship(
        "HouseholdUser",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    sessions: Mapped[List["UserSession"]] = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        CheckConstraint("email = lower(email)", name="users_email_lower_check"),
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'", name="users_email_format_check"),
        CheckConstraint("length(name) >= 1", name="users_name_length_check"),
        Index("idx_users_email", "email", postgresql_where=text("deleted_at IS NULL")),
        Index("idx_users_external_auth", "auth_provider", "external_auth_id", postgresql_where=text("deleted_at IS NULL")),
        Index("idx_users_last_active", text("last_active_at DESC"), postgresql_where=text("is_active = true")),
    )
    
    @validates("email")
    def validate_email(self, key: str, value: str) -> str:
        """Ensure email is lowercase"""
        return value.lower() if value else value
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, name={self.name})>"


class UserSession(Base):
    """User session for JWT refresh tokens"""
    __tablename__ = "user_sessions"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    refresh_token_hash: Mapped[str] = mapped_column(Text, nullable=False)
    device_info: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="sessions")
    
    __table_args__ = (
        CheckConstraint("expires_at > created_at", name="sessions_not_expired"),
        Index("idx_sessions_user", "user_id", postgresql_where=text("revoked_at IS NULL")),
        Index("idx_sessions_token", "refresh_token_hash", postgresql_where=text("revoked_at IS NULL")),
    )


# ============================================================================
# HOUSEHOLD MANAGEMENT
# ============================================================================

class Household(Base, TimestampMixin, SoftDeleteMixin):
    """Household (multi-tenant) model"""
    __tablename__ = "households"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    organization_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    
    # Settings
    timezone: Mapped[str] = mapped_column(Text, server_default="'Asia/Kolkata'", nullable=False)
    locale: Mapped[str] = mapped_column(Text, server_default="'en-IN'")
    currency: Mapped[CurrencyCode] = mapped_column(SQLEnum(CurrencyCode), server_default="'INR'")
    
    # Metrics (denormalized)
    members_count: Mapped[int] = mapped_column(Integer, server_default="1", nullable=False)
    active_items_count: Mapped[int] = mapped_column(Integer, server_default="0")
    total_waste_value_cents: Mapped[int] = mapped_column(BigInteger, server_default="0")
    
    # Ownership
    created_by: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Metadata (renamed from 'metadata' to avoid SQLAlchemy conflict)
    meta_data: Mapped[dict] = mapped_column(
        JSONB,
        server_default=text('\'{"dietary_restrictions": [], "shopping_budget_monthly_cents": null, "favorite_stores": [], "waste_reduction_goal": "medium"}\'')
    )
    
    # Relationships
    members: Mapped[List["HouseholdUser"]] = relationship(
        "HouseholdUser",
        back_populates="household",
        cascade="all, delete-orphan"
    )
    pantry_entries: Mapped[List["PantryEntry"]] = relationship(
        "PantryEntry",
        back_populates="household",
        cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        CheckConstraint("members_count >= 0", name="households_members_count_check"),
        CheckConstraint("length(name) >= 1", name="households_name_length_check"),
        Index("idx_households_created_by", "created_by", postgresql_where=text("deleted_at IS NULL")),
    )
    
    def __repr__(self) -> str:
        return f"<Household(id={self.id}, name={self.name}, members={self.members_count})>"


class HouseholdUser(Base):
    """Many-to-many relationship between households and users"""
    __tablename__ = "household_users"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    household_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("households.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    role: Mapped[UserRole] = mapped_column(SQLEnum(UserRole), server_default="'member'", nullable=False)
    
    # Invitation tracking
    invited_by: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    invitation_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    invitation_sent_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    invitation_accepted_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    joined_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    left_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationships
    household: Mapped["Household"] = relationship("Household", back_populates="members")
    user: Mapped["User"] = relationship("User", back_populates="households", foreign_keys=[user_id])
    
    __table_args__ = (
        UniqueConstraint("household_id", "user_id", name="unique_household_user"),
        Index("idx_household_users_household", "household_id", postgresql_where=text("left_at IS NULL")),
        Index("idx_household_users_user", "user_id", postgresql_where=text("left_at IS NULL")),
    )


# ============================================================================
# ITEM CATALOG
# ============================================================================

class ItemCatalog(Base, TimestampMixin, SoftDeleteMixin):
    """Canonical item catalog"""
    __tablename__ = "items_catalog"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    # Identification
    barcode: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    brand: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Categorization
    category: Mapped[str] = mapped_column(Text, nullable=False)
    subcategory: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tags: Mapped[List[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    
    # Storage & shelf life
    typical_shelf_life_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    shelf_life_after_opening_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    storage_type: Mapped[StorageType] = mapped_column(SQLEnum(StorageType), server_default="'pantry'")
    
    # Measurement
    unit: Mapped[str] = mapped_column(Text, server_default="'units'", nullable=False)
    typical_package_size: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    
    # Nutrition
    nutrition: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    allergens: Mapped[List[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    
    # Pricing
    typical_price_cents: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    price_per_unit_cents: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    
    # Data source
    data_source: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data_quality_score: Mapped[Optional[float]] = mapped_column(Numeric(3, 2), nullable=True)
    external_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Search
    search_vector: Mapped[Optional[str]] = mapped_column(TSVECTOR, nullable=True)
    # embedding: Mapped[Optional[Any]] = mapped_column(Vector(384), nullable=True)  # Requires pgvector
    
    region_code: Mapped[str] = mapped_column(Text, server_default="'IN'")
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    # Relationships
    pantry_entries: Mapped[List["PantryEntry"]] = relationship("PantryEntry", back_populates="item_catalog")
    
    __table_args__ = (
        CheckConstraint("typical_shelf_life_days > 0", name="items_shelf_life_positive"),
        CheckConstraint("length(name) >= 1", name="items_name_length_check"),
        Index("idx_items_catalog_barcode", "barcode", unique=True, postgresql_where=text("barcode IS NOT NULL AND deleted_at IS NULL")),
        Index("idx_items_catalog_category", "category", "subcategory", postgresql_where=text("deleted_at IS NULL")),
        Index("idx_items_catalog_search_vector", "search_vector", postgresql_using="gin"),
    )
    
    def __repr__(self) -> str:
        return f"<ItemCatalog(id={self.id}, name={self.name}, barcode={self.barcode})>"


# ============================================================================
# PANTRY INVENTORY
# ============================================================================

class PantryEntry(Base, TimestampMixin, SoftDeleteMixin):
    """Per-batch pantry inventory tracking"""
    __tablename__ = "pantry_entries"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    # Ownership
    household_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("households.id", ondelete="CASCADE"), nullable=False)
    added_by: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    # Item
    item_catalog_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("items_catalog.id", ondelete="SET NULL"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_item: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Batch
    batch_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lot_number: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Quantity
    quantity: Mapped[float] = mapped_column(Numeric, nullable=False)
    original_quantity: Mapped[float] = mapped_column(Numeric, nullable=False)
    unit: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Dates
    purchase_date: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    opened_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    expiry_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    estimated_shelf_life_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    predicted_expiry_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    predicted_waste_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    # Storage
    storage_location: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    storage_type: Mapped[StorageType] = mapped_column(SQLEnum(StorageType), server_default="'pantry'")
    
    # Purchase
    store: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    price_cents: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    currency: Mapped[CurrencyCode] = mapped_column(SQLEnum(CurrencyCode), server_default="'INR'")
    
    # Status
    active: Mapped[bool] = mapped_column(Boolean, server_default="true")
    consumed_completely: Mapped[bool] = mapped_column(Boolean, server_default="false")
    wasted: Mapped[bool] = mapped_column(Boolean, server_default="false")
    
    # Photos
    photo_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    receipt_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    last_modified: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    deactivated_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    # Relationships
    household: Mapped["Household"] = relationship("Household", back_populates="pantry_entries")
    item_catalog: Mapped[Optional["ItemCatalog"]] = relationship("ItemCatalog", back_populates="pantry_entries")
    consumption_logs: Mapped[List["ConsumptionLog"]] = relationship("ConsumptionLog", back_populates="pantry_entry", cascade="all, delete-orphan")
    waste_events: Mapped[List["WasteEvent"]] = relationship("WasteEvent", back_populates="pantry_entry")
    predictions: Mapped[List["Prediction"]] = relationship("Prediction", back_populates="pantry_entry", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint("quantity > 0", name="pantry_quantity_positive"),
        CheckConstraint("quantity <= original_quantity", name="pantry_quantity_lte_original"),
        CheckConstraint("opened_at IS NULL OR opened_at >= purchase_date", name="pantry_opened_after_purchase"),
        Index("idx_pantry_household", "household_id", postgresql_where=text("active = true AND deleted_at IS NULL")),
        Index("idx_pantry_item", "item_catalog_id", postgresql_where=text("deleted_at IS NULL")),
        Index("idx_pantry_expiry", "household_id", "expiry_date", postgresql_where=text("active = true AND expiry_date IS NOT NULL")),
    )
    
    def __repr__(self) -> str:
        return f"<PantryEntry(id={self.id}, name={self.name}, quantity={self.quantity}, active={self.active})>"


# ============================================================================
# CONSUMPTION & WASTE TRACKING
# ============================================================================

class ConsumptionLog(Base):
    """Time-series consumption tracking"""
    __tablename__ = "consumption_logs"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    pantry_entry_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("pantry_entries.id", ondelete="CASCADE"), nullable=False)
    household_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("households.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    quantity: Mapped[float] = mapped_column(Numeric, nullable=False)
    unit: Mapped[str] = mapped_column(Text, nullable=False)
    
    recipe_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    recipe_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    timestamp: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    consumption_method: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    # Relationships
    pantry_entry: Mapped["PantryEntry"] = relationship("PantryEntry", back_populates="consumption_logs")
    
    __table_args__ = (
        CheckConstraint("quantity > 0", name="consumption_quantity_positive"),
        Index("idx_consumption_pantry", "pantry_entry_id"),
        Index("idx_consumption_household_time", "household_id", text("timestamp DESC")),
    )


class WasteEvent(Base):
    """Food waste event tracking"""
    __tablename__ = "waste_events"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    pantry_entry_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("pantry_entries.id", ondelete="SET NULL"))
    household_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("households.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    
    quantity: Mapped[float] = mapped_column(Numeric, nullable=False)
    unit: Mapped[str] = mapped_column(Text, nullable=False)
    
    reason: Mapped[WasteReason] = mapped_column(SQLEnum(WasteReason), nullable=False)
    reason_detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    photo_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    photo_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), server_default=text("'{}'"))
    
    estimated_value_cents: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    estimated_carbon_kg: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    waste_date: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    # Relationships
    pantry_entry: Mapped[Optional["PantryEntry"]] = relationship("PantryEntry", back_populates="waste_events")
    
    __table_args__ = (
        Index("idx_waste_household", "household_id"),
        Index("idx_waste_household_date", "household_id", text("waste_date DESC")),
    )


# ============================================================================
# ML PREDICTIONS
# ============================================================================

class Prediction(Base):
    """ML model predictions for waste risk"""
    __tablename__ = "predictions"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    pantry_entry_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("pantry_entries.id", ondelete="CASCADE"), nullable=False)
    household_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("households.id", ondelete="CASCADE"), nullable=False)
    
    model_version: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    
    risk_score: Mapped[float] = mapped_column(Numeric, nullable=False)
    risk_class: Mapped[RiskClass] = mapped_column(SQLEnum(RiskClass), nullable=False)
    
    predicted_waste_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    predicted_expiry_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    
    explanation: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    features_used: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    actual_outcome: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    outcome_date: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    # Relationships
    pantry_entry: Mapped["PantryEntry"] = relationship("PantryEntry", back_populates="predictions")
    
    __table_args__ = (
        CheckConstraint("risk_score BETWEEN 0 AND 1", name="predictions_risk_score_range"),
        Index("idx_predictions_pantry", "pantry_entry_id"),
        Index("idx_predictions_household", "household_id"),
        Index("idx_predictions_model", "model_name", "model_version"),
    )


# ============================================================================
# NOTIFICATIONS
# ============================================================================

class Notification(Base):
    """Multi-channel notifications"""
    __tablename__ = "notifications"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    household_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("households.id", ondelete="CASCADE"))
    user_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"))
    
    type: Mapped[NotificationType] = mapped_column(SQLEnum(NotificationType), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    
    payload: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    actions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    channels: Mapped[List[str]] = mapped_column(ARRAY(Text), server_default=text("'{push}'"))
    
    scheduled_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    sent_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    delivered: Mapped[bool] = mapped_column(Boolean, server_default="false")
    delivered_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    opened: Mapped[bool] = mapped_column(Boolean, server_default="false")
    opened_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    failed: Mapped[bool] = mapped_column(Boolean, server_default="false")
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, server_default="0")
    
    priority: Mapped[int] = mapped_column(Integer, server_default="0")
    
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    __table_args__ = (
        Index("idx_notifications_household", "household_id"),
        Index("idx_notifications_scheduled", "scheduled_at", postgresql_where=text("sent_at IS NULL")),
    )


# ============================================================================
# AUDIT LOGGING
# ============================================================================

class AuditLog(Base):
    """Audit trail for compliance"""
    __tablename__ = "audit_logs"
    
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, server_default=text("uuid_generate_v4()"))
    
    user_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"))
    household_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), ForeignKey("households.id", ondelete="SET NULL"))
    
    action: Mapped[str] = mapped_column(Text, nullable=False)
    resource_type: Mapped[str] = mapped_column(Text, nullable=False)
    resource_id: Mapped[Optional[UUID]] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    
    old_values: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    new_values: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    ip_address: Mapped[Optional[str]] = mapped_column(INET, nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    request_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    meta_data: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'"))
    
    __table_args__ = (
        Index("idx_audit_logs_user", "user_id", text("created_at DESC")),
        Index("idx_audit_logs_household", "household_id", text("created_at DESC")),
        Index("idx_audit_logs_resource", "resource_type", "resource_id"),
    )


# Export all models
__all__ = [
    "Base",
    "User",
    "UserSession",
    "Household",
    "HouseholdUser",
    "ItemCatalog",
    "PantryEntry",
    "ConsumptionLog",
    "WasteEvent",
    "Prediction",
    "Notification",
    "AuditLog",
    # Enums
    "UserRole",
    "StorageType",
    "WasteReason",
    "NotificationType",
    "RiskClass",
    "CurrencyCode",
]

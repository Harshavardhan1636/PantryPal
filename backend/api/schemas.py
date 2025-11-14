"""
Pydantic Schemas for API Request/Response Models
Type-safe data validation and serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict

from backend.shared.models_v2 import (
    UserRole,
    StorageType,
    WasteReason,
    NotificationType,
    RiskClass,
    CurrencyCode,
)


# ============================================================================
# Base Schemas
# ============================================================================

class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime
    updated_at: datetime


class SoftDeleteMixin(BaseModel):
    """Mixin for soft delete."""
    deleted_at: Optional[datetime] = None


# ============================================================================
# Authentication Schemas
# ============================================================================

class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    name: str = Field(..., min_length=1, max_length=100)
    
    @validator("password")
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            raise ValueError("Password must contain at least one special character")
        return v


class LoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserResponse(BaseModel):
    """User profile response."""
    id: UUID
    email: EmailStr
    name: str
    email_verified: bool
    is_active: bool
    is_admin: bool
    created_at: datetime
    preferences: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Household Schemas
# ============================================================================

class HouseholdCreate(BaseModel):
    """Create household request."""
    name: str = Field(..., min_length=1, max_length=100)
    timezone: str = Field(default="UTC", max_length=50)
    members_count: Optional[int] = Field(default=1, ge=1, le=100)
    currency: Optional[CurrencyCode] = CurrencyCode.USD
    settings: Optional[Dict[str, Any]] = None


class HouseholdUpdate(BaseModel):
    """Update household request."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    timezone: Optional[str] = Field(None, max_length=50)
    members_count: Optional[int] = Field(None, ge=1, le=100)
    currency: Optional[CurrencyCode] = None
    settings: Optional[Dict[str, Any]] = None


class HouseholdResponse(BaseModel):
    """Household response."""
    id: UUID
    name: str
    timezone: str
    members_count: int
    currency: CurrencyCode
    created_at: datetime
    settings: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class HouseholdUserResponse(BaseModel):
    """Household user membership response."""
    id: UUID
    household_id: UUID
    user_id: UUID
    role: UserRole
    joined_at: datetime
    user: Optional[UserResponse] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Pantry Schemas
# ============================================================================

class PantryEntryCreate(BaseModel):
    """Create pantry entry request."""
    household_id: UUID
    barcode: Optional[str] = Field(None, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., max_length=20)
    purchase_date: datetime
    expiry_date: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    storage_location: StorageType = StorageType.PANTRY
    purchase_price: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=500)


class PantryEntryUpdate(BaseModel):
    """Update pantry entry request."""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    quantity: Optional[float] = Field(None, gt=0)
    unit: Optional[str] = Field(None, max_length=20)
    expiry_date: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    storage_location: Optional[StorageType] = None
    notes: Optional[str] = Field(None, max_length=500)


class ItemCatalogResponse(BaseModel):
    """Canonical item from catalog."""
    id: UUID
    name: str
    barcode: Optional[str] = None
    category: Optional[str] = None
    brand: Optional[str] = None
    default_shelf_life_days: Optional[int] = None
    nutritional_info: Optional[Dict[str, Any]] = None
    
    model_config = ConfigDict(from_attributes=True)


class PredictionResponse(BaseModel):
    """Waste risk prediction."""
    id: UUID
    pantry_entry_id: UUID
    risk_score: float
    risk_class: RiskClass
    predicted_waste_date: Optional[datetime] = None
    confidence_score: Optional[float] = None
    recommended_actions: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class PantryEntryResponse(BaseModel):
    """Pantry entry with relationships."""
    id: UUID
    household_id: UUID
    canonical_item_id: Optional[UUID] = None
    name: str
    barcode: Optional[str] = None
    quantity: float
    unit: str
    purchase_date: datetime
    expiry_date: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    storage_location: StorageType
    purchase_price: Optional[float] = None
    notes: Optional[str] = None
    created_at: datetime
    canonical_item: Optional[ItemCatalogResponse] = None
    latest_prediction: Optional[PredictionResponse] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Purchase Schemas
# ============================================================================

class PurchaseResponse(BaseModel):
    """Purchase record response."""
    id: UUID
    household_id: UUID
    purchased_at: datetime
    total_amount: Optional[float] = None
    currency: CurrencyCode
    receipt_image_url: Optional[str] = None
    receipt_ocr_result: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Consumption Schemas
# ============================================================================

class ConsumptionLogCreate(BaseModel):
    """Log consumption request."""
    pantry_entry_id: UUID
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., max_length=20)
    recipe_id: Optional[UUID] = None
    notes: Optional[str] = Field(None, max_length=500)


class ConsumptionLogResponse(BaseModel):
    """Consumption log response."""
    id: UUID
    household_id: UUID
    pantry_entry_id: UUID
    quantity: float
    unit: str
    consumed_at: datetime
    recipe_id: Optional[UUID] = None
    notes: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Waste Schemas
# ============================================================================

class WasteEventCreate(BaseModel):
    """Log waste event request."""
    pantry_entry_id: UUID
    quantity: float = Field(..., gt=0)
    unit: str = Field(..., max_length=20)
    reason: WasteReason
    photo_url: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=500)


class WasteEventResponse(BaseModel):
    """Waste event response."""
    id: UUID
    household_id: UUID
    pantry_entry_id: UUID
    quantity: float
    unit: str
    reason: WasteReason
    wasted_at: datetime
    estimated_cost: Optional[float] = None
    carbon_footprint_kg: Optional[float] = None
    photo_url: Optional[str] = None
    notes: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Prediction Schemas
# ============================================================================

class PredictionListResponse(BaseModel):
    """List of predictions with risk ranking."""
    predictions: List[PredictionResponse]
    total: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


# ============================================================================
# Recipe Schemas
# ============================================================================

class RecipeCreate(BaseModel):
    """Create recipe request."""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    ingredients: List[str] = Field(..., min_items=1)
    instructions: Optional[str] = None
    servings: Optional[int] = Field(None, ge=1)
    prep_time_minutes: Optional[int] = Field(None, ge=0)
    cook_time_minutes: Optional[int] = Field(None, ge=0)
    cuisine_type: Optional[str] = Field(None, max_length=50)
    dietary_tags: Optional[List[str]] = None
    source_url: Optional[str] = None


class RecipeResponse(BaseModel):
    """Recipe response."""
    id: UUID
    name: str
    description: Optional[str] = None
    ingredients: List[str]
    instructions: Optional[str] = None
    servings: Optional[int] = None
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    cuisine_type: Optional[str] = None
    dietary_tags: Optional[List[str]] = None
    source_url: Optional[str] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class RecipeBulkImport(BaseModel):
    """Bulk import recipes request."""
    recipes: List[RecipeCreate] = Field(..., min_items=1, max_items=1000)


# ============================================================================
# Notification Schemas
# ============================================================================

class NotificationScheduleCreate(BaseModel):
    """Schedule notification request."""
    household_id: UUID
    notification_type: NotificationType
    payload: Dict[str, Any]
    scheduled_at: datetime
    channels: Optional[List[str]] = None  # ["push", "sms", "email"]


class NotificationResponse(BaseModel):
    """Notification response."""
    id: UUID
    household_id: UUID
    notification_type: NotificationType
    message: str
    payload: Optional[Dict[str, Any]] = None
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Admin Schemas
# ============================================================================

class ModelDeployRequest(BaseModel):
    """Deploy ML model request."""
    model_artifact_uri: str = Field(..., min_length=1)
    metadata: Dict[str, Any]
    model_type: str = Field(..., max_length=50)  # "prediction", "recommendation", etc.
    version: str = Field(..., max_length=20)


class ModelResponse(BaseModel):
    """ML model response."""
    id: UUID
    model_type: str
    version: str
    artifact_uri: str
    metadata: Dict[str, Any]
    deployed_at: datetime
    is_active: bool


# ============================================================================
# Metrics Schemas
# ============================================================================

class HouseholdMetricsResponse(BaseModel):
    """Household metrics response."""
    household_id: UUID
    period_start: datetime
    period_end: datetime
    waste_estimate: Dict[str, Any] = Field(
        description="Waste statistics (count, weight, cost, carbon footprint)"
    )
    cost_saved: float = Field(description="Estimated cost saved by reducing waste")
    engagement: Dict[str, Any] = Field(
        description="User engagement metrics (active users, pantry entries, consumptions)"
    )
    top_wasted_items: List[Dict[str, Any]] = Field(
        description="Items with highest waste frequency"
    )
    waste_reasons_breakdown: Dict[str, int] = Field(
        description="Breakdown of waste reasons"
    )
    prediction_accuracy: Optional[float] = Field(
        None, description="ML prediction accuracy (if available)"
    )


# ============================================================================
# Common Response Schemas
# ============================================================================

class MessageResponse(BaseModel):
    """Generic message response."""
    message: str
    success: bool = True


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    message: str
    details: Optional[Any] = None
    timestamp: datetime


class PaginatedResponse(BaseModel):
    """Paginated list response."""
    items: List[Any]
    total: int
    skip: int
    limit: int
    has_more: bool

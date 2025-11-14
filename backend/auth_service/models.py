"""Authentication service database models."""
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Boolean, ForeignKey, Integer, JSON, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from shared.database import Base, TimestampMixin, SoftDeleteMixin


class Organization(Base, TimestampMixin, SoftDeleteMixin):
    """Organization/company account."""
    
    __tablename__ = "organizations"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    plan: Mapped[str] = mapped_column(String(50), nullable=False, default="free")
    max_households: Mapped[int] = mapped_column(Integer, default=1)
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Relationships
    households = relationship("Household", back_populates="organization")


class Household(Base, TimestampMixin, SoftDeleteMixin):
    """Household/family unit within an organization."""
    
    __tablename__ = "households"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    org_id: Mapped[UUID] = mapped_column(ForeignKey("organizations.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    members_count: Mapped[int] = mapped_column(Integer, default=1)
    storage_types: Mapped[list] = mapped_column(JSON, default=lambda: ["refrigerator", "pantry", "freezer"])
    dietary_preferences: Mapped[list] = mapped_column(JSON, default=list)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Relationships
    organization = relationship("Organization", back_populates="households")
    members = relationship("HouseholdMember", back_populates="household")


class User(Base, TimestampMixin, SoftDeleteMixin):
    """User account."""
    
    __tablename__ = "users"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    auth_provider: Mapped[str] = mapped_column(String(50), default="local")
    auth_provider_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    phone_number: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    phone_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    avatar_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    locale: Mapped[str] = mapped_column(String(10), default="en-US")
    last_login_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
    # Relationships
    households = relationship("HouseholdMember", back_populates="user")
    
    __table_args__ = (
        UniqueConstraint("auth_provider", "auth_provider_id", name="unique_auth_provider"),
    )


class HouseholdMember(Base):
    """Association between users and households with roles."""
    
    __tablename__ = "household_members"
    
    household_id: Mapped[UUID] = mapped_column(ForeignKey("households.id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role: Mapped[str] = mapped_column(String(50), default="member")
    permissions: Mapped[list] = mapped_column(JSON, default=list)
    joined_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    left_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    
    # Relationships
    household = relationship("Household", back_populates="members")
    user = relationship("User", back_populates="households")

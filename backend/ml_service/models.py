"""ML service database models."""
from datetime import datetime, date
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Numeric, ForeignKey, JSON, Date, CheckConstraint
from sqlalchemy.orm import Mapped, mapped_column

from shared.database import Base


class RiskPrediction(Base):
    """ML waste risk predictions (time-series)."""
    
    __tablename__ = "risk_predictions"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    batch_id: Mapped[UUID] = mapped_column(ForeignKey("inventory_batches.id", ondelete="CASCADE"))
    household_id: Mapped[UUID] = mapped_column(ForeignKey("households.id", ondelete="CASCADE"))
    risk_score: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    risk_class: Mapped[Optional[str]] = mapped_column(String(20))
    predicted_waste_date: Mapped[Optional[date]] = mapped_column(Date)
    confidence: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 4))
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    features: Mapped[dict] = mapped_column(JSON, default=dict)
    predicted_at: Mapped[datetime] = mapped_column(nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        CheckConstraint("risk_score >= 0 AND risk_score <= 1", name="valid_risk_score"),
    )


class ModelExperiment(Base):
    """ML model training experiments tracking."""
    
    __tablename__ = "model_experiments"
    
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[Optional[str]]
    hyperparameters: Mapped[dict] = mapped_column(JSON, default=dict)
    training_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    validation_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    test_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    deployed_at: Mapped[Optional[datetime]]
    deprecated_at: Mapped[Optional[datetime]]
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

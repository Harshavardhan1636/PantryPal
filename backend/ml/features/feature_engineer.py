"""
Feature Engineering Pipeline for Waste Risk Prediction.

Extracts and engineers features from raw data for ML model training and inference.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ...shared.models_v2 import (
    PantryEntry,
    ItemsCatalog,
    ConsumptionLog,
    WasteEvent,
    Household,
    HouseholdUser,
)
from ..config import ENGINEERED_FEATURES, FEATURE_LOOKBACK_WINDOWS


logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Engineered feature vector for a single pantry entry."""
    pantry_entry_id: str
    features: Dict[str, float]
    feature_names: List[str]
    metadata: Dict[str, Any]


class FeatureEngineer:
    """Feature engineering for waste prediction models."""
    
    def __init__(self, db_session: AsyncSession):
        """
        Initialize feature engineer.
        
        Args:
            db_session: Database session for data extraction
        """
        self.db = db_session
        self.feature_names = ENGINEERED_FEATURES
        
        logger.info(f"Initialized FeatureEngineer with {len(self.feature_names)} features")
    
    async def extract_features(
        self,
        pantry_entry_id: str,
        reference_date: Optional[datetime] = None,
    ) -> FeatureVector:
        """
        Extract all features for a single pantry entry.
        
        Args:
            pantry_entry_id: Pantry entry UUID
            reference_date: Date for feature extraction (default: now)
            
        Returns:
            FeatureVector with all engineered features
        """
        if reference_date is None:
            reference_date = datetime.utcnow()
        
        logger.debug(f"Extracting features for {pantry_entry_id}")
        
        # Fetch pantry entry with related data
        query = select(PantryEntry).where(
            PantryEntry.entry_id == pantry_entry_id
        )
        result = await self.db.execute(query)
        entry = result.scalar_one_or_none()
        
        if not entry:
            raise ValueError(f"Pantry entry {pantry_entry_id} not found")
        
        # Fetch item catalog
        item_query = select(ItemsCatalog).where(
            ItemsCatalog.item_id == entry.item_id
        )
        item_result = await self.db.execute(item_query)
        item = item_result.scalar_one_or_none()
        
        # Fetch household info
        household_query = select(Household).where(
            Household.household_id == entry.household_id
        )
        household_result = await self.db.execute(household_query)
        household = household_result.scalar_one_or_none()
        
        # Extract features
        features = {}
        
        # ===== Temporal Features =====
        features["age_days"] = (reference_date - entry.purchase_date).days
        
        if entry.expiry_date:
            features["days_to_expiry"] = (entry.expiry_date - reference_date).days
        elif item and item.typical_shelf_life_days:
            expected_expiry = entry.purchase_date + timedelta(days=item.typical_shelf_life_days)
            features["days_to_expiry"] = (expected_expiry - reference_date).days
        else:
            features["days_to_expiry"] = -999  # Unknown
        
        features["days_since_opened"] = (
            (reference_date - entry.opened_at).days
            if entry.opened_at else 0
        )
        
        features["purchase_day_of_week"] = entry.purchase_date.weekday()
        features["purchase_week_of_year"] = entry.purchase_date.isocalendar()[1]
        
        # ===== Quantity Features =====
        features["quantity_on_hand"] = float(entry.quantity_on_hand)
        features["initial_quantity"] = float(entry.quantity)
        
        if entry.quantity > 0:
            features["quantity_consumed_ratio"] = (
                (entry.quantity - entry.quantity_on_hand) / entry.quantity
            )
        else:
            features["quantity_consumed_ratio"] = 0.0
        
        # ===== Consumption Pattern Features =====
        consumption_features = await self._extract_consumption_features(
            entry,
            reference_date,
        )
        features.update(consumption_features)
        
        # ===== Household Features =====
        if household:
            # Count household members
            member_query = select(func.count(HouseholdUser.user_id)).where(
                HouseholdUser.household_id == household.household_id
            )
            member_result = await self.db.execute(member_query)
            household_size = member_result.scalar() or 1
            features["household_size"] = float(household_size)
        else:
            features["household_size"] = 1.0
        
        # Household turnover rate
        household_turnover = await self._calculate_household_turnover(
            entry.household_id,
            reference_date,
        )
        features["household_pantry_turnover_rate"] = household_turnover
        
        # Historical waste rate
        waste_rate = await self._calculate_household_waste_rate(
            entry.household_id,
            reference_date,
        )
        features["household_waste_rate_historical"] = waste_rate
        
        # ===== Item Features =====
        features["storage_type"] = self._encode_storage_type(entry.storage_location)
        features["is_opened"] = float(entry.opened_at is not None)
        features["price_per_unit"] = float(entry.price_per_unit or 0.0)
        features["is_bulk_purchase"] = float(entry.quantity >= 5.0)  # Threshold
        
        if item:
            features["typical_shelf_life_days"] = float(
                item.typical_shelf_life_days or 7
            )
        else:
            features["typical_shelf_life_days"] = 7.0
        
        # ===== Category Features =====
        if item:
            category_features = await self._extract_category_features(
                item.category,
                reference_date,
            )
            features.update(category_features)
        else:
            features["category_waste_rate"] = 0.5
            features["category_turnover_rate"] = 0.5
            features["category_seasonality_score"] = 0.0
        
        # ===== Interaction Features =====
        features["family_size_x_quantity"] = (
            features["household_size"] * features["quantity_on_hand"]
        )
        features["price_x_quantity"] = (
            features["price_per_unit"] * features["quantity_on_hand"]
        )
        
        if features["days_to_expiry"] > 0:
            features["age_x_expiry_ratio"] = (
                features["age_days"] / features["days_to_expiry"]
            )
        else:
            features["age_x_expiry_ratio"] = 999.0  # Very high risk
        
        # ===== External Features =====
        features["is_holiday_season"] = self._is_holiday_season(reference_date)
        features["day_of_week_encoded"] = reference_date.weekday() / 6.0  # Normalize
        features["promotion_flag"] = 0.0  # TODO: Integrate with promotions data
        
        # Recommendation response rate (TODO: track user actions)
        features["recommendation_response_rate"] = 0.5  # Default
        
        # Fill missing features with defaults
        for feature_name in self.feature_names:
            if feature_name not in features:
                features[feature_name] = 0.0
        
        return FeatureVector(
            pantry_entry_id=pantry_entry_id,
            features=features,
            feature_names=self.feature_names,
            metadata={
                "reference_date": reference_date,
                "household_id": str(entry.household_id),
                "item_id": str(entry.item_id),
            },
        )
    
    async def _extract_consumption_features(
        self,
        entry: PantryEntry,
        reference_date: datetime,
    ) -> Dict[str, float]:
        """Extract consumption pattern features."""
        features = {}
        
        # Calculate average daily consumption for different windows
        for window_days in FEATURE_LOOKBACK_WINDOWS:
            start_date = reference_date - timedelta(days=window_days)
            
            # Query consumption logs
            query = select(func.sum(ConsumptionLog.quantity_consumed)).where(
                ConsumptionLog.pantry_entry_id == entry.entry_id,
                ConsumptionLog.consumed_at >= start_date,
                ConsumptionLog.consumed_at <= reference_date,
            )
            result = await self.db.execute(query)
            total_consumed = result.scalar() or 0.0
            
            avg_daily = total_consumed / window_days if window_days > 0 else 0.0
            features[f"avg_daily_consumption_{window_days}d"] = float(avg_daily)
        
        # Consumption velocity (acceleration)
        if len(FEATURE_LOOKBACK_WINDOWS) >= 2:
            short_window = FEATURE_LOOKBACK_WINDOWS[0]
            long_window = FEATURE_LOOKBACK_WINDOWS[-1]
            
            velocity = (
                features[f"avg_daily_consumption_{short_window}d"] -
                features[f"avg_daily_consumption_{long_window}d"]
            )
            features["consumption_velocity"] = velocity
        else:
            features["consumption_velocity"] = 0.0
        
        # Cohort average consumption (from similar items)
        cohort_avg = await self._get_cohort_consumption(
            entry.item_id,
            entry.household_id,
            reference_date,
        )
        features["cohort_avg_consumption"] = cohort_avg
        
        return features
    
    async def _get_cohort_consumption(
        self,
        item_id: str,
        household_id: str,
        reference_date: datetime,
    ) -> float:
        """Get average consumption rate for similar items in cohort."""
        window_days = 30
        start_date = reference_date - timedelta(days=window_days)
        
        # Query consumption for same item across all households
        query = select(func.avg(ConsumptionLog.quantity_consumed)).where(
            ConsumptionLog.consumed_at >= start_date,
            ConsumptionLog.consumed_at <= reference_date,
        ).join(PantryEntry).where(
            PantryEntry.item_id == item_id,
            PantryEntry.household_id != household_id,  # Exclude same household
        )
        
        result = await self.db.execute(query)
        cohort_avg = result.scalar() or 0.0
        
        return float(cohort_avg)
    
    async def _calculate_household_turnover(
        self,
        household_id: str,
        reference_date: datetime,
    ) -> float:
        """
        Calculate pantry turnover rate for household.
        
        Turnover = (items consumed + wasted) / total items purchased
        """
        window_days = 30
        start_date = reference_date - timedelta(days=window_days)
        
        # Count consumed items
        consumed_query = select(func.count(ConsumptionLog.log_id)).where(
            ConsumptionLog.consumed_at >= start_date,
        ).join(PantryEntry).where(
            PantryEntry.household_id == household_id,
        )
        consumed_result = await self.db.execute(consumed_query)
        consumed_count = consumed_result.scalar() or 0
        
        # Count wasted items
        wasted_query = select(func.count(WasteEvent.event_id)).where(
            WasteEvent.wasted_at >= start_date,
        ).join(PantryEntry).where(
            PantryEntry.household_id == household_id,
        )
        wasted_result = await self.db.execute(wasted_query)
        wasted_count = wasted_result.scalar() or 0
        
        # Count purchased items
        purchased_query = select(func.count(PantryEntry.entry_id)).where(
            PantryEntry.household_id == household_id,
            PantryEntry.purchase_date >= start_date,
        )
        purchased_result = await self.db.execute(purchased_query)
        purchased_count = purchased_result.scalar() or 1  # Avoid division by zero
        
        turnover = (consumed_count + wasted_count) / purchased_count
        return float(turnover)
    
    async def _calculate_household_waste_rate(
        self,
        household_id: str,
        reference_date: datetime,
    ) -> float:
        """Calculate historical waste rate for household."""
        window_days = 90  # 3 months
        start_date = reference_date - timedelta(days=window_days)
        
        # Count wasted items
        wasted_query = select(func.count(WasteEvent.event_id)).where(
            WasteEvent.wasted_at >= start_date,
        ).join(PantryEntry).where(
            PantryEntry.household_id == household_id,
        )
        wasted_result = await self.db.execute(wasted_query)
        wasted_count = wasted_result.scalar() or 0
        
        # Count total items (consumed + wasted)
        consumed_query = select(func.count(ConsumptionLog.log_id)).where(
            ConsumptionLog.consumed_at >= start_date,
        ).join(PantryEntry).where(
            PantryEntry.household_id == household_id,
        )
        consumed_result = await self.db.execute(consumed_query)
        consumed_count = consumed_result.scalar() or 0
        
        total_items = consumed_count + wasted_count
        if total_items == 0:
            return 0.5  # Default
        
        waste_rate = wasted_count / total_items
        return float(waste_rate)
    
    async def _extract_category_features(
        self,
        category: str,
        reference_date: datetime,
    ) -> Dict[str, float]:
        """Extract category-level aggregated features."""
        features = {}
        window_days = 90
        start_date = reference_date - timedelta(days=window_days)
        
        # Category waste rate
        wasted_query = select(func.count(WasteEvent.event_id)).join(
            PantryEntry
        ).join(ItemsCatalog).where(
            ItemsCatalog.category == category,
            WasteEvent.wasted_at >= start_date,
        )
        wasted_result = await self.db.execute(wasted_query)
        wasted_count = wasted_result.scalar() or 0
        
        consumed_query = select(func.count(ConsumptionLog.log_id)).join(
            PantryEntry
        ).join(ItemsCatalog).where(
            ItemsCatalog.category == category,
            ConsumptionLog.consumed_at >= start_date,
        )
        consumed_result = await self.db.execute(consumed_query)
        consumed_count = consumed_result.scalar() or 0
        
        total_items = consumed_count + wasted_count
        if total_items > 0:
            features["category_waste_rate"] = float(wasted_count / total_items)
            features["category_turnover_rate"] = float(total_items / window_days)
        else:
            features["category_waste_rate"] = 0.5
            features["category_turnover_rate"] = 0.1
        
        # Seasonality score (simple heuristic based on month)
        month = reference_date.month
        # TODO: Implement proper seasonality modeling
        features["category_seasonality_score"] = 0.0
        
        return features
    
    def _encode_storage_type(self, storage_location: Optional[str]) -> float:
        """Encode storage type as numerical feature."""
        storage_mapping = {
            "refrigerator": 0.0,
            "freezer": 0.25,
            "pantry": 0.5,
            "counter": 0.75,
            "other": 1.0,
        }
        
        if not storage_location:
            return 0.5  # Default
        
        return storage_mapping.get(storage_location.lower(), 0.5)
    
    def _is_holiday_season(self, date: datetime) -> float:
        """Check if date falls in holiday season (Nov-Dec)."""
        return float(date.month in [11, 12])
    
    async def extract_batch_features(
        self,
        pantry_entry_ids: List[str],
        reference_date: Optional[datetime] = None,
    ) -> List[FeatureVector]:
        """Extract features for multiple entries efficiently."""
        features = []
        for entry_id in pantry_entry_ids:
            try:
                fv = await self.extract_features(entry_id, reference_date)
                features.append(fv)
            except Exception as e:
                logger.error(f"Feature extraction failed for {entry_id}: {e}")
        
        return features
    
    def to_dataframe(self, feature_vectors: List[FeatureVector]) -> pd.DataFrame:
        """Convert feature vectors to pandas DataFrame."""
        records = []
        for fv in feature_vectors:
            record = {"pantry_entry_id": fv.pantry_entry_id}
            record.update(fv.features)
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Ensure all feature columns are present
        for feature_name in self.feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0.0
        
        return df[["pantry_entry_id"] + self.feature_names]

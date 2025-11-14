"""
Label Schema and Data Quality Validation

Defines:
- Waste reason taxonomy
- Product category taxonomy
- Data validation rules
- Quality check functions
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# Waste Reason Taxonomy
# ============================================================================

class WasteReason(str, Enum):
    """
    Standardized waste reason taxonomy.
    
    Categories:
    - spoilage: Expired, moldy, rotten, or spoiled
    - overcooked: Burned, overcooked, dried out, or ruined during cooking
    - portion: Made too much, didn't finish, leftovers not eaten
    - packaging: Opened too early, improper storage, damaged packaging
    - other: User-specified reason not fitting above categories
    """
    
    SPOILAGE = "spoilage"
    OVERCOOKED = "overcooked"
    PORTION = "portion"
    PACKAGING = "packaging"
    OTHER = "other"
    
    @classmethod
    def get_description(cls, reason: str) -> str:
        """Get human-readable description."""
        descriptions = {
            cls.SPOILAGE: "Item expired, molded, or spoiled before use",
            cls.OVERCOOKED: "Item was burned or ruined during cooking",
            cls.PORTION: "Made too much or leftovers not eaten",
            cls.PACKAGING: "Packaging issues or improper storage",
            cls.OTHER: "Other reason for waste",
        }
        return descriptions.get(reason, "Unknown reason")
    
    @classmethod
    def get_prevention_tip(cls, reason: str) -> str:
        """Get prevention tip for waste reason."""
        tips = {
            cls.SPOILAGE: "Store properly and use before expiration. Check regularly.",
            cls.OVERCOOKED: "Use timers and follow recipes carefully.",
            cls.PORTION: "Plan portions better and store leftovers promptly.",
            cls.PACKAGING: "Reseal packages properly and transfer to airtight containers.",
            cls.OTHER: "Review your waste patterns for improvement opportunities.",
        }
        return tips.get(reason, "")


# ============================================================================
# Product Category Taxonomy
# ============================================================================

class ProductCategory(str, Enum):
    """Standardized product categories."""
    
    DAIRY = "dairy"
    PRODUCE = "produce"
    MEAT = "meat"
    SEAFOOD = "seafood"
    BAKERY = "bakery"
    PANTRY = "pantry"
    FROZEN = "frozen"
    BEVERAGES = "beverages"
    SNACKS = "snacks"
    CONDIMENTS = "condiments"
    OTHER = "other"


class StorageLocation(str, Enum):
    """Storage location taxonomy."""
    
    REFRIGERATOR = "refrigerator"
    FREEZER = "freezer"
    PANTRY = "pantry"
    COUNTER = "counter"
    CABINET = "cabinet"


# ============================================================================
# Unit Standards
# ============================================================================

class UnitType(str, Enum):
    """Standardized unit types."""
    
    # Weight
    POUND = "pound"
    OUNCE = "ounce"
    GRAM = "gram"
    KILOGRAM = "kilogram"
    
    # Volume
    GALLON = "gallon"
    QUART = "quart"
    PINT = "pint"
    CUP = "cup"
    FLUID_OUNCE = "fluid_ounce"
    MILLILITER = "milliliter"
    LITER = "liter"
    
    # Count
    UNIT = "unit"
    PACKAGE = "package"
    BUNCH = "bunch"
    DOZEN = "dozen"


# Canonical unit mappings
UNIT_ALIASES = {
    # Weight
    "lb": UnitType.POUND,
    "lbs": UnitType.POUND,
    "oz": UnitType.OUNCE,
    "g": UnitType.GRAM,
    "kg": UnitType.KILOGRAM,
    
    # Volume
    "gal": UnitType.GALLON,
    "qt": UnitType.QUART,
    "pt": UnitType.PINT,
    "c": UnitType.CUP,
    "fl oz": UnitType.FLUID_OUNCE,
    "ml": UnitType.MILLILITER,
    "l": UnitType.LITER,
    
    # Count
    "ea": UnitType.UNIT,
    "each": UnitType.UNIT,
    "pkg": UnitType.PACKAGE,
    "dz": UnitType.DOZEN,
}


# ============================================================================
# Validation Models
# ============================================================================

class PantryItemValidation(BaseModel):
    """Validation schema for pantry items."""
    
    name: str = Field(..., min_length=1, max_length=200)
    category: ProductCategory
    quantity: float = Field(..., gt=0, lt=10000)
    unit: UnitType
    expiration_date: Optional[datetime] = None
    purchase_date: Optional[datetime] = None
    storage_location: Optional[StorageLocation] = None
    
    @validator('quantity')
    def validate_quantity_range(cls, v, values):
        """Validate quantity is in reasonable range."""
        if v < 0.01:
            raise ValueError('Quantity too small (min 0.01)')
        if v > 1000:
            raise ValueError('Quantity too large (max 1000)')
        return v
    
    @validator('expiration_date')
    def validate_expiration_date(cls, v, values):
        """Validate expiration date is in the future."""
        if v and v < datetime.utcnow():
            raise ValueError('Expiration date must be in the future')
        
        # Warn if expiration is too far in the future (>5 years)
        if v and (v - datetime.utcnow()).days > 1825:
            raise ValueError('Expiration date too far in future (>5 years)')
        
        return v
    
    @validator('purchase_date')
    def validate_purchase_date(cls, v, values):
        """Validate purchase date is not in the future."""
        if v and v > datetime.utcnow():
            raise ValueError('Purchase date cannot be in the future')
        
        # Warn if purchase date is too old (>1 year)
        if v and (datetime.utcnow() - v).days > 365:
            raise ValueError('Purchase date too old (>1 year)')
        
        return v


class WastePredictionValidation(BaseModel):
    """Validation schema for waste predictions."""
    
    waste_risk_score: float = Field(..., ge=0, le=1)
    days_until_waste: int = Field(..., ge=0)
    waste_reason: WasteReason
    confidence_score: float = Field(..., ge=0, le=1)
    
    @validator('days_until_waste')
    def validate_days_until_waste(cls, v):
        """Validate days until waste is reasonable."""
        if v < 0:
            raise ValueError('Days until waste cannot be negative')
        if v > 365:
            raise ValueError('Days until waste too large (max 365)')
        return v


# ============================================================================
# Data Quality Rules
# ============================================================================

class DataQualityRule(BaseModel):
    """Base class for data quality rules."""
    
    name: str
    description: str
    severity: str  # "critical", "warning", "info"


class QualityCheckResult(BaseModel):
    """Result of a quality check."""
    
    rule_name: str
    passed: bool
    severity: str
    message: str
    affected_count: int = 0


# ============================================================================
# Quality Check Functions
# ============================================================================

def check_missing_expiration_dates(df) -> QualityCheckResult:
    """Check for missing expiration dates in perishable items."""
    
    perishable_categories = [
        ProductCategory.DAIRY,
        ProductCategory.PRODUCE,
        ProductCategory.MEAT,
        ProductCategory.SEAFOOD,
    ]
    
    perishable_items = df[df['category'].isin(perishable_categories)]
    missing_expiration = perishable_items['expiration_date'].isnull().sum()
    
    passed = missing_expiration == 0
    
    return QualityCheckResult(
        rule_name="missing_expiration_dates",
        passed=passed,
        severity="warning",
        message=f"Found {missing_expiration} perishable items without expiration dates",
        affected_count=missing_expiration,
    )


def check_negative_quantities(df) -> QualityCheckResult:
    """Check for negative quantities."""
    
    negative_count = (df['quantity'] < 0).sum()
    
    passed = negative_count == 0
    
    return QualityCheckResult(
        rule_name="negative_quantities",
        passed=passed,
        severity="critical",
        message=f"Found {negative_count} items with negative quantities",
        affected_count=negative_count,
    )


def check_duplicate_entries(df, window_hours: int = 24) -> QualityCheckResult:
    """Check for duplicate entries within time window."""
    
    # Group by name, household, and time window
    df['time_window'] = df['created_at'].dt.floor(f'{window_hours}H')
    
    duplicates = df.groupby(['name', 'household_id', 'time_window']).size()
    duplicate_count = (duplicates > 1).sum()
    
    passed = duplicate_count == 0
    
    return QualityCheckResult(
        rule_name="duplicate_entries",
        passed=passed,
        severity="warning",
        message=f"Found {duplicate_count} potential duplicate entries within {window_hours}h",
        affected_count=duplicate_count,
    )


def check_unit_consistency(df) -> QualityCheckResult:
    """Check for unit consistency by product category."""
    
    # Volume units should be used with liquids
    volume_units = [UnitType.GALLON, UnitType.QUART, UnitType.CUP, UnitType.MILLILITER, UnitType.LITER]
    weight_units = [UnitType.POUND, UnitType.OUNCE, UnitType.GRAM, UnitType.KILOGRAM]
    
    # Check beverages with weight units
    beverages_with_weight = df[
        (df['category'] == ProductCategory.BEVERAGES) &
        (df['unit'].isin(weight_units))
    ]
    
    # Check produce with volume units (except some liquids like juice)
    produce_with_volume = df[
        (df['category'] == ProductCategory.PRODUCE) &
        (df['unit'].isin(volume_units))
    ]
    
    inconsistent_count = len(beverages_with_weight) + len(produce_with_volume)
    
    passed = inconsistent_count == 0
    
    return QualityCheckResult(
        rule_name="unit_consistency",
        passed=passed,
        severity="warning",
        message=f"Found {inconsistent_count} items with inconsistent units for category",
        affected_count=inconsistent_count,
    )


def check_extreme_quantities(df) -> QualityCheckResult:
    """Check for extreme quantity outliers."""
    
    # Define category-specific reasonable ranges
    category_ranges = {
        ProductCategory.DAIRY: (0.1, 20),      # 0.1 to 20 lbs
        ProductCategory.PRODUCE: (0.1, 50),    # 0.1 to 50 lbs
        ProductCategory.MEAT: (0.1, 20),       # 0.1 to 20 lbs
        ProductCategory.BEVERAGES: (0.1, 10),  # 0.1 to 10 gallons
    }
    
    outliers = 0
    for category, (min_qty, max_qty) in category_ranges.items():
        category_items = df[df['category'] == category]
        outliers += ((category_items['quantity'] < min_qty) | (category_items['quantity'] > max_qty)).sum()
    
    passed = outliers == 0
    
    return QualityCheckResult(
        rule_name="extreme_quantities",
        passed=passed,
        severity="warning",
        message=f"Found {outliers} items with extreme quantities outside typical ranges",
        affected_count=outliers,
    )


def check_low_confidence_predictions(df, threshold: float = 0.3) -> QualityCheckResult:
    """Check for very low confidence predictions that should be dropped."""
    
    low_confidence = (df['confidence_score'] < threshold).sum()
    
    passed = low_confidence == 0
    
    return QualityCheckResult(
        rule_name="low_confidence_predictions",
        passed=passed,
        severity="info",
        message=f"Found {low_confidence} predictions with confidence <{threshold} (consider dropping)",
        affected_count=low_confidence,
    )


def check_invalid_waste_reasons(df) -> QualityCheckResult:
    """Check for invalid waste reasons."""
    
    valid_reasons = [reason.value for reason in WasteReason]
    invalid_reasons = ~df['waste_reason'].isin(valid_reasons + [None])
    invalid_count = invalid_reasons.sum()
    
    passed = invalid_count == 0
    
    return QualityCheckResult(
        rule_name="invalid_waste_reasons",
        passed=passed,
        severity="critical",
        message=f"Found {invalid_count} items with invalid waste reasons",
        affected_count=invalid_count,
    )


# ============================================================================
# Run All Quality Checks
# ============================================================================

def run_all_quality_checks(df) -> List[QualityCheckResult]:
    """
    Run all data quality checks.
    
    Args:
        df: DataFrame to check
    
    Returns:
        List of quality check results
    """
    
    checks = [
        check_missing_expiration_dates(df),
        check_negative_quantities(df),
        check_duplicate_entries(df),
        check_unit_consistency(df),
        check_extreme_quantities(df),
        check_low_confidence_predictions(df),
        check_invalid_waste_reasons(df),
    ]
    
    return checks


def print_quality_report(results: List[QualityCheckResult]):
    """Print formatted quality check report."""
    
    print("="*80)
    print("Data Quality Report")
    print("="*80)
    
    critical_failed = [r for r in results if r.severity == "critical" and not r.passed]
    warning_failed = [r for r in results if r.severity == "warning" and not r.passed]
    info_failed = [r for r in results if r.severity == "info" and not r.passed]
    
    if critical_failed:
        print("\n❌ CRITICAL ISSUES:")
        for result in critical_failed:
            print(f"  - {result.message}")
    
    if warning_failed:
        print("\n⚠️  WARNINGS:")
        for result in warning_failed:
            print(f"  - {result.message}")
    
    if info_failed:
        print("\n ℹ️  INFO:")
        for result in info_failed:
            print(f"  - {result.message}")
    
    if not (critical_failed or warning_failed or info_failed):
        print("\n✅ All quality checks passed!")
    
    print("\n" + "="*80)

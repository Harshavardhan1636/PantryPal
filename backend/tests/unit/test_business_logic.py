"""
Unit Tests for PantryPal Business Logic

Tests:
- Canonicalization functions
- Unit conversion utilities
- Date/time calculations
- Waste prediction logic
- Recipe recommendation scoring
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from backend.shared.utils.canonicalization import (
    canonicalize_product_name,
    match_product_to_database,
    extract_quantity_from_text
)
from backend.shared.utils.unit_conversion import (
    convert_units,
    normalize_unit,
    parse_quantity_with_unit
)
from backend.shared.utils.datetime_utils import (
    calculate_days_until_expiration,
    is_expired,
    calculate_consumption_rate
)


# ============================================================================
# Canonicalization Tests
# ============================================================================

class TestCanonicaliz:
    """Test product name canonicalization."""
    
    def test_canonicalize_simple_name(self):
        assert canonicalize_product_name("Apple") == "apple"
        assert canonicalize_product_name("BANANA") == "banana"
    
    def test_canonicalize_with_brand(self):
        assert canonicalize_product_name("Organic Valley Whole Milk") == "organic valley whole milk"
        assert canonicalize_product_name("Trader Joe's Orange Juice") == "trader joes orange juice"
    
    def test_canonicalize_remove_special_chars(self):
        assert canonicalize_product_name("Milk (2%)") == "milk 2"
        assert canonicalize_product_name("Eggs - Large") == "eggs large"
    
    def test_canonicalize_plurals(self):
        # Should normalize plurals to singular
        assert canonicalize_product_name("Apples") == "apple"
        assert canonicalize_product_name("Tomatoes") == "tomato"
    
    def test_canonicalize_with_quantity(self):
        # Should remove quantity information
        assert canonicalize_product_name("2 lb Chicken Breast") == "chicken breast"
        assert canonicalize_product_name("1 gallon Milk") == "milk"
    
    def test_match_product_fuzzy(self):
        # Test fuzzy matching
        result = match_product_to_database("Orgnic Milk", threshold=0.8)
        assert result == "organic milk"
        
        result = match_product_to_database("Chckn Brest", threshold=0.7)
        assert result == "chicken breast"
    
    def test_extract_quantity_from_text(self):
        assert extract_quantity_from_text("2 lbs") == (2, "lb")
        assert extract_quantity_from_text("1.5 gallons") == (1.5, "gallon")
        assert extract_quantity_from_text("500 ml") == (500, "ml")


# ============================================================================
# Unit Conversion Tests
# ============================================================================

class TestUnitConversion:
    """Test unit conversion utilities."""
    
    def test_normalize_unit_volume(self):
        assert normalize_unit("gallon") == "gallon"
        assert normalize_unit("gal") == "gallon"
        assert normalize_unit("gallons") == "gallon"
    
    def test_normalize_unit_weight(self):
        assert normalize_unit("lb") == "pound"
        assert normalize_unit("lbs") == "pound"
        assert normalize_unit("pounds") == "pound"
    
    def test_convert_volume_to_ml(self):
        # Convert gallons to ml
        result = convert_units(1, "gallon", "ml")
        assert abs(result - 3785.41) < 0.1
        
        # Convert quarts to ml
        result = convert_units(1, "quart", "ml")
        assert abs(result - 946.35) < 0.1
    
    def test_convert_weight_to_grams(self):
        # Convert pounds to grams
        result = convert_units(1, "pound", "gram")
        assert abs(result - 453.592) < 0.1
        
        # Convert ounces to grams
        result = convert_units(1, "ounce", "gram")
        assert abs(result - 28.35) < 0.1
    
    def test_convert_same_unit(self):
        # No conversion needed
        result = convert_units(100, "ml", "ml")
        assert result == 100
    
    def test_parse_quantity_with_unit(self):
        assert parse_quantity_with_unit("2 lbs") == (2.0, "pound")
        assert parse_quantity_with_unit("1.5 gallons") == (1.5, "gallon")
        assert parse_quantity_with_unit("500ml") == (500.0, "ml")
    
    def test_invalid_unit_conversion(self):
        # Cannot convert volume to weight
        with pytest.raises(ValueError):
            convert_units(1, "gallon", "pound")


# ============================================================================
# Date/Time Calculation Tests
# ============================================================================

class TestDateTimeUtils:
    """Test date/time calculation utilities."""
    
    def test_calculate_days_until_expiration(self):
        today = datetime.now()
        tomorrow = today + timedelta(days=1)
        week_from_now = today + timedelta(days=7)
        
        assert calculate_days_until_expiration(tomorrow) == 1
        assert calculate_days_until_expiration(week_from_now) == 7
    
    def test_calculate_days_until_expiration_past(self):
        yesterday = datetime.now() - timedelta(days=1)
        assert calculate_days_until_expiration(yesterday) == -1
    
    def test_is_expired(self):
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        tomorrow = today + timedelta(days=1)
        
        assert is_expired(yesterday) is True
        assert is_expired(tomorrow) is False
    
    def test_calculate_consumption_rate(self):
        # Item purchased 10 days ago, 50% consumed
        purchase_date = datetime.now() - timedelta(days=10)
        current_quantity = 5
        original_quantity = 10
        
        rate = calculate_consumption_rate(
            purchase_date,
            current_quantity,
            original_quantity
        )
        
        # Should be 0.5 units per day
        assert abs(rate - 0.5) < 0.01
    
    def test_calculate_consumption_rate_zero_days(self):
        # Item purchased today
        purchase_date = datetime.now()
        rate = calculate_consumption_rate(purchase_date, 5, 10)
        
        # Should handle division by zero
        assert rate == 0


# ============================================================================
# Waste Prediction Logic Tests
# ============================================================================

class TestWastePredictionLogic:
    """Test waste prediction business logic."""
    
    def test_calculate_waste_risk_score(self):
        from backend.ml.prediction_service import calculate_waste_risk_score
        
        # High risk: expiring soon, low consumption
        risk = calculate_waste_risk_score(
            days_until_expiration=2,
            consumption_rate=0.1,
            current_quantity=10
        )
        assert risk > 0.7
        
        # Low risk: far expiration, high consumption
        risk = calculate_waste_risk_score(
            days_until_expiration=30,
            consumption_rate=2.0,
            current_quantity=10
        )
        assert risk < 0.3
    
    def test_calculate_days_until_waste(self):
        from backend.ml.prediction_service import calculate_days_until_waste
        
        # 10 units, consuming 1 per day
        days = calculate_days_until_waste(
            current_quantity=10,
            consumption_rate=1.0
        )
        assert days == 10
        
        # Zero consumption rate
        days = calculate_days_until_waste(
            current_quantity=10,
            consumption_rate=0
        )
        assert days == float('inf')


# ============================================================================
# Recipe Recommendation Scoring Tests
# ============================================================================

class TestRecipeRecommendationScoring:
    """Test recipe recommendation scoring logic."""
    
    def test_calculate_ingredient_match_score(self):
        from backend.ai.recipe_service import calculate_ingredient_match_score
        
        pantry_items = ["tomato", "onion", "garlic", "basil"]
        recipe_ingredients = ["tomato", "onion", "garlic"]
        
        score = calculate_ingredient_match_score(pantry_items, recipe_ingredients)
        
        # All recipe ingredients are available
        assert score == 1.0
    
    def test_calculate_ingredient_match_partial(self):
        from backend.ai.recipe_service import calculate_ingredient_match_score
        
        pantry_items = ["tomato", "onion"]
        recipe_ingredients = ["tomato", "onion", "garlic", "basil"]
        
        score = calculate_ingredient_match_score(pantry_items, recipe_ingredients)
        
        # 50% of ingredients available
        assert score == 0.5
    
    def test_prioritize_at_risk_items(self):
        from backend.ai.recipe_service import calculate_recipe_priority_score
        
        # Recipe uses items with high waste risk
        score = calculate_recipe_priority_score(
            ingredient_match_score=0.8,
            at_risk_items_used=3,
            total_recipe_items=5
        )
        
        assert score > 0.7


# ============================================================================
# Data Validation Tests
# ============================================================================

class TestDataValidation:
    """Test data validation utilities."""
    
    def test_validate_quantity_positive(self):
        from backend.shared.validators import validate_quantity
        
        assert validate_quantity(10) is True
        assert validate_quantity(0.5) is True
        
        with pytest.raises(ValueError):
            validate_quantity(-1)
    
    def test_validate_expiration_date_format(self):
        from backend.shared.validators import validate_date_format
        
        assert validate_date_format("2024-12-31") is True
        assert validate_date_format("2024/12/31") is False
        assert validate_date_format("invalid") is False
    
    def test_validate_unit_supported(self):
        from backend.shared.validators import validate_unit
        
        assert validate_unit("pound") is True
        assert validate_unit("gallon") is True
        assert validate_unit("invalid_unit") is False


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_string_canonicalization(self):
        assert canonicalize_product_name("") == ""
        assert canonicalize_product_name("   ") == ""
    
    def test_unicode_characters(self):
        assert canonicalize_product_name("Café Latté") == "cafe latte"
        assert canonicalize_product_name("Jalapeño") == "jalapeno"
    
    def test_very_large_quantities(self):
        result = convert_units(1000000, "ml", "gallon")
        assert result > 0
    
    def test_zero_quantity_consumption(self):
        rate = calculate_consumption_rate(
            datetime.now() - timedelta(days=10),
            0,  # current quantity
            10  # original quantity
        )
        assert rate == 1.0  # Fully consumed


# ============================================================================
# Pytest Configuration
# ============================================================================

@pytest.fixture
def sample_pantry_item():
    """Sample pantry item for testing."""
    return {
        "id": 1,
        "name": "Whole Milk",
        "category": "dairy",
        "quantity": 1.0,
        "unit": "gallon",
        "purchase_date": datetime.now() - timedelta(days=5),
        "expiration_date": datetime.now() + timedelta(days=7),
        "location": "refrigerator"
    }


@pytest.fixture
def sample_household():
    """Sample household for testing."""
    return {
        "id": 1,
        "name": "Test Household",
        "member_count": 4,
        "created_at": datetime.now() - timedelta(days=30)
    }


# Run with: pytest backend/tests/unit/test_business_logic.py -v

"""
Unit Normalization Service

Converts various unit formats to standardized base units:
- Volume: ml, L, gallon, cup, tbsp, tsp → milliliters (ml)
- Weight: g, kg, lb, oz → grams (g)
- Count: piece, unit, item, serving → pieces
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from decimal import Decimal
import re
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# Conversion Constants
# ============================================================================

# Volume conversions to milliliters (ml)
VOLUME_TO_ML: Dict[str, float] = {
    # Metric
    "ml": 1.0,
    "milliliter": 1.0,
    "millilitre": 1.0,
    "l": 1000.0,
    "liter": 1000.0,
    "litre": 1000.0,
    "dl": 100.0,
    "deciliter": 100.0,
    "cl": 10.0,
    "centiliter": 10.0,
    
    # US customary
    "gallon": 3785.41,
    "gal": 3785.41,
    "quart": 946.353,
    "qt": 946.353,
    "pint": 473.176,
    "pt": 473.176,
    "cup": 236.588,
    "c": 236.588,
    "fluid_ounce": 29.5735,
    "fl_oz": 29.5735,
    "floz": 29.5735,
    "tablespoon": 14.7868,
    "tbsp": 14.7868,
    "teaspoon": 4.92892,
    "tsp": 4.92892,
    
    # Imperial
    "imperial_gallon": 4546.09,
    "imperial_pint": 568.261,
}

# Weight conversions to grams (g)
WEIGHT_TO_G: Dict[str, float] = {
    # Metric
    "g": 1.0,
    "gram": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "mg": 0.001,
    "milligram": 0.001,
    
    # US/Imperial
    "lb": 453.592,
    "pound": 453.592,
    "oz": 28.3495,
    "ounce": 28.3495,
    "ton": 907185.0,
    "tonne": 1000000.0,
}

# Count-based units (normalized to "piece")
COUNT_UNITS = {
    "piece",
    "pieces",
    "unit",
    "units",
    "item",
    "items",
    "count",
    "each",
    "ea",
    "serving",
    "servings",
    "portion",
    "portions",
    "package",
    "packages",
    "pkg",
    "container",
    "containers",
    "can",
    "cans",
    "bottle",
    "bottles",
    "box",
    "boxes",
    "bag",
    "bags",
}

# Composite units (e.g., "6x330ml" → 6 cans of 330ml each)
COMPOSITE_PATTERN = re.compile(r"(\d+)\s*[xX×]\s*(\d+\.?\d*)\s*([a-zA-Z]+)")


@dataclass
class NormalizedQuantity:
    """Normalized quantity result."""
    quantity: float
    unit: str
    original_quantity: float
    original_unit: str
    conversion_factor: float
    is_composite: bool = False
    composite_parts: Optional[Tuple[int, float, str]] = None


class UnitNormalizer:
    """Service to normalize units to base units."""
    
    def __init__(self):
        """Initialize unit normalizer."""
        self.volume_units = VOLUME_TO_ML
        self.weight_units = WEIGHT_TO_G
        self.count_units = COUNT_UNITS
        
        logger.info("UnitNormalizer initialized")
    
    def normalize(
        self,
        quantity: float,
        unit: str,
    ) -> Tuple[float, str]:
        """
        Normalize quantity and unit to base unit.
        
        Args:
            quantity: Input quantity
            unit: Input unit (e.g., "gallon", "lb", "piece")
            
        Returns:
            Tuple of (normalized_quantity, normalized_unit)
            
        Examples:
            >>> normalizer.normalize(1.0, "gallon")
            (3785.41, "ml")
            
            >>> normalizer.normalize(2.5, "lb")
            (1133.98, "g")
            
            >>> normalizer.normalize(12, "piece")
            (12.0, "piece")
            
            >>> normalizer.normalize(1, "6x330ml")
            (1980.0, "ml")
        """
        # Clean unit string
        unit_clean = self._clean_unit(unit)
        
        # Check for composite units (e.g., "6x330ml")
        composite_match = COMPOSITE_PATTERN.match(unit_clean)
        if composite_match:
            count, sub_quantity, sub_unit = composite_match.groups()
            count = int(count)
            sub_quantity = float(sub_quantity)
            
            # Recursively normalize sub-unit
            normalized_sub, normalized_unit = self.normalize(sub_quantity, sub_unit)
            
            # Total quantity = count × normalized_sub × quantity
            total_quantity = count * normalized_sub * quantity
            
            logger.debug(
                f"Composite: {quantity} × {count}×{sub_quantity}{sub_unit} "
                f"→ {total_quantity} {normalized_unit}"
            )
            
            return total_quantity, normalized_unit
        
        # Try volume conversion
        if unit_clean in self.volume_units:
            conversion_factor = self.volume_units[unit_clean]
            normalized_quantity = quantity * conversion_factor
            
            logger.debug(
                f"Volume: {quantity} {unit} → {normalized_quantity} ml "
                f"(factor: {conversion_factor})"
            )
            
            return normalized_quantity, "ml"
        
        # Try weight conversion
        if unit_clean in self.weight_units:
            conversion_factor = self.weight_units[unit_clean]
            normalized_quantity = quantity * conversion_factor
            
            logger.debug(
                f"Weight: {quantity} {unit} → {normalized_quantity} g "
                f"(factor: {conversion_factor})"
            )
            
            return normalized_quantity, "g"
        
        # Count-based units
        if unit_clean in self.count_units:
            logger.debug(f"Count: {quantity} {unit} → {quantity} piece")
            return quantity, "piece"
        
        # Unknown unit - pass through with warning
        logger.warning(f"Unknown unit '{unit}', passing through as-is")
        return quantity, unit_clean
    
    def _clean_unit(self, unit: str) -> str:
        """
        Clean and normalize unit string.
        
        Args:
            unit: Raw unit string
            
        Returns:
            Cleaned unit string (lowercase, no spaces)
        """
        # Convert to lowercase
        cleaned = unit.lower().strip()
        
        # Remove plural 's' (but not from "oz", "tsp", etc.)
        if cleaned.endswith("s") and len(cleaned) > 3:
            singular = cleaned[:-1]
            if singular in self.volume_units or singular in self.weight_units:
                cleaned = singular
        
        # Replace spaces with underscores
        cleaned = cleaned.replace(" ", "_")
        
        # Remove periods
        cleaned = cleaned.replace(".", "")
        
        return cleaned
    
    def convert(
        self,
        quantity: float,
        from_unit: str,
        to_unit: str,
    ) -> float:
        """
        Convert between arbitrary units.
        
        Args:
            quantity: Input quantity
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted quantity
            
        Examples:
            >>> normalizer.convert(1.0, "gallon", "liter")
            3.78541
            
            >>> normalizer.convert(16, "oz", "lb")
            1.0
        """
        # Normalize both
        normalized, base_unit = self.normalize(quantity, from_unit)
        
        # If target is already base unit, return normalized
        to_unit_clean = self._clean_unit(to_unit)
        if to_unit_clean == base_unit:
            return normalized
        
        # Convert from base unit to target unit
        if base_unit == "ml" and to_unit_clean in self.volume_units:
            conversion_factor = self.volume_units[to_unit_clean]
            return normalized / conversion_factor
        
        if base_unit == "g" and to_unit_clean in self.weight_units:
            conversion_factor = self.weight_units[to_unit_clean]
            return normalized / conversion_factor
        
        # Can't convert (different unit types)
        raise ValueError(
            f"Cannot convert from {from_unit} to {to_unit} "
            f"(different unit types: {base_unit} vs {to_unit_clean})"
        )
    
    def get_display_unit(self, quantity: float, base_unit: str) -> Tuple[float, str]:
        """
        Get human-friendly display unit for quantity.
        
        Converts large quantities to larger units for readability.
        
        Args:
            quantity: Quantity in base unit
            base_unit: Base unit ("ml", "g", "piece")
            
        Returns:
            Tuple of (display_quantity, display_unit)
            
        Examples:
            >>> normalizer.get_display_unit(5000, "ml")
            (5.0, "l")
            
            >>> normalizer.get_display_unit(1500, "g")
            (1.5, "kg")
            
            >>> normalizer.get_display_unit(24, "piece")
            (24, "piece")
        """
        if base_unit == "ml":
            if quantity >= 1000:
                return quantity / 1000, "l"
            return quantity, "ml"
        
        if base_unit == "g":
            if quantity >= 1000:
                return quantity / 1000, "kg"
            return quantity, "g"
        
        # Count units don't need conversion
        return quantity, base_unit
    
    def parse_quantity_string(self, quantity_str: str) -> Tuple[float, str]:
        """
        Parse quantity string with embedded unit.
        
        Args:
            quantity_str: String like "2.5kg", "1 gallon", "6x330ml"
            
        Returns:
            Tuple of (quantity, unit)
            
        Examples:
            >>> normalizer.parse_quantity_string("2.5kg")
            (2.5, "kg")
            
            >>> normalizer.parse_quantity_string("1 gallon")
            (1.0, "gallon")
            
            >>> normalizer.parse_quantity_string("6x330ml")
            (1.0, "6x330ml")  # Will be parsed as composite in normalize()
        """
        # Try parsing composite first
        composite_match = COMPOSITE_PATTERN.match(quantity_str.strip())
        if composite_match:
            # Return as-is for normalize() to handle
            return 1.0, quantity_str.strip()
        
        # Try parsing "number unit" format
        pattern = r"(\d+\.?\d*)\s*([a-zA-Z_]+)"
        match = re.match(pattern, quantity_str.strip())
        
        if match:
            quantity = float(match.group(1))
            unit = match.group(2)
            return quantity, unit
        
        # Fallback - try parsing as number
        try:
            quantity = float(quantity_str.strip())
            return quantity, "piece"
        except ValueError:
            raise ValueError(f"Cannot parse quantity string: {quantity_str}")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    normalizer = UnitNormalizer()
    
    # Test volume conversions
    print("\n=== Volume Conversions ===")
    print(normalizer.normalize(1.0, "gallon"))  # → 3785.41 ml
    print(normalizer.normalize(2.0, "liter"))   # → 2000.0 ml
    print(normalizer.normalize(8, "cup"))       # → 1892.7 ml
    print(normalizer.normalize(1, "6x330ml"))   # → 1980.0 ml
    
    # Test weight conversions
    print("\n=== Weight Conversions ===")
    print(normalizer.normalize(2.5, "lb"))      # → 1133.98 g
    print(normalizer.normalize(16, "oz"))       # → 453.592 g
    print(normalizer.normalize(1.5, "kg"))      # → 1500.0 g
    
    # Test count units
    print("\n=== Count Units ===")
    print(normalizer.normalize(12, "piece"))    # → 12.0 piece
    print(normalizer.normalize(6, "cans"))      # → 6.0 piece
    print(normalizer.normalize(1, "package"))   # → 1.0 piece
    
    # Test conversions
    print("\n=== Unit Conversions ===")
    print(normalizer.convert(1.0, "gallon", "liter"))  # → 3.78541
    print(normalizer.convert(16, "oz", "lb"))          # → 1.0
    print(normalizer.convert(1000, "g", "kg"))         # → 1.0
    
    # Test display units
    print("\n=== Display Units ===")
    print(normalizer.get_display_unit(5000, "ml"))    # → 5.0 l
    print(normalizer.get_display_unit(1500, "g"))     # → 1.5 kg
    print(normalizer.get_display_unit(500, "ml"))     # → 500 ml
    
    # Test parsing
    print("\n=== String Parsing ===")
    print(normalizer.parse_quantity_string("2.5kg"))      # → (2.5, "kg")
    print(normalizer.parse_quantity_string("1 gallon"))   # → (1.0, "gallon")
    print(normalizer.parse_quantity_string("6x330ml"))    # → (1.0, "6x330ml")

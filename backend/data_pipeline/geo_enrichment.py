"""
Geographic & Seasonal Enrichment Service

Enriches purchase data with:
- Regional pricing data
- Seasonal availability
- Climate zone information
- Local holiday/event data
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, date
from dataclasses import dataclass
import logging

from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import httpx


logger = logging.getLogger(__name__)


@dataclass
class GeoEnrichment:
    """Geographic enrichment data."""
    # Location
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country_code: Optional[str] = None
    state_province: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    
    # Climate
    climate_zone: Optional[str] = None  # "tropical", "temperate", "arctic"
    hardiness_zone: Optional[str] = None  # USDA zone (e.g., "7a")
    
    # Seasonal
    season: Optional[str] = None  # "spring", "summer", "fall", "winter"
    is_harvest_season: bool = False
    local_produce: Optional[List[str]] = None
    
    # Economic
    price_index: Optional[float] = None  # Regional price multiplier
    avg_household_size: Optional[float] = None
    
    # Events
    upcoming_holidays: Optional[List[str]] = None
    is_holiday_season: bool = False


class GeoEnrichmentService:
    """Service to enrich purchases with geographic/seasonal data."""
    
    def __init__(
        self,
        geocoding_user_agent: str = "PantryPal/1.0",
        enable_caching: bool = True,
    ):
        """
        Initialize geo enrichment service.
        
        Args:
            geocoding_user_agent: User agent for geocoding API
            enable_caching: Cache geocoding results
        """
        self.geolocator = Nominatim(user_agent=geocoding_user_agent)
        self.enable_caching = enable_caching
        self._cache: Dict[str, Any] = {}
        
        logger.info("GeoEnrichmentService initialized")
    
    async def enrich(
        self,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        purchase_date: Optional[datetime] = None,
    ) -> GeoEnrichment:
        """
        Enrich purchase with geographic/seasonal data.
        
        Args:
            location: Location string (e.g., "San Francisco, CA")
            latitude: Latitude (if location not provided)
            longitude: Longitude (if location not provided)
            purchase_date: Purchase date for seasonal data
            
        Returns:
            GeoEnrichment data
        """
        enrichment = GeoEnrichment()
        
        # Geocode location
        if location:
            coords = await self._geocode_location(location)
            if coords:
                enrichment.latitude = coords["latitude"]
                enrichment.longitude = coords["longitude"]
                enrichment.country_code = coords.get("country_code")
                enrichment.state_province = coords.get("state")
                enrichment.city = coords.get("city")
                enrichment.postal_code = coords.get("postal_code")
        elif latitude and longitude:
            enrichment.latitude = latitude
            enrichment.longitude = longitude
            
            # Reverse geocode
            reverse_data = await self._reverse_geocode(latitude, longitude)
            if reverse_data:
                enrichment.country_code = reverse_data.get("country_code")
                enrichment.state_province = reverse_data.get("state")
                enrichment.city = reverse_data.get("city")
        
        # Climate zone
        if enrichment.latitude and enrichment.longitude:
            enrichment.climate_zone = self._get_climate_zone(
                enrichment.latitude,
                enrichment.longitude,
            )
            
            # USDA hardiness zone (US only)
            if enrichment.country_code == "US":
                enrichment.hardiness_zone = self._get_hardiness_zone(
                    enrichment.latitude,
                    enrichment.longitude,
                )
        
        # Seasonal data
        if purchase_date:
            purchase_date_obj = purchase_date.date() if isinstance(purchase_date, datetime) else purchase_date
            
            enrichment.season = self._get_season(
                purchase_date_obj,
                enrichment.latitude or 0.0,
            )
            
            # Local produce
            if enrichment.state_province and enrichment.season:
                enrichment.local_produce = self._get_local_produce(
                    enrichment.state_province,
                    enrichment.season,
                )
                enrichment.is_harvest_season = len(enrichment.local_produce) > 0
            
            # Holidays
            enrichment.upcoming_holidays = self._get_upcoming_holidays(
                purchase_date_obj,
                enrichment.country_code or "US",
            )
            enrichment.is_holiday_season = len(enrichment.upcoming_holidays) > 0
        
        # Economic data
        if enrichment.state_province:
            enrichment.price_index = self._get_price_index(enrichment.state_province)
            enrichment.avg_household_size = self._get_avg_household_size(
                enrichment.state_province
            )
        
        logger.debug(f"Enriched location '{location}': {enrichment}")
        
        return enrichment
    
    async def _geocode_location(self, location: str) -> Optional[Dict[str, Any]]:
        """Geocode location string to coordinates."""
        # Check cache
        cache_key = f"geocode:{location}"
        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Geocode
            result = self.geolocator.geocode(location, addressdetails=True)
            
            if result:
                data = {
                    "latitude": result.latitude,
                    "longitude": result.longitude,
                    "country_code": result.raw.get("address", {}).get("country_code", "").upper(),
                    "state": result.raw.get("address", {}).get("state"),
                    "city": result.raw.get("address", {}).get("city"),
                    "postal_code": result.raw.get("address", {}).get("postcode"),
                }
                
                # Cache result
                if self.enable_caching:
                    self._cache[cache_key] = data
                
                return data
        
        except Exception as e:
            logger.error(f"Geocoding failed for '{location}': {e}")
        
        return None
    
    async def _reverse_geocode(
        self,
        latitude: float,
        longitude: float,
    ) -> Optional[Dict[str, Any]]:
        """Reverse geocode coordinates to location."""
        cache_key = f"reverse:{latitude},{longitude}"
        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            result = self.geolocator.reverse(
                (latitude, longitude),
                addressdetails=True,
            )
            
            if result:
                data = {
                    "country_code": result.raw.get("address", {}).get("country_code", "").upper(),
                    "state": result.raw.get("address", {}).get("state"),
                    "city": result.raw.get("address", {}).get("city"),
                }
                
                if self.enable_caching:
                    self._cache[cache_key] = data
                
                return data
        
        except Exception as e:
            logger.error(f"Reverse geocoding failed for {latitude},{longitude}: {e}")
        
        return None
    
    def _get_climate_zone(self, latitude: float, longitude: float) -> str:
        """Determine climate zone based on latitude."""
        abs_lat = abs(latitude)
        
        if abs_lat < 23.5:
            return "tropical"
        elif abs_lat < 40:
            return "subtropical"
        elif abs_lat < 60:
            return "temperate"
        else:
            return "arctic"
    
    def _get_hardiness_zone(self, latitude: float, longitude: float) -> Optional[str]:
        """Get USDA hardiness zone (simplified)."""
        # Simplified mapping - in production use USDA API
        # https://planthardiness.ars.usda.gov/
        
        if latitude > 48:
            return "3a"
        elif latitude > 45:
            return "4a"
        elif latitude > 42:
            return "5a"
        elif latitude > 39:
            return "6a"
        elif latitude > 36:
            return "7a"
        elif latitude > 33:
            return "8a"
        elif latitude > 30:
            return "9a"
        else:
            return "10a"
    
    def _get_season(self, purchase_date: date, latitude: float) -> str:
        """Determine season based on date and hemisphere."""
        month = purchase_date.month
        
        # Northern hemisphere
        if latitude >= 0:
            if month in [12, 1, 2]:
                return "winter"
            elif month in [3, 4, 5]:
                return "spring"
            elif month in [6, 7, 8]:
                return "summer"
            else:
                return "fall"
        # Southern hemisphere
        else:
            if month in [12, 1, 2]:
                return "summer"
            elif month in [3, 4, 5]:
                return "fall"
            elif month in [6, 7, 8]:
                return "winter"
            else:
                return "spring"
    
    def _get_local_produce(self, state: str, season: str) -> List[str]:
        """Get local produce for state/season."""
        # Simplified lookup - in production use USDA Seasonal Produce Guide
        # https://www.ams.usda.gov/local-food-directories/farmersmarkets
        
        PRODUCE_GUIDE = {
            "spring": ["asparagus", "strawberries", "lettuce", "peas", "spinach"],
            "summer": ["tomatoes", "corn", "watermelon", "berries", "zucchini"],
            "fall": ["apples", "pumpkins", "squash", "sweet_potatoes", "pears"],
            "winter": ["citrus", "kale", "brussels_sprouts", "cabbage", "root_vegetables"],
        }
        
        return PRODUCE_GUIDE.get(season, [])
    
    def _get_upcoming_holidays(
        self,
        purchase_date: date,
        country_code: str,
    ) -> List[str]:
        """Get upcoming holidays within 2 weeks."""
        # Simplified US holidays - in production use holiday API
        # https://date.nager.at/Api
        
        US_HOLIDAYS = {
            (1, 1): "New Year's Day",
            (7, 4): "Independence Day",
            (12, 25): "Christmas",
            (11, 24): "Thanksgiving",  # Approximate
        }
        
        upcoming = []
        
        for (month, day), holiday in US_HOLIDAYS.items():
            holiday_date = date(purchase_date.year, month, day)
            days_until = (holiday_date - purchase_date).days
            
            if 0 <= days_until <= 14:
                upcoming.append(holiday)
        
        return upcoming
    
    def _get_price_index(self, state: str) -> float:
        """Get regional price index (1.0 = national average)."""
        # Simplified - in production use BLS Cost of Living Index
        # https://www.bls.gov/cpi/
        
        HIGH_COST_STATES = {
            "California": 1.15,
            "New York": 1.20,
            "Hawaii": 1.30,
            "Massachusetts": 1.12,
        }
        
        LOW_COST_STATES = {
            "Mississippi": 0.85,
            "Arkansas": 0.87,
            "Oklahoma": 0.88,
        }
        
        if state in HIGH_COST_STATES:
            return HIGH_COST_STATES[state]
        elif state in LOW_COST_STATES:
            return LOW_COST_STATES[state]
        else:
            return 1.0
    
    def _get_avg_household_size(self, state: str) -> float:
        """Get average household size for state."""
        # Simplified - in production use Census Bureau data
        # https://data.census.gov/
        
        # National average
        return 2.53


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    async def main():
        service = GeoEnrichmentService()
        
        # Enrich location
        enrichment = await service.enrich(
            location="San Francisco, CA",
            purchase_date=datetime(2024, 7, 15),
        )
        
        print(f"\n=== Enrichment Results ===")
        print(f"Location: {enrichment.city}, {enrichment.state_province}")
        print(f"Coordinates: {enrichment.latitude}, {enrichment.longitude}")
        print(f"Climate Zone: {enrichment.climate_zone}")
        print(f"Hardiness Zone: {enrichment.hardiness_zone}")
        print(f"Season: {enrichment.season}")
        print(f"Local Produce: {enrichment.local_produce}")
        print(f"Upcoming Holidays: {enrichment.upcoming_holidays}")
        print(f"Price Index: {enrichment.price_index}")
    
    asyncio.run(main())

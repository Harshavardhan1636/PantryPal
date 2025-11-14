"""
Data Pipeline Package

Ingestion, normalization, and enrichment services for PantryPal.

Components:
- ingestion_pipeline: Multi-source ingestion with Pub/Sub
- unit_normalizer: Unit conversion (volume, weight, count)
- geo_enrichment: Geographic and seasonal data enrichment
"""

from .ingestion_pipeline import (
    IngestionPipeline,
    IngestionMessage,
    IngestionSource,
    IngestionStatus,
    IngestionResult,
)

from .unit_normalizer import (
    UnitNormalizer,
    NormalizedQuantity,
)

from .geo_enrichment import (
    GeoEnrichmentService,
    GeoEnrichment,
)

__all__ = [
    # Ingestion
    "IngestionPipeline",
    "IngestionMessage",
    "IngestionSource",
    "IngestionStatus",
    "IngestionResult",
    
    # Normalization
    "UnitNormalizer",
    "NormalizedQuantity",
    
    # Enrichment
    "GeoEnrichmentService",
    "GeoEnrichment",
]

__version__ = "1.0.0"

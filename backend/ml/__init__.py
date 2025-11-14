"""
PantryPal ML System - Production Machine Learning Pipelines

This package contains all ML components for the PantryPal food waste reduction platform:
- Item canonicalization (text/OCR/barcode → canonical items)
- Waste risk prediction (LightGBM classifier + regressor with SHAP)
- Recipe retrieval & ranking (SentenceTransformers + FAISS)
- Shopping list optimization (OR-Tools constraint programming)

Author: Senior SDE-3
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Senior SDE-3"

# Import main configuration
from .config import (
    MLModelType,
    ConfidenceLevel,
    ItemMatchResult,
    WastePredictionResult,
    RecipeRecommendation,
    ShoppingListOptimization,
)

# Import key services
from .services.canonicalization_service import ItemCanonicalizationService

# Import models
from .models.fuzzy_matcher import FuzzyMatcher
from .models.embedding_matcher import EmbeddingMatcher
from .models.consumption_forecaster import ConsumptionForecaster
from .models.waste_predictor import WastePredictor

# Import utilities
from .utils.text_normalizer import TextNormalizer

# Import feature engineering
from .features.feature_engineer import FeatureEngineer

__all__ = [
    # Config
    "MLModelType",
    "ConfidenceLevel",
    "ItemMatchResult",
    "WastePredictionResult",
    "RecipeRecommendation",
    "ShoppingListOptimization",
    # Services
    "ItemCanonicalizationService",
    # Models
    "FuzzyMatcher",
    "EmbeddingMatcher",
    "ConsumptionForecaster",
    "WastePredictor",
    # Utilities
    "TextNormalizer",
    "FeatureEngineer",
]

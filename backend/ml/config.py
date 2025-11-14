"""
ML System Design for PantryPal
Comprehensive machine learning pipelines for food waste prediction and optimization.

Components:
1. Item Canonicalization & NER - Map free text/OCR/barcode to items_catalog
2. Waste Risk Prediction - Core ML model predicting waste probability
3. Recipe Retrieval & Ranking - Recommendation system for at-risk items
4. Shopping List Optimizer - Constraint optimization for purchase planning

Author: Senior SDE-3
Date: 2025-11-12
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


class MLModelType(str, Enum):
    """ML model types in the system."""
    ITEM_CANONICALIZATION = "item_canonicalization"
    WASTE_PREDICTION = "waste_prediction"
    CONSUMPTION_FORECAST = "consumption_forecast"
    RECIPE_RANKING = "recipe_ranking"
    SHOPPING_OPTIMIZER = "shopping_optimizer"


class ConfidenceLevel(str, Enum):
    """Match confidence levels."""
    HIGH = "high"  # >= 0.9
    MEDIUM = "medium"  # 0.75 - 0.9
    LOW = "low"  # < 0.75


@dataclass
class ItemMatchResult:
    """Result from item canonicalization pipeline."""
    canonical_item_id: Optional[str]
    match_confidence: float
    match_method: str  # "barcode", "exact", "fuzzy", "embedding"
    candidate_items: List[Dict[str, Any]]
    needs_human_review: bool
    explanation: Dict[str, Any]


@dataclass
class WastePredictionResult:
    """Result from waste risk prediction model."""
    pantry_entry_id: str
    waste_probability: float
    risk_class: str  # LOW, MEDIUM, HIGH
    predicted_waste_date: Optional[datetime]
    days_until_waste: Optional[int]
    confidence_score: float
    feature_contributions: Dict[str, float]  # SHAP values
    recommended_actions: List[str]


@dataclass
class RecipeRecommendation:
    """Recipe recommendation for at-risk items."""
    recipe_id: str
    recipe_name: str
    score: float
    at_risk_ingredients_used: List[str]
    embedding_similarity: float
    popularity_score: float
    prep_time_minutes: int
    explanation: str


@dataclass
class ShoppingListOptimization:
    """Optimized shopping list result."""
    items_to_buy: List[Dict[str, Any]]
    total_cost: float
    expected_waste_reduction: float
    constraints_satisfied: bool
    optimization_details: Dict[str, Any]


# ============================================================================
# Model Configuration
# ============================================================================

# Item Canonicalization
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # 384d
EMBEDDING_MODEL_LITE = "paraphrase-MiniLM-L6-v2"  # 384d, faster
FAISS_INDEX_TYPE = "IVF_PQ"  # For >100k items
FUZZY_MATCH_THRESHOLD = 0.85
EMBEDDING_SIMILARITY_THRESHOLD = 0.7
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = 0.75

# Waste Prediction
WASTE_PREDICTION_MODEL = "lightgbm"  # or "neural_net", "tft"
WASTE_PREDICTION_HORIZON_DAYS = 14
CONSUMPTION_FORECAST_MODEL = "prophet"  # or "exponential_smoothing"
FEATURE_LOOKBACK_WINDOWS = [7, 14, 30]  # days
COST_AWARE_FN_WEIGHT = 3.0  # Weight false negatives higher
CALIBRATION_METHOD = "isotonic"  # or "platt"

# Recipe Ranking
RECIPE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RECIPE_RANKING_WEIGHTS = {
    "alpha": 0.4,  # at-risk ingredient fraction
    "beta": 0.3,  # embedding similarity
    "gamma": 0.2,  # popularity
    "delta": 0.1,  # prep time penalty
}

# Shopping Optimizer
OPTIMIZATION_ENGINE = "ortools"  # Google OR-Tools CP-SAT
OPTIMIZATION_TIMEOUT_SECONDS = 30


# ============================================================================
# Feature Engineering Constants
# ============================================================================

ENGINEERED_FEATURES = [
    # Temporal features
    "age_days",
    "days_to_expiry",
    "days_since_opened",
    "purchase_day_of_week",
    "purchase_week_of_year",
    
    # Quantity features
    "quantity_on_hand",
    "initial_quantity",
    "quantity_consumed_ratio",
    
    # Consumption patterns
    "avg_daily_consumption_7d",
    "avg_daily_consumption_14d",
    "avg_daily_consumption_30d",
    "consumption_velocity",
    "cohort_avg_consumption",
    
    # Household features
    "household_size",
    "household_pantry_turnover_rate",
    "household_waste_rate_historical",
    
    # Item features
    "storage_type",
    "is_opened",
    "price_per_unit",
    "is_bulk_purchase",
    "typical_shelf_life_days",
    
    # Category features
    "category_waste_rate",
    "category_turnover_rate",
    "category_seasonality_score",
    
    # Interaction features
    "family_size_x_quantity",
    "price_x_quantity",
    "age_x_expiry_ratio",
    
    # External features
    "is_holiday_season",
    "day_of_week_encoded",
    "promotion_flag",
    "recommendation_response_rate",
]


# ============================================================================
# Model Serving Configuration
# ============================================================================

MODEL_ARTIFACT_STORE = "s3://pantrypal-models"  # or "gs://pantrypal-models"
MODEL_REGISTRY = "mlflow"  # or "bentoml"
MLFLOW_TRACKING_URI = "http://mlflow:5000"  # Docker service name
MLFLOW_EXPERIMENT_NAME = "waste_risk_prediction"
MLFLOW_ARTIFACT_LOCATION = "/mlflow/artifacts"  # Local in development, S3 in production
MLFLOW_MODEL_REGISTRY_URI = "postgresql://pantrypal:pantrypal_dev_password@postgres:5432/mlflow"
BATCH_PREDICTION_SCHEDULE = "0 2 * * *"  # Daily at 2 AM
REALTIME_PREDICTION_ENDPOINT = "http://ml-service:5000/predict"
MODEL_MONITORING_METRICS = [
    "prediction_latency_p95",
    "batch_throughput",
    "prediction_accuracy",
    "data_drift_score",
    "model_drift_score",
]


# ============================================================================
# Training Pipeline Configuration
# ============================================================================

TRAINING_PIPELINE_CONFIG = {
    "orchestrator": "airflow",  # or "prefect"
    "schedule": "0 3 * * 0",  # Weekly on Sunday at 3 AM
    "validation_strategy": "time_series_split",
    "n_folds": 5,
    "early_stopping_rounds": 50,
    "auto_retrain_threshold": 0.05,  # Retrain if accuracy drops by 5%
}


# ============================================================================
# Evaluation Metrics
# ============================================================================

EVALUATION_METRICS = {
    "waste_prediction": {
        "classification": ["auc", "f1", "precision_at_k", "recall_at_k"],
        "regression": ["mae", "rmse", "mape"],
        "business": ["cost_saved", "waste_prevented", "user_satisfaction"],
    },
    "item_canonicalization": {
        "accuracy": "exact_match_rate",
        "human_review_rate": "below_confidence_threshold",
        "latency": "p95_ms",
    },
    "recipe_ranking": {
        "relevance": ["ndcg", "map", "mrr"],
        "engagement": ["ctr", "conversion_rate", "time_to_action"],
    },
    "shopping_optimizer": {
        "optimization": ["constraint_satisfaction_rate", "solution_quality"],
        "business": ["waste_reduction", "cost_efficiency", "user_adoption"],
    },
}


# ============================================================================
# Data Quality Thresholds
# ============================================================================

DATA_QUALITY_CHECKS = {
    "min_samples_per_household": 10,
    "max_null_rate": 0.1,
    "max_outlier_rate": 0.05,
    "min_prediction_confidence": 0.6,
    "max_data_drift_score": 0.3,
}

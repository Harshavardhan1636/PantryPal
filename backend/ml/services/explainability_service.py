"""
Explainability Service - Task #10

Computes and stores SHAP explanations for waste predictions.

Features:
- Top-5 feature importance with SHAP values
- Human-readable explanations
- Feature contribution visualization data
- Support for user feedback integration
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import shap


logger = logging.getLogger(__name__)


@dataclass
class Explanation:
    """
    Explanation for a single prediction.
    
    Attributes:
        features: Top-5 feature names
        values: Top-5 SHAP values (positive = increases waste risk)
        base_value: Model's base prediction (average)
        prediction: Final prediction value
        text: Human-readable explanation
    """
    features: List[str]
    values: List[float]
    base_value: float
    prediction: float
    text: str


class ExplainabilityService:
    """
    Service for generating SHAP-based explanations.
    
    Converts raw SHAP values into human-readable reasons like:
    "Because: 5 days to expiry + low consumption history"
    """
    
    def __init__(self, explainer: shap.TreeExplainer):
        """
        Initialize explainability service.
        
        Args:
            explainer: Pre-trained SHAP TreeExplainer for waste model
        """
        self.explainer = explainer
        
        # Feature name to human-readable mapping
        self.feature_names_readable = {
            "days_to_expiry": "days until expiry",
            "age_days": "item age",
            "quantity_on_hand": "quantity remaining",
            "avg_daily_consumption_7d": "consumption rate",
            "days_since_last_used": "days since last use",
            "household_size": "household size",
            "category_waste_rate": "category waste rate",
            "is_perishable": "perishability",
            "storage_temp_mismatch": "storage condition",
            "seasonal_demand": "seasonal demand",
            "price_per_unit": "item value",
            "consumption_volatility": "usage consistency",
            "expiry_risk_score": "expiry urgency",
            "stock_days": "days of supply",
            "is_opened": "opened status",
            "household_waste_rate": "household waste history",
            "time_to_consume": "estimated consumption time",
            "is_holiday_week": "holiday timing",
            "local_produce": "local availability",
        }
    
    def explain(
        self,
        features: pd.DataFrame,
        prediction: float,
        top_k: int = 5,
    ) -> Explanation:
        """
        Generate explanation for a single prediction.
        
        Args:
            features: Feature values (single row DataFrame)
            prediction: Model prediction (waste probability)
            top_k: Number of top features to include
            
        Returns:
            Explanation with top features and human-readable text
        """
        # Compute SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Handle binary classification (shap_values may be 2D)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]  # Positive class
        
        # Flatten SHAP values (single prediction)
        shap_values_flat = shap_values.flatten()
        
        # Get top-k features by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_values_flat))[-top_k:][::-1]
        
        top_features = []
        top_values = []
        
        for idx in top_indices:
            feature_name = features.columns[idx]
            shap_value = shap_values_flat[idx]
            feature_value = features.iloc[0, idx]
            
            top_features.append(feature_name)
            top_values.append(float(shap_value))
        
        # Generate human-readable text
        explanation_text = self._generate_explanation_text(
            features=features,
            top_features=top_features,
            top_values=top_values,
            prediction=prediction,
        )
        
        return Explanation(
            features=top_features,
            values=top_values,
            base_value=float(base_value),
            prediction=prediction,
            text=explanation_text,
        )
    
    def explain_batch(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        top_k: int = 5,
    ) -> List[Explanation]:
        """
        Generate explanations for batch of predictions.
        
        Args:
            features: Feature values (multi-row DataFrame)
            predictions: Model predictions (waste probabilities)
            top_k: Number of top features per prediction
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for i in range(len(features)):
            feature_row = features.iloc[[i]]
            prediction = predictions[i]
            
            explanation = self.explain(feature_row, prediction, top_k)
            explanations.append(explanation)
        
        return explanations
    
    def _generate_explanation_text(
        self,
        features: pd.DataFrame,
        top_features: List[str],
        top_values: List[float],
        prediction: float,
    ) -> str:
        """
        Generate human-readable explanation text.
        
        Example:
        "Because: 5 days to expiry + low consumption history + opened package"
        """
        reasons = []
        
        for feature, shap_value in zip(top_features, top_values):
            feature_value = features[feature].iloc[0]
            
            # Get human-readable feature name
            readable_name = self.feature_names_readable.get(
                feature,
                feature.replace("_", " ")
            )
            
            # Generate reason based on feature type
            reason = self._generate_reason(
                feature=feature,
                value=feature_value,
                shap_value=shap_value,
                readable_name=readable_name,
            )
            
            if reason:
                reasons.append(reason)
        
        # Combine top 3 reasons
        if not reasons:
            return "Standard waste risk assessment"
        
        top_reasons = reasons[:3]
        explanation = "Because: " + " + ".join(top_reasons)
        
        return explanation
    
    def _generate_reason(
        self,
        feature: str,
        value: Any,
        shap_value: float,
        readable_name: str,
    ) -> Optional[str]:
        """
        Generate reason string for a single feature.
        
        Rules:
        - Positive SHAP value = increases waste risk
        - Negative SHAP value = decreases waste risk
        """
        # Handle numeric features
        if feature == "days_to_expiry":
            if value <= 3 and shap_value > 0:
                return f"{int(value)} days to expiry"
            elif value <= 7 and shap_value > 0:
                return "expiring soon"
        
        elif feature == "avg_daily_consumption_7d":
            if value < 0.1 and shap_value > 0:
                return "low consumption history"
            elif value > 1.0 and shap_value < 0:
                return "high usage rate"
        
        elif feature == "days_since_last_used":
            if value > 7 and shap_value > 0:
                return "unused for days"
            elif value > 14 and shap_value > 0:
                return "unused for weeks"
        
        elif feature == "quantity_on_hand":
            if value > 5 and shap_value > 0:
                return "large quantity"
            elif value < 1 and shap_value < 0:
                return "nearly depleted"
        
        elif feature == "age_days":
            if value > 14 and shap_value > 0:
                return "old item"
            elif value > 30 and shap_value > 0:
                return "very old item"
        
        elif feature == "is_opened":
            if value == 1 and shap_value > 0:
                return "opened package"
        
        elif feature == "storage_temp_mismatch":
            if value == 1 and shap_value > 0:
                return "improper storage"
        
        elif feature == "is_perishable":
            if value == 1 and shap_value > 0:
                return "perishable item"
        
        elif feature == "category_waste_rate":
            if value > 0.3 and shap_value > 0:
                return "high-waste category"
        
        elif feature == "household_waste_rate":
            if value > 0.2 and shap_value > 0:
                return "household waste pattern"
        
        elif feature == "consumption_volatility":
            if value > 0.5 and shap_value > 0:
                return "inconsistent usage"
        
        elif feature == "seasonal_demand":
            if value < -0.5 and shap_value > 0:
                return "out of season"
        
        elif feature == "is_holiday_week":
            if value == 1 and shap_value > 0:
                return "holiday disruption"
        
        elif feature == "price_per_unit":
            if value > 10 and shap_value < 0:
                return "valuable item"
        
        # Generic fallback
        if abs(shap_value) > 0.05:  # Only include significant features
            direction = "high" if shap_value > 0 else "low"
            return f"{direction} {readable_name}"
        
        return None
    
    def get_feature_contributions(
        self,
        features: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Get all feature contributions (SHAP values).
        
        Useful for visualization in UI.
        
        Args:
            features: Feature values (single row)
            
        Returns:
            Dictionary of feature name -> SHAP value
        """
        shap_values = self.explainer.shap_values(features)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        shap_values_flat = shap_values.flatten()
        
        contributions = {}
        for i, feature_name in enumerate(features.columns):
            contributions[feature_name] = float(shap_values_flat[i])
        
        return contributions
    
    def format_for_api(
        self,
        explanation: Explanation,
    ) -> Dict[str, Any]:
        """
        Format explanation for API response.
        
        Returns:
            {
                "text": "Because: 5 days to expiry + low consumption history",
                "features": ["days_to_expiry", "avg_daily_consumption_7d", ...],
                "values": [0.15, 0.12, ...],
                "contributions": {
                    "days_to_expiry": 0.15,
                    "avg_daily_consumption_7d": 0.12,
                    ...
                }
            }
        """
        contributions = {
            feature: value
            for feature, value in zip(explanation.features, explanation.values)
        }
        
        return {
            "text": explanation.text,
            "features": explanation.features,
            "values": explanation.values,
            "contributions": contributions,
            "base_value": explanation.base_value,
            "prediction": explanation.prediction,
        }


# ============================================================================
# Integration with Prediction Pipeline
# ============================================================================

def add_explanation_to_prediction(
    prediction: Dict[str, Any],
    features: pd.DataFrame,
    explainer: shap.TreeExplainer,
) -> Dict[str, Any]:
    """
    Augment prediction result with explanation.
    
    This is called by the batch prediction service or BentoML service.
    
    Args:
        prediction: Raw prediction from model
        features: Feature values used for prediction
        explainer: SHAP explainer
        
    Returns:
        Prediction with "explanation" field added
    """
    service = ExplainabilityService(explainer)
    
    explanation = service.explain(
        features=features,
        prediction=prediction["waste_probability"],
        top_k=5,
    )
    
    prediction["explanation"] = service.format_for_api(explanation)
    
    return prediction


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Demo: Generate explanation
    
    # Mock data
    features = pd.DataFrame({
        "days_to_expiry": [3.0],
        "avg_daily_consumption_7d": [0.05],
        "quantity_on_hand": [2.0],
        "is_opened": [1.0],
        "age_days": [10.0],
        "household_size": [3.0],
        "category_waste_rate": [0.25],
    })
    
    # Mock explainer (in production, load from model artifacts)
    # explainer = shap.TreeExplainer(model)
    
    print("Feature values:")
    print(features.T)
    print()
    
    # Mock explanation generation
    explanation = Explanation(
        features=["days_to_expiry", "avg_daily_consumption_7d", "is_opened"],
        values=[0.15, 0.12, 0.08],
        base_value=0.20,
        prediction=0.65,
        text="Because: 3 days to expiry + low consumption history + opened package",
    )
    
    print("Explanation:")
    print(f"  Text: {explanation.text}")
    print(f"  Prediction: {explanation.prediction:.2%}")
    print(f"  Base value: {explanation.base_value:.2%}")
    print()
    print("Top features:")
    for feature, value in zip(explanation.features, explanation.values):
        direction = "↑" if value > 0 else "↓"
        print(f"  {direction} {feature}: {value:+.3f}")

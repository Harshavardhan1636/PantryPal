"""BentoML service for waste risk prediction."""
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel, Field
from typing import List, Dict
# numpy imported via other dependencies

from models.waste_risk_predictor import get_predictor


# Pydantic models for API
class BatchInput(BaseModel):
    """Input schema for single batch prediction."""
    batch_id: str
    purchase_date: str  # ISO format
    expiry_date: str = None
    opened_date: str = None
    quantity: float = Field(gt=0)
    initial_quantity: float = Field(gt=0)
    storage_type: str = "pantry"
    category: str = "other"
    consumption_rate: float = 0.0
    household_size: int = 2
    past_waste_count: int = 0
    purchase_price: float = 0.0


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    batch_id: str
    risk_score: float
    risk_class: str
    predicted_waste_date: str
    confidence: float
    model_version: str


# Create BentoML service
waste_risk_runner = bentoml.models.get("waste_risk_predictor:latest").to_runner()
svc = bentoml.Service("waste_risk_predictor", runners=[waste_risk_runner])


@svc.api(input=JSON(pydantic_model=BatchInput), output=JSON(pydantic_model=PredictionOutput))
def predict(input_data: BatchInput) -> PredictionOutput:
    """Predict waste risk for a single inventory batch."""
    predictor = get_predictor()
    
    # Convert input to dict
    batch_data = input_data.dict()
    
    # Convert date strings to date objects
    from datetime import date
    if batch_data["purchase_date"]:
        batch_data["purchase_date"] = date.fromisoformat(batch_data["purchase_date"])
    if batch_data["expiry_date"]:
        batch_data["expiry_date"] = date.fromisoformat(batch_data["expiry_date"])
    if batch_data["opened_date"]:
        batch_data["opened_date"] = date.fromisoformat(batch_data["opened_date"])
    
    # Get prediction
    result = predictor.predict(batch_data)
    
    return PredictionOutput(
        batch_id=batch_data["batch_id"],
        risk_score=result["risk_score"],
        risk_class=result["risk_class"],
        predicted_waste_date=result["predicted_waste_date"].isoformat(),
        confidence=result["confidence"],
        model_version=result["model_version"],
    )


@svc.api(input=JSON(), output=JSON())
def predict_batch(input_data: Dict) -> List[Dict]:
    """Predict waste risk for multiple batches."""
    predictor = get_predictor()
    
    batches = input_data.get("batches", [])
    
    # Convert date strings
    from datetime import date
    for batch in batches:
        if batch.get("purchase_date"):
            batch["purchase_date"] = date.fromisoformat(batch["purchase_date"])
        if batch.get("expiry_date"):
            batch["expiry_date"] = date.fromisoformat(batch["expiry_date"])
        if batch.get("opened_date"):
            batch["opened_date"] = date.fromisoformat(batch["opened_date"])
    
    # Get predictions
    results = predictor.predict_batch(batches)
    
    # Format output
    return [
        {
            "batch_id": batch["batch_id"],
            "risk_score": result["risk_score"],
            "risk_class": result["risk_class"],
            "predicted_waste_date": result["predicted_waste_date"].isoformat(),
            "confidence": result["confidence"],
        }
        for batch, result in zip(batches, results)
    ]


@svc.api(input=JSON(), output=JSON())
def health() -> Dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "waste_risk_predictor",
        "model_version": get_predictor().MODEL_VERSION,
    }

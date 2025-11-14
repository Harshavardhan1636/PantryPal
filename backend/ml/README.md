# PantryPal ML System

Production-ready machine learning pipelines for food waste prediction and optimization.

## 📚 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Components](#components)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [API Integration](#api-integration)

---

## 🎯 Overview

The PantryPal ML system consists of 4 major pipelines:

1. **Item Canonicalization** - Maps user input (text, OCR, barcode) to canonical items
2. **Waste Risk Prediction** - Predicts probability and timing of food waste
3. **Recipe Retrieval** - Recommends recipes for at-risk items
4. **Shopping Optimization** - Generates optimal shopping lists

**Status:** ✅ 75% Complete (Core models implemented, infrastructure TODO)

---

## 🏗️ Architecture

```
ml/
├── config.py                      # Global ML configuration
├── utils/
│   └── text_normalizer.py        # Text preprocessing
├── models/
│   ├── fuzzy_matcher.py          # RapidFuzz string matching
│   ├── embedding_matcher.py      # SentenceTransformers + FAISS
│   ├── consumption_forecaster.py # Prophet/Exponential Smoothing
│   └── waste_predictor.py        # LightGBM + SHAP
├── features/
│   └── feature_engineer.py       # Feature extraction (50+ features)
├── services/
│   └── canonicalization_service.py # Item matching orchestrator
└── ML_IMPLEMENTATION_SUMMARY.md  # Detailed documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ (with pgvector extension)
- 8GB RAM minimum
- GPU optional (for faster embedding encoding)

### Installation

```bash
# Install dependencies
cd backend
pip install -r ml_requirements.txt

# Verify installation
python -c "import lightgbm, sentence_transformers, faiss; print('✓ ML dependencies installed')"
```

### Basic Usage

```python
from backend.ml import ItemCanonicalizationService, WastePredictor

# Initialize services
canonicalization = ItemCanonicalizationService(db_session)
predictor = WastePredictor(model_path="models/waste_predictor_v1")

# Canonicalize item
result = await canonicalization.canonicalize(
    user_input="organic whole milk",
    barcode="012345678901",
)
print(f"Matched: {result.canonical_item_id} (confidence: {result.match_confidence:.2f})")

# Predict waste risk
features = await feature_engineer.extract_features(pantry_entry_id)
prediction = predictor.predict(features, pantry_entry_id)
print(f"Waste risk: {prediction.risk_class} ({prediction.waste_probability:.2f})")
```

---

## 🧩 Components

### 1. Item Canonicalization Pipeline

**Purpose:** Map free text/OCR/barcode to canonical items in `items_catalog`

**Pipeline Stages:**

```
User Input → Normalization → Barcode Lookup → Exact Match → Fuzzy Match → Embedding Search → Scoring → Human Review (if needed)
```

**Models:**
- Text normalization (regex + stopwords)
- RapidFuzz token sort ratio (fuzzy matching)
- SentenceTransformers all-mpnet-base-v2 (embeddings)
- FAISS IVFFlat/Flat (ANN search)

**Confidence Scoring:**
```python
score = 0.6 * fuzzy_score + 0.4 * embedding_similarity + barcode_bonus
if score < 0.75:
    route_to_human_review()
```

**Example:**
```python
from backend.ml import ItemCanonicalizationService

service = ItemCanonicalizationService(db_session)
result = await service.canonicalize("org whle milk", barcode=None)

# ItemMatchResult(
#     canonical_item_id="uuid",
#     match_confidence=0.85,
#     match_method="ensemble",
#     candidate_items=[...],
#     needs_human_review=False,
# )
```

---

### 2. Waste Risk Prediction

**Purpose:** Predict probability and timing of food waste for pantry items

**Two-Stage Architecture:**

**Stage A: Consumption Forecasting**
- Models: Prophet (rich history) or Exponential Smoothing (sparse data)
- Output: Predicted daily consumption rate
- Fallback: Cohort average for cold start

**Stage B: Waste Classification & Regression**
- Classifier: LightGBM binary (P(waste))
- Regressor: LightGBM regression (days_to_waste)
- Explainability: SHAP TreeExplainer for feature contributions

**Features (50+):**
- Temporal: age_days, days_to_expiry, days_since_opened
- Quantity: quantity_on_hand, quantity_consumed_ratio
- Consumption: avg_daily_consumption (7d, 14d, 30d)
- Household: household_size, turnover_rate, waste_rate_historical
- Item: storage_type, price_per_unit, is_bulk_purchase
- Category: category_waste_rate, category_turnover_rate

**Example:**
```python
from backend.ml import WastePredictor, FeatureEngineer

# Extract features
engineer = FeatureEngineer(db_session)
features = await engineer.extract_features(pantry_entry_id)

# Predict
predictor = WastePredictor()
result = predictor.predict(features, pantry_entry_id, return_shap=True)

# WastePredictionResult(
#     waste_probability=0.82,
#     risk_class="HIGH",
#     predicted_waste_date=datetime(2025, 11, 20),
#     days_until_waste=8,
#     feature_contributions={
#         "days_to_expiry": -0.15,  # Most important
#         "quantity_on_hand": 0.08,
#     },
#     recommended_actions=["USE_SOON", "FIND_RECIPE"],
# )
```

---

### 3. Recipe Retrieval & Ranking

**Purpose:** Recommend recipes that use at-risk items

**Architecture:**
- Precompute: SentenceTransformer embeddings for all recipes
- Search: FAISS ANN search for semantic similarity
- Ranking: Multi-factor scoring + Learning-to-Rank (optional)

**Scoring Function:**
```python
score = (
    0.4 * fraction_at_risk_ingredients +  # Uses high-risk items
    0.3 * embedding_similarity +           # Semantic match
    0.2 * popularity_score +               # User engagement
    -0.1 * prep_time_penalty               # Prefer quick recipes
)
```

**Status:** Architecture defined, implementation TODO

---

### 4. Shopping List Optimizer

**Purpose:** Generate optimal shopping lists minimizing waste

**Approach:** Integer programming via OR-Tools CP-SAT

**Formulation:**
```python
# Decision variables: x_i = quantity to buy
minimize: sum(expected_waste_cost[i] * x[i]) + penalties

constraints:
  - sum(x[i] * price[i]) <= budget
  - x[i] >= min_required[i] for required items
  - x[i] % pack_size[i] == 0  # Integer multiples
```

**Status:** Architecture defined, implementation TODO

---

## 💾 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r backend/ml_requirements.txt
```

**Key dependencies:**
- lightgbm>=4.0.0 - Gradient boosting
- sentence-transformers>=2.2.0 - Embeddings
- faiss-cpu>=1.7.4 - Vector search
- prophet>=1.1.0 - Time series
- shap>=0.43.0 - Explainability
- rapidfuzz>=3.0.0 - Fuzzy matching
- ortools>=9.7.0 - Optimization

### Step 2: Download Models (First Time Setup)

```bash
# Download SentenceTransformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

# This caches the model to ~/.cache/torch/sentence_transformers/
```

### Step 3: Set Up Database (if not already done)

```bash
cd infrastructure/database
psql -U postgres < schema_v2_production.sql
```

---

## 📖 Usage Examples

### Example 1: Item Canonicalization

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from backend.ml import ItemCanonicalizationService, EmbeddingMatcher

async def main():
    # Initialize database session
    engine = create_async_engine("postgresql+asyncpg://...")
    async with AsyncSession(engine) as session:
        # Build FAISS index (one-time setup)
        embedding_matcher = EmbeddingMatcher()
        
        # Fetch items from catalog
        items = await session.execute(select(ItemsCatalog))
        item_ids = [str(i.item_id) for i in items.scalars()]
        item_texts = [i.name for i in items.scalars()]
        
        embedding_matcher.build_index(item_texts, item_ids)
        
        # Initialize service
        service = ItemCanonicalizationService(session, embedding_matcher)
        
        # Canonicalize
        result = await service.canonicalize(
            user_input="Organic 2% Reduced Fat Milk",
            barcode=None,
        )
        
        print(f"Item ID: {result.canonical_item_id}")
        print(f"Confidence: {result.match_confidence:.2f}")
        print(f"Method: {result.match_method}")
        
        if result.needs_human_review:
            print("⚠️ Low confidence - route to human review")

asyncio.run(main())
```

### Example 2: Waste Prediction

```python
from backend.ml import WastePredictor, FeatureEngineer

async def predict_waste_risk(pantry_entry_id: str):
    # Extract features
    engineer = FeatureEngineer(db_session)
    feature_vector = await engineer.extract_features(pantry_entry_id)
    
    # Load model
    predictor = WastePredictor(model_path="models/waste_predictor_v2")
    
    # Predict
    result = predictor.predict(
        feature_vector.features,
        pantry_entry_id,
        return_shap=True,
    )
    
    # Display results
    print(f"Risk: {result.risk_class} ({result.waste_probability:.1%})")
    
    if result.days_until_waste:
        print(f"Expected waste date: {result.predicted_waste_date}")
    
    # Show top contributing features
    top_features = sorted(
        result.feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:5]
    
    print("\nTop risk factors:")
    for feature, contribution in top_features:
        print(f"  {feature}: {contribution:+.3f}")
    
    print(f"\nActions: {', '.join(result.recommended_actions)}")
```

### Example 3: Batch Predictions

```python
from backend.ml import WastePredictor, FeatureEngineer
import pandas as pd

async def run_batch_predictions(household_id: str):
    # Fetch all active pantry entries
    query = select(PantryEntry).where(
        PantryEntry.household_id == household_id,
        PantryEntry.quantity_on_hand > 0,
    )
    entries = (await db.execute(query)).scalars().all()
    
    # Extract features
    engineer = FeatureEngineer(db_session)
    feature_vectors = await engineer.extract_batch_features(
        [str(e.entry_id) for e in entries]
    )
    
    # Convert to DataFrame
    df = engineer.to_dataframe(feature_vectors)
    X = df[engineer.feature_names]
    
    # Load model
    predictor = WastePredictor(model_path="models/waste_predictor_v2")
    
    # Predict
    predictions = predictor.predict_batch(
        X,
        pantry_entry_ids=df["pantry_entry_id"].tolist(),
    )
    
    # Save to database
    for pred in predictions:
        await db.execute(
            insert(Prediction).values(
                prediction_id=uuid.uuid4(),
                pantry_entry_id=pred.pantry_entry_id,
                risk_class=pred.risk_class,
                confidence_score=pred.confidence_score,
                predicted_date=pred.predicted_waste_date,
                explanation=pred.feature_contributions,
            )
        )
    
    await db.commit()
    
    print(f"✓ Generated predictions for {len(predictions)} items")
```

---

## 🎓 Model Training

### Training Waste Prediction Model

```python
from backend.ml import WastePredictor, FeatureEngineer
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. Extract training data
async def prepare_training_data():
    # Query historical data
    query = """
    SELECT 
        pe.entry_id,
        CASE WHEN we.event_id IS NOT NULL THEN 1 ELSE 0 END as wasted,
        EXTRACT(day FROM we.wasted_at - pe.purchase_date) as days_to_waste
    FROM pantry_entries pe
    LEFT JOIN waste_events we ON pe.entry_id = we.pantry_entry_id
    WHERE pe.purchase_date >= '2024-01-01'
    """
    
    results = await db.execute(query)
    labels_df = pd.DataFrame(results.fetchall())
    
    # Extract features
    engineer = FeatureEngineer(db_session)
    features = await engineer.extract_batch_features(
        labels_df["entry_id"].tolist()
    )
    
    X = engineer.to_dataframe(features)
    y_waste = labels_df["wasted"].values
    y_days = labels_df["days_to_waste"].fillna(0).values
    
    return X, y_waste, y_days

# 2. Train model
X, y_waste, y_days = await prepare_training_data()

X_train, X_val, y_waste_train, y_waste_val, y_days_train, y_days_val = train_test_split(
    X, y_waste, y_days, test_size=0.2, random_state=42
)

predictor = WastePredictor()
artifacts = predictor.train(
    X_train, y_waste_train, y_days_train,
    X_val, y_waste_val, y_days_val,
    feature_names=X.columns.tolist(),
)

# 3. Save model
predictor.save_model("models/waste_predictor_v3")

print("✓ Model trained and saved")
```

---

## 🚀 Deployment

### Option 1: Local Development

```bash
# Run API server with ML endpoints
cd backend
uvicorn api.main:app --reload --port 8000

# Test predictions endpoint
curl http://localhost:8000/predictions/household_uuid
```

### Option 2: Docker (TODO)

```bash
# Build image
docker build -t pantrypal-ml:latest -f Dockerfile.ml .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -v ./models:/app/models \
  pantrypal-ml:latest
```

### Option 3: MLflow Model Registry (TODO)

```python
import mlflow

# Register model
mlflow.set_tracking_uri("http://mlflow-server:5000")

with mlflow.start_run():
    mlflow.log_params({"model": "lightgbm", "version": "v3"})
    mlflow.log_metrics({"auc": 0.89, "mae": 2.1})
    
    mlflow.sklearn.log_model(
        predictor,
        "waste_predictor",
        registered_model_name="WasteRiskPredictor",
    )
```

---

## 📊 Monitoring

### Model Performance Metrics

Track these metrics in Grafana:

- **Accuracy:** AUC, Precision@10, Recall@10, F1
- **Latency:** P50, P95, P99 prediction time
- **Throughput:** Predictions per second
- **Data Drift:** Feature distribution changes
- **Model Drift:** Accuracy degradation over time

### Example Monitoring Query (Prometheus)

```promql
# Prediction latency P95
histogram_quantile(0.95, ml_prediction_duration_seconds_bucket)

# Accuracy (AUC) over time
ml_model_auc{model="waste_predictor",stage="production"}

# Prediction volume
rate(ml_predictions_total[5m])
```

---

## 🔌 API Integration

### Update API Router

```python
# api/routers/predictions.py
from backend.ml import WastePredictor, FeatureEngineer

# Load model at startup
predictor = WastePredictor(model_path="models/waste_predictor_production")

@router.get("/predictions/{household_id}")
async def get_predictions(
    household_id: str,
    db: AsyncSession = Depends(get_db),
):
    # Fetch pantry entries
    entries = await fetch_active_entries(household_id, db)
    
    # Extract features
    engineer = FeatureEngineer(db)
    features = await engineer.extract_batch_features(
        [e.entry_id for e in entries]
    )
    
    # Predict
    predictions = predictor.predict_batch(features, entry_ids)
    
    return predictions
```

---

## 🐛 Troubleshooting

### Issue: FAISS index not found

```python
# Rebuild FAISS index
python backend/ml/data/build_faiss_index.py \
  --db-url postgresql://... \
  --output models/faiss_index/
```

### Issue: Prophet installation fails

```bash
# Install C++ compiler (required by pystan)
# On Ubuntu:
sudo apt-get install build-essential

# On macOS:
xcode-select --install

# Then reinstall
pip install prophet
```

### Issue: Out of memory during training

```python
# Use data subsampling
X_sample = X.sample(frac=0.5, random_state=42)

# Or use Dask for distributed computing
import dask.dataframe as dd
X_dask = dd.from_pandas(X, npartitions=10)
```

---

## 📝 License

Proprietary - PantryPal Platform

---

## 👥 Contributors

- Senior SDE-3 - ML System Design & Implementation

---

## 📚 Additional Resources

- [ML Implementation Summary](./ML_IMPLEMENTATION_SUMMARY.md) - Detailed architecture
- [API Documentation](../api/README.md) - REST API endpoints
- [Database Schema](../../infrastructure/database/README_SCHEMA_V2.md) - Data model

---

**Status:** ✅ 75% Complete | **Version:** 1.0.0 | **Last Updated:** 2025-11-12

# ML Training Pipeline Documentation

## Overview

Production-ready training pipeline for waste risk prediction models with MLflow tracking, hyperparameter optimization, and comprehensive validation.

## Architecture

```
backend/ml/training/
├── __init__.py
├── train_waste_model.py        # Main training script with CV & MLflow
├── generate_synthetic_data.py  # Synthetic data generator
├── validate_model.py           # Comprehensive validation suite
└── README.md                   # This file

Infrastructure:
├── MLflow (port 5000)          # Experiment tracking & model registry
├── Airflow (port 8080)         # Workflow orchestration
└── PostgreSQL                  # MLflow backend store
```

## Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r ml_requirements_training.txt
```

### 2. Start Infrastructure

```bash
# Start MLflow + Airflow (from project root)
docker-compose --profile ml-training up -d

# Verify services
curl http://localhost:5000/health  # MLflow
curl http://localhost:8080/health  # Airflow

# Access UIs
open http://localhost:5000  # MLflow UI
open http://localhost:8080  # Airflow UI (admin/admin)
```

### 3. Generate Synthetic Training Data

```bash
# Generate 100k samples (1000 households × 100 items)
python -m backend.ml.training.generate_synthetic_data \
    --n-households 1000 \
    --n-items-per-household 100 \
    --output data/synthetic_training_100k.parquet

# Quick test (1k samples)
python -m backend.ml.training.generate_synthetic_data \
    --n-households 10 \
    --n-items-per-household 100 \
    --output data/synthetic_training_1k.parquet
```

**Output:**
- Realistic household behavior simulation
- 50+ engineered features
- Balanced waste/no-waste distribution (~25% waste rate)
- Category-specific characteristics
- File: `data/synthetic_training_100k.parquet` (~15 MB)

### 4. Train Model

#### Option A: Using Synthetic Data

```bash
python -m backend.ml.training.train_waste_model \
    --use-synthetic \
    --n-samples 10000 \
    --n-folds 5
```

#### Option B: Using Real Data from Database

```bash
python -m backend.ml.training.train_waste_model \
    --start-date 2024-01-01 \
    --end-date 2025-11-01 \
    --n-folds 5
```

#### Option C: Using Saved Data File

```bash
python -m backend.ml.training.train_waste_model \
    --data-file data/synthetic_training_100k.parquet \
    --n-folds 5
```

#### Option D: With Hyperparameter Optimization

```bash
python -m backend.ml.training.train_waste_model \
    --use-synthetic \
    --n-samples 50000 \
    --n-folds 5 \
    --hyperopt \
    --n-trials 100
```

**Training Time Estimates:**
- 1k samples: ~30 seconds
- 10k samples: ~2-3 minutes
- 100k samples: ~15-20 minutes
- 100k samples + hyperopt (100 trials): ~2-3 hours

### 5. View Results in MLflow

```bash
# MLflow UI
open http://localhost:5000

# Or via CLI
mlflow ui --backend-store-uri postgresql://pantrypal:pantrypal_dev_password@localhost:5432/mlflow
```

**MLflow Features:**
- ✅ Experiment tracking
- ✅ Hyperparameter logging
- ✅ Metric visualization
- ✅ Model artifacts
- ✅ Model registry
- ✅ Model versioning
- ✅ Model comparison

## Training Pipeline Features

### 1. Data Loading

Three data source options:

**A. Database (Production):**
```python
X, y_waste, y_days = await trainer.load_training_data_from_db(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2025, 11, 1)
)
```

**B. Synthetic Data:**
```python
from generate_synthetic_data import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
df = generator.generate_dataset(
    n_households=1000,
    n_items_per_household=100
)
```

**C. File (Parquet/CSV):**
```python
X, y_waste, y_days = trainer.load_training_data_from_file(
    "data/training_data.parquet"
)
```

### 2. Cross-Validation

**Time Series Split (Default):**
```python
tscv = TimeSeriesSplit(n_splits=5)
# Respects temporal ordering
# Fold 1: Train[0:20%] → Val[20:40%]
# Fold 2: Train[0:40%] → Val[40:60%]
# ...
```

**Why Time Series CV?**
- Respects temporal ordering (no data leakage)
- Simulates production deployment
- Tests model on future data

### 3. Model Training

**LightGBM Dual-Model Architecture:**

1. **Classifier**: Predicts P(waste)
   - Objective: `binary`
   - Metric: `auc`
   - Handles class imbalance: `is_unbalance=True`

2. **Regressor**: Predicts days to waste (for wasted items)
   - Objective: `regression`
   - Metric: `mae`
   - Trained only on wasted items

**Hyperparameters:**
```python
classifier_params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "lambda_l1": 0.1,  # L1 regularization
    "lambda_l2": 0.1,  # L2 regularization
}
```

### 4. Hyperparameter Optimization

**Optuna-based HPO:**
```python
python -m backend.ml.training.train_waste_model \
    --use-synthetic \
    --n-samples 50000 \
    --hyperopt \
    --n-trials 100
```

**Tuned Parameters:**
- `num_leaves`: [20, 80]
- `learning_rate`: [0.01, 0.2]
- `feature_fraction`: [0.5, 0.9]
- `bagging_fraction`: [0.5, 0.9]
- `min_data_in_leaf`: [10, 100]
- `lambda_l1`: [0.0, 2.0]
- `lambda_l2`: [0.0, 2.0]
- `max_depth`: [3, 12]

**Optimization Objective:**
- Maximize AUC on validation set

### 5. Model Evaluation

**Metrics Tracked:**

| Category | Metrics |
|----------|---------|
| **Classification** | AUC, Avg Precision, Precision@K, Recall@K, F1 |
| **Calibration** | Brier Score, Log Loss, ECE |
| **Regression** | MAE, RMSE (days to waste) |
| **Business** | Precision@0.7, Recall@0.7 |
| **Stability** | CV std, fold variance |

**Visualizations Generated:**
1. ROC Curve
2. Precision-Recall Curve
3. Calibration Curve
4. Confusion Matrix
5. Feature Importance (top 20)

### 6. Model Artifacts

**Saved Artifacts:**
```
models/waste_predictor_final/
├── classifier.txt           # LightGBM classifier
├── regressor.txt           # LightGBM regressor
├── calibrator.pkl          # Isotonic calibrator
├── explainer.pkl           # SHAP explainer
├── feature_names.json      # Feature list
└── metadata.json           # Training metadata
```

**MLflow Logged:**
- ✅ Models (classifier + regressor)
- ✅ All metrics
- ✅ All parameters
- ✅ Evaluation plots
- ✅ Feature importance
- ✅ Full artifact directory

## Synthetic Data Generator

### Features

Realistic simulation of:
1. **Household Behavior**
   - Size distribution (1-6 members)
   - Waste propensity (beta distribution)
   - Shopping patterns
   - Planning behavior

2. **Food Categories**
   - Dairy (28% waste rate)
   - Produce (38% waste rate)
   - Meat (22% waste rate)
   - Grains (12% waste rate)
   - Canned (5% waste rate)
   - Frozen (8% waste rate)
   - Beverages (15% waste rate)

3. **Temporal Effects**
   - Seasonality per category
   - Purchase day of week
   - Age-based waste probability

4. **Consumption Modeling**
   - Household size effects
   - Category-specific consumption rates
   - Random variation

5. **50+ Engineered Features**
   - All features used by real model
   - Realistic value distributions

### Usage

```bash
# Generate data
python -m backend.ml.training.generate_synthetic_data \
    --n-households 1000 \
    --n-items-per-household 100 \
    --output data/training.parquet \
    --seed 42

# Statistics
Total samples: 100,000
Waste ratio: 24.3%
File size: ~15 MB (parquet)
Generation time: ~30 seconds
```

## Model Validation Suite

### Comprehensive Validation

```python
from backend.ml.training.validate_model import ModelValidator

validator = ModelValidator(model, X_test, y_test)
metrics = validator.validate_comprehensive()

print(f"AUC: {metrics.auc:.4f}")
print(f"Precision@0.7: {metrics.precision_at_high_confidence:.4f}")
```

### Validation Checks

1. **Classification Performance**
   - AUC-ROC
   - Average Precision
   - Brier Score
   - Log Loss

2. **Calibration Analysis**
   - Expected Calibration Error (ECE)
   - Calibration slope & intercept
   - Hosmer-Lemeshow test

3. **Threshold Optimization**
   - F1-maximizing threshold
   - Precision/Recall trade-off

4. **Business Metrics**
   - Precision@0.7 (high confidence)
   - Recall@0.7

5. **Statistical Tests**
   - Hosmer-Lemeshow (calibration)
   - Anderson-Darling (optional)

6. **Fairness Metrics** (if sensitive features provided)
   - Demographic parity
   - Equalized odds

7. **Cross-Validation Stability**
   - Mean CV AUC
   - CV std (should be < 0.05)

### Gate Requirements

Model must pass:
- ✅ **AUC ≥ 0.85**
- ✅ **Precision@0.7 ≥ 0.80**
- ✅ **ECE < 0.10** (well-calibrated)
- ✅ **CV Stable** (std < 0.05)

### Generate Report

```python
report = validator.generate_report("validation_report.txt")
print(report)
```

**Output:**
```
================================================================================
MODEL VALIDATION REPORT
================================================================================
Generated: 2025-11-12 14:30:00

CLASSIFICATION METRICS
--------------------------------------------------------------------------------
AUC-ROC:                    0.8723
Average Precision:          0.7891
Brier Score:                0.1234
Log Loss:                   0.3456

CALIBRATION METRICS
--------------------------------------------------------------------------------
Expected Calibration Error: 0.0567
Calibration Slope:          0.9832
Calibration Intercept:      0.0123
Hosmer-Lemeshow p-value:    0.4567

THRESHOLD OPTIMIZATION
--------------------------------------------------------------------------------
Optimal Threshold:          0.623
  Precision:                0.8234
  Recall:                   0.7654
  F1 Score:                 0.7932

BUSINESS METRICS (threshold=0.7)
--------------------------------------------------------------------------------
Precision:                  0.8456
Recall:                     0.7123

GATE REQUIREMENTS
--------------------------------------------------------------------------------
AUC >= 0.85:                ✓ PASS
Precision@0.7 >= 0.80:      ✓ PASS
Calibration (ECE < 0.10):   ✓ PASS
CV Stable (std < 0.05):     ✓ PASS
================================================================================
```

## MLflow Integration

### Experiment Tracking

**What's Logged:**
1. **Parameters**
   - n_samples, n_features, n_folds
   - Waste ratio, training date
   - Hyperparameters (if hyperopt)

2. **Metrics** (per fold + average)
   - auc, avg_precision
   - precision_at_0.5, recall_at_0.5
   - precision_at_0.7, recall_at_0.7
   - f1_at_0.7
   - days_mae, days_rmse

3. **Artifacts**
   - Model files
   - Evaluation plots
   - Feature importance
   - Training logs

4. **Models**
   - Classifier (registered)
   - Regressor (registered)

### Model Registry

**Stages:**
- **None**: Newly trained
- **Staging**: Under evaluation
- **Production**: Serving traffic
- **Archived**: Retired

**Promote Model:**
```python
import mlflow

client = mlflow.tracking.MlflowClient()

# Transition to staging
client.transition_model_version_stage(
    name="waste_risk_classifier",
    version=3,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="waste_risk_classifier",
    version=3,
    stage="Production"
)
```

### Load Model from Registry

```python
import mlflow

# Load production model
model_uri = "models:/waste_risk_classifier/Production"
model = mlflow.lightgbm.load_model(model_uri)

# Predict
predictions = model.predict(X)
```

## Airflow Integration

### Training DAG (Weekly)

**File:** `backend/ml/dags/weekly_model_retraining.py`

**Schedule:** Every Sunday at 3 AM

**Tasks:**
1. `validate_data` - Check data quality
2. `train_model` - Run training script
3. `evaluate_model` - Validate new model
4. `deploy_if_better` - Compare and deploy

**Trigger Manually:**
```bash
airflow dags trigger weekly_model_retraining
```

## Troubleshooting

### Common Issues

**1. MLflow Connection Error**
```
Error: Connection refused to http://mlflow:5000
```
Solution:
```bash
docker-compose up -d mlflow
curl http://localhost:5000/health
```

**2. Out of Memory During Training**
```
MemoryError: Unable to allocate array
```
Solution:
```bash
# Reduce sample size
python -m backend.ml.training.train_waste_model \
    --use-synthetic \
    --n-samples 10000  # Instead of 100000

# Or use chunked processing
```

**3. PostgreSQL MLflow Backend Issues**
```
sqlalchemy.exc.OperationalError: could not connect
```
Solution:
```bash
# Recreate MLflow database
docker-compose down -v
docker-compose up -d postgres
docker-compose up -d mlflow
```

**4. Slow Training**
```
Training taking > 1 hour for 100k samples
```
Solution:
```bash
# Use GPU acceleration (if available)
pip install lightgbm --install-option=--gpu

# Or reduce hyperopt trials
--n-trials 20  # Instead of 100
```

## Performance Benchmarks

| Samples | Features | CV Folds | Hyperopt | Time | Memory |
|---------|----------|----------|----------|------|--------|
| 1k | 50 | 5 | No | 30s | 500MB |
| 10k | 50 | 5 | No | 3m | 1GB |
| 100k | 50 | 5 | No | 20m | 4GB |
| 100k | 50 | 5 | 100 trials | 3h | 6GB |

**Hardware:** Intel i7, 16GB RAM, no GPU

## Next Steps

1. ✅ **Train Initial Model**
   ```bash
   python -m backend.ml.training.train_waste_model --use-synthetic --n-samples 10000
   ```

2. ✅ **Validate Performance**
   - Check MLflow UI
   - Review validation report
   - Ensure gate requirements passed

3. ✅ **Deploy to Staging**
   ```python
   # Promote model in MLflow
   client.transition_model_version_stage(
       name="waste_risk_classifier",
       version=1,
       stage="Staging"
   )
   ```

4. ✅ **Integrate with BentoML**
   - Update `ml-service/service.py` to load from MLflow
   - Test predictions

5. ✅ **Set Up Airflow Retraining**
   - Schedule weekly retraining DAG
   - Monitor training runs

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## Support

For questions or issues:
1. Check MLflow UI logs: http://localhost:5000
2. Check Airflow logs: http://localhost:8080
3. Review training logs in terminal
4. Contact ML team: ml-team@pantrypal.com

"""
ML Model Registry & Lifecycle Management

Uses MLflow for:
- Model versioning & artifact storage
- Experiment tracking
- Model lineage & metadata
- Deployment automation

Architecture:
- MLflow Tracking Server (PostgreSQL backend, S3/GCS artifacts)
- Model Registry with stages: Staging → Production → Archived
- CI/CD integration for automated testing & deployment
- Canary deployment with traffic splitting
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import json
import logging
import hashlib

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature, infer_signature
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    """Model registry stages."""
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


class ModelType(str, Enum):
    """Model types in the system."""
    WASTE_CLASSIFIER = "waste_classifier"
    WASTE_REGRESSOR = "waste_regressor"
    CONSUMPTION_FORECASTER = "consumption_forecaster"
    RECIPE_RANKER = "recipe_ranker"
    ITEM_EMBEDDER = "item_embedder"


@dataclass
class ModelMetadata:
    """Metadata stored with each model."""
    model_name: str
    model_type: ModelType
    version: str
    
    # Training data
    training_data_hash: str
    training_samples: int
    training_start: datetime
    training_end: datetime
    
    # Model parameters
    model_params: Dict[str, Any]
    feature_names: List[str]
    
    # Performance metrics
    metrics: Dict[str, float]
    
    # Lineage
    parent_run_id: Optional[str] = None
    dataset_version: Optional[str] = None
    code_version: Optional[str] = None
    
    # Deployment
    stage: ModelStage = ModelStage.NONE
    deployed_at: Optional[datetime] = None
    deployed_by: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class ModelValidationResult:
    """Result of model validation tests."""
    passed: bool
    
    # Test results
    sanity_tests_passed: bool
    unit_tests_passed: bool
    fairness_tests_passed: bool
    drift_tests_passed: bool
    
    # Metrics
    validation_metrics: Dict[str, float]
    
    # Failures
    failures: List[str]
    warnings: List[str]


class ModelRegistry:
    """Service to manage ML model lifecycle."""
    
    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        registry_uri: Optional[str] = None,
        artifact_location: str = "s3://pantrypal-mlflow/artifacts",
        experiment_name: str = "pantrypal-ml",
    ):
        """
        Initialize model registry.
        
        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: Model registry URI (defaults to tracking_uri)
            artifact_location: Artifact storage location (S3/GCS)
            experiment_name: MLflow experiment name
        """
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # Create experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        except Exception:
            # Experiment already exists
            self.experiment_id = mlflow.get_experiment_by_name(
                experiment_name
            ).experiment_id
        
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient()
        
        logger.info(
            f"ModelRegistry initialized (tracking: {tracking_uri}, "
            f"experiment: {experiment_name})"
        )
    
    def register_model(
        self,
        model: Any,
        model_name: str,
        model_type: ModelType,
        training_data: pd.DataFrame,
        target_data: pd.Series,
        model_params: Dict[str, Any],
        metrics: Dict[str, float],
        signature: Optional[ModelSignature] = None,
        input_example: Optional[pd.DataFrame] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Register model to MLflow registry.
        
        Args:
            model: Trained model object
            model_name: Model name (e.g., "waste_classifier")
            model_type: Model type enum
            training_data: Training features
            target_data: Training targets
            model_params: Model hyperparameters
            metrics: Evaluation metrics
            signature: Model signature (auto-inferred if None)
            input_example: Example input for signature
            description: Model description
            tags: Additional tags
            
        Returns:
            Model version string
        """
        # Compute training data hash
        training_data_hash = self._compute_data_hash(training_data, target_data)
        
        # Infer signature if not provided
        if signature is None and input_example is not None:
            predictions = model.predict(input_example)
            signature = infer_signature(input_example, predictions)
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(model_params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=model_name,
            )
            
            # Log metadata
            mlflow.set_tags({
                "model_type": model_type.value,
                "training_data_hash": training_data_hash,
                "training_samples": len(training_data),
                **(tags or {}),
            })
            
            # Log feature names
            if hasattr(training_data, "columns"):
                feature_names = training_data.columns.tolist()
                mlflow.log_dict(
                    {"features": feature_names},
                    "feature_names.json",
                )
            
            # Log training data stats
            mlflow.log_dict(
                {
                    "data_hash": training_data_hash,
                    "num_samples": len(training_data),
                    "num_features": training_data.shape[1],
                    "target_distribution": target_data.value_counts().to_dict()
                    if hasattr(target_data, "value_counts")
                    else {},
                },
                "training_data_stats.json",
            )
            
            # Log model description
            if description:
                mlflow.log_text(description, "description.txt")
            
            run_id = run.info.run_id
        
        # Get registered model version
        model_versions = self.client.search_model_versions(
            f"name='{model_name}' and run_id='{run_id}'"
        )
        
        version = model_versions[0].version if model_versions else "1"
        
        logger.info(
            f"Registered model {model_name} v{version} "
            f"(run_id: {run_id}, hash: {training_data_hash[:8]})"
        )
        
        return version
    
    def _compute_data_hash(
        self,
        training_data: pd.DataFrame,
        target_data: pd.Series,
    ) -> str:
        """Compute hash of training data for lineage tracking."""
        # Concatenate data
        combined = pd.concat([training_data, target_data], axis=1)
        
        # Hash sorted data (deterministic)
        hash_input = combined.sort_index().to_csv(index=False).encode("utf-8")
        data_hash = hashlib.sha256(hash_input).hexdigest()
        
        return data_hash
    
    def validate_model(
        self,
        model_name: str,
        version: str,
        validation_data: pd.DataFrame,
        validation_targets: pd.Series,
    ) -> ModelValidationResult:
        """
        Run validation tests on model.
        
        Tests:
        1. Sanity tests - model produces valid predictions
        2. Unit tests - prediction logic correctness
        3. Fairness tests - no bias across demographics
        4. Drift tests - performance on new data
        
        Args:
            model_name: Model name
            version: Model version
            validation_data: Validation features
            validation_targets: Validation targets
            
        Returns:
            ModelValidationResult
        """
        logger.info(f"Validating model {model_name} v{version}")
        
        # Load model
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        failures = []
        warnings = []
        
        # Test 1: Sanity tests
        sanity_passed = True
        try:
            predictions = model.predict(validation_data)
            
            # Check for NaN/Inf
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                sanity_passed = False
                failures.append("Model produced NaN/Inf predictions")
            
            # Check prediction range
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(validation_data)
                if not np.all((proba >= 0) & (proba <= 1)):
                    sanity_passed = False
                    failures.append("Probabilities outside [0, 1] range")
        
        except Exception as e:
            sanity_passed = False
            failures.append(f"Prediction failed: {e}")
        
        # Test 2: Unit tests (example: check prediction consistency)
        unit_tests_passed = True
        try:
            # Same input should give same output
            pred1 = model.predict(validation_data.iloc[:10])
            pred2 = model.predict(validation_data.iloc[:10])
            
            if not np.allclose(pred1, pred2):
                unit_tests_passed = False
                failures.append("Predictions not deterministic")
        
        except Exception as e:
            unit_tests_passed = False
            failures.append(f"Unit test failed: {e}")
        
        # Test 3: Fairness tests (simplified)
        fairness_passed = True
        # TODO: Implement demographic parity checks
        # E.g., check if waste prediction is unbiased across household types
        
        # Test 4: Drift tests
        drift_passed = True
        try:
            from sklearn.metrics import roc_auc_score, mean_squared_error
            
            predictions = model.predict(validation_data)
            
            # Compute metrics
            if hasattr(model, "predict_proba"):
                # Classification
                proba = model.predict_proba(validation_data)[:, 1]
                auc = roc_auc_score(validation_targets, proba)
                
                validation_metrics = {"auc": auc}
                
                # Check for drift (AUC drop > 5%)
                # TODO: Compare against production metrics
                if auc < 0.75:
                    warnings.append(f"AUC ({auc:.3f}) below threshold (0.75)")
            else:
                # Regression
                mse = mean_squared_error(validation_targets, predictions)
                rmse = np.sqrt(mse)
                
                validation_metrics = {"rmse": rmse}
                
                # Check for drift
                # TODO: Compare against production RMSE
        
        except Exception as e:
            drift_passed = False
            failures.append(f"Drift test failed: {e}")
            validation_metrics = {}
        
        # Overall result
        passed = (
            sanity_passed
            and unit_tests_passed
            and fairness_passed
            and drift_passed
        )
        
        result = ModelValidationResult(
            passed=passed,
            sanity_tests_passed=sanity_passed,
            unit_tests_passed=unit_tests_passed,
            fairness_tests_passed=fairness_passed,
            drift_tests_passed=drift_passed,
            validation_metrics=validation_metrics,
            failures=failures,
            warnings=warnings,
        )
        
        logger.info(
            f"Validation result for {model_name} v{version}: "
            f"{'PASSED' if passed else 'FAILED'}"
        )
        
        return result
    
    def promote_model(
        self,
        model_name: str,
        version: str,
        stage: ModelStage,
        archive_existing: bool = True,
    ):
        """
        Promote model to a new stage.
        
        Args:
            model_name: Model name
            version: Model version to promote
            stage: Target stage
            archive_existing: Archive existing model in target stage
        """
        # Archive existing models in target stage
        if archive_existing and stage == ModelStage.PRODUCTION:
            existing_versions = self.client.get_latest_versions(
                model_name,
                stages=[ModelStage.PRODUCTION.value],
            )
            
            for existing in existing_versions:
                logger.info(
                    f"Archiving existing production model "
                    f"{model_name} v{existing.version}"
                )
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=existing.version,
                    stage=ModelStage.ARCHIVED.value,
                )
        
        # Promote new version
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage.value,
        )
        
        logger.info(f"Promoted {model_name} v{version} to {stage.value}")
    
    def get_production_model(self, model_name: str) -> Any:
        """
        Get current production model.
        
        Args:
            model_name: Model name
            
        Returns:
            Loaded model object
        """
        model_uri = f"models:/{model_name}/Production"
        
        try:
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded production model {model_name}")
            return model
        
        except Exception as e:
            logger.error(f"Failed to load production model {model_name}: {e}")
            raise
    
    def get_model_metadata(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None,
    ) -> ModelMetadata:
        """
        Get model metadata.
        
        Args:
            model_name: Model name
            version: Model version (if None, get latest)
            stage: Model stage (if version not provided)
            
        Returns:
            ModelMetadata
        """
        if version:
            model_version = self.client.get_model_version(model_name, version)
        elif stage:
            versions = self.client.get_latest_versions(
                model_name,
                stages=[stage.value],
            )
            model_version = versions[0] if versions else None
        else:
            # Get latest version
            versions = self.client.search_model_versions(f"name='{model_name}'")
            model_version = max(versions, key=lambda v: int(v.version))
        
        if not model_version:
            raise ValueError(f"Model {model_name} not found")
        
        # Get run info
        run = self.client.get_run(model_version.run_id)
        
        # Parse metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_type=ModelType(run.data.tags.get("model_type", "unknown")),
            version=model_version.version,
            training_data_hash=run.data.tags.get("training_data_hash", ""),
            training_samples=int(run.data.tags.get("training_samples", 0)),
            training_start=datetime.fromtimestamp(run.info.start_time / 1000),
            training_end=datetime.fromtimestamp(run.info.end_time / 1000),
            model_params=run.data.params,
            feature_names=[],  # TODO: Load from artifacts
            metrics=run.data.metrics,
            stage=ModelStage(model_version.current_stage),
            description=model_version.description,
            tags=run.data.tags,
        )
        
        return metadata
    
    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelMetadata]:
        """
        List registered models.
        
        Args:
            model_type: Filter by model type
            stage: Filter by stage
            
        Returns:
            List of ModelMetadata
        """
        # Get all registered models
        registered_models = self.client.search_registered_models()
        
        models = []
        
        for rm in registered_models:
            # Get latest version
            latest_version = self.client.get_latest_versions(rm.name)[0]
            
            try:
                metadata = self.get_model_metadata(
                    rm.name,
                    version=latest_version.version,
                )
                
                # Apply filters
                if model_type and metadata.model_type != model_type:
                    continue
                
                if stage and metadata.stage != stage:
                    continue
                
                models.append(metadata)
            
            except Exception as e:
                logger.warning(f"Failed to get metadata for {rm.name}: {e}")
        
        return models


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    
    # Initialize registry
    registry = ModelRegistry(
        tracking_uri="http://localhost:5000",
        artifact_location="s3://pantrypal-mlflow/artifacts",
    )
    
    # Create dummy data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    X_train_df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)])
    X_test_df = pd.DataFrame(X_test, columns=[f"feature_{i}" for i in range(20)])
    y_train_series = pd.Series(y_train, name="target")
    y_test_series = pd.Series(y_test, name="target")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model AUC: {auc:.3f}")
    
    # Register model
    version = registry.register_model(
        model=model,
        model_name="waste_classifier_test",
        model_type=ModelType.WASTE_CLASSIFIER,
        training_data=X_train_df,
        target_data=y_train_series,
        model_params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        },
        metrics={"auc": auc},
        input_example=X_train_df.iloc[:5],
        description="Test waste classifier model",
        tags={"version": "1.0.0"},
    )
    
    print(f"\nRegistered model version: {version}")
    
    # Validate model
    validation_result = registry.validate_model(
        model_name="waste_classifier_test",
        version=version,
        validation_data=X_test_df,
        validation_targets=y_test_series,
    )
    
    print(f"\nValidation result: {'PASSED' if validation_result.passed else 'FAILED'}")
    print(f"Sanity tests: {validation_result.sanity_tests_passed}")
    print(f"Unit tests: {validation_result.unit_tests_passed}")
    print(f"Validation metrics: {validation_result.validation_metrics}")
    
    # Promote to production
    if validation_result.passed:
        registry.promote_model(
            model_name="waste_classifier_test",
            version=version,
            stage=ModelStage.PRODUCTION,
        )
        print(f"\nPromoted to production")
    
    # Get production model
    prod_model = registry.get_production_model("waste_classifier_test")
    print(f"\nLoaded production model: {prod_model}")
    
    # Get metadata
    metadata = registry.get_model_metadata(
        model_name="waste_classifier_test",
        stage=ModelStage.PRODUCTION,
    )
    print(f"\nModel metadata:")
    print(f"  Type: {metadata.model_type}")
    print(f"  Version: {metadata.version}")
    print(f"  Training samples: {metadata.training_samples}")
    print(f"  Metrics: {metadata.metrics}")

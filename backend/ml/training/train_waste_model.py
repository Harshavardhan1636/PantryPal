"""
Production Training Script for Waste Risk Prediction Model.

Features:
- MLflow experiment tracking and model registry
- Cross-validation with time series splits
- Hyperparameter optimization with Optuna
- Model versioning and artifact storage
- Comprehensive evaluation metrics
- SHAP feature importance analysis

Usage:
    # Basic training
    python -m backend.ml.training.train_waste_model \\
        --start-date 2024-01-01 \\
        --end-date 2025-11-01

    # With hyperparameter optimization
    python -m backend.ml.training.train_waste_model \\
        --start-date 2024-01-01 \\
        --end-date 2025-11-01 \\
        --hyperopt \\
        --n-trials 100

    # Using synthetic data
    python -m backend.ml.training.train_waste_model \\
        --use-synthetic \\
        --n-samples 10000

Author: Senior SDE-3
Date: 2025-11-12
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Add backend to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from ml.models.waste_predictor import WastePredictor, WasteModelArtifacts
from ml.features.feature_engineer import FeatureEngineer
from ml.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_ARTIFACT_LOCATION,
    EVALUATION_METRICS,
    ENGINEERED_FEATURES,
)
from shared.database import get_async_session
from sqlalchemy import text


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WasteModelTrainer:
    """Production trainer for waste risk prediction models."""
    
    def __init__(
        self,
        experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
    ):
        """
        Initialize trainer.
        
        Args:
            experiment_name: MLflow experiment name
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.experiment_name = experiment_name
        
        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        logger.info(f"Initialized WasteModelTrainer")
        logger.info(f"  Experiment: {experiment_name}")
        logger.info(f"  Tracking URI: {mlflow_tracking_uri}")
    
    async def load_training_data_from_db(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load training data from database.
        
        Args:
            start_date: Training data start date
            end_date: Training data end date
            
        Returns:
            (X, y_waste, y_days) - Features, waste labels, days to waste
        """
        logger.info(f"Loading training data from database")
        logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
        
        async with get_async_session() as db:
            feature_engineer = FeatureEngineer(db)
            
            # Query all pantry entries in date range with waste events
            query = text("""
                SELECT 
                    pe.entry_id,
                    pe.purchase_date,
                    we.waste_event_id IS NOT NULL as was_wasted,
                    CASE 
                        WHEN we.waste_event_id IS NOT NULL 
                        THEN EXTRACT(EPOCH FROM (we.waste_date - pe.purchase_date)) / 86400
                        ELSE NULL
                    END as days_to_waste
                FROM pantry_entries pe
                LEFT JOIN waste_events we ON pe.entry_id = we.entry_id
                WHERE pe.purchase_date BETWEEN :start_date AND :end_date
                AND pe.status IN ('consumed', 'wasted', 'expired')
                ORDER BY pe.purchase_date
            """)
            
            result = await db.execute(query, {
                "start_date": start_date,
                "end_date": end_date,
            })
            
            entries = result.fetchall()
            logger.info(f"Found {len(entries)} pantry entries")
            
            if len(entries) == 0:
                raise ValueError("No training data found in database for specified date range")
            
            # Extract features for each entry
            X_list = []
            y_waste_list = []
            y_days_list = []
            failed_count = 0
            
            logger.info("Extracting features...")
            for idx, (entry_id, purchase_date, was_wasted, days_to_waste) in enumerate(entries):
                if idx % 100 == 0 and idx > 0:
                    logger.info(f"  Processed {idx}/{len(entries)} entries")
                
                try:
                    # Use purchase date as reference for historical simulation
                    reference_date = purchase_date + timedelta(days=7)  # 7 days after purchase
                    
                    features = await feature_engineer.extract_features(
                        entry_id,
                        reference_date=reference_date
                    )
                    
                    X_list.append(features.features)
                    y_waste_list.append(1 if was_wasted else 0)
                    y_days_list.append(days_to_waste if days_to_waste else 0)
                    
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 5:  # Log first 5 failures
                        logger.warning(f"Failed to extract features for {entry_id}: {e}")
            
            if failed_count > 0:
                logger.warning(f"Failed to extract features for {failed_count} entries")
            
            # Convert to arrays
            X = pd.DataFrame(X_list)
            y_waste = np.array(y_waste_list)
            y_days = np.array(y_days_list)
            
            logger.info(f"Feature extraction complete:")
            logger.info(f"  Samples: {len(X)}")
            logger.info(f"  Features: {X.shape[1]}")
            logger.info(f"  Waste ratio: {y_waste.mean():.2%}")
            logger.info(f"  Wasted items: {y_waste.sum()}")
            
            return X, y_waste, y_days
    
    def load_training_data_from_file(
        self,
        data_path: str,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load training data from file (parquet, csv).
        
        Args:
            data_path: Path to data file
            
        Returns:
            (X, y_waste, y_days) - Features, waste labels, days to waste
        """
        logger.info(f"Loading training data from file: {data_path}")
        
        # Load data
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Separate features and labels
        y_waste = df['was_wasted'].values
        y_days = df['days_to_waste'].fillna(0).values
        
        # Get feature columns
        feature_cols = [col for col in ENGINEERED_FEATURES if col in df.columns]
        X = df[feature_cols]
        
        logger.info(f"Data loaded:")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Features: {X.shape[1]}")
        logger.info(f"  Waste ratio: {y_waste.mean():.2%}")
        
        return X, y_waste, y_days
    
    def train_with_cross_validation(
        self,
        X: pd.DataFrame,
        y_waste: np.ndarray,
        y_days: np.ndarray,
        n_folds: int = 5,
        hyperopt: bool = False,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Train model with cross-validation and MLflow tracking.
        
        Args:
            X: Feature DataFrame
            y_waste: Binary waste labels (0=no waste, 1=waste)
            y_days: Days to waste (for wasted items)
            n_folds: Number of CV folds
            hyperopt: Whether to run hyperparameter optimization
            n_trials: Number of Optuna trials
            
        Returns:
            Training results dictionary
        """
        logger.info("="*80)
        logger.info("Starting model training with cross-validation")
        logger.info("="*80)
        
        with mlflow.start_run(run_name=f"waste_model_{datetime.now():%Y%m%d_%H%M%S}") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")
            
            # Log parameters
            mlflow.log_param("n_samples", len(X))
            mlflow.log_param("n_features", X.shape[1])
            mlflow.log_param("n_folds", n_folds)
            mlflow.log_param("hyperopt", hyperopt)
            mlflow.log_param("waste_ratio", y_waste.mean())
            mlflow.log_param("training_date", datetime.now().isoformat())
            
            # Log feature names
            mlflow.log_param("feature_names", ",".join(X.columns.tolist()))
            
            # Hyperparameter optimization
            best_params = {}
            if hyperopt:
                logger.info(f"Running hyperparameter optimization ({n_trials} trials)...")
                best_params = self._hyperparameter_optimization(
                    X, y_waste, y_days, n_trials
                )
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            
            # Cross-validation
            logger.info(f"Starting {n_folds}-fold time series cross-validation...")
            tscv = TimeSeriesSplit(n_splits=n_folds)
            
            fold_metrics = []
            fold_models = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                logger.info(f"\nTraining fold {fold_idx + 1}/{n_folds}")
                logger.info(f"  Train samples: {len(train_idx)}")
                logger.info(f"  Val samples: {len(val_idx)}")
                
                # Split data
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_waste_train_fold = y_waste[train_idx]
                y_waste_val_fold = y_waste[val_idx]
                y_days_train_fold = y_days[train_idx]
                y_days_val_fold = y_days[val_idx]
                
                # Train model
                predictor = WastePredictor()
                artifacts = predictor.train(
                    X_train_fold,
                    y_waste_train_fold,
                    y_days_train_fold,
                    X_val_fold,
                    y_waste_val_fold,
                    y_days_val_fold,
                    feature_names=list(X.columns),
                )
                
                # Evaluate
                val_metrics = self._evaluate_model(
                    predictor,
                    X_val_fold,
                    y_waste_val_fold,
                    y_days_val_fold,
                    fold_idx=fold_idx,
                )
                
                fold_metrics.append(val_metrics)
                fold_models.append(artifacts)
                
                # Log fold metrics
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"fold_{fold_idx}_{metric_name}", value)
                
                logger.info(f"  Fold {fold_idx + 1} AUC: {val_metrics['auc']:.4f}")
                logger.info(f"  Fold {fold_idx + 1} Precision@0.7: {val_metrics['precision_at_0.7']:.4f}")
            
            # Aggregate CV metrics
            logger.info("\nAggregating cross-validation metrics...")
            avg_metrics = self._aggregate_fold_metrics(fold_metrics)
            
            # Log average CV metrics
            for metric_name, value in avg_metrics.items():
                mlflow.log_metric(f"cv_{metric_name}", value)
            
            logger.info("\nCross-validation results:")
            logger.info(f"  CV AUC: {avg_metrics['auc']:.4f} ± {avg_metrics['auc_std']:.4f}")
            logger.info(f"  CV Precision@0.7: {avg_metrics['precision_at_0.7']:.4f}")
            logger.info(f"  CV Recall@0.7: {avg_metrics['recall_at_0.7']:.4f}")
            logger.info(f"  CV Days MAE: {avg_metrics['days_mae']:.2f}")
            
            # Train final model on full dataset
            logger.info("\n" + "="*80)
            logger.info("Training final model on full dataset...")
            logger.info("="*80)
            
            train_size = int(0.8 * len(X))
            X_train_final = X.iloc[:train_size]
            X_val_final = X.iloc[train_size:]
            y_waste_train_final = y_waste[:train_size]
            y_waste_val_final = y_waste[train_size:]
            y_days_train_final = y_days[:train_size]
            y_days_val_final = y_days[train_size:]
            
            predictor_final = WastePredictor()
            artifacts_final = predictor_final.train(
                X_train_final,
                y_waste_train_final,
                y_days_train_final,
                X_val_final,
                y_waste_val_final,
                y_days_val_final,
                feature_names=list(X.columns),
            )
            
            # Final evaluation
            final_metrics = self._evaluate_model(
                predictor_final,
                X_val_final,
                y_waste_val_final,
                y_days_val_final,
                fold_idx="final",
            )
            
            for metric_name, value in final_metrics.items():
                mlflow.log_metric(f"final_{metric_name}", value)
            
            logger.info("\nFinal model performance:")
            logger.info(f"  AUC: {final_metrics['auc']:.4f}")
            logger.info(f"  Precision@0.7: {final_metrics['precision_at_0.7']:.4f}")
            logger.info(f"  Recall@0.7: {final_metrics['recall_at_0.7']:.4f}")
            
            # Generate visualizations
            logger.info("\nGenerating evaluation plots...")
            self._generate_evaluation_plots(
                predictor_final,
                X_val_final,
                y_waste_val_final,
                y_days_val_final,
            )
            
            # Save model artifacts
            logger.info("\nSaving model artifacts...")
            model_path = "models/waste_predictor_final"
            predictor_final.save_model(model_path)
            
            # Log models to MLflow
            logger.info("Logging models to MLflow...")
            
            # Log classifier
            mlflow.lightgbm.log_model(
                artifacts_final.classifier,
                "classifier",
                registered_model_name="waste_risk_classifier",
            )
            
            # Log regressor
            mlflow.lightgbm.log_model(
                artifacts_final.regressor,
                "regressor",
                registered_model_name="waste_days_regressor",
            )
            
            # Log full artifacts
            mlflow.log_artifacts(model_path)
            
            # Log feature importance
            self._log_feature_importance(artifacts_final.classifier, X.columns)
            
            logger.info("\n" + "="*80)
            logger.info("TRAINING COMPLETE")
            logger.info("="*80)
            logger.info(f"MLflow Run ID: {run_id}")
            logger.info(f"CV AUC: {avg_metrics['auc']:.4f}")
            logger.info(f"Final AUC: {final_metrics['auc']:.4f}")
            logger.info(f"View results: {MLFLOW_TRACKING_URI}/#/experiments")
            logger.info("="*80)
            
            return {
                "cv_metrics": avg_metrics,
                "final_metrics": final_metrics,
                "final_model": artifacts_final,
                "mlflow_run_id": run_id,
                "fold_models": fold_models,
            }
    
    def _evaluate_model(
        self,
        predictor: WastePredictor,
        X: pd.DataFrame,
        y_waste: np.ndarray,
        y_days: np.ndarray,
        fold_idx: Any = None,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictor: Trained predictor
            X: Feature DataFrame
            y_waste: True waste labels
            y_days: True days to waste
            fold_idx: Fold index for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predict waste probabilities
        predictions = []
        for idx in range(len(X)):
            try:
                result = predictor.predict(
                    X.iloc[[idx]],
                    f"entry_{idx}",
                    return_shap=False,
                )
                predictions.append(result.waste_probability)
            except Exception as e:
                logger.warning(f"Prediction failed for index {idx}: {e}")
                predictions.append(0.5)  # Default
        
        y_pred_proba = np.array(predictions)
        
        # Classification metrics
        auc = roc_auc_score(y_waste, y_pred_proba)
        avg_precision = average_precision_score(y_waste, y_pred_proba)
        
        # Precision/Recall at different thresholds
        precision_at_05 = self._precision_at_threshold(y_waste, y_pred_proba, 0.5)
        recall_at_05 = self._recall_at_threshold(y_waste, y_pred_proba, 0.5)
        
        precision_at_07 = self._precision_at_threshold(y_waste, y_pred_proba, 0.7)
        recall_at_07 = self._recall_at_threshold(y_waste, y_pred_proba, 0.7)
        
        # F1 score at 0.7 threshold
        if precision_at_07 + recall_at_07 > 0:
            f1_at_07 = 2 * (precision_at_07 * recall_at_07) / (precision_at_07 + recall_at_07)
        else:
            f1_at_07 = 0.0
        
        # Regression metrics (for wasted items)
        wasted_mask = y_waste == 1
        mae = 0.0
        rmse = 0.0
        
        if wasted_mask.sum() > 0:
            days_pred = []
            for idx in np.where(wasted_mask)[0]:
                try:
                    result = predictor.predict(
                        X.iloc[[idx]],
                        f"entry_{idx}",
                        return_shap=False,
                    )
                    days_pred.append(result.days_until_waste or 0)
                except:
                    days_pred.append(7)  # Default
            
            days_pred = np.array(days_pred)
            days_true = y_days[wasted_mask]
            
            mae = mean_absolute_error(days_true, days_pred)
            rmse = np.sqrt(mean_squared_error(days_true, days_pred))
        
        return {
            "auc": auc,
            "avg_precision": avg_precision,
            "precision_at_0.5": precision_at_05,
            "recall_at_0.5": recall_at_05,
            "precision_at_0.7": precision_at_07,
            "recall_at_0.7": recall_at_07,
            "f1_at_0.7": f1_at_07,
            "days_mae": mae,
            "days_rmse": rmse,
        }
    
    @staticmethod
    def _precision_at_threshold(y_true, y_pred_proba, threshold):
        """Compute precision at specific threshold."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    @staticmethod
    def _recall_at_threshold(y_true, y_pred_proba, threshold):
        """Compute recall at specific threshold."""
        y_pred = (y_pred_proba >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def _aggregate_fold_metrics(self, fold_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across CV folds."""
        metric_names = fold_metrics[0].keys()
        
        aggregated = {}
        for metric_name in metric_names:
            values = [fold[metric_name] for fold in fold_metrics]
            aggregated[metric_name] = np.mean(values)
            aggregated[f"{metric_name}_std"] = np.std(values)
        
        return aggregated
    
    def _hyperparameter_optimization(
        self,
        X: pd.DataFrame,
        y_waste: np.ndarray,
        y_days: np.ndarray,
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter optimization.
        
        Args:
            X: Features
            y_waste: Waste labels
            y_days: Days to waste
            n_trials: Number of trials
            
        Returns:
            Best hyperparameters
        """
        logger.info(f"Running hyperparameter optimization with Optuna")
        
        def objective(trial):
            # Suggest hyperparameters
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 80),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 0.9),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 2.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 2.0),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
            }
            
            # 80/20 train/val split
            train_size = int(0.8 * len(X))
            X_train = X.iloc[:train_size]
            X_val = X.iloc[train_size:]
            y_waste_train = y_waste[:train_size]
            y_waste_val = y_waste[train_size:]
            y_days_train = y_days[:train_size]
            y_days_val = y_days[train_size:]
            
            # Train model with suggested params (simplified)
            try:
                import lightgbm as lgb
                
                train_data = lgb.Dataset(X_train, label=y_waste_train)
                val_data = lgb.Dataset(X_val, label=y_waste_val, reference=train_data)
                
                model = lgb.train(
                    {**params, "objective": "binary", "metric": "auc", "verbose": -1},
                    train_data,
                    num_boost_round=200,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                )
                
                # Evaluate
                y_pred = model.predict(X_val)
                auc = roc_auc_score(y_waste_val, y_pred)
                
                return auc
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.5  # Return poor score for failed trials
        
        # Run optimization
        study = optuna.create_study(direction="maximize", study_name="waste_risk_optimization")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Optimization complete!")
        logger.info(f"  Best AUC: {study.best_value:.4f}")
        logger.info(f"  Best params: {study.best_params}")
        
        return study.best_params
    
    def _generate_evaluation_plots(
        self,
        predictor: WastePredictor,
        X_val: pd.DataFrame,
        y_waste_val: np.ndarray,
        y_days_val: np.ndarray,
    ):
        """Generate and log evaluation plots."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Predict
        y_pred_proba = []
        for idx in range(len(X_val)):
            result = predictor.predict(X_val.iloc[[idx]], f"entry_{idx}", return_shap=False)
            y_pred_proba.append(result.waste_probability)
        y_pred_proba = np.array(y_pred_proba)
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve
        
        fpr, tpr, thresholds = roc_curve(y_waste_val, y_pred_proba)
        auc = roc_auc_score(y_waste_val, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Waste Risk Prediction')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('roc_curve.png')
        plt.close()
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_waste_val, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {average_precision_score(y_waste_val, y_pred_proba):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('precision_recall_curve.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('precision_recall_curve.png')
        plt.close()
        
        # 3. Calibration Curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_waste_val, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('calibration_curve.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('calibration_curve.png')
        plt.close()
        
        # 4. Confusion Matrix
        y_pred = (y_pred_proba >= 0.7).astype(int)
        cm = confusion_matrix(y_waste_val, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix (threshold=0.7)')
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('confusion_matrix.png')
        plt.close()
        
        logger.info("  Saved evaluation plots to MLflow")
    
    def _log_feature_importance(self, model, feature_names):
        """Log feature importance plot."""
        import matplotlib
        matplotlib.use('Agg')
        
        importance = model.feature_importance(importance_type='gain')
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(20)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.xlabel('Importance (Gain)')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        mlflow.log_artifact('feature_importance.png')
        plt.close()
        
        # Log as CSV
        feature_importance_df.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        
        logger.info("  Logged feature importance")


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train waste risk prediction model with MLflow tracking"
    )
    
    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--start-date",
        type=str,
        help="Training data start date (YYYY-MM-DD). Requires --end-date"
    )
    data_group.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic training data"
    )
    data_group.add_argument(
        "--data-file",
        type=str,
        help="Path to training data file (parquet/csv)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        help="Training data end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of synthetic samples (if --use-synthetic)"
    )
    
    # Training config
    parser.add_argument(
        "--experiment-name",
        default=MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run hyperparameter optimization"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials (if --hyperopt)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start_date and not args.end_date:
        parser.error("--start-date requires --end-date")
    
    # Initialize trainer
    trainer = WasteModelTrainer(args.experiment_name)
    
    # Load data
    if args.use_synthetic:
        logger.info("Generating synthetic training data...")
        from .generate_synthetic_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator()
        df = generator.generate_dataset(
            n_households=args.n_samples // 100,
            n_items_per_household=100,
        )
        
        # Save temporarily
        df.to_parquet('temp_synthetic_data.parquet', index=False)
        X, y_waste, y_days = trainer.load_training_data_from_file('temp_synthetic_data.parquet')
        
    elif args.data_file:
        X, y_waste, y_days = trainer.load_training_data_from_file(args.data_file)
        
    else:
        # Load from database
        start_date = datetime.fromisoformat(args.start_date)
        end_date = datetime.fromisoformat(args.end_date)
        X, y_waste, y_days = asyncio.run(
            trainer.load_training_data_from_db(start_date, end_date)
        )
    
    # Train model
    results = trainer.train_with_cross_validation(
        X, y_waste, y_days,
        n_folds=args.n_folds,
        hyperopt=args.hyperopt,
        n_trials=args.n_trials,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"MLflow Run ID: {results['mlflow_run_id']}")
    print(f"MLflow UI: {MLFLOW_TRACKING_URI}")
    print(f"\nCross-Validation Metrics:")
    print(f"  AUC: {results['cv_metrics']['auc']:.4f} ± {results['cv_metrics']['auc_std']:.4f}")
    print(f"  Precision@0.7: {results['cv_metrics']['precision_at_0.7']:.4f}")
    print(f"  Recall@0.7: {results['cv_metrics']['recall_at_0.7']:.4f}")
    print(f"  F1@0.7: {results['cv_metrics']['f1_at_0.7']:.4f}")
    print(f"  Days MAE: {results['cv_metrics']['days_mae']:.2f} days")
    print(f"\nFinal Model Metrics:")
    print(f"  AUC: {results['final_metrics']['auc']:.4f}")
    print(f"  Precision@0.7: {results['final_metrics']['precision_at_0.7']:.4f}")
    print(f"  Recall@0.7: {results['final_metrics']['recall_at_0.7']:.4f}")
    print("="*80)
    print("\n✓ Training complete! View results in MLflow UI.")


if __name__ == "__main__":
    main()

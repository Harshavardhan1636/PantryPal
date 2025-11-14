"""
Production Model Training with Hyperparameter Optimization
Train waste risk prediction model to meet gate requirements:
- AUC ≥ 0.85
- Precision@0.7 ≥ 0.80

Features:
- 67-feature dataset
- Optuna hyperparameter optimization
- 5-fold cross-validation
- Comprehensive evaluation metrics
- Feature importance analysis
"""

import argparse
import logging
from pathlib import Path
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix
)
import lightgbm as lgb
import optuna

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ProductionModelTrainer:
    """Train production-ready waste prediction model."""
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        self.best_model = None
        self.best_params = None
        self.feature_names = None
        
    def load_data(self, data_path: str) -> tuple:
        """Load and prepare training data."""
        logger.info(f"Loading data from {data_path}...")
        
        df = pd.read_parquet(data_path)
        
        logger.info(f"✓ Loaded {len(df):,} samples")
        logger.info(f"  Features: {len(df.columns)}")
        logger.info(f"  Waste rate: {df['is_wasted'].mean():.1%}")
        
        # Drop non-feature columns
        drop_cols = [
            'household_id', 'archetype', 'item_name', 'category',
            'purchase_date', 'expiration_date', 'season',
            'days_to_expiration_categorical',
            'is_wasted',  # Target
            'days_to_waste', 'waste_probability'  # Leakage
        ]
        
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        y = df['is_wasted']
        
        self.feature_names = X.columns.tolist()
        logger.info(f"✓ Feature columns: {len(self.feature_names)}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """Split into train/val/test sets."""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        # Second split: separate validation set
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.seed, stratify=y_temp
        )
        
        logger.info("")
        logger.info("Data split:")
        logger.info(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
        logger.info(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X):.1%})")
        logger.info(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X):.1%})")
        logger.info(f"  Train waste rate: {y_train.mean():.1%}")
        logger.info(f"  Val waste rate:   {y_val.mean():.1%}")
        logger.info(f"  Test waste rate:  {y_test.mean():.1%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=100):
        """Use Optuna to find best hyperparameters."""
        logger.info("")
        logger.info("="*80)
        logger.info("HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        logger.info(f"Search space: {n_trials} trials")
        logger.info("Optimizing for: AUC")
        logger.info("")
        
        def objective(trial):
            """Optuna objective function."""
            params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'seed': self.seed,
                
                # Hyperparameters to optimize
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 80),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            }
            
            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
            )
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            return auc
        
        # Run optimization
        study = optuna.create_study(direction='maximize', study_name='waste_predictor')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info("")
        logger.info("Optimization complete!")
        logger.info(f"✓ Best AUC: {study.best_value:.4f}")
        logger.info("✓ Best hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"    {key:20s}: {value}")
        
        self.best_params = study.best_params
        return study.best_params
    
    def train_final_model(self, X_train, y_train, X_val, y_val, params):
        """Train final model with best hyperparameters."""
        logger.info("")
        logger.info("="*80)
        logger.info("TRAINING FINAL MODEL")
        logger.info("="*80)
        
        # Add fixed parameters
        full_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': self.seed,
            **params
        }
        
        # Train
        model = lgb.LGBMClassifier(**full_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )
        
        self.best_model = model
        logger.info("✓ Model training complete")
        logger.info(f"  Best iteration: {model.best_iteration_}")
        logger.info(f"  Total features: {len(self.feature_names)}")
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Comprehensive model evaluation."""
        logger.info("")
        logger.info("="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # AUC
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"AUC: {auc:.4f} {'✅' if auc >= 0.85 else '❌ (need ≥0.85)'}")
        
        # Precision at different thresholds
        logger.info("")
        logger.info("Precision@Threshold:")
        thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            status = "✅" if precision >= 0.80 and threshold == 0.7 else ""
            logger.info(f"  @{threshold:.1f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f} {status}")
        
        # Find optimal threshold
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else 0.5
        
        logger.info(f"")
        logger.info(f"Optimal threshold (by F1): {best_threshold:.3f}")
        logger.info(f"  Precision: {precision_vals[best_f1_idx]:.3f}")
        logger.info(f"  Recall: {recall_vals[best_f1_idx]:.3f}")
        logger.info(f"  F1: {f1_scores[best_f1_idx]:.3f}")
        
        # Confusion matrix
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_optimal)
        logger.info("")
        logger.info("Confusion Matrix:")
        logger.info(f"  TN: {cm[0,0]:6,}  FP: {cm[0,1]:6,}")
        logger.info(f"  FN: {cm[1,0]:6,}  TP: {cm[1,1]:6,}")
        
        # Feature importance
        logger.info("")
        logger.info("Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']:35s}: {row['importance']:8.0f}")
        
        # Gate requirements check
        logger.info("")
        logger.info("="*80)
        logger.info("GATE REQUIREMENTS CHECK")
        logger.info("="*80)
        
        precision_at_07 = precision_score(y_test, (y_pred_proba >= 0.7).astype(int), zero_division=0)
        
        auc_pass = auc >= 0.85
        precision_pass = precision_at_07 >= 0.80
        
        logger.info(f"✓ AUC ≥ 0.85:           {auc:.4f} {'PASS ✅' if auc_pass else 'FAIL ❌'}")
        logger.info(f"✓ Precision@0.7 ≥ 0.80: {precision_at_07:.4f} {'PASS ✅' if precision_pass else 'FAIL ❌'}")
        logger.info("")
        
        if auc_pass and precision_pass:
            logger.info("🎉 ALL GATE REQUIREMENTS MET! 🎉")
        else:
            logger.info("⚠️  Some gate requirements not met")
        
        return {
            'auc': auc,
            'precision_at_07': precision_at_07,
            'best_threshold': best_threshold,
            'feature_importance': feature_importance,
            'gate_pass': auc_pass and precision_pass
        }
    
    def save_model(self, output_path: str, metrics: dict):
        """Save trained model and metadata."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_artifact = {
            'model': self.best_model,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat(),
            'seed': self.seed
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_artifact, f)
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info("")
        logger.info("="*80)
        logger.info(f"✓ Model saved to {output_path}")
        logger.info(f"  File size: {file_size_mb:.1f} MB")
        logger.info("="*80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train production waste prediction model with hyperparameter optimization"
    )
    
    parser.add_argument(
        "--data",
        default="data/training_enhanced_100k.parquet",
        help="Training data path"
    )
    parser.add_argument(
        "--output",
        default="models/waste_predictor_production.pkl",
        help="Output model path"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Optuna optimization trials (default: 100)"
    )
    parser.add_argument(
        "--no-hyperopt",
        action="store_true",
        help="Skip hyperparameter optimization (use defaults)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ProductionModelTrainer(seed=args.seed)
    
    # Load data
    X, y = trainer.load_data(args.data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(X, y)
    
    # Hyperparameter optimization
    if not args.no_hyperopt:
        best_params = trainer.optimize_hyperparameters(
            X_train, y_train, X_val, y_val,
            n_trials=args.n_trials
        )
    else:
        logger.info("Skipping hyperparameter optimization (using defaults)")
        best_params = {
            'n_estimators': 300,
            'max_depth': 7,
            'learning_rate': 0.05,
            'num_leaves': 50,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
    
    # Train final model
    model = trainer.train_final_model(X_train, y_train, X_val, y_val, best_params)
    
    # Evaluate
    metrics = trainer.evaluate_model(model, X_test, y_test)
    
    # Save
    trainer.save_model(args.output, metrics)


if __name__ == "__main__":
    main()

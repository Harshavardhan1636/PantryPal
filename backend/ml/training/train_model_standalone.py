"""
Standalone Model Training Script - No Database/MLflow Dependencies

Trains a waste risk prediction model on synthetic data.

Usage:
    python train_model_standalone.py --data data/synthetic_test.parquet
"""

import argparse
import logging
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_features(df):
    """Prepare features from raw data."""
    logger.info("Preparing features...")
    
    # Select numeric features
    feature_cols = [
        'price',
        'shelf_life_days',
        'days_until_expiration',
        'household_size',
        'household_waste_rate',
    ]
    
    X = df[feature_cols].copy()
    y = df['is_wasted'].values
    
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Waste rate: {y.mean():.1%}")
    
    return X, y, feature_cols


def train_model(X_train, y_train, X_val, y_val):
    """Train LightGBM classifier."""
    logger.info("\nTraining LightGBM model...")
    
    # Model parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'is_unbalance': True,  # Handle class imbalance
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50),
        ]
    )
    
    logger.info(f"✓ Training complete (best iteration: {model.best_iteration})")
    
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance."""
    logger.info("\n" + "="*80)
    logger.info("MODEL EVALUATION")
    logger.info("="*80)
    
    # Predictions
    y_pred_proba = model.predict(X_test)
    
    # Classification metrics at different thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9]
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"\nThreshold = {threshold}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall:    {recall:.3f}")
        logger.info(f"  F1 Score:  {f1:.3f}")
    
    # AUC (main metric)
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"\n{'='*40}")
    logger.info(f"AUC-ROC: {auc:.4f}")
    logger.info(f"{'='*40}")
    
    # Gate requirements
    y_pred_07 = (y_pred_proba >= 0.7).astype(int)
    precision_07 = precision_score(y_test, y_pred_07, zero_division=0)
    
    logger.info(f"\nGate Requirements:")
    logger.info(f"  AUC >= 0.85:           {'✓ PASS' if auc >= 0.85 else '✗ FAIL'} ({auc:.3f})")
    logger.info(f"  Precision@0.7 >= 0.80: {'✓ PASS' if precision_07 >= 0.80 else '✗ FAIL'} ({precision_07:.3f})")
    
    # Feature importance
    logger.info(f"\n{'='*40}")
    logger.info("FEATURE IMPORTANCE")
    logger.info(f"{'='*40}")
    
    importance = model.feature_importance(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        logger.info(f"  {row['feature']:<30} {row['importance']:>10.1f}")
    
    return {
        'auc': auc,
        'precision_at_07': precision_07,
        'feature_importance': feature_importance,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train waste risk prediction model"
    )
    
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training data (parquet or csv)"
    )
    parser.add_argument(
        "--output",
        default="models/waste_predictor.pkl",
        help="Output model file path"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set proportion (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    elif args.data.endswith('.csv'):
        df = pd.read_csv(args.data)
    else:
        raise ValueError(f"Unsupported file format: {args.data}")
    
    logger.info(f"✓ Loaded {len(df):,} samples")
    
    # Prepare features
    X, y, feature_names = prepare_features(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    logger.info(f"\nData split:")
    logger.info(f"  Training:   {len(X_train):,} samples ({y_train.mean():.1%} waste)")
    logger.info(f"  Test:       {len(X_test):,} samples ({y_test.mean():.1%} waste)")
    
    # Train model
    model = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_names)
    
    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(output_path))
    logger.info(f"\n✓ Model saved to {args.output}")
    
    # Also save as pickle with metadata
    pickle_path = output_path.with_suffix('.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'metrics': metrics,
        }, f)
    logger.info(f"✓ Model + metadata saved to {pickle_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review metrics above")
    logger.info(f"  2. Check feature importance")
    logger.info(f"  3. Test predictions on new data")
    logger.info(f"  4. Deploy model if metrics meet requirements")


if __name__ == "__main__":
    main()

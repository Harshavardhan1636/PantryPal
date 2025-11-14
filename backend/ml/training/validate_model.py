"""
Model Validation Suite for Waste Risk Prediction.

Comprehensive validation framework including:
- Cross-validation strategies
- Calibration analysis
- Fairness metrics (demographic parity, equalized odds)
- Threshold optimization
- Statistical tests
- Performance guarantees

Usage:
    from backend.ml.training.validate_model import ModelValidator
    
    validator = ModelValidator(model, X_test, y_test)
    report = validator.generate_validation_report()

Author: Senior SDE-3
Date: 2025-11-12
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Validation metrics container."""
    # Classification metrics
    auc: float
    avg_precision: float
    brier_score: float
    log_loss: float
    
    # Calibration metrics
    expected_calibration_error: float
    calibration_slope: float
    calibration_intercept: float
    
    # Threshold-specific metrics
    optimal_threshold: float
    precision_at_optimal: float
    recall_at_optimal: float
    f1_at_optimal: float
    
    # Business metrics
    precision_at_high_confidence: float  # Precision at 0.7 threshold
    recall_at_high_confidence: float
    
    # Statistical tests
    hosmer_lemeshow_pvalue: float
    anderson_darling_pvalue: Optional[float]
    
    # Fairness metrics
    demographic_parity_diff: Optional[float]
    equalized_odds_diff: Optional[float]
    
    # Cross-validation
    cv_auc_mean: float
    cv_auc_std: float
    cv_stable: bool  # True if CV std < 0.05


class ModelValidator:
    """Comprehensive model validation framework."""
    
    def __init__(
        self,
        model,  # Trained model (must have predict_proba method)
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        sensitive_features: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize validator.
        
        Args:
            model: Trained model with predict_proba method
            X_test: Test features
            y_test: Test labels
            sensitive_features: Optional sensitive features for fairness analysis
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.sensitive_features = sensitive_features
        
        # Get predictions
        self.y_pred_proba = self._get_predictions(model, X_test)
        
        logger.info("Initialized ModelValidator")
        logger.info(f"  Test samples: {len(X_test)}")
        logger.info(f"  Positive class ratio: {y_test.mean():.2%}")
    
    def _get_predictions(self, model, X):
        """Get probability predictions from model."""
        try:
            # Try predict_proba (sklearn-style)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                if proba.ndim == 2:
                    return proba[:, 1]  # Positive class
                return proba
            # Try predict (LightGBM-style)
            elif hasattr(model, 'predict'):
                return model.predict(X)
            else:
                raise ValueError("Model must have predict_proba or predict method")
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            raise
    
    def validate_comprehensive(self) -> ValidationMetrics:
        """
        Run comprehensive validation suite.
        
        Returns:
            ValidationMetrics with all validation results
        """
        logger.info("="*80)
        logger.info("COMPREHENSIVE MODEL VALIDATION")
        logger.info("="*80)
        
        # 1. Classification metrics
        logger.info("\n1. Computing classification metrics...")
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        brier = brier_score_loss(self.y_test, self.y_pred_proba)
        logloss = log_loss(self.y_test, self.y_pred_proba)
        
        logger.info(f"   AUC: {auc:.4f}")
        logger.info(f"   Average Precision: {avg_precision:.4f}")
        logger.info(f"   Brier Score: {brier:.4f}")
        logger.info(f"   Log Loss: {logloss:.4f}")
        
        # 2. Calibration analysis
        logger.info("\n2. Analyzing calibration...")
        ece, cal_slope, cal_intercept = self._compute_calibration_metrics()
        logger.info(f"   Expected Calibration Error: {ece:.4f}")
        logger.info(f"   Calibration slope: {cal_slope:.4f}")
        logger.info(f"   Calibration intercept: {cal_intercept:.4f}")
        
        # 3. Threshold optimization
        logger.info("\n3. Optimizing decision threshold...")
        optimal_threshold, opt_precision, opt_recall, opt_f1 = self._optimize_threshold()
        logger.info(f"   Optimal threshold: {optimal_threshold:.3f}")
        logger.info(f"   Precision: {opt_precision:.4f}")
        logger.info(f"   Recall: {opt_recall:.4f}")
        logger.info(f"   F1: {opt_f1:.4f}")
        
        # 4. Business metrics (high confidence = 0.7)
        logger.info("\n4. Computing business metrics (threshold=0.7)...")
        prec_07, recall_07 = self._metrics_at_threshold(0.7)
        logger.info(f"   Precision@0.7: {prec_07:.4f}")
        logger.info(f"   Recall@0.7: {recall_07:.4f}")
        
        # 5. Statistical tests
        logger.info("\n5. Running statistical tests...")
        hl_pvalue = self._hosmer_lemeshow_test()
        ad_pvalue = None  # Can add Anderson-Darling if needed
        logger.info(f"   Hosmer-Lemeshow p-value: {hl_pvalue:.4f}")
        if hl_pvalue > 0.05:
            logger.info("   ✓ Model is well-calibrated (p > 0.05)")
        else:
            logger.warning("   ✗ Model may be mis-calibrated (p < 0.05)")
        
        # 6. Fairness metrics
        logger.info("\n6. Checking fairness...")
        dp_diff, eo_diff = self._compute_fairness_metrics()
        if dp_diff is not None:
            logger.info(f"   Demographic parity diff: {dp_diff:.4f}")
            logger.info(f"   Equalized odds diff: {eo_diff:.4f}")
            if abs(dp_diff) < 0.1:
                logger.info("   ✓ Demographic parity satisfied")
            else:
                logger.warning("   ✗ Demographic parity may be violated")
        else:
            logger.info("   Skipped (no sensitive features provided)")
        
        # 7. Cross-validation stability
        logger.info("\n7. Validating cross-validation stability...")
        cv_auc_mean, cv_auc_std = self._check_cv_stability()
        cv_stable = cv_auc_std < 0.05
        logger.info(f"   CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
        if cv_stable:
            logger.info("   ✓ Model is stable across folds")
        else:
            logger.warning("   ✗ High CV variance detected")
        
        logger.info("\n" + "="*80)
        logger.info("VALIDATION COMPLETE")
        logger.info("="*80)
        
        return ValidationMetrics(
            auc=auc,
            avg_precision=avg_precision,
            brier_score=brier,
            log_loss=logloss,
            expected_calibration_error=ece,
            calibration_slope=cal_slope,
            calibration_intercept=cal_intercept,
            optimal_threshold=optimal_threshold,
            precision_at_optimal=opt_precision,
            recall_at_optimal=opt_recall,
            f1_at_optimal=opt_f1,
            precision_at_high_confidence=prec_07,
            recall_at_high_confidence=recall_07,
            hosmer_lemeshow_pvalue=hl_pvalue,
            anderson_darling_pvalue=ad_pvalue,
            demographic_parity_diff=dp_diff,
            equalized_odds_diff=eo_diff,
            cv_auc_mean=cv_auc_mean,
            cv_auc_std=cv_auc_std,
            cv_stable=cv_stable,
        )
    
    def _compute_calibration_metrics(self) -> Tuple[float, float, float]:
        """
        Compute calibration metrics.
        
        Returns:
            (ECE, slope, intercept)
        """
        # Expected Calibration Error (ECE)
        n_bins = 10
        fraction_of_positives, mean_predicted_value = calibration_curve(
            self.y_test, self.y_pred_proba, n_bins=n_bins, strategy='uniform'
        )
        
        # ECE = average absolute calibration error
        ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        # Calibration slope and intercept (linear fit)
        if len(mean_predicted_value) > 1:
            slope, intercept, _, _, _ = stats.linregress(
                mean_predicted_value, fraction_of_positives
            )
        else:
            slope, intercept = 1.0, 0.0
        
        return ece, slope, intercept
    
    def _optimize_threshold(self) -> Tuple[float, float, float, float]:
        """
        Find optimal threshold by maximizing F1 score.
        
        Returns:
            (threshold, precision, recall, f1)
        """
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba
        )
        
        # Compute F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresholds):
            best_threshold = thresholds[best_idx]
        else:
            best_threshold = 0.5  # Default
        
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        best_f1 = f1_scores[best_idx]
        
        return best_threshold, best_precision, best_recall, best_f1
    
    def _metrics_at_threshold(self, threshold: float) -> Tuple[float, float]:
        """
        Compute precision and recall at specific threshold.
        
        Args:
            threshold: Classification threshold
            
        Returns:
            (precision, recall)
        """
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        
        tp = np.sum((y_pred == 1) & (self.y_test == 1))
        fp = np.sum((y_pred == 1) & (self.y_test == 0))
        fn = np.sum((y_pred == 0) & (self.y_test == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def _hosmer_lemeshow_test(self, n_bins: int = 10) -> float:
        """
        Hosmer-Lemeshow goodness-of-fit test for calibration.
        
        Args:
            n_bins: Number of bins for grouping
            
        Returns:
            p-value (higher is better, >0.05 indicates good calibration)
        """
        # Group predictions into bins
        bin_edges = np.percentile(self.y_pred_proba, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 0.001  # Ensure last bin includes max value
        
        observed = []
        expected = []
        
        for i in range(n_bins):
            mask = (self.y_pred_proba >= bin_edges[i]) & (self.y_pred_proba < bin_edges[i + 1])
            
            if mask.sum() == 0:
                continue
            
            obs = self.y_test[mask].sum()
            exp = self.y_pred_proba[mask].sum()
            
            observed.append(obs)
            expected.append(exp)
        
        # Chi-squared test
        observed = np.array(observed)
        expected = np.array(expected)
        
        chi2 = np.sum((observed - expected) ** 2 / (expected + 1e-10))
        
        # Degrees of freedom = n_bins - 2
        df = len(observed) - 2
        
        if df > 0:
            p_value = 1 - stats.chi2.cdf(chi2, df)
        else:
            p_value = 1.0
        
        return p_value
    
    def _compute_fairness_metrics(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute fairness metrics if sensitive features provided.
        
        Returns:
            (demographic_parity_diff, equalized_odds_diff)
        """
        if self.sensitive_features is None or len(self.sensitive_features) == 0:
            return None, None
        
        # Use first sensitive feature (e.g., household_size groups)
        sensitive_col = self.sensitive_features.iloc[:, 0]
        
        # Binarize sensitive feature (high vs low)
        sensitive_median = sensitive_col.median()
        group_a = sensitive_col <= sensitive_median  # Privileged group
        group_b = sensitive_col > sensitive_median   # Unprivileged group
        
        # Predictions at threshold 0.7
        y_pred = (self.y_pred_proba >= 0.7).astype(int)
        
        # Demographic parity: P(Y_pred=1|A) - P(Y_pred=1|B)
        pr_a = y_pred[group_a].mean()
        pr_b = y_pred[group_b].mean()
        dp_diff = abs(pr_a - pr_b)
        
        # Equalized odds: TPR and FPR difference
        # TPR_A - TPR_B
        tpr_a = y_pred[group_a & (self.y_test == 1)].mean() if (group_a & (self.y_test == 1)).sum() > 0 else 0
        tpr_b = y_pred[group_b & (self.y_test == 1)].mean() if (group_b & (self.y_test == 1)).sum() > 0 else 0
        
        # FPR_A - FPR_B
        fpr_a = y_pred[group_a & (self.y_test == 0)].mean() if (group_a & (self.y_test == 0)).sum() > 0 else 0
        fpr_b = y_pred[group_b & (self.y_test == 0)].mean() if (group_b & (self.y_test == 0)).sum() > 0 else 0
        
        eo_diff = max(abs(tpr_a - tpr_b), abs(fpr_a - fpr_b))
        
        return dp_diff, eo_diff
    
    def _check_cv_stability(self, n_folds: int = 5) -> Tuple[float, float]:
        """
        Check model stability via cross-validation.
        
        Args:
            n_folds: Number of CV folds
            
        Returns:
            (mean_auc, std_auc)
        """
        # Use stratified K-fold for better balance
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        auc_scores = []
        
        for train_idx, val_idx in cv.split(self.X_test, self.y_test):
            X_val = self.X_test.iloc[val_idx]
            y_val = self.y_test[val_idx]
            
            y_pred_val = self._get_predictions(self.model, X_val)
            auc = roc_auc_score(y_val, y_pred_val)
            auc_scores.append(auc)
        
        return np.mean(auc_scores), np.std(auc_scores)
    
    def generate_report(self, output_path: str = "validation_report.txt") -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            output_path: Output file path
            
        Returns:
            Report as string
        """
        metrics = self.validate_comprehensive()
        
        report_lines = [
            "="*80,
            "MODEL VALIDATION REPORT",
            "="*80,
            f"Generated: {pd.Timestamp.now()}",
            f"Test samples: {len(self.X_test):,}",
            f"Positive class ratio: {self.y_test.mean():.2%}",
            "",
            "CLASSIFICATION METRICS",
            "-"*80,
            f"AUC-ROC:                    {metrics.auc:.4f}",
            f"Average Precision:          {metrics.avg_precision:.4f}",
            f"Brier Score:                {metrics.brier_score:.4f}",
            f"Log Loss:                   {metrics.log_loss:.4f}",
            "",
            "CALIBRATION METRICS",
            "-"*80,
            f"Expected Calibration Error: {metrics.expected_calibration_error:.4f}",
            f"Calibration Slope:          {metrics.calibration_slope:.4f}",
            f"Calibration Intercept:      {metrics.calibration_intercept:.4f}",
            f"Hosmer-Lemeshow p-value:    {metrics.hosmer_lemeshow_pvalue:.4f}",
            "",
            "THRESHOLD OPTIMIZATION",
            "-"*80,
            f"Optimal Threshold:          {metrics.optimal_threshold:.3f}",
            f"  Precision:                {metrics.precision_at_optimal:.4f}",
            f"  Recall:                   {metrics.recall_at_optimal:.4f}",
            f"  F1 Score:                 {metrics.f1_at_optimal:.4f}",
            "",
            "BUSINESS METRICS (threshold=0.7)",
            "-"*80,
            f"Precision:                  {metrics.precision_at_high_confidence:.4f}",
            f"Recall:                     {metrics.recall_at_high_confidence:.4f}",
            "",
            "CROSS-VALIDATION STABILITY",
            "-"*80,
            f"CV AUC:                     {metrics.cv_auc_mean:.4f} ± {metrics.cv_auc_std:.4f}",
            f"Stable:                     {'✓ Yes' if metrics.cv_stable else '✗ No'}",
            "",
        ]
        
        # Add fairness if available
        if metrics.demographic_parity_diff is not None:
            report_lines.extend([
                "FAIRNESS METRICS",
                "-"*80,
                f"Demographic Parity Diff:    {metrics.demographic_parity_diff:.4f}",
                f"Equalized Odds Diff:        {metrics.equalized_odds_diff:.4f}",
                "",
            ])
        
        # Gate requirements check
        report_lines.extend([
            "GATE REQUIREMENTS",
            "-"*80,
            f"AUC >= 0.85:                {'✓ PASS' if metrics.auc >= 0.85 else '✗ FAIL'}",
            f"Precision@0.7 >= 0.80:      {'✓ PASS' if metrics.precision_at_high_confidence >= 0.80 else '✗ FAIL'}",
            f"Calibration (ECE < 0.10):   {'✓ PASS' if metrics.expected_calibration_error < 0.10 else '✗ FAIL'}",
            f"CV Stable (std < 0.05):     {'✓ PASS' if metrics.cv_stable else '✗ FAIL'}",
            "",
            "="*80,
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save to file
        Path(output_path).write_text(report_text)
        logger.info(f"Saved validation report to {output_path}")
        
        return report_text


def main():
    """Example usage."""
    # This is for demonstration
    print("Model Validation Suite")
    print("Usage:")
    print("  from backend.ml.training.validate_model import ModelValidator")
    print("  validator = ModelValidator(model, X_test, y_test)")
    print("  report = validator.generate_report()")


if __name__ == "__main__":
    main()

"""
Model Evaluation Module for FinSecure AI Fraud Detection System.

This module provides comprehensive evaluation of the trained fraud detection
model including confusion matrix, classification metrics, ROC/PR curves,
and feature importance analysis.

Author: FinSecure AI Team
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    matthews_corrcoef
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import feature engineering
from src.feature_engineering import engineer_features, transform_with_scaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_metadata(models_dir: Path):
    """
    Load trained model, scaler, and metadata.

    Args:
        models_dir: Path to models directory.

    Returns:
        Tuple of (model, scaler, metadata).
    """
    logger.info("Loading model and related files")

    # Load model
    model_path = models_dir / "xgboost_model.joblib"
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Load scaler
    scaler_path = models_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path)
    logger.info(f"Scaler loaded from {scaler_path}")

    # Load metadata
    metadata_path = models_dir / "model_metadata.joblib"
    metadata = joblib.load(metadata_path)
    logger.info(f"Metadata loaded: threshold={metadata['threshold']}")

    return model, scaler, metadata


def prepare_test_data(parquet_path: Path, models_dir: Path) -> tuple:
    """
    Load cleaned data and prepare test set.

    Args:
        parquet_path: Path to cleaned Parquet file.
        models_dir: Path to models directory.

    Returns:
        Tuple of (X_test, y_test, feature_columns).
    """
    logger.info("Preparing test data")

    # Load cleaned data
    df = pd.read_parquet(parquet_path)

    # Apply feature engineering
    df = engineer_features(df)

    # Sort by step for chronological split
    df = df.sort_values('step').reset_index(drop=True)

    # Get last 15% as test set (same as training)
    n = len(df)
    test_start = int(n * 0.85)
    test_df = df.iloc[test_start:]

    logger.info(f"Test set size: {len(test_df):,} rows")

    # Get feature columns
    original_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                    'oldbalanceDest', 'newbalanceDest', 'isFraud']
    one_hot_cols = [col for col in df.columns if col.startswith('type_')]
    feature_cols = [col for col in df.columns
                   if col not in original_cols + one_hot_cols]

    # Extract features and target
    X_test = test_df[feature_cols]
    y_test = test_df['isFraud']

    return X_test, y_test, feature_cols


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute and format confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary with confusion matrix data.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return {
        'raw': cm,
        'normalized': cm_normalized,
        'tn': cm[0, 0],
        'fp': cm[0, 1],
        'fn': cm[1, 0],
        'tp': cm[1, 1]
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path: Path) -> None:
    """
    Create confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Confusion matrix saved to {save_path}")


def plot_feature_importance(model: XGBClassifier, feature_cols: list,
                           save_path: Path) -> None:
    """
    Create feature importance bar chart (top 20).

    Args:
        model: Trained XGBClassifier.
        feature_cols: List of feature column names.
        save_path: Path to save the plot.
    """
    # Get feature importance
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Take top 20
    top_20 = feature_importance.head(20)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_20)), top_20['importance'].values, color='steelblue')
    plt.yticks(range(len(top_20)), top_20['feature'].values)
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Feature importance plot saved to {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray,
                   save_path: Path, auc_score: float) -> None:
    """
    Create ROC curve plot.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_path: Path to save the plot.
        auc_score: ROC-AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"ROC curve saved to {save_path}")


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray,
                  save_path: Path) -> None:
    """
    Create Precision-Recall curve plot.

    Args:
        y_true: True labels.
        y_proba: Predicted probabilities.
        save_path: Path to save the plot.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = np.trapz(precision, -np.gradient(recall))

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2,
             label=f'PR curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"PR curve saved to {save_path}")


def print_evaluation_results(cm: dict, report: dict, roc_auc: float,
                            pr_auc: float, mcc: float) -> None:
    """
    Print comprehensive evaluation results.

    Args:
        cm: Confusion matrix dictionary.
        report: Classification report.
        roc_auc: ROC-AUC score.
        pr_auc: Precision-Recall AUC.
        mcc: Matthews Correlation Coefficient.
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 60)

    # Confusion Matrix
    logger.info("\nConfusion Matrix (Raw):")
    logger.info(f"  True Negatives:  {cm['tn']:,}")
    logger.info(f"  False Positives: {cm['fp']:,}")
    logger.info(f"  False Negatives: {cm['fn']:,}")
    logger.info(f"  True Positives:  {cm['tp']:,}")

    logger.info("\nConfusion Matrix (Normalized):")
    logger.info(f"  Legitimate correctly: {cm['normalized'][0,0]*100:.2f}%")
    logger.info(f"  Fraud correctly:     {cm['normalized'][1,1]*100:.2f}%")

    # Classification Report
    logger.info("\nClassification Report:")
    logger.info(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    logger.info(f"  {'-'*44}")
    for cls in ['0', '1']:
        logger.info(f"  {cls:<12} {report[cls]['precision']:>10.4f} "
                   f"{report[cls]['recall']:>10.4f} {report[cls]['f1-score']:>10.4f}")

    # Summary Metrics
    logger.info("\nSummary Metrics:")
    logger.info(f"  ROC-AUC Score:         {roc_auc:.4f}")
    logger.info(f"  Precision-Recall AUC: {pr_auc:.4f}")
    logger.info(f"  MCC Score:            {mcc:.4f}")

    logger.info("=" * 60)


def main() -> None:
    """Main entry point for model evaluation."""
    # Define paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "cleaned_transactions.parquet"
    models_dir = project_root / "models"

    # Check if model exists
    if not (models_dir / "xgboost_model.joblib").exists():
        logger.error("Model not found. Please run: python src/train_model.py")
        return

    # Check if data exists
    if not data_path.exists():
        logger.error("Cleaned data not found. Please run: python src/preprocess.py")
        return

    # Load model and metadata
    model, scaler, metadata = load_model_and_metadata(models_dir)

    # Prepare test data
    X_test, y_test, feature_cols = prepare_test_data(data_path, models_dir)

    # Scale features
    X_test_scaled = transform_with_scaler(X_test, scaler)

    # Get predictions
    threshold = metadata['threshold']
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    cm = compute_confusion_matrix(y_test.values, y_pred)
    report = classification_report(y_test.values, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Compute PR AUC
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = np.trapz(precision, -np.gradient(recall))

    # Compute MCC
    mcc = matthews_corrcoef(y_test.values, y_pred)

    # Print results
    print_evaluation_results(cm, report, roc_auc, pr_auc, mcc)

    # Generate plots
    plot_confusion_matrix(y_test.values, y_pred, models_dir / "confusion_matrix.png")
    plot_feature_importance(model, feature_cols, models_dir / "feature_importance.png")
    plot_roc_curve(y_test.values, y_proba, models_dir / "roc_curve.png", roc_auc)
    plot_pr_curve(y_test.values, y_proba, models_dir / "pr_curve.png")

    logger.info("\nEvaluation complete! All plots saved to models/ directory")


if __name__ == "__main__":
    main()
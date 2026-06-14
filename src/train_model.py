"""
Model Training Module for FinSecure AI Fraud Detection System.

This module handles XGBoost model training with chronological data splitting,
early stopping, and threshold optimization for fraud detection.

Author: FinSecure AI Team
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, classification_report, roc_auc_score
import joblib

# Import feature engineering
from src.feature_engineering import engineer_features, fit_and_save_scaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EARLY_STOPPING_ROUNDS = 20
OPTIMAL_THRESHOLD = 0.8  # Default, will be tuned


def load_and_prepare_data(parquet_path: Path) -> pd.DataFrame:
    """
    Load cleaned Parquet file and apply feature engineering.

    Args:
        parquet_path: Path to cleaned_transactions.parquet

    Returns:
        DataFrame with engineered features.
    """
    logger.info(f"Loading cleaned data from {parquet_path}")

    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Apply feature engineering
    df = engineer_features(df)

    return df


def chronological_split(df: pd.DataFrame):
    """
    Perform chronological train/val/test split sorted by step.

    Args:
        df: DataFrame sorted by 'step'.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Performing chronological data split")

    # Sort by step (time)
    df = df.sort_values('step').reset_index(drop=True)

    # Get feature columns (exclude original columns and target)
    original_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                     'oldbalanceDest', 'newbalanceDest', 'isFraud']
    one_hot_cols = [col for col in df.columns if col.startswith('type_')]

    feature_cols = [col for col in df.columns
                    if col not in original_cols + one_hot_cols]

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Calculate split indices
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    # Split data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Train: {len(train_df):,} rows (steps {train_df['step'].min()}-{train_df['step'].max()})")
    logger.info(f"Validation: {len(val_df):,} rows (steps {val_df['step'].min()}-{val_df['step'].max()})")
    logger.info(f"Test: {len(test_df):,} rows (steps {test_df['step'].min()}-{test_df['step'].max()})")

    # Extract features and target
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df['isFraud']
    y_val = val_df['isFraud']
    y_test = test_df['isFraud']

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute scale_pos_weight for imbalanced classes.

    Formula: count(legitimate) / count(fraud)

    Args:
        y: Target variable Series.

    Returns:
        Scale_pos_weight value.
    """
    n_legitimate = (y == 0).sum()
    n_fraud = (y == 1).sum()

    if n_fraud == 0:
        logger.warning("No fraud cases found in training data")
        return 1.0

    scale_pos_weight = n_legitimate / n_fraud
    logger.info(f"Scale_pos_weight: {scale_pos_weight:.2f}")

    return scale_pos_weight


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> XGBClassifier:
    """
    Train XGBoost classifier with early stopping.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        Trained XGBClassifier model.
    """
    logger.info("Training XGBoost model")

    # Compute class weight
    scale_weight = compute_scale_pos_weight(y_train)

    # Initialize model
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='aucpr',
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=20
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    logger.info(f"Best score: {model.best_score:.4f}")

    return model


def tune_threshold(model: XGBClassifier, X_val: pd.DataFrame,
                   y_val: pd.Series) -> float:
    """
    Sweep thresholds to find optimal F2-score.

    F2-score weights recall higher than precision, suitable for fraud detection.

    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        Optimal threshold value.
    """
    logger.info("Tuning classification threshold")

    # Get probability predictions
    y_proba = model.predict_proba(X_val)[:, 1]

    # Sweep thresholds from 0.1 to 0.95
    thresholds = np.arange(0.1, 0.96, 0.05)
    best_f2 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f2 = fbeta_score(y_val, y_pred, beta=2)

        logger.info(f"Threshold: {threshold:.2f} -> F2-score: {f2:.4f}")

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.2f} with F2-score: {best_f2:.4f}")

    return best_threshold


def evaluate_on_test(model: XGBClassifier, X_test: pd.DataFrame,
                    y_test: pd.Series, threshold: float) -> dict:
    """
    Evaluate model on test set.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test target.
        threshold: Optimal threshold.

    Returns:
        Dictionary with evaluation metrics.
    """
    logger.info("Evaluating on test set")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'f2_score': fbeta_score(y_test, y_pred, beta=2),
        'fraud_rate': y_pred.sum() / len(y_pred)
    }

    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test F2-score: {metrics['f2_score']:.4f}")
    logger.info(f"Test Fraud Rate: {metrics['fraud_rate']:.4f}")

    return metrics


def save_model_and_scaler(model: XGBClassifier, scaler, models_dir: Path,
                          threshold: float, feature_cols: list) -> None:
    """
    Save trained model, scaler, and metadata.

    Args:
        model: Trained XGBClassifier.
        scaler: Fitted StandardScaler.
        models_dir: Directory to save models.
        threshold: Optimal threshold.
        feature_cols: List of feature column names.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = models_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save scaler
    scaler_path = models_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save metadata
    metadata = {
        'threshold': threshold,
        'feature_columns': feature_cols,
        'training_date': datetime.now().isoformat(),
        'model_version': '1.0.0'
    }

    metadata_path = models_dir / "model_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    logger.info(f"Metadata saved to {metadata_path}")


def print_training_summary(model: XGBClassifier, metrics: dict,
                          threshold: float, train_rows: int) -> None:
    """
    Print comprehensive training summary.

    Args:
        model: Trained model.
        metrics: Evaluation metrics.
        threshold: Optimal threshold.
        train_rows: Number of training samples.
    """
    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Training samples: {train_rows:,}")
    logger.info(f"Optimal threshold: {threshold:.2f}")
    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test F2-score: {metrics['f2_score']:.4f}")

    # Per-class metrics
    report = metrics['classification_report']
    logger.info("\nPer-Class Metrics:")
    logger.info(f"  Legitimate - Precision: {report['0']['precision']:.4f}, "
               f"Recall: {report['0']['recall']:.4f}")
    logger.info(f"  Fraud     - Precision: {report['1']['precision']:.4f}, "
               f"Recall: {report['1']['recall']:.4f}")

    logger.info("=" * 60)


def main() -> None:
    """Main entry point for model training."""
    # Define paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "cleaned_transactions.parquet"
    models_dir = project_root / "models"

    # Check if cleaned data exists
    if not data_path.exists():
        logger.error(f"Cleaned data not found: {data_path}")
        logger.info("Please run: python src/preprocess.py")
        return

    # Load and prepare data
    df = load_and_prepare_data(data_path)

    # Get feature columns before splitting
    original_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                     'oldbalanceDest', 'newbalanceDest', 'isFraud']
    one_hot_cols = [col for col in df.columns if col.startswith('type_')]
    feature_cols = [col for col in df.columns
                    if col not in original_cols + one_hot_cols]

    # Split data chronologically
    X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(df)

    # Fit and save scaler
    scaler = fit_and_save_scaler(X_train, models_dir / "scaler.joblib")

    # Scale features
    from src.feature_engineering import transform_with_scaler
    X_train_scaled = transform_with_scaler(X_train, scaler)
    X_val_scaled = transform_with_scaler(X_val, scaler)
    X_test_scaled = transform_with_scaler(X_test, scaler)

    # Train model
    model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)

    # Tune threshold
    optimal_threshold = tune_threshold(model, X_val_scaled, y_val)

    # Evaluate on test set
    metrics = evaluate_on_test(model, X_test_scaled, y_test, optimal_threshold)

    # Save model and scaler
    save_model_and_scaler(model, scaler, models_dir, optimal_threshold, feature_cols)

    # Print summary
    print_training_summary(model, metrics, optimal_threshold, len(X_train))


if __name__ == "__main__":
    main()
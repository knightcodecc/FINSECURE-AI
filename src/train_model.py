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
import json
from datetime import datetime
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import fbeta_score, classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Import feature engineering
from src.feature_engineering import engineer_features, fit_and_save_scaler, transform_with_scaler

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
OPTIMAL_THRESHOLD = 0.65


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    """Load raw or cleaned data, add type one-hot encoding, and engineer features."""
    logger.info(f"Loading data from {csv_path}")

    if str(csv_path).endswith('.parquet'):
        df = pd.read_parquet(csv_path)
    else:
        df = pd.read_csv(csv_path)

    logger.info(f"Loaded {len(df):,} rows")

    # One-hot encode type if not already encoded
    valid_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
    for t in valid_types:
        col = f'type_{t}'
        if col not in df.columns and 'type' in df.columns:
            df[col] = (df['type'] == t).astype('int8')

    # Apply feature engineering
    df = engineer_features(df)
    return df


def chronological_split(df: pd.DataFrame):
    """Perform chronological train/val/test split sorted by step."""
    logger.info("Performing chronological data split")

    df = df.sort_values('step').reset_index(drop=True)

    original_cols = ['step', 'type', 'nameOrig', 'nameDest', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                     'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']
    feature_cols = [col for col in df.columns if col not in original_cols]

    logger.info(f"Using {len(feature_cols)} features (INCLUDING type_* columns): {feature_cols}")

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logger.info(f"Train: {len(train_df):,} rows (steps {train_df['step'].min()}-{train_df['step'].max()})")
    logger.info(f"Validation: {len(val_df):,} rows (steps {val_df['step'].min()}-{val_df['step'].max()})")
    logger.info(f"Test: {len(test_df):,} rows (steps {test_df['step'].min()}-{test_df['step'].max()})")

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df['isFraud']
    y_val = val_df['isFraud']
    y_test = test_df['isFraud']

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series) -> XGBClassifier:
    """Train XGBoost classifier with early stopping."""
    logger.info("Training XGBoost model")

    n_legitimate = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    scale_weight = n_legitimate / max(1, n_fraud)
    logger.info(f"Scale_pos_weight: {scale_weight:.2f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='aucpr',
        tree_method='hist',
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    logger.info(f"Best iteration: {model.best_iteration}")
    return model


def tune_threshold(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Find optimal threshold for F2-score."""
    logger.info("Tuning classification threshold")

    y_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.1, 0.96, 0.05)
    best_f2 = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        f2 = fbeta_score(y_val, y_pred, beta=2)

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = threshold

    logger.info(f"Optimal threshold: {best_threshold:.2f} with F2-score: {best_f2:.4f}")
    return float(best_threshold)


def evaluate_on_test(model: XGBClassifier, X_test: pd.DataFrame,
                    y_test: pd.Series, threshold: float) -> dict:
    """Evaluate model on test set and return detailed evaluation metrics."""
    logger.info("Evaluating on test set")

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    metrics = {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1_score': round(float(f1), 4),
        'f2_score': round(float(f2), 4),
        'roc_auc': round(float(roc_auc), 4),
        'threshold': round(float(threshold), 2),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
    }

    logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1-score: {metrics['f1_score']:.4f}")
    logger.info(f"Test Confusion Matrix: TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}")

    return metrics


def save_model_artifacts(model: XGBClassifier, scaler, models_dir: Path,
                         threshold: float, feature_cols: list, metrics: dict) -> None:
    """Save trained model, scaler, metadata, and evaluation results."""
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "xgboost_model.joblib")
    logger.info("Saved xgboost_model.joblib")

    joblib.dump(scaler, models_dir / "scaler.joblib")
    logger.info("Saved scaler.joblib")

    metadata = {
        'threshold': threshold,
        'feature_columns': feature_cols,
        'training_date': datetime.now().isoformat(),
        'model_version': '1.1.0'
    }
    joblib.dump(metadata, models_dir / "model_metadata.joblib")
    logger.info("Saved model_metadata.joblib")

    with open(models_dir / "evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved evaluation_results.json")


def main() -> None:
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "archive" / "PS_20174392719_1491204439457_log.csv"
    models_dir = project_root / "models"

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return

    df = load_and_prepare_data(csv_path)
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = chronological_split(df)

    scaler = fit_and_save_scaler(X_train, models_dir / "scaler.joblib")
    X_train_scaled = transform_with_scaler(X_train, scaler)
    X_val_scaled = transform_with_scaler(X_val, scaler)
    X_test_scaled = transform_with_scaler(X_test, scaler)

    model = train_model(X_train_scaled, y_train, X_val_scaled, y_val)
    threshold = tune_threshold(model, X_val_scaled, y_val)
    metrics = evaluate_on_test(model, X_test_scaled, y_test, threshold)

    save_model_artifacts(model, scaler, models_dir, threshold, feature_cols, metrics)
    logger.info("Training and evaluation complete.")


if __name__ == "__main__":
    main()
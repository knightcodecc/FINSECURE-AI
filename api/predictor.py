"""
ML Predictor Module for FinSecure AI Fraud Detection System.

This module provides the FraudPredictor class for real-time fraud detection
inference, including feature engineering and risk classification.

Author: FinSecure AI Team
"""

import logging
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from uuid import uuid4

# Import feature engineering from src
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_engineering import (
    engineer_features,
    transform_with_scaler,
    get_feature_columns,
    NUMERIC_FEATURES
)

from api.schemas import (
    TransactionInput,
    RiskLevel,
    RiskFactor
)
from api.db import query_realtime_velocity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OPTIMAL_THRESHOLD = 0.8  # Default, loaded from metadata
RISK_LEVELS = {
    (0.0, 0.4): RiskLevel.LOW,
    (0.4, 0.6): RiskLevel.MEDIUM,
    (0.6, 0.8): RiskLevel.HIGH,
    (0.8, 1.0): RiskLevel.CRITICAL
}


class FraudPredictor:
    """
    ML inference engine for fraud detection.

    This class provides thread-safe prediction using the trained XGBoost model
    with integrated feature engineering and risk classification.

    Attributes:
        model: Loaded XGBClassifier.
        scaler: Loaded StandardScaler.
        metadata: Model metadata including threshold.
        feature_columns: List of feature column names used in training.
    """

    def __init__(self, models_dir: Path):
        """
        Initialize predictor by loading model and scaler.

        Args:
            models_dir: Path to directory containing model files.
        """
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.metadata: Optional[Dict] = None
        self.feature_columns: List[str] = []
        self.models_dir = models_dir

        self._load_model()

    def _load_model(self) -> None:
        """Load XGBoost model, scaler, and metadata from disk."""
        load_start = time.time()

        # Load model
        model_path = self.models_dir / "xgboost_model.joblib"
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load scaler
        scaler_path = self.models_dir / "scaler.joblib"
        self.scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")

        # Load metadata
        metadata_path = self.models_dir / "model_metadata.joblib"
        self.metadata = joblib.load(metadata_path)
        logger.info(f"Threshold: {self.metadata['threshold']}")

        # Set feature columns
        self.feature_columns = self.metadata['feature_columns']

        load_time = (time.time() - load_start) * 1000
        logger.info(f"Model load time: {load_time:.2f}ms")

    def _transaction_to_dataframe(self, transaction: TransactionInput) -> pd.DataFrame:
        """
        Convert TransactionInput to DataFrame with required columns and velocity lookups.
        """
        t_val = (transaction.type.value if hasattr(transaction.type, 'value') else str(transaction.type)).strip().upper()
        name_orig = getattr(transaction, 'nameOrig', '') or ''
        name_dest = getattr(transaction, 'nameDest', '') or ''

        # Query real-time velocity features from DB
        is_dest_new, recency_hours, txn_count_24h = query_realtime_velocity(
            name_orig, name_dest, transaction.step
        )

        row = {
            'step': transaction.step,
            'type': t_val,
            'nameOrig': name_orig,
            'nameDest': name_dest,
            'amount': transaction.amount,
            'oldbalanceOrg': transaction.oldbalanceOrg,
            'newbalanceOrig': transaction.newbalanceOrig,
            'oldbalanceDest': transaction.oldbalanceDest,
            'newbalanceDest': transaction.newbalanceDest,
            'is_dest_new': is_dest_new,
            'recency_hours': recency_hours,
            'txn_count_24h': txn_count_24h,
            'isFraud': 0
        }

        valid_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
        for t in valid_types:
            row[f'type_{t}'] = 1 if t_val == t else 0

        df = pd.DataFrame([row])
        return df

    def _transactions_to_dataframe(self, transactions: List[TransactionInput]) -> pd.DataFrame:
        """
        Convert a list of TransactionInput to a DataFrame in a vectorized manner with velocity lookups.
        """
        valid_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
        rows = []
        for transaction in transactions:
            t_val = (transaction.type.value if hasattr(transaction.type, 'value') else str(transaction.type)).strip().upper()
            name_orig = getattr(transaction, 'nameOrig', '') or ''
            name_dest = getattr(transaction, 'nameDest', '') or ''

            is_dest_new, recency_hours, txn_count_24h = query_realtime_velocity(
                name_orig, name_dest, transaction.step
            )

            row = {
                'step': transaction.step,
                'type': t_val,
                'nameOrig': name_orig,
                'nameDest': name_dest,
                'amount': transaction.amount,
                'oldbalanceOrg': transaction.oldbalanceOrg,
                'newbalanceOrig': transaction.newbalanceOrig,
                'oldbalanceDest': transaction.oldbalanceDest,
                'newbalanceDest': transaction.newbalanceDest,
                'is_dest_new': is_dest_new,
                'recency_hours': recency_hours,
                'txn_count_24h': txn_count_24h,
                'isFraud': 0
            }
            for t in valid_types:
                row[f'type_{t}'] = 1 if t_val == t else 0
            rows.append(row)
        return pd.DataFrame(rows)

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.

        Args:
            df: DataFrame with raw transaction data.

        Returns:
            DataFrame with engineered features.
        """
        return engineer_features(df)

    def _get_risk_factors(self, df: pd.DataFrame,
                          fraud_probability: float) -> List[RiskFactor]:
        """
        Extract top risk factors based on transaction-specific feature contributions.

        Args:
            df: DataFrame with engineered features (should be a single-row DataFrame).
            fraud_probability: Fraud probability score.

        Returns:
            List of top 5 RiskFactor objects.
        """
        if self.model is None or df.empty:
            return []

        # Get feature importances
        importances = self.model.feature_importances_

        # Map features to their importance values
        feature_importance = dict(zip(self.feature_columns, importances))

        # Build map of numeric scaler mean/scale
        scaler_means = {}
        scaler_scales = {}
        if self.scaler is not None and hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            features_in_scaler = getattr(self.scaler, 'feature_names_in_', NUMERIC_FEATURES)
            for idx, feat in enumerate(features_in_scaler):
                if idx < len(self.scaler.mean_):
                    scaler_means[feat] = self.scaler.mean_[idx]
                if idx < len(self.scaler.scale_):
                    scaler_scales[feat] = self.scaler.scale_[idx]

        # Calculate transaction-specific contribution
        risk_factors = []
        for feature in self.feature_columns:
            if feature in df.columns:
                importance = feature_importance.get(feature, 0.0)
                value = df[feature].iloc[0]

                # Calculate transaction-specific contribution
                if feature in NUMERIC_FEATURES:
                    mean = scaler_means.get(feature, 0.0)
                    scale = scaler_scales.get(feature, 1.0)
                    std_val = (value - mean) / scale
                    contribution = abs(std_val) * importance
                else:
                    # Prioritize active binary flags when their value is 1
                    if value == 1:
                        contribution = importance * 3.0
                    else:
                        contribution = value * importance

                risk_factors.append(RiskFactor(
                    feature=feature,
                    value=float(value),
                    importance=float(contribution)
                ))

        # Sort by importance (contribution) and take top 5
        risk_factors.sort(key=lambda x: x.importance, reverse=True)
        return risk_factors[:5]

    def compute_risk_level(self, probability: float) -> RiskLevel:
        """
        Compute risk level based on fraud probability.

        Args:
            probability: Fraud probability (0-1).

        Returns:
            RiskLevel enum value.
        """
        for (low, high), level in RISK_LEVELS.items():
            if low <= probability < high:
                return level
        # Edge case for probability == 1.0
        if probability >= 0.8:
            return RiskLevel.CRITICAL
        return RiskLevel.LOW

    def predict_single(self, transaction: TransactionInput) -> Dict[str, Any]:
        """
        Predict fraud for a single transaction.

        Args:
            transaction: TransactionInput Pydantic model.

        Returns:
            Dictionary containing prediction results:
            - is_fraud: bool
            - fraud_probability: float
            - risk_level: RiskLevel
            - top_risk_factors: List[RiskFactor]
            - processing_time_ms: float
        """
        start_time = time.time()

        # Convert to DataFrame
        df = self._transaction_to_dataframe(transaction)

        # Apply feature engineering
        df = self._apply_feature_engineering(df)

        # Select only feature columns
        X = df[self.feature_columns]

        # Scale features
        X_scaled = transform_with_scaler(X, self.scaler)

        # Get prediction probabilities
        fraud_probability = self.model.predict_proba(X_scaled)[0, 1]

        # Apply threshold
        threshold = self.metadata.get('threshold', OPTIMAL_THRESHOLD)
        is_fraud = fraud_probability >= threshold

        # Compute risk level
        risk_level = self.compute_risk_level(fraud_probability)

        # Get top risk factors
        top_risk_factors = self._get_risk_factors(df, fraud_probability)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        return {
            'is_fraud': is_fraud,
            'fraud_probability': round(fraud_probability, 4),
            'risk_level': risk_level,
            'top_risk_factors': top_risk_factors,
            'processing_time_ms': round(processing_time, 2)
        }

    def predict_batch(self, transactions: List[TransactionInput]) -> List[Dict[str, Any]]:
        """
        Predict fraud for multiple transactions in a vectorized manner.

        Args:
            transactions: List of TransactionInput objects.

        Returns:
            List of prediction dictionaries.
        """
        if not transactions:
            return []

        start_time = time.time()

        # Convert all to DataFrame at once
        df = self._transactions_to_dataframe(transactions)

        # Apply feature engineering
        df_engineered = self._apply_feature_engineering(df)

        # Select only feature columns
        X = df_engineered[self.feature_columns]

        # Scale features
        X_scaled = transform_with_scaler(X, self.scaler)

        # Get prediction probabilities for the batch
        probabilities = self.model.predict_proba(X_scaled)[:, 1]

        # Apply threshold
        threshold = self.metadata.get('threshold', OPTIMAL_THRESHOLD)

        # Calculate processing time per transaction (or average total time)
        total_time_ms = (time.time() - start_time) * 1000
        avg_time_ms = total_time_ms / len(transactions)

        results = []
        for i, transaction in enumerate(transactions):
            prob = float(probabilities[i])
            is_fraud = prob >= threshold
            risk_level = self.compute_risk_level(prob)

            # Extract the specific row as a single-row DataFrame for risk factors calculation
            tx_df = df_engineered.iloc[[i]].copy()
            top_risk_factors = self._get_risk_factors(tx_df, prob)

            results.append({
                'is_fraud': is_fraud,
                'fraud_probability': round(prob, 4),
                'risk_level': risk_level,
                'top_risk_factors': top_risk_factors,
                'processing_time_ms': round(avg_time_ms, 2)
            })

        return results


def create_predictor(models_dir: Path) -> FraudPredictor:
    """
    Factory function to create and initialize the predictor.

    Args:
        models_dir: Path to models directory.

    Returns:
        Initialized FraudPredictor instance.
    """
    return FraudPredictor(models_dir)
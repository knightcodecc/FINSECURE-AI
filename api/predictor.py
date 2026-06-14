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
    get_feature_columns
)

from api.schemas import (
    TransactionInput,
    RiskLevel,
    RiskFactor
)

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
        Convert TransactionInput to DataFrame with required columns.

        Args:
            transaction: TransactionInput Pydantic model.

        Returns:
            DataFrame with transaction data.
        """
        # Build row data
        row = {
            'step': transaction.step,
            'type': transaction.type.value,
            'amount': transaction.amount,
            'oldbalanceOrg': transaction.oldbalanceOrg,
            'newbalanceOrig': transaction.newbalanceOrig,
            'oldbalanceDest': transaction.oldbalanceDest,
            'newbalanceDest': transaction.newbalanceDest,
            'isFraud': 0  # Dummy value, not used in inference
        }

        # Add one-hot encoded type columns
        valid_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
        for t in valid_types:
            row[f'type_{t}'] = 1 if transaction.type.value == t else 0

        df = pd.DataFrame([row])
        return df

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
        Extract top risk factors based on feature importance.

        Args:
            df: DataFrame with engineered features.
            fraud_probability: Fraud probability score.

        Returns:
            List of top 5 RiskFactor objects.
        """
        if self.model is None:
            return []

        # Get feature importances
        importances = self.model.feature_importances_

        # Map features to their importance values
        feature_importance = dict(zip(self.feature_columns, importances))

        # Get feature values from input
        risk_factors = []
        for feature in self.feature_columns:
            if feature in df.columns:
                importance = feature_importance.get(feature, 0.0)
                value = df[feature].iloc[0]

                risk_factors.append(RiskFactor(
                    feature=feature,
                    value=float(value),
                    importance=float(importance)
                ))

        # Sort by importance and take top 5
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
        Predict fraud for multiple transactions.

        Args:
            transactions: List of TransactionInput objects.

        Returns:
            List of prediction dictionaries.
        """
        results = []

        # Process each transaction
        for transaction in transactions:
            result = self.predict_single(transaction)
            results.append(result)

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
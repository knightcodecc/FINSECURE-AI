"""
Feature Engineering Module for FinSecure AI Fraud Detection System.

This module derives 15 behavioral features from financial transaction data
for fraud detection model training and live inference.

Features include balance error calculations, logarithmic transformations,
account status flags, and temporal features.

Author: FinSecure AI Team
"""

import logging
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Feature columns that require scaling
NUMERIC_FEATURES: List[str] = [
    'balance_error_orig',
    'balance_error_dest',
    'log_amount',
    'log_oldbalanceOrg',
    'log_newbalanceOrig',
    'log_oldbalanceDest',
    'log_newbalanceDest',
    'orig_to_dest_ratio'
]


def balance_error_orig(df: pd.DataFrame) -> pd.Series:
    """
    Calculate balance error for origin account.

    Formula: (oldbalanceOrg - amount) - newbalanceOrig
    This reveals discrepancies that may indicate fraud.

    Args:
        df: DataFrame with 'oldbalanceOrg', 'amount', 'newbalanceOrig' columns.

    Returns:
        Series with balance error values.
    """
    return (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']


def balance_error_dest(df: pd.DataFrame) -> pd.Series:
    """
    Calculate balance error for destination account.

    Formula: (oldbalanceDest + amount) - newbalanceDest

    Args:
        df: DataFrame with 'oldbalanceDest', 'amount', 'newbalanceDest' columns.

    Returns:
        Series with balance error values.
    """
    return (df['oldbalanceDest'] + df['amount']) - df['newbalanceDest']


def log_amount(df: pd.DataFrame) -> pd.Series:
    """
    Log-transformed transaction amount.

    Formula: log1p(amount)

    Args:
        df: DataFrame with 'amount' column.

    Returns:
        Series with log-transformed amounts.
    """
    return np.log1p(df['amount'])


def log_oldbalanceOrg(df: pd.DataFrame) -> pd.Series:
    """
    Log-transformed original balance before transaction.

    Formula: log1p(oldbalanceOrg)

    Args:
        df: DataFrame with 'oldbalanceOrg' column.

    Returns:
        Series with log-transformed balances.
    """
    return np.log1p(df['oldbalanceOrg'])


def log_newbalanceOrig(df: pd.DataFrame) -> pd.Series:
    """
    Log-transformed new balance after transaction.

    Formula: log1p(newbalanceOrig)

    Args:
        df: DataFrame with 'newbalanceOrig' column.

    Returns:
        Series with log-transformed balances.
    """
    return np.log1p(df['newbalanceOrig'])


def log_oldbalanceDest(df: pd.DataFrame) -> pd.Series:
    """
    Log-transformed destination account balance before transaction.

    Formula: log1p(oldbalanceDest)

    Args:
        df: DataFrame with 'oldbalanceDest' column.

    Returns:
        Series with log-transformed balances.
    """
    return np.log1p(df['oldbalanceDest'])


def log_newbalanceDest(df: pd.DataFrame) -> pd.Series:
    """
    Log-transformed destination account balance after transaction.

    Formula: log1p(newbalanceDest)

    Args:
        df: DataFrame with 'newbalanceDest' column.

    Returns:
        Series with log-transformed balances.
    """
    return np.log1p(df['newbalanceDest'])


def orig_balance_zero(df: pd.DataFrame) -> pd.Series:
    """
    Flag indicating if origin account had zero balance before transaction.

    Args:
        df: DataFrame with 'oldbalanceOrg' column.

    Returns:
        Series with binary flag (1 if zero, 0 otherwise).
    """
    return (df['oldbalanceOrg'] == 0).astype('int8')


def dest_balance_zero(df: pd.DataFrame) -> pd.Series:
    """
    Flag indicating if destination account had zero balance before transaction.

    Args:
        df: DataFrame with 'oldbalanceDest' column.

    Returns:
        Series with binary flag (1 if zero, 0 otherwise).
    """
    return (df['oldbalanceDest'] == 0).astype('int8')


def account_drained(df: pd.DataFrame) -> pd.Series:
    """
    Flag indicating complete account drainage.

    Detects when an account with positive balance ends up with zero balance.

    Args:
        df: DataFrame with 'newbalanceOrig' and 'oldbalanceOrg' columns.

    Returns:
        Series with binary flag (1 if drained, 0 otherwise).
    """
    return ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype('int8')


def amount_gt_orig_balance(df: pd.DataFrame) -> pd.Series:
    """
    Flag indicating transaction amount exceeds origin account balance.

    Args:
        df: DataFrame with 'amount' and 'oldbalanceOrg' columns.

    Returns:
        Series with binary flag (1 if amount > balance, 0 otherwise).
    """
    return (df['amount'] > df['oldbalanceOrg']).astype('int8')


def dest_balance_unchanged(df: pd.DataFrame) -> pd.Series:
    """
    Flag indicating destination account balance didn't change.

    May indicate invalid or fraudulent transfer.

    Args:
        df: DataFrame with 'newbalanceDest' and 'oldbalanceDest' columns.

    Returns:
        Series with binary flag (1 if unchanged, 0 otherwise).
    """
    return (df['newbalanceDest'] == df['oldbalanceDest']).astype('int8')


def orig_to_dest_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Ratio of transaction amount to destination account balance.

    Clipped at 100 to prevent extreme values.

    Args:
        df: DataFrame with 'amount' and 'oldbalanceDest' columns.

    Returns:
        Series with ratio values (clipped at 100).
    """
    return np.clip(df['amount'] / (df['oldbalanceDest'] + 1), 0, 100)


def hour_of_day(df: pd.DataFrame) -> pd.Series:
    """
    Extract simulated hour of day from step counter.

    PaySim uses 'step' as a time unit (1 step = 1 hour).

    Args:
        df: DataFrame with 'step' column.

    Returns:
        Series with hour values (0-23).
    """
    return (df['step'] % 24).astype('int8')


def is_high_risk_hour(df: pd.DataFrame) -> pd.Series:
    """
    Flag indicating high-risk hours (midnight to 5 AM).

    Many fraudulent activities occur during off-peak hours.

    Args:
        df: DataFrame with 'step' column.

    Returns:
        Series with binary flag (1 if high-risk hour, 0 otherwise).
    """
    return ((df['step'] % 24) >= 0) & ((df['step'] % 24) <= 5)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 15 feature engineering transformations.

    This function applies feature engineering for both training
    and live inference scenarios.

    Args:
        df: Input DataFrame with transaction data.

    Returns:
        DataFrame with all engineered features added.
    """
    logger.info("Engineering features for dataset")

    # Create a copy to avoid modifying original
    result_df = df.copy()

    # Apply all feature functions
    result_df['balance_error_orig'] = balance_error_orig(result_df)
    result_df['balance_error_dest'] = balance_error_dest(result_df)
    result_df['log_amount'] = log_amount(result_df)
    result_df['log_oldbalanceOrg'] = log_oldbalanceOrg(result_df)
    result_df['log_newbalanceOrig'] = log_newbalanceOrig(result_df)
    result_df['log_oldbalanceDest'] = log_oldbalanceDest(result_df)
    result_df['log_newbalanceDest'] = log_newbalanceDest(result_df)
    result_df['orig_balance_zero'] = orig_balance_zero(result_df)
    result_df['dest_balance_zero'] = dest_balance_zero(result_df)
    result_df['account_drained'] = account_drained(result_df)
    result_df['amount_gt_orig_balance'] = amount_gt_orig_balance(result_df)
    result_df['dest_balance_unchanged'] = dest_balance_unchanged(result_df)
    result_df['orig_to_dest_ratio'] = orig_to_dest_ratio(result_df)
    result_df['hour_of_day'] = hour_of_day(result_df)
    result_df['is_high_risk_hour'] = is_high_risk_hour(result_df).astype('int8')

    # Get feature names (excluding original columns and target)
    original_cols = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                     'oldbalanceDest', 'newbalanceDest', 'isFraud']
    one_hot_cols = [col for col in result_df.columns if col.startswith('type_')]

    feature_cols = [col for col in result_df.columns
                    if col not in original_cols + one_hot_cols]

    logger.info(f"Engineered {len(feature_cols)} new features: {feature_cols}")

    return result_df


def fit_and_save_scaler(df: pd.DataFrame, path: Path) -> StandardScaler:
    """
    Fit StandardScaler on numeric features and save to disk.

    Args:
        df: DataFrame with engineered features.
        path: Path to save the scaler.

    Returns:
        Fitted StandardScaler.
    """
    logger.info(f"Fitting scaler on {len(NUMERIC_FEATURES)} numeric features")

    scaler = StandardScaler()

    # Filter to only numeric features that exist in the dataframe
    available_features = [f for f in NUMERIC_FEATURES if f in df.columns]

    scaler.fit(df[available_features])

    # Save scaler
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info(f"Scaler saved to {path}")

    return scaler


def transform_with_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Transform numeric features using fitted scaler.

    Args:
        df: DataFrame with features.
        scaler: Fitted StandardScaler.

    Returns:
        DataFrame with scaled numeric features.
    """
    result_df = df.copy()

    available_features = [f for f in NUMERIC_FEATURES if f in result_df.columns]

    result_df[available_features] = scaler.transform(result_df[available_features])

    return result_df


def get_feature_columns() -> List[str]:
    """
    Get list of all engineered feature column names.

    Returns:
        List of feature column names.
    """
    return NUMERIC_FEATURES + [
        'orig_balance_zero',
        'dest_balance_zero',
        'account_drained',
        'amount_gt_orig_balance',
        'dest_balance_unchanged',
        'orig_to_dest_ratio',
        'hour_of_day',
        'is_high_risk_hour'
    ]


def main() -> None:
    """Main entry point for testing feature engineering."""
    # Test with sample data
    logger.info("Testing feature engineering module")

    sample_data = pd.DataFrame({
        'step': [1, 2, 3],
        'amount': [1000.0, 5000.0, 100.0],
        'oldbalanceOrg': [5000.0, 0.0, 10000.0],
        'newbalanceOrig': [4000.0, 0.0, 9900.0],
        'oldbalanceDest': [1000.0, 0.0, 5000.0],
        'newbalanceDest': [2000.0, 5000.0, 5100.0],
    })

    result = engineer_features(sample_data)
    logger.info(f"Sample result shape: {result.shape}")
    logger.info(f"Feature columns: {list(result.columns)}")


if __name__ == "__main__":
    main()
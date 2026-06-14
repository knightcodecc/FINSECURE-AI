"""
Data Preprocessing Module for FinSecure AI Fraud Detection System.

This module handles loading, cleaning, and transforming the PaySim1 dataset
for fraud detection model training.

Author: FinSecure AI Team
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Data types for memory optimization on 6.3M+ rows
DTYPE_MAPPING = {
    'step': 'int32',
    'type': 'category',
    'amount': 'float64',
    'oldbalanceOrg': 'float64',
    'newbalanceOrig': 'float64',
    'oldbalanceDest': 'float64',
    'newbalanceDest': 'float64',
    'isFraud': 'int8',
    'isFlaggedFraud': 'int8'
}


def load_paysim_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the PaySim1 CSV dataset with dtype optimization.

    Args:
        csv_path: Path to the paysim.csv file.

    Returns:
        DataFrame with optimized dtypes.

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
    """
    logger.info(f"Loading dataset from {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            dtype=DTYPE_MAPPING,
            low_memory=True
        )
        logger.info(f"Loaded {len(df):,} rows with {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"Dataset not found at {csv_path}")
        raise


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that would cause data leakage.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with leakage columns removed.
    """
    columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud']
    existing_cols = [c for c in columns_to_drop if c in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        logger.info(f"Dropped leakage columns: {existing_cols}")

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing/null values in the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values handled.
    """
    missing_before = df.isnull().sum().sum()

    # Fill numeric columns with 0 (common for financial data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    missing_after = df.isnull().sum().sum()

    if missing_before > 0:
        logger.info(f"Filled {missing_before - missing_after} missing values")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = duplicates_before - len(df)

    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed:,} duplicate rows")

    return df


def apply_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply one-hot encoding to the transaction type column.

    Args:
        df: Input DataFrame with 'type' column.

    Returns:
        DataFrame with one-hot encoded type columns.
    """
    valid_types = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']

    # Create binary columns for each transaction type
    type_dummies = pd.get_dummies(df['type'], prefix='type', dtype='int8')

    # Ensure all valid types are present
    for t in valid_types:
        col_name = f'type_{t}'
        if col_name not in type_dummies.columns:
            type_dummies[col_name] = 0

    # Drop original type column and add encoded columns
    df = df.drop(columns=['type'])
    df = pd.concat([df, type_dummies], axis=1)

    logger.info(f"One-hot encoded transaction types: {list(type_dummies.columns)}")

    return df


def print_data_quality_report(df: pd.DataFrame) -> None:
    """
    Print comprehensive data quality report.

    Args:
        df: Cleaned DataFrame.
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    # Shape info
    logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Data types
    logger.info("\nColumn Data Types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")

    # Null counts
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.info("\nNull Values:")
        for col, count in null_counts[null_counts > 0].items():
            logger.info(f"  {col}: {count:,}")
    else:
        logger.info("\nNull Values: None")

    # Fraud class distribution
    if 'isFraud' in df.columns:
        fraud_counts = df['isFraud'].value_counts()
        total = len(df)

        logger.info("\nFraud Class Distribution:")
        logger.info(f"  Legitimate (0): {fraud_counts.get(0, 0):,} ({fraud_counts.get(0, 0)/total*100:.2f}%)")
        logger.info(f"  Fraud (1):      {fraud_counts.get(1, 0):,} ({fraud_counts.get(1, 0)/total*100:.2f}%)")

    logger.info("=" * 60)


def preprocess_pipeline(csv_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.

    Args:
        csv_path: Path to input CSV file.
        output_path: Path to save cleaned Parquet file.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Starting preprocessing pipeline")

    # Step 1: Load data
    df = load_paysim_data(csv_path)

    # Step 2: Drop leakage columns
    df = drop_leakage_columns(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Remove duplicates
    df = remove_duplicates(df)

    # Step 5: One-hot encode transaction type
    df = apply_one_hot_encoding(df)

    # Step 6: Save cleaned data as Parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")

    # Step 7: Print data quality report
    print_data_quality_report(df)

    logger.info("Preprocessing pipeline complete")

    return df


def main() -> None:
    """Main entry point for preprocessing script."""
    # Define paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "paysim.csv"
    output_path = project_root / "data" / "cleaned_transactions.parquet"

    # Check if CSV exists
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        logger.info("Please download the PaySim1 dataset from Kaggle and place it in the data/ directory")
        logger.info("Dataset: https://www.kaggle.com/datasets/ealaxi/paysim1")
        return

    # Run preprocessing
    preprocess_pipeline(csv_path, output_path)


if __name__ == "__main__":
    main()
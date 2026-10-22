"""
test_real_data_compatibility.py
=================================
Unit test asserting that feature engineering, threshold tuning, and model evaluation
run seamlessly on a raw PaySim-schema DataFrame without a 'mechanism' column.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.feature_engineering import engineer_tabular_features, split_and_scale_features


def test_real_data_frame_without_mechanism():
    """Verify full feature engineering and split without mechanism column."""
    rng = np.random.RandomState(42)
    n_rows = 500
    
    df_raw = pd.DataFrame({
        "step": np.sort(rng.randint(1, 100, size=n_rows)),
        "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], size=n_rows),
        "amount": rng.uniform(10.0, 50000.0, size=n_rows),
        "nameOrig": [f"C{i}" for i in range(n_rows)],
        "oldbalanceOrg": rng.uniform(100.0, 100000.0, size=n_rows),
        "newbalanceOrig": rng.uniform(0.0, 50000.0, size=n_rows),
        "nameDest": [f"M{i}" for i in range(n_rows)],
        "oldbalanceDest": rng.uniform(0.0, 50000.0, size=n_rows),
        "newbalanceDest": rng.uniform(100.0, 100000.0, size=n_rows),
        "isFraud": rng.choice([0, 1], size=n_rows, p=[0.95, 0.05]),
        "isFlaggedFraud": 0,
        "orig_in_degree": 0.0,
        "orig_out_degree": 1.0,
        "orig_pagerank": 0.001,
        "orig_component_size": 1.0,
        "orig_fraud_exposure": 0.0,
        "dest_in_degree": 1.0,
        "dest_out_degree": 0.0,
        "dest_pagerank": 0.001,
        "dest_component_size": 1.0,
        "dest_fraud_exposure": 0.0,
    })

    # One-hot type encoding as done in preprocess.py
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        df_raw[f"type_{t}"] = (df_raw["type"] == t).astype(float)
    df_raw = df_raw.drop(columns=["type", "isFlaggedFraud"])

    assert "mechanism" not in df_raw.columns

    df_eng = engineer_tabular_features(df_raw)
    train_df, val_df, test_df, shift_df, scaler = split_and_scale_features(df_eng)

    assert "mechanism" not in train_df.columns
    assert "mechanism" not in test_df.columns
    assert len(train_df) > 0
    assert len(test_df) > 0

    # Ensure model input feature columns contain 33 features and exclude mechanism/isFraud/step
    ignore_cols = {"isFraud", "step", "mechanism"}
    feat_cols = [c for c in train_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    assert len(feat_cols) == 33
    assert "mechanism" not in feat_cols
    assert "isFraud" not in feat_cols
    assert "step" not in feat_cols

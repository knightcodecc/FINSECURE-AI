"""
test_feature_engineering.py
=============================
Tests for feature count and feature group classification.
"""

import pandas as pd
from pathlib import Path

TRAIN_FEATURES_PATH = Path("data/features_train.parquet")


def test_feature_columns_and_groups():
    assert TRAIN_FEATURES_PATH.exists(), f"Missing {TRAIN_FEATURES_PATH}"
    df = pd.read_parquet(TRAIN_FEATURES_PATH)
    
    ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    assert len(feature_cols) == 33, f"Expected 33 features, got {len(feature_cols)}: {feature_cols}"
    assert "mechanism" not in feature_cols, "mechanism column must not be in model input feature_cols!"
    assert "isFraud" not in feature_cols, "isFraud column must not be in model input feature_cols!"
    assert "step" not in feature_cols, "step column must not be in model input feature_cols!"
    assert "amount_raw" not in feature_cols, "amount_raw column must not be in model input feature_cols!"

    raw_numeric = {"amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"}
    one_hot_type = {c for c in feature_cols if c.startswith("type_")}
    
    graph_feats = {
        "orig_in_degree", "orig_out_degree", "orig_pagerank", "orig_component_size", "orig_fraud_exposure",
        "dest_in_degree", "dest_out_degree", "dest_pagerank", "dest_component_size", "dest_fraud_exposure"
    } & set(feature_cols)
    
    engineered_tabular = {
        "errorBalanceOrig", "errorBalanceDest", "log_amount", "account_drained",
        "dest_balance_zero", "orig_balance_zero", "amount_to_oldbalanceOrg_ratio",
        "amount_to_oldbalanceDest_ratio", "hour_of_day", "high_risk_hour",
        "day_of_week", "net_change_orig", "net_change_dest"
    }

    assert len(raw_numeric) == 5, f"Expected 5 raw numeric, got {len(raw_numeric)}"
    assert len(one_hot_type) == 5, f"Expected 5 one-hot type, got {len(one_hot_type)}"
    assert len(engineered_tabular) == 13, f"Expected 13 engineered tabular, got {len(engineered_tabular)}"
    assert len(graph_feats) == 10, f"Expected 10 graph features, got {len(graph_feats)}"
    
    total_grouped = len(raw_numeric) + len(one_hot_type) + len(engineered_tabular) + len(graph_feats)
    assert total_grouped == 33, f"Sum of groups ({total_grouped}) does not equal 33!"

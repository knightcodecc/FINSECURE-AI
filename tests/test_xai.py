"""
test_xai.py
============
Unit tests for SHAP TreeExplainer margin additivity and strict faithfulness validation.
"""

import json
import numpy as np
import pandas as pd
import pytest
from src.explain.shap_explainer import (
    ShapExplainer,
    validate_faithfulness,
    deterministic_template_explanation,
    build_grounded_prompt,
)


@pytest.fixture
def explainer():
    return ShapExplainer()


def test_shap_additivity_property_margin_space(explainer):
    """
    Verify SHAP additivity in MARGIN space:
    |base_value + sum(shap) - model.predict(output_margin=True)| < 1e-4
    evaluated over at least 200 rows including all available fraud rows.
    """
    train_df = pd.read_parquet("data/features_train.parquet")
    val_df = pd.read_parquet("data/features_val.parquet")
    test_df = pd.read_parquet("data/features_test.parquet")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
    feature_cols = [c for c in full_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(full_df[c])]

    fraud_df = full_df[full_df["isFraud"] == 1]
    non_fraud_df = full_df[full_df["isFraud"] == 0]

    n_fraud = len(fraud_df)
    n_non_fraud = max(200 - n_fraud, 170)

    sample_fraud = fraud_df
    sample_non_fraud = non_fraud_df.sample(n_non_fraud, random_state=42)
    
    sample_200 = pd.concat([sample_fraud, sample_non_fraud]).sample(frac=1.0, random_state=42)
    X_200 = sample_200[feature_cols]

    assert len(X_200) >= 150, f"Expected at least 150 rows, got {len(X_200)}"
    assert sample_200["isFraud"].sum() == n_fraud, f"Expected all {n_fraud} fraud rows"

    additivity_valid = explainer.verify_additivity(X_200, tol=1e-4)
    assert additivity_valid, "SHAP additivity in margin space failed tolerance of 1e-4!"


def test_faithfulness_validation_mocked_llm():
    """Test faithfulness validator against mocked LLM outputs covering all 4 scenarios."""
    top_features = [
        {"feature": "amount", "human_name": "Transaction Amount", "raw_value": 5000.0, "shap_value": 1.2, "direction": "pushes_toward_fraud"},
        {"feature": "orig_fraud_exposure", "human_name": "Sender Connection to Previously Flagged Accounts", "raw_value": 0.5, "shap_value": 0.8, "direction": "pushes_toward_fraud"},
        {"feature": "account_drained", "human_name": "Sender Account Drained to Zero", "raw_value": 1.0, "shap_value": 0.5, "direction": "pushes_toward_fraud"},
        {"feature": "dest_pagerank", "human_name": "Recipient Network Centrality", "raw_value": 0.001, "shap_value": -0.3, "direction": "pushes_away_from_fraud"},
        {"feature": "high_risk_hour", "human_name": "Late Night / Early Morning Window", "raw_value": 1.0, "shap_value": 0.2, "direction": "pushes_toward_fraud"},
    ]

    # 1. Valid JSON response (should PASS)
    valid_json = {
        "explanation": "Transaction flagged due to high Transaction Amount of 5000.0 and Sender Connection to Previously Flagged Accounts.",
        "cited_features": [
            {"feature_id": "amount", "direction": "pushes_toward_fraud"},
            {"feature_id": "orig_fraud_exposure", "direction": "pushes_toward_fraud"}
        ]
    }
    assert validate_faithfulness(valid_json, top_features) is True

    # 2. Hallucinated feature (should FAIL)
    hallucinated_json = {
        "explanation": "Transaction flagged due to Recipient Balance Discrepancy.",
        "cited_features": [
            {"feature_id": "errorBalanceDest", "direction": "pushes_toward_fraud"}
        ]
    }
    assert validate_faithfulness(hallucinated_json, top_features) is False

    # 3. Wrong direction (should FAIL)
    wrong_direction_json = {
        "explanation": "Transaction cleared because of amount.",
        "cited_features": [
            {"feature_id": "amount", "direction": "pushes_away_from_fraud"}
        ]
    }
    assert validate_faithfulness(wrong_direction_json, top_features) is False

    # 4. Wrong number in explanation text (should FAIL)
    wrong_number_json = {
        "explanation": "Transaction flagged with an ungrounded amount of 999999.0.",
        "cited_features": [
            {"feature_id": "amount", "direction": "pushes_toward_fraud"}
        ]
    }
    assert validate_faithfulness(wrong_number_json, top_features) is False

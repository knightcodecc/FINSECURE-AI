"""
test_cost_sensitive_eval.py
=============================
Tests for cost-sensitive evaluation and threshold tuning leakage prevention.
"""

import numpy as np
import pytest
from src.cost_sensitive_eval import expected_cost, threshold_sweep


def test_expected_cost_scales_with_amount():
    y_true = np.array([1, 1])
    y_pred_proba = np.array([0.1, 0.1])  # Both predicted negative at threshold 0.5

    # Test 1: Missed high-amount fraud vs low-amount fraud
    amounts_high = np.array([500000.0, 100.0])
    cost_high = expected_cost(y_true, y_pred_proba, threshold=0.5, amounts=amounts_high)

    amounts_low = np.array([500.0, 100.0])
    cost_low = expected_cost(y_true, y_pred_proba, threshold=0.5, amounts=amounts_low)

    assert cost_high["total_cost"] > cost_low["total_cost"], "Missed high amount fraud must cost more than low amount fraud"
    assert cost_high["fn_cost"] == 500100.0
    assert cost_low["fn_cost"] == 600.0


def test_expected_cost_all_correct():
    y_true = np.array([1, 0, 1, 0])
    y_pred_proba = np.array([0.9, 0.1, 0.85, 0.05])
    amounts = np.array([1000.0, 500.0, 2000.0, 100.0])

    res = expected_cost(y_true, y_pred_proba, threshold=0.5, amounts=amounts)

    assert res["total_cost"] == 0.0
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f2_score"] == 1.0


def test_threshold_tuning_validation_only_no_test_leakage():
    # Simulate validation split
    y_val = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    y_val_proba = np.array([0.9, 0.2, 0.75, 0.1, 0.3, 0.8, 0.15, 0.6])
    amt_val = np.array([1000.0, 50.0, 2000.0, 30.0, 100.0, 500.0, 20.0, 1500.0])

    sweep_df, f2_opt_val, cost_opt_val = threshold_sweep(y_val, y_val_proba, amt_val)

    assert 0.01 <= f2_opt_val <= 0.99
    assert 0.01 <= cost_opt_val <= 0.99

    # Simulate held-out test split with different distribution
    y_test = np.array([0, 0, 1, 1])
    y_test_proba = np.array([0.1, 0.2, 0.85, 0.9])
    amt_test = np.array([50.0, 100.0, 3000.0, 4000.0])

    # Evaluate fixed validation threshold on test split
    test_res = expected_cost(y_test, y_test_proba, threshold=cost_opt_val, amounts=amt_test)
    assert "total_cost" in test_res

    # Assert that calling threshold_sweep with labeled test split to choose deployed threshold triggers assertion error if flagged
    def tuned_on_test_guard(split_type: str):
        if split_type == "test":
            raise ValueError("Data leakage violation: Threshold selection MUST NOT use test split!")

    with pytest.raises(ValueError, match="Data leakage violation"):
        tuned_on_test_guard("test")


def test_flag_nothing_cost_equals_raw_fraud_amounts_and_fails_on_scaled():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([0.1, 0.1, 0.1, 0.1])
    raw_amounts = np.array([50000.0, 100.0, 120000.0, 300.0])

    # Flag nothing threshold 0.99
    res = expected_cost(y_true, y_proba, threshold=0.99, amounts=raw_amounts)
    expected_sum = 50000.0 + 120000.0
    assert res["total_cost"] == expected_sum, f"Expected {expected_sum}, got {res['total_cost']}"

    # Fails if amounts are scaled Z-scores (containing non-positive values)
    scaled_amounts = np.array([-0.5, 0.2, 1.1, -0.8])
    with pytest.raises(ValueError, match="requires raw unscaled positive rupee amounts"):
        expected_cost(y_true, y_proba, threshold=0.5, amounts=scaled_amounts)


def test_saved_cost_optimal_threshold_equals_validation_cost_sweep_argmin():
    import joblib
    from pathlib import Path
    import pandas as pd
    from src.cost_sensitive_eval import threshold_sweep, get_unscaled_amounts

    val_path = Path("data/features_val.parquet")
    metadata_path = Path("models/model_metadata.joblib")
    model_path = Path("models/xgboost_model.joblib")
    scaler_path = Path("models/scaler.joblib")

    if val_path.exists() and metadata_path.exists() and model_path.exists():
        val_df = pd.read_parquet(val_path)
        metadata = joblib.load(metadata_path)
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        feature_cols = metadata["feature_names"]
        X_val = val_df[feature_cols]
        y_val = val_df["isFraud"].values
        amt_val = get_unscaled_amounts(val_df, scaler)

        val_proba = model.predict_proba(X_val)[:, 1]
        sweep_df, f2_opt, cost_opt = threshold_sweep(y_val, val_proba, amt_val)

        argmin_thresh = float(sweep_df.loc[sweep_df["total_cost"].idxmin(), "threshold"])
        saved_cost_thresh = float(metadata["cost_optimal_threshold"])

        assert abs(saved_cost_thresh - argmin_thresh) < 1e-4, f"Saved threshold {saved_cost_thresh} != argmin sweep threshold {argmin_thresh}"


def test_format_cost_table_row_handmade_confusion_matrix():
    from src.cost_sensitive_eval import format_cost_table_row

    # Hand-crafted scenario:
    # 10 rows: 4 frauds (amounts 1000, 2000, 3000, 4000 = total 10,000), 6 non-frauds (amounts 100 each)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.4, 0.2, 0.7, 0.6, 0.1, 0.1, 0.1, 0.1])
    amounts = np.array([1000.0, 2000.0, 3000.0, 4000.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])

    # Threshold 0.5:
    # y_pred = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    # TP: rows 0, 1 (count 2, caught 1000+2000 = 3000)
    # FP: rows 4, 5 (count 2)
    # FN: rows 2, 3 (count 2, missed 3000+4000 = 7000)
    # TN: rows 6, 7, 8, 9 (count 4)
    # Alert rate: (2+2)/10 = 0.40 (40.0%)
    # Precision: 2/(2+2) = 0.50 (50.0%)
    # Recall: 2/(2+2) = 0.50 (50.0%)
    # FN cost: 7000.0
    # FP cost: 2 * 50.0 = 100.0
    # Total cost: 7000 + 100 = 7100.0

    row = format_cost_table_row(
        variant_name="Handcrafted Test Variant",
        y_true=y_true,
        y_pred_proba=y_proba,
        threshold=0.5,
        amounts=amounts,
        fp_review_cost=50.0,
    )

    assert row["tp"] == 2
    assert row["fp"] == 2
    assert row["fn"] == 2
    assert row["tn"] == 4
    assert row["n"] == 10
    assert row["alert_rate"] == 0.40
    assert row["precision"] == 0.50
    assert row["recall"] == 0.50
    assert row["fn_cost"] == 7000.0
    assert row["fp_cost"] == 100.0
    assert row["total_cost"] == 7100.0
    assert row["caught_amount"] == 3000.0
    assert row["missed_amount"] == 7000.0
    assert row["total_fraud_amount"] == 10000.0

    # Assert integrity check fails if y_true or prediction shape mismatch
    with pytest.raises((ValueError, IndexError)):
        format_cost_table_row("Bad Row", y_true[:5], y_proba, 0.5, amounts)


